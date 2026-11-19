"""Assess the §13.13 paired arms with the §13.12 focus and Query interface."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import groupby
import shutil
import time

import numpy as np
import torch

import route3_ovg_data as data
from route3_g3_r14_train import learned_hidden
from route3_ldr_u1r import model as lpips
from route3_ovg_1b_common import CONFIG, SOURCE, read
from route3_ovg_1b_optical_data import background_samples, b_oracle, geometry_root, place
from route3_ovg_1b_prime_evaluate import A, ORIGINAL, PRIME_EMA
from route3_ovg_1b_train import load_model
from route3_ovg_1c import OUT, aligned_input, aligned_sample
from route3_ovg_footprint import query_net
from route3_ovg_refocus import OUT as REFOCUS, case_frames as frozen_frames, focus_records, gpu_setup, valid_depth
from route3_ovg_refocus_query import cache_path, prepare_scene, method_band
from route3_ovg_refocus_render import check_cache, merged_metrics
from route3_ovg_subpixel import save_json, save_npz
from route3_s2a_evaluate import masks
from route3_s2a_learn import Store
from route3_upper_bound_evaluate import display, exr, score
from src.aperture_renderer import aperture_disk
from src.footprint_order import apply_footprint_order
from src.layered_aperture import merge_layers
from src.ovg_optics import fixed_visible_surfels, raster

ALIGNED_FIXED = 'fixed-original-aligned'
TRUE = 'GT-visible-reference'
NEW_MODELS = tuple((f'1c-{arm}-{suffix}', arm, key)
                   for arm in ('control','aligned') for suffix,key in (('EMA','ema'),('raw','model')))
NEW_METHODS = tuple(name for name,_,_ in NEW_MODELS) + (ALIGNED_FIXED,TRUE)
NO_EXPLICIT = (PRIME_EMA, ORIGINAL)


def guard():
    if shutil.disk_usage(OUT).free < 60*2**30:
        raise RuntimeError('Data disk below 60 GiB')


def case_frames():
    for *details, folder in frozen_frames():
        yield *details, OUT/'C3'/folder.relative_to(REFOCUS)


def old_folder(folder):
    return REFOCUS / folder.relative_to(OUT/'C3')


def fixed_aligned(name, frame, old):
    native = data.b_input(name, frame)
    valid = native['confidence'][0] > 0
    np.testing.assert_array_equal(old[valid],native['depth'][0,valid])
    return np.where(valid, aligned_input(name,frame)['depth'][0], old)


@torch.inference_mode()
def prepare():
    models = [(label,load_model(OUT/'train'/arm/'checkpoints/step-005000.pt',key).eval(),arm)
              for label,arm,key in NEW_MODELS]
    rows = []
    for name,frame,_,_,_,_,_,items,folder in case_frames():
        guard()
        prior = old_folder(folder)
        target = data.load(geometry_root(name)/'geometry'/f'f{frame:03d}-center.npz',
                           ('depth_axial_m','center_visible'))
        base_masks = data.load(prior/'R1-masks.npz')
        old = data.load(prior/'visible-depths.npz',(ORIGINAL,PRIME_EMA))
        depths = {ALIGNED_FIXED:fixed_aligned(name,frame,old[ORIGINAL]), TRUE:target['depth_axial_m']}
        item = ('B', name, data.TRAIN_FRAMES.index(frame))
        inputs = {'control':data.sample(item,'cuda')[0], 'aligned':aligned_sample(item,'cuda')[0]}
        for label,net,arm in models:
            pred = net(inputs[arm],protect_residual=True,render_heads=False)
            depths[label] = pred['depth1'][0].cpu().numpy()
        valid = {k:valid_depth(z) for k,z in depths.items()}
        valid[ALIGNED_FIXED] &= base_masks[ORIGINAL+'__valid']
        valid[TRUE] &= target['center_visible']
        common = base_masks['common_visible_background']
        if any(np.any(common & ~v) for v in valid.values()):
            raise ValueError(f'New geometry invalid on the fixed P1 fit set {name} f{frame:03d}')
        focus,_,_ = focus_records(name,frame,items,depths,valid,None,methods=NEW_METHODS,common_background=common)
        save_npz(folder/'visible-depths.npz',**depths)
        save_json(folder/'R1.json',dict(rows=[r for r in focus if r['method']!=TRUE]))
        frozen = [r for r in read(prior/'R1.json')['rows'] if r['method'] in (*NO_EXPLICIT,A)]
        save_json(folder/'focus.json',dict(rows=focus+frozen))
        rows.extend(focus+frozen)
        print('1C_FOCUS',name,frame,sum(r['失败'] for r in focus),flush=True)
    save_json(OUT/'C3-focus.json',dict(rows=rows, lens_focal_length_mm=50,
              background='exact 13.12 common fit mask', GT_hidden_and_Query_anchor=PRIME_EMA))


def prepare_queries():
    start = time.monotonic()
    groups = [list(group) for _,group in groupby(case_frames(),key=lambda row:row[0])]
    build = partial(prepare_scene,output=OUT/'C3',reuse_audit=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = [row for group in pool.map(build,groups) for row in group]
    save_json(OUT/'C3-query.json',dict(rows=rows,seconds=time.monotonic()-start,
              history='same §13.12 frozen sources, poses, observations and Query weights',
              GT_reference='keeps exact placed GT and Query layers of the 1b′ EMA row'))


def hidden_layers(query, store, inp_path, depth, parents, true_depth, true_bg, uv, item, manifest,
                  K, focus, band):
    case = item['case']
    active = check_cache(store,str(inp_path),focus,case['fstop'],band,depth.cpu().numpy())
    conditioned = dict(item,input=str(inp_path),case=dict(case,focus_m=focus))
    candidate = learned_hidden(query,store,conditioned,manifest,K,band)
    retrieved, qs = apply_footprint_order(candidate,depth,K)
    supported = dict(parents,recoverable=parents['recoverable'] & band[uv[:,1],uv[:,0]])
    aligned, gs = place(supported,true_depth,true_bg,depth,K,512)
    return [('GT',aligned),('Query',retrieved)], dict(cache_active_pixels=active,GT=gs,Query=qs)


@torch.inference_mode()
def render():
    if not (OUT/'C3-query.json').is_file():
        raise RuntimeError('Build the method-conditioned Query cache first')
    start = time.monotonic()
    query = query_net(SOURCE/'V1/retrained/step-020000.pt')
    metric = lpips()
    store = Store()
    rows, placements, times = [], [], []
    for name,frame,entry,_,_,_,manifest,items,folder in case_frames():
        guard()
        frame_start = time.monotonic()
        prior = old_folder(folder)
        focus_rows = {(r['protocol'],r['id'],r['method']):r for r in read(folder/'focus.json')['rows']}
        pending = []
        for item in items:
            case = item['case']
            path = folder/'render'/case['protocol']/case['id']/'case.json'
            if path.exists():
                cached = read(path)
                rows.extend(cached['rows'])
                placements.extend(cached['placements'])
            else:
                pending.append((item,path))
        if not pending:
            continue
        maps = dict(data.load(folder/'visible-depths.npz'),**data.load(prior/'visible-depths.npz',NO_EXPLICIT))
        bands = data.load(folder/'R2-bands.npz')
        ref_bands = data.load(prior/'R2-bands.npz')
        x,_ = data.sample(('B',name,data.TRAIN_FRAMES.index(frame)),'cuda')
        K = x['K'][0]
        size = tuple(x['rgb'].shape[-2:])
        center = K.new_zeros((1,2))
        center_target = data.load(entry['input_npz'],('rgb_srgb',))['rgb_srgb']
        zn = {'zero_visible_full_image':np.ones(size,bool)}
        spec = dict(tiers='B',entry=entry)
        a_center = background_samples(spec,K,1.4,center,size)
        models = {}
        for method,z in maps.items():
            surrogate = np.where(valid_depth(z),z,1e4).astype(np.float32)
            depth = torch.as_tensor(surrogate,device='cuda')
            cloud = fixed_visible_surfels(dict(depth1=depth[None]),x)
            zero = raster(cloud,K,1.4,center,*size)[0]
            models[method] = dict(depth=depth,cloud=cloud,zero=score(display(zero.cpu().numpy()),center_target,zn,metric))
        parents,true_depth,true_bg = b_oracle(name,frame)
        uv = np.floor(parents['uv']).astype(int).clip(0,511)
        for item,path in pending:
            guard()
            case = item['case']
            protocol,cid,aperture = case['protocol'],case['id'],case['fstop']
            target = display(exr(geometry_root(name)/'dof'/(cid+'.exr')))
            fixed_masks = masks(geometry_root(name),case)
            regions = {k:fixed_masks[k] for k in ('full_image','in_focus','band','recoverable')}
            zero_masks = {k:regions[k] for k in ('full_image','in_focus','recoverable')}
            offsets = aperture_disk(CONFIG['native_eval_rays'],.05/(2*aperture),device=K.device,dtype=K.dtype)
            local_rows,local_placements = [],[]
            for method in (*NEW_METHODS,*NO_EXPLICIT):
                plan = focus_rows[protocol,cid,method]
                if plan['失败']:
                    continue
                focus = plan['实际对焦距离_m']
                fmt = dict(scene=name,frame=frame,id=cid,protocol=protocol,aperture=aperture,
                           名义对焦距离_m=plan['名义对焦距离_m'],实际对焦距离_m=focus)
                F2 = background_samples(spec,K,focus,offsets,size)
                model = models[method]
                if method in NO_EXPLICIT:
                    layers, info = [('none',None)], {}
                else:
                    anchor = PRIME_EMA if method==TRUE else method
                    key = '__'.join((anchor,protocol,cid))
                    if method==TRUE:
                        anchor_focus = focus_rows[protocol,cid,anchor]['实际对焦距离_m']
                        band, inp_file = ref_bands[key], cache_path(prior,anchor)
                        layers, info = hidden_layers(query,store,inp_file,models[anchor]['depth'],parents,
                                    true_depth,true_bg,uv,item,manifest,K,anchor_focus,band)
                        info.update(hidden_layer_anchor=PRIME_EMA,
                                    own_band_pixels=int(method_band(maps[method],K.cpu().numpy(),plan).sum()))
                    else:
                        layers, info = hidden_layers(query,store,cache_path(folder,method),model['depth'],
                                    parents,true_depth,true_bg,uv,item,manifest,K,focus,bands[key])
                local_placements.append(dict(fmt,method=method,**info))
                for tag,hidden in layers:
                    cloud = merge_layers(model['cloud'],hidden)
                    rgb = raster(cloud,K,focus,offsets,*size,F2).mean(0)
                    zero = raster(cloud,K,focus,center,*size,a_center)[0]
                    image,zero_image,metrics = merged_metrics(rgb,zero,target,center_target,regions,zero_masks,metric)
                    metrics.update(model['zero'])
                    local_rows.extend(dict(fmt,method=method,hidden=tag,region=k,**v) for k,v in metrics.items())
                    save_npz(path.parent/f'{method}-{tag}.npz',prediction_srgb=image,zero_srgb=zero_image)
            save_json(path,dict(rows=local_rows,placements=local_placements))
            rows.extend(local_rows)
            placements.extend(local_placements)
            print('1C_RENDER', name,protocol,cid,len(local_rows),flush=True)
        times.append(dict(scene=name,frame=frame,seconds=time.monotonic()-frame_start))
        store.inputs.cache_clear()
        store.active.cache_clear()
    save_json(OUT/'C3-optical.json',dict(rows=rows,placements=placements,frame_seconds=times,
              seconds=time.monotonic()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,visible_geometry=rendered_geometry()))


def rendered_geometry():
    """Score cached visible maps against subpixel front surfaces on one common support."""
    result = []
    for name,frame,_,_,_,_,_,_,folder in case_frames():
        old = data.load(old_folder(folder)/'visible-depths.npz',(ORIGINAL,PRIME_EMA,A))
        depths = dict(old,**data.load(folder/'visible-depths.npz'))
        labels = data.load(data.OUT/'B'/name/'pinhole-subpixels'/f'f{frame:03d}.npz',
                           ('valid','depth1','K','depth_boundary'))
        common = labels['valid'] & np.logical_and.reduce([valid_depth(z) for z in depths.values()])
        regions = {'main_surface':common,'boundary_front':common&labels['depth_boundary']}
        for method,z in depths.items():
            for region,ok in regions.items():
                inverse_error = np.abs(1./z[ok].astype(np.float64)-1./labels['depth1'][ok])
                for focus in (1.4,5.):
                    factor = labels['K'][0,0]*.05/(2*1.4)*focus/(focus-.05)
                    result.append(dict(scene=name,frame=frame,method=method,region=region,
                                       nominal_focus_distance_m=focus,pixels=int(ok.sum()),
                                       inverse_absolute_error_sum=float(inverse_error.sum()),
                                       CoC_absolute_error_sum_px=float(inverse_error.sum()*factor)))
    return result



def cover_agreement():
    rows = []
    for _,name,index in data.inventory('dev')['B']:
        frame = data.TRAIN_FRAMES[index]
        y = data.load(data.OUT/'B'/name/'pinhole-subpixels'/f'f{frame:03d}.npz',('valid','depth_boundary'))
        original = data.b_input(name,frame)
        conf = original['confidence']
        available = conf > 0
        high = conf[1] >= .6
        high_edge = high & y['valid'] & y['depth_boundary']
        for arm,x in (('before',original),('after',aligned_input(name,frame))):
            d = x['depth']
            inverse = np.divide(1.,d,out=np.zeros_like(d),where=available)
            agree = available & (np.abs(inverse-inverse[1])<=.03*inverse[1])
            common = high & available[0]
            rows.append(dict(scene=name,frame=frame,arm=arm,
                all_SHARP_matches=int((common & agree[0]).sum()),all_joint_pixels=int(common.sum()),
                edge_SHARP_matches=int((high_edge & agree[0]).sum()),
                edge_SHARP_joint=int((high_edge & available[0]).sum()),
                edge_agreement_any=int((high_edge & (agree.sum(0)>=2)).sum()),
                edge_single=int((high_edge & (available.sum(0)==1)).sum()),
                edge_multiple=int((high_edge & (available.sum(0)>=2)).sum())))
    save_json(OUT/'C3-agreement.json',dict(rows=rows, fraction=.03,depthsplat_confidence=.6))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','query','render','agreement'))
    operation = parser.parse_args().operation
    from route3_ovg_1b_common import OUT as DATA_RUN
    data.OUT = DATA_RUN
    if operation in ('prepare','render'):
        gpu_setup()
        torch.cuda.reset_peak_memory_stats()
    {'prepare':prepare,'query':prepare_queries,'render':render,'agreement':cover_agreement}[operation]()
