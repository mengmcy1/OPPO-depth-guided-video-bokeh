"""Render the §13.12 cases with method-specific focus, bands and hidden placement."""
import time
import numpy as np
import torch

from route3_ovg_refocus import (OUT, METHODS, FIRST, LEGACY_HEADS, case_frames,
                                gpu_setup, guard, valid_depth)
from route3_ovg_refocus_query import cache_path
from route3_ovg_1b_common import CONFIG, SOURCE, read
from route3_ovg_1b_optical_data import background_samples, b_oracle, geometry_root, place
from route3_ovg_1b_prime_evaluate import A, OLD, OLD_FIXED
import route3_ovg_data as data
from route3_ovg_footprint import query_net
from route3_g3_r14_train import learned_hidden
from route3_ldr_u1r import model as lpips
from route3_s2a_evaluate import masks
from route3_s2a_learn import Store
from route3_ovg_subpixel import save_json, save_npz
from route3_upper_bound_evaluate import display, exr, score
from src.aperture_renderer import aperture_disk
from src.footprint_order import apply_footprint_order
from src.layered_aperture import merge_layers
from src.ovg_optics import fixed_visible_surfels, raster, visible_surfels


def render_scenes(maps, x):
    figures={}
    for method in METHODS:
        if method==A:
            continue
        key=OLD if method==OLD_FIXED else method
        pred=dict(depth1=torch.as_tensor(maps[key],device='cuda')[None])
        if method in (FIRST,OLD):
            pred.update({k:torch.as_tensor(maps[key+'__'+k],device='cuda') for k in LEGACY_HEADS})
            cloud=visible_surfels(pred,x)
        else:
            cloud=fixed_visible_surfels(pred,x)
        figures[method]=dict(depth=pred['depth1'][0],cloud=cloud)
    return figures


def check_cache(store,path,focus,aperture,band,depth):
    inp=store.inputs(path)
    if not np.array_equal(inp['target_depth'],np.where(valid_depth(depth),depth,1e4)):
        raise ValueError('Store.forward target depth differs from method geometry')
    expected=band.ravel()[inp['query_pixel']]
    active=store.active(path,focus,aperture)
    if not np.array_equal(active,np.flatnonzero(expected)) or len(active)!=int(band.sum()):
        raise ValueError('Store.active re-screen does not match the method band')
    return len(active)


def merged_metrics(rgb,zero,target,zero_target,regions,zero_masks,metric):
    srgb=display(rgb.cpu().numpy())
    zrgb=display(zero.cpu().numpy())
    metrics=score(srgb,target,regions,metric)
    metrics.update({'zero_'+k:v for k,v in score(zrgb,zero_target,zero_masks,metric).items()})
    return srgb,zrgb,metrics


@torch.inference_mode()
def render():
    if not (OUT/'R2.json').is_file():
        raise RuntimeError('R2 cache preparation must complete before rendering')
    start=time.monotonic()
    query=query_net(SOURCE/'V1/retrained/step-020000.pt')
    metric=lpips()
    store=Store()
    rows,placements,verified=[],[],[]
    timings=[]
    for name,frame,entry,_,_,_,manifest,items,folder in case_frames():
        guard()
        frame_start=time.monotonic()
        r1=read(folder/'R1.json')['rows']
        case_focus={(r['protocol'],r['id'],r['method']):r for r in r1}
        pending=[]
        for item in items:
            case=item['case']
            record=folder/'render'/case['protocol']/case['id']/'case.json'
            if not record.exists():
                pending.append((item,record))
                continue
            cached=read(record)
            rows.extend(cached['rows'])
            placements.extend(cached['placements'])
            verified.extend(cached['cache_checks'])
        if not pending:
            continue
        maps=data.load(folder/'visible-depths.npz')
        bands=data.load(folder/'R2-bands.npz')
        x,_=data.sample(('B',name,data.TRAIN_FRAMES.index(frame)),'cuda')
        models=render_scenes(maps,x)
        K=x['K'][0]
        size=tuple(x['rgb'].shape[-2:])
        center=K.new_zeros((1,2))
        obs=data.load(entry['input_npz'],('rgb_srgb',))
        zero_target=obs['rgb_srgb']
        zn={'zero_visible_full_image':np.ones(size,bool)}
        spec=dict(tiers='B',entry=entry)
        a_center=background_samples(spec,K,1.4,center,size)
        for layer in models.values():
            identity=raster(layer['cloud'],K,1.4,center,*size)[0]
            layer['visible_zero']=score(display(identity.cpu().numpy()),zero_target,zn,metric)
        parents,true_depth,true_bg=b_oracle(name,frame)
        uv=np.floor(parents['uv']).astype(int).clip(0,511)

        for item,record in pending:
            guard()
            case=item['case']
            cid,protocol,aperture=case['id'],case['protocol'],case['fstop']
            truth_root=geometry_root(name)
            target=display(exr(truth_root/'dof'/(cid+'.exr')))
            fixed_masks=masks(truth_root,case)
            regions={k:fixed_masks[k] for k in ('full_image','in_focus','band','recoverable')}
            zero_masks={k:regions[k] for k in ('full_image','in_focus','recoverable')}
            radius=.05/(2*aperture)
            offsets=aperture_disk(CONFIG['native_eval_rays'],radius,device=K.device,dtype=K.dtype)
            per_rows,per_places,per_checks=[],[],[]
            for method in METHODS:
                if method==OLD_FIXED:
                    continue  # This arm is rendered inside the same hidden/F2 block as OLD.
                plan=case_focus[protocol,cid,method]
                if plan['失败']:
                    continue
                focus=plan['实际对焦距离_m']
                fields=dict(scene=name,frame=frame,id=cid,protocol=protocol,aperture=aperture,
                            名义对焦距离_m=plan['名义对焦距离_m'],实际对焦距离_m=focus)
                F2=background_samples(spec,K,focus,offsets,size)
                if method==A:
                    pred,zero,values=merged_metrics(F2.mean(0),a_center[0],target,zero_target,
                                                   regions,zero_masks,metric)
                    per_rows.extend(dict(fields,method=A,hidden='standalone',region=reg,**v)
                                    for reg,v in values.items())
                    save_npz(record.parent/f'{method}-standalone.npz',prediction_srgb=pred,zero_srgb=zero)
                    continue
                band=bands['__'.join((method,protocol,cid))]
                inp_path=str(cache_path(folder,method))
                if method==OLD:
                    matched_plan=case_focus[protocol,cid,OLD_FIXED]
                    sibling=bands['__'.join((OLD_FIXED,protocol,cid))]
                    if matched_plan['实际对焦距离_m']!=focus or not np.array_equal(sibling,band):
                        raise ValueError('Primitive pair must share focus distance and band')
                active=check_cache(store,inp_path,focus,aperture,band,maps[method])
                query_row=dict(item,input=inp_path,case=dict(case,focus_m=focus))
                candidate=learned_hidden(query,store,query_row,manifest,K,band)
                depth=models[method]['depth']
                retrieved,qs=apply_footprint_order(candidate,depth,K)
                supported=dict(parents,recoverable=parents['recoverable']&band[uv[:,1],uv[:,0]])
                aligned,gs=place(supported,true_depth,true_bg,depth,K,512)
                per_checks.append(dict(fields,method=method,cache_active_pixels=active))
                for visible in ((method,OLD_FIXED) if method==OLD else (method,)):
                    layer=models[visible]
                    per_places.append(dict(fields,method=visible,GT=gs,Query=qs,band_pixels=active))
                    for tag,hidden in (('GT',aligned),('Query',retrieved)):
                        cloud=merge_layers(layer['cloud'],hidden)
                        rgb=raster(cloud,K,focus,offsets,*size,F2).mean(0)
                        zero=raster(cloud,K,focus,center,*size,a_center)[0]
                        pred,z,vals=merged_metrics(rgb,zero,target,zero_target,regions,zero_masks,metric)
                        vals.update(layer['visible_zero'])
                        per_rows.extend(dict(fields,method=visible,hidden=tag,region=k,**v) for k,v in vals.items())
                        save_npz(record.parent/f'{visible}-{tag}.npz',prediction_srgb=pred,zero_srgb=z)
                        del cloud
            save_json(record,dict(rows=per_rows,placements=per_places,cache_checks=per_checks))
            rows.extend(per_rows)
            placements.extend(per_places)
            verified.extend(per_checks)
            print('REFOCUS_R3',name,protocol,cid,len(per_rows),flush=True)
        timings.append(dict(scene=name,frame=frame,seconds=time.monotonic()-frame_start))
        # Store caches are keyed by method-specific depth files. Release all
        # previous-frame inputs while retaining the frozen source image tensors.
        store.inputs.cache_clear()
        store.active.cache_clear()
        del models
    save_json(OUT/'R3.json',dict(rows=rows,placements=placements,cache_checks=verified,
        K=CONFIG['native_eval_rays'],seconds=time.monotonic()-start,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,frame_timings=timings,
        primitive_pair='one focus, candidate cache, placed GT layer, placed Query layer and F2 per case',
        A='standalone SHARP+A-color at own focus distance',weights_optimized=False))
    print('REFOCUS_R3_DONE',len(rows),flush=True)


if __name__=='__main__':
    guard()
    gpu_setup()
    torch.cuda.reset_peak_memory_stats()
    render()
