"""1b′ optical comparisons with fixed clocks, focus and candidate identities."""
import argparse
from collections import defaultdict
import os
import subprocess
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import numpy as np
import torch

from route3_ovg_1b_common import CONFIG, OUT as OLD_RUN, SOURCE, read
from route3_ovg_1b_optical_data import background_samples, b_oracle, geometry_root, place
from route3_ovg_1b_prime import OUT, guard
from route3_ovg_1b_train import load_model
from route3_ovg_footprint import query_net
from route3_g3_r14_train import learned_hidden
from route3_ldr_u1r import model as lpips
from route3_s2a_evaluate import masks
from route3_s2a_learn import Store
from route3_upper_bound_evaluate import display, exr, pool, score
import route3_ovg_data as data
from route3_ovg_subpixel import save_json, save_npz
from src.aperture_renderer import aperture_disk
from src.dual_layer import depth_band
from src.footprint_order import apply_footprint_order
from src.layered_aperture import merge_layers, pixel_layer
from src.ovg_optics import fixed_visible_surfels, raster, visible_surfels


ORIGINAL = 'fixed-original'
DEPTHSPLAT = 'fixed-DepthSplat'
OLD = 'OVG-1b-EMA'
OLD_FIXED = 'OVG-1b-EMA-fixed'
PRIME_EMA = 'OVG-1b-prime-EMA'
PRIME_RAW = 'OVG-1b-prime-raw'
A = 'A-SHARP-A-color'
OLD_METHODS = (ORIGINAL, DEPTHSPLAT, OLD)


def by(record, keys):
    return tuple(record[key] for key in keys)


def summarize(rows):
    names = ('scene','protocol','nominal_focus_m','fstop','hidden','visible','region')
    groups = defaultdict(list)
    for row in rows:
        key = by(row,names[1:])
        for scene in (row['scene'],'pooled'):
            groups[(scene,*key)].append(row)
    index = {key:dict(zip(names,key),**pool(values)) for key,values in sorted(groups.items())}
    pairs = [
        (OLD_FIXED,OLD,'primitive'),
        (PRIME_EMA,PRIME_EMA+'-gate-off','gate-EMA'),
        (PRIME_RAW,PRIME_RAW+'-gate-off','gate-raw'),
    ]
    pairs += [(PRIME_EMA, ref, 'system') for ref in (ORIGINAL,DEPTHSPLAT,OLD_FIXED,OLD,A)]
    deltas = []
    for method,reference,comparison in pairs:
        for key,row in index.items():
            if row['visible']!=method:
                continue
            hidden = 'standalone' if reference==A else key[4]
            expected = index.get((*key[:4],hidden,reference,key[-1]))
            if expected is None or not row['pixels']:
                continue
            if expected['pixels']!=row['pixels']:
                raise ValueError('Paired optical evaluation mask changed')
            deltas.append(dict({k:row[k] for k in names}, reference=reference,
                               comparison=comparison, pixels=row['pixels'],
                               PSNR_delta=row['psnr']-expected['psnr'],
                               LPIPS_delta=row['patch_lpips']-expected['patch_lpips']))
    return dict(pooled=list(index.values()), paired=deltas)


def compare_placement(reference, actual):
    for hidden in ('GT','Query'):
        for field in ('eligible','placed','total','moved','shrunk','dropped'):
            if actual[hidden].get(field)!=reference[hidden].get(field):
                raise RuntimeError(f'Primitive comparison changed {hidden} placement: {field}')


@torch.inference_mode()
def run(references_only):
    start=time.monotonic()
    old=read(OLD_RUN/'P3/terminal/results.json')
    placement_key=('scene','protocol','id','visible')
    old_placements={by(row,placement_key):row for row in old['placements']}
    old_by_case=defaultdict(list)
    for row in old['rows']:
        if row['visible'] in OLD_METHODS:
            old_by_case[by(row,('scene','protocol','id'))].append(row)
    historical=OUT/'Q3/reference/results.json'
    if not references_only and not historical.exists():
        raise FileNotFoundError('Q3 matched existing-1b and A references are required')
    if references_only:
        src=load_model(OLD_RUN/'checkpoints/step-030000.pt','ema').eval()
        models=[(OLD_FIXED,src,False)]
        total_rows,total_places=[],[]
    else:
        ck=OUT/'checkpoints/step-030000.pt'
        ema=load_model(ck,'ema').eval()
        raw=load_model(ck,'model').eval()
        models=[(PRIME_EMA,ema,True),(PRIME_RAW,raw,True),
                (PRIME_EMA+'-gate-off',ema,False),(PRIME_RAW+'-gate-off',raw,False)]
        h=read(historical)
        total_rows,total_places=h['rows'],h['placements']
    mode = 'reference' if references_only else 'terminal'
    work=OUT/'Q3'/mode
    query=query_net(SOURCE/'V1/retrained/step-020000.pt')
    store=Store()
    metric=lpips()
    root=data.PREVIOUS
    names=read(root/'split.json')['heldout']
    gauss_peak=0
    tensor=lambda a:torch.as_tensor(a,device='cuda')
    for name in names:
        geometry=root/'geometry/original'/name
        manifest=read(root/'evaluation-input/original'/name/'512-13/manifest.json')
        entries=read(geometry/'prepared.json')['rows']
        for entry in entries:
            guard()
            frame=entry['frame']
            cases=[r for r in manifest['rows'] if r['frame']==frame]
            pending=[r for r in cases
                     if not (work/name/r['case']['protocol']/r['case']['id']/'case.json').exists()]
            for item in (r for r in cases if r not in pending):
                case=item['case']
                cached=read(work/name/case['protocol']/case['id']/'case.json')
                total_rows.extend(cached['rows'])
                total_places.extend(cached['placements'])
            if not pending:
                continue
            x,_=data.sample(('B',name,data.TRAIN_FRAMES.index(frame)),'cuda')
            K=x['K'][0]
            size = tuple(x['rgb'].shape[-2:])
            rgb=x['rgb'][0].permute(1,2,0)
            eye=torch.eye(4,device='cuda')
            # Keep the same pinhole reference and method support as 1b.
            observed=data.load(entry['input_npz'],('depth','rgb_srgb'))
            zero_target=observed['rgb_srgb']
            ref_zero_mask={'zero_visible_full_image':np.ones(size,bool)}
            spec=dict(tiers='B',entry=entry)
            center=K.new_zeros((1,2))
            a_center=background_samples(spec,K,1.4,center,size)
            old_zero={}
            if references_only:
                depth_splat=root/'geometry/depthsplat-official'/name
                row=next(e for e in read(depth_splat/'prepared.json')['rows'] if e['frame']==frame)
                depths={ORIGINAL:observed['depth'],
                        DEPTHSPLAT:data.load(row['input_npz'],('depth',))['depth']}
                for label,depth in depths.items():
                    cloud=pixel_layer(rgb,tensor(depth),K,eye)
                    zero=raster(cloud,K,1.4,center,*size)[0]
                    old_zero[label]=score(display(zero.cpu().numpy()),zero_target,ref_zero_mask,metric)
                    del cloud
                legacy=visible_surfels(src(x),x)
                zero=raster(legacy,K,1.4,center,*size)[0]
                old_zero[OLD]=score(display(zero.cpu().numpy()),zero_target,ref_zero_mask,metric)
                del legacy
            models_for_frame={}
            for label,model,gate in models:
                pred=model(x,protect_residual=gate,render_heads=False)
                models_for_frame[label]=(pred, fixed_visible_surfels(pred,x))
            zero_for_frame={}
            for label,(_,cloud) in models_for_frame.items():
                gauss_peak=max(gauss_peak,len(cloud.means))
                vis_zero=raster(cloud,K,1.4,center,*size)[0]
                zero_for_frame[label]=score(display(vis_zero.cpu().numpy()),zero_target,ref_zero_mask,metric)
            parents,gt_depth,gt_bg=b_oracle(name,frame)
            for item in pending:
                case=item['case']
                cid,protocol,focus,fstop=by(case,('id','protocol','focus_m','fstop'))
                signature=(name,protocol,cid)
                baseline=old_by_case[signature]
                if not baseline or not all(r['focus_m']==focus for r in baseline):
                    raise ValueError('Q3 case and focus do not match existing 1b')
                radius=.05/(2*fstop)
                offsets=aperture_disk(CONFIG['native_eval_rays'],radius,device=K.device,dtype=K.dtype)
                F2=background_samples(spec,K,focus,offsets,size)
                true=geometry_root(name)
                target=display(exr(true/'dof'/(cid+'.exr')))
                regions=masks(true,case)
                regions={k:regions[k] for k in ('full_image','in_focus','band','recoverable')}
                zero_masks={k:regions[k] for k in ('full_image','in_focus','recoverable')}
                support=depth_band(observed['depth'],float(K[0,0]),focus,radius)[0]
                uv=np.floor(parents['uv']).astype(int).clip(0,511)
                common_parents=dict(parents,recoverable=parents['recoverable'] & support[uv[:,1],uv[:,0]])
                shared_candidate=learned_hidden(query,store,item,manifest,K,support)
                dest=work/name/protocol/cid
                rows,placements=[],[]
                common=dict(scene=name,frame=frame,id=cid,protocol=protocol,focus_m=focus,
                            nominal_focus_m=case['nominal_focus_m'],fstop=fstop)

                def append(label,branch,image,zero,visible_zero):
                    rendered=display(image.cpu().numpy())
                    pinhole=display(zero.cpu().numpy())
                    values=score(rendered,target,regions,metric)
                    values.update({'zero_'+k:v for k,v in score(pinhole,zero_target,zero_masks,metric).items()})
                    values.update(visible_zero)
                    rows.extend(dict(common,visible=label,hidden=branch,region=k,**v) for k,v in values.items())
                    save_npz(dest/f'{label}-{branch}.npz',prediction_srgb=rendered,zero_srgb=pinhole)
                if references_only:
                    rows.extend(baseline)
                    for label,metrics in old_zero.items():
                        placements.append(old_placements[name,protocol,cid,label])
                        for branch in ('GT','Query'):
                            rows.extend(dict(common,visible=label,hidden=branch,region=k,**v)
                                        for k,v in metrics.items())
                    append(A,'standalone',F2.mean(0),a_center[0],{})
                for label,(pred,cloud) in models_for_frame.items():
                    depth=pred['depth1'][0]
                    aligned,aligned_stats=place(common_parents,gt_depth,gt_bg,depth,K,512)
                    retrieved,retrieved_stats=apply_footprint_order(shared_candidate,depth,K)
                    stats=dict(common,visible=label,GT=aligned_stats,Query=retrieved_stats)
                    if label==OLD_FIXED:
                        compare_placement(old_placements[name,protocol,cid,OLD],stats)
                    placements.append(stats)
                    for branch,hidden in [('GT',aligned),('Query',retrieved)]:
                        merged=merge_layers(cloud,hidden)
                        rgb=raster(merged,K,focus,offsets,*size,F2).mean(0)
                        zero=raster(merged,K,focus,center,*size,a_center)[0]
                        append(label,branch,rgb,zero,zero_for_frame[label])
                save_json(dest/'case.json',dict(rows=rows,placements=placements))
                total_rows.extend(rows)
                total_places.extend(placements)
                print('PRIME_Q3',mode,name,protocol,cid,flush=True)
            del models_for_frame
    report=summarize(total_rows)
    save_json(work/'results.json',dict(rows=total_rows,placements=total_places,**report,
        seconds=time.monotonic()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        max_fixed_visible_gaussians=gauss_peak,K=CONFIG['native_eval_rays'],
        A_definition='standalone SHARP+A-color per-ray F2 source, same focus and masks',
        reused_1b_optical_rows=True,primitive_control_verified_same_placement=True))
    print('PRIME_Q3_DONE',mode,len(total_rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--references', action='store_true')
    args=parser.parse_args()
    guard()
    data.OUT = OLD_RUN
    subprocess.run(['nvidia-smi'],check=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.mem_get_info()[0]<8*2**30:
        raise RuntimeError('Available assigned-GPU memory is insufficient for paired optical evaluation')
    torch.set_num_threads(5)
    torch.cuda.reset_peak_memory_stats()
    run(args.references)
