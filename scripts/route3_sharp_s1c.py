"""S1c input-only dense flow, geometric support and inverse-depth fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.dense_flow_geometry import pair_depth,multiview_depth,stratified_points
from route3_sharp_s1b_calibrate import fit_inverse,inverse_stats,bilinear
from route3_upper_bound_evaluate import BASE,OUT as UPPER,RECORDS
from route3_sharp_s1 import OUT as S1,FRAMES
OUT=ROOT/'outputs/route3-sharp-s1c-2026-11-10-01';PREFIX='route3-sharp-s1c-2026-11-10'
CRITERIA=RECORDS/'route3-sharp-s1c-criteria-2026-11-10.json'
WEIGHTS=ROOT/'models/metrics/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth'


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def stop(reason,record):
    write(OUT/'stop.json',dict(reason=reason,record=record,scope_expansion_stopped=True));raise RuntimeError('S1C_STOP '+reason)


def select_pairs():
    selected={};directed=set()
    for variant in ['E1','E2']:
        poses={f:np.load(UPPER/'U2-da3'/variant/f'f{f:03d}.npz')['c2w'] for f in FRAMES};selected[variant]={}
        for f in FRAMES:
            candidates=[]
            for g in FRAMES:
                if f==g:continue
                delta=poses[g][:3,3]-poses[f][:3,3];baseline=float(np.linalg.norm(delta))
                if baseline<.01:continue
                local=poses[f][:3,:3].T@delta;angle=np.arctan2(local[1],local[0]);quadrant=int(np.floor((angle+np.pi)/(np.pi/2)))%4
                candidates.append(dict(frame=g,baseline_m=baseline,quadrant=quadrant))
            candidates.sort(key=lambda x:(-x['baseline_m'],x['frame']));chosen=[]
            for quadrant in range(4):
                row=next((x for x in candidates if x['quadrant']==quadrant),None)
                if row is not None:chosen.append(row)
            for row in candidates:
                if row not in chosen and len(chosen)<6:chosen.append(row)
            chosen.sort(key=lambda x:(-x['baseline_m'],x['frame']));selected[variant][str(f)]=chosen
            for row in chosen:directed.add((f,row['frame']));directed.add((row['frame'],f))
    result=dict(selected=selected,directed_pairs=sorted(directed));write(OUT/'flow-plan.json',result);return result


def flow(c):
    from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
    import torchvision
    import torchvision.version
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
    subprocess.run(['nvidia-smi'],check=True);plan=select_pairs();folder=OUT/'flow';folder.mkdir(exist_ok=False)
    net=raft_large(weights=None,progress=False).eval();net.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True);net.to('cuda');transform=Raft_Large_Weights.C_T_SKHT_V2.transforms()
    images={f:torch.tensor(np.uint8(np.rint(np.load(S1/'depth'/f'f{f:03d}.npz')['rgb_srgb']*255))).permute(2,0,1)[None].cuda() for f in FRAMES};rows=[]
    for a,b in plan['directed_pairs']:
        first,second=transform(images[a],images[b]);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        with torch.inference_mode():result=net(first,second,num_flow_updates=32)[-1]
        torch.cuda.synchronize();seconds=time.monotonic()-start;value=result[0].permute(1,2,0).cpu().numpy()
        if not np.isfinite(value).all():stop('nonfinite dense optical flow','flow')
        path=folder/f'f{a:03d}-f{b:03d}.npz';np.savez_compressed(path,flow=value)
        rows.append(dict(first=a,second=b,path=str(path),seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20));print('DENSE_FLOW',a,b,seconds,flush=True)
        write(OUT/'flow-progress.json',dict(completed=False,rows=rows))
    record=dict(completed=True,rows=rows,model='TorchVision official RAFT-large C_T_SKHT_V2',torchvision=torchvision.__version__,implementation_revision=torchvision.version.git_version,weights=str(WEIGHTS),
        original_RAFT_download_failed='Official Dropbox direct request failed ENETUNREACH; use existing official TorchVision public model under user-authorized fallback',
        iterations=32,resolution=[512,512],mixed_precision=False,training=False,GT_depth_or_masks_read=False)
    write(OUT/'flow.json',record);write(RECORDS/(PREFIX+'-flow.json'),record)


def support_stats(xy,depth,held):
    train=~held;cells=set((np.floor(xy[train,1]/128)*4+np.floor(xy[train,0]/128)).astype(int).tolist());near=depth<2.5;middle=(depth>=2.5)&(depth<4);far=depth>=4
    span=float(np.quantile(1/depth[train],.9)-np.quantile(1/depth[train],.1)) if train.any() else 0.
    values=dict(points=len(depth),train_points=int(train.sum()),heldout_points=int(held.sum()),occupied_train_cells=len(cells),train_inverse_depth_span_P90_P10=span,
        strata={name:dict(all=int(m.sum()),train=int((m&train).sum()),heldout=int((m&held).sum())) for name,m in [('near',near),('middle',middle),('far',far)]},depth_p10_p50_p90=np.quantile(depth,[.1,.5,.9]).tolist() if len(depth) else None)
    values['supported']=bool(train.sum()>=100 and held.sum()>=25 and len(cells)>=6 and span>=.03 and (near&train).sum()>=25 and (near&held).sum()>=10 and (far&train).sum()>=25 and (far&held).sum()>=10)
    return values


def triangulate_calibrate(c):
    plan=json.loads((OUT/'flow-plan.json').read_text());folder=OUT/'calibration';folder.mkdir(exist_ok=False);rows=[];start=time.monotonic()
    K=np.load(S1/'depth/f001.npz')['K'].astype(np.float64)
    for variant_index,variant in enumerate(['E1','E2']):
        poses={f:np.load(UPPER/'U2-da3'/variant/f'f{f:03d}.npz')['c2w'] for f in FRAMES}
        for f in FRAMES:
            pairs=[];audits=[]
            for row in plan['selected'][variant][str(f)]:
                g=row['frame'];forward=np.load(OUT/'flow'/f'f{f:03d}-f{g:03d}.npz')['flow'];back=np.load(OUT/'flow'/f'f{g:03d}-f{f:03d}.npz')['flow']
                z,uv,stats=pair_depth(forward,back,K,poses[f],poses[g]);pairs.append(dict(frame=g,depth=z,other_xy=uv));audits.append(dict(source_frame=g,**stats))
            depth,valid,support,count=multiview_depth(pairs,K,poses[f],poses);xy,tri,held=stratified_points(depth,valid,f,variant_index);stats=support_stats(xy,tri,held)
            dest=folder/variant;dest.mkdir(exist_ok=True);np.savez_compressed(dest/f'f{f:03d}-reference.npz',depth=depth,valid=valid,support=support,initial_pair_support=count,xy=xy,triangulated_depth=tri,heldout=held)
            result=dict(variant=variant,frame=f,dense_valid_pixels=int(valid.sum()),support=stats,pairs=audits,models={});rows.append(result)
            record=dict(completed=False,rows=rows,seconds=time.monotonic()-start,GT_depth_or_mask_used_in_geometric_filter_or_fit=False)
            if not stats['supported']:
                record['stop_reason']=f'{variant} frame{f}: dense triangulation below frozen support';write(OUT/'calibration.json',record);write(RECORDS/(PREFIX+'-calibration.json'),record);stop(record['stop_reason'],'calibration')
            for source in ['SHARP','DA3']:
                data=np.load(S1/'depth'/f'f{f:03d}.npz') if source=='SHARP' else np.load(UPPER/'U2-da3/E1'/f'f{f:03d}.npz');pred=data['depth'];observed=bilinear(pred,xy)
                fitted=fit_inverse(observed[~held],tri[~held],pred);before=inverse_stats(observed[held],tri[held]);after=inverse_stats(observed[held],tri[held],fitted['a'],fitted['b'])
                improved=after['inverse_RMSE']<=before['inverse_RMSE']+1e-10 and after['sparse_depth_AbsRel']<=before['sparse_depth_AbsRel']+1e-10
                result['models'][source]=dict(**fitted,heldout_before=before,heldout_after=after,heldout_not_worse=improved,
                    train_before=inverse_stats(observed[~held],tri[~held]),train_after=inverse_stats(observed[~held],tri[~held],fitted['a'],fitted['b']))
                if not improved:
                    record['stop_reason']=f'{variant} frame{f} {source}: heldout calibration error worsened';write(OUT/'calibration.json',record);write(RECORDS/(PREFIX+'-calibration.json'),record);stop(record['stop_reason'],'calibration')
                z=1/np.maximum(fitted['a']/pred+fitted['b'],1e-4);target=dest/source;target.mkdir(exist_ok=True);np.savez_compressed(target/f'f{f:03d}.npz',depth=z.astype(np.float32),K=data['K'],c2w=poses[f])
            print('DENSE_CALIB',variant,f,stats['points'],stats['strata'],flush=True);write(OUT/'calibration-progress.json',record)
    report=dict(completed=True,rows=rows,seconds=time.monotonic()-start,GT_depth_or_masks_used_for_fit=False)
    write(OUT/'calibration.json',report);write(RECORDS/(PREFIX+'-calibration.json'),report)


def report_geometry(c):
    """GT read only here, after geometric selection/fitting are final."""
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    from route3_sharp_s1_evaluate import edge_maps,depth_stats
    result=json.loads((OUT/'calibration.json').read_text());folder=OUT/'geometry-report';folder.mkdir(exist_ok=False);rows=[];figures=[]
    for row in result['rows']:
        f=row['frame'];variant=row['variant'];d=np.load(OUT/'calibration'/variant/f'f{f:03d}-reference.npz');g,edge,_,_=edge_maps(f);valid=d['valid'];known=g['center_visible'];tri=d['depth'];xy=d['xy'];z=d['triangulated_depth'];x=xy[:,0].astype(int);y=xy[:,1].astype(int)
        geometry=ROOT/'outputs/route3-a2b-2026-11-10-01/blender-b2/B2-high/scene-geometry.npz'
        dynamic=np.isin(g['object_id'],np.load(geometry)['dynamic_object_ids'])
        metrics=dict(variant=variant,frame=f,triangulation=depth_stats(tri,g['depth_axial_m'],valid&known),selected_points_GT=depth_stats(z,g['depth_axial_m'][y,x],known[y,x]),
            dynamic_pixels=int(dynamic.sum()),dynamic_accepted=int((dynamic&valid).sum()),dynamic_rejected_fraction=float((dynamic&~valid).sum()/dynamic.sum()) if dynamic.any() else None,
            static_pixels=int((known&~dynamic).sum()),static_accepted=int((known&~dynamic&valid).sum()),calibrated={})
        for source,fitted in row['models'].items():
            pred=np.load(S1/'depth'/f'f{f:03d}.npz')['depth'] if source=='SHARP' else np.load(UPPER/'U2-da3/E1'/f'f{f:03d}.npz')['depth'];mapped=1/np.maximum(fitted['a']/pred+fitted['b'],1e-4)
            masks=dict(full=known,boundary=edge,near=known&(g['depth_axial_m']<2.5),middle=known&(g['depth_axial_m']>=2.5)&(g['depth_axial_m']<4),far=known&(g['depth_axial_m']>=4))
            metrics['calibrated'][source]={key:dict(before=depth_stats(pred,g['depth_axial_m'],mask),after=depth_stats(mapped,g['depth_axial_m'],mask)) for key,mask in masks.items() if mask.any()}
        rows.append(metrics);fig,axes=plt.subplots(1,3,figsize=(12,4));axes[0].imshow(np.load(S1/'depth'/f'f{f:03d}.npz')['rgb_srgb']);axes[0].scatter(x,y,c=z,s=2,cmap='viridis',vmin=1,vmax=6);axes[0].set_title('Balanced sparse sample, color=tri depth');axes[1].imshow(np.where(valid,tri,np.nan),vmin=1,vmax=6);axes[1].set_title('Dense geometric reference');axes[2].hist(z,bins=40);axes[2].set_xlabel('triangulated axial depth(m)')
        fig.suptitle(f'{variant} frame{f}; GT unused duringselection/fit');fig.tight_layout();path=folder/f'{variant}-f{f:03d}.png';fig.savefig(path,dpi=140);plt.close(fig);figures.append(str(path))
    write(RECORDS/(PREFIX+'-geometry-evaluation.json'),dict(completed=True,calibration_completed=result['completed'],rows=rows,visualizations=figures,GT_evaluation_only=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['flow','triangulate_calibrate','report_geometry'],required=True);a=p.parse_args();torch.set_num_threads(8);cv2.setNumThreads(1);c=json.loads(CRITERIA.read_text());assert c['frozen'];globals()[a.mode](c)


if __name__=='__main__':main()
