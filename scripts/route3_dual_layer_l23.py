"""L2/L3 use the same sampled lens, visible layer and input-only Cfill.

Prediction never reads target hidden GT in L3. GT data is loaded only in the
separate CPU evaluation branch. Binary output activation of hidden Gaussians
is equivalent to turning their alpha off outside the band on every lens ray;
it is not the descriptive O3/Oracle splice.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.layered_aperture import pixel_layer,merge_layers
from src.analytic_aperture import sampled
from src.aperture_renderer import linear_to_srgb
from src.dual_layer import depth_band
from src.c1_bokehme import region_sse,patch_metrics
from route3_b2_oracle_3d import sparse_pixel_layer
from route3_c1_bokehme_oracle import lpips_cpu
from types import SimpleNamespace


def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def display(x):return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,1e-12)**(1/2.4)-.055).clip(0,1).astype(np.float32)


def entries(a):return [x for x in json.loads((a.base/'manifests/oracle.json').read_text())['items'] if x['fstop'] in [1.4,2.]]


def render(a,c):
    a.output.mkdir(parents=True,exist_ok=False);groups={};rows=[]
    for x in entries(a):groups.setdefault((x['scene'],x['frame_id']),[]).append(x)
    profile=c['profile'];inputs=json.loads(a.shared.read_text())['items'];prior={}
    if a.stage=='L3':
        prior={r['id']:r for r in json.loads(a.L2_record.read_text())['rows']}
        if any(r.get('observed_sky_visible_layer',False)!=a.include_observed_sky for r in prior.values()):raise ValueError('L2/L3 observed-sky contract mismatch')
    for (scene,frame),items in sorted(groups.items()):
        inp=next(x for x in inputs if x['scene']==scene and x['frame']==frame);d=np.load(inp['input_npz'])
        K=torch.tensor(d['K'],device='cuda',dtype=torch.float32);cam=torch.tensor(d['camera_c2w'],device='cuda',dtype=torch.float32);E=torch.linalg.inv(cam)
        C=torch.tensor(d['visible_rgb_linear'],device='cuda');Z=torch.tensor(d['center_depth'],device='cuda');visible=pixel_layer(C,Z,K,cam,None if a.include_observed_sky else torch.tensor(d['center_surface_valid'],device='cuda'))
        fill_data=np.load(a.fill_root/inp['id']/'fill.npz');fill=torch.tensor(fill_data['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda')
        parent_path=(a.base/'methods-3d/prepared'/inp['id']/'parents.npz') if a.stage=='L2' else a.parents/inp['id']/'parents.npz'
        parent=np.load(parent_path);valid=parent['recoverable'] if a.stage=='L2' else np.ones(len(parent['depth']),bool)
        full_hidden=sparse_pixel_layer(parent,valid,K,cam,profile,'cuda') if a.stage=='L2' else None
        for x in items:
            band,edge,CoC=depth_band(d['center_depth'],d['K'][0,0],x['focus_distance_m'],x['aperture_radius_m'],c['band']['multiplier'],c['band']['relative_inverse_depth_jump'])
            uv=parent['uv'];px=np.clip(uv[:,0].astype(int),0,511);py=np.clip(uv[:,1].astype(int),0,511)
            chosen=valid&band[py,px];hidden=sparse_pixel_layer(parent,chosen,K,cam,profile,'cuda');selected_count=int(chosen.sum())
            torchmask=torch.tensor(band,device='cuda')
            methods=[('O5_shared',visible),('O6_full_shared',merge_layers(visible,full_hidden)),('band_combined',merge_layers(visible,hidden))] if a.stage=='L2' else [('band_combined',merge_layers(visible,hidden))]
            predictions={};times={};peaks={}
            if a.stage=='L3':
                if prior[x['id']]['input_npz']!=inp['input_npz']:raise ValueError('L2/L3 current-visible input differs')
                base=np.load(prior[x['id']]['outputs']['O5_shared']);predictions['O5_shared']={k:torch.tensor(base[k],device='cuda') for k in ['rgb_linear','alpha']}
                times['O5_shared']=prior[x['id']]['timing']['O5_shared'];peaks['O5_shared']=prior[x['id']]['peaks_MiB']['O5_shared']
            for name,pc in methods:
                def run():return sampled(pc,K,E,512,512,x['focus_distance_m'],x['aperture_radius_m'],64,fill,bg,far_plane=1e5 if a.include_observed_sky else 1e4)
                with torch.inference_mode():
                    warm=run();del warm;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();measured=[]
                    for _ in range(3):
                        start=time.monotonic();result=run();torch.cuda.synchronize();measured.append(time.monotonic()-start)
                predictions[name]=result;times[name]=dict(repetitions_seconds=measured,median_seconds=float(np.median(measured)));peaks[name]=torch.cuda.max_memory_allocated()/2**20
            mixed={k:torch.where(torchmask[...,None] if k=='rgb_linear' else torchmask,predictions['band_combined'][k],predictions['O5_shared'][k]) for k in ['rgb_linear','alpha']}
            predictions[a.stage]=mixed;dest=a.output/x['id'];dest.mkdir();paths={}
            for name,value in predictions.items():
                if name=='band_combined':continue
                rgb=linear_to_srgb(value['rgb_linear']).clamp(0,1).cpu().numpy();path=dest/(name+'.npz')
                np.savez_compressed(path,prediction_srgb=rgb,rgb_linear=value['rgb_linear'].cpu().numpy(),alpha=value['alpha'].cpu().numpy(),band=band)
                cv2.imwrite(str(dest/(name+'.png')),np.uint8(rgb[...,::-1]*255));paths[name]=str(path)
            if a.stage=='L2':
                O3=np.load(a.base/'methods-2d/bokehme'/x['id']/'prediction.npz')['prediction_srgb'];oracle=np.load(paths['O6_full_shared'])['prediction_srgb']
                descriptive=np.where(band[...,None],oracle,O3);path=dest/'descriptive-O3-outside.npz';np.savez_compressed(path,prediction_srgb=descriptive);paths['descriptive_O3_outside']=str(path)
            rows.append(dict(id=x['id'],scene=scene,frame=frame,focus_m=x['focus_distance_m'],fstop=x['fstop'],stage=a.stage,
                visible_form='V-G',K=64,band_pixels=int(band.sum()),hidden_parents_available=int(valid.sum()),hidden_parents_selected=selected_count,
                input_npz=inp['input_npz'],parent_input=str(parent_path),fill=str(a.fill_root/inp['id']/'fill.npz'),
                uses_true_hidden_geometry=a.stage=='L2',uses_true_hidden_texture=a.stage=='L2',GT_segmentation_in_fill=False,true_hidden_depth_in_fill=False,
                observed_sky_visible_layer=a.include_observed_sky,sky_depth_is_GT=False,sky_method_depth_m=10000.,far_clip_m=100000. if a.include_observed_sky else 10000.,
                timing=times,peaks_MiB=peaks,outputs=paths,
                two_pass_stage_render_seconds=times['O5_shared']['median_seconds']+times['band_combined']['median_seconds'],
                spatial_activation='only parent centers projected in input-depth band; additionally hidden alpha disabled per output ray outside band; average commutes with fixed binary output activation'))
            write(a.output/'runtime.json',dict(completed=False,rows=rows));print('DUAL_RENDER',a.stage,x['id'],selected_count,flush=True)
            del predictions,hidden
        del visible,full_hidden,parent;torch.cuda.empty_cache()
    write(a.record,dict(completed=True,stage=a.stage,rows=rows,criteria=str(a.criteria),profile=profile,K=64,training=False,
        forbidden_L3_inputs_used=False,old_results_modified=False,descriptive_splice_is_method=False))


def pool(rows):
    n=sum(r['pixels'] for r in rows);s=sum(r['sse'] for r in rows)
    return dict(pixels=n,sse=s,psnr=float(-10*np.log10(max(s/(3*n),1e-12))),patch_lpips=sum(r['pixels']*r['patch_lpips'] for r in rows if r['pixels'])/n) if n else dict(pixels=0,sse=0.,psnr=None,patch_lpips=None)


def geometry_coverage(parent_path,stage,band,g):
    p=np.load(parent_path);uv=p['uv'];xx=np.clip(uv[:,0].astype(int),0,511);yy=np.clip(uv[:,1].astype(int),0,511)
    chosen=band[yy,xx]
    if stage=='L2':chosen&=p['recoverable']
    dep=np.full(512*512,np.inf,np.float32);np.minimum.at(dep,yy[chosen]*512+xx[chosen],p['depth'][chosen]);dep=dep.reshape(512,512)
    supported=g['recoverable']&~g['residual']&band[None]
    q=g['reference_uv_center'][supported];z=g['depth_axial_m'][supported]
    inside=np.isfinite(q).all(-1)&(q[:,0]>=0)&(q[:,0]<512)&(q[:,1]>=0)&(q[:,1]<512)
    estimate=np.full(len(z),np.inf,np.float32);estimate[inside]=dep[np.floor(q[inside,1]).astype(int),np.floor(q[inside,0]).astype(int)]
    match=np.isfinite(estimate)&(abs(estimate-z)<=np.maximum(.005,.02*z));dynamic=g['object_id'][supported]==7
    def result(m):return dict(rays=int(m.sum()),matched_rays=int((match&m).sum()),fraction=float(match[m].mean()) if m.any() else None)
    return dict(all_recoverable=result(np.ones(len(z),bool)),dynamic_hidden_surface=result(dynamic),static_hidden_surface=result(~dynamic),
        definition='lookup nearest selected hidden-parent depth at GT ray surface central projection;within max5mm,2%depth; evaluation only',
        retrieved_parent_pixel_fraction_of_band=float(np.isfinite(dep)[band].mean()),GT_used_only_for_evaluation=True)


def evaluate(a,c):
    a.output.mkdir(parents=True,exist_ok=False);run=json.loads(a.input.read_text());lookup={x['id']:x for x in entries(a)}
    model=lpips_cpu(SimpleNamespace(alexnet_weights=Path('/root/autodl-tmp/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')))
    records=[];allmetrics={};flat=[];plots=[];prior={}
    reuse=json.loads(a.reuse_evaluation.read_text())['per_case_metrics'] if a.reuse_evaluation else {}
    if a.stage=='L3':prior={r['id']:r for r in json.loads(a.L2_record.read_text())['rows']}
    for r in run['rows']:
        x=lookup[r['id']];d=np.load(r['input_npz']);g=np.load(x['geometry_gt_npz'])
        raw=cv2.imread(x['gt_rgb_exr'],cv2.IMREAD_UNCHANGED)[...,:3][...,::-1];target=display(raw)
        band,edge,R=depth_band(d['center_depth'],d['K'][0,0],x['focus_distance_m'],x['aperture_radius_m'],c['band']['multiplier'],c['band']['relative_inverse_depth_jump'])
        residual=g['residual_pixels'].astype(bool)
        center=np.load(a.base/x['scene']/'geometry'/f'f{x["frame_id"]:03d}-center.npz')
        masks=dict(full_image=np.ones(band.shape,bool),in_focus=d['center_surface_valid']&(R<=1),out_of_focus=d['center_surface_valid']&(R>=2),band=band&~residual,
            recoverable=band&g['mask_recoverable_total'],motion_required=band&g['mask_motion_required'],parallax_sufficient=band&g['mask_parallax_sufficient'],dynamic_visible=center['object_id']==7,
            seam=cv2.dilate(band.astype(np.uint8),np.ones((3,3),np.uint8))!=cv2.erode(band.astype(np.uint8),np.ones((3,3),np.uint8)))
        images={'O3':np.load(a.base/'methods-2d/bokehme'/r['id']/'prediction.npz')['prediction_srgb']}
        images.update({k:np.load(p)['prediction_srgb'] for k,p in r['outputs'].items()})
        if a.stage=='L3':images['L2']=np.load(prior[r['id']]['outputs']['L2'])['prediction_srgb']
        allmetrics[r['id']]={}
        for method,pred in images.items():
            if not np.isfinite(pred).all() or pred.shape!=(512,512,3):raise ValueError('Invalid output '+method)
            stats=reuse.get(r['id'],{}).get(method)
            if stats is None:
                ps=patch_metrics(pred,target,np.ones(band.shape,bool),model,patch_size=64,device='cpu');score=np.array(ps['patch_lpips_scores']);stats={}
                for region,mask in masks.items():
                    weight=np.array([mask[y:y+64,z:z+64].sum() for y in range(0,512,64) for z in range(0,512,64)])
                    stats[region]=dict(**region_sse(pred,target,mask),patch_lpips=float(score@weight/weight.sum()) if weight.sum() else None)
            for region in masks:
                flat.append(dict(id=r['id'],scene=r['scene'],focus_m=r['focus_m'],fstop=r['fstop'],frame=r['frame'],method=method,region=region,**stats[region]))
            allmetrics[r['id']][method]=stats
        D=g['mask_disocclusion'].astype(bool);valid_GT=(g['known'].all(0)|D)&~residual
        TP=int((band&D).sum());FP=int((band&~D&valid_GT).sum());FN=int((~band&D).sum())
        records.append(dict(id=r['id'],scene=r['scene'],focus_m=r['focus_m'],fstop=r['fstop'],frame=r['frame'],
            region_pixels={k:int(v.sum()) for k,v in masks.items()},band_image_fraction=float(band.mean()),CoC_visible_in_band_p10_p50_p90=np.quantile(R[band],[.1,.5,.9]).tolist(),
            band_vs_true_disocclusion=dict(TP=TP,FP=FP,FN=FN,recall=TP/max(TP+FN,1),precision=TP/max(TP+FP,1),unknown_GT_band_pixels=int((band&~valid_GT).sum()),residual_pixels=int(residual.sum())),
            geometry_coverage=geometry_coverage(r['parent_input'],a.stage,band,g)))
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
        names=['GT','O3',a.stage]+(['L2'] if a.stage=='L3' else ['O5_shared','O6_full_shared','descriptive_O3_outside']);images['GT']=target
        y,z=np.nonzero(band);cy,cz=np.median(y),np.median(z);closest=np.argmin((y-cy)**2+(z-cz)**2);y0=int(np.clip(y[closest]-64,0,384));z0=int(np.clip(z[closest]-64,0,384))
        fig,axes=plt.subplots(2,len(names),figsize=(3*len(names),6),squeeze=False)
        for col,name in enumerate(names):
            axes[0,col].imshow(images[name]);axes[1,col].imshow(images[name][y0:y0+128,z0:z0+128]);axes[0,col].set_title(name)
            for j in [0,1]:axes[j,col].axis('off')
        fig.suptitle(r['id']+'; input-depth band selected crop;GT only in evaluation');fig.tight_layout();path=a.output/(r['id']+'.png');fig.savefig(path,dpi=120);plt.close(fig);plots.append(str(path))
        print('DUAL_SCORED',a.stage,r['id'],allmetrics[r['id']][a.stage]['band']['psnr'],flush=True)
    pooled={};decisions={};primary=a.stage
    for scene,focuses in c['scope']['focuses'].items():
        pooled[scene]={};decisions[scene]={}
        for focus in focuses:
            cases=[r for r in records if r['scene']==scene and r['focus_m']==focus];ids=[r['id'] for r in cases]
            methods=list(allmetrics[ids[0]]);regions=list(allmetrics[ids[0]][primary])
            table={m:{reg:pool([allmetrics[i][m][reg] for i in ids]) for reg in regions} for m in methods};pooled[scene][str(focus)]=table
            if scene=='B2-low':decisions[scene][str(focus)]=dict(status='low_CoC_descriptive_control',passed=None);continue
            valid=[r for r in cases if r['region_pixels']['band']>=c['min_pixels']];comparisons={}
            for baseline in ['O5_shared','O3']:
                if baseline not in table:continue
                left=table[primary]['band'];right=table[baseline]['band'];gain=left['psnr']-right['psnr'];delta=left['patch_lpips']-right['patch_lpips']
                noise=sum(c['noise']['per_lens_band'][f'N{r["fstop"]:g}-F{focus:g}']*r['region_pixels']['band'] for r in cases)/sum(r['region_pixels']['band'] for r in cases)
                quality=gain>=.5 or (-delta>=.1*right['patch_lpips'] and -delta>3*noise)
                secondary=gain>=-.2 and delta<=3*noise
                same=0;ties=0
                for r in valid:
                    i=r['id'];l=allmetrics[i][primary]['band'];b=allmetrics[i][baseline]['band'];n=c['noise']['per_lens_band'][f'N{r["fstop"]:g}-F{focus:g}']
                    same+=l['psnr']>b['psnr'] and l['patch_lpips']-b['patch_lpips'] < -3*n
                    ties+=abs(l['patch_lpips']-b['patch_lpips'])<=3*n
                guards={reg:table[primary][reg]['psnr']-table['O3'][reg]['psnr'] for reg in ['full_image','in_focus']}
                guardpass=all(v>=-.2 for v in guards.values());stable=len(valid)>=6 and same>=5
                comparisons[baseline]=dict(PSNR_gain_dB=gain,LPIPS_change=delta,relative_LPIPS_change=delta/right['patch_lpips'],noise=noise,LPIPS_noise_ties=ties,
                    quality_passed=quality,secondary_metric_passed=secondary,same_direction_count=int(same),valid_groups=len(valid),required_same_direction=5,
                    stable_passed=stable,O3_guardrails_dB=guards,guardrails_passed=guardpass,passed=quality and secondary and stable and guardpass)
            decision=dict(comparisons=comparisons,passed=all(v['passed'] for v in comparisons.values()) and len(comparisons)==2)
            if a.stage=='L3':
                mseO3=table['O3']['band']['sse']/(3*table['O3']['band']['pixels']);mse2=table['L2']['band']['sse']/(3*table['L2']['band']['pixels']);mse3=table['L3']['band']['sse']/(3*table['L3']['band']['pixels'])
                decision['retained_L2_MSE_reduction_vs_O3']=(mseO3-mse3)/(mseO3-mse2) if mseO3>mse2 else None
                decision['L3_minus_L2']=dict(PSNR_dB=table['L3']['band']['psnr']-table['L2']['band']['psnr'],LPIPS=table['L3']['band']['patch_lpips']-table['L2']['band']['patch_lpips'])
            decisions[scene][str(focus)]=decision
    write(a.record,dict(completed=True,stage=a.stage,criteria=str(a.criteria),cases=records,per_case_metrics=allmetrics,pooled=pooled,decisions=decisions,visualizations=plots,
        DrBokeh='excluded: prerequisite focus check failed',method_results_include_descriptive_splice=False,formal_visible='V-G',VP='L0 diagnosis only, not promoted',GT_read_in_prediction=False if a.stage=='L3' else 'L2 explicit GT Oracle'))
    import csv
    with a.record.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['render','evaluate'],required=True);p.add_argument('--stage',choices=['L2','L3'],required=True)
    p.add_argument('--base',type=Path,default=ROOT/'outputs/route3-b2only-2026-11-10-01');p.add_argument('--criteria',type=Path,required=True);p.add_argument('--shared',type=Path,required=True)
    p.add_argument('--fill-root',type=Path,required=True);p.add_argument('--parents',type=Path);p.add_argument('--L2-record',dest='L2_record',type=Path)
    p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--reuse-evaluation',type=Path);p.add_argument('--include-observed-sky',action='store_true');a=p.parse_args()
    c=json.loads(a.criteria.read_text());assert c['frozen'];torch.set_num_threads(8 if a.mode=='evaluate' else 4);cv2.setNumThreads(1)
    if a.record.exists():raise FileExistsError(a.record)
    if a.mode=='render':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0 required')
        subprocess.run(['nvidia-smi'],check=True);render(a,c)
    else:evaluate(a,c)


if __name__=='__main__':main()
