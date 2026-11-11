"""One fixed S1d stage at a time, with immediate focus-loss diagnostics."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import numpy as np
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1d import OUT,PREFIX,CRITERIA,S1C,BASE,RECORDS,entries,write
from route3_sharp_s1c_apply import anchored
from route3_upper_bound_evaluate import items,display,exr,masks_for,pool as pooled_score
from route3_sharp_s1d_diagnose import associate
from route3_b2_oracle_3d import sparse_pixel_layer
from src.dual_layer import depth_band
from src.aperture_renderer import GaussianScene
from src.layered_aperture import merge_layers
from src.analytic_aperture import sampled


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['R1','R2','R3','R4','Oracle_depth_only','Oracle_color_only','Oracle_missing_only'],required=True);a=p.parse_args();torch.set_num_threads(8);c=json.loads(CRITERIA.read_text());assert c['frozen'] and (RECORDS/(PREFIX+'-decomposition.json')).exists()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
    subprocess.run(['nvidia-smi'],check=True);folder=OUT/'render'/a.method;folder.mkdir(parents=True,exist_ok=False);old=json.loads((RECORDS/'route3-sharp-s1c-2026-11-10-evaluation.json').read_text());profile=c['source_profile'];rows=[];diagnostics=[];focus=[];previous=[];current_frame=None
    for case in items():
        frame=case['frame_id'];entry=next(e for e in entries() if e['frame']==frame)
        if frame!=current_frame:
            if current_frame is not None:del cloud;torch.cuda.empty_cache()
            cloud=anchored(entry,'A_color');d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');E=torch.eye(4,device='cuda');fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda');current_frame=frame
        lens=next(x for x in entry['cases'] if x['id']==case['id']);band=depth_band(d['depth'],d['K'][0,0],lens['focus_m'],case['aperture_radius_m'])[0];source=OUT/'candidates'/f'f{frame:03d}'/(a.method+'.npz') if a.method.startswith('R') else OUT/'replacements'/case['id']/(a.method+'.npz');parent=np.load(source);uv=parent['uv'];x=np.clip(uv[:,0].astype(int),0,511);y=np.clip(uv[:,1].astype(int),0,511);selected=band[y,x];hidden=sparse_pixel_layer(parent,selected,K,E,profile,'cuda');basepath=old['predictions'][case['id']]['cal_E1__A_color'];base=np.load(basepath)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        with torch.inference_mode():r=sampled(merge_layers(cloud,hidden),K,E,512,512,lens['focus_m'],case['aperture_radius_m'],64,fill,bg,far_plane=1e5,return_fill_contribution=True)
        torch.cuda.synchronize();seconds=time.monotonic()-start;values={k:np.where(band[...,None] if v.ndim==3 else band,v.cpu().numpy(),base[k]) for k,v in r.items()};image=display(values['rgb_linear']);path=folder/(case['id']+'.npz');np.savez_compressed(path,prediction_srgb=image,**values)
        rows.append(dict(id=case['id'],method=a.method,path=str(path),seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,parents=int(selected.sum()),Gaussians=len(hidden.means)))
        # GT is first read after writing the ordinary prediction above.
        center=np.load(BASE/'B2-high/geometry'/f'f{frame:03d}-center.npz');known=center['center_visible'][y,x];zvis=center['depth_axial_m'][y,x];wrong=known&(parent['depth']<zvis-np.maximum(.001,.005*zvis));gt=np.load(OUT/'geometry'/(case['id']+'.npz'));eligible=gt['recoverable'];q=gt['uv'][eligible];rec_pixel=gt['pixel'][eligible];ass=associate(parent,q,band);hit=(ass>=0)&band.flat[rec_pixel]&band[np.clip(q[:,1].astype(int),0,511),np.clip(q[:,0].astype(int),0,511)];safe=np.maximum(ass,0)
        good_depth=hit&(abs(parent['depth'][safe]-gt['depth'][eligible])<=np.maximum(.02,.02*gt['depth'][eligible])) if len(parent['depth']) else np.zeros(len(q),bool)
        colorvalid=gt['GT_parent'][eligible]>=0;good_color=good_depth&colorvalid&(abs(parent['colors_linear'][safe]-gt['GT_color'][eligible]).max(-1)<=.05) if len(parent['depth']) else np.zeros(len(q),bool)
        tagparent={key:parent[key] for key in parent.files};tagparent['colors_linear']=np.repeat(wrong[:,None].astype(np.float32),3,axis=1);taghidden=sparse_pixel_layer(tagparent,selected,K,E,profile,'cuda');visible_tag=GaussianScene(cloud.means,cloud.quats,cloud.scales,cloud.opacities,torch.zeros_like(cloud.colors),cloud.source_frames)
        with torch.inference_mode():tag=sampled(merge_layers(visible_tag,taghidden),K,E,512,512,lens['focus_m'],case['aperture_radius_m'],64,torch.zeros_like(fill),bg,far_plane=1e5)
        heat=np.where(band,tag['rgb_linear'][...,0].cpu().numpy(),0);masks=masks_for(case,c);truth=display(exr(case['gt_rgb_exr']));mask=masks['in_focus'];sse=float(((image.astype(np.float64)-truth)**2)[mask].sum());focus.append(dict(id=case['id'],focus_m=case['focus_distance_m'],pixels=int(mask.sum()),sse=sse,psnr=float(-10*np.log10(max(sse/(3*mask.sum()),1e-12))),patch_lpips=0.))
        vismetric=old['per_case_metrics']['Cycles'][case['id']]['cal_E1__A_color']['in_focus'];previous.append(dict(focus_m=case['focus_distance_m'],**vismetric))
        diagnostic=dict(id=case['id'],method=a.method,focus_m=case['focus_distance_m'],parents=int(selected.sum()),known_GT_parents=int((selected&known).sum()),wrong_parents=int((selected&wrong).sum()),wrong_fraction=float((selected&wrong).sum()/max((selected&known).sum(),1)),rec_rays=int(eligible.sum()),covered_rec_rays=int(hit.sum()),depth_correct_rec_rays=int(good_depth.sum()),depth_color_correct_rec_rays=int(good_color.sum()),tag_wrong_mean_full=float(heat.mean()),tag_wrong_mean_band=float(heat[band].mean()),focus_error_in_wrong_tag_pixels_SSE=float((((image.astype(np.float64)-truth)**2).sum(-1)*(heat>.001)*mask).sum()),all_RGB_changes_inside_fixed_band=bool(np.array_equal(image[~band],base['prediction_srgb'][~band])))
        assert diagnostic['all_RGB_changes_inside_fixed_band'];diagnostics.append(diagnostic);np.savez_compressed(folder/(case['id']+'-diagnostic.npz'),wrong_tag_alpha=heat);print('S1D_RENDER',a.method,case['id'],seconds,flush=True);del hidden,taghidden,visible_tag,r,tag
    summary={}
    for f in [1.4,5.]:
        now=pooled_score([x for x in focus if x['focus_m']==f]);before=pooled_score([x for x in previous if x['focus_m']==f]);summary[str(f)]=dict(in_focus_PSNR=now['psnr'],visible_PSNR=before['psnr'],change_vs_visible_dB=now['psnr']-before['psnr'])
        if a.method in ['R2','R3','R4']:
            prior=json.loads((RECORDS/(PREFIX+'-render-R'+str(int(a.method[1])-1)+'.json')).read_text());summary[str(f)]['change_vs_preceding_stage_dB']=now['psnr']-prior['focus_quick_check'][str(f)]['in_focus_PSNR']
    write(RECORDS/(PREFIX+'-render-'+a.method+'.json'),dict(completed=True,rows=rows,diagnostics=diagnostics,focus_quick_check=summary,fixed_visible_and_fill=True,GT_only_post_prediction_diagnostic=True));write(folder/'predictions.json',dict(completed=True,rows=rows));print('S1D_FOCUS_CHECK',a.method,summary,flush=True)


if __name__=='__main__':main()
