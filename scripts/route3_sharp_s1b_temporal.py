"""S1b native eight-frame diagnosis, with two separately authorized focuses."""
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
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1b import OUT,PREFIX,CRITERIA,write,fitted_scene
from route3_sharp_s1 import OUT as S1,WEIGHT,OFFICIAL,raster
from route3_upper_bound_evaluate import BASE,RECORDS,display,exr
from src.aperture_renderer import GaussianScene
from src.sharp_anchor import anchor
TEMP=OUT/'temporal';FRAMES=list(range(27,35))


def full_scene(frame):
    source=(S1/'gaussians'/f'f{frame:03d}.pt') if frame==30 else TEMP/'gaussians'/f'f{frame:03d}.pt'
    return GaussianScene(**torch.load(source,map_location='cuda',weights_only=True))


def infer(c):
    from sharp.models import PredictorParams,create_predictor
    from sharp.cli.predict import predict_image
    (TEMP/'gaussians').mkdir(parents=True,exist_ok=False);(TEMP/'sharp-depth').mkdir();(TEMP/'DA3-images').mkdir()
    pin=BASE/'inputs/B2-high/pinhole';timing={r['frame']:r for r in json.loads((pin/'timings.json').read_text())}
    state=torch.load(WEIGHT,map_location='cpu',weights_only=True);predictor=create_predictor(PredictorParams());predictor.load_state_dict(state,strict=True);del state
    predictor.eval().to('cuda');K=torch.tensor([[512*50/36,0,256],[0,512*50/36,256],[0,0,1]],device='cuda',dtype=torch.float32);rows=[];paths=[]
    for frame in FRAMES:
        source=pin/(timing[frame]['key']+'.exr');raw=np.clip(exr(source),0,1);rgb=display(raw)
        image=TEMP/'DA3-images'/f'f{frame:03d}.png';cv2.imwrite(str(image),np.uint8(np.rint(rgb[...,::-1]*255)));paths.append(str(image))
        if frame==30:
            old=np.load(S1/'depth/f030.npz');z=old['depth'];valid=old['valid'];seconds=0.;peak=0.
        else:
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();p=predict_image(predictor,rgb*255,float(K[0,0]),torch.device('cuda'));torch.cuda.synchronize();seconds=time.monotonic()-start
            n=p.mean_vectors.shape[1];assert n==1179648
            d=dict(means=p.mean_vectors[0],quats=p.quaternions[0],scales=p.singular_values[0],opacities=p.opacities[0].reshape(-1),colors=p.colors[0],source_frames=torch.arange(2,device='cuda').repeat_interleave(n//2))
            torch.save({k:v.detach().cpu() for k,v in d.items()},TEMP/'gaussians'/f'f{frame:03d}.pt')
            with torch.inference_mode():_,dep,alpha=raster(GaussianScene(**{k:v[:n//2] for k,v in d.items()}),K)
            z=dep.cpu().numpy();valid=np.isfinite(z)&(z>0)&(alpha.cpu().numpy()>=.1)
            if not valid.any():raise RuntimeError('S1B_STOP invalid temporal SHARP depth')
            if not valid.all():indices=distance_transform_edt(~valid,return_distances=False,return_indices=True);z[~valid]=z[tuple(indices[:,~valid])]
            peak=torch.cuda.max_memory_allocated()/2**20;del p,d
        np.savez_compressed(TEMP/'sharp-depth'/f'f{frame:03d}.npz',depth=z,rgb_srgb=rgb,rgb_linear=raw,K=K.cpu().numpy(),valid=valid)
        rows.append(dict(frame=frame,SHARP_seconds=seconds,SHARP_peak_MiB=peak,reused_S1=frame==30));print('TEMP_SHARP',frame,seconds,flush=True)
    del predictor;torch.cuda.empty_cache()
    from depth_anything_3.api import DepthAnything3
    from route3_upper_bound_da3 import sim3
    model=DepthAnything3.from_pretrained(str(ROOT/'models/DA3-BASE')).to('cuda:0').eval();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    pred=model.inference(paths,process_res=518,process_res_method='upper_bound_resize',export_dir=None);torch.cuda.synchronize();seconds=time.monotonic()-start;peak=torch.cuda.max_memory_allocated()/2**20
    poses=np.broadcast_to(np.eye(4),(8,4,4)).copy();poses[:,:3,:]=pred.extrinsics[:,:3,:];poses=np.linalg.inv(poses)
    gt=np.stack([np.load(BASE/'B2-high/geometry'/f'f{f:03d}-center.npz')['camera_c2w'] for f in FRAMES]);scale,R,t,sv=sim3(poses[:,:3,3],gt[:,:3,3]);aligned=poses.copy();aligned[:,:3,:3]=R@poses[:,:3,:3];aligned[:,:3,3]=scale*(poses[:,:3,3]@R.T)+t
    directory=TEMP/'DA3-depth';directory.mkdir()
    for i,frame in enumerate(FRAMES):
        depth=cv2.resize(pred.depth[i].astype(np.float32),(512,512),interpolation=cv2.INTER_LINEAR)*scale
        np.savez_compressed(directory/f'f{frame:03d}.npz',depth=depth,K=K.cpu().numpy(),c2w=aligned[i])
    record=dict(completed=True,frames=FRAMES,SHARP=rows,DA3=dict(joint_input_frames=FRAMES,seconds=seconds,peak_MiB=peak,global_Sim3_scale=scale,R=R.tolist(),translation=t.tolist(),
        camera_center_RMSE_m=float(np.sqrt(np.mean(np.sum((aligned[:,:3,3]-gt[:,:3,3])**2,axis=1)))),GT_depth_read=False,
        no_per_frame_GT_depth_alignment=True,context_different_from_main13=True,network_finetuning=False))
    write(TEMP/'inference.json',record);write(RECORDS/(PREFIX+'-temporal-inference.json'),record)


def prepare(c):
    from src.dual_layer import depth_band
    directory=TEMP/'prepared';directory.mkdir(exist_ok=False);rows=[];fills=[]
    old=json.loads((S1/'prepared.json').read_text())['rows']
    for frame in FRAMES:
        sharp=np.load(TEMP/'sharp-depth'/f'f{frame:03d}.npz');g=np.load(BASE/'B2-high/geometry'/f'f{frame:03d}-center.npz');K=sharp['K']
        for source in ['SHARP','DA3']:
            z=sharp['depth'] if source=='SHARP' else np.load(TEMP/'DA3-depth'/f'f{frame:03d}.npz')['depth'];union=np.zeros((512,512),bool);maxradius=0.;cases=[]
            for focus in [1.4,5.]:
                for stop in [1.4,2.]:
                    nominal=abs(g['K'][0,0]*(.05/(2*stop))*(1/np.maximum(g['depth_axial_m'],1e-8)-1/focus));selection=g['center_visible']&(nominal<=1)
                    chosen=float(np.median(z[selection]));band,_,radius=depth_band(z,K[0,0],chosen,.05/(2*stop));union|=band;maxradius=max(maxradius,float(radius.max()))
                    if stop==1.4:cases.append(dict(frame=frame,nominal_focus_m=focus,focus_m=chosen,fstop=stop,id=f'B2-high-f{frame:03d}-N1.4-F{focus:g}'))
            name=f'{source}-f{frame:03d}';path=directory/(name+'.npz')
            if source=='SHARP' and frame==30:
                prior=next(x for x in old if x['scale_label']=='native' and x['frame']==frame);path=Path(prior['input_npz']);fill=str(S1/'fill'/prior['id']/'fill.npz')
                for case in cases:assert case['focus_m']==next(x['focus_m'] for x in prior['cases'] if x['id']==case['id'])
            else:
                mask=cv2.dilate(union.astype(np.uint8),np.ones((5,5),np.uint8),iterations=2)>0;radius=int(np.ceil(2*maxradius))+4
                bg=cv2.dilate(z,np.ones((2*radius+1,2*radius+1),np.uint8))
                np.savez_compressed(path,depth=z,K=K,rgb_srgb=sharp['rgb_srgb'],rgb_linear=sharp['rgb_linear'],fill_mask=mask,background_depth=bg)
                fills.append(dict(id=name,input_npz=str(path),uses_GT_segmentation=False,uses_true_hidden_depth=False));fill=str(TEMP/'fill'/name/'fill.npz')
            rows.append(dict(id=name,frame=frame,source=source,input_npz=str(path),fill=fill,cases=cases))
    write(TEMP/'prepared.json',dict(completed=True,rows=rows));write(TEMP/'fill-manifest.json',dict(items=fills))


def fit(c):
    entries=json.loads((TEMP/'prepared.json').read_text())['rows'];rows=[]
    for frame in FRAMES:
        if frame==30:
            for variant in ['A_color','A_color_opacity']:
                prior=json.loads((OUT/'anchor/native/f030'/variant/'result.json').read_text());assert prior['completed'];rows.append(dict(frame=frame,method=variant,reused_main=True,attributes=prior['attributes']))
            continue
        entry=next(x for x in entries if x['frame']==frame and x['source']=='SHARP');d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');target=torch.tensor(d['rgb_srgb'],device='cuda');base=full_scene(frame)
        for variant,op in [('A_color',False),('A_color_opacity',True)]:
            folder=TEMP/'anchor'/f'f{frame:03d}'/variant;folder.mkdir(parents=True,exist_ok=False);torch.manual_seed(2026111121+frame)
            fitted,result=anchor(base,K,fill,target,op,lambda value:write(folder/'monitor.json',value))
            path=folder/'attributes-final.pt';torch.save(dict(colors=fitted.colors.detach().cpu(),opacities=fitted.opacities.detach().cpu(),completed_updates=result['completed_updates']),path)
            result.update(frame=frame,method=variant,attributes=str(path));write(folder/'result.json',result);write(RECORDS/(PREFIX+f'-anchor-temporal-f{frame:03d}-{variant}.json'),result)
            if not result['completed']:raise RuntimeError('S1B_STOP temporal anchoring '+str(result['stop_reason']))
            rows.append(dict(frame=frame,method=variant,reused_main=False,attributes=str(path)))
        del base;torch.cuda.empty_cache()
    write(TEMP/'anchor.json',dict(completed=True,rows=rows))


def bokehme(c):
    from src.baselines.bokehme import setup,predict
    pipeline,settings,nets,_=setup(ROOT.parent/'BokehMe',ROOT/'models/BokehMe',dict(gamma=2.2,highlight=False,defocus_scale=10.))
    root=TEMP/'O3';root.mkdir(exist_ok=False);rows=[]
    for entry in json.loads((TEMP/'prepared.json').read_text())['rows']:
        d=np.load(entry['input_npz'])
        for case in entry['cases']:
            control=(d['K'][0,0]*.05/2.8*(1/d['depth']-1/case['focus_m'])/10).astype(np.float32)
            pred,_=predict(pipeline,settings,nets,np.ascontiguousarray(d['rgb_srgb']),np.ascontiguousarray(control));path=root/(case['id']+'-O3_'+entry['source']+'.npz');np.savez_compressed(path,prediction_srgb=pred.clip(0,1))
            rows.append(dict(id=case['id'],frame=entry['frame'],focus_m=case['nominal_focus_m'],method='O3_'+entry['source'],path=str(path)))
    write(TEMP/'O3-predictions.json',dict(completed=True,rows=rows))


def render(c):
    from src.layered_aperture import pixel_layer
    from src.analytic_aperture import sampled
    directory=TEMP/'render';directory.mkdir(exist_ok=False);entries=json.loads((TEMP/'prepared.json').read_text())['rows'];attributes=json.loads((TEMP/'anchor.json').read_text())['rows'];rows=[]
    for frame in FRAMES:
        sd=next(x for x in entries if x['frame']==frame and x['source']=='SHARP');dd=next(x for x in entries if x['frame']==frame and x['source']=='DA3');s=np.load(sd['input_npz']);d=np.load(dd['input_npz']);K=torch.tensor(s['K'],device='cuda');E=torch.eye(4,device='cuda');base=full_scene(frame)
        vg=pixel_layer(torch.tensor(d['rgb_linear'],device='cuda'),torch.tensor(d['depth'],device='cuda'),K,E)
        for method in ['VG_DA3','SHARP_full','A_color','A_color_opacity']:
            cloud=vg if method=='VG_DA3' else base
            if method.startswith('A_color'):
                a=next(r for r in attributes if r['frame']==frame and r['method']==method);p=torch.load(a['attributes'],map_location='cuda',weights_only=True)
                cloud=GaussianScene(base.means,base.quats,base.scales,p['opacities'],p['colors'],base.source_frames)
            entry=dd if method=='VG_DA3' else sd;data=d if method=='VG_DA3' else s;fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');bg=torch.tensor(data['background_depth'],device='cuda')
            for case in entry['cases']:
                with torch.inference_mode():r=sampled(cloud,K,E,512,512,case['focus_m'],.05/2.8,64,fill,bg,far_plane=1e5)
                path=directory/(case['id']+'-'+method+'.npz');np.savez_compressed(path,prediction_srgb=display(r['rgb_linear'].cpu().numpy()),rgb_linear=r['rgb_linear'].cpu().numpy(),alpha=r['alpha'].cpu().numpy())
                rows.append(dict(id=case['id'],frame=frame,focus_m=case['nominal_focus_m'],method=method,path=str(path)));print('TEMP_RENDER',frame,method,case['nominal_focus_m'],flush=True)
        del base,vg,cloud;torch.cuda.empty_cache()
    rows+=json.loads((TEMP/'O3-predictions.json').read_text())['rows'];write(TEMP/'predictions.json',dict(completed=True,rows=rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['infer','prepare','fit','bokehme','render'],required=True);a=parser.parse_args();c=json.loads(CRITERIA.read_text());assert c['frozen']
    torch.set_num_threads(8);cv2.setNumThreads(1)
    if a.mode!='prepare':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
        subprocess.run(['nvidia-smi'],check=True)
    globals()[a.mode](c)


if __name__=='__main__':main()
