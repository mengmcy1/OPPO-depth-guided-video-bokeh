"""U1 snapshot fitting from disjoint translated-pinhole supervision only."""
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
import torch.nn.functional as F
from gsplat import rasterization
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.aperture_renderer import GaussianScene,linear_to_srgb
from src.layered_aperture import pixel_layer,merge_layers
from src.upper_bound_optimization import FittedVisible,ssim
from src.dual_layer import depth_band
from src.analytic_aperture import sampled
from route3_b2_oracle_3d import sparse_pixel_layer
from route3_upper_bound_evaluate import exr,display,OUT,BASE,RECORDS


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def inputs(frame):
    name=f'B2-high-f{frame:03d}'
    d=np.load(ROOT/'outputs/route3-dual-layer-2026-11-10-01/shared-inputs'/(name+'.npz'))
    K=torch.tensor(d['K'],device='cuda',dtype=torch.float32);cam=torch.tensor(d['camera_c2w'],device='cuda',dtype=torch.float32)
    visible=pixel_layer(torch.tensor(d['visible_rgb_linear'],device='cuda'),torch.tensor(d['center_depth'],device='cuda'),K,cam)
    parent=np.load(BASE/'methods-3d/prepared'/name/'parents.npz')
    filled=np.load(ROOT/'outputs/route3-dual-layer-2026-11-10-01/shared-fill'/name/'fill.npz')
    fill=torch.tensor(filled['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda')
    profile=json.loads((RECORDS/'route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile']
    return d,K,cam,visible,parent,fill,bg,profile


def hidden_selection(d,parent,focus=None,stop=None):
    # Input-depth method bands only: no residuals, GT aperture masks or scores.
    band=np.zeros((512,512),bool)
    for f in ([focus] if focus is not None else [1.4,5.]):
        for n in ([stop] if stop is not None else [1.4,2.]):
            band|=depth_band(d['center_depth'],d['K'][0,0],f,.05/(2*n),1.5,.03)[0]
    uv=parent['uv'];x=np.clip(uv[:,0].astype(int),0,511);y=np.clip(uv[:,1].astype(int),0,511)
    return parent['recoverable']&band[y,x],band


def translated_fill(fill,bg,K,xy):
    y,x=torch.meshgrid(torch.arange(512,device='cuda'),torch.arange(512,device='cuda'),indexing='ij')
    uv=torch.stack([x+.5,y+.5],-1)+torch.tensor(xy,device='cuda')*torch.diag(K)[:2]/bg[...,None]
    grid=2*uv/512-1
    return F.grid_sample(fill.permute(2,0,1)[None],grid[None],align_corners=False,padding_mode='border')[0].permute(1,2,0)


def pinhole(pc,K,E,background):
    image,alpha,_=rasterization(pc.means,pc.quats,pc.scales,pc.opacities,pc.colors,E[None],K[None],512,512,
        packed=False,rasterize_mode='antialiased',near_plane=1e-4,far_plane=1e5,eps2d=.0015)
    return linear_to_srgb(image[0]+(1-alpha[0])*background).clamp(0,1)


def fit(a):
    folder=OUT/'U1-fit'/f'f{a.frame:03d}';folder.mkdir(parents=True,exist_ok=False)
    plan=json.loads((OUT/'U1-positions.json').read_text());d,K,cam,initial,parent,fill,bg,profile=inputs(a.frame)
    views=[]
    for r in plan['positions']:
        if r['frame']!=a.frame:continue
        path=Path(r['existing_exr']) if 'existing_exr' in r else OUT/'U1-supervision'/(r['key']+'.exr')
        rgb=torch.tensor(display(exr(path)),device='cuda')
        E=torch.linalg.inv(cam);E[:2,3]-=torch.tensor(r['xy_m'],device='cuda')
        views.append(dict(**r,rgb=rgb,E=E,background=translated_fill(fill,bg,K,r['xy_m'])))
    train=[v for v in views if v['role'] in ['U1-train','U1-center']];held=[v for v in views if v['role']=='U1-heldout']
    assert len(train)==65 and len(held)==16
    chosen,union=hidden_selection(d,parent);hidden=sparse_pixel_layer(parent,chosen,K,cam,profile,'cuda')
    z=torch.tensor(d['center_depth'],device='cuda').repeat_interleave(5,0).repeat_interleave(5,1).reshape(-1)
    fitted=FittedVisible(initial,z);optimizer=fitted.optimizer();del initial,z
    config=dict(frame=a.frame,steps=a.steps,train_views=65,heldout_views=16,center_included=True,seed=2026111105+a.frame,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        loss='0.8displayRGB L1+0.2(1-SSIM11sigma1.5)',position_bound='min(2cm,1%initialdepth)',scale_bound='0.1x..10x initialscale',
        color_bound='nonnegative learned linear radiance;no uppercap;supervision and evaluation remain sameLDR;no rawHDR labels',
        learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},learning_rate_schedule='cosine1to0.01 overfixedbudget',
        hidden_parents=int(chosen.sum()),visible_gaussians=len(fitted.means),
        hidden_training_band='union of main-lens input-depth method bands only;score masks never read',
        DOF_GT_read=False,evaluation_positions_read=False,evaluation_masks_read=False,
        fixed_hidden=True,heldout_selects_checkpoint=False)
    write(folder/'config.json',config)
    rng=np.random.default_rng(config['seed']);rows=[];step_times=[];overfit_count=0;diverged_count=0;initial_train_loss=None
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();stop_reason=None;first_gradients={}
    for step in range(a.steps+1):
        if step%500==0 or step in [0,100,a.steps]:
            with torch.inference_mode():
                pc=merge_layers(fitted.scene(),hidden);monitor={}
                for name,group in [('train',train[::4]),('heldout',held)]:
                    sse=0.;pixels=0;losses=[]
                    for v in group:
                        pred=pinhole(pc,K,v['E'],v['background']);sse+=float((pred-v['rgb']).square().double().sum());pixels+=pred.numel()
                        losses.append(float(.8*abs(pred-v['rgb']).mean()+.2*(1-ssim(pred,v['rgb']))))
                    monitor[name]=dict(psnr=float(-10*np.log10(max(sse/pixels,1e-12))),loss=float(np.mean(losses)),views=len(group))
                del pc
            gap=monitor['train']['psnr']-monitor['heldout']['psnr']
            row=dict(step=step,seconds=time.monotonic()-start,metrics=monitor,train_minus_heldout_PSNR_dB=gap,
                median_step_seconds=float(np.median(step_times[-100:])) if step_times else None,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
            rows.append(row);write(folder/'monitor.json',dict(completed=False,rows=rows,config=config));print('U1_MONITOR',a.frame,json.dumps(row),flush=True)
            if initial_train_loss is None:initial_train_loss=monitor['train']['loss']
            if step>=1000:
                overfit_count=overfit_count+1 if gap>3 else 0
                diverged_count=diverged_count+1 if monitor['train']['loss']>1.05*initial_train_loss else 0
            if overfit_count>=3:stop_reason='training PSNR exceeds heldout by>3dB for3 consecutive monitors'
            if diverged_count>=3:stop_reason='persistent training loss divergence above initial loss'
            if stop_reason:break
            if step>0 and step%1000==0:
                torch.save(dict(step=step,parameters=fitted.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state),folder/'latest.pt')
        if step==a.steps:break
        factor=.01+.99*.5*(1+np.cos(np.pi*step/a.steps))
        for group in optimizer.param_groups:group['lr']=config['learning_rates'][group['name']]*factor
        v=train[int(rng.integers(len(train)))];optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();begin=time.monotonic()
        pc=merge_layers(fitted.scene(),hidden);pred=pinhole(pc,K,v['E'],v['background'])
        loss=.8*abs(pred-v['rgb']).mean()+.2*(1-ssim(pred,v['rgb']))
        if not torch.isfinite(loss):stop_reason='nonfinite training loss';break
        loss.backward()
        if step==0:
            for name,param in fitted.named_parameters():
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise RuntimeError('Missing/nonfinite first backward gradient: '+name)
                first_gradients[name]=dict(norm=float(param.grad.norm()),max_abs=float(param.grad.abs().max()),parameters=param.numel())
        optimizer.step();fitted.project_constraints();torch.cuda.synchronize()
        step_times.append(time.monotonic()-begin);del pc,pred,loss
        if step==99:
            write(folder/'backward-profile.json',dict(steps=100,median_step_seconds=float(np.median(step_times)),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,first_backward_gradients=first_gradients,config=config))
    with torch.no_grad():
        scene=fitted.scene();torch.save({key:getattr(scene,key).detach().cpu() for key in scene.__dataclass_fields__},folder/'visible-final.pt')
    report=dict(completed=stop_reason is None and step==a.steps,stop_reason=stop_reason,config=config,rows=rows,
        learned_linear_color_max=float(scene.colors.max()),learned_Gaussian_fraction_any_color_above1=float((scene.colors>1).any(-1).float().mean()),
        wall_seconds=time.monotonic()-start,median_step_seconds=float(np.median(step_times)) if step_times else None,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,first_backward_gradients=first_gradients,output=str(folder/'visible-final.pt'))
    write(folder/'result.json',report);write(RECORDS/f'route3-upper-bound-2026-11-11-U1-f{a.frame:03d}.json',report)
    if stop_reason:raise RuntimeError('U1_STOP '+stop_reason)


def render(a):
    dest=OUT/'U1-render';dest.mkdir(exist_ok=False);rows=[];timings=[]
    for frame in [1,30,60]:
        d,K,cam,unused,parent,fill,bg,profile=inputs(frame);del unused
        data=torch.load(OUT/'U1-fit'/f'f{frame:03d}'/'visible-final.pt',weights_only=True,map_location='cuda')
        visible=GaussianScene(**data);E=torch.linalg.inv(cam)
        for focus in [1.4,5.]:
            for stop in [1.4,2.]:
                case=f'B2-high-f{frame:03d}-N{stop:g}-F{focus:g}';chosen,band=hidden_selection(d,parent,focus,stop)
                hidden=sparse_pixel_layer(parent,chosen,K,cam,profile,'cuda');values={};times={}
                for name,pc in [('U1_visible',visible),('U1_joint',merge_layers(visible,hidden))]:
                    with torch.inference_mode():
                        torch.cuda.synchronize();start=time.monotonic();r=sampled(pc,K,E,512,512,focus,.05/(2*stop),64,fill,bg,far_plane=1e5)
                        torch.cuda.synchronize();times[name]=time.monotonic()-start
                    values[name]=r['rgb_linear'].cpu().numpy()
                values['U1']=np.where(band[...,None],values.pop('U1_joint'),values['U1_visible'])
                for name,raw in values.items():
                    path=dest/(case+'-'+name+'.npz');np.savez_compressed(path,rgb_linear=raw,prediction_srgb=display(raw),band=band)
                    rows.append(dict(id=case,method=name,path=str(path)))
                timings.append(dict(id=case,times=times));print('U1_RENDER',case,flush=True);del hidden
        del visible,data;torch.cuda.empty_cache()
    write(OUT/'U1-predictions.json',dict(rows=rows));write(RECORDS/'route3-upper-bound-2026-11-11-U1-render.json',dict(completed=True,rows=timings,K=64,one_optimized_fit_per_frame=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['fit','render'],required=True)
    p.add_argument('--frame',type=int,choices=[1,30,60]);p.add_argument('--steps',type=int,default=8000);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
    subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    if a.mode=='fit':fit(a)
    else:render(a)


if __name__=='__main__':main()
