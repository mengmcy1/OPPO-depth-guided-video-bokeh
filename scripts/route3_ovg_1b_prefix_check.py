"""One P1 toy check against the per-Gaussian differentiable reference."""
import os
import subprocess

import torch
from gsplat.cuda._wrapper import fully_fused_projection

from route3_ovg_1b_common import CONFIG, OUT, guard
from route3_ovg_subpixel import save_json
from src.aperture_renderer import GaussianScene
from src.prefix_transmittance import differentiable_prefix_transmittance, prefix_transmittance


def brute(scene,K,V,depth_limit):
    h,w=depth_limit.shape
    radii,m,z,c,comp=fully_fused_projection(scene.means,None,scene.quats,scene.scales,V[None],K[None],w,h,
             eps2d=.0015,near_plane=1e-4,far_plane=1e5,calc_compensations=True)
    y,x=torch.meshgrid(torch.arange(h,device=K.device),torch.arange(w,device=K.device),indexing='ij')
    dx=m[0,:,0]-x[...,None]-.5
    dy=m[0,:,1]-y[...,None]-.5
    ex=.5*(c[0,:,0]*dx.square()+c[0,:,2]*dy.square())+c[0,:,1]*dx*dy
    a=(scene.opacities*comp[0]*torch.exp(-ex)).clamp(max=.999)
    # Independently enumerate every Gaussian within each gsplat tile rectangle.
    tile=torch.stack([x//16,y//16],-1)[...,None,:]
    lower=torch.floor((m[0]-radii[0])/16)
    upper=torch.ceil((m[0]+radii[0])/16)
    tile_hit=((tile>=lower)&(tile<upper)).all(-1)
    enabled=tile_hit&(radii[0]>0).all(-1)&(ex>=0)&(a>=1/255)&(z[0]<depth_limit[...,None])
    return (1.-torch.where(enabled,a,0.)).prod(-1)


def run():
    guard()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    subprocess.run(['nvidia-smi'],check=True)
    conf=CONFIG['prefix_check']
    dev='cuda'
    K=torch.tensor([[28.,0,15.5],[0,29.,10.5],[0,0,1.]],device=dev)
    eye=torch.eye(4,device=dev)
    dim=(21,31)
    means=torch.tensor([[-.21,-.05,1.2],[-.12,.02,2.9],[.03,-.03,1.35],[.22,.18,5.],
                        [.54,.3,2.2],[-.4,-.15,2.]],device=dev,requires_grad=True)
    scales=torch.tensor([[.08,.13,.009],[.1,.10,.012],[.09,.12,.005],
                         [.17,.092,.01],[.085,.05,.005],[.13,.077,.009]],device=dev,requires_grad=True)
    quats=torch.tensor([[1.,0,0,0],[.96,.1,.05,.03],[.92,.1,.02,.31],[1,0,0,0],
                        [.91,.0,.3,.08],[.92,.13,.0,.08]],device=dev,requires_grad=True)
    opacity=torch.tensor([.81,.35,.65,.92,.5,.78],device=dev,requires_grad=True)
    s=GaussianScene(means,quats,scales,opacity,torch.zeros(6,3,device=dev),torch.zeros(6,device=dev,dtype=torch.long))
    y,x=torch.meshgrid(torch.arange(dim[0],device=dev),torch.arange(dim[1],device=dev),indexing='ij')
    limit=torch.where(x<15,2.45,3.3)+.008*y
    limit[0,:]=.8
    limit[-1,:]=2.
    out=differentiable_prefix_transmittance(s,K,eye,limit)
    old=prefix_transmittance(s,K,eye,limit)
    ref=brute(s,K,eye,limit)
    weight=torch.sin(x*.3+y*.2)+.25
    vs=(means,quats,scales,opacity)
    grad=torch.autograd.grad((out*weight).mean(),vs)
    truth=torch.autograd.grad((ref*weight).mean(),vs)
    torch.testing.assert_close(out,ref,rtol=0,atol=conf['forward_atol'])
    torch.testing.assert_close(out,old,rtol=0,atol=conf['forward_atol'])
    errors={}
    for name,value,want in zip(('means','quats','scales','opacity'),grad,truth):
        torch.testing.assert_close(value,want,rtol=conf['gradient_rtol'],atol=conf['gradient_atol'])
        errors[name]=float((value-want).abs().max())
    save_json(OUT/'P1/prefix.json',dict(passed=True,dimensions=list(dim),gaussians=6,
                  cutoff='strict axial depth; hard membership, gradient through projection',
                  peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20,
                  max_abs_forward_vs_brute=float((out-ref).abs().max()),
                  max_abs_forward_vs_historical=float((out-old).abs().max()),
                  max_abs_backward_vs_brute=errors,tolerances=conf,
                  foreground_control_pixels=int((out<.7).sum()),
                  disocclusion_control_pixels=int((out>.97).sum())))


if __name__=='__main__':
    run()
