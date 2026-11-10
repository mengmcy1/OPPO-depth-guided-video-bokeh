"""Minimal physics checks for merged alpha sorting and per-aperture fill."""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.aperture_renderer import GaussianScene,render_aperture,aperture_disk
from src.layered_aperture import aperture_fill_samples,render_layered,pixel_layer


def one(z,color,source):
    return GaussianScene(torch.tensor([[0.,0.,z]],device='cuda'),torch.tensor([[1.,0,0,0]],device='cuda'),
        torch.tensor([[.1,.1,.001]],device='cuda'),torch.tensor([.5],device='cuda'),
        torch.tensor([color],device='cuda'),torch.tensor([source],device='cuda'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    K=torch.tensor([[40.,0,16.5],[0,40,16.5],[0,0,1.]],device='cuda');eye=torch.eye(4,device='cuda')
    fill=torch.tensor([0.,0.,1.],device='cuda').expand(32,32,3).clone().requires_grad_()
    depth=torch.full((32,32),6.,device='cuda')
    visible=one(4.,[0.,1.,0.],0);hidden=one(2.,[1.,0.,0.],1)
    output=render_layered(visible,hidden,K,eye,32,32,4.,0.,1,fill,depth)
    error=float((output['rgb_linear'][16,16]-torch.tensor([.5,.25,.25],device='cuda')).abs().max())
    assert error<1e-5,error
    reverse=render_layered(one(2.,[0.,1.,0.],0),one(4.,[1.,0.,0.],1),K,eye,32,32,4.,0.,1,fill,depth)
    reverse_error=float((reverse['rgb_linear'][16,16]-torch.tensor([.25,.5,.25],device='cuda')).abs().max())
    assert reverse_error<1e-5,reverse_error
    output['rgb_linear'][16,16].sum().backward();assert fill.grad is not None and float(fill.grad.abs().sum())>0
    plane=torch.full((32,32),4.,device='cuda');ramp=torch.arange(32,device='cuda').float()[None,:,None].expand(32,32,3)/31
    samples,known,_=aperture_fill_samples(ramp,plane,K,4.,.1,16)
    focus_error=float((samples-ramp[None]).abs().max());assert focus_error<1e-6,focus_error
    z=torch.full_like(plane,2.);shifted,_,_=aperture_fill_samples(ramp,z,K,4.,.1,2)
    xy=aperture_disk(2,.1,device='cuda',dtype=torch.float32)
    expected=ramp[None,:,3:29,:]+(40*xy[:,0]*(1/2-1/4))[:,None,None,None]/31
    projection_error=float((shifted[:,:,3:29]-expected).abs().max());assert projection_error<1e-6,projection_error
    # Old background-vector API and new per-ray constant image agree.
    base=render_aperture(visible,K,eye,32,32,4.,.1,4,background=torch.tensor([0.,0.,1.],device='cuda'))
    new=render_aperture(visible,K,eye,32,32,4.,.1,4,sample_backgrounds=fill.detach()[None].expand(4,32,32,3))
    legacy_error=float((base['rgb_linear']-new['rgb_linear']).abs().max());assert legacy_error==0
    yy,xx=torch.meshgrid(torch.arange(32,device='cuda'),torch.arange(32,device='cuda'),indexing='ij')
    checker=(((xx//4+yy//4)%2)*.6+.2).float()[...,None].expand(32,32,3).contiguous()
    plane=torch.full((32,32),4.,device='cuda');PK=K.clone();PK[0,2]=PK[1,2]=16
    cloud=pixel_layer(checker,plane,PK,eye)
    sharp=render_layered(cloud,None,PK,eye,32,32,4.,0.,1,fill.detach(),plane)
    local=(sharp['rgb_linear'][2:-2,2:-2]-checker[2:-2,2:-2]).abs()
    preservation_psnr=float(-10*local.square().mean().clamp_min(1e-15).log10())
    assert preservation_psnr>50,preservation_psnr
    shifted=render_layered(cloud,None,PK,eye,32,32,2.,.1,16,fill.detach(),plane,return_samples=True)
    coverage=float(shifted['sample_alphas'][:,2:-2,2:-2].min());assert coverage>.98,coverage
    result=dict(depth_sort_hidden_in_front_max_error=error,depth_sort_visible_in_front_max_error=reverse_error,
        fill_transmission_gradient=True,focus_plane_fill_alignment_max_error=focus_error,
        background_projection_sign_max_error=projection_error,legacy_constant_background_max_error=legacy_error,
        synthetic_pinhole_checker_PSNR=preservation_psnr,synthetic_off_grid_plane_minimum_alpha=coverage,
        fixed_pixel_initialization=dict(subdivisions=5,sigma_pixels=.7,eps2d=.0015,opacity=.999),
        interpretation='LaMa is only residual transmitted color; no B2 method outputs evaluated',passed=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
