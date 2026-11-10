"""Full-attribute constrained Gaussian fitting for the authorized U1 proxy.

This fits a separate scene snapshot, not a feedforward network. No scoring
regions, evaluation aperture locations or defocused images enter this module.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .aperture_renderer import GaussianScene


class FittedVisible(nn.Module):
    def __init__(self, scene, axial_depth):
        super().__init__()
        self.means=nn.Parameter(scene.means.detach().clone())
        self.log_scales=nn.Parameter(scene.scales.detach().log())
        self.quats=nn.Parameter(scene.quats.detach().clone())
        self.opacity_logits=nn.Parameter(torch.logit(scene.opacities.detach().clamp(1e-5,1-1e-5)))
        self.colors=nn.Parameter(scene.colors.detach().clone())
        self.register_buffer('initial_means',scene.means.detach().clone(),persistent=False)
        self.register_buffer('initial_log_scales',scene.scales.detach().log(),persistent=False)
        self.register_buffer('bound',torch.minimum(.01*axial_depth,torch.full_like(axial_depth,.02)),persistent=False)
        self.sources=scene.source_frames

    def scene(self):
        return GaussianScene(self.means,F.normalize(self.quats,dim=-1),self.log_scales.exp(),
            self.opacity_logits.sigmoid(),self.colors,self.sources)

    def optimizer(self):
        return torch.optim.Adam([
            dict(params=[self.means],lr=1e-4,name='position'),
            dict(params=[self.log_scales],lr=5e-3,name='scale'),
            dict(params=[self.quats],lr=1e-3,name='rotation'),
            dict(params=[self.opacity_logits],lr=1e-2,name='opacity'),
            dict(params=[self.colors],lr=2e-2,name='color')],eps=1e-15)

    @torch.no_grad()
    def project_constraints(self):
        delta=self.means-self.initial_means
        ratio=(self.bound/delta.norm(dim=-1).clamp_min(1e-12)).clamp_max(1)
        self.means.copy_(self.initial_means+delta*ratio[:,None])
        self.log_scales.copy_(torch.maximum(self.initial_log_scales-2.302585093,
            torch.minimum(self.log_scales,self.initial_log_scales+2.302585093)))
        self.quats.copy_(F.normalize(self.quats,dim=-1))
        self.opacity_logits.clamp_(-12,12)
        self.colors.clamp_(0,1)


def ssim(x,y):
    """Standard local SSIM, 11x11 Gaussian sigma1.5, RGB range0..1."""
    x=x.permute(2,0,1)[None];y=y.permute(2,0,1)[None]
    v=torch.arange(11,device=x.device,dtype=x.dtype)-5
    w=torch.exp(-v.square()/(2*1.5**2));w=w/w.sum()
    kernel=(w[:,None]*w[None,:])[None,None].expand(3,1,11,11).contiguous()
    mux=F.conv2d(x,kernel,groups=3);muy=F.conv2d(y,kernel,groups=3)
    vx=F.conv2d(x*x,kernel,groups=3)-mux.square()
    vy=F.conv2d(y*y,kernel,groups=3)-muy.square()
    cov=F.conv2d(x*y,kernel,groups=3)-mux*muy
    return (((2*mux*muy+.01**2)*(2*cov+.03**2))/
            ((mux.square()+muy.square()+.01**2)*(vx+vy+.03**2))).mean()
