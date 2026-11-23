"""Appearance heads and depth-locked subdivided surfels for OVG stage 2."""
import torch
from torch import nn
from torch.nn import functional as F

from .aperture_renderer import GaussianScene
from .layered_aperture import pixel_layer
from .ovg_net import conv


class AppearanceHeads(nn.Module):
    def __init__(self, coverage):
        super().__init__()
        self.fusion = nn.Sequential(conv(64+3+3+1,48), conv(48,48))
        self.color = nn.Conv2d(48,3,1)
        self.coverage_delta = nn.Conv2d(48,1,1)
        self.surface = nn.Conv2d(48,5,1)
        self.coverage = nn.Conv2d(64,1,1)
        self.coverage.load_state_dict(coverage.state_dict())
        self.coverage.requires_grad_(False)
        for layer in (self.color,self.coverage_delta,self.surface):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, geometry, rgb, memory, memory_q):
        phi=geometry['features']
        hidden=self.fusion(torch.cat((phi,rgb,memory,memory_q[:,None]),1))
        surface=self.surface(hidden)
        alpha=(self.coverage(phi)+self.coverage_delta(hidden))[:,0].sigmoid()
        color=(rgb+.5*self.color(hidden).tanh()).clamp_min(0)
        opacity=(.999+.5*surface[:,0].tanh()).clamp(.001,.999)
        scale=surface[:,1:3].tanh().exp()
        tilt=.55*surface[:,3:5].tanh()
        normal=F.normalize(torch.cat((tilt,-torch.ones_like(tilt[:,:1])),1),dim=1)
        return dict(geometry,alpha=alpha,foreground=color,opacity=opacity,
                    radius_multiplier=scale,normal=normal)


def cutoff(front, back):
    return 2/(front.clamp_min(1e-6).reciprocal()+back.clamp_min(1e-6).reciprocal())


def boundary_candidate(appearance):
    return ((appearance['alpha'][0] < .95)&
            (appearance['depth2'][0] >= 1.03*appearance['depth1'][0]))


def visible_layer(appearance, inputs, split):
    color=torch.where(split[None],appearance['foreground'][0],inputs['rgb'][0])
    z=appearance['depth1'][0]
    K=inputs['K'][0]
    base=pixel_layer(color.permute(1,2,0),z,K,torch.eye(4,device=K.device),
                     sigma_pixels=.7,opacity=.999,subdivisions=5)
    def subpixels(values):
        return values.repeat_interleave(5,dim=0).repeat_interleave(5,dim=1)
    scale=subpixels(appearance['radius_multiplier'][0].permute(1,2,0)).reshape(-1,2)
    opacity=torch.where(split,appearance['opacity'][0],.999)
    vec=-appearance['normal'][0].permute(1,2,0)
    nx,ny,nz=vec.unbind(-1)
    quat=F.normalize(torch.stack((1+nz,-ny,nx,torch.zeros_like(nz)),-1),dim=-1)
    return GaussianScene(base.means,subpixels(quat).reshape(-1,4),
        torch.cat((base.scales[:,:2]*scale,base.scales[:,2:]),-1),
        subpixels(opacity).flatten(),base.colors,base.source_frames)


def boundary_layer(appearance, inputs, split, color, depth):
    K=inputs['K'][0]
    # Front-to-back blending applies the foreground transmittance to this layer.
    return pixel_layer(color,depth,K,torch.eye(4,device=K.device),
                       valid=split,layer_id=2,sigma_pixels=.7,opacity=.999,subdivisions=5)
