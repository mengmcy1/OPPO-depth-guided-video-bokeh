"""Inference-only surface variants over a frozen native pixel-layer geometry."""
from dataclasses import replace

import torch
from torch.nn import functional as F

from .aperture_renderer import GaussianScene


VARIANTS = ("V0", "V1a", "V1b-0.5", "V1b-0.75", "V1b-1.0", "V3")
THRESHOLDS = (.03, .10, .25)


def neighbors(points, axis, links):
    dim = 1-axis
    pad = (0,1) if dim == 1 else (0,0,0,1)
    forward = F.pad(links, pad)
    back = torch.roll(forward, 1, dim)
    delta = torch.roll(points, -1, dim)-points
    ahead = torch.where(forward[...,None], delta, 0.)
    behind = torch.where(back[...,None], torch.roll(delta, 1, dim), 0.)
    return ahead+behind, ahead.norm(dim=-1)+behind.norm(dim=-1), forward.int()+back.int(), forward, back


def sobel_tangents(points, links):
    terms = [neighbors(points, axis, edge) for axis, edge in enumerate(links)]
    tangents, distances = [], []
    for axis, (value, span, weight, _, _) in enumerate(terms):
        other = 1-axis
        _, _, _, forward, back = terms[other]
        channel = other ^ 1
        tangent = 2*value+torch.where(forward[...,None], torch.roll(value,-1,channel), 0.) \
            +torch.where(back[...,None], torch.roll(value,1,channel), 0.)
        spacing = 2*span+torch.where(forward,torch.roll(span,-1,channel),0.) \
            +torch.where(back,torch.roll(span,1,channel),0.)
        n = 2*weight+torch.where(forward,torch.roll(weight,-1,channel),0) \
            +torch.where(back,torch.roll(weight,1,channel),0)
        tangents.append(tangent/n.clamp_min(1)[...,None])
        distances.append(spacing/n.clamp_min(1))
    return tangents, torch.stack(distances,-1)


def geometry_basis(depth, K, links):
    y,x = torch.meshgrid(torch.arange(depth.shape[0],device=K.device,dtype=K.dtype)+.5,
                         torch.arange(depth.shape[1],device=K.device,dtype=K.dtype)+.5,indexing="ij")
    pixels=torch.stack((x,y,torch.ones_like(x)),-1)
    points=(pixels@torch.linalg.inv(K).T)*depth[...,None]
    (sx,sy), spacing=sobel_tangents(points,links)
    cross=torch.linalg.cross(sx,sy)
    normal=F.normalize(cross,dim=-1)
    valid=(spacing>0).all(-1)&torch.isfinite(normal).all(-1)&(cross.norm(dim=-1)>1e-12)
    normal=torch.where(normal[...,2:]<0,-normal,normal)
    nx,ny,nz=normal.unbind(-1)
    align=F.normalize(torch.stack((1+nz,-ny,nx,torch.zeros_like(nz)),-1),dim=-1)
    axis=torch.zeros_like(normal)
    axis[...,0]=1
    vector=align[...,1:]
    q_cross=torch.linalg.cross(vector,axis)
    local_x=axis+2*(align[...,:1]*q_cross+torch.linalg.cross(vector,q_cross))
    local_y=torch.linalg.cross(normal,local_x)
    angle=.5*torch.atan2((sx*local_y).sum(-1),(sx*local_x).sum(-1))
    c,s=angle.cos(),angle.sin()
    qw,qx,qy,qz=align.unbind(-1)
    quats=torch.stack((qw*c-qz*s,qx*c+qy*s,qy*c-qx*s,qz*c+qw*s),-1)
    return points, quats, spacing, valid


def representation(base, depth, color, K, valid_pixels, links, name, basis=None):
    if name=="V0":
        return base
    points,quats,spacing,eligible=geometry_basis(depth,K,links) if basis is None else basis
    eligible=eligible&valid_pixels
    if name=="V3":
        selected=spacing[eligible]
        scales=torch.cat((selected,selected.amin(dim=-1,keepdim=True)*.01),dim=-1)
        filler=GaussianScene((points[eligible]*1.005).contiguous(),quats[eligible].contiguous(),scales,
                             scales.new_full((len(scales),),.999),color[eligible].contiguous(),
                             torch.zeros(len(scales),dtype=torch.long,device=K.device))
        return base,filler
    parents_grid=torch.arange(valid_pixels.numel(),device=K.device).reshape(valid_pixels.shape)
    native=lambda x:x.repeat_interleave(5,0).repeat_interleave(5,1)
    parent=native(parents_grid)[native(valid_pixels)]
    selected=eligible.flatten()[parent]
    parents=parent[selected]
    rotated=base.quats.clone()
    rotated[selected]=quats.view(-1,4)[parents]
    scales=base.scales
    if name.startswith("V1b-"):
        scales=base.scales.clone()
        scales[selected,:2]=float(name.split("-")[1])*spacing.reshape(-1,2)[parents]
    return replace(base,quats=rotated,scales=scales)
