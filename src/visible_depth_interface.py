"""Depth and dominant-surface focus from the visible renderer's actual weights.

No Gaussian is changed. gsplat's public intersection enumerator supplies the
same front-to-back hits, including its alpha threshold and early termination.
Focus clusters are depth-contiguous intervals whose extrema are within 3%
of their ordinary median; among valid intervals we take the largest alpha mass.
"""
import math
import numpy as np
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _cluster(GI, START, COUNT, PIX, M, C, Z, O, DEPTH, MASS, ALPHA,
             W: tl.constexpr, REL: tl.constexpr, L: tl.constexpr):
    job = tl.program_id(0)
    pixel = tl.load(PIX + job)
    start = tl.load(START + pixel)
    n = tl.load(COUNT + pixel)
    j = tl.arange(0, L)
    ids = tl.load(GI + start + j, j < n, other=0)
    z = tl.load(Z + ids, j < n, other=0.)
    dx = tl.load(M + 2 * ids) - pixel % W - .5
    dy = tl.load(M + 2 * ids + 1) - pixel // W - .5
    sigma = .5 * (tl.load(C + 3 * ids) * dx * dx +
                   tl.load(C + 3 * ids + 2) * dy * dy) + tl.load(C + 3 * ids + 1) * dx * dy
    a = tl.where(j < n, tl.minimum(.999, tl.load(O + ids) * tl.exp(-sigma)), 0.)
    t = tl.cumprod(1 - a, 0)
    weight = a * t / tl.maximum(1 - a, 1e-8)
    prefix = tl.cumsum(weight, 0)
    total = tl.sum(weight, 0)
    i, e = j[:, None], j[None, :]
    lo, hi = (i + e) // 2, (i + e + 1) // 2
    il = tl.load(GI + start + lo, (i <= e) & (e < n), other=0)
    ih = tl.load(GI + start + hi, (i <= e) & (e < n), other=0)
    median = .5 * (tl.load(Z + il) + tl.load(Z + ih))
    valid = (i <= e) & (e < n) & (z[:, None] >= (1 - REL) * median) & (z[None, :] <= (1 + REL) * median)
    # Inclusive prefix differences give each interval's exact alpha mass.
    mass = prefix[None, :] - (prefix - weight)[:, None]
    mass = tl.where(valid, mass, -1.)
    best = tl.max(tl.max(mass, 1), 0)
    index = tl.min(tl.min(tl.where(mass == best, i * L + e, 2147483647), 1), 0)
    ii, ee = index // L, index % L
    ml = tl.load(GI + start + (ii + ee) // 2, n > 0, other=0)
    mh = tl.load(GI + start + (ii + ee + 1) // 2, n > 0, other=0)
    md = .5 * (tl.load(Z + ml) + tl.load(Z + mh))
    tl.store(DEPTH + pixel, tl.where(n > 0, md, 0.))
    tl.store(MASS + pixel, tl.where(total > 0, best / total, 0.))
    tl.store(ALPHA + pixel, total)


def cluster_reference(depths, weights, relative=.03):
    """Small exact CPU reference, also used for unusually long hit lists."""
    z, w = np.asarray(depths), np.asarray(weights)
    best = (-1., 0.)
    for i in range(len(z)):
        for e in range(i, len(z)):
            median = .5 * (z[(i + e) // 2] + z[(i + e + 1) // 2])
            if z[i] >= (1-relative)*median and z[e] <= (1+relative)*median:
                mass = w[i:e+1].sum()
                if mass > best[0]:
                    best = (float(mass), float(median))
    return best[1], best[0] / max(float(w.sum()), 1e-30)


@torch.no_grad()
def dominant_surface(scene, K, selection, relative=.03):
    """Return cluster maps only where selection is true; no depth-map fallback."""
    from gsplat.cuda._wrapper import (fully_fused_projection, isect_tiles,
        isect_offset_encode, rasterize_to_indices_in_range)
    h, w = selection.shape
    view = torch.eye(4, device=K.device, dtype=K.dtype)
    r, m, z, c, comp = fully_fused_projection(
        scene.means, None, scene.quats, scene.scales, view[None], K[None], w, h,
        eps2d=.0015, near_plane=1e-4, far_plane=1e5, calc_compensations=True)
    opacity = (scene.opacities[None] * comp).contiguous()
    _, ids, flat = isect_tiles(m, r, z, 16, math.ceil(w/16), math.ceil(h/16))
    offsets = isect_offset_encode(ids, 1, math.ceil(w/16), math.ceil(h/16))
    gi, pi, _ = rasterize_to_indices_in_range(0, 100000,
        torch.ones((1,h,w), device=K.device), m, c, opacity, w, h, 16, offsets, flat)
    if not bool((pi[1:] >= pi[:-1]).all()):
        raise RuntimeError('gsplat enumerator is not grouped by pixel')
    count = torch.bincount(pi.long(), minlength=h*w).contiguous()
    start = (torch.cumsum(count, 0)-count).contiguous()
    selected = torch.as_tensor(selection, device=K.device).flatten()
    depth, mass, alpha = [torch.zeros(h*w, device=K.device) for _ in range(3)]
    last = 0
    for length in [16,32,64,128,256]:
        pixels = torch.nonzero(selected & (count > last) & (count <= length)).flatten().contiguous()
        if len(pixels):
            _cluster[(len(pixels),)](gi,start,count,pixels,m.contiguous(),c.contiguous(),z.contiguous(),opacity,
                depth,mass,alpha,w,relative,length,num_warps=4 if length<=64 else 8)
        last = length
    # Rare very long lists: exact CPU path, no truncation or relaxed clustering.
    rare = torch.nonzero(selected & (count>256)).flatten().cpu().numpy()
    for pixel in rare:
        b, n = int(start[pixel]), int(count[pixel])
        ix = gi[b:b+n].long()
        dx = m.reshape(-1,2)[ix] - torch.tensor([pixel%w+.5,pixel//w+.5],device=K.device)
        cc = c.reshape(-1,3)[ix]
        sigma = .5*(cc[:,0]*dx[:,0]**2+cc[:,2]*dx[:,1]**2)+cc[:,1]*dx[:,0]*dx[:,1]
        a = torch.minimum(torch.tensor(.999,device=K.device),opacity.flatten()[ix]*torch.exp(-sigma))
        ww = a*torch.cumprod(torch.cat([torch.ones(1,device=K.device),1-a[:-1]]),0)
        dd, mm = cluster_reference(z.flatten()[ix].cpu().numpy(),ww.cpu().numpy(),relative)
        depth[pixel],mass[pixel],alpha[pixel] = dd,mm,ww.sum()
    return dict(cluster_depth=depth.reshape(h,w).cpu().numpy(),cluster_fraction=mass.reshape(h,w).cpu().numpy(),
        contribution_alpha=alpha.reshape(h,w).cpu().numpy(),
        selected_hit_count=count.reshape(h,w).cpu().numpy(),rare_exact_CPU_pixels=int(len(rare)))


@torch.no_grad()
def visible_depth_interface(scene,K,selection,relative=.03):
    """Unified visible ED/alpha plus dominant clusters for subject selection.

    Invalid pixels use an explicit far value and valid=False, never an input
    first-layer depth fallback. This function does not change Gaussian params.
    """
    from gsplat import rasterization
    h,w=selection.shape
    value,alpha,_=rasterization(scene.means,scene.quats,scene.scales,
        scene.opacities,scene.colors,torch.eye(4,device=K.device)[None],K[None],
        w,h,render_mode='RGB+ED',rasterize_mode='antialiased',eps2d=.0015,
        packed=False,near_plane=1e-4,far_plane=1e5)
    depth=value[0,...,3].cpu().numpy();alpha=alpha[0,...,0].cpu().numpy()
    valid=np.isfinite(depth)&(depth>0)&(alpha>1e-6)
    clusters=dominant_surface(scene,K,selection,relative)
    return dict(depth=np.where(valid,depth,1e4),alpha=alpha,valid=valid,**clusters)
