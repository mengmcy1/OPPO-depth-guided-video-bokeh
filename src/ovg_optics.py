"""Stage 1b continuous surfels and differentiable thin-lens losses."""

import torch
from torch.nn import functional as F
from gsplat import rasterization

from .aperture_renderer import GaussianScene
from .layered_aperture import pixel_layer
from .ovg_net import masked_mean, rays
from .prefix_transmittance import differentiable_prefix_transmittance


def visible_surfels(pred, inputs):
    """Camera-aligned centers, view-normal covariances, observed linear RGB.

    Stage 1b emits one surface per pixel. Coverage alpha supervises geometry;
    a bounded near-opaque footprint opacity controls rasterization.
    """
    z = pred['depth1'][0]
    K = inputs['K'][0]
    directions = rays(K[None], z.shape, z.dtype, z.device)[0]
    xyz = directions * z[..., None]
    unit = -pred['normal'][0].permute(1, 2, 0)
    ux, uy, uz = unit.unbind(-1)
    quat = F.normalize(torch.stack((1 + uz, -uy, ux, torch.zeros_like(uz)), -1), dim=-1)
    tangent = pred['sigma_px'][0].permute(1, 2, 0) * z[..., None] / K.diag()[:2]
    scale = torch.cat((tangent, z[..., None] * 2e-4), -1)
    opacity = .90 + .099 * pred['opacity'][0]
    return GaussianScene(xyz.reshape(-1, 3), quat.reshape(-1, 4), scale.reshape(-1, 3), opacity.flatten(),
                         inputs['rgb'][0].permute(1, 2, 0).reshape(-1, 3),
                         torch.zeros(z.numel(), device=z.device, dtype=torch.long))


def fixed_visible_surfels(pred, inputs):
    depth = pred['depth1'][0]
    K = inputs['K'][0]
    return pixel_layer(inputs['rgb'][0].permute(1, 2, 0), depth, K,
                       torch.eye(4, device=K.device, dtype=K.dtype),
                       sigma_pixels=.7, opacity=.999, subdivisions=5)


def subcamera(K, focus, offset):
    view = torch.eye(4, device=K.device, dtype=K.dtype)
    view[:2, 3] = -offset
    intr = K.clone()
    intr[:2, 2] += K.diag()[:2] * offset / focus
    return intr, view


def raster(scene, K, focus, offsets, height, width, background=None, *, return_alpha=False):
    images, coverage = [], []
    for j, offset in enumerate(offsets):
        J, view = subcamera(K, focus, offset)
        rgb, a, _ = rasterization(scene.means, scene.quats, scene.scales, scene.opacities, scene.colors,
                                  view[None], J[None], width, height, packed=False, render_mode='RGB',
                                  rasterize_mode='antialiased', eps2d=.0015, near_plane=1e-4, far_plane=1e5)
        image = rgb[0]
        if background is not None:
            image = image + (1 - a[0]) * background[j]
        images.append(image)
        if return_alpha:
            coverage.append(a[0,...,0].detach())
    result = torch.stack(images)
    return (result, torch.stack(coverage)) if return_alpha else result


def occlusion_loss(visible, K, focus, offsets, cutoff, reveal, block, *, sparse=False, sample_scale=1):
    """Balanced, two-sided clipped-depth error on labeled non-mixed rays."""
    active = reveal | block if sparse else (None,)*len(cutoff)
    T = torch.stack([
        differentiable_prefix_transmittance(visible, *subcamera(K, focus, xy), z.contiguous(),
                                            active=mask, sample_scale=sample_scale)
        for xy, z, mask in zip(offsets, cutoff, active)
    ])
    return masked_mean(1 - T, reveal) + masked_mean(T, block)


def stratified_pupil(count, radius, generator, K):
    # Equal-area annuli crossed with a randomized common azimuth shift. Each
    # annulus is sampled independently and uniforms are redrawn for each subset.
    u = torch.rand((2, count), generator=generator, device=K.device, dtype=K.dtype)
    an = (torch.arange(count, device=K.device, dtype=K.dtype) + u[0]) / count
    r = radius * an.sqrt()
    theta = 2 * torch.pi * u[1]
    return torch.stack((r * theta.cos(), r * theta.sin()), -1)


def resized_prediction(pred, inputs, target, size):
    old_h, old_w = pred['depth1'].shape[-2:]
    h, w = size
    if size == (old_h, old_w):
        return pred, inputs, target
    if old_h % h or old_w % w:
        raise ValueError('Optical reduction must tile whole pixels')
    K = inputs['K'].clone()
    K[:, 0] *= w / old_w
    K[:, 1] *= h / old_h
    def nearest(value):
        return F.interpolate(value, size=size, mode='nearest-exact')
    out = dict(pred, depth1=nearest(pred['depth1'][:, None])[:, 0])
    if 'normal' in pred:
        out['normal'] = nearest(pred['normal'])
        out['opacity'] = nearest(pred['opacity'][:, None])[:, 0]
        out['sigma_px'] = nearest(pred['sigma_px']) * K.new_tensor([w/old_w, h/old_h])[None, :, None, None]
    # Sampling a center of the input's observed pixel footprint for visibility
    # is distinct from integrating the full-res photometric reference.
    obs = dict(inputs, rgb=F.interpolate(inputs['rgb'], size=size, mode='area'), K=K)
    return out, obs, target
