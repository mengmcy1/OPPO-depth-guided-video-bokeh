"""Evaluation-only orientation of existing b=0 Gaussian surfels."""
from dataclasses import replace
import math

import torch
from torch.nn import functional as F


def _slope(inv, valid, axis):
    prev = torch.roll(inv, 1, axis)
    follow = torch.roll(inv, -1, axis)
    prev_valid = torch.roll(valid, 1, axis)
    next_valid = torch.roll(valid, -1, axis)
    prev_valid.select(axis, 0).zero_()
    next_valid.select(axis, -1).zero_()
    back = torch.where(prev_valid, inv - prev, 0.)
    forward = torch.where(next_valid, follow - inv, 0.)
    return torch.where(prev_valid & next_valid, (back + forward) * .5, back + forward)


def _surface_normals(depth, b0, K):
    inv = torch.where(b0, depth.reciprocal(), 0.)
    dy, dx = (_slope(inv, b0, i) for i in (0, 1))
    y, x = torch.meshgrid(torch.arange(depth.shape[0], device=depth.device, dtype=depth.dtype) + .5,
                          torch.arange(depth.shape[1], device=depth.device, dtype=depth.dtype) + .5,
                          indexing='ij')
    # A plane in camera space has affine inverse axial depth in pixel space.
    pixels = torch.stack((x, y, torch.ones_like(x)), -1)
    n = torch.stack((dx, dy, inv - dx*x - dy*y), -1) @ K
    ray = pixels @ torch.linalg.inv(K).T
    lateral = n[..., :2].norm(dim=-1)
    angle = math.radians(80)
    # Bound incidence against the actual pinhole ray as well as axial tilt:
    # a plane grazing that ray has an unbounded perspective footprint lift.
    direction = F.normalize(n[..., :2], dim=-1)
    k = (direction*ray[..., :2]).sum(-1)
    a = ray.norm(dim=-1) * math.cos(angle)
    incidence_min = (-k + a*(1+k.square()-a.square()).clamp_min(0).sqrt())/(1-a.square())
    min_z = lateral * incidence_min.clamp_min(1/math.tan(angle))
    n[..., 2] = torch.maximum(n[..., 2], min_z)
    return F.normalize(n, dim=-1)


def _rotate(q, v):
    u, v = torch.broadcast_tensors(q[..., 1:], v)
    cross = torch.linalg.cross(u, v)
    return v + 2 * (q[..., :1]*cross + torch.linalg.cross(u, cross))


@torch.no_grad()
def tilt_visible(scene, depth, b0, flags, K, *, enabled=False):
    """Keep every center and opacity; lift its image covariance onto the tangent.

    Perspective projection annihilates the radial component of a 3D vector.
    Subtracting that component to move each old covariance axis onto the new
    plane therefore preserves the 2D pinhole covariance to first order. Its
    small retained normal thickness is compensated in the tangent covariance.
    """
    if not enabled:
        return scene
    normals = _surface_normals(depth, b0, K)
    indices = torch.where(flags)[0]
    quats = scene.quats.clone()
    scales = scene.scales.clone()
    basis = torch.eye(3, device=K.device, dtype=K.dtype)
    for idx in indices.split(262144):
        means = scene.means[idx]
        uv = means @ K.T
        uv = (uv[:, :2] / uv[:, 2:]).long()
        n = normals[uv[:, 1], uv[:, 0]]
        denom = (n * means).sum(-1)
        # At a tangential viewing ray the lift has no finite solution.
        good = denom.abs() > 1e-4 * means.norm(dim=-1)
        idx, means, n, denom = idx[good], means[good], n[good], denom[good]
        nx, ny, nz = n.unbind(-1)
        align = F.normalize(torch.stack((1+nz, -ny, nx, torch.zeros_like(nz)), -1), dim=-1)
        u, v = _rotate(align, basis[0]), _rotate(align, basis[1])
        old_q = F.normalize(scene.quats[idx], dim=-1)
        old_scales = scene.scales[idx]
        axes = _rotate(old_q[:, None, :], basis[None, :, :] * old_scales[..., None])
        lifted = axes - ((axes*n[:, None, :]).sum(-1) / denom[:, None])[..., None] * means[:, None, :]
        a, b = (lifted*u[:, None, :]).sum(-1), (lifted*v[:, None, :]).sum(-1)
        normal_lift = (n - means/denom[:, None])*old_scales[:, 2:]
        da, db = (normal_lift*u).sum(-1), (normal_lift*v).sum(-1)
        uu, vv = a.square().sum(-1)-da.square(), b.square().sum(-1)-db.square()
        uv_cov = (a*b).sum(-1)-da*db
        angle = .5 * torch.atan2(2*uv_cov, uu-vv)
        c, s = angle.cos(), angle.sin()
        # Evaluate the eigenvalues through their projections to avoid
        # subtracting two nearly equal numbers at a large oblique stretch.
        major = (a*c[:, None]+b*s[:, None]).square().sum(-1)-(da*c+db*s).square()
        minor = (-a*s[:, None]+b*c[:, None]).square().sum(-1)-(-da*s+db*c).square()
        scales[idx, 0] = major.clamp_min(1e-18).sqrt()
        scales[idx, 1] = minor.clamp_min(1e-18).sqrt()
        half_c, half_s = (angle*.5).cos(), (angle*.5).sin()
        qw, qx, qy, _ = align.unbind(-1)
        quats[idx] = torch.stack((qw*half_c, qx*half_c+qy*half_s,
                                  qy*half_c-qx*half_s, qw*half_s), -1)
    return replace(scene, quats=quats, scales=scales)
