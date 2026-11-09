"""Differentiable thin-lens integration with gsplat (OpenCV camera convention).

Distances, Gaussian positions/scales and aperture radius share one world unit.
Focus is axial camera depth, not Euclidean ray length. Colors are linear RGB.
Provenance statistics are alpha-composited contributions, not hard ray hits.
"""
from dataclasses import dataclass
from typing import Optional, Sequence
import math

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class GaussianScene:
    means: Tensor
    quats: Tensor  # wxyz
    scales: Tensor  # activated positive world-space standard deviations
    opacities: Tensor  # activated [0, 1], shape N
    colors: Tensor  # N,3, linear RGB (no SH in this interface)
    source_frames: Tensor  # N integer t_ref values

    def select(self, sources: Optional[Sequence[int]]):
        if sources is None:
            return self
        keep = torch.isin(self.source_frames, torch.as_tensor(sources, device=self.means.device))
        return GaussianScene(*(getattr(self, k)[keep] for k in self.__dataclass_fields__))


def srgb_to_linear(rgb: Tensor) -> Tensor:
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).clamp_min(0).pow(2.4))


def linear_to_srgb(rgb: Tensor) -> Tensor:
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb.clamp_min(1e-12).pow(1 / 2.4) - 0.055)


def aperture_disk(count: int, radius, *, device, dtype) -> Tensor:
    """Deterministic antithetic equal-area disk samples; K=1 is the center."""
    if count < 1:
        raise ValueError("Sample count must be positive")
    radius = torch.as_tensor(radius, device=device, dtype=dtype)
    if bool((radius < 0).any()):
        raise ValueError("Aperture radius must be nonnegative")
    half = count // 2
    index = torch.arange(half, device=device, dtype=dtype)
    r = ((index + 0.5) / max(half, 1)).sqrt()
    theta = index * (math.pi * (3 - math.sqrt(5)))
    xy = torch.stack((r * theta.cos(), r * theta.sin()), -1)
    xy = torch.cat((xy, -xy), 0)
    if count % 2:
        xy = torch.cat((torch.zeros(1, 2, device=device, dtype=dtype), xy), 0)
    return xy * radius


def _rasterize_with_intrinsic_gradients(scene, features, K, view, width, height, mode):
    """Use gsplat's torch projection and CUDA rasterizer for trainable Ks.

    The public gsplat 1.5 projection omits K gradients. This equivalent path
    avoids silently losing focus-distance gradients or adding an image warp.
    """
    from gsplat.cuda._torch_impl import _quat_scale_to_covar_preci, _fully_fused_projection
    from gsplat.cuda._wrapper import isect_tiles, isect_offset_encode, rasterize_to_pixels
    covariance, _ = _quat_scale_to_covar_preci(scene.quats, scene.scales, compute_preci=False)
    radii, means2d, depths, conics, compensation = _fully_fused_projection(
        scene.means, covariance, view[None], K[None], width, height,
        near_plane=1e-4, far_plane=1e4, calc_compensations=mode == "antialiased")
    opacities = scene.opacities.reshape(1, -1)
    if compensation is not None:
        opacities = opacities * compensation
    tile = 16
    tw, th = math.ceil(width / tile), math.ceil(height / tile)
    _, ids, flattened = isect_tiles(means2d, radii, depths, tile, tw, th)
    offsets = isect_offset_encode(ids, 1, tw, th)
    colors = torch.cat((features[None], depths[..., None]), -1)
    rendered, alpha = rasterize_to_pixels(means2d, conics, colors, opacities,
                                          width, height, tile, offsets, flattened)
    rendered = torch.cat((rendered[..., :-1], rendered[..., -1:] / alpha.clamp_min(1e-10)), -1)
    return rendered, alpha, {}


def render_aperture(scene: GaussianScene, intrinsics: Tensor, world_to_camera: Tensor,
                    width: int, height: int, focus_distance, aperture_radius,
                    samples: int, current_frame: int = 0,
                    source_frames: Optional[Sequence[int]] = None,
                    background: Optional[Tensor] = None, return_samples: bool = False,
                    rasterize_mode: str = "classic") -> dict:
    """Render scalar focus by off-axis projection or a per-pixel focus field.

    A focus field traces each output pixel through its own focal depth. It is
    implemented by sampling an enlarged subcamera image at the corresponding
    ray coordinate. A constant field and scalar focus differ only by bilinear
    resampling. Returned RGB, alpha and provenance remain differentiable.
    """
    from gsplat import rasterization

    scene = scene.select(source_frames)
    device, dtype = scene.means.device, scene.means.dtype
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    world_to_camera = world_to_camera.to(device=device, dtype=dtype)
    focus = torch.as_tensor(focus_distance, device=device, dtype=dtype)
    if not bool(torch.isfinite(focus).all()) or bool((focus <= 0).any()):
        raise ValueError("Focus depth must be finite and positive")
    if focus.ndim and focus.shape != (height, width):
        raise ValueError("Focus field must have shape (height,width)")
    aperture = aperture_disk(samples, aperture_radius, device=device, dtype=dtype)
    source_ids, inverse = torch.unique(scene.source_frames, sorted=True, return_inverse=True)
    provenance = F.one_hot(inverse, num_classes=max(len(source_ids), 1)).to(dtype)[:, :len(source_ids)]
    features = torch.cat((scene.colors, provenance), -1)
    bg = torch.zeros(3, device=device, dtype=dtype) if background is None else background.to(scene.means)
    yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                            torch.arange(width, device=device, dtype=dtype), indexing="ij")
    # Pixel centers: gsplat uses coordinates u+0.5,v+0.5 internally.
    pixel_xy = torch.stack((xx + 0.5, yy + 0.5), -1)
    sum_image = torch.zeros(height, width, 3, device=device, dtype=dtype)
    sum_alpha = torch.zeros(height, width, device=device, dtype=dtype)
    sum_hits = torch.zeros(height, width, len(source_ids), device=device, dtype=dtype)
    sum_depth_mass = torch.zeros_like(sum_alpha)
    sample_depths, sample_alphas, sample_views, sample_Ks, sample_pixels = [], [], [], [], []
    for xy in aperture:
        shift = torch.cat((-xy, torch.zeros(1, device=device, dtype=dtype)))
        view = torch.cat((torch.cat((world_to_camera[:3, :3],
                                    (world_to_camera[:3, 3] + shift)[:, None]), 1), world_to_camera[3:]), 0)
        if focus.ndim == 0:
            delta = torch.stack((intrinsics[0, 0] * xy[0] / focus,
                                 intrinsics[1, 1] * xy[1] / focus, torch.zeros_like(focus)))
            K = intrinsics + torch.stack((torch.zeros_like(delta), torch.zeros_like(delta), delta), 1)
            render_K, rw, rh, grid = K, width, height, None
            rays_xy = pixel_xy
        else:
            ray_delta = xy * torch.stack((intrinsics[0, 0], intrinsics[1, 1])) / focus[..., None]
            rays_xy = pixel_xy - ray_delta
            pad = int(torch.ceil(ray_delta.detach().abs().max()).item()) + 2
            pad_delta = torch.tensor([float(pad), float(pad), 0.], device=device, dtype=dtype)
            render_K = intrinsics + torch.stack((torch.zeros_like(pad_delta), torch.zeros_like(pad_delta), pad_delta), 1)
            K, rw, rh = intrinsics, width + 2 * pad, height + 2 * pad
            coords = rays_xy + pad
            grid = torch.stack((2 * coords[..., 0] / rw - 1, 2 * coords[..., 1] / rh - 1), -1)[None]
        if len(scene.means):
            if render_K.requires_grad:
                rendered, alpha, _ = _rasterize_with_intrinsic_gradients(
                    scene, features, render_K, view, rw, rh, rasterize_mode)
            else:
                rendered, alpha, _ = rasterization(
                    means=scene.means, quats=scene.quats, scales=scene.scales,
                    opacities=scene.opacities.reshape(-1), colors=features,
                    viewmats=view[None], Ks=render_K[None], width=rw, height=rh,
                    near_plane=1e-4, far_plane=1e4, packed=False,
                    render_mode="RGB+ED", rasterize_mode=rasterize_mode,
                )
            if grid is not None:
                # Resample alpha*depth, then divide, so empty pixels do not bias depth.
                depth_mass = rendered[..., -1:] * alpha
                merged = torch.cat((rendered[..., :-1], depth_mass, alpha), -1)
                warped = F.grid_sample(merged.permute(0, 3, 1, 2), grid,
                                       mode="bilinear", padding_mode="zeros", align_corners=False)
                warped = warped.permute(0, 2, 3, 1)[0]
                alpha = warped[..., -1]
                depth = warped[..., -2] / alpha.clamp_min(1e-8)
                rendered = warped[..., :-2]
            else:
                depth, rendered, alpha = rendered[0, ..., -1], rendered[0, ..., :-1], alpha[0, ..., 0]
            image, hits = rendered[..., :3], rendered[..., 3:]
        else:
            alpha, depth = torch.zeros_like(sum_alpha), torch.zeros_like(sum_alpha)
            image, hits = torch.zeros_like(sum_image), torch.zeros_like(sum_hits)
        sum_image = sum_image + image + (1 - alpha[..., None]) * bg
        sum_alpha = sum_alpha + alpha
        sum_hits = sum_hits + hits
        sum_depth_mass = sum_depth_mass + depth * alpha
        if return_samples:
            sample_depths.append(depth)
            sample_alphas.append(alpha)
            sample_views.append(view)
            sample_Ks.append(K)
            sample_pixels.append(rays_xy)
    alpha, hits = sum_alpha / samples, sum_hits / samples
    current = hits[..., source_ids == current_frame].sum(-1)
    other = hits[..., source_ids != current_frame].sum(-1)
    output = {"rgb_linear": sum_image / samples, "alpha": alpha,
              "source_ids": source_ids, "t_ref_weights": hits,
              "current_frame_weight": current, "other_frame_weight": other,
              "hole_fraction": 1 - alpha,
              "depth": sum_depth_mass / sum_alpha.clamp_min(1e-8),
              "aperture_xy": aperture, "sample_count": samples}
    if return_samples:
        output.update(sample_depths=torch.stack(sample_depths), sample_alphas=torch.stack(sample_alphas),
                      sample_world_to_camera=torch.stack(sample_views), sample_intrinsics=torch.stack(sample_Ks),
                      sample_pixel_coordinates=torch.stack(sample_pixels))
    return output


def observability_statistics(rendered: dict) -> dict:
    """Expose both conventions because the plan calls other-or-hole 'O'.

    O_other_or_hole grows with missing current-frame support. It must not be
    interpreted as a reliability score whose low values trigger repair.
    """
    return {"O_other_or_hole": rendered["other_frame_weight"] + rendered["hole_fraction"],
            "current_frame_support": rendered["current_frame_weight"],
            "observed_support": rendered["alpha"], "hole_fraction": rendered["hole_fraction"]}


def disocclusion_mask(rendered: dict, reference_depth: Tensor, reference_K: Tensor,
                      reference_world_to_camera: Tensor, reference_valid: Optional[Tensor] = None,
                      relative_tolerance: float = 0.02, absolute_tolerance: float = 0.01,
                      hit_threshold: float = 0.9) -> dict:
    """Reproject aperture hits into the central depth map and test occlusion.

    Supply geometrically accurate first-surface depths for GT masks. Expected
    Gaussian depths give an approximation. Unknown geometry is reported
    separately and never counted as known occlusion or a zero disocclusion.
    """
    depth = rendered["sample_depths"]
    K, view, pixels = rendered["sample_intrinsics"], rendered["sample_world_to_camera"], rendered["sample_pixel_coordinates"]
    h, w = reference_depth.shape
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
    rays = torch.einsum("kij,khwj->khwi", torch.linalg.inv(K), homogeneous)
    camera_points = rays * depth[..., None]
    world = torch.einsum("kij,khwj->khwi", torch.linalg.inv(view)[:, :3, :3], camera_points)
    world = world + torch.linalg.inv(view)[:, None, None, :3, 3]
    central = torch.einsum("ij,khwj->khwi", reference_world_to_camera[:3, :3], world)
    central = central + reference_world_to_camera[:3, 3]
    projected = torch.einsum("ij,khwj->khwi", reference_K, central)
    uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
    grid = torch.stack((2 * uv[..., 0] / w - 1, 2 * uv[..., 1] / h - 1), -1)
    count = len(depth)
    center_depth = F.grid_sample(reference_depth[None, None].expand(count, 1, h, w), grid,
                                 mode="nearest", padding_mode="zeros", align_corners=False)[:, 0]
    valid = reference_depth > 0 if reference_valid is None else reference_valid
    center_valid = F.grid_sample(valid.to(depth)[None, None].expand(count, 1, h, w), grid,
                                 mode="nearest", padding_mode="zeros", align_corners=False)[:, 0] > 0.5
    inside = (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
    known = center_valid & inside & (central[..., 2] > 0) & torch.isfinite(depth) & (depth > 0)
    known = known & (rendered["sample_alphas"] >= hit_threshold)
    occluded = known & (central[..., 2] > center_depth * (1 + relative_tolerance) + absolute_tolerance)
    fraction = occluded.float().mean(0)
    return {"mask": occluded.any(0), "ray_fraction": fraction,
            "known_ray_fraction": known.float().mean(0),
            "unknown_ray_fraction": (~known).float().mean(0),
            "image_fraction": occluded.any(0).float().mean()}
