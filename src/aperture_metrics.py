"""Small region and temporal metric interfaces for route-three preparation."""
from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F


def regional_metrics(prediction: Tensor, target: Tensor, mask: Tensor,
                     lpips_model=None) -> dict:
    """HWC RGB in [0,1]; empty regions return None, never a perfect score.

    SSIM averages an 11x11 Gaussian SSIM map over region centers. LPIPS requires
    a caller-provided spatial LPIPS network; its receptive field crosses masks.
    """
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("Expected matching HWC RGB tensors")
    mask = mask.bool() & torch.isfinite(prediction).all(-1) & torch.isfinite(target).all(-1)
    count = int(mask.sum())
    result = {"pixels": count, "fraction": count / mask.numel(), "psnr": None, "ssim": None,
              "lpips": None, "lpips_status": "not_requested" if lpips_model is None else "empty_region"}
    if count == 0:
        return result
    mse = (prediction[mask] - target[mask]).square().mean()
    result["psnr"] = float(-10 * mse.clamp_min(1e-12).log10())
    x, y = prediction.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    axis = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel = torch.exp(-axis.square() / (2 * 1.5 ** 2))
    kernel = (kernel[:, None] * kernel[None, :]); kernel = kernel / kernel.sum()
    kernel = kernel[None, None].expand(3, 1, 11, 11)
    def local(z):
        return F.conv2d(F.pad(z, (5, 5, 5, 5), mode="replicate"), kernel, groups=3)
    ux, uy = local(x), local(y)
    vx, vy, cov = local(x * x) - ux * ux, local(y * y) - uy * uy, local(x * y) - ux * uy
    score = ((2 * ux * uy + 0.01 ** 2) * (2 * cov + 0.03 ** 2) /
             ((ux.square() + uy.square() + 0.01 ** 2) * (vx + vy + 0.03 ** 2)))
    result["ssim"] = float(score[0].mean(0)[mask].mean())
    if lpips_model is not None:
        distance = lpips_model(2 * x - 1, 2 * y - 1)
        if distance.ndim != 4 or distance.shape[-2:] == (1, 1):
            raise ValueError("Regional LPIPS needs a spatial=True LPIPS model")
        distance = F.interpolate(distance, size=mask.shape, mode="bilinear", align_corners=False)[0, 0]
        result.update(lpips=float(distance[mask].mean()), lpips_status="spatial_map_region_centers")
    return result


def temporal_warp_error(current: Tensor, previous: Tensor, backward_flow: Tensor,
                        valid_mask: Tensor, target_current: Optional[Tensor] = None,
                        target_previous: Optional[Tensor] = None) -> dict:
    """Backward flow maps current pixels into previous pixels, in pixel units.

    The caller supplies an occlusion/flow consistency mask. Reference-corrected
    error additionally subtracts the target's temporal residual when supplied.
    """
    h, w, _ = current.shape
    if previous.shape != current.shape or backward_flow.shape != (h, w, 2):
        raise ValueError("Mismatched image or flow shapes")
    y, x = torch.meshgrid(torch.arange(h, device=current.device, dtype=current.dtype),
                          torch.arange(w, device=current.device, dtype=current.dtype), indexing="ij")
    uv = torch.stack((x, y), -1) + backward_flow
    finite = torch.isfinite(uv).all(-1)
    valid = valid_mask.bool() & finite & (uv[..., 0] >= 0) & (uv[..., 0] <= w - 1)
    valid = valid & (uv[..., 1] >= 0) & (uv[..., 1] <= h - 1)
    grid = torch.stack((2 * (uv[..., 0] + .5) / w - 1, 2 * (uv[..., 1] + .5) / h - 1), -1)
    def warp(image):
        return F.grid_sample(image.permute(2, 0, 1)[None], grid[None],
                             align_corners=False, padding_mode="zeros")[0].permute(1, 2, 0)
    residual = current - warp(previous)
    valid = valid & torch.isfinite(residual).all(-1)
    count = int(valid.sum())
    result = {"valid_pixels": count, "valid_fraction": count / (h * w),
              "warp_l1": float(residual[valid].abs().mean()) if count else None,
              "reference_corrected_warp_l1": None}
    if target_current is not None or target_previous is not None:
        if target_current is None or target_previous is None:
            raise ValueError("Both temporal target images are required")
        target_residual = target_current - warp(target_previous)
        valid = valid & torch.isfinite(target_residual).all(-1)
        if bool(valid.any()):
            result["reference_corrected_warp_l1"] = float((residual - target_residual)[valid].abs().mean())
    return result
