"""C1 development switches for the same forward pass; no learned parameters."""
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class FusionSettings:
    static_filter: bool = False
    visibility_filter: bool = False
    neighbor_steps: int = 2  # positions within the actual 13-frame input list
    motion_threshold: float = .005  # normalized MoVieS world units
    visibility_relative_margin: float = .1
    visibility_absolute_margin_m: float = .01


def multiquery_dynamic(query_offsets: Tensor, threshold: float) -> Tensor:
    """Same source Gaussian compared across query times, shape Q,N,3.

    Classification uses maximum displacement from the first query. A constant
    nonzero absolute offset is static under this definition. It is not GT.
    """
    if query_offsets.ndim != 3 or query_offsets.shape[-1] != 3 or len(query_offsets) < 2:
        raise ValueError('At least two query times of the same Gaussian IDs are required')
    # Pairwise range avoids missing a displacement between two non-first queries.
    span = torch.zeros(query_offsets.shape[1], device=query_offsets.device)
    for i in range(len(query_offsets)):
        for j in range(i):
            span = torch.maximum(span, (query_offsets[i] - query_offsets[j]).norm(dim=-1))
    return span > threshold


def select_gaussians(means: Tensor, source_frames: Tensor, dynamic: Tensor,
                     input_frame_indices: Tensor, current_frame: int,
                     K: Tensor, world_to_camera: Tensor, current_depth: Tensor,
                     settings: FusionSettings, reference_motion_mask: Tensor = None):
    """M1/M2a/M2b mask and deletion diagnostics.

    Visibility rejects an OTHER-source Gaussian lying in front of the current
    predicted surface. Unknown/out-of-image depth is kept and counted. The
    diagnostic reference mask never changes selection.
    """
    if current_frame not in input_frame_indices.tolist():
        raise ValueError('Current time must be one of the fixed input frames')
    current_pos = int((input_frame_indices == current_frame).nonzero()[0])
    eligible = input_frame_indices[max(0, current_pos - settings.neighbor_steps):current_pos + settings.neighbor_steps + 1]
    near = torch.isin(source_frames, eligible)
    keep = (~dynamic | near) if settings.static_filter else torch.ones_like(dynamic)
    camera = means @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    projected = camera @ K.T
    uv = (projected[:, :2] / projected[:, 2:].clamp_min(1e-8)).floor().long()
    h, w = current_depth.shape
    inside = (camera[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    u, v = uv[:, 0].clamp(0, w - 1), uv[:, 1].clamp(0, h - 1)
    depth = current_depth[v, u]
    known = inside & torch.isfinite(depth) & (depth > 0)
    margin = depth * settings.visibility_relative_margin + settings.visibility_absolute_margin_m
    in_front = known & (camera[:, 2] < depth - margin) & (source_frames != current_frame)
    deleted = keep & in_front if settings.visibility_filter else torch.zeros_like(keep)
    statistics = dict(total=len(keep),dynamic=int(dynamic.sum()),outside_or_unknown=int((~known).sum()),
        removed_by_static_dynamic=int((~keep).sum()),removed_by_visibility=int(deleted.sum()),
        static_removed_by_visibility=int((deleted & ~dynamic).sum()),
        current_source_removed_by_visibility=int((deleted & (source_frames == current_frame)).sum()),
        false_deletion_GT_available=False)
    if reference_motion_mask is not None:
        # A risk proxy, not a false-positive ground truth or a selection signal.
        statistics['deleted_outside_reference_motion_proxy'] = int((deleted & known & ~reference_motion_mask[v,u]).sum())
    return keep & ~deleted, statistics
