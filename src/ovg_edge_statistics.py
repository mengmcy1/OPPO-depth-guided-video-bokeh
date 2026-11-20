"""Diagnostic edge statistics. Labels and counts never enter a training loss."""
from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.nn.functional import interpolate


BANDS = ((1, '0_to_2_px'), (2, 'gt_2_to_8_px'), (3, 'gt_8_px'))


def distance_bands(boundary):
    if not np.any(boundary):
        return np.full(boundary.shape, 3, np.uint8)
    distance = distance_transform_edt(~boundary)
    return np.where(distance <= 2, 1, np.where(distance <= 8, 2, 3)).astype(np.uint8)


def adjacent_residuals(prediction, gt, object_id, foreground, valid, boundary, support, jump=.03):
    """Native right/down pairs. Distance is the nearer endpoint's distance."""
    assert all(x.shape == gt.shape for x in (prediction, object_id, valid, boundary, support))
    gt, prediction = gt.astype(np.float64, copy=False), prediction.astype(np.float64, copy=False)
    foreground = np.isin(object_id, foreground)
    valid = valid & np.isfinite(gt) & (gt > 0)
    comparable = valid & support & np.isfinite(prediction) & (prediction > 0)
    inv_error = (np.divide(1., prediction, out=np.zeros_like(gt, dtype=np.float64), where=comparable)
                 - np.divide(1., gt, out=np.zeros_like(gt, dtype=np.float64), where=comparable))
    bands = distance_bands(boundary)
    for direction, p, q in (
        ('right', np.s_[:, :-1], np.s_[:, 1:]),
        ('down', np.s_[:-1, :], np.s_[1:, :]),
    ):
        same = object_id[p] == object_id[q]
        # A jump within one mesh must also have a GT subpixel boundary.
        depth_jump = (np.abs(gt[q]-gt[p]) > jump*np.minimum(gt[p],gt[q])) & (boundary[p] | boundary[q])
        paired = comparable[p] & comparable[q]
        continuous = paired & foreground[p] & foreground[q] & same & ~depth_jump
        crossing = paired & (foreground[p] | foreground[q]) & (~same | depth_jump)
        residual = inv_error[q]-inv_error[p]
        bucket = np.minimum(bands[p],bands[q])
        for code, band in BANDS[:2]:
            yield 'same_foreground_surface', direction, band, residual[continuous & (bucket==code)]
        for code, band in BANDS:
            yield 'GT_occlusion_crossing', direction, band, residual[crossing & (bucket==code)]


class EdgeStatistics:
    def __init__(self):
        self.samples = defaultdict(list)

    def add(self, scene, method, prediction, reference, support):
        for relation, direction, band, samples in adjacent_residuals(prediction, **reference, support=support):
            self.samples[scene, method, relation, direction, band].append(samples)

    def rows(self):
        for (scene, method, relation, direction, band), chunks in sorted(self.samples.items()):
            samples = np.concatenate(chunks)
            yield dict(scene=scene, method=method, relation=relation, direction=direction, distance_band=band,
                       pair_count=int(samples.size), signed_median_m_inv=float(np.median(samples)) if samples.size else None,
                       absolute_p95_m_inv=float(np.quantile(np.abs(samples), .95)) if samples.size else None)


@torch.no_grad()
def occupancy_distance_counts(boundary, reveal, block):
    """Count the exact existing L_occ masks at the effective render resolution."""
    bins = torch.from_numpy(distance_bands(boundary.detach().cpu().numpy())).to(block.device)
    if tuple(bins.shape) != tuple(block.shape[-2:]):
        # Same representative native pixel as resized_prediction's depth.
        bins = interpolate(bins[None, None].float(), size=block.shape[-2:], mode='nearest-exact')[0, 0]
    counts = {}
    for code, band in BANDS:
        members = bins == code
        counts[f'occ_{band}_reveal'] = int((reveal & members).sum())
        counts[f'occ_{band}_block'] = int((block & members).sum())
    return counts
