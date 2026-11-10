"""Method-independent C1 proxy regions for fixed CamVFX cameras.

A temporal median does not establish the existence of a clean hidden surface.
Agreement counts measure photometric consistency, not geometric correctness.
"""
import cv2
import numpy as np


def linear_rgb(x):
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4).astype(np.float32)


def display_rgb(x):
    return np.clip(np.where(x <= .0031308, 12.92 * x,
                           1.055 * np.maximum(x, 1e-12) ** (1 / 2.4) - .055), 0, 1)


def background_and_motion(images_linear, config):
    board = np.median(images_linear, axis=0)
    difference = np.max(np.abs(images_linear - board[None]), axis=-1)
    agreement = (difference <= config['linear_rgb_background_agreement_threshold']).sum(0).astype(np.uint8)
    valid = agreement >= config['background_min_agreement_frames']
    raw = difference > config['linear_rgb_motion_max_channel_threshold']
    kernel = np.ones((config['morphology_open_close_kernel'],) * 2, np.uint8)
    masks = []
    for m in raw:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= config['minimum_motion_component_pixels'])
        keep = keep[keep != 0]
        masks.append(np.isin(labels, keep))
    return board, agreement, valid, np.asarray(masks), raw


def disk_counts(mask, radius):
    r = int(radius)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = ((xx * xx + yy * yy) <= r * r).astype(np.float32)
    return np.maximum(0, np.rint(cv2.filter2D(mask.astype(np.float32), -1, kernel,
                                          borderType=cv2.BORDER_CONSTANT)))


def proxy_regions(motion_masks, background_valid, frame_index, input_indices,
                  radius, static_disocclusion, static_known, config):
    """Disk-footprint approximation; unknown background is never recoverable.

    Motion depth is unknown: radius is the maximum physical CoC radius over the
    nearest/farthest positive supplied mesh depth. The mesh is not dynamic GT.
    """
    m = motion_masks[frame_index]
    blocked = disk_counts(m, radius)
    total = disk_counts(np.ones_like(m), radius)
    boundary = (blocked > 0) & (blocked < total)
    others = [i for i in input_indices if i != frame_index]
    exposed = (~motion_masks[others]).any(0) if others else np.zeros_like(m)
    exposed50 = (~motion_masks[np.arange(len(motion_masks)) != frame_index]).any(0)
    numerator = disk_counts(m & exposed & background_valid, radius)
    numerator50 = disk_counts(m & exposed50 & background_valid, radius)
    rho = numerator / np.maximum(blocked, 1)
    rho50 = numerator50 / np.maximum(blocked, 1)
    confidence = disk_counts(m & background_valid, radius) / np.maximum(blocked, 1)
    reliable = confidence >= config['minimum_footprint_background_confidence']
    cut = config['recoverable_ratio_threshold']
    recoverable = boundary & reliable & (rho >= cut)
    unrecoverable = boundary & reliable & (rho < cut)
    unknown = boundary & ~reliable
    motion_footprint = blocked > 0
    static = static_disocclusion & ~motion_footprint & static_known
    return dict(M_t=m, D_t=boundary, M_C1_proxy=recoverable,
                dynamic_unrecoverable=unrecoverable, dynamic_unknown=unknown,
                static_static=static, rho_t=rho.astype(np.float32),
                rho_50=rho50.astype(np.float32), confidence=confidence.astype(np.float32),
                M_C1_proxy_50=boundary & reliable & (rho50 >= cut))


def mask_summary(mask, confidence, flags=()):
    count = int(mask.sum())
    return dict(valid_pixels=count, image_fraction=float(mask.mean()),
                mean_confidence=float(np.mean(confidence[mask])) if count else None,
                failure_flags=list(flags) + ([] if count else ['empty_region']))
