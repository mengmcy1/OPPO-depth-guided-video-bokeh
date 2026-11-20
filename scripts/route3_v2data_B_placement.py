"""Place the sealed multiple-layer foreground actors using pinhole geometry."""
from itertools import combinations, product

import numpy as np


MARGIN_PX = 8
MIN_WORLD_HEIGHT_M = 0.
MID_FRAME = 30
TARGET_RANGE = (.3, .7)
TARGET_TOLERANCE = .02
MAX_FRAME_OCCLUSION = .90
MIN_GAP_FRACTION = .10
MAX_THICKNESS_FRACTION = .88
GRID_SIZE = 65
FRAMING_SCALES = (1., .9, .8, .65, .5, .35)


def policy(frames):
    return dict(
        middle_frame=MID_FRAME, label_frames=list(frames), margin_px=MARGIN_PX,
        minimum_actor_bbox_height_m=MIN_WORLD_HEIGHT_M,
        middle_overlap_target_range=list(TARGET_RANGE), target_tolerance=TARGET_TOLERANCE,
        all_frame_max_overlap=MAX_FRAME_OCCLUSION, minimum_depth_gap_fraction=MIN_GAP_FRACTION,
        maximum_actor_thickness_fraction=MAX_THICKNESS_FRACTION,
        candidate_grid_per_axis=GRID_SIZE, center_search_half_extent_camera_half_field=1.2,
        framing_scales=list(FRAMING_SCALES),
        definition="area of rear projected world AABB covered by the union of earlier layer AABBs",
        random_draws=["uniform overlap target", "horizontal preference sign", "vertical preference sign"],
        selection="first largest scale with a target match, then preferred sides, overlap error and grid order",
        fallback=[
            "move the first layer to the nearest feasible projected center if its original anchor clips",
            "try the documented smaller uniform scales in order while preserving every geometric constraint",
            "if no scale achieves the sampled target tolerance, take the closest in-range result at the largest feasible scale",
            "fail before rendering if no such placement exists",
        ],
    )


def projected_bounds(centers, offsets, cameras, focal, width):
    corners = centers[:, None, :] + offsets
    bounds = []
    for camera in cameras:
        xyz = (corners - camera[:3, 3]) @ camera[:3, :3]
        depth = -xyz[..., 2]
        projected = focal * xyz[..., :2] / np.maximum(depth[..., None], 1e-12) + width / 2
        bounds.append(np.concatenate((projected.min(axis=1), projected.max(axis=1),
                                      depth.min(axis=1)[:, None]), axis=1))
    return np.stack(bounds)


def overlaps(boxes, earlier):
    area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    covered = np.zeros_like(area)
    for count in range(1, len(earlier) + 1):
        for group in combinations(earlier, count):
            low, high = boxes[..., :2], boxes[..., 2:4]
            for front in group:
                low = np.maximum(low, front[:, None, :2])
                high = np.minimum(high, front[:, None, 2:4])
            covered += (-1)**(count + 1) * np.maximum(0, high - low).prod(axis=-1)
    return np.clip(covered / np.maximum(area, 1e-12), 0., 1.)


def centers_for_depth(anchor, camera, focal, width):
    axis = -camera[:3, 2]
    distance = (anchor[1] - camera[1, 3]) / axis[1]
    center = camera[:3, 3] + distance * axis
    radius = distance * width / (2 * focal)
    steps = np.linspace(-1.2, 1.2, GRID_SIZE) * radius
    shifts = np.array(list(product(steps, steps)))
    grid = np.repeat(center[None], len(shifts), axis=0)
    grid[:, (0, 2)] += shifts
    return np.concatenate((anchor[None], grid)), center


def place_layers(bounds, trajectory, frames, focal, width, nominal_gap, rng):
    """Return center/scale transforms and a projection audit, without rendering."""
    cameras = trajectory[np.asarray(frames) - 1]
    mid = frames.index(MID_FRAME)
    projections, placements, audits = [], [], []
    depths = []
    for index, (name, lower, upper, anchor) in enumerate(bounds):
        size = upper - lower
        offsets = np.array(list(product(*zip(-size / 2, size / 2))))
        thickness_scale = min(1., MAX_THICKNESS_FRACTION * nominal_gap / max(size[1], 1e-12))
        centers, optical_axis = centers_for_depth(anchor, cameras[mid], focal, width)
        if index:
            target = float(rng.uniform(*TARGET_RANGE))
            signs = 2 * rng.integers(0, 2, size=2) - 1
            side_penalty = (((centers[:, (0, 2)] - optical_axis[(0, 2),]) * signs) < 0).sum(axis=1)
        best = match = None
        for relative_scale in FRAMING_SCALES:
            scale = relative_scale * thickness_scale
            projected = projected_bounds(centers, offsets * scale, cameras, focal, width)
            above_floor = centers[:, 2] - size[2] * scale / 2 >= MIN_WORLD_HEIGHT_M
            inside = ((projected[..., :2] >= MARGIN_PX).all(axis=(0, 2))
                      & (projected[..., 2:4] <= width - MARGIN_PX).all(axis=(0, 2))
                      & (projected[..., 4] >= .003).all(axis=0) & above_floor)
            if not index:
                valid = np.flatnonzero(inside)
                if len(valid):
                    choice = 0 if inside[0] else valid[np.argmin(
                        np.linalg.norm(centers[valid] - anchor, axis=1)
                    )]
                    match = (choice, scale, relative_scale, projected[:, choice], np.zeros(len(frames)), None)
                    break
                continue
            ratios = overlaps(projected, projections)
            midpoint = ratios[mid]
            valid = np.flatnonzero(inside & (ratios.max(axis=0) <= MAX_FRAME_OCCLUSION)
                                   & (midpoint >= TARGET_RANGE[0]) & (midpoint <= TARGET_RANGE[1]))
            if not len(valid):
                continue
            error = np.abs(midpoint - target)
            closest = valid[np.lexsort((valid, side_penalty[valid], error[valid]))[0]]
            if best is None:
                best = (closest, scale, relative_scale, projected[:, closest], ratios[:, closest], target)
            good = valid[error[valid] <= TARGET_TOLERANCE]
            if len(good):
                choice = good[np.lexsort((good, error[good], side_penalty[good]))[0]]
                match = (choice, scale, relative_scale, projected[:, choice], ratios[:, choice], target)
                break
        used_target_fallback = match is None and best is not None
        if match is None:
            match = best
        if match is None:
            raise ValueError(f"no all-frame placement for {name} at depth {anchor[1]:.3f} m")
        choice, scale, relative_scale, projected, ratios, target = match
        center = centers[choice]
        depth_range = (center[1] - size[1] * scale / 2, center[1] + size[1] * scale / 2)
        gap = depth_range[0] - depths[-1][1] if depths else None
        if gap is not None and gap < MIN_GAP_FRACTION * nominal_gap:
            raise ValueError(f"{name} would violate the minimum depth gap")
        depths.append(depth_range)
        projections.append(projected)
        placements.append((center, scale))
        fallback = []
        if not index and choice:
            fallback.append("front_recentered")
        if relative_scale < 1.:
            fallback.append("reduced_framing_scale")
        if used_target_fallback:
            fallback.append("closest_in_range")
        audits.append(dict(
            name=name, center_m=center.tolist(), scale=scale, thickness_scale=thickness_scale,
            minimum_world_height_m=float(center[2] - size[2] * scale / 2),
            depth_range_m=list(depth_range), gap_to_previous_m=gap, target_overlap=target,
            middle_overlap=float(ratios[mid]), fallback=fallback,
            frames=[dict(frame=frame, projected_xyxy=box[:4].tolist(), overlap=float(fraction))
                    for frame, box, fraction in zip(frames, projected, ratios)],
        ))
    return placements, dict(layers=audits, minimum_depth_gap_m=nominal_gap * MIN_GAP_FRACTION,
                            fallback_count=sum(bool(row["fallback"]) for row in audits))
