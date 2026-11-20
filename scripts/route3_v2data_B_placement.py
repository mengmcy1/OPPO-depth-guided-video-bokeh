"""Place the sealed multiple-layer foreground actors using pinhole geometry."""
from itertools import combinations, product

import numpy as np


MARGIN_PX = 8
MIN_WORLD_HEIGHT_M = 0.
MID_FRAME = 30
TARGET_RANGE = (.3, .7)
TARGET_TOLERANCE = .02
MIN_CONTIGUOUS_FRAMES = 9
MIN_GAP_FRACTION = .10
MAX_THICKNESS_FRACTION = .88
GRID_SIZE = 65
FRAMING_SCALES = (1., .9, .8, .65, .5, .35, .24, .16, .10, .06, .035)
LOCAL_GRID = 25


def policy(frames):
    return dict(
        middle_frame=MID_FRAME, label_frames=list(frames), margin_px=MARGIN_PX,
        minimum_actor_bbox_height_m=MIN_WORLD_HEIGHT_M,
        middle_overlap_target_range=list(TARGET_RANGE), target_tolerance=TARGET_TOLERANCE,
        minimum_contiguous_framed_label_frames=MIN_CONTIGUOUS_FRAMES,
        framing_tiers=["all_13_label_frames", "contiguous_at_least_9_including_f030"],
        minimum_depth_gap_fraction=MIN_GAP_FRACTION, maximum_actor_thickness_fraction=MAX_THICKNESS_FRACTION,
        candidate_grid_per_axis=GRID_SIZE, center_search_half_extent_camera_half_field=1.2,
        framing_scales=list(FRAMING_SCALES), local_grid_per_axis=LOCAL_GRID,
        definition="area of rear projected world AABB covered by the union of earlier layer AABBs",
        coverage="all sealed scenes with more than one designated foreground actor, including procedural middle layers",
        depth_axis="scene Y, the nominal foreground-to-background axis",
        preflight="record per-layer projection, overlap, offscreen frames and adjacent world-box depth gaps before and after repair",
        depth_gap_numeric_tolerance_m=1e-5,
        random_draws=["uniform overlap target", "horizontal preference sign", "vertical preference sign"],
        selection="first largest scale with a target match, then preferred sides, overlap error and grid order",
        fallback=[
            "try all 13 frames first, then allow a contiguous run of at least nine including f030",
            "within either tier try smaller uniform scales with the same depth and sampled target",
            "use the closest feasible midpoint ratio within [0.3, 0.7] if the target tolerance is unreachable",
            "fail before rendering if no candidate meets either tier and all depth and overlap constraints",
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


def centers_around_front(anchor, camera, focal, width, projections, mid, offsets):
    span = np.ptp(offsets, axis=0)
    nominal_depth = anchor[1] - camera[1, 3]
    padding = np.maximum(2, focal * span[[0, 2]] / nominal_depth)
    grids = []
    for row in projections:
        box = row[mid, :4]
        low, high = box[:2] - padding, box[2:4] + padding
        axes = [np.unique(np.concatenate((
            np.linspace(lo, hi, LOCAL_GRID),
            left + step * np.linspace(-1, 1, 13),
            right + step * np.linspace(-1, 1, 13),
        ))) for lo, hi, left, right, step in zip(low, high, box[:2], box[2:4], padding)]
        image = np.array(list(product(*axes)))
        rays = np.c_[(image - width/2) / focal, -np.ones(len(image))] @ camera[:3, :3].T
        grids.append(camera[:3, 3] + (nominal_depth / rays[:, 1])[:, None] * rays)
    return grids


def midpoint_run(mask, mid):
    """Count consecutive true frames surrounding the required middle frame."""
    left = np.logical_and.accumulate(mask[:mid][::-1], axis=0).sum(axis=0)
    right = np.logical_and.accumulate(mask[mid+1:], axis=0).sum(axis=0)
    return np.where(mask[mid], 1 + left + right, 0)


def framing_mask(projected, width):
    return ((projected[..., :2] >= MARGIN_PX).all(-1)
            & (projected[..., 2:4] <= width - MARGIN_PX).all(-1)
            & (projected[..., 4] >= .003))


def choose_layer(anchor, offsets, cameras, focal, width, mid, earlier, thickness, target, signs):
    base, optical_axis = centers_for_depth(anchor, cameras[mid], focal, width)
    best = {}
    for relative_scale in FRAMING_SCALES:
        scale = thickness * relative_scale
        local = centers_around_front(anchor, cameras[mid], focal, width, earlier, mid, offsets * scale)
        centers = np.concatenate((base, *local))
        projected = projected_bounds(centers, offsets * scale, cameras, focal, width)
        runs = midpoint_run(framing_mask(projected, width), mid)
        above_floor = centers[:, 2] + offsets[:, 2].min() * scale >= MIN_WORLD_HEIGHT_M
        ratios = overlaps(projected, earlier)
        errors = np.abs(ratios[mid] - target) if earlier else np.zeros(len(centers))
        in_range = ((ratios[mid] >= TARGET_RANGE[0]) & (ratios[mid] <= TARGET_RANGE[1])) if earlier else above_floor
        preference = (((centers[:, (0, 2)] - optical_axis[[0, 2]]) * signs) < 0).sum(axis=1) if earlier else (
            np.linalg.norm(centers - anchor, axis=1)
        )
        for required in (len(cameras), MIN_CONTIGUOUS_FRAMES):
            valid = np.flatnonzero((runs >= required) & above_floor & in_range)
            if not len(valid):
                continue
            within = valid[errors[valid] <= TARGET_TOLERANCE]
            if len(within):
                index = within[np.lexsort((within, errors[within], preference[within]))[0]]
                score = (0, -relative_scale, preference[index], errors[index], index)
            else:
                index = valid[np.lexsort((valid, preference[valid], errors[valid]))[0]]
                score = (1, errors[index], -relative_scale, preference[index], index)
            if required not in best or score < best[required][0]:
                best[required] = (score, centers[index], scale, relative_scale,
                                  projected[:, index], ratios[:, index], int(runs[index]), index)
        if len(cameras) in best and best[len(cameras)][0][0] == 0:
            break
    if not best:
        return None
    tier = len(cameras) if len(cameras) in best else MIN_CONTIGUOUS_FRAMES
    return tier, best[tier]


def place_layers(bounds, trajectory, frames, focal, width, nominal_gap, rng):
    """Return fixed-depth center/scale transforms from the preregistered search."""
    cameras = trajectory[np.asarray(frames)-1]
    mid = frames.index(MID_FRAME)
    targets = [(float(rng.uniform(*TARGET_RANGE)), 2*rng.integers(0, 2, size=2)-1)
               for _ in bounds[1:]]
    projections, placements, audits = [], [], []
    previous_far = None
    for index, (name, lower, upper, anchor) in enumerate(bounds):
        size = upper - lower
        offsets = np.array(list(product(*zip(-size / 2, size / 2))))
        thickness = min(1., MAX_THICKNESS_FRACTION * nominal_gap / max(size[1], 1e-12))
        target, signs = targets[index-1] if index else (None, None)
        chosen = choose_layer(anchor, offsets, cameras, focal, width, mid, projections, thickness, target, signs)
        if chosen is None:
            raise ValueError(f"no feasible framed placement for {name} at fixed depth {anchor[1]:.3f} m")
        tier, (score, center, scale, relative_scale, projected, ratios, run, choice) = chosen
        low, high = (center[1] - size[1]*scale/2, center[1] + size[1]*scale/2)
        gap = low - previous_far if previous_far is not None else None
        if gap is not None and gap < MIN_GAP_FRACTION*nominal_gap:
            raise ValueError(f"{name} would violate the minimum depth gap")
        previous_far = high
        projections.append(projected)
        placements.append((center, scale))
        fallback = []
        if not index and choice:
            fallback.append("front_recentered")
        if relative_scale < 1.:
            fallback.append("reduced_framing_scale")
        if score[0]:
            fallback.append("closest_in_range")
        if tier != len(cameras):
            fallback.append("contiguous_midpoint_framing")
        audits.append(dict(
            name=name, center_m=center.tolist(), scale=scale, thickness_scale=thickness,
            depth_range_m=[float(low),float(high)], gap_to_previous_m=gap, target_overlap=target,
            middle_overlap=float(ratios[mid]), framing_tier=tier, contiguous_framed_label_frames=run,
            fallback=fallback,
            frames=[dict(frame=f, projected_xyxy=box[:4].tolist(), overlap=float(ratio),
                         framed_with_margin=bool(inside)) for f,box,ratio,inside in zip(
                frames, projected, ratios, framing_mask(projected,width))],
        ))
    return placements, dict(layers=audits, minimum_depth_gap_m=nominal_gap*MIN_GAP_FRACTION,
                            fallback_count=sum(len(row["fallback"]) for row in audits))


def audit_layers(bounds, trajectory, frames, focal, width, nominal_gap):
    """Check all label-frame projections and actual world-space layer bounds."""
    cameras = trajectory[np.asarray(frames) - 1]
    previous, rows = [], []
    last_far = None
    required_gap = nominal_gap * MIN_GAP_FRACTION
    for name, lower, upper, _ in bounds:
        corners = np.array(list(product(*zip(lower, upper))))
        projected = projected_bounds(np.zeros((1, 3)), corners, cameras, focal, width)
        ratios = overlaps(projected, previous)[:, 0]
        boxes = projected[:, 0]
        clearance = np.minimum(boxes[:, :2].min(1), width - boxes[:, 2:4].max(1))
        outside = (clearance < 0) | (boxes[:, 4] < .003)
        fully_hidden = ratios >= 1 - 1e-7
        gap = float(lower[1] - last_far) if last_far is not None else None
        midpoint = float(ratios[frames.index(MID_FRAME)])
        flags = dict(
            fully_hidden_frames=[f for f, flag in zip(frames, fully_hidden) if flag],
            out_of_frame=[f for f, flag in zip(frames, outside) if flag],
            below_margin_frames=[f for f, value in zip(frames, clearance) if value < MARGIN_PX - 1e-3],
            depth_interleaves=gap is not None and gap < -1e-5,
            insufficient_gap=gap is not None and gap + 1e-5 < required_gap,
            above_thickness_cap=bool(upper[1] - lower[1] > nominal_gap * MAX_THICKNESS_FRACTION + 1e-5),
            below_ground=bool(lower[2] < MIN_WORLD_HEIGHT_M - 1e-5),
            outside_middle_overlap=bool(previous) and not TARGET_RANGE[0]-1e-5 <= midpoint <= TARGET_RANGE[1]+1e-5,
            insufficient_contiguous_framing=int(midpoint_run((clearance >= MARGIN_PX-1e-3) & (boxes[:,4] >= .003),
                                                              frames.index(MID_FRAME))) < MIN_CONTIGUOUS_FRAMES,
        )
        rows.append(dict(
            name=name, flags=flags, minimum_margin_px=float(clearance.min()), middle_overlap=midpoint,
            gap_to_previous_m=gap, minimum_gap_m=required_gap, depth_range_m=[float(lower[1]), float(upper[1])],
            bounds_m=[lower.tolist(), upper.tolist()],
            frames=[dict(frame=f, projected_xyxy=box[:4].tolist(), overlap=float(ratio), margin_px=float(margin))
                    for f, box, ratio, margin in zip(frames, boxes, ratios, clearance)],
        ))
        previous.append(boxes)
        last_far = upper[1]
    return rows
