"""Target-only point controls for the unchanged MagicBokeh depth/renderer path."""
import numpy as np
from PIL import Image
from scipy import ndimage


def visible_target_alpha(folder, metadata, frame):
    rank = int(metadata["source"]["focus_object_depth_rank"])
    depths = np.asarray(frame["foreground_depths_m"], dtype=float)
    if not 0 <= rank < len(depths) or not (np.diff(depths) > 0).all():
        raise ValueError("Invalid front-to-back target metadata")
    if abs(float(depths[rank]) - frame["focus_distance_m"]) > 1e-8:
        raise ValueError("Focus plane does not match the designated object")
    alpha = np.asarray(Image.open(folder / frame["foreground_files"][rank]).convert("RGBA"), dtype=np.float32)[..., 3] / 255
    for near in range(rank):
        foreground = np.asarray(Image.open(folder / frame["foreground_files"][near]).convert("RGBA"), dtype=np.float32)[..., 3] / 255
        alpha *= 1 - foreground
    source_index = int(frame["source_order_by_depth"][rank])
    identity = metadata["source"]["objects"][source_index]["source_id"]
    return alpha, identity


def safe_target_region(alpha):
    """Fixed alpha fallback; largest component; image edges count as boundaries."""
    for threshold in [.9, .5, 0.]:
        binary = alpha > threshold
        if not binary.any():
            continue
        labels, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        largest = np.flatnonzero(sizes == sizes.max())
        if len(largest) > 1:
            largest = sorted(largest, key=lambda x: np.flatnonzero(labels.ravel() == x)[0])
        region = labels == largest[0]
        distance = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        return region, distance, dict(status="AVAILABLE", alpha_threshold=threshold,
            component_pixels=int(region.sum()), visible_pixels_gt0=int((alpha > 0).sum()),
            visible_pixels_gt05=int((alpha > .5).sum()),
            point_A_xy=[int(x), int(y)], point_A_margin_pixels=float(distance[y, x]))
    return None, None, dict(status="focus_point_unavailable", visible_pixels_gt0=0, visible_pixels_gt05=0)


def medoid_point(region, distance, quantized_depth):
    median = float(np.median(quantized_depth[region]))
    error = np.where(region, np.abs(quantized_depth.astype(float) - median), np.inf)
    eligible = error == error.min()
    margins = np.where(eligible, distance, -1)
    y, x = np.unravel_index(np.argmax(margins), margins.shape)
    return [int(x), int(y)], dict(target_native_depth_median_png8=median,
        selected_depth_error_png8=float(error[y, x]), boundary_margin_pixels=float(distance[y, x]))


def control_diagnostics(alpha, quantized_depth, focus, K, *, native_focused=None):
    # Match the author's FP32 ToTensor and GPU pointwise arithmetic.
    depth = quantized_depth.astype(np.float32) / np.float32(255)
    defocus = np.abs(depth - np.float32(focus)) * np.float32(K / 32)
    focused = defocus <= np.float32(.1) if native_focused is None else native_focused
    target = alpha > .5
    non_target = ~target
    return dict(target_focus_coverage=None if not target.any() else float(focused[target].mean()),
        focus_leakage=None if not non_target.any() else float(focused[non_target].mean()),
        focused_image_fraction=float(focused.mean()), target_eval_pixels=int(target.sum()),
        target_eval_mask="Visible target alpha >0.5; complement defines non-target",
        max_defocus_condition=float(defocus.max()))
