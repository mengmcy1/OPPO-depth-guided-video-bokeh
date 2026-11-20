"""Cycles RGB observations, clipped layer targets, and optical reference pairs."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import bpy
from mathutils import Matrix
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route3_ovg_subpixel import save_json, save_npz
from route3_ovg_phase2_blender import gate_materials


SIZE = 512
LDR_LIMIT = 1.0
SUBVIEWS = 64
TOLERANCE = dict(
    composed_display_MAE=.025, composed_display_p95=.10, rendered_vs_mesh_coverage_MAE=.075,
    same_seed_complement_linear_MAE=.002,
)


def srgb(rgb):
    return np.where(rgb <= .0031308, 12.92*rgb, 1.055*np.power(rgb, 1/2.4)-.055)


def configure(scene, cpu):
    scene.cycles.device = "CPU" if cpu else "GPU"
    if not cpu:
        enable_optix()
    scene.render.use_compositing = False
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.cycles.use_persistent_data = True
    scene.render.film_transparent = True


def enable_optix():
    pref = bpy.context.preferences.addons["cycles"].preferences
    pref.compute_device_type = "OPTIX"
    pref.get_devices()
    for device in pref.devices:
        device.use = device.type == "OPTIX"
    if sum(d.use for d in pref.devices) != 1:
        raise RuntimeError("Single dedicated OptiX device required")


def render_image(scene, filename, seed, spp, *, transparent, size=SIZE):
    scene.render.film_transparent = transparent
    scene.cycles.seed, scene.cycles.samples = seed, spp
    scene.render.filepath = str(filename)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(filename), check_existing=False)
    try:
        pixels = np.empty(size*size*4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgb = pixels.reshape(size, size, 4)[::-1].copy()
        if not np.all(np.isfinite(rgb)):
            raise ValueError("Cycles returned non-finite RGB")
        return rgb
    finally:
        bpy.data.images.remove(image)
        filename.unlink()


def integration_offsets(fstop):
    # Symmetric equal-weight disk sampling as in R1 and the K16 labels.
    i = np.arange(SUBVIEWS//2)
    radius = np.sqrt((i+.5)/len(i))
    theta = i * np.pi * (3-np.sqrt(5))
    positive = np.stack([radius*np.cos(theta), radius*np.sin(theta)], axis=-1)
    return np.concatenate([positive, -positive]) * .05 / (2*fstop)


def optical(scene, camera, base_transform, out, case_id, focus, fstop, seed, spp):
    target = out/"labels"/f"{case_id}-optics.npz"
    hdr_path = out/"labels"/f"{case_id}-hdr.npz"
    offsets = integration_offsets(fstop)
    metadata = dict(id=case_id, focus_m=focus, fstop=fstop, subviews=SUBVIEWS,
                    hdr_sha_excluded_from_input=True)
    if target.is_file() and hdr_path.is_file():
        with np.load(target) as saved, np.load(hdr_path) as raw:
            if np.array_equal(saved["aperture_xy"],offsets) and all(
                file["rgb_linear"].shape == (SIZE,SIZE,3) and np.isfinite(file["rgb_linear"]).all()
                for file in (saved,raw)
            ):
                return metadata
        raise ValueError(f"Saved optical pair does not match the requested configuration: {case_id}")
    hdr = np.zeros((SIZE,SIZE,3), np.float64)
    ldr = np.zeros_like(hdr)
    for index, (ax, ay) in enumerate(offsets):
        camera.matrix_world = base_transform @ Matrix.Translation((ax, -ay, 0))
        camera.data.shift_x = -ax / (.036*focus/.05)
        camera.data.shift_y = ay / (.036*focus/.05)
        rgba = render_image(scene, out/"rebuildable-render.exr", seed+index, spp, transparent=False)
        rgb = rgba[..., :3]
        hdr += rgb
        ldr += np.clip(rgb, 0, LDR_LIMIT)
    camera.matrix_world = base_transform
    camera.data.shift_x = camera.data.shift_y = 0
    save_npz(target, rgb_linear=(ldr/SUBVIEWS).astype(np.float16), aperture_xy=offsets)
    hdr = hdr / SUBVIEWS
    if abs(hdr).max() > np.finfo(np.float16).max:
        raise ValueError("HDR overflows float16 across the integrated reference")
    save_npz(hdr_path, rgb_linear=hdr.astype(np.float16))
    return metadata


def straight(premultiplied):
    alpha = premultiplied[..., 3:]
    return np.divide(premultiplied[..., :3], alpha, out=np.zeros_like(premultiplied[..., :3]), where=alpha>1e-7)


def layer_labels(scene, gate, controls, background_switch, observed, path, filename, seed, spp):
    with np.load(path) as file:
        cut = file["cutoff_m"]
        alpha = file["alpha"].astype(np.float32)
        reliable = file["valid_alpha"]
        dual = file["valid_dual"]
    rgba = np.ones((SIZE,SIZE,4), np.float32)
    rgba[..., :3] = cut[::-1,:,None]
    gate.pixels.foreach_set(rgba.ravel())
    gate.update()
    values = []
    for sign in (0.,1.):
        for toggle in controls:
            toggle.default_value = sign
        values.append(render_image(scene, filename, seed, spp, transparent=True))
    raw_fg, raw_bg = (straight(value) for value in values)
    interior = reliable & (alpha == 1)
    raw_fg[interior] = observed[interior]
    F, B, I = [np.clip(rgb, 0, LDR_LIMIT) for rgb in (raw_fg, raw_bg, observed)]
    boundary = reliable & (alpha>0) & (alpha<1)
    saturated = boundary & np.logical_or.reduce([(rgb.astype(np.float16)>=LDR_LIMIT).any(axis=-1) for rgb in (F,B,I)])
    valid_F = reliable & (alpha > 0) & ((values[0][...,3] > 0) | interior) & ~saturated
    valid_B = reliable & (alpha < 1) & (values[1][...,3] > 0) & ~saturated
    valid_comp = boundary & valid_F & valid_B
    mixed = alpha[...,None]*F + (1-alpha[...,None])*B
    color_error = np.abs(srgb(mixed)-srgb(I))[valid_comp]
    cover_error = np.abs(alpha-values[0][...,3])[boundary]
    complement_error = np.abs(values[0][...,:3] + values[1][...,:3] - observed)[valid_comp]
    audit = dict(
        same_seed_complement_linear_MAE=float(complement_error.mean()) if complement_error.size else None,
        valid_boundary=int(valid_comp.sum()), saturated_boundary=int(saturated.sum()),
        composed_display_MAE=float(color_error.mean()) if color_error.size else None,
        composed_display_p95=float(np.quantile(color_error,.95)) if color_error.size else None,
        rendered_vs_mesh_coverage_MAE=float(cover_error.mean()) if cover_error.size else None,
    )
    audit["passed"] = all(audit[key] is None or audit[key] <= limit for key, limit in TOLERANCE.items())
    save_npz(
        path.with_name(path.name.replace("-pinhole", "-colors")),
        foreground_rgb=F.astype(np.float16), complementary_background_rgb=B.astype(np.float16),
        foreground_valid=valid_F, background_valid=valid_B, compositing_valid=valid_comp,
        saturated_boundary_invalid=saturated, valid_dual=dual,
    )
    for node in background_switch:
        node.default_value = 0.
    return audit


def expose_evaluated_material_slots(scene):
    graph = bpy.context.evaluated_depsgraph_get()
    extra = []
    for ob in scene.objects:
        if ob.type != "MESH" or not ob.modifiers:
            continue
        evaluated = ob.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            for mat in mesh.materials:
                if mat and mat.original.name not in ob.data.materials:
                    extra.append((ob.data, mat.original))
        finally:
            evaluated.to_mesh_clear()
    for mesh, mat in extra:
        if mat.name not in mesh.materials:
            mesh.materials.append(mat)


def thin_interior(root, frame, object_id):
    with np.load(root/"labels"/f"f{frame:03d}-pinhole.npz") as labels:
        mask = labels["object_id"] == object_id
    pixels = np.pad(mask, 5)
    return np.lib.stride_tricks.sliding_window_view(pixels,(11,11)).all(axis=(-2,-1))


def half_pixel_lanczos(rgb):
    padded = np.pad(rgb, ((0,0),(3,4),(0,0)), mode="reflect")
    taps = np.arange(-3,5)-.5
    weights = np.sinc(taps)*np.sinc(taps/4)
    weights /= weights.sum()
    return sum(weight*padded[:,offset:offset+SIZE] for offset,weight in enumerate(weights))


def thin_texture_check(root, scene, camera, frames, source, filename, seed, spp):
    if source["category"] != "thin_structure" or 30 not in frames:
        return None
    with np.load(root/"labels/scene-geometry.npz") as geometry:
        names = geometry["object_names"].tolist()
        object_id = int(geometry["object_ids"][names.index("foreground_structure")])
    scene.frame_set(30)
    mask = thin_interior(root,30,object_id)
    pixels = int(mask.sum())
    if pixels < 25:
        return dict(frame=30, pure_interior_pixels=pixels, status="not_applicable")
    seed += 30
    base = render_image(scene,filename,seed,spp,transparent=False)[...,:3]
    noise = render_image(scene,filename,seed+1,spp,transparent=False)[...,:3]
    camera.data.shift_x = .5/SIZE
    shift = render_image(scene,filename,seed,spp,transparent=False)[...,:3]
    camera.data.shift_x = 0.
    scene.render.resolution_x = scene.render.resolution_y = 2*SIZE
    double = render_image(scene,filename,seed,spp,transparent=False,size=2*SIZE)[...,:3]
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    resized = double.reshape(SIZE,2,SIZE,2,3).mean(axis=(1,3))
    display = lambda value: srgb(np.clip(value, 0, LDR_LIMIT))
    phase = float(np.mean((display(shift)[mask]-display(half_pixel_lanczos(base))[mask])**2))
    baseline = float(np.mean((display(noise)[mask]-display(base)[mask])**2))
    excess = max(0.,phase-baseline)**.5
    if excess > .01:
        for frame in frames:
            save_npz(root/"labels"/f"f{frame:03d}-phase-invalid.npz", invalid=thin_interior(root,frame,object_id))
    return dict(
        frame=30, pure_interior_pixels=pixels, status="degraded_texture" if excess>.01 else "passed",
        phase_excess_rmse=excess, auxiliary_2x_rmse=float(np.sqrt(np.mean(
            (display(resized)[mask]-display(base)[mask])**2
        ))),
    )


def reference_views(scene, camera, root, frames, source, seed, spp):
    lens = camera.copy()
    lens.data = camera.data.copy()
    lens.animation_data_clear()
    lens.data.animation_data_clear()
    scene.collection.objects.link(lens)
    scene.camera = lens
    metadata = []
    try:
        for frame in frames:
            scene.frame_set(frame)
            pose = camera.matrix_world.copy()
            near = source["near_anchor_m"]
            for index, focus in enumerate((near, near*source["depth_ratio"])):
                for fstop in (1.4, 2.0):
                    case = f"f{frame:03d}-focus{index}-N{fstop:g}"
                    metadata.append(optical(scene,lens,pose,root,case,focus,fstop,seed+frame,spp))
    finally:
        scene.camera = camera
        bpy.data.objects.remove(lens,do_unlink=True)
    return metadata


def run(root, frames, references, samples, scene_seed, cpu, reference_samples, optical_only):
    estimated = 2 * 2**30
    if shutil.disk_usage(root).free < 60 * 2**30 + estimated:
        raise RuntimeError("Disk reserve would be exceeded for this render batch")
    bpy.ops.wm.open_mainfile(filepath=str(root/"scene.blend"))
    scene = bpy.context.scene
    camera = scene.camera
    configure(scene, cpu)
    workfile = root/"rebuildable-render.exr"
    if workfile.exists():
        raise FileExistsError(workfile)
    source = json.loads((root/"scene-spec.json").read_text())
    start = time.monotonic()
    accepted = [frame for frame in frames if (root/"labels"/f"f{frame:03d}-pinhole.npz").is_file()]
    if optical_only:
        if not accepted:
            raise ValueError("No validated geometry frames for the optical reference")
        rows = reference_views(scene,camera,root,accepted,source,scene_seed,reference_samples)
        save_json(root/"labels/optical-audit.json", dict(optical=rows,exposure_ev=0,
            color_space="scene-linear RGB",subviews=SUBVIEWS,cycles_samples_per_subview=reference_samples,
            seconds=time.monotonic()-start))
        return
    # Passes are inserted after RGB/optical rendering, never into the observation stream.
    for frame in frames:
        scene.frame_set(frame)
        rgb = render_image(scene, workfile, scene_seed+frame, samples, transparent=False)[..., :3]
        save_npz(root/"observations"/f"f{frame:03d}.npz", rgb_linear=np.clip(rgb, 0, LDR_LIMIT).astype(np.float16))
    render_meta = reference_views(scene,camera,root,accepted,source,scene_seed,reference_samples) if references else []
    texture_phase = thin_texture_check(root, scene, camera, accepted, source, workfile, scene_seed, samples)
    cutter = bpy.data.images.new("B axial depth cutter", SIZE, SIZE, alpha=True, float_buffer=True)
    cutter.colorspace_settings.name = "Non-Color"
    cutter.alpha_mode = "CHANNEL_PACKED"
    expose_evaluated_material_slots(scene)
    controls = gate_materials(scene, cutter)
    switches = []
    for control in controls:
        node_tree = control.node.id_data
        surface = next(node for node in node_tree.nodes if node.type=="OUTPUT_MATERIAL" and node.is_active_output)
        mix = surface.inputs["Surface"].links[0].from_node
        enabled = node_tree.nodes.new("ShaderNodeMixShader")
        node_tree.links.new(mix.outputs[0], enabled.inputs[2])
        node_tree.links.new(mix.inputs[1].links[0].from_socket, enabled.inputs[1])
        node_tree.links.new(enabled.outputs[0], surface.inputs["Surface"])
        switches.append(enabled.inputs[0])

    audits = []
    for frame in accepted:
        scene.frame_set(frame)
        for switch in switches:
            switch.default_value = 0.
        rgb = render_image(scene, workfile, scene_seed+frame, samples, transparent=False)[..., :3]
        for switch in switches:
            switch.default_value = 1.
        stats = layer_labels(
            scene, cutter, controls, switches, rgb, root/"labels"/f"f{frame:03d}-pinhole.npz",
            workfile, scene_seed+frame, samples,
        )
        hidden = [ob for ob in scene.objects if ob.name.startswith("foreground_")]
        previous = [ob.visible_camera for ob in hidden]
        for ob in hidden:
            ob.visible_camera = False
        peeled = render_image(scene,workfile,scene_seed+frame,samples,transparent=False)[...,:3]
        for ob, visible in zip(hidden,previous):
            ob.visible_camera = visible
        save_npz(root/"labels"/f"f{frame:03d}-peeled-color.npz",
                 rgb_linear=np.clip(peeled,0,LDR_LIMIT).astype(np.float16))
        audits.append(dict(frame=frame, **stats))
    save_json(root/"labels/radiance-audit.json", dict(
        frames=frames, elapsed_seconds=time.monotonic()-start, audits=audits, optical=render_meta,
        thin_texture_phase=texture_phase,
        exposure_ev=0, color_space="scene-linear RGB", render_filter="BOX 1.0",
        pinhole_samples=samples, optical_samples_per_subview=reference_samples, clipping_threshold_linear=LDR_LIMIT,
    ))
    if not all(row["passed"] for row in audits):
        raise ValueError("Fixed B layer composition check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=int, default=list(range(1,61)))
    parser.add_argument("--optical", action="store_true")
    parser.add_argument("--optical-only", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Use CPU Cycles for a diagnostic render")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--reference-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    run(args.scene.resolve(), args.frames, args.optical, args.samples, args.seed, args.cpu, args.reference_samples, args.optical_only)


if __name__ == "__main__":
    main()
