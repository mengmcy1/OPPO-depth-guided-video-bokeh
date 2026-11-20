"""Build a B-tier Blender scene and export the evaluated mesh to labels only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route3_blender_b1 import cube, material
from route3_ovg_subpixel import TRAIN_FRAMES, save_json, save_npz


WIDTH = 512
LENS_MM = 50.0
SENSOR_MM = 36.0
FOCAL = WIDTH * LENS_MM / SENSOR_MM
CAMERA_FRAMES = 60
RESERVE_GIB = 60


def surface(name, rng, weak=False, texture_path=None):
    color = rng.uniform(.3, .36, 3) if weak else rng.uniform(.08, .66, 3)
    mat = material(name, color, .78 if weak else .63)
    if texture_path and not weak:
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        image = nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(str(texture_path), check_existing=True)
        links.new(image.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
    return mat


def world(scene, spec, rng):
    scene.world = bpy.data.worlds.new("Illumination")
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    light = nodes["Background"]
    light.inputs["Color"].default_value = (.25, .28, .33, 1)
    light.inputs["Strength"].default_value = rng.uniform(.3, .7)
    if spec.get("hdri_path"):
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(spec["hdri_path"], check_existing=True)
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value[2] = rng.uniform(0, 2*np.pi)
        coord = nodes.new("ShaderNodeTexCoord")
        links = scene.world.node_tree.links
        links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        links.new(env.outputs["Color"], light.inputs["Color"])
    for location, energy, size in (([-3, -1, 4], 380, 4), ([2.5, 2, 3], 180, 2.5)):
        bpy.ops.object.light_add(type="AREA", location=location)
        fill = bpy.context.object
        fill.data.energy = energy * rng.uniform(.8, 1.2)
        fill.data.size = size
        fill.rotation_euler = (Vector([0, 2, 1.2])-fill.location).to_track_quat("-Z", "Y").to_euler()


def add_tube_group(name, edges, radius, mat):
    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    join = nodes.new("GeometryNodeJoinGeometry")
    profile = nodes.new("GeometryNodeCurvePrimitiveCircle")
    profile.inputs["Resolution"].default_value = 12
    profile.inputs["Radius"].default_value = radius
    for begin, end in edges:
        line = nodes.new("GeometryNodeCurvePrimitiveLine")
        line.inputs["Start"].default_value = begin
        line.inputs["End"].default_value = end
        tube = nodes.new("GeometryNodeCurveToMesh")
        tube.inputs["Fill Caps"].default_value = True
        links.new(line.outputs["Curve"], tube.inputs["Curve"])
        links.new(profile.outputs["Curve"], tube.inputs["Profile Curve"])
        links.new(tube.outputs["Mesh"], join.inputs["Geometry"])
    assign = nodes.new("GeometryNodeSetMaterial")
    assign.inputs["Material"].default_value = mat
    links.new(join.outputs["Geometry"], assign.inputs["Geometry"])
    out = nodes.new("NodeGroupOutput")
    links.new(assign.outputs["Geometry"], out.inputs["Geometry"])
    ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    bpy.context.scene.collection.objects.link(ob)
    modifier = ob.modifiers.new(name, "NODES")
    modifier.node_group = group
    return ob


def make_structure(spec, near, material, rng):
    segment = spec["structure"]
    shape = segment["shape"]
    width = rng.uniform(*segment["width_px_at_512"])
    radius = near * width / (2 * FOCAL)
    xspan, height, y = near * .27, near * .50, near
    z0 = 1.15 - height/2
    count = 6 + segment["parameter_segment"]
    xs = np.linspace(-xspan, xspan, count)
    vertical = [((x, y, z0), (x, y, z0+height)) for x in xs]
    horizontal = [
        ((-xspan, y, z), (xspan, y, z))
        for z in np.linspace(z0, z0+height, count if shape == "mesh" else 3)
    ]
    if shape == "wire":
        edges = [((a, y, z0), (b, y, z0+height)) for a, b in zip(xs, xs[::-1])]
    elif shape == "branch":
        edges = [vertical[count//2]]
        edges.extend(((0, y, z0 + height*.35), (x, y, z0+height*.93)) for x in xs)
    else:
        edges = vertical + (horizontal if shape != "railing" else horizontal[:2])
    add_tube_group("foreground_structure", edges, radius, material)
    return float(width)


def add_mesh_asset(path, name, position, extent):
    with bpy.data.libraries.load(str(path), link=False) as (source, target):
        target.objects = source.objects
    assets = [ob for ob in target.objects if ob and ob.type == "MESH"]
    if not assets:
        raise ValueError(f"No mesh objects for {name}")
    for ob in assets:
        if not ob.users_collection:
            bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    corners = np.asarray([tuple(ob.matrix_world @ Vector(c)) for ob in assets for c in ob.bound_box])
    lower, upper = corners.min(0), corners.max(0)
    scale = extent / max(upper-lower)
    center = Vector((lower+upper)/2)
    for ix, ob in enumerate(assets):
        ob.name = f"{name}_{ix:03d}"
        ob.location = Vector(position) + (ob.location-center)*scale
        ob.scale *= scale
    return assets


def populate(spec, layout_rng, foreground_rng, mat_rng):
    category = spec["category"]
    near = float(spec["near_anchor_m"])
    far = near * spec["depth_ratio"]
    weak = category == "weak_texture"
    position = (0, far+.04, 1.2)
    tex = spec.get("texture_path")
    wall = surface("Background diffuse", mat_rng, weak=weak, texture_path=tex)
    floor = surface("Ground diffuse", mat_rng, weak=weak)
    near_mat = surface("Foreground diffuse", mat_rng, weak=weak)
    mid_mat = surface("Middle diffuse", mat_rng, weak=weak)
    wall_at = far*1.2 if spec.get("layout_path") else position[1]
    cube("background_wall", (position[0],wall_at,position[2]), (max(6, far), .08, max(3, far)), wall)
    cube("background_floor", (0, far/2, -.055), (max(6, far), far+4, .1), floor)
    for k in range(5 if category == "complex_layout" else 3):
        x, z = layout_rng.uniform(-.5, .5), layout_rng.uniform(.35, 1.8)
        size = (.14*far, .07, .24)
        if spec.get("priority") == "large_scale_small_disparity":
            size = (near*.065, near*.036, near*.07)
        cube(f"background_layout_{k}", (x*far, far-.05, z), size, floor)
    if spec.get("layout_path"):
        add_mesh_asset(spec["layout_path"], "background_import", (0, far, 1.15), far*.26)
    width = None
    if category == "thin_structure":
        width = make_structure(spec, near, near_mat, foreground_rng)
    elif spec.get("occluder_assets"):
        gap = (far-near)/(spec["depth_bands"]-1)
        for index, asset in enumerate(spec["occluder_assets"]):
            distance = near+gap*index
            extent = distance*(.81 if category=="large_foreground" else .64)
            if len(spec["occluder_assets"])>1:
                axes = asset["scale_m"]
                extent = min(extent,.82*gap*max(axes)/max(axes[1],.0005))
            x = 0 if index==0 else (-1)**index*.12*distance
            anchor = Vector((x,distance,1.16))
            placed = add_mesh_asset(asset["path"],f"foreground_import_{index:02d}",anchor,extent)
            if category == "curved_self_occlusion":
                angle = foreground_rng.uniform(.24,.52)
                turn = Matrix.Translation(anchor) @ Matrix.Rotation(angle,4,"Z") @ Matrix.Translation(-anchor)
                for ob in placed:
                    ob.matrix_world = turn @ ob.matrix_world
            if weak:
                for ob in placed:
                    ob.data.materials.clear()
                    ob.data.materials.append(near_mat)
    elif category == "curved_self_occlusion":
        if spec.get("sloped_surface"):
            ob = cube("foreground_sloped", (0, near, 1.15), (near*.52, near*.18, near*.65), near_mat)
            ob.rotation_euler.z = foreground_rng.uniform(.45, .8)
        else:
            bpy.ops.mesh.primitive_torus_add(
                major_radius=near*.20, minor_radius=near*.11, location=(0, near, 1.15),
                rotation=(np.pi/2.8, 0.3, 0.25),
            )
            bpy.context.object.name = "foreground_curved"
            bpy.context.object.data.materials.append(near_mat)
    elif category == "complex_layout" and spec.get("model_path"):
        add_mesh_asset(spec["model_path"], "foreground_import", (0, near*1.16, 1.16), near*.64)
    else:
        dims = (near*.48, near*.13, near*.46) if category == "large_foreground" else (near*.29, near*.10, near*.45)
        offset = foreground_rng.uniform(-.09, .09)
        cube("foreground_main", (offset, near, 1.15), dims, near_mat)
    for layer in range(max(2,1+len(spec.get("occluder_assets",[]))), spec["depth_bands"]):
        fraction = (layer-1)/(spec["depth_bands"]-1)
        distance = near + (far-near)*fraction
        cube(f"foreground_middle_{layer}", (.09*near, distance, 1.05), (.30*distance, .06, .44), mid_mat)
    if spec.get("hole_area_fraction", 0):
        # Frame side lengths yield the requested centered rectangular opening.
        outer = near*.56
        inner = outer * np.sqrt(spec["hole_area_fraction"])
        bar = (outer-inner)/2
        x, y, z = -.20*near, near*.90, 1.15
        for sign, side in ((-1,"left"),(1,"right")):
            cube(f"foreground_window_{side}", (x+sign*(inner+bar)/2,y,z), (bar,.07,outer), near_mat)
        for sign, side in ((-1,"bottom"),(1,"top")):
            cube(f"foreground_window_{side}", (x,y,z+sign*(inner+bar)/2), (inner,.07,bar), near_mat)
    if spec.get("highlights"):
        emitter = surface("Warm emissive fixture", mat_rng)
        shader = emitter.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Emission Color"].default_value = (1, .68, .38, 1)
        shader.inputs["Emission Strength"].default_value = 12
        for k, x in enumerate((-.35*far, .16*far, .24*far)):
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=.025*far, location=(x, .93*far, 1.26))
            bpy.context.object.name = f"background_emitter_{k}"
            bpy.context.object.data.materials.append(emitter)
    return width


def camera_path(scene, spec):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    scene.camera = camera
    lens = camera.data
    lens.lens, lens.sensor_width, lens.sensor_fit = LENS_MM, SENSOR_MM, "HORIZONTAL"
    lens.clip_start, lens.clip_end = 0.003, 1000
    baseline = spec["camera_baseline_m"]
    rotation = np.deg2rad(spec["rotation_span_deg"])
    cameras = []
    for frame, t in enumerate(np.linspace(-.5, .5, CAMERA_FRAMES), 1):
        camera.location = (baseline*t, 0, 1.15)
        yaw = rotation*t
        ray = Vector((np.sin(yaw), np.cos(yaw), .005 * np.sin(2*np.pi*t)))
        camera.rotation_euler = ray.to_track_quat("-Z", "Y").to_euler()
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
        scene.frame_set(frame)
        cameras.append(np.array(camera.matrix_world))
    return np.stack(cameras)


def export_labels(scene, cameras, dest):
    objects = sorted((ob for ob in scene.objects if ob.type == "MESH"), key=lambda ob: ob.name)
    arrays, ids, tri_object, tri_poly, transforms, foreground = {}, [], [], [], [], []
    graph = bpy.context.evaluated_depsgraph_get()
    for index, ob in enumerate(objects, 1):
        evaluated = ob.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            if not mesh.loop_triangles:
                raise ValueError(f"No renderable evaluated triangles: {ob.name}")
            triangles = np.array([face.vertices[:] for face in mesh.loop_triangles], dtype=np.int32)
            arrays[f"object_{index}_vertices"] = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)
            arrays[f"object_{index}_faces"] = triangles
            tri_object.append(np.full(len(triangles), index, dtype=np.int16))
            tri_poly.append(np.array([face.polygon_index for face in mesh.loop_triangles], dtype=np.int32))
            transforms.append(np.array(ob.matrix_world))
            ob.pass_index = index
            ids.append(index)
            if ob.name.startswith("foreground_"):
                foreground.append(index)
        finally:
            evaluated.to_mesh_clear()
    intrinsics = np.array([[FOCAL, 0, WIDTH/2], [0, FOCAL, WIDTH/2], [0, 0, 1]])
    cv = cameras @ np.diag([1., -1., -1., 1.])
    save_npz(
        dest / "labels/scene-geometry.npz", **arrays,
        object_names=np.array([ob.name for ob in objects]), object_ids=np.array(ids),
        triangle_object_id=np.concatenate(tri_object), triangle_polygon_id=np.concatenate(tri_poly),
        object_world_matrices=np.repeat(np.array(transforms)[None, ...], len(cameras), axis=0),
        camera_c2w_blender=cameras, camera_c2w_opencv=cv, intrinsics=intrinsics,
        foreground_object_ids=np.array(foreground, np.int16), dynamic_object_ids=np.empty(0, np.int16),
        input_frames=np.arange(1, CAMERA_FRAMES+1, dtype=np.int16),
    )
    save_json(dest / "observations/cameras.json", dict(K=intrinsics.tolist(), poses=cv.tolist()))
    return len(objects), int(sum(len(faces) for faces in tri_object))


def build(spec, dest):
    if (dest / "scene.blend").is_file():
        raise FileExistsError(dest)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, CAMERA_FRAMES
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = scene.render.resolution_y = WIDTH
    scene.render.resolution_percentage = 100
    scene.render.threads_mode, scene.render.threads = "FIXED", 6
    scene.render.use_motion_blur = False
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.pixel_filter_type, scene.cycles.filter_width = "BOX", 1.0
    scene.view_settings.view_transform, scene.view_settings.look = "Standard", "None"
    scene.view_settings.exposure, scene.view_settings.gamma = 0, 1
    rngs = [np.random.default_rng(seed) for seed in np.random.SeedSequence(spec["seed"]).spawn(4)]
    width = populate(spec, *rngs[:3])
    world(scene, spec, rngs[3])
    cameras = camera_path(scene, spec)
    objects, faces = export_labels(scene, cameras, dest)
    scene.frame_set(1)
    bpy.ops.file.pack_all()
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(dest / "scene.blend"))
    save_json(dest/"scene-build.json", dict(
        category=spec["category"], seed=spec["seed"], blender=bpy.app.version_string,
        frame_count=CAMERA_FRAMES, labeled_frames=TRAIN_FRAMES,
        objects=objects, triangles=faces, designated_width_px=width,
        input_metadata="observations/cameras.json", supervision="labels/",
    ))


def validate_structures(candidates, output):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mat = material("Opaque geometry check", (.3, .4, .6), .6)
    checked = []
    for row in json.loads(candidates.read_text())["candidates"]:
        spec = {"structure": row}
        rng = np.random.default_rng(int(row["parameter_segment"]))
        width = make_structure(spec, 1.5, mat, rng)
        ob = bpy.data.objects["foreground_structure"]
        group = ob.modifiers[0].node_group
        empty_mesh = ob.data
        evaluated = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            points = np.asarray([tuple(v.co) for v in mesh.vertices])
            triangles = [tuple(face.vertices) for face in mesh.loop_triangles]
            if not triangles:
                checked.append(dict(row, status="excluded", reason="no evaluated triangles"))
                continue
            bvh = BVHTree.FromPolygons(points, triangles, all_triangles=True)
            face = triangles[len(triangles)//2]
            origin = Vector((0., 0., 1.15))
            target = Vector(points[list(face)].mean(0))
            hit = bvh.ray_cast(origin, (target-origin).normalized())[2]
            if hit is None:
                checked.append(dict(row, status="excluded", reason="representative curve becomes unreachable in mesh BVH"))
                continue
            scale = np.ptp(points,axis=0)
            canonical = np.unique(np.rint((points-points.min(0))/max(scale)*100000).astype("<i4"),axis=0)
            checked.append(dict(
                row, status="passed", face_count=len(triangles), scale_m=scale.tolist(),
                geometry_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),
                evaluated_as_mesh=True, source_probe_width_px=width,
            ))
        finally:
            evaluated.to_mesh_clear()
            bpy.data.objects.remove(ob,do_unlink=True)
            bpy.data.node_groups.remove(group)
            bpy.data.meshes.remove(empty_mesh)
    save_json(output, checked)
    print("Procedural segment BVH checks",sum(row["status"]=="passed" for row in checked),len(checked))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--validate-structures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    if args.validate_structures:
        validate_structures(args.validate_structures, args.output)
    elif args.spec:
        build(json.loads(args.spec.read_text()), args.output)
    else:
        parser.error("Use --spec or --validate-structures")


if __name__ == "__main__":
    main()
