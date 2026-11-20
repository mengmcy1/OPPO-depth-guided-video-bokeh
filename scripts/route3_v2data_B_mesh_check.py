"""Blender batch filter for B-tier opaque, ray-intersectable asset geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile

import bmesh
import bpy
from mathutils import Vector
import numpy as np


UNSUPPORTED_NODES = {
    "BSDF_TRANSPARENT", "BSDF_GLASS", "BSDF_REFRACTION",
    "VOLUME_ABSORPTION", "VOLUME_SCATTER", "PRINCIPLED_VOLUME",
}
MESH_SUFFIXES = {".gltf", ".glb", ".obj", ".blend"}


class Rejected(ValueError):
    """An asset that cannot be used under the mesh/radiance contract."""


def material_nodes(tree, visited):
    if not tree or tree in visited:
        return
    visited.add(tree)
    for node in tree.nodes:
        yield node
        if node.type == "GROUP":
            yield from material_nodes(node.node_tree, visited)


def check_materials(objects):
    materials = {slot.material for ob in objects for slot in ob.material_slots if slot.material}
    for material in materials:
        for node in material_nodes(material.node_tree, set()):
            if node.type in UNSUPPORTED_NODES:
                raise Rejected(f"transparent, refractive, or volumetric shader: {node.type}")
            if node.type == "BSDF_PRINCIPLED":
                for key, threshold in (("Alpha", 1.0), ("Transmission Weight", 0.0)):
                    socket = node.inputs[key]
                    if socket.is_linked or abs(socket.default_value - threshold) > 1e-6:
                        raise Rejected(f"potential alpha card or transmission: {key}")
            if node.type != "TEX_IMAGE":
                continue
            image = node.image
            if image is None:
                raise Rejected("image texture is missing")
            source_path = bpy.path.abspath(image.filepath, library=image.library)
            if not image.packed_file and image.source != "GENERATED" and not Path(source_path).is_file():
                raise Rejected("external image texture is missing")


def import_objects(path):
    if path.suffix.lower() in {".gltf", ".glb"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
        return list(bpy.context.scene.objects)
    if path.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
        return list(bpy.context.scene.objects)
    with bpy.data.libraries.load(str(path), link=False) as (source, target):
        target.objects = source.objects
    objects = [ob for ob in target.objects if ob is not None]
    for ob in objects:
        if not ob.users_collection:
            bpy.context.scene.collection.objects.link(ob)
    return objects


def unwrap(files, location):
    candidates = []
    for file in files:
        if file.suffix.lower() in MESH_SUFFIXES:
            candidates.append(file)
        if file.suffix.lower() != ".zip":
            continue
        unpack = location / file.stem
        with ZipFile(file) as archive:
            members = archive.infolist()
            total = sum(item.file_size for item in members)
            if shutil.disk_usage(location).free < 60 * 2**30 + total:
                raise Rejected("not enough disk for validated extraction")
            if any(Path(item.filename).is_absolute() or ".." in Path(item.filename).parts for item in members):
                raise Rejected("unsafe archive path")
            for member in members:
                if not member.is_dir():
                    target = unpack / member.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        with archive.open(member) as source, target.open("xb") as stream:
                            shutil.copyfileobj(source, stream)
                    if target.suffix.lower() in MESH_SUFFIXES and "__MACOSX" not in target.parts:
                        candidates.append(target)
    if not candidates:
        raise Rejected("no supported mesh in the official asset files")
    return sorted(candidates)


def audit_mesh(ob):
    mesh = bmesh.new()
    try:
        mesh.from_mesh(ob.data)
        if not mesh.faces:
            raise Rejected("mesh has no faces")
        extent = max(ob.dimensions)
        bmesh.ops.remove_doubles(mesh, verts=list(mesh.verts), dist=max(1e-8, extent * 1e-6))
        if any(not edge.is_manifold for edge in mesh.edges):
            raise Rejected("non-manifold mesh after welding duplicated seam vertices")
        if any(not edge.is_contiguous for edge in mesh.edges) or mesh.calc_volume(signed=True) <= 0:
            raise Rejected("inconsistent winding or inward-facing closed surface")
        if any(face.calc_area() <= 1e-14 for face in mesh.faces):
            raise Rejected("zero-area face")
    finally:
        mesh.free()


def audit_asset(objects, source, dest):
    if any(ob.type in {"CURVES", "CURVE", "SURFACE"} for ob in objects):
        raise Rejected("unevaluated hair or curve geometry")
    meshes = [ob for ob in objects if ob.type == "MESH" and not ob.hide_render]
    if not meshes:
        raise Rejected("no visible mesh")
    check_materials(meshes)
    bpy.context.view_layer.update()
    dependency = bpy.context.evaluated_depsgraph_get()
    output_objects, coords, faces = [], [], 0
    for ob in meshes:
        evaluated = ob.evaluated_get(dependency)
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=dependency)
        normalized = bpy.data.objects.new("B mesh " + ob.name, mesh)
        bpy.context.scene.collection.objects.link(normalized)
        normalized.matrix_world = ob.matrix_world.copy()
        bpy.context.view_layer.objects.active = normalized
        if source.get("decimate") and len(mesh.polygons) > 30000:
            modifier = normalized.modifiers.new("Route 3 scan decimation", "DECIMATE")
            modifier.ratio = 30000 / len(mesh.polygons)
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        audit_mesh(normalized)
        faces += len(normalized.data.polygons)
        coords.extend(normalized.matrix_world @ vertex.co for vertex in normalized.data.vertices)
        output_objects.append(normalized)
    points = np.asarray(coords, dtype=np.float64)
    size = np.ptp(points, axis=0)
    largest = float(size.max())
    if not np.isfinite(points).all() or not 0.0005 <= largest <= 500:
        raise Rejected("invalid asset scale")
    canonical = np.unique(np.rint((points-points.min(axis=0))/largest*100000).astype("<i4"), axis=0)
    for ob in list(bpy.context.scene.objects):
        if ob not in output_objects:
            bpy.data.objects.remove(ob, do_unlink=True)
    bpy.ops.file.pack_all()
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(dest))
    return dict(scale_m=size.tolist(), face_count=faces, geometry_sha256=hashlib.sha256(canonical.tobytes()).hexdigest())


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def run(row, base):
    files = [base / item["path"] for item in row["files"]]
    work = files[0].parent
    paths = unwrap(files, work)
    main = min(paths, key=lambda p: (p.suffix.lower() not in {".gltf", ".glb"}, str(p)))
    if row["source"] == "Google Scanned Objects" and main.suffix == ".obj":
        texture = main.parent.parent / "materials/textures/texture.png"
        relative_texture = main.parent / "texture.png"
        if texture.is_file() and not relative_texture.exists():
            relative_texture.hardlink_to(texture)
    reset_scene()
    objects = import_objects(main)
    if row["category"] == "mesh_bundle":
        assets = [ob for ob in objects if ob.asset_data and ob.type == "MESH"]
        if not assets:
            assets = [ob for ob in objects if ob.type == "MESH" and not ob.hide_render]
        names = [ob.name for ob in assets]
        for index, name in enumerate(names):
            reset_scene()
            with bpy.data.libraries.load(str(main), link=False) as (_source, target):
                target.objects = [name]
            ob = target.objects[0]
            bpy.context.scene.collection.objects.link(ob)
            dest = work / f"accepted-{index:03d}.blend"
            entry = dict(row, asset_id=name, category="mesh", bundle=row["asset_id"])
            yield inspect(entry, [ob], dest, base)
    else:
        yield inspect(row, objects, work / "accepted.blend", base)


def inspect(row, objects, path, base):
    try:
        result = audit_asset(objects, row, path)
    except (Rejected, RuntimeError, ValueError) as exc:
        return dict(row, status="excluded", reason=str(exc))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(2**20):
            digest.update(data)
    return dict(row, status="passed", mesh_path=str(path.relative_to(base)), mesh_sha256=digest.hexdigest(), **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    rows = json.loads(args.candidates.read_text())["candidates"]
    output = []
    for row in rows:
        if row["status"] != "pending_mesh_checks":
            output.append(row)
            continue
        try:
            output.extend(run(row, args.base.resolve()))
        except (Rejected, RuntimeError, ValueError, OSError) as exc:
            output.append(dict(row, status="excluded", reason=str(exc)))
    temp = args.output.with_suffix(".part")
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    temp.replace(args.output)


if __name__ == "__main__":
    main()
