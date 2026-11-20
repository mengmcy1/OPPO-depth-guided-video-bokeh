"""Extract one complete 3D-FRONT room from the verified archives and load it in Blender."""
import argparse
from collections import defaultdict
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from urllib.parse import urlsplit
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route3_v2data_B_front import ARCHIVES, DATA, RESERVE_BYTES, TERMS, write_json


def texture_id(url):
    return PurePosixPath(urlsplit(url).path).parts[-2]


def group_files(archive, suffixes):
    grouped = defaultdict(dict)
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
            continue
        folder, name = path.parts[-2:]
        if re.fullmatch(r"[a-zA-Z0-9_-]+",folder) and path.suffix.lower() in suffixes:
            grouped[folder][name] = member
    return grouped


def selected_room(layout, models, textures):
    shapes = {row["uid"]:row for row in layout["mesh"]}
    furniture = {row["uid"]:row for row in layout["furniture"]}
    materials = {row["uid"]:row for row in layout["material"]}
    for room in layout["scene"]["room"]:
        if room.get("type") not in {"Bedroom", "LivingRoom", "LivingDiningRoom", "DiningRoom"}:
            continue
        if any("furniture" in x.get("instanceid", "") and x["ref"] not in furniture for x in room["children"]):
            continue
        architecture = [(shapes[x["ref"]],x) for x in room["children"] if x["ref"] in shapes]
        placed = [(furniture[x["ref"]],x) for x in room["children"] if x["ref"] in furniture]
        if not placed or any(
            row["jid"] not in models or "raw_model.obj" not in models[row["jid"]]
            or "texture.png" not in models[row["jid"]] for row,_ in placed
        ):
            continue
        kinds = {str(row["type"]).casefold() for row,_ in architecture}
        if not ("floor" in kinds and any("wall" in name for name in kinds)):
            continue
        material_ids = {row["material"] for row,_ in architecture if "material" in row}
        relevant = {key:materials[key] for key in material_ids if key in materials}
        required = {
            texture_id(mat[slot]) for mat in relevant.values()
            for slot in ("texture","normaltexture") if mat.get(slot)
        }
        if required <= textures.keys():
            return room,architecture,placed,relevant,required
    return None


def extract_room(destination, skip=0):
    archives = DATA/"sources/3d-front"
    status = json.loads((DATA/"3d-front-status.json").read_text())
    confirmed = {row["file"]:row for row in status.get("downloaded_files",[]) if row.get("verified")}
    if set(confirmed) != set(ARCHIVES):
        raise ValueError("all three 3D-FRONT source archives need verified Drive hashes")
    for name,row in confirmed.items():
        if (archives/name).stat().st_size != row["bytes"]:
            raise ValueError(f"3D-FRONT source archive size changed: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    with ZipFile(archives/ARCHIVES[0]) as layouts, ZipFile(archives/ARCHIVES[1]) as tex, ZipFile(archives/ARCHIVES[2]) as geo:
        textures = group_files(tex, {".png",".jpg",".jpeg"})
        models = group_files(geo, {".obj",".mtl",".png",".jpg",".jpeg"})
        members = sorted(name for name in layouts.namelist() if name.endswith(".json"))
        for member in members:
            layout = json.loads(layouts.read(member))
            try:
                chosen = selected_room(layout, models, textures)
            except (KeyError,TypeError,IndexError):
                continue
            if chosen and skip == 0:
                break
            skip -= chosen is not None
        else:
            raise ValueError("no room could resolve all referenced furniture, textures and architecture")
        room, architecture, placed, materials, required = chosen
        fetch = [(geo,member,destination/"models"/jid/name)
                 for jid in sorted({row["jid"] for row,_ in placed})
                 for name,member in models[jid].items()]
        fetch += [(tex,member,destination/"textures"/key/name)
                  for key in sorted(required) for name,member in textures[key].items()]
        if shutil.disk_usage(destination).free < RESERVE_BYTES+sum(entry.file_size for _,entry,_ in fetch):
            raise RuntimeError("not enough disk for the room dependencies and 60 GiB reserve")
        for archive, entry, path in fetch:
            path.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(entry) as src, path.open("xb") as dst:
                shutil.copyfileobj(src,dst)
    spec = dict(layout_member=member,
                layout_id=layout["uid"], room_id=room["instanceid"], room_type=room["type"],
                architecture=[row for row,_ in architecture],
                furniture=[{"jid":row["jid"],"instance":x} for row,x in placed], materials=materials,
                texture_ids=sorted(required), source_sha256={k:v["sha256"] for k,v in confirmed.items()})
    write_json(destination/"room.json",spec)
    print("3D-FRONT room resolved",spec["room_type"],len(architecture),len(placed),len(required),flush=True)


def load_material(name, color, image_path=None, normal_path=None):
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get("Principled BSDF")
    rgba = np.asarray(color,dtype=float).ravel()[:3]
    if rgba.shape != (3,) or not np.isfinite(rgba).all():
        rgba = np.full(3,180.)
    shader.inputs["Base Color"].default_value = (*np.clip(rgba/255,0,1),1)
    if image_path:
        image = nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(str(image_path.resolve()),check_existing=True)
        links.new(image.outputs["Color"],shader.inputs["Base Color"])
        if image.image.channels == 4:
            rgba_pixels = np.empty(len(image.image.pixels),np.float32)
            image.image.pixels.foreach_get(rgba_pixels)
            if np.any(rgba_pixels[3::4] < .999):
                links.new(image.outputs["Alpha"],shader.inputs["Alpha"])
    if normal_path:
        image = nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(str(normal_path.resolve()),check_existing=True)
        image.image.colorspace_settings.name = "Non-Color"
        normal = nodes.new("ShaderNodeNormalMap")
        links.new(image.outputs["Color"],normal.inputs["Color"])
        links.new(normal.outputs["Normal"],shader.inputs["Normal"])
    return mat


def build_room(root):
    import bpy
    from mathutils import Matrix, Quaternion, Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    spec = json.loads((root/"room.json").read_text())
    baked_materials = {}
    dataset_to_blender = Matrix.Rotation(np.pi/2,4,"X")
    for uid, mat in spec["materials"].items():
        maps = []
        for slot in ("texture","normaltexture"):
            directory = root/"textures"/texture_id(mat[slot]) if mat.get(slot) else None
            maps.append(next(iter(sorted(directory.glob("texture.*"))),None) if directory else None)
            if directory and maps[-1] is None:
                raise ValueError(f"missing {slot} for room material {uid}")
        baked_materials[uid] = load_material(uid,mat.get("color",[180,180,180]),*maps)

    floor = []
    for row in spec["architecture"]:
        verts = np.asarray(row["xyz"],dtype=np.float32).reshape(-1,3)[:,[0,2,1]] * np.array([1,-1,1],np.float32)
        faces = np.asarray(row["faces"],dtype=np.int32).reshape(-1,3)
        if not np.isfinite(verts).all() or faces.min()<0 or faces.max()>=len(verts):
            raise ValueError("invalid architecture geometry indices")
        normals = np.asarray(row["normal"],dtype=np.float32).reshape(-1,3)[:,[0,2,1]] * np.array([1,-1,1],np.float32)
        direction = np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]])
        if len(normals)==len(verts) and np.median(np.sum(direction*normals[faces].mean(1),axis=1))<0:
            faces = faces[:,::-1]
        mesh = bpy.data.meshes.new(row["uid"])
        mesh.from_pydata(verts,[],faces.tolist())
        mesh.update()
        ob = bpy.data.objects.new(row["type"],mesh)
        scene.collection.objects.link(ob)
        mat = baked_materials.get(row.get("material"))
        if mat:
            mesh.materials.append(mat)
        uv = np.asarray(row.get("uv",[]),dtype=np.float32)
        if uv.size==len(verts)*2 and np.isfinite(uv).all():
            layer = mesh.uv_layers.new(name="UVMap")
            uv = uv.reshape(-1,2)
            transform = spec["materials"].get(row.get("material"),{}).get("UVTransform")
            if transform is not None:
                uv = uv @ np.asarray(transform,dtype=np.float32).reshape(3,3)[:2,:2].T
            layer.data.foreach_set("uv",uv[faces].ravel())
        if row["type"].casefold()=="floor":
            floor.extend(verts)

    furniture = []
    for row in spec["furniture"]:
        model_dir = root/"models"/row["jid"]
        instance = row["instance"]
        before = set(scene.objects)
        bpy.ops.wm.obj_import(filepath=str((model_dir/"raw_model.obj").resolve()),
                              forward_axis="Y",up_axis="Z")
        parts = [ob for ob in scene.objects if ob not in before and ob.type == "MESH"]
        if not parts:
            raise ValueError(f"3D-FUTURE model has no meshes: {row['jid']}")
        pos, scale, rotation = [np.asarray(instance[key],dtype=float) for key in ("pos","scale","rot")]
        if (not all(np.isfinite(arr).all() for arr in (pos,scale,rotation))
                or np.any(scale==0) or np.linalg.norm(rotation)<1e-6):
            raise ValueError("invalid furniture transformation")
        # The dataset toolbox applies scale and its XYZW rotation in Y-up coordinates.
        position = dataset_to_blender @ Vector(pos)
        orientation = Quaternion(rotation[[3,0,1,2]]).normalized().to_matrix().to_4x4()
        matrix = (Matrix.Translation(position) @ dataset_to_blender
                  @ orientation @ Matrix.Diagonal((*scale,1)))
        mat = load_material(row["jid"],[180,180,180],model_dir/"texture.png")
        for ob in parts:
            if not ob.data.uv_layers or not ob.data.polygons:
                raise ValueError("furniture mesh lacks texture UVs or faces")
            ob.matrix_world = matrix
            ob.data.materials.clear()
            ob.data.materials.append(mat)
        furniture.append(dict(jid=row["jid"],objects=len(parts),scale=scale.tolist(),
                              source_position_m=pos.tolist(),source_rotation_xyzw=rotation.tolist(),
                              blender_position_m=list(position)))

    extent = np.ptp(floor,axis=0)
    lower, upper = np.min(floor,axis=0),np.max(floor,axis=0)
    if not (1 < min(extent[:2]) and max(extent[:2]) < 50 and extent[2]<.3):
        raise ValueError(f"invalid room floor extent in meters: {extent.tolist()}")
    center = (lower+upper)/2
    bpy.ops.object.camera_add(location=(center[0], lower[1]+.2*extent[1],center[2]+1.45))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((*center[:2],center[2]+1.0))-camera.location).to_track_quat("-Z","Y").to_euler()
    camera.data.lens = 23
    scene.camera = camera
    scene.world = bpy.data.worlds.new("Room preview environment")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value=.65
    bpy.ops.object.light_add(type="AREA",location=(*center[:2],center[2]+2.2))
    bpy.context.object.data.energy = 700
    bpy.context.object.data.size = 2
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 64
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str((root/"room.blend").resolve()))
    write_json(root/"room-load.json",dict(layout_id=spec["layout_id"],room_id=spec["room_id"],
        room_type=spec["room_type"],floor_extent_m=extent.tolist(),furniture=furniture,
        architecture=len(spec["architecture"]),archive_textures=spec["texture_ids"]))


def render_room(root):
    import bpy
    from route3_v2data_B_render import configure

    bpy.ops.wm.open_mainfile(filepath=str((root/"room.blend").resolve()))
    scene = bpy.context.scene
    configure(scene,False)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str((root/"room-preview.png").resolve())
    bpy.ops.render.render(write_still=True)


def review_layout(source, destination):
    import bpy
    from route3_v2data_B_mesh_check import Rejected, audit_mesh, check_materials

    destination.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(source.resolve()))
    scene = bpy.context.scene
    originals = list(scene.objects)
    graph = bpy.context.evaluated_depsgraph_get()
    accepted, excluded = [], []
    for obj in originals:
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = bpy.data.meshes.new_from_object(evaluated,preserve_all_data_layers=True,depsgraph=graph)
        copy = bpy.data.objects.new(obj.name+" opaque",mesh)
        copy.matrix_world = obj.matrix_world.copy()
        try:
            check_materials([copy])
            if not mesh.polygons or not mesh.materials:
                raise Rejected("mesh has no renderable material or polygons")
            audit_mesh(copy)
        except (Rejected, ValueError, KeyError, RuntimeError) as exc:
            excluded.append(dict(object=obj.name,reason=str(exc)))
            bpy.data.objects.remove(copy,do_unlink=True)
            bpy.data.meshes.remove(mesh)
            continue
        scene.collection.objects.link(copy)
        accepted.append(copy)
    if not accepted:
        write_json(destination/"mesh-review.json",dict(accepted=0,excluded=excluded,status="excluded"))
        return
    for obj in originals:
        bpy.data.objects.remove(obj,do_unlink=True)
    bpy.context.view_layer.update()
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str((destination/"layout.blend").resolve()))
    write_json(destination/"mesh-review.json",dict(accepted=len(accepted),excluded=excluded,status="passed",
        render_path="layout.blend", faces=sum(len(obj.data.polygons) for obj in accepted)))


def register_room(source, review):
    loaded = json.loads((source/"room-load.json").read_text())
    room = json.loads((source/"room.json").read_text())
    result = json.loads((review/"mesh-review.json").read_text())
    source_name = "3D-FRONT"
    asset_id = f"{loaded['layout_id']}/{loaded['room_id']}"
    row = dict(source=source_name,asset_id=asset_id,category="layout",status=result["status"],
               author="3D-FRONT and 3D-FUTURE dataset authors",
               license="3D-FRONT / 3D-FUTURE Data Sets Use License Agreements",
               page="https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset",
               terms=TERMS["3D-FRONT"],furniture_terms=TERMS["3D-FUTURE"],
               license_terms_version={"3D-FRONT":"2020-06-18","3D-FUTURE":"2020-03-28"},
               download_channel="authorized dataset archives",can_redistribute=False,
               archive_sha256=room["source_sha256"],floor_extent_m=loaded["floor_extent_m"],
               mesh_review=str((review/"mesh-review.json").resolve().relative_to(DATA.resolve())))
    if row["status"] == "passed":
        from hashlib import file_digest

        asset = (review/"layout.blend").resolve()
        with asset.open("rb") as stream:
            identifier = file_digest(stream,"sha256").hexdigest()
        row.update(mesh_path=str(asset.relative_to(DATA.resolve())),geometry_sha256=identifier,
                   face_count=result["faces"],mesh_components=result["accepted"])
    else:
        row["reason"] = "no individual opaque mesh passed material and topology checks"
    write_json(DATA/"source-candidates/3d-front.json",dict(source=source_name,candidates=[row]))
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("extract","build","render","review"))
    parser.add_argument("destination",type=Path)
    parser.add_argument("--skip",type=int,default=0,help="Skip an independently invalid room candidate")
    parser.add_argument("--source",type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    if args.mode == "extract":
        extract_room(args.destination,args.skip)
    elif args.mode == "review":
        if not args.source:
            parser.error("review requires --source")
        review_layout(args.source,args.destination)
    else:
        {"build":build_room,"render":render_room}[args.mode](args.destination)
