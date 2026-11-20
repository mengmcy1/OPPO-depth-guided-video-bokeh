"""Preseal imported-asset coverage and release the B v2 generation plans."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp

from route3_v2data_B_front import DATA, RECORD, write_json
from route3_v2data_B_inputs import FORMAL_DATA as V2
from route3_v2data_B_plan import MODES, QUOTA, digest, generator_code, integer_seed, is_reserved, load, plan
from route3_v2data_B_registry import register
from route3_v2data_B_sources import canonical_pool


TERMINAL_SOURCES = ("gso","poly-haven","smithsonian","blendkit","sketchfab","objaverse","3d-future","3d-front","reused-v1-meshes")


def reused_assets():
    rows = load(DATA/"asset-registry-v1.json")["usable"]
    def file_link(path):
        source = DATA/path
        target = V2/"reused-v1"/path
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            target.symlink_to(source.resolve())
        return str(target.relative_to(V2))

    for source in rows:
        if source["category"] not in {"material","hdri","procedural_structure","layout"}:
            continue
        row = {k:v for k,v in source.items() if k not in {"key","pool","fingerprint"}}
        for name in ("mesh_path","render_path"):
            if row.get(name):
                row[name] = file_link(row[name])
        if row.get("files"):
            row["files"] = [dict(file,path=file_link(file["path"])) for file in row["files"]]
        yield row


def stage_legacy_meshes():
    from route3_v2data_B_batch import sha256

    rows = load(DATA/"asset-registry-v1.json")["usable"]
    fresh = {"Google Scanned Objects","Poly Haven","Smithsonian Open Access","Blendkit","Sketchfab","Objaverse"}
    keep = []
    for row in rows:
        if row["category"] != "mesh" or row["source"] in fresh:
            continue
        original = DATA/row["mesh_path"]
        key = hashlib.sha256(row["key"].encode()).hexdigest()[:20]
        staged = V2/"source-files"/key/"inherited.blend"
        staged.parent.mkdir(exist_ok=True,parents=True)
        if not staged.exists():
            staged.hardlink_to(original.resolve())
        entries = {k:v for k,v in row.items() if k not in {
            "key","pool","fingerprint","mesh_path","render_path","files","geometry_sha256",
        }}
        entries.update(status="pending_mesh_checks",
            files=[dict(path=str(staged.relative_to(V2)),bytes=staged.stat().st_size,sha256=sha256(staged))],
            inherited_registry="v1; reviewed again under v2 topology")
        keep.append(entries)
    write_json(V2/"source-candidates/reused-v1-meshes.json",
               dict(source="Previously reviewed mesh sources",status="complete", catalog_complete=True,
                    catalog_count=len(keep),candidates=keep))


def is_curved_asset(row):
    values = row.get("scale_m") or []
    return len(values)==3 and row.get("face_count",0)>=48 and min(values)>.025*max(values)


def assignment(specs, registry):
    indexed=defaultdict(list)
    for row in registry["usable"]:
        indexed[row["category"], row["pool"]].append(row)
    frequency=defaultdict(Counter)

    def draw(spec, kind, used):
        category=spec["category"]
        pool=spec["material_pool"]
        eligible=[row for row in indexed[kind,pool] if row["key"] not in used]
        if kind=="mesh" and category=="curved_self_occlusion":
            eligible=[row for row in eligible if is_curved_asset(row)]
        if kind=="mesh" and category=="large_foreground":
            eligible=[row for row in eligible if min(row["scale_m"][::2]) >= .32*max(row["scale_m"])]
        if not eligible:
            raise ValueError(f"no eligible imported {kind} for {spec['partition']}/{category}/{pool}")
        seen=frequency[spec["partition"],category,kind]
        row=min(eligible,key=lambda item:(seen[item["key"]],integer_seed(spec["seed"],item["key"])))
        seen[row["key"]]+=1
        used.add(row["key"])
        return row

    for rows in specs.values():
        for spec in rows:
            specs_used=set(spec["assets"])
            spec.update(data_version="v2",imported_meshes=[],imported_layouts=[])
            if spec["category"]=="thin_structure":
                continue
            count=spec["depth_bands"]-1 if spec["category"]=="multiple_layers" else 1
            occluders=[]
            for _ in range(count):
                item=draw(spec,"mesh",specs_used)
                occluders.append(dict(key=item["key"],path=str((V2/item["mesh_path"]).resolve()),scale_m=item["scale_m"]))
                spec["imported_meshes"].append(item["key"])
            spec["occluder_assets"]=occluders
            if spec["category"]=="complex_layout":
                layout=draw(spec,"layout",specs_used)
                spec["imported_layouts"].append(layout["key"])
                spec["layout_path"]=str((V2/layout["mesh_path"]).resolve())
            spec["assets"] += spec["imported_meshes"]+spec["imported_layouts"]
    return specs


def usage(specs, categories):
    groups={**specs,"test":specs["test_unseen_assets"]+specs["test_unseen_occlusions"]}
    results={}
    for mode, scenes in groups.items():
        rows={}
        for category in categories:
            subset=[scene for scene in scenes if scene["category"]==category]
            meters=[Counter(key for row in subset for key in set(row[name])) for name in ("imported_meshes","imported_layouts")]
            rows[category]=dict(scenes=len(subset),unique_meshes=len(meters[0]),unique_layouts=len(meters[1]),
                                max_mesh_reuse=max(meters[0].values(),default=0),
                                max_layout_reuse=max(meters[1].values(),default=0))
        results[mode]=rows
    return results


def check_plan(specs, config):
    expected={"training":300,"development":15,"test_unseen_assets":15,"test_unseen_occlusions":15}
    if {key:len(rows) for key,rows in specs.items()}!=expected:
        raise ValueError("unexpected v2 scene counts")
    for mode, scenes in specs.items():
        pool,reserved=MODES[mode]
        for row in scenes:
            if not all(canonical_pool(key)==pool for key in row["assets"]):
                raise ValueError(f"asset pool violation in {row['scene_id']}")
            if is_reserved(row,config["reserved_configuration"])!=reserved:
                raise ValueError(f"held-out config violation in {row['scene_id']}")
            if row["category"]!="thin_structure" and not row["imported_meshes"]:
                raise ValueError(f"missing imported primary occluder in {row['scene_id']}")
    actual = Counter(tuple(row[key] for key in ("category","distance","motion"))
                     for row in specs["training"])
    targets = {tuple(cell[key] for key in ("occlusion","distance","motion")):cell["training_scenes"]
               for cell in config["geometry"]["quota_cells"]}
    if actual != targets:
        raise ValueError("v2 did not preserve the 54 training quota cells")


def evidence():
    labels=V2/"qualification/open-room-labels-world-meters/validation.json"
    sources={}
    unfinished=[]
    for slug in TERMINAL_SOURCES:
        path=V2/"source-candidates"/(slug+".json")
        row=load(path)
        if not row.get("catalog_complete") and row.get("status") not in {"待凭据","catalog_unavailable"}:
            unfinished.append(slug)
        if row:
            sources[slug]={"status":row["status"],"sha256":digest(path),"catalog_count":row.get("catalog_count"),
                           "reason":row.get("reason")}
    return labels,sources,unfinished


def freeze(registry, specs, coverage, labels, sources):
    record=load(RECORD)
    row=record["v2"]
    rules=json.dumps(row["rules"],sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()
    if hashlib.sha256(rules).hexdigest()!=row["rules_sha256"]:
        raise ValueError("the pre-registered v2 rules digest changed")
    if (V2/"sealed-scene-plans").exists():
        raise FileExistsError("v2 is already sealed")
    row.update(status="coverage_written_before_seal",preseal_coverage=coverage,source_dispositions=sources)
    write_json(RECORD,record)

    registry=dict(registry,status="frozen-v2",open_mesh_audit_sha256=digest(labels),sources=sources)
    registry_path=V2/"asset-registry-v2.json"
    checksum=write_json(registry_path,registry)
    authors=V2/"asset-attribution-v2.json"
    attributions=load(V2/"asset-attribution-provisional.json")
    write_json(authors,attributions)
    code=generator_code()
    release_bytes=json.dumps([code,checksum,row["rules_sha256"]],sort_keys=True).encode()
    version="B-v2-"+hashlib.sha256(release_bytes).hexdigest()[:12]
    stage=Path(mkdtemp(dir=V2,prefix=".sealing-"))
    hashes={}
    for mode, scenes in specs.items():
        hashes[mode]=write_json(stage/(mode+".json"),dict(generator_version=version, generator_code=code,
                quota_sha256=digest(QUOTA),registry_sha256=checksum, rules_sha256=row["rules_sha256"],scenes=scenes))
    write_json(stage/"sha256.json",hashes)
    stage.rename(V2/"sealed-scene-plans")
    manifest=V2/"manifest.json"
    target=row["rules"]["generation_targets"]
    manifest_sha=write_json(manifest,dict(archive="B/v2",status="in_progress",data_version="v2",
                 eligible_for_formal_use=True,generator_version=version,units=[],counts={k:0 for k in target},
                 targets=target,asset_registry_sha256=checksum,asset_attribution_sha256=digest(authors),
                 rules_sha256=row["rules_sha256"],scene_plan_sha256=hashes))
    row.update(status="sealed",generator_version=version,manifest_sha256=manifest_sha,
               scene_plan_sha256=hashes,asset_registry_sha256=checksum,
               source_license_counts=registry["counts_by_license"],source_pool_counts=registry["counts_by_source_category_pool"])
    write_json(RECORD,record)
    print(version,hashes,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",action="store_true",help="Seal the reviewed v2 pool and all scene plans")
    parser.add_argument("--stage-existing",action="store_true",help="Stage unaffected v1 meshes for rechecking")
    args=parser.parse_args()
    if args.stage_existing:
        stage_legacy_meshes()
        return
    labels,sources,unfinished=evidence()
    registry=register(V2,reused_assets())
    print("registry",len(registry["usable"]),"pending",len(registry["pending"]),"excluded",len(registry["exclusions"]),flush=True)
    if unfinished or registry["pending"] or not load(labels).get("passed"):
        if args.freeze:
            raise ValueError(f"source or qualification disposition remains outstanding: {unfinished}")
        return
    config=load(QUOTA)
    config["seed"]=integer_seed(config["seed"],"B-v2")
    specs=assignment(plan(config,registry,archive=V2,legacy_placement=False),registry)
    check_plan(specs,config)
    coverage=usage(specs,sorted({row["occlusion"] for row in config["geometry"]["quota_cells"]}))
    write_json(V2/"preseal-coverage.json",coverage)
    if args.freeze:
        freeze(registry,specs,coverage,labels,sources)


if __name__=="__main__":
    main()
