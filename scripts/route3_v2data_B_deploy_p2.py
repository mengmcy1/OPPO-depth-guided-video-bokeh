"""Archive the affected admitted B units and install the qualified p2 release."""
import argparse
from collections import Counter
from pathlib import Path
import time

import psutil

from route3_v2data_B_batch import archive_status, sha256
from route3_v2data_B_dispatch import PARTITIONS, STOP, dispatch, production_lock
from route3_v2data_B_front import RECORD, write_json
from route3_v2data_B_inputs import FORMAL_DATA
from route3_v2data_B_placement import policy
from route3_v2data_B_plan import MODES, load
from route3_v2data_B_release import (
    OPTICAL_FRAMES, PLAN_DIR, code_hash, dispatch_code, load_production_plan,
    p2_changes, p2_generation_code, parent_p1_code, sealed_plan, validate_p2,
)
from route3_ovg_subpixel import TRAIN_FRAMES


ARCHIVE = FORMAL_DATA / "archived-p1"
AFFECTED = {"curved_self_occlusion", "multiple_layers"}
EXPECTED = {
    ("training", "curved_self_occlusion"): 2,
    ("development", "curved_self_occlusion"): 3,
    ("test_unseen_assets", "curved_self_occlusion"): 2,
    ("test_unseen_occlusions", "curved_self_occlusion"): 4,
    ("test_unseen_assets", "multiple_layers"): 4,
    ("test_unseen_occlusions", "multiple_layers"): 2,
}


def qualification_summary(proof_path, proof):
    controls = {name: dict(category=row["category"], passed=row["passed"],
                           structure_equal=row.get("structure_equal"), differing_labels=row.get("differing_labels"),
                           label_file_count=row.get("label_file_count"), failure=row.get("failure"))
                for name, row in proof["invariance"].items()}
    return dict(
        path=str(proof_path.resolve().relative_to(FORMAL_DATA.resolve())), sha256=sha256(proof_path),
        status=proof["status"], seconds=proof["seconds"], smoke=proof["smoke"],
        invariance=controls, smoke_passed=sum(row["passed"] for row in proof["smoke"].values()),
        invariance_passed=sum(row["passed"] for row in controls.values()),
        fallback_count=sum(row.get("fallback_count", 0) for row in proof["smoke"].values()),
        unchanged_category_proof=proof.get("unchanged_category_proof"),
    )


def archive_rows(state, specs):
    rows = []
    for unit in state["units"]:
        scene_id = unit["scene_id"]
        spec = specs[scene_id]
        category = spec["category"]
        if category not in AFFECTED:
            continue
        source = Path(unit["manifest"])
        dest = Path(ARCHIVE.name) / spec["partition"] / source.parent.name / source.name
        rows.append(dict(scene_id=scene_id, partition=spec["partition"], category=category,
                         prior_generator_version=unit["generator_version"], original_manifest=str(source),
                         archived_manifest=str(dest), original_manifest_sha256=unit["manifest_sha256"],
                         bytes=unit["bytes"]))
    if Counter((row["partition"], row["category"]) for row in rows) != EXPECTED:
        raise ValueError("admitted curve and multilayer archive inventory differs from the specified 17 units")
    if state["counts"]["train_optical"]:
        raise ValueError("review the inventory of attached training optical units before migration")
    return rows


def move_archives(rows):
    for row in rows:
        original = FORMAL_DATA / row["original_manifest"]
        archived = FORMAL_DATA / row["archived_manifest"]
        original_exists, archived_exists = original.parent.is_dir(), archived.parent.is_dir()
        if original_exists == archived_exists:
            raise ValueError(f"expected exactly one copy of {row['scene_id']}")
        manifest = original if original_exists else archived
        if sha256(manifest) != row["original_manifest_sha256"]:
            raise ValueError(f"old unit manifest changed: {row['scene_id']}")
    for row in rows:
        original = (FORMAL_DATA / row["original_manifest"]).parent
        archived = (FORMAL_DATA / row["archived_manifest"]).parent
        if original.is_dir():
            archived.parent.mkdir(parents=True, exist_ok=True)
            original.rename(archived)


def install(proof_path):
    with production_lock():
        if not STOP.is_file():
            raise ValueError("p2 installation requires the queue to remain drained")
        plan, state = sealed_plan(PLAN_DIR/"training.json")
        version = plan["generator_version"] + "-p2"
        parent = state["generator_patch"]
        if parent["version"] == version:
            validated = load_production_plan(PLAN_DIR/"training.json")["effective_generator_version"]
            archive_status()
            return validated
        tracked = load(RECORD)
        record = tracked["v2"]
        preregistered = record["p2_qualification_policy"]
        proof = load(proof_path)
        if not proof:
            raise ValueError("the isolated p2 qualification has not produced a report")
        summary = qualification_summary(proof_path, proof)
        if (
            proof["status"] != "passed" or proof["version"] != version
            or proof["generation_code"] != p2_generation_code() or proof["policy"] != preregistered
            or record["multilayer_placement_policy_p2"] != policy(TRAIN_FRAMES)
            or set(proof["smoke"]) != set(preregistered["smoke_training_scene_ids"])
            or set(proof["invariance"]) != set(preregistered["invariance_scene_ids"])
            or summary["smoke_passed"] != 12 or summary["invariance_passed"] != 4
        ):
            raise ValueError("p2 requires all 12 smoke scenes and four unchanged controls")
        specs = {}
        for mode in MODES:
            other, _ = sealed_plan(PLAN_DIR/(mode+".json"))
            if plan["generator_code"] != other["generator_code"]:
                raise ValueError("generator provenance diverges across sealed plans")
            specs.update((row["scene_id"], row) for row in other["scenes"])
        rows = archive_rows(state, specs)
        parent_code = parent_p1_code(plan, state, parent)
        current = p2_generation_code()
        receipt = dict(
            base_version=plan["generator_version"], patch_version=version,
            base_generator_code_sha256=code_hash(plan["generator_code"]), parent_patch=parent,
            parent_generator_code_sha256=code_hash(parent_code), generator_code_sha256=code_hash(current),
            changes=p2_changes(parent_code, current), dispatch_code=dispatch_code(p2=True),
            sealed_plan_sha256=state["scene_plan_sha256"], optical_frames=list(OPTICAL_FRAMES),
            qualification={key:summary[key] for key in ("path", "sha256")},
            foreground_presence_policy=record["foreground_presence_policy_p2"],
            multilayer_placement_policy=record["multilayer_placement_policy_p2"],
        )
        validate_p2(plan, state, receipt)
        ARCHIVE.mkdir(exist_ok=True)
        inventory = ARCHIVE / "migration-plan.json"
        if inventory.exists() and load(inventory) != rows:
            raise ValueError("existing archive inventory differs from the planned migration")
        inventory_sha = write_json(inventory, rows)
        move_archives(rows)
        receipt["archive"] = dict(path=str(inventory.relative_to(FORMAL_DATA)), sha256=inventory_sha, count=len(rows))
        patch_path = FORMAL_DATA / "generator-patch-p2.json"
        patch_ref = dict(path=patch_path.name, sha256=write_json(patch_path, receipt), version=version)
        state["generator_patch"] = patch_ref
        state["archived_p1_units"] = rows
        state["p2_archive_inventory"] = receipt["archive"]
        archived_ids = {row["scene_id"] for row in rows}
        state["units"] = [row for row in state["units"] if row["scene_id"] not in archived_ids]
        state["counts"].update(train_geometry=0, development=1, test=18)
        write_json(FORMAL_DATA/"manifest.json", state)
        record["generator_patch_p2"] = dict(receipt, status="installed", **patch_ref)
        record["p2_qualification"] = summary
        record["p2_invariance"] = dict(status="passed", scenes=summary["invariance"])
        record["p2_rebuild"] = dict(status="archived_awaiting_dispatch", archive=receipt["archive"],
                                    archived_units=rows, affected_test_pending=12)
        record["p2_dispatch"].update(status="installed_queue_drained")
        write_json(RECORD, tracked)
        archive_status()
        return version


def wait_for_proof(path, pid):
    while True:
        proof = load(path) if path.is_file() and path.stat().st_size else {}
        status = proof.get("status")
        if status == "passed":
            return
        try:
            alive = psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            alive = False
        if status == "failed" or not alive:
            with production_lock():
                record = load(RECORD)["v2"]
                if proof and proof["generation_code"] != p2_generation_code():
                    record["p2_previous_qualification"] = qualification_summary(path, proof)
                else:
                    if proof:
                        record["p2_qualification"] = qualification_summary(path, proof)
                    record["p2_dispatch"].update(status="paused_qualification_failed", stop_after_current=True)
                    record["p2_rebuild"]["status"] = "blocked_before_archival"
                    record["p2_finish_eta"] = dict(status="blocked_by_qualification", reason=status or "no final proof")
                all_records = load(RECORD)
                all_records["v2"] = record
                write_json(RECORD, all_records)
            raise RuntimeError("B p2 qualification failed or ended before a final report")
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--qualification-pid", type=int)
    parser.add_argument("--resume-with-unit-byte-budget", type=int)
    args = parser.parse_args()
    estimate = args.resume_with_unit_byte_budget
    if estimate is not None and estimate <= 0:
        parser.error("the measured per-unit budget must be positive")
    if args.qualification_pid is not None:
        wait_for_proof(args.proof, args.qualification_pid)
    print(install(args.proof), flush=True)
    if estimate is not None:
        paths = [PLAN_DIR/(partition+".json") for partition in PARTITIONS]
        dispatch(paths, estimate, optics=True, resume_p2=True)
