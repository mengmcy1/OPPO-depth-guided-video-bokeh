"""Failure accounting and ranked optical replacements for formal B production."""
from datetime import datetime, timezone
import re

from route3_v2data_B_batch import FORMAL_DATA, OPTICAL_TAGS, sha256, write_json
from route3_v2data_B_plan import load


ROLLING_ATTEMPTS = 30
FAILURE_LIMIT = 10
POLICY = dict(
    single_scene_failure="record_gap_and_continue",
    stop_conditions=["insufficient_disk_reserve", "unrecoverable_gpu_or_environment",
                     "admitted_data_corruption", "rolling_failure_rate"],
    rolling_attempts=ROLLING_ATTEMPTS, rolling_failure_limit=FAILURE_LIMIT,
    optical_start="sealed_geometry_queue_has_drained",
    replacement="next unassigned candidate for the same primary difficulty",
    principle_zh="数据准备效率优先。单个场景失败就记为缺口，然后跳过，不暂停类别或难度标签，也不卡住下一阶段。",
    systematic_stop_zh=["磁盘余量不足","GPU 或运行环境崩溃，且无法自动恢复",
                        "发现已入列的数据损坏","最近 30 次尝试里有 10 次以上失败"],
    geometry_slot_postcheck_zh="在训练光学全部完成后执行。结果只用于记录，不拦生产。",
)
SYSTEM_FAILURE = re.compile(
    r"out[ _]of[ _]memory|failed to allocate|CUDA[ _]error|OptiX error|segmentation fault|illegal memory access|"
    r"no space left on device|combined.*(?:RSS|VRAM).*cap|combined dispatch resource limit|"
    r"could not measure combined VRAM|GPU resource measurement failed|driver/library version mismatch|"
    r"nvidia-smi.*returned non-zero|cannot be opened|not a directory|permission denied|"
    r"error while loading shared libraries|no module named",
    flags=re.IGNORECASE,
)


class ProductionStop(RuntimeError):
    def __init__(self, condition, detail):
        super().__init__(detail)
        self.condition = condition


def check_admitted_inventory(archive):
    for row in (*archive["units"], *archive.get("optical_units", [])):
        path = FORMAL_DATA/row["manifest"]
        if not path.is_file() or sha256(path)!=row["manifest_sha256"]:
            raise ProductionStop("admitted_data_corruption",f"admitted manifest missing or changed: {row['manifest']}")
        unit = load(path)
        for item in unit["files"]:
            stored = FORMAL_DATA/item["path"] if "geometry_manifest_sha256" in unit else path.parent/item["path"]
            if not stored.is_file() or stored.stat().st_size!=item["bytes"]:
                raise ProductionStop("admitted_data_corruption",f"admitted product missing or wrong size: {stored}")


class ProductionHealth:
    def __init__(self, path=None):
        self.path = path or FORMAL_DATA/"p2-review/production-attempt-window.json"
        self.recent = (load(self.path) or {}).get("recent", [])
        self.hit = (load(self.path) or {}).get("halt")

    def seed_admitted_geometry(self, state):
        if self.path.exists():
            return
        geometry = [(FORMAL_DATA/row["manifest"],row["scene_id"]) for row in state["units"] if row["partition"]=="training"]
        for path,scene in sorted(geometry,key=lambda pair:pair[0].stat().st_mtime_ns)[-ROLLING_ATTEMPTS:]:
            self.add("geometry",scene,True)

    def add(self, phase, scene_id, passed, *, attempt="final", reason=None):
        row = dict(phase=phase, scene_id=scene_id, attempt=attempt, passed=passed, reason=reason,
                   recorded_at_utc=datetime.now(timezone.utc).isoformat())
        key = ("phase","scene_id","attempt")
        if any(all(prior[item]==row[item] for item in key) for prior in self.recent):
            return
        self.recent = [*self.recent, row][-ROLLING_ATTEMPTS:]
        failure_count = sum(not item["passed"] for item in self.recent)
        if failure_count>=FAILURE_LIMIT and self.hit is None:
            self.hit = dict(attempts=len(self.recent), failures=failure_count, triggered_by=row)
        write_json(self.path,dict(recent=self.recent,halt=self.hit))

    def check(self):
        if self.hit:
            raise ProductionStop("rolling_failure_rate",
                                 f"{self.hit['failures']} failures in the latest {self.hit['attempts']} attempts")


def system_failure(reason, log, *, exit_code=None):
    if exit_code is not None and (exit_code<0 or exit_code in {126,127,134,137,139}):
        return True
    if SYSTEM_FAILURE.search(reason or ""):
        return True
    if not log.is_file():
        return False
    with log.open("rb") as stream:
        stream.seek(max(0,log.stat().st_size-8192))
        return bool(SYSTEM_FAILURE.search(stream.read().decode(errors="replace")))


def ranked_optical_candidates(units):
    # The frozen B_batch.optical_selection scorer uses these same physical metrics and tie breaks.
    rankings = {tag:[] for tag in OPTICAL_TAGS}
    for unit in units:
        if unit["partition"]!="training":
            continue
        supervision = unit["scene_supervision"]
        boundary = sum(frame["boundary"] for frame in supervision)
        scores = {
            "foreground_focus": sum(frame["pixels"] for frame in supervision)-boundary,
            "background_focus_with_disocclusion": sum(
                lens["recoverable_rays"] for frame in supervision for lens in frame["K16"]),
            "thin_and_complex_edge_transmission": boundary if unit["category"] in
                {"thin_structure","curved_self_occlusion","complex_layout"} else 0,
            "multiple_layers_and_local_occlusion": boundary if unit["depth_bands"]>=3 else 0,
            "highlight_and_bokeh": sum(frame["saturated_boundary"] for frame in unit["radiance_audit"]),
        }
        tags = [tag for tag in OPTICAL_TAGS if scores[tag]>0]
        for tag in tags:
            rankings[tag].append(dict(scene_id=unit["scene_id"], primary_difficulty=tag,
                                      difficulties=tags, scores=scores))
    return {tag:sorted(rows,key=lambda row:(-row["scores"][tag],row["scene_id"])) for tag,rows in rankings.items()}


class OpticalQueue:
    def __init__(self, selection, rankings, path):
        self.path = path
        self.rankings = rankings
        self.assignments = (load(path) or {}).get("assignments", [[row] for row in selection])

    def pending(self, failed, admitted):
        reserved = {row["scene_id"] for slot in self.assignments for row in slot}
        for index, chain in enumerate(self.assignments):
            current = chain[-1]
            name = current["scene_id"]
            if name in admitted:
                continue
            if name in failed:
                tag = current["primary_difficulty"]
                replacement = next((row for row in self.rankings[tag] if row["scene_id"] not in reserved),None)
                if replacement is None:
                    continue
                chain.append(replacement)
                reserved.add(replacement["scene_id"])
                self.save()
                print("B optical supplement",tag,name,replacement["scene_id"],flush=True)
                current = replacement
            yield index,current

    def exhausted(self, failed):
        reserved = {row["scene_id"] for slot in self.assignments for row in slot}
        return [dict(slot=index+1,primary_difficulty=row["primary_difficulty"])
                for index,slot in enumerate(self.assignments) for row in [slot[-1]]
                if row["scene_id"] in failed and not any(
                    option["scene_id"] not in reserved for option in self.rankings[row["primary_difficulty"]])]

    def save(self):
        write_json(self.path,dict(assignments=self.assignments))

    def replacements(self):
        return [dict(slot=index+1, replaced=old["scene_id"], replacement=new["scene_id"],
                     primary_difficulty=new["primary_difficulty"])
                for index,chain in enumerate(self.assignments) for old,new in zip(chain,chain[1:])]
