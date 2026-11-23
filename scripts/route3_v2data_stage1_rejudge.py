"""Rejudge Stage 1 using saved EMA case metrics, after the original decision."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

from route3_v2data_B_batch import write_json
from route3_v2data_B_plan import load
from route3_v2data_stage1_evaluate import METRICS, adoption, summary

BRANCHES = ("original", "same-surface")
WEIGHTS = ("ema", "model")
FLOORS = ("zero_visible", "zero_combined")
RECORD = Path("03_实验记录/records/route3-v2data-stage1-2026-11-22.json")
PROGRESS = Path("03_实验记录/当前实验进度与方法论.md")
NOTE = "**新阶段 1 选支修正**"
REASON = ("旧汇总要求全部计划 case 都已评分。development-010 的 f001、f030、f060 近焦 case "
          "由真值判定无有效名义主体，使旧零下限 all_passed 恒为 False。修正后按有评分的 case 检查，"
          "另记不可评分的 case。")


def rejudge(folder):
    saved = [load(folder / branch / "ema.json") for branch in BRANCHES]
    updated = [dict(row, development=summary(row["development_rows"])) for row in saved]
    details = {}
    for branch, row in zip(BRANCHES, updated):
        dev = row["development"]
        legacy = summary(row["legacy_rows"])
        details[branch] = dict(
            zero_floors={split: {key: values[key] for key in (
                *FLOORS, "unscored_cases", "unscored_case_ids",
            )} for split, values in (("development", dev), ("legacy", legacy))},
            p2_scene_standard_deviation_db={metric: dev[metric]["scene_standard_deviation_db"]
                                            for metric in METRICS},
        )
    return dict(original_decision=adoption(*saved), corrected_decision=adoption(*updated),
                reason=REASON, ema_checks=details)


def optical_table(milestones):
    result = {}
    for step in (10_000, 20_000, 30_000):
        milestone = milestones[f"1b_step_{step}"]
        rows = []
        for branch in BRANCHES:
            for weight in WEIGHTS:
                row = milestone["by_branch"][branch][weight]
                values = {}
                for split in ("development", "legacy"):
                    source = row[split]
                    values[split] = {
                        metric: dict(open_db=source[metric]["psnr_db"],
                                     query_open_minus_closed_db=source["query_pair"][metric]["delta_db"])
                        for metric in METRICS
                    }
                rows.append(dict(branch=branch, weights=weight, **values,
                                 legacy_far_per_scene_db=row["legacy"]["far_recoverable"]["per_scene"]))
        result[str(step)] = rows
    return result


def local_records(checkout, milestones):
    def git(*args, capture_output=False):
        return subprocess.run(["git", *args], cwd=checkout, check=True,
                              capture_output=capture_output, text=True).stdout

    if git("status", "--porcelain", capture_output=True):
        raise RuntimeError(f"record worktree has local changes: {checkout}")
    git("fetch", "origin", "main")
    git("merge", "--ff-only", "origin/main")
    report = milestones["adoption_corrected"]
    tracked = load(checkout / RECORD)
    tracked.update(milestones)
    write_json(checkout / RECORD, tracked)
    corrected = report["corrected_decision"]
    checks = corrected["checks"]
    distances = corrected["same_surface_p95"]
    near, far = (corrected["p2_candidate_minus_original_db"][m] for m in METRICS)
    zero = report["ema_checks"]["same-surface"]["zero_floors"]["development"]
    floors = "、".join(f"{zero[k]['passed_cases']}/{zero[k]['scored_cases']}" for k in FLOORS)
    label = lambda passed: "通过" if passed else "未通过"
    note = (
        f"{NOTE}。原决定为 `{report['original_decision']['selected']}`，"
        f"修正后为 `{corrected['selected']}`。{report['reason']}"
        f"同表面台阶从 {distances['original']:.6f} 到 {distances['candidate']:.6f} m⁻¹"
        f"（{label(checks['same_surface_step_improved'])}）。"
        f"候选开发集零光圈两项为 {floors}（{label(checks['both_zero_floors'])}）。"
        f"P2 候选减原配方的近焦与远焦为 {near:+.3f} 和 {far:+.3f} dB"
        f"（{label(checks['both_p2_noninferior'])}）。[阶段 1 记录](records/{RECORD.name})\n"
    )
    progress = checkout / PROGRESS
    lines = progress.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(NOTE):
            lines[index] = note
            break
    else:
        index = next(i for i, line in enumerate(lines) if line.startswith("**新阶段 1 配对训练**"))
        lines[index + 1:index + 1] = ["\n", note]
    progress.write_text("".join(lines))
    git("diff", "--check")
    git("add", "--", str(RECORD), str(PROGRESS))
    git("diff", "--cached", "--check")
    if git("diff", "--cached", "--name-only", capture_output=True):
        git("commit", "-m", "记录阶段 1 零光圈下限选支修正",
            "-m", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    return git("rev-parse", "--short", "HEAD", capture_output=True).strip()


def finalize(run, checkout, b_data):
    milestone_path = run / "milestones.json"
    folder = run / "evaluation/1b/step-030000"
    print("Waiting for the original 1b follow-up to release its lock.", flush=True)
    with (run / "followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = load(run / "dispatch.json")
        milestones = load(milestone_path)
        if state.get("status") != "complete" or state.get("decision") != milestones.get("decision"):
            raise RuntimeError("The original final decision has not been registered.")
        correction = rejudge(folder)
        if correction["original_decision"] != milestones["decision"]:
            raise ValueError("The saved EMA summaries do not reproduce the registered decision.")
        path = folder / "adoption-corrected.json"
        report = load(path)
        if not report:
            manifest = load(b_data / "manifest.json")
            complete_f2 = sum(
                all((b_data / "derived-f2" / row["scene_id"] / f"f{f:03d}" / "manifest.json").is_file()
                    for f in (1, 30, 60)) for row in manifest["optical_units"]
            )
            legacy = milestones["1b_step_30000"]["legacy_1c_5k_ema"]
            report = dict(correction, evaluated_at_utc=state["finished_at_utc"],
                          recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                          evaluation_wall_seconds=milestones["1b_step_30000"]["runtime"]["wall_seconds"],
                          optical_counts=dict(admitted=manifest["counts"]["train_optical"], f2_complete=complete_f2),
                          legacy_1c_5k_ema=dict(query_pair=legacy["query_pair"],
                                               far_per_scene_db=legacy["summary"]["far_recoverable"]["per_scene"]),
                          checkpoints=optical_table(milestones))
            write_json(path, report)
        milestones["adoption_corrected"] = {k: v for k, v in report.items() if k != "checkpoints"}
        milestones["adoption_corrected"]["artifact"] = path.relative_to(run).as_posix()
        write_json(milestone_path, milestones)
        commit = local_records(checkout, milestones)
        print(f"Saved {path}. Local results commit {commit}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Inspect an existing checkpoint pair without writing results.")
    check.add_argument("folder", type=Path)
    finish = commands.add_parser("finalize", help="Wait for the original decision and record the correction locally.")
    finish.add_argument("--run", type=Path, required=True)
    finish.add_argument("--records-worktree", type=Path, required=True)
    finish.add_argument("--b-data", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        print(json.dumps(rejudge(args.folder), ensure_ascii=False, indent=2))
    else:
        finalize(args.run.resolve(), args.records_worktree.resolve(), args.b_data.resolve())
