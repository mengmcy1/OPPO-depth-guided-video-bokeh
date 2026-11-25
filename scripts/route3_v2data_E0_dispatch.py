"""Run the matched E0 arms in a persistent server process and index their results."""
import argparse
import os
import subprocess
import sys
import time

from route3_v2data_E_labels import ROOT,load,save_json
from route3_v2data_E0_data import CONFIG,RUN
from route3_v2data_E0_evaluate import FULL,LABELS
from route3_v2data_E0_train import COMPETITOR,choose,verify_registration
from src.ovg_e import FEATURE_MODES

RECORD=ROOT/"03_实验记录/records/route3-v2data-E0-2026-11-26.json"


def progress(phase,**details):
    save_json(RUN/"dispatch.json",dict(phase=phase,unix_seconds=time.time(),pid=os.getpid(),**details))


def command(script,tag,*arguments):
    prefix=RUN/"logs"/tag
    prefix.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(2):
        log=prefix.with_suffix(f".{time.time_ns()}.log")
        progress(tag,log=str(log))
        with log.open("w") as out:
            p=subprocess.run([sys.executable,"-u",f"scripts/route3_v2data_E0_{script}.py",
                              *map(str,arguments)],stdout=out,stderr=subprocess.STDOUT,cwd=ROOT)
        if p.returncode==0:
            return
        text=log.read_text()
        if attempt or ("OutOfMemoryError" not in text and "CUDA out of memory" not in text):
            raise RuntimeError(f"E0 phase {tag} stopped, read {log}")
        time.sleep(15)


def training(mode):
    return load(RUN/mode/"training.json")


def select(modes):
    all_rows=[dict(training(mode)["selected"],mode=mode) for mode in modes]
    chosen,eligible=choose(all_rows)
    return dict(mode=chosen["mode"],step=chosen["step"],phase=chosen["phase"],rod_eligible=eligible,
                holdout=chosen["holdout"],candidates=all_rows)


def measure_competitor():
    running={"3b":COMPETITOR/"progress.json"}
    for work in ROOT.parent.glob("route3-stage3c*"):
        for item in (work/"outputs").glob("*/progress.json"):
            running["3c"]=item
    return {name:load(path) for name,path in running.items() if path.is_file()}


def run():
    verify_registration(load(RUN/"data.json"))
    competition=RUN/"competitor-start.json"
    if not competition.is_file():
        save_json(competition,measure_competitor())
    for mode in FEATURE_MODES[:2]:
        command("train",f"train-{mode}",mode)
    first=select(FEATURE_MODES[:2])
    for mode in FEATURE_MODES[:2]:
        step=training(mode)["selected"]["step"]
        command("evaluate",f"depth-{mode}",mode,step,"depth")
        if mode!=first["mode"]:
            command("evaluate",f"E-only-{mode}",mode,step,"E-only")
    mode,step=first["mode"],first["step"]
    command("tears",f"tear-{mode}",mode,"--step",step)
    command("evaluate",f"full-{mode}",mode,step,"full")
    command("windows","cache-E-H2")
    command("train","train-E-H2","E-H2")
    h2step=training("E-H2")["selected"]["step"]
    command("evaluate","depth-E-H2","E-H2",h2step,"depth")
    final=select(FEATURE_MODES)
    if final["mode"]=="E-H2":
        command("tears","tear-E-H2","E-H2","--step",h2step)
        command("evaluate","full-E-H2","E-H2",h2step,"full")
    else:
        command("evaluate","E-only-E-H2","E-H2",h2step,"E-only")
    save_json(RUN/"selection.json",dict(first=first,final=final,paired_geometry_frozen=True))
    save_json(RUN/"competitor-end.json",measure_competitor())
    progress("awaiting-panel-review",selected=final["mode"],step=final["step"])


def optics_report(mode,step,kind):
    return load(RUN/mode/f"evaluation-{step:06d}-{kind}/report.json")


def compact_curve(curve):
    return [dict(step=r["step"],phase=r["phase"],holdout=r["holdout"]) for r in curve]


def focus_table(report):
    result=[]
    for row in report["by_case"]:
        one={k:row[k] for k in ("scene","frame","cohort","case","status","nominal_focus_m","GT_focus_m")}
        one["methods"]={k:m["focus_m"] for k,m in row["methods"].items()}
        result.append(one)
    return result


def directory_summary(result,observations):
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"
    lines=["# E0 深度与光学对照","",f"选定配置 {result['selection']['final']['mode']}。",""]
    for mode,row in result["modes"].items():
        lines.extend([f"## {mode}",""])
        t=row["training"]
        lines.extend([f"选中 {t['selected']['step']:,} 步 EMA，第 {t['selected']['phase']} 期。"
                      f"训练 {t['step_seconds']:.4f} 秒/步，峰值保留显存 {t['peak_reserved_gib']:.2f} GiB。",""])
        stats=row["depth"]["cohorts"]["development"]["errors"]
        lines.extend(["| 区域 | OVG p50 / p95 | E p50 / p95 |","|---|---:|---:|"])
        for zone in ("rod","gap","normal","boundary"):
            a,b=(stats[f"{zone}/{k}"] for k in ("OVG","decoded"))
            lines.append(f"| {zone} | {fmt(a['p50'])} / {fmt(a['p95'])} | {fmt(b['p50'])} / {fmt(b['p95'])} |")
        for group,block in row["optics"]["summary"].items():
            lines.extend(["",f"### {group}","",
                "| 方法 | 近焦 | GT−F2 | 远焦 | 边界 | 全图 | 零光圈可见 / 完整 | T≥0.1 正向 SSE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"])
            for arm,measures in block["methods"].items():
                cells=[fmt(measures[k]["psnr_db"]) for k in
                       ("near_in_focus","near_gt_minus_f2","far_recoverable","boundary_2px","full_image")]
                cells.append(" / ".join(f'{fmt(measures[k]["psnr_db"])}（{measures[k]["passed_cases"]}/{measures[k]["scored_cases"]}）'
                                        for k in ("zero_visible","zero_combined")))
                cells.append(fmt(measures["transmission"]["positive_sse_bins"].get("T_ge_0.1")))
                lines.append("| "+" | ".join([LABELS[arm],*cells])+" |")
        lines.append("")
    lines.extend(["## 观察清单","",*observations,""])
    return "\n".join(lines)


def collect():
    selection=load(RUN/"selection.json")
    observation=RUN/"观察清单.md"
    if not observation.is_file():
        raise FileNotFoundError(f"Review the fixed depth and optical panels first: {observation}")
    full_modes={selection[k]["mode"] for k in ("first","final")}
    modes={}
    for mode in FEATURE_MODES:
        t=training(mode)
        step=t["selected"]["step"]
        depth=optics_report(mode,step,"depth")
        kind="full" if mode in full_modes else "E-only"
        optics=optics_report(mode,step,kind)
        t["curve"]=compact_curve(t["curve"])
        tear_path=RUN/mode/f"tear-{step:06d}/training.json"
        if kind=="full" and set(optics["summary"]["development"]["methods"])!=set(FULL):
            raise ValueError(f"The complete comparison is incomplete for {mode}")
        modes[mode]=dict(training=t,depth=depth,optics={k:optics[k] for k in
                         ("summary","skipped_legacy","tear_audit","compatibility","wall_seconds")},
                         focus=focus_table(optics),panels=[p["panel"] for p in optics["by_case"] if p.get("panel")],
                         tear=load(tear_path) if tear_path.is_file() else None)
    begin,end=(load(RUN/k) for k in ("competitor-start.json","competitor-end.json"))
    shared={}
    for name in set(begin)&set(end):
        steps=end[name]["step"]-begin[name]["step"]
        seconds=end[name]["seconds"]-begin[name]["seconds"]
        shared[name]=dict(before=begin[name],after=end[name],steps=steps,seconds=seconds,
                          concurrent_wall_per_step=seconds/steps if steps else None)
    result=dict(selection=selection,modes=modes,h2_window_cache=load(RUN/"window-cache.json"),
                shared_resources=dict(begin=begin,end=end,overlap=shared),
                test_cases_used=0,run=RUN.name,observation=observation.read_text().splitlines())
    save_json(RECORD,result)
    cfg=load(CONFIG)
    cfg["training"]["phase2_actual_term_scales"]={mode:row["training"]["phase2"] for mode,row in modes.items()}
    save_json(CONFIG,cfg)
    (RUN/"汇总.md").write_text(directory_summary(result,result["observation"]))
    progress("complete",selected=selection["final"]["mode"],record=str(RECORD))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("run","collect"))
    a=p.parse_args()
    if a.phase=="collect":
        collect()
    else:
        try:
            run()
        except Exception as error:
            progress("halted",reason=str(error))
            raise
