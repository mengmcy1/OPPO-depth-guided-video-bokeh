"""Evaluate and collect the version-three E-H2 comparison on development scenes."""
import argparse
from collections import defaultdict
from html import escape
from pathlib import Path

import numpy as np
import torch

import route3_v2data_E0_evaluate as depth_eval
import route3_v2data_E0_rerender as replay
import route3_v2data_E0_rerender_report as panels
import route3_v2data_E0_v2 as v2
import route3_v2data_module2_infer as ref
from route3_v2data_B_batch import sha256
from route3_v2data_E_labels import load, save_json
from route3_v2data_E0_tears import load_model
from route3_v2data_E0_train import gpu_gate, predict, single_back_target
from route3_v2data_E0_v3 import BASES, CONFIG, MODE, ORIGINAL_STEPS, RESULT, RUN, dispatch, retrain, tear_root
from route3_v2data_E0_v3_metrics import Diagnostics
from src.ovg_edge_statistics import EdgeStatistics


EVAL=RUN/"evaluation"
FOCI=("P","G")
FIXED_KEYS=("V0_pred","v2a","E_learned","E2_4k_tear","E2_2k","E2_2k_tear",
            "E3","E3_tear","E3_gt","V0_GT")
TITLES=dict(zip(FIXED_KEYS,("V0 预测","OVG + v2a","E1 2k + 撕裂","E2 4k + 撕裂","E2 2k",
                           "E2 2k + 撕裂","E3","E3 + 撕裂","E3 + GT 撕裂","V0 GT")))
ARCHIVE_ARMS={"V0_pred":"V0_pred","v2a":"v2a","E_learned":"E_learned","E2_4k_tear":"E_B_learned"}


def steps():
    hist=load(RUN/"retrospective.json")
    own=load(RUN/MODE/"training.json")
    return hist,own["selected"]["step"]


def groups(retrospective, chosen):
    result={
        "E1_2k":("v1",2000,{a:a for a in ("V0_pred","v2a","E_learned","V0_GT")}),
        "E2_4k":("v2",4000,{"E2_4k_tear":"E_B_learned"}),
        "E2_2k":("v2",2000,{"E2_2k":"E_B","E2_2k_tear":"E_B_learned"}),
        "E3":("v3",chosen,{"E3":"E_B","E3_tear":"E_B_learned","E3_gt":"E_B_gt"}),
    }
    for version,item in retrospective.items():
        step=item["selected"]["step"]
        if step==item["old_step"] or (version=="v2" and step==2000):
            continue
        alias=f"E{version[-1]}_retro"
        result[alias]=(version,step,{alias:"E_B",alias+"_tear":"E_B_learned"})
    return result


def unique_arms(arm_groups):
    return (*FIXED_KEYS,*(a for _,_,mapping in arm_groups.values() for a in mapping if a not in FIXED_KEYS))


def merge_phases(phase_rows, arm_groups, final_arms):
    merged=[]
    for frame_rows in zip(*phase_rows.values(),strict=True):
        baseline=frame_rows[0]["rows"]["P"][0]
        output=dict(relative=f"{baseline['scene']}/f{baseline['frame']:03d}.json",rows={})
        for tag in FOCI:
            output["rows"][tag]=[]
            for paired in zip(*(block["rows"][tag] for block in frame_rows),strict=True):
                first=paired[0]
                key=panels.case_key(first),first["status"]
                if any((panels.case_key(row),row["status"])!=key for row in paired):
                    raise ValueError(f"Different scenes or masks between E3 optical arms: {key}")
                methods={}
                if first["status"]=="evaluated":
                    for row,(_,_,mapping) in zip(paired,arm_groups.values(),strict=True):
                        methods.update({new:row["methods"][old] for new,old in mapping.items()})
                    methods={a:methods[a] for a in final_arms}
                output["rows"][tag].append(dict(first,methods=methods))
        merged.append(output)
    if len(merged)!=54:
        raise ValueError("Expected 54 sealed development/continuity frames")
    return merged


def replay_check(parts, arms):
    original=load(v2.RESULT)["optical"]["P"]
    checks={}
    summary=replay.core_summary(parts,arms,"P")
    for cohort in ("development","legacy","thin_structure","dev008"):
        rows=[r for p in parts for r in p["rows"]["P"] if belongs(r,cohort)]
        f2=v2.far_f2(rows,arms=arms,new_arms=())["pooled"]
        checks[cohort]={}
        for new,old in ARCHIVE_ARMS.items():
            actual=summary[cohort]["methods"][new]
            expected=original[cohort]["methods"][old]
            measurements={k:(actual[k]["psnr_db"],expected[k]["psnr_db"]) for k in ref.REGIONS}
            measurements["far_F2"]=(f2[new]["psnr_db"],original[cohort]["far_f2"]["pooled"][old]["psnr_db"])
            checks[cohort][new]={k:dict(observed=a,registered=b,delta_db=a-b if a is not None and b is not None else None)
                                  for k,(a,b) in measurements.items()}
            if any((a is None)!=(b is None) or (a is not None and abs(a-b)>.01) for a,b in measurements.values()):
                save_json(EVAL/"P-self-check.json",checks)
                raise ValueError(f"E3 archived P comparison exceeded .01 dB in {cohort}/{new}: {checks[cohort][new]}")
    save_json(EVAL/"P-self-check.json",checks)
    return checks


def belongs(row, cohort):
    if cohort in ("development","legacy"):
        return row["cohort"]==cohort
    slug=ref.title(row["scene"])
    return slug=="dev008" if cohort=="dev008" else slug in ref.THIN_SCENES


def optical_summary(parts, arms):
    report={}
    new=tuple(a for a in arms if a.startswith("E3"))
    for tag in FOCI:
        table=replay.core_summary(parts,arms,tag)
        if (table["development"]["scored"],table["legacy"]["scored"])!=(87,18):
            raise ValueError("E3 development or legacy case coverage mismatch")
        for cohort,scope in table.items():
            rows=[r for p in parts for r in p["rows"][tag] if belongs(r,cohort)]
            scope["far_f2"]=v2.far_f2(rows,arms=arms,new_arms=new)
            scope["paired_vs_E1"]={a:ref.compare(scope["per_scene"],"E_learned",a,scope["methods"],
                                                  include_absolute_near=True) for a in new}
        report[tag]=table
    return report


def subject_info(frame, case, K, native, target, out):
    mask=ref.near_subject(native["gt"],native["valid"],K,case["focus_m"])
    def field(data):
        return data[0,0].cpu().numpy()
    def median(data):
        return float(np.median(data[mask])) if mask.any() else None
    zone,source,selected=(field(x) for x in (target["loss_zone"],target["mv_source"],out["selected_back"]))
    return dict(frame=frame,case=case["id"],pixels=int(mask.sum()),
                regions={z:int((mask&(zone==i)).sum()) for i,z in enumerate(v2.training.GROUPS)},
                selected_front=int((mask&~selected).sum()),selected_back=int((mask&selected).sum()),
                rear_base_source={s:int((mask&(source==i)).sum()) for i,s in enumerate(("rho2","DS","tri"))},
                raw_rear_log_inverse_p50=median(field(out["raw_back"])),
                raw_rear_inverse_p50=median(field(out["raw_back"].exp())),
                clipped_rear_inverse_p50=median(field(out["means"][:,1:2].exp())),
                single_rear_target_log_inverse_p50=median(field(single_back_target(out,target))),
                OVG_rho1_p50=median(field(target["rho1"])),decoded_rho_p50=median(field(out["decoded"])),
                GT_center_rho_p50=median(field(target["center"])),
                actual_focus_m=ref.p2_focus(1/field(out["decoded"]),mask))


@torch.inference_mode()
def evaluate_depth(alias, network, inputs, geometry):
    dest=EVAL/"depth"/alias/"report.json"
    if dest.is_file():
        return load(dest)
    stats=defaultdict(Diagnostics)
    edges=defaultdict(EdgeStatistics)
    data,coc=defaultdict(list),defaultdict(list)
    planned_subjects=[]
    for cohort,name,frame,unit,_,cases in ref.frames():
        gpu_gate()
        item=("B",name,ref.ovg.TRAIN_FRAMES.index(frame))
        x,labels=ref.sample(item)
        native=ref.ovg.b_edge_reference(item,labels)
        if unit:
            target=inputs.native(name,frame)
        else:
            forecast=geometry(x,protect_residual=True,render_heads=False,return_features=True)
            target=inputs.observations(name,frame,x,forecast)
            legacy=depth_eval.legacy_targets(labels,native)
            depth2=labels["depth2"][:,None]
            dual=legacy["visible"]&labels["valid_dual"][:,None]&torch.isfinite(depth2)&(depth2>0)
            target.update(legacy,valid_dual=dual,back=torch.where(dual,depth2.clamp_min(1e-8).reciprocal(),0.))
        out=predict(network,target)
        row=dict(scene=name,cohort=cohort)
        for scope in (cohort,"thin_structure","dev008"):
            if belongs(row,scope):
                stats[scope].add(out,target)
        for method,d in depth_eval.source_depth(target,out).items():
            key=cohort,method
            depth_eval.add_guard_stats(data[key],edges[key],coc[key],method,name,d,labels,native,x,unit,cases)
        if ref.title(name)=="dev003":
            case=next(c for c in cases if c["focus_index"]==1)
            planned_subjects.append(subject_info(frame,case,x["K"][0].cpu().numpy(),native,target,out))
        print("E3_DEPTH",alias,ref.title(name),frame,flush=True)
    guards={}
    limits=load(CONFIG)["geometry_limits"]
    for cohort in ("development","legacy"):
        a,b=(depth_eval.guard_report(data[cohort,k],coc[cohort,k],edges[cohort,k]) for k in ("OVG","E"))
        relative={k:b[k]/a[k]-1 for k in depth_eval.GUARDS}
        guards[cohort]=dict(OVG=a,E=b,relative=relative,limits=limits,
                            passed={k:v<=limits[k] for k,v in relative.items()})
    result=dict(cohorts={k:v.summary() for k,v in stats.items()},guards=guards,dev003_subject=planned_subjects)
    save_json(dest,result)
    return result


def write_panels(parts, arm_groups, arms):
    panels.RUN=EVAL
    titles=dict(TITLES)
    for name,(_,step,mapping) in arm_groups.items():
        if name.endswith("_retro"):
            prefix=name.split("_")[0]+f" {step//1000}k 回溯"
            titles.update({a:prefix+(" + 撕裂" if a.endswith("_tear") else "") for a in mapping})
    panels.TITLES=tuple(titles[a] for a in arms)
    (EVAL/"figures").mkdir(parents=True,exist_ok=True)
    paths=[]
    for entry in parts:
        row=entry["rows"]["P"][0]
        slug=ref.title(row["scene"])
        if row["frame"]!=30 or slug not in replay.HIGHLIGHT:
            continue
        relative=Path(entry["relative"]).with_suffix(".npz")
        images={}
        for name,(_,_,mapping) in arm_groups.items():
            lookup={v:k for k,v in mapping.items()}
            with np.load(EVAL/name/"frames"/relative) as contents:
                for key in contents:
                    keys=key.split("/")
                    if keys[0] in FOCI:
                        alias=lookup.get(keys[2])
                        if alias is None:
                            continue
                        keys[2]=alias
                    elif keys[0] in ("tear","depth"):
                        alias=lookup.get(keys[1])
                        if alias is None:
                            continue
                        keys[1]=alias
                    images["/".join(keys)]=contents[key]
        for tag in FOCI:
            for r in entry["rows"][tag]:
                if r["status"]!="evaluated":
                    continue
                i=r["focus_index"]
                if i==1:
                    methods={a:dict(m,optical=dict(m["optical"],far_recoverable=m["far_off"]))
                             for a,m in r["methods"].items()}
                    r=dict(r,methods=methods)
                    with_f2=dict(images,**{f"{tag}/{i}/{a}":images[f"{tag}/{i}/{a}/F2"] for a in arms})
                else:
                    with_f2=images
                paths.append(panels.optical_panel(slug,tag,i,r,with_f2,arms,far_f2=i==1))
    if len(paths)!=38:
        raise ValueError(f"Expected 38 E3 panels, received {len(paths)}")
    return paths,titles


@torch.inference_mode()
def evaluate(arm_groups):
    v2.training.gpu_setup(dict(training=dict(max_process_gpu_gib=replay.GPU_GIB)))
    gpu_gate(replay.GPU_GIB)
    replay.RUN=EVAL
    geometry=v2.geometry(load(CONFIG)["geometry"])
    inputs=depth_eval.EInputs(MODE)
    arms=unique_arms(arm_groups)
    phases={}
    depth={}
    checks=None
    for name,(version,step,mapping) in arm_groups.items():
        dispatch("evaluating-optics",group=name,version=version,step=step)
        if name=="E1_2k":
            bundle=v2.arm_bundle(geometry,inputs,BASES[version],step,baseline=True)
        else:
            model,_=load_model(MODE,step,run=BASES[version])
            tear=replay.load_tear(tear_root(version,step)/"selected.pt")
            bundle=(geometry,None,inputs,model,tear)
        phases[name]=replay.phase(EVAL/name,bundle,tuple(mapping.values()),FOCI,render_options=v2.PHASE_OPTIONS)
        if name=="E2_4k":
            initial={k:arm_groups[k] for k in phases}
            before=tuple(a for _,_,m in initial.values() for a in m)
            checks=replay_check(merge_phases(phases,initial,before),before)
        dispatch("evaluating-depth",group=name,version=version,step=step)
        depth[name]=evaluate_depth(name,bundle[3],inputs,geometry)
        del bundle
        torch.cuda.empty_cache()
    parts=merge_phases(phases,arm_groups,arms)
    optical=optical_summary(parts,arms)
    paths,titles=write_panels(parts,arm_groups,arms)
    return dict(optical=optical,depth=depth,panels=paths,titles=titles,arm_order=list(arms),P_self_check=checks,
                dev003_subject={k:v["dev003_subject"] for k,v in depth.items()})


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def text_summary(report):
    lines=["# E0 第三版方案 A", "", "## 怎么读", "",
           "留出表每行是一版的一个 EMA 检查点。杆和后分量误差越低越好，均衡 CoC 采用 px 且越低越好。",
           "光学表每行是同一几何和撕裂头组合，PSNR 单位 dB 且越高越好。近焦按各方法对焦的 P，远焦主要读 F2 的 P 和 G。", "",
           "## 留出集重新选点", "", "| 版本 | 原选点 | 新规则选点 | 通过杆门槛 | 均衡 CoC px |",
           "|---|---:|---:|---|---:|"]
    for version,r in report["holdout"].items():
        chosen=r["selected"]
        lines.append(f"| {version} | {r['old_step'] or '—'} | {chosen['step']} | "
                     f"{'是' if r['passed_rod_gate'] else '否，回退'} | {fmt(chosen['holdout']['coc_balanced_px'])} |")
    lines.extend(["","| 版本 | 步数 | 均衡 CoC px | 杆 p50 m⁻¹ | OVG 杆 p50 m⁻¹ | NLL | 相邻分量切换 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for version,r in report["holdout"].items():
        for p in r["curve"]:
            h=p["holdout"]
            lines.append(f"| {version} | {p['step']} | {fmt(h['coc_balanced_px'])} | "
                         f"{fmt(h['errors']['rod/decoded']['p50'],5)} | {fmt(h['errors']['rod/OVG']['p50'],5)} | "
                         f"{fmt(h['nll'])} | {h['adjacent_switch']['fraction']:.2%} |")
    lines.extend(["", "## 选中检查点的后分量诊断", "",
                  "每行是某版选中点在一个留出区域的结果。误差和峰宽用对数逆深度，饱和列给 |tanh| > 0.95 的像素比。", "",
                  "| 版本 | 区域 | 单层后分量误差 p50／p95 | 双层后分量误差 p50／p95 | 前／后峰宽 p50 | ρ₂／多视角饱和率 |",
                  "|---|---|---:|---:|---:|---:|"])
    def pair(entry):
        return f"{fmt(entry['p50'])}／{fmt(entry['p95'])}"
    def rate(value):
        return "—" if value is None else f"{value:.2%}"
    for version,r in report["holdout"].items():
        h=r["selected"]["holdout"]
        comp=h["component"]
        for zone in v2.training.GROUPS:
            a,b=(h["saturation"][f"{zone}/{source}"]["fraction"] for source in ("rho2","multiview"))
            lines.append(f"| {version} | {zone} | {pair(comp[f'{zone}/rear.single'])} | "
                         f"{pair(comp[f'{zone}/rear.dual'])} | {fmt(comp[f'{zone}/front_scale']['p50'])}／"
                         f"{fmt(comp[f'{zone}/back_scale']['p50'])} | {rate(a)}／{rate(b)} |")
    lines.extend(["", "## 开发集深度与护栏", "", "分区误差是 m⁻¹ 的绝对误差 p50／p95，越低越好。护栏各列给相对冻结 OVG 的变化率。", "",
                  "| 检查点 | 杆 | 缝 | 正常 | 边界 | 逆深度 RMSE | 同表面台阶 | 近焦 CoC | 跨断层误差 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for alias,data in report["depth"].items():
        h=data["cohorts"]["development"]["errors"]
        rg=[pair(h[zone+"/decoded"]) for zone in ("rod","gap","normal_loss_zone","boundary")]
        guard=data["guards"]["development"]
        g=[f"{guard['relative'][k]:+.2%}（{'通过' if guard['passed'][k] else '超限'}）" for k in depth_eval.GUARDS]
        lines.append("| "+" | ".join([alias,*rg,*g])+" |")
    for group in ("development","legacy","thin_structure","dev008"):
        p,g=(report["optical"][t][group] for t in FOCI)
        lines.extend(["",f"## {group} 光学", "", "括号中为相对 V0_pred 的逐场景配对中位数，单位 dB。", "",
                      "| 档 | P 近焦 | P 远焦 F2 | G 远焦 F2 | P／G 含替身待查 |", "|---|---:|---:|---:|---:|"])
        for arm in report["arm_order"]:
            a,b=(s["methods"][arm] for s in (p,g))
            off=[s["far_f2"]["pooled"][arm]["psnr_db"] for s in (p,g)]
            md=[np.median(list(s["far_f2"]["paired"][arm]["per_scene_db"].values())).item()
                if arm!="V0_pred" else 0. for s in (p,g)]
            lines.append(f"| {report['titles'][arm]} | {fmt(a['near_in_focus']['psnr_db'])} | "
                         f"{fmt(off[0])} ({md[0]:+.3f}) | {fmt(off[1])} ({md[1]:+.3f}) | {fmt(a['far_recoverable']['psnr_db'])}／"
                         f"{fmt(b['far_recoverable']['psnr_db'])} |")
    lines.extend(["", "## dev003 远焦主体", "", "区域、分量和锚点列给像素数。后分量给原始逆深度与截断后逆深度的中位数，单位 m⁻¹。对焦距离单位 m。", "",
                  "| 检查点 | 帧 | 杆／缝／正常／边界 | 选前／后 | ρ₂／DS／三角化 | 后分量原始／截断 | 实际焦距 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for alias,rows in report["dev003_subject"].items():
        for r in rows:
            region="／".join(map(str,r["regions"].values()))
            source="／".join(map(str,r["rear_base_source"].values()))
            lines.append(f"| {alias} | {r['frame']} | {region} | {r['selected_front']}／{r['selected_back']} | "
                         f"{source} | {fmt(r['raw_rear_inverse_p50'])}／{fmt(r['clipped_rear_inverse_p50'])} | "
                         f"{fmt(r['actual_focus_m'])} |")
    return "\n".join(lines)+"\n"


def collect():
    cached=load(EVAL/"full-summary.json")
    retro,step=steps()
    fit=load(RUN/MODE/"training.json")
    hist=dict(retro,v3=dict(curve=fit["curve"],selected=fit["selected"],passed_rod_gate=fit["rod_baseline_eligible"],
                          old_step=None,missing_steps=[]))
    arm_groups=groups(retro,step)
    heads={name:load(tear_root(v,s)/"training.json") for name,(v,s,_) in arm_groups.items()}
    training={k:v for k,v in fit.items() if k not in ("curve","selected")}
    report=dict(cached,holdout=hist,training=training,config_file=CONFIG.name,test_cases_used=0,
                checkpoint_step=step,checkpoint_sha256=sha256(RUN/MODE/f"step-{step:06d}.pt"),
                tear_heads={name:{k:r[k] for k in ("config","selected","wall_seconds")} for name,r in heads.items()},
                far_primary="F2-only",substitute="pending validation")
    save_json(RESULT,report)
    (EVAL/"汇总.md").write_text(text_summary(report))
    gallery='<!doctype html><meta charset="utf-8"><title>E0 第三版</title><h1>E0 第三版</h1><p><a href="汇总.md">汇总</a></p>'
    (EVAL/"index.html").write_text(gallery+"\n".join(
        f'<figure><a href="{escape(p)}"><img src="{escape(p)}" width="1080" loading="lazy"></a>'
        f'<figcaption>{escape(Path(p).stem)}</figcaption></figure>' for p in report["panels"]))
    dispatch("awaiting-publication",selected=step,panels=len(report["panels"]))


def follow():
    retro,step=steps()
    arm_groups=groups(retro,step)
    for version,s,_ in arm_groups.values():
        if s==ORIGINAL_STEPS.get(version):
            continue
        dispatch("training-tear-head",version=version,step=s)
        retrain(version,s)
    path=EVAL/"full-summary.json"
    if not path.is_file():
        save_json(path,evaluate(arm_groups))
    collect()


def publish():
    report=load(RESULT)
    if len(report["panels"])!=38 or any(not (EVAL/p).is_file() for p in report["panels"]):
        raise ValueError("Cannot publish an incomplete E3 panel set")
    # Sync first, keeping the generated experiment record outside the rebase.
    v2.git("fetch","origin","main")
    v2.git("rebase","origin/main")
    progress=v2.PROGRESS.read_text()
    anchor=next(s for s in progress.splitlines() if s.startswith("**E0 第三版登记"))
    tag="**E0 第三版结果（方案 A）**"
    if tag not in progress:
        p,g=(report["optical"][f]["development"] for f in FOCI)
        score=p["methods"]["E3_tear"]["near_in_focus"]["psnr_db"]
        far=[s["far_f2"]["pooled"]["E3_tear"]["psnr_db"] for s in (p,g)]
        picked="／".join(f"{h['selected']['step']//1000}k" for h in report["holdout"].values())
        line=(f"{tag}。统一留出选点第一／二／三版为 {picked}。第三版配重训撕裂头的开发集近焦 P "
              f"为 {score:.3f} dB，远焦仅 F2 的 P／G 为 {far[0]:.3f}／{far[1]:.3f} dB。"
              f"含 GT 替身的远焦列待查。 [机器记录](records/{RESULT.name})")
        v2.PROGRESS.write_text(progress.replace(anchor,anchor+"\n\n"+line,1))
    v2.publish((RESULT,v2.PROGRESS),"记录 E0 第三版、三版统一选点及配对评价")
    dispatch("complete",commit=v2.git("rev-parse","HEAD",capture=True).stdout.strip(),panels=38)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("follow","collect","publish"))
    command=parser.parse_args().command
    try:
        {"follow":follow,"collect":collect,"publish":publish}[command]()
    except Exception as error:
        dispatch("halted",reason=str(error))
        raise
