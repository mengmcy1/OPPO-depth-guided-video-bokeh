"""Collect hidden-colour results with the same region pixels in each pair."""
import argparse
from collections import defaultdict
from html import escape
from pathlib import Path
import time

import numpy as np

import route3_v2data_module2_infer as ref
from route3_v2data_B_batch import sha256
from route3_v2data_E0_rerender import STAGE3C
from route3_v2data_E_labels import save_json
from route3_v2data_hidden_color_purity import OLD_ARM, raw_s3_source_coverage, target_metadata, zero_file
from route3_v2data_hidden_color_sources import OUT, ROOT, read, read_witnesses, witness_path
from route3_v2data_hidden_color_zero import (COMBINED_FLOOR, FAILURE_CASE, START_TOLERANCE, VISIBLE_FLOOR,
                                            VISIBLE_TOLERANCE)
from route3_v2data_stage3b import QUERY

RECORD=ROOT/"03_实验记录/records/route3-v2data-hidden-color-purity-2026-11-26.json"
FAMILIES=("V0_pred","V0_GT","E2k","Query")
TITLES={"V0_pred":"OVG 预测 V0","V0_GT":"GT 中心深度 V0","E2k":"E-H2@2k + 重训撕裂头","Query":"实际 Query"}


def sum_dicts(values):
    out=defaultdict(float)
    for value in values:
        for k,v in value.items():
            out[k]+=v
    return dict(out)


def items(row,family,focus):
    r=row["render"][focus][family]
    return r["methods"] if family=="Query" else r


def combine(metrics, fields=("patch_lpips","direct_lpips_to_baseline")):
    pooled=ref.pooled([r["score"] for r in metrics])
    for field in fields:
        values=[r for r in metrics if r.get(field) is not None and r["score"]["pixels"]]
        if values:
            weight=sum(r["score"]["pixels"] for r in values)
            pooled[field]=sum(r[field]*r["score"]["pixels"] for r in values)/weight
    return pooled


def pair(metrics,base,k):
    # Keep frame support paired, including the restricted f030 S3 table.
    pairs=defaultdict(lambda: ([],[]))
    for r,a,b in zip(metrics,base,k):
        if r["score"]["pixels"]!=a["score"]["pixels"]:
            raise ValueError("The pair uses unequal region support")
        pairs[b][0].append(r)
        pairs[b][1].append(a)
    deltas=[]
    lpips=[]
    for a,b in pairs.values():
        current,old=combine(a),combine(b)
        if current["pixels"]:
            deltas.append(current["psnr_db"]-old["psnr_db"])
            if "patch_lpips" in current and "patch_lpips" in old:
                lpips.append(current["patch_lpips"]-old["patch_lpips"])
    def describe(values):
        x=np.asarray(values,dtype=float)
        return dict(scenes=len(x),median=float(np.median(x)) if len(x) else None,
                    std=float(x.std(ddof=0)) if len(x) else None,
                    positive=int((x>1e-9).sum()),negative=int((x< -1e-9).sum()),
                    ties=int((abs(x)<=1e-9).sum()))
    current,reference=combine(metrics),combine(base)
    return dict(PSNR=dict(describe(deltas),pooled_db=current["psnr_db"]-reference["psnr_db"] if current["pixels"] else None),
                LPIPS=dict(describe(lpips),pooled_delta=current["patch_lpips"]-reference["patch_lpips"]
                           if "patch_lpips" in current and "patch_lpips" in reference else None))


def aggregate_strata(rows,total):
    selected=[r["stratified"] for r in rows if "stratified" in r]
    if not selected:
        return None
    def pooled(all_counts):
        r=ref.pooled(all_counts)
        return dict(r,sse_share=r["sse"]/total if total>0 else None)
    return dict(
        purity={key:pooled([x["bins"][key] for x in selected]) for key in selected[0]["bins"]},
        threshold={key:{side:pooled([x["thresholds"][key][side] for x in selected])
                        for side in ("recoverable","unrecoverable")}
                   for key in selected[0]["thresholds"]})


def optical_table(rows,focus,family):
    methods=sorted(set().union(*(items(r,family,focus).keys() for r in rows)))
    table={}
    reference="Query" if family=="Query" else "S0"
    for method in methods:
        subset=[r for r in rows if method in items(r,family,focus)]
        group=[items(r,family,focus)[method] for r in subset]
        off=[items(r,family,focus)["F2"] for r in subset]
        base=[items(r,family,focus)[reference] for r in subset]
        y=combine(group)
        names=[r["scene"] for r in subset]
        table[method]=dict(pooled=y,cases=len(subset),vs_F2=pair(group,off,names),
                           vs_original=pair(group,base,names),
                           stratification=aggregate_strata(group,y["sse"]))
        if family=="Query" and method.startswith("QPred@"):
            same_cut=method.replace("QPred","QGT",1)
            gt=[items(r,family,focus)[same_cut] for r in subset]
            table[method]["vs_GT_purity"]=pair(group,gt,names)
        if method=="S3":
            table[method]["reference_same_f030"]={name:combine([items(r,family,focus)[name] for r in subset])
                                                   for name in methods if name!="S3"}
    return table


def source_query(rows):
    items=[r["render"]["P"]["Query"]["source"] for r in rows]
    available=[x for x in items if x["retained_heads"]]
    n=sum(x["retained_heads"] for x in available)
    if not n:
        return dict(retained_heads=0)
    bins=np.sum([x["dominant_gt"]["bins"] for x in available],axis=0)
    contributing=np.sum([x["contributing_gt"]["bins"] for x in available],axis=0)
    unknown=sum(x["contributing_gt"]["unknown"] for x in available)
    count=int(contributing.sum())+unknown
    effective=np.sum([x["weighted_gt"] for x in available],axis=0)
    def peel_errors(field):
        result={}
        for k in ("0","1","2","3"):
            d=sum_dicts(x[field][k] for x in available)
            samples=int(d["samples"])
            result[k]=dict(samples=samples,linear_L1=d["absolute"]/(3*samples) if samples else None,
                           linear_PSNR=-10*np.log10(max(1e-12,d["squared"]/(3*samples))) if samples else None)
        return result
    admitted={t:sum_dicts(x[t]["admitted"] for x in available) for t in ("QGT","QPred")}
    return dict(retained_heads=n,dominant_bins=bins.tolist(),dominant_fractions=(bins/n).tolist(),
                dominant_unknown=int(n-bins.sum()),soft_rgb_weight_fractions=(effective/n).tolist(),
                unknown_rgb_weight=sum(x["unknown_weight"] for x in available)/n,
                contributing_samples=count,contributing_bins=contributing.tolist(),
                contributing_fractions=(contributing/count).tolist(),contributing_unknown=unknown,
                matched_f030_peel_count=sum(x["peeled_matched"] for x in available),
                f030_color_error_by_bin=peel_errors("peeled_color_error"),
                f030_contributing_color_error=peel_errors("contributing_color_error"),admitted_heads=admitted)


def source_witnesses(rows):
    parts=[r["substitute_source"] for r in rows]
    n=sum(v["parents"] for v in parts)
    counts=sum_dicts(v["any_at_threshold"] for v in parts)
    def hist(key):
        return np.sum([r[key]["bins"] for r in parts],axis=0).tolist()
    qualified=sum(r["S3_qualified"] or 0 for r in parts)
    available=sum(r["parents"] for r in parts if r["S3_qualified"] is not None)
    return dict(parents=n,highest_bin_counts=hist("best"),original_bin_counts=hist("S0"),
                highest_unknown=int(n-sum(hist("best"))),any_at_threshold=counts,
                fractions={k:v/n if n else None for k,v in counts.items()},
                S3=dict(qualified=qualified,f030_parents=available))


def complete_witness_rows():
    files={p for items in read(OUT/"source-witnesses.json")["files"].values() for p in items}
    rows=[]
    for cohort,name,frame,_,_,cases in ref.frames():
        for case in cases:
            path=str(witness_path(name,case).relative_to(OUT))
            if path not in files:
                continue
            parents,variants,best,observed=read_witnesses(name,case)
            rows.append(dict(cohort=cohort,scene=name,frame=frame,focus_index=case["focus_index"],
                             substitute_source=raw_s3_source_coverage(best,observed,parents,variants)))
    if len(rows)!=105:
        raise ValueError("Expected 105 CPU witness audits")
    return rows


def cpu_query_source():
    path=OUT/"source-query-cpu.json"
    if not path.is_file():
        return None
    receipt=read(path)
    data=receipt["rows"]
    if len(data)!=87:
        raise ValueError("Expected the 87 scored development cases in the CPU Query trace")

    def pool(rows):
        h=sum(r["heads"] for r in rows)
        available=[r for r in rows if r["heads"]]
        source=dict(cases=len(rows),heads=h)
        for mode in ("dominant","contributors"):
            hist=np.sum([r[mode]["bins"] for r in available],axis=0)
            unknown=sum(r[mode]["unknown"] for r in available)
            n=int(hist.sum())+unknown
            source[mode]=dict(bins=hist.tolist(),unknown=unknown,total=n,fractions=(hist/n).tolist())
        source["f030_linear_color_error_by_bin"]={}
        for k in ("0","1","2","3"):
            v=sum_dicts(r["peeled_color_error"][k] for r in available)
            n=int(v["samples"])
            source["f030_linear_color_error_by_bin"][k]=dict(samples=n,mae=v["absolute"]/(3*n) if n else None,
                psnr_db=float(-10*np.log10(max(1e-12,v["squared"]/(3*n)))) if n else None)
        return source

    return dict(source=str(path.relative_to(OUT)),focus_reference=receipt["focus_source"],
                all=pool(data),far=pool([r for r in data if r["focus_index"]==1]))


def focal_stats(subset):
    out={}
    for arm in FAMILIES[:3]:
        values=[r["focus_error"][arm] for r in subset]
        v=np.array([x["delta_rho"] for x in values],np.float64)
        if not len(v):
            continue
        out[arm]=dict(cases=len(v),mean=float(np.mean(v)),p50=float(np.median(v)),
                      p95=float(np.quantile(v,.95)),mean_CoC_px=float(v.mean()*25.4),
                      p50_CoC_px=float(np.median(v)*25.4),p95_CoC_px=float(np.quantile(v,.95)*25.4))
    return out


def original_query_check(rows):
    path=STAGE3C/"evaluation/step-000000/ema/report.json"
    if not path.exists():
        return dict(status="unavailable",source=str(path))
    older=read(path)
    cases={(r["scene"],r["case"]):r for r in older["rows"] if r["status"]=="evaluated"}
    deltas=[]
    for r in rows:
        key=r["scene"],r["case"]
        if key not in cases:
            continue
        old=cases[key]
        a=old["on"]["far_recoverable" if r["focus_index"] else "near_in_focus"]
        b=items(r,"Query","P")["Query"]["score"]
        x,y=(ref.pooled([p])["psnr_db"] for p in (a,b))
        if x is not None:
            deltas.append(abs(x-y))
    return dict(source=str(path),cases=len(deltas),max_difference_db=max(deltas) if deltas else None)


def s0_reference_check(rows):
    maxima={a:0. for a in FAMILIES[:3]}
    for r in rows:
        region="far_recoverable" if r["focus_index"] else "near_in_focus"
        expected=target_metadata(r["scene"],r["frame"],{"id":r["case"]})["methods"]
        for arm in maxima:
            old=expected[OLD_ARM[arm]]["optical"][region]
            new=items(r,arm,"G")["S0"]["score"]
            a,b=(ref.score_psnr(x) for x in (old,new))
            if old["pixels"]!=new["pixels"] or (a is None)!=(b is None):
                raise ValueError(f"S0 evaluation support changed for {r['scene']}/{r['case']}/{arm}")
            if a is not None:
                maxima[arm]=max(maxima[arm],abs(a-b))
    if max(maxima.values())>.01:
        raise ValueError(f"S0 did not reproduce the archived G rendering within .01 dB: {maxima}")
    return dict(cases=len(rows),max_abs_difference_db=maxima,tolerance_db=.01)


def zero_floors(rows, tag, family):
    arms=sorted(set().union(*(items(r,family,tag).keys() for r in rows)))
    def check(scores, floor):
        values=[s for _,s in scores]
        below=[dict(scene=r["scene"],case=r["case"],psnr_db=s["psnr"]) for r,s in scores if s["psnr"]<floor]
        return dict(ref.pooled(values),threshold_db=floor,passed=len(values)-len(below),below_floor=len(below),
                    below_floor_cases=below,evaluated=len(values),min_case_db=min(s["psnr"] for s in values))
    composed={}
    for arm in arms:
        scores=[(r,items(r,family,tag)[arm]["zero"]) for r in rows if arm in items(r,family,tag)]
        composed[arm]=check(scores,COMBINED_FLOOR)
    visible_method="Query" if family=="Query" else "F2"
    visible=[(r,items(r,family,tag)[visible_method]["visible_zero"]) for r in rows]
    return dict(visible=check(visible,VISIBLE_FLOOR),composed=composed)


def frame_rows(path, *, require_zero):
    rows=read(path)["rows"]
    first=rows[0]
    sidecar=zero_file(first["scene"],first["frame"])
    checks={r["case"]:r for r in read(sidecar)["rows"]} if sidecar.is_file() else {}
    for r in rows:
        if r["status"]!="evaluated" or "zero_checks" in r:
            continue
        supplement=checks.get(r["case"])
        if supplement is None:
            if require_zero:
                raise ValueError(f"Missing zero-aperture measurements for {r['scene']}/{r['case']}")
            continue
        for tag in ("G","P"):
            for family in FAMILIES:
                arms=supplement["render"][tag][family]
                arms=arms["methods"] if family=="Query" else arms
                for method,measurements in arms.items():
                    items(r,family,tag)[method].update(measurements)
        r["zero_checks"]=supplement["zero_checks"]
    return rows


def build(*, halted=False):
    paths=sorted((OUT/"frames").glob("*/*.json"))
    full_frames=len(paths)
    qualification=[p for p in sorted((OUT/"smoke/frames").glob("*/*.json"))
                   if not (OUT/"frames"/p.parent.name/p.name).exists()]
    paths.extend(qualification)
    if len(paths)!=54 and not halted:
        raise ValueError(f"Expected all 54 frames, received {len(paths)}")
    if halted:
        phase=read(OUT/"dispatch.json")
        if phase["phase"]!="halted":
            raise ValueError("A partial record requires a halted GPU run")
    all_rows=[r for p in paths for r in frame_rows(p,require_zero=not halted)]
    rows=[r for r in all_rows if r["status"]=="evaluated"]
    dev=[r for r in rows if r["cohort"]=="development"]
    legacy=[r for r in rows if r["cohort"]=="legacy"]
    if (len(dev),len(legacy))!=(87,18) and not halted:
        raise ValueError("Expected the registered 87 dev and 18 legacy cases")
    scopes=dict(development=dev,legacy=legacy,
                thin_structure=[r for r in dev if ref.title(r["scene"]) in ("dev003","dev004","dev005")],
                dev008=[r for r in dev if ref.title(r["scene"])=="dev008"])
    zero_scopes={name:[r for r in cases if "zero_checks" in r]
                 for name,cases in dict(all=rows,**scopes).items()}
    zero_tables={name:{tag:{arm:zero_floors(cases,tag,arm) for arm in FAMILIES} for tag in ("G","P")}
                 for name,cases in zero_scopes.items() if cases}
    audited=zero_scopes["all"]
    zero_policy=dict(visible_floor_db=VISIBLE_FLOOR,visible_max_delta_exclusive_db=VISIBLE_TOLERANCE,
        original_Query_composed_floor_db=COMBINED_FLOOR,original_Query_P_3c_max_delta_inclusive_db=START_TOLERANCE,
        comparison_convention="3c stores P; G uses its own focus-conditioned Query and keeps the original Query floor.",
        altered_and_F2_below_floor="record only",
        verified_cases=len(audited),
        visible_max_delta_db=max((max(r["zero_checks"]["visible_max_delta_db"].values()) for r in audited),default=None),
        original_Query_P_max_delta_from_3c_db=max((r["zero_checks"]["query_P_3c_abs_delta_db"]
                                                  for r in audited),default=None))
    failure=next((r for r in rows if (r["scene"],r["case"])==FAILURE_CASE[:2]),None)
    repro=None if failure is None else failure["render"][FAILURE_CASE[2]]["Query"].get("zero_change")
    if not halted and repro is None:
        raise ValueError("Missing zero-aperture candidate and ray audit for dev002 f030 far G")
    optical={}
    focus={}
    for name,parts in scopes.items():
        near=[r for r in parts if r["focus_index"]==0]
        far=[r for r in parts if r["focus_index"]==1]
        focus[name]=dict(near=focal_stats(near),far=focal_stats(far))
        optical[name]={tag:dict(
            far={arm:optical_table(far,tag,arm) for arm in FAMILIES},
            near={arm:{key:s["vs_F2"]["PSNR"] for key,s in optical_table(near,tag,arm).items()}
                  for arm in FAMILIES if near}) for tag in ("G","P")}
    by_scene=defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)
    witness_rows=complete_witness_rows()
    witness_by_scene=defaultdict(list)
    for r in witness_rows:
        witness_by_scene[r["scene"]].append(r)
    witness_scopes=dict(development=[r for r in witness_rows if r["cohort"]=="development"],
                        legacy=[r for r in witness_rows if r["cohort"]=="legacy"],
                        thin_structure=[r for r in witness_rows if ref.title(r["scene"]) in ("dev003","dev004","dev005")],
                        dev008=[r for r in witness_rows if ref.title(r["scene"])=="dev008"])
    crop=read(OUT/"source-crop-summary.json")
    crop["same_denominator_comparison"]=dict(
        development_far_only=crop["development_far"]["recoverable_geometry_fraction"],
        development_all_scored_cases=crop["development_far"]["recoverable_geometry_pixels"]/(87*512**2),
        training_far_only=crop["training"]["far"]["recoverable_geometry_fraction"],
        training_all_optical_crops=crop["training"]["far"]["recoverable_geometry_pixels"]/crop["training"]["all"]["pixels"])
    derived=dict(status="halted" if halted else "complete",
                 case_counts=dict(development=len(dev),legacy=len(legacy),test=0),
                 optical_scope=dict(completed_production_frames=full_frames,planned_frames=54,
                                    qualification_frames_added=len(qualification),
                                    planned_development_cases=87,planned_legacy_cases=18),
                 source_protocol=dict(
                     direct_supervision="route3_v2data_query_prepare.py placement/label_case: associate 256-grid source candidate with its GT 512 centre point, project into the target, require matching deeper peeled surface and ray support; take target fNNN-peeled-color.npz whole pixel at integer projected coordinates. No first_visible_other_frame in target colour. C uses analytic material/UV texture lookup.",
                     actual_Query="src/light_candidate_fusion.py: two heads, frozen q/softmax weights, fused world point, source-frame reprojection then bilinear RGB plus bounded learned residual. Count all contributors with positive RGB weights. Dominant results use the highest effective RGB weight per retained head; weighted results sum the entire effective mixture.",
                     original_substitute="Stage 3a hidden_parents: first_visible_other_frame and nearest whole-pixel RGB in new B dev. Legacy S0 uses the archived same-frame peeled EXR oracle, as in E0.",
                     Query_purity="GT anchor is the surface at the dominant source candidate's original centre pixel. The exact four bilinear taps in each actual RGB source are weighted by their 4x4 occupancy of that surface. Purity is spatial coverage of that identity, not correctness of the inferred hidden surface.",
                     predicted="For every actual E-H2@2k source pixel, purity = 2*abs(w_back_eff-.5), interpolated at the same four taps. Query geometry, source slots, O1 and per-head opacity are frozen. Drop unreliable RGB weights and renormalize the original mixture on survivors; keep the original residual. A head without admissible colour is omitted so F2 receives the residual transmittance.",
                     S1_S2="Measure 4x4 fixed subpixel casts per actual pixel. S1 uses highest strictly positive occupancy among point-visible frames, ties use the earliest listed source. S2 uses linear RGB channelwise median for all positive occupancy and thresholds .5, .9, .99. Empty support goes to F2. For S3 f030 require matching object and axial depth to the centre peeled surface, with max(.01 m, 0.5%) tolerance. Uncovered parents are left to F2.",
                     stratification="Rank hidden colour-parent purity by greatest compositing weight across K64, at 1e-6 visibility tolerance. At each threshold record recoverable if any baseline hidden contribution in a qualifying bin is visible, otherwise unrecoverable. Ground-truth purity strata are held fixed across Query variants. Separate point-level purity and image-level radiance.",
                     perceptual="Support-weighted 64x64 full fixed-grid Alex LPIPS patches in display RGB. Direct distances are to each family's original S0 or Query; paired LPIPS differences also compare those methods to Cycles.",
                     paired="Pool PSNR by SSE/(3*pixels); scene deltas use the same frame support; report population standard deviation and signed scene counts. S3 comparison subsets f030.",
                     anchors="Only f030 uses centre peel for representative RGB error and S3. The .99 cutoff controls the fixed figures only; all requested thresholds are kept in numbers.",
                     optics="G uses GT P2 nominal subject focus and P uses each geometry's own focus, both evaluate original masks. Focus shift uses K=25.4 px per inverse metre.",
                 ),
                 crop_audit=crop,
                 source_query={"all":source_query(dev),"by_scene":{k:source_query(g) for k,g in by_scene.items()
                                                               if g[0]["cohort"]=="development"}},
                 source_query_cpu=cpu_query_source(),
                 witness_availability={name:source_witnesses(v) for name,v in witness_scopes.items()},
                 witness_by_scene={name:source_witnesses(v) for name,v in witness_by_scene.items()},
                 far_witness_by_scene={name:source_witnesses([r for r in v if r["focus_index"]==1])
                                       for name,v in witness_by_scene.items()},
                 focus_error=focus,focus_by_case=[{k:r[k] for k in ("scene","case","frame","focus_index","focus_error")}
                                                  for r in rows],
                 optical=optical,query_original_P=original_query_check(rows),
                 S0_archived_G={scope:s0_reference_check(parts) for scope,parts in scopes.items() if parts},
                 zero_aperture=dict(policy=zero_policy,by_scope=zero_tables,
                                    dev002_f030_far_G=repro),
                 panels=[item for r in rows for item in r.get("panel",[])],
                 skipped=[dict(scene=r["scene"],case=r["case"],reason=r["status"]) for r in all_rows
                          if r["status"]!="evaluated"],
                 checkpoints=dict(Query=str(QUERY),Query_sha256=sha256(QUERY),E_H2_step=2000,
                                  E_H2_SHA256=sha256(ROOT/"outputs/route3-v2data-E0-2026-11-26-01/E-H2/step-002000.pt")))
    if (not halted and len(derived["panels"])!=10) or any(not (OUT/f).is_file() for f in derived["panels"]):
        raise ValueError("Expected all ten requested interleaved image panels")
    if halted:
        derived["stop"]=dict(phase,log="inference-resumed.log")
    if RECORD.is_file():
        prior=read(RECORD)
        historical=prior.get("resumed_after") or prior.get("stop",{})
        if historical.get("arm")=="QGT@0.50":
            derived["resumed_after"]=dict(historical,classification="停止条件过严（规划方），已放宽后续跑")
        if "plot_observations" in prior:
            derived["plot_observations"]=prior["plot_observations"]
    return derived


def lines_for(report):
    fmt=lambda x,precision=3:"—" if x is None else f"{x:.{precision}f}"
    n=report["case_counts"]
    caption=f"光学表覆盖新开发 {n['development']} 个 case、原场景 {n['legacy']} 个 case。"
    if report["status"]=="halted":
        caption+="GPU 触发零光圈下限后停止。以下是已完成配对，含三帧资格核对。CPU 来源统计覆盖全部新开发 87 个 case。"
    lines=["# 隐藏层见证颜色取色对照","",caption,
           "远焦以 G 为主。S3 只在 f030。",
           "所有数值按对应支持合并，S3 的配对在共同的 f030 上算。","",
           "## 3a 裁块的几何命中","",
           "| 分母口径 | 训练远焦可恢复占比 | 开发集远焦可恢复占比 |",
           "|---|---:|---:|"]
    comparison=report["crop_audit"]["same_denominator_comparison"]
    lines.extend([
           f"| 只计远焦 | {comparison['training_far_only']:.3%} | {comparison['development_far_only']:.3%} |",
           f"| 计入全部焦点 | {comparison['training_all_optical_crops']:.3%} | {comparison['development_all_scored_cases']:.3%} |",
           "", "训练日志存有裁块与父点数。像素比例用相同的几何光线定义统计。", "",
           "## 见证可用率","",
           "| 场景 | 父点数 | 任意正覆盖 | ≥.50 | ≥.90 | ≥.99 | f030 S3 |",
           "|---|---:|---:|---:|---:|---:|---:|"])
    for scene,value in report["witness_by_scene"].items():
        rates=[f"{v:.1%}" if v is not None else "—" for v in value["fractions"].values()]
        n=value["S3"]
        lines.append(f"| {ref.title(scene)} | {value['parents']} | "+" | ".join(rates)+f" | {n['qualified']}/{n['f030_parents']} |")
    if report["source_query_cpu"]:
        source=report["source_query_cpu"]["all"]
        lines.extend(["", "## 实际 Query 取色来源", "",
                      "| 覆盖率 | RGB 贡献样本数 | 占比 | f030 剥离匹配数 | 线性 RGB MAE |",
                      "|---|---:|---:|---:|---:|"])
        c=source["contributors"]
        for i,band in enumerate(("<0.50","0.50–0.90","0.90–0.99","≥0.99")):
            e=source["f030_linear_color_error_by_bin"][str(i)]
            lines.append(f"| {band} | {c['bins'][i]:,} | {c['fractions'][i]:.2%} | {e['samples']:,} | {fmt(e['mae'],4)} |")
        lines.extend(["",f"另有 {c['unknown']:,} 个 RGB 贡献样本的覆盖率未知。"])
    for scope,table in report["optical"].items():
        for f,focus in table.items():
            lines.extend(["",f"## {scope} · {f} 远焦","",
                "| 几何 | 方法 | 场次 | PSNR | LPIPS | 相对 F2 配对中位 | 相对原色配对中位 | 正／负 |",
                "|---|---|---:|---:|---:|---:|---:|---:|"])
            for arm,methods in focus["far"].items():
                for tag,m in methods.items():
                    y=m["pooled"]
                    p=m["vs_original"]["PSNR"]
                    lines.append(f"| {TITLES[arm]} | {tag} | {m['cases']} | {fmt(y['psnr_db'])} | "
                                 f"{fmt(y.get('patch_lpips'),5)} | {fmt(m['vs_F2']['PSNR']['median'])} | "
                                 f"{fmt(p['median'])} | {p['positive']}/{p['negative']} |")
    lines.extend(["","## 焦距误差","",
        "| 范围 | 焦层 | 几何 | 绝对 Δρ 中位 / p95 | CoC 中位 / p95 px |",
        "|---|---|---|---:|---:|"])
    for school,focus in report["focus_error"].items():
        for label,models in focus.items():
            for k,r in models.items():
                lines.append(f"| {school} | {label} | {TITLES[k]} | {fmt(r['p50'])} / {fmt(r['p95'])} | "
                             f"{fmt(r['p50_CoC_px'])} / {fmt(r['p95_CoC_px'])} |")
    zero=report["zero_aperture"]
    policy=zero["policy"]
    if policy["verified_cases"]:
        lines.extend(["", "## 零光圈", "",
                      f"核对 {policy['verified_cases']} 个 case。可见层同几何最大差 "
                      f"{policy['visible_max_delta_db']:.8g} dB，原版 Query P 与 3c 起点最大差 "
                      f"{policy['original_Query_P_max_delta_from_3c_db']:.8g} dB。",
                      "", "| 几何 / 系统 | 档 | G <60 / 总数 | G 最低 dB | P <60 / 总数 | P 最低 dB |",
                      "|---|---|---:|---:|---:|---:|"])
        overall=zero["by_scope"]["all"]
        for family in FAMILIES:
            for name,g in overall["G"][family]["composed"].items():
                p=overall["P"][family]["composed"][name]
                lines.append(f"| {TITLES[family]} | {name} | {g['below_floor']}/{g['evaluated']} | "
                             f"{fmt(g['min_case_db'])} | {p['below_floor']}/{p['evaluated']} | {fmt(p['min_case_db'])} |")
    issue=zero["dev002_f030_far_G"]
    if issue:
        rays=issue["pinhole_rays"]["positive_compositing_weight"]
        lines.extend(["", "### dev002 f030 远焦 G", "",
                      f"原版 Query {issue['original_query_db']:.6f} dB，QGT@0.50 {issue['variant_db']:.6f} dB，"
                      f"差 {issue['variant_minus_original_db']:+.6f} dB。去掉 {issue['removed_hidden_points']:,} 个隐藏点，"
                      f"其余隐藏点中有 {issue['recolored_hidden_points']:,} 个发生实质换色。"
                      f"有 {rays['either']:,} 条针孔光线从两类改动点收到正合成权重。"])
    return "\n".join(lines)+"\n"


def collect(*, halted=False):
    report=build(halted=halted)
    save_json(RECORD,report)
    (OUT/"汇总.md").write_text(lines_for(report))
    page=('<!doctype html><meta charset="utf-8"><title>隐藏层颜色纯度对照</title>'
          '<h1>隐藏层颜色纯度对照</h1><p>G 远焦 · f030，面板采用 .99 阈值。</p><p><a href="汇总.md">汇总表</a></p>\n')
    (OUT/"index.html").write_text(page+"\n".join(
        f'<figure><a href="{escape(p)}"><img src="{escape(p)}" width="1120" loading="lazy"></a>'
        f'<figcaption>{escape(Path(p).stem)}</figcaption></figure>' for p in report["panels"]))
    if not halted:
        save_json(OUT/"dispatch.json",dict(phase="collected",at=time.time(),record=str(RECORD)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--halted",action="store_true")
    collect(halted=parser.parse_args().halted)
