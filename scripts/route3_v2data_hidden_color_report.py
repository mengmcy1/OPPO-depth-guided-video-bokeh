"""Collect hidden-colour results with the same region pixels in each pair."""
from collections import defaultdict
from html import escape
from pathlib import Path
import time

import numpy as np

import route3_v2data_module2_infer as ref
from route3_v2data_B_batch import sha256
from route3_v2data_E0_rerender import STAGE3C
from route3_v2data_E_labels import save_json
from route3_v2data_hidden_color_sources import OUT, ROOT, read
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


def build():
    paths=sorted((OUT/"frames").glob("*/*.json"))
    if len(paths)!=54:
        raise ValueError(f"Expected all 54 frames, received {len(paths)}")
    all_rows=[r for p in paths for r in read(p)["rows"]]
    rows=[r for r in all_rows if r["status"]=="evaluated"]
    dev=[r for r in rows if r["cohort"]=="development"]
    legacy=[r for r in rows if r["cohort"]=="legacy"]
    if (len(dev),len(legacy))!=(87,18):
        raise ValueError("Expected the registered 87 dev and 18 legacy cases")
    scopes=dict(development=dev,legacy=legacy,
                thin_structure=[r for r in dev if ref.title(r["scene"]) in ("dev003","dev004","dev005")],
                dev008=[r for r in dev if ref.title(r["scene"])=="dev008"])
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
    crop=read(OUT/"source-crop-summary.json")
    crop["same_denominator_comparison"]=dict(
        development_far_only=crop["development_far"]["recoverable_geometry_fraction"],
        development_all_scored_cases=crop["development_far"]["recoverable_geometry_pixels"]/(len(dev)*512**2),
        training_far_only=crop["training"]["far"]["recoverable_geometry_fraction"],
        training_all_optical_crops=crop["training"]["far"]["recoverable_geometry_pixels"]/crop["training"]["all"]["pixels"])
    derived=dict(case_counts=dict(development=len(dev),legacy=len(legacy),test=0),
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
                 witness_availability={name:source_witnesses(v) for name,v in scopes.items()},
                 witness_by_scene={name:source_witnesses(v) for name,v in by_scene.items()},
                 far_witness_by_scene={name:source_witnesses([r for r in v if r["focus_index"]==1])
                                       for name,v in by_scene.items()},
                 focus_error=focus,focus_by_case=[{k:r[k] for k in ("scene","case","frame","focus_index","focus_error")}
                                                  for r in rows],
                 optical=optical,query_original_P=original_query_check(rows),
                 panels=[item for r in rows for item in r.get("panel",[])],
                 skipped=[dict(scene=r["scene"],case=r["case"],reason=r["status"]) for r in all_rows
                          if r["status"]!="evaluated"],
                 checkpoints=dict(Query=str(QUERY),Query_sha256=sha256(QUERY),E_H2_step=2000,
                                  E_H2_SHA256=sha256(ROOT/"outputs/route3-v2data-E0-2026-11-26-01/E-H2/step-002000.pt")))
    if len(derived["panels"])!=10 or any(not (OUT/f).is_file() for f in derived["panels"]):
        raise ValueError("Expected all ten requested interleaved image panels")
    return derived


def lines_for(report):
    fmt=lambda x,precision=3:"—" if x is None else f"{x:.{precision}f}"
    lines=["# 隐藏层见证颜色取色对照","",
           "新开发集 87 个 case，原 3 场景 18 个 case。远焦以 G 为主。S3 只在 f030。",
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
    return "\n".join(lines)+"\n"


def collect():
    report=build()
    save_json(RECORD,report)
    (OUT/"汇总.md").write_text(lines_for(report))
    page=('<!doctype html><meta charset="utf-8"><title>隐藏层颜色纯度对照</title>'
          '<h1>隐藏层颜色纯度对照</h1><p>G 远焦 · f030，面板采用 .99 阈值。</p><p><a href="汇总.md">汇总表</a></p>\n')
    (OUT/"index.html").write_text(page+"\n".join(
        f'<figure><a href="{escape(p)}"><img src="{escape(p)}" width="1120" loading="lazy"></a>'
        f'<figcaption>{escape(Path(p).stem)}</figcaption></figure>' for p in report["panels"]))
    save_json(OUT/"dispatch.json",dict(phase="collected",at=time.time(),record=str(RECORD)))


if __name__=="__main__":
    collect()
