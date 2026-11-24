"""Compare prior-only geometry using the fixed Stage 3a optical protocol."""
from collections import defaultdict

import numpy as np
import torch

import route3_ovg_data as ovg
from route3_ovg_subpixel import save_npz
from route3_v2data_B_inputs import library_unit_paths
from route3_v2data_baseline_dev3_extended_report import pooled
from route3_v2data_stage1_evaluate import (
    development_cases, load_paired_checkpoint, near_subject, new_regions, p2_focus,
)
from route3_v2data_stage2_eval import f2_for, optical_target
from route3_v2data_stage2_train import INITIAL, sample
from route3_v2data_stage3a_prior_diagnostic import TARGETS, THIN_SCENES, quantiles, thin_regions
from src.aperture_renderer import aperture_disk
from src.c1_bokehme import region_sse
from src.layered_aperture import merge_layers
from src.ovg_optics import fixed_visible_surfels

PRIORS = ("P1", "P2", "P3", "P4", "P5")
METHODS = ("start", "endpoint", *PRIORS)
PANEL_PRIORS = ("P1", "P2", "P3", "P5")
REGIONS = ("near_in_focus", "far_recoverable")
LABELS = {
    "start": "起点网络", "endpoint": "20k EMA",
    "P1": "P1 融合先验", "P2": "P2 DepthSplat", "P3": "P3 校准 SHARP",
    "P4": "P4 三角化", "P5": "P5 逐像素最优先验",
}
PROTOCOL = (
    "全部有效 f001/f030/f060，近焦与远焦，f/1.4。V0、不拆分、64 条光圈光线，"
    "各方法从自己的前层深度计算 P2 实际焦距，并按该深度重新放置同一 GT 隐藏层替身和投影同一 F2 缓存。"
    "PSNR 使用同一 GT 评分掩码，合并 SSE 除以 3×像素数。起点和 20k EMA 复用正式光学分数。"
    "P1 取 20k EMA 的融合先验。P2 和 P4 无有效输入时以 P1 填补，P3 沿用同样规则。"
    "P5 只从真实有效的 P2/P3/P4 按 GT 逆深度最小绝对误差选取，并列时依次优先 P2/P3/P4；"
    "无有效 GT 或有效候选时取 P1。P5 使用真值，只作选择空间的参照。"
    "逆深度沿用 10k 的 GT 中心命中分组，绝对误差单位 m⁻¹。"
    "前景 CoC 误差沿用固定标称焦点的逆深度缩放，按有效光学 case 汇总，单位 px。"
)


def prior_depths(inputs, fused, reference):
    depths = inputs["prior_depth"][0, [1,0,2]]
    confidence = inputs["prior_conf"][0, [1,0,2]]
    valid = torch.isfinite(depths) & (depths > 0) & (confidence > 0)
    gt = torch.as_tensor(reference["gt"], device=fused.device, dtype=fused.dtype)
    gt_valid = torch.as_tensor(reference["valid"], device=fused.device) & torch.isfinite(gt) & (gt > 0)
    completed = torch.where(valid, depths, fused)
    errors = (completed.reciprocal()-torch.where(gt_valid,gt,1.).reciprocal()).abs()
    best = torch.where(valid,errors,torch.inf).argmin(0)
    usable = gt_valid & valid.any(0)
    oracle = torch.where(usable,completed.gather(0,best[None])[0],fused)
    fallback = dict(zip(PRIORS[1:4],~valid))
    fallback["P5"] = ~usable
    return dict(zip(PRIORS,(fused,*completed,oracle))), fallback


def stats(samples):
    data = np.concatenate(samples) if samples else np.empty(0)
    p50,p95 = quantiles(data,(.5,.95))
    return dict(pixels=int(data.size),p50=p50,p95=p95)


def summarize(rows, samples, missing):
    result = {}
    for method in METHODS:
        regions = {}
        for region in REGIONS:
            scenes = defaultdict(list)
            for row in rows:
                if row["region"] == region:
                    score = row["methods"][method]
                    if score["pixels"]:
                        scenes[row["scene"]].append(score)
            combined = pooled([score for scores in scenes.values() for score in scores])
            per_scene = {name:pooled(scores) for name,scores in scenes.items()}
            psnr = [s["psnr"] for s in per_scene.values()]
            regions[region] = dict(combined=combined,scene_median_db=quantiles(np.asarray(psnr),(.5,))[0],
                                   per_scene=per_scene)
        fallback = {}
        for scope in ("full", "gap_background", "foreground"):
            totals = [v[method][scope] for v in missing if method in v]
            if totals:
                count,pixels = (sum(t[i] for t in totals) for i in (0,1))
                fallback[scope] = dict(filled=count,pixels=pixels,ratio=count/pixels if pixels else None)
        result[method] = dict(regions=regions,
                              **{k:stats(v) for k,v in samples[method].items()},
                              fallback=fallback)
    return result


def evaluated_frames():
    for unit in library_unit_paths("dev"):
        slug = "dev"+unit.name.split("-")[1]
        if slug in TARGETS:
            cases = development_cases(unit)
            for frame in (1,30,60):
                yield slug,unit,frame,[c for c in cases if c["frame"] == frame and c["fstop"] == 1.4]


def case_score(row, region):
    return dict(row["scores"]["stage3"]["metrics"][region],focus_m=row["focus_m"])


def evaluate_prior_comparison(net, placer, output, fixed, current_rows, start_rows):
    from route3_v2data_stage3a_eval import hidden_layer, render

    current = {(r["scene"],r["case"]):r for r in current_rows}
    start = {(r["scene"],r["case"]):r for r in start_rows}
    fixed = set(map(tuple,fixed))
    rows, unscored, artifacts, groups = [], [], {}, {}
    initial = load_paired_checkpoint(INITIAL,"ema").eval()
    for slug,unit,frame,cases in evaluated_frames():
        name = unit.name
        item = ("B",name,ovg.TRAIN_FRAMES.index(frame))
        inputs, target = sample(item)
        ref = ovg.b_edge_reference(item,target)
        center = ovg.b_foreground_reference(name,frame)
        masks, _ = thin_regions(slug,unit,center)
        pred = net(inputs,protect_residual=True,render_heads=False)
        estimates, fallbacks = prior_depths(inputs,pred["fused_depth"][0],center)
        estimates["start"] = initial(inputs,protect_residual=True,render_heads=False)["depth1"][0]
        estimates["endpoint"] = pred["depth1"][0]
        depths = {key:value.float().cpu().numpy() for key,value in estimates.items()}
        if any(not np.all(np.isfinite(d) & (d > 0)) for d in depths.values()):
            raise FloatingPointError(f"Invalid depth in prior comparison for {name} f{frame:03d}")
        values, missing = groups.setdefault(slug,({m:defaultdict(list) for m in METHODS},[]))
        counts = {}
        for method, invalid in fallbacks.items():
            mask = invalid.cpu().numpy()
            counts[method] = dict(full=(int(mask.sum()),mask.size),
                                 **{k:(int((mask & v).sum()),int(v.sum())) for k,v in masks.items()})
        missing.append(counts)
        gt_inv = np.divide(1.,center["gt"],where=center["gt"]>0,
                           out=np.zeros(center["gt"].shape,np.float32),dtype=np.float32)
        for method, depth in depths.items():
            actual = abs(1/depth-gt_inv)
            for region, mask in masks.items():
                values[method][region].append(actual[mask])
        K = inputs["K"][0]
        k = K.cpu().numpy()
        offsets = aperture_disk(64,.05/(2*1.4),device=K.device,dtype=K.dtype)
        for case in cases:
            ident, idx, nominal = case["id"],case["focus_index"],case["focus_m"]
            key = (name,ident)
            base, end = start[key], current[key]
            if bool(base["scores"]) != bool(end["scores"]):
                raise ValueError(f"Unmatched official case validity for {key}")
            if not end["scores"]:
                unscored.append(dict(scene=slug,case=ident,reason=end["status"]))
                continue
            region = REGIONS[idx]
            ground = new_regions(unit,case,ref["gt"],ref["valid"],k)
            area = ground["in_focus"] if idx == 0 else ground["recoverable"]
            subject = near_subject(ref["gt"],ref["valid"],k,nominal)
            truth = optical_target(unit,name,case)
            scale = float(k[0,0])*.05/(2*case["fstop"])*nominal/(nominal-.05)
            for method in METHODS:
                values[method]["foreground_coc_px"].append(values[method]["foreground"][-1] * scale)
            row = dict(scene=slug,case=ident,region=region,methods={
                "start":case_score(base,region),"endpoint":case_score(end,region)})
            pictures = {}
            for method in PRIORS:
                depth = estimates[method]
                focus = p2_focus(depths[method],subject)
                if focus is None:
                    raise ValueError(f"Missing P2 focus for {key} {method}")
                visible = fixed_visible_surfels({"depth1":depth[None]},inputs)
                hidden,_ = hidden_layer(placer,unit,name,frame,case,ref,depth,K)
                cloud = merge_layers(visible,hidden)
                fill = f2_for(unit,None,frame,inputs,focus,offsets)
                image = render(cloud,K,focus,offsets,fill)
                row["methods"][method] = dict(region_sse(image,truth,area),focus_m=focus)
                if method in PANEL_PRIORS and (slug,frame,idx) in fixed:
                    pictures[method+"_rgb"] = image.astype(np.float16)
                    pictures[method+"_depth"] = depths[method].astype(np.float16)
                del visible, hidden, cloud, fill
            if any(s["pixels"] != int(area.sum()) for s in row["methods"].values()):
                raise ValueError(f"Mismatched prior comparison scoring masks for {key}")
            rows.append(row)
            if pictures:
                path = output/"prior-comparison"/f"{slug}-{ident}.npz"
                path.parent.mkdir(exist_ok=True)
                save_npz(path,**pictures)
                artifacts[f"{slug}/{ident}"] = str(path.relative_to(output))
        print("STAGE3A_PRIOR_COMPARE",slug,frame,flush=True)
    if not rows or not groups:
        raise ValueError("No scored prior comparison case")
    combined = {m:defaultdict(list) for m in METHODS}
    combined_missing = []
    for slug in THIN_SCENES:
        if slug in groups:
            values, missing = groups[slug]
            combined_missing.extend(missing)
            for method,regions in values.items():
                for key,samples in regions.items():
                    combined[method][key].extend(samples)
    summaries = {"thin_structure":summarize([r for r in rows if r["scene"] in THIN_SCENES],
                                             combined,combined_missing)}
    summaries.update({slug:summarize([r for r in rows if r["scene"] == slug],values,missing)
                      for slug,(values,missing) in groups.items()})
    return dict(protocol=PROTOCOL,valid_cases=len(rows),unscored=unscored,cases=rows,
                groups=summaries,panel_artifacts=artifacts)


def comparison_markdown(report):
    lines = ["## 先验对照", "",report["protocol"],"",
             f'有效光学 case {report["valid_cases"]} 个，无有效主体 {len(report["unscored"])} 个。',""]
    def fmt(v,n=3):
        return "—" if v is None else f"{v:.{n}f}"
    for group, methods in report["groups"].items():
        lines.extend([f"### {'dev003/004/005 合并' if group == 'thin_structure' else group}","",
                      "| 方法 | 近焦 PSNR 合并 / 场景中位 | 远焦 PSNR 合并 / 场景中位 | "
                      "前景 CoC p50 / p95 | 框内背景逆深度误差 p50 / p95 | 前景逆深度误差 p50 / p95 |",
                      "|---|---:|---:|---:|---:|---:|"])
        for method, values in methods.items():
            cols = [f'{fmt(r["combined"]["psnr"])} / {fmt(r["scene_median_db"])}'
                    for r in values["regions"].values()]
            cols.extend(f'{fmt(values[k]["p50"])} / {fmt(values[k]["p95"])}' if k in values else "—"
                        for k in ("foreground_coc_px","gap_background","foreground"))
            lines.append("| "+" | ".join([LABELS[method],*cols])+" |")
        lines.extend(["","| 方法 | 全帧 P1 补值 | 框内背景 P1 补值 | 前景 P1 补值 |",
                      "|---|---:|---:|---:|"])
        for method, values in methods.items():
            if values["fallback"]:
                cols = [f'{r["filled"]:,}/{r["pixels"]:,}（{100*r["ratio"]:.2f}%）'
                        if r["ratio"] is not None else "—" for r in values["fallback"].values()]
                lines.append("| "+" | ".join([LABELS[method],*cols])+" |")
        lines.append("")
    return lines
