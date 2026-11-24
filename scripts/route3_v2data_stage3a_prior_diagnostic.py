"""Trace inverse-depth errors on fixed thin foreground and background pixels."""
import numpy as np

import route3_ovg_data as ovg
from route3_ovg_subpixel import save_npz
from route3_v2data_B_inputs import library_unit_paths
from route3_v2data_B_plan import load
from route3_v2data_stage1_evaluate import load_paired_checkpoint
from route3_v2data_stage2_train import INITIAL, sample
from route3_v2data_stage3a import checkpoint_model, thin_surface_ids
from src.ovg_net import residual_attenuation

THIN_SCENES = ("dev003", "dev004", "dev005")
TARGETS = (*THIN_SCENES, "dev008")
PRIOR_KEYS = ("sharp_aligned", "depthsplat", "triangulation")
ERROR_KEYS = (*PRIOR_KEYS, "fused", "start_fused", "start", "prediction")
PLOT_KEYS = ("attenuation", "gate", "residual", "absolute_residual", "residual_before_attenuation", "start_residual")
LABELS = {
    "gap_background": "框内 GT 背景", "foreground": "GT 细结构前景",
    "sharp_aligned": "SHARP 校准", "depthsplat": "DepthSplat", "triangulation": "三角化",
    "fused": "当前融合先验", "start_fused": "起点融合先验", "start": "起点预测",
    "prediction": "当前预测", "attenuation": "总残差衰减系数", "gate": "残差门控",
    "residual": "实际逆深度残差", "absolute_residual": "实际残差绝对值",
    "residual_before_attenuation": "衰减前逆深度残差",
    "start_residual": "起点实际残差",
}
DEFINITION = (
    "使用固定 f030 的原生 GT 中心命中与轴向深度。对每个指定物体取 GT 投影包围框。"
    "框内背景须命中非指定前景表面，且深度大于该框物体最远的可见深度。多个框取并集并去重。"
    "早期场景缺少逐物体出现性清单时，按生成类别 thin_structure 与真值指定前景 ID 取掩码。"
    "dev008 无 thin_structure 标签，单列两件导入前景及其框内背景。"
    "各先验按模型输入的有效置信度单独筛选并报告像素数，融合和预测使用全部有效分组像素。"
    "误差为绝对逆深度差，另列有符号中位数，正值表示偏近。实际残差为预测逆深度减该检查点的融合先验，单位 m⁻¹。"
    "衰减系数来自输入可信度与先验一致性，门控是多视角残差分支的 sigmoid 输出。"
    "衰减前残差由实际残差除以衰减系数得到。"
)


def gt_regions(reference, ids):
    depth = reference["gt"]
    obj = reference["object_id"]
    valid = reference["valid"] & np.isfinite(depth) & (depth > 0)
    background = valid & ~np.isin(obj, reference["foreground"])
    front = valid & np.isin(obj, ids)
    gaps = np.zeros_like(front)
    boxes = []
    for identity in ids:
        y, x = np.where(front & (obj == identity))
        if not y.size:
            continue
        y0, y1, x0, x1 = map(int, (y.min(), y.max()+1, x.min(), x.max()+1))
        r = np.s_[y0:y1, x0:x1]
        deepest = float(depth[y, x].max())
        gaps[r] |= background[r] & (depth[r] > deepest)
        boxes.append(dict(surface_id=int(identity), yxyx_exclusive=[y0, x0, y1, x1],
                          deepest_foreground_m=deepest))
    return {"gap_background": gaps, "foreground": front}, boxes


def thin_regions(slug, unit, reference):
    ids = thin_surface_ids(unit)
    if slug == "dev008" or (not ids and load(unit/"scene-spec.json")["category"] == "thin_structure"):
        ids = reference["foreground"]
    if len(ids) == 0:
        raise ValueError(f"Missing designated foreground labels for {unit.name}")
    return gt_regions(reference, ids)


def inputs(unit, frame):
    item = ("B", unit.name, ovg.TRAIN_FRAMES.index(frame))
    return sample(item)[0]


def predictions(net, x):
    pred = net(x, protect_residual=True, render_heads=False)
    return tuple(pred[k][0].float().cpu().numpy() for k in ("inverse1", "fused_depth", "confidence"))


def scene_data(net, case, initial, output):
    slug, unit, frame = case
    x = inputs(unit, frame)
    prediction, fused_depth, gate = predictions(net, x)
    start, start_fused_depth, _ = initial
    attenuation = residual_attenuation(x["prior_depth"], x["prior_conf"])[0].cpu().numpy()
    prior_z = x["prior_depth"][0].float().cpu().numpy()
    prior_conf = x["prior_conf"][0].float().cpu().numpy()
    prior_valid = (prior_conf > 0) & np.isfinite(prior_z) & (prior_z > 0)
    priors = np.where(prior_valid, 1 / prior_z.clip(min=.05), np.nan)
    reference = ovg.b_foreground_reference(unit.name, frame)
    masks, boxes = thin_regions(slug,unit,reference)
    gt = np.divide(1., reference["gt"], where=reference["gt"] > 0,
                   out=np.zeros(reference["gt"].shape, dtype=np.float32), dtype=np.float32)
    fused = 1 / fused_depth
    initial_fused = 1 / start_fused_depth
    estimates = dict(zip(ERROR_KEYS, (*priors, fused, initial_fused, start, prediction)))
    residual = prediction-fused
    plots = dict(attenuation=attenuation, gate=gate, residual=residual,
                 absolute_residual=np.abs(residual), residual_before_attenuation=residual/attenuation,
                 start_residual=start-initial_fused)
    samples = {
        name: {"errors": {k: v[mask]-gt[mask] for k, v in estimates.items()},
               "distributions": {k: v[mask] for k, v in plots.items()}}
        for name, mask in masks.items()
    }
    name = f"{slug}-f{frame:03d}"
    path = output/"prior-residual"/f"{name}.npz"
    save_npz(path, fused_depth=fused_depth, attenuation=attenuation)
    return name, samples, dict(scene=unit.name, frame=frame, artifact=str(path.relative_to(output)),
                               bounding_boxes=boxes)


def quantiles(values, qs):
    if not values.size:
        return [None]*len(qs)
    return [float(v) for v in np.quantile(values, qs)]


def metrics(groups):
    result = {}
    for mask in ("gap_background", "foreground"):
        errors, plots = {}, {}
        for k in ERROR_KEYS:
            e = np.concatenate([g[mask]["errors"][k] for g in groups])
            e = e[np.isfinite(e)]
            signed, = quantiles(e, (.50,))
            median, p95 = quantiles(np.abs(e), (.50, .95))
            errors[k] = dict(pixels=int(e.size), p50=median, p95=p95, signed_p50=signed)
        for k in PLOT_KEYS:
            v = np.concatenate([g[mask]["distributions"][k] for g in groups])
            if not np.isfinite(v).all():
                raise FloatingPointError(f"Invalid diagnostic distribution {mask}/{k}")
            p10, p50, p90 = quantiles(v, (.10, .50, .90))
            plots[k] = dict(p10=p10, p50=p50, p90=p90)
            if k == "attenuation":
                plots[k]["counts"] = {str(a): int((v==a).sum()) for a in (.05, .20, 1.)}
        result[mask] = dict(pixels=errors["prediction"]["pixels"], errors=errors, distributions=plots)
    return result


def evaluate_prior_residual(src, weights, output, fixed):
    units = {"dev"+unit.name.split("-")[1]: unit for unit in library_unit_paths("dev")}
    frames = sorted({(slug, frame) for slug, frame, _ in fixed if slug in TARGETS})
    cases = [(slug, units[slug], frame) for slug, frame in frames]
    if set(TARGETS) != {slug for slug, _, _ in cases}:
        raise ValueError("A fixed thin-structure diagnostic case could not be identified")
    (output/"prior-residual").mkdir(exist_ok=True)
    start = load_paired_checkpoint(INITIAL, "ema").eval()
    initial = [predictions(start, inputs(unit, frame)) for _, unit, frame in cases]
    del start
    net = checkpoint_model(src, weights).eval()
    blocks, thin = {}, []
    for case, base in zip(cases, initial):
        name, groups, row = scene_data(net, case, base, output)
        blocks[name] = dict(row, regions=metrics([groups]))
        if case[0] in THIN_SCENES:
            thin.append(groups)
    return dict(definition=DEFINITION, frames=blocks, thin_pooled=metrics(thin))


def diagnostic_markdown(report, *, per_frame=True):
    sections = {"dev003/004/005 合并": report["thin_pooled"]}
    if per_frame:
        sections.update({key: row["regions"] for key, row in report["frames"].items()})
    rows = ["## 固定帧细结构深度诊断", "", report["definition"], ""]
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"
    for scene, groups in sections.items():
        rows.extend([f"### {scene}", "",
                     "| 区域 | 输入或预测 | 有效像素 | 绝对逆深度误差 p50 / p95 | 有符号中位数 |",
                     "|---|---|---:|---:|---:|"])
        for region, block in groups.items():
            label = "GT 导入前景" if scene.startswith("dev008") and region == "foreground" else LABELS[region]
            for key, r in block["errors"].items():
                rows.append(f"| {label} | {LABELS[key]} | {r['pixels']:,} | "
                            f"{fmt(r['p50'])} / {fmt(r['p95'])} | {fmt(r['signed_p50'])} |")
        rows.extend(["", "| 区域 | 像素 | 分布 | p10 / p50 / p90 | 衰减系数 0.05 / 0.20 / 1 的像素数 |",
                     "|---|---:|---|---:|---:|"])
        for region, block in groups.items():
            label = "GT 导入前景" if scene.startswith("dev008") and region == "foreground" else LABELS[region]
            for key, r in block["distributions"].items():
                values = " / ".join(fmt(r[q]) for q in ("p10", "p50", "p90"))
                counts = " / ".join(f"{n:,}" for n in r["counts"].values()) if "counts" in r else "—"
                rows.append(f"| {label} | {block['pixels']:,} | {LABELS[key]} | {values} | {counts} |")
        rows.append("")
    return rows
