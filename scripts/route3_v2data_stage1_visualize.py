"""Render a shared P2 gallery of the old and expanded Stage 1 systems."""
import argparse
from datetime import datetime
from html import escape
from pathlib import Path
import time
from urllib.parse import quote
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import route3_ovg_data as ovg
from route3_g3_r14_train import learned_hidden
from route3_ovg_1b_common import CONFIG
from route3_ovg_1b_optical_data import geometry_root
from route3_ovg_1b_train import load_model
from route3_ovg_1c import aligned_sample
from route3_ovg_1c_evaluate import case_frames
from route3_ovg_footprint import query_net
from route3_ovg_refocus_query import original_views
from route3_s2a_data import query_input_512, query_memory
from route3_s2a_evaluate import masks as legacy_masks
from route3_upper_bound_evaluate import display
from route3_v2data_B_batch import GIB, ROOT, reserve, sha256, write_json
from route3_v2data_B_inputs import library_unit_paths, observation
from route3_v2data_B_plan import load
from route3_v2data_stage1_evaluate import (
    LEGACY_CHECKPOINT, QUERY as OLD_QUERY, FrameQueries, development_cases, load_paired_checkpoint,
    near_subject, new_regions, p2_focus, pooled, region_errors, view_from_observation,
)
from route3_v2data_stage2_eval import f2_for, k16, optical_target, oracle_hidden, peeled_truth, render_offsets
from route3_v2data_stage2_train import INITIAL, QUERY, configuration, fixed_inputs
from src.aperture_renderer import aperture_disk
from src.dual_layer import depth_band
from src.footprint_order import apply_footprint_order
from src.layered_aperture import merge_layers
from src.ovg_optics import fixed_visible_surfels, raster


OUT = ROOT.parent/"route3-cache/visual-20261123"
ZONE = ZoneInfo("Asia/Shanghai")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
NAMES = {
    "input": "输入 RGB", "GT": "GT · Cycles", "S1": "S1 · 旧 1c 完整系统",
    "S2": "S2 · 新 1b + 旧 Query", "S3": "S3 · 新 1b + 新 Query",
    "S4": "S4 · 新 1b + F2", "S5": "S5 · 新 1b + GT 隐藏层",
}
LEGACY = ("122104", "122105", "122107")
PRIORITY = ("dev001", "dev006", "dev007", "dev000", "dev011", "dev005")
WHY = {
    "122104": "原评测场景，大面积遮挡的跨版本连续性对照",
    "122105": "原评测场景，多层遮挡的跨版本连续性对照",
    "122107": "原评测场景，大面积前景的跨版本连续性对照",
    "dev001": "指定重点场景，补首尾帧查看时序变化",
    "dev006": "2a 零光圈误差集中场景，核对阶段 1 的观感",
    "dev007": "指定重点场景，补首尾帧查看时序变化",
    "dev000": "阶段 2 第 0 步连续性核对场景",
    "dev011": "指定重点场景，补首尾帧查看时序变化",
    "dev005": "阶段 1 远焦合并 SSE 的主要贡献场景",
}
TILE, GAP, LEFT = 272, 10, 20
NEAR_MASK, FAR_MASK = (67, 239, 220), (246, 86, 193)
ERROR_SCALE, TRANS_SCALE = cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS


def stamp():
    return datetime.now(ZONE).isoformat(timespec="seconds")


def planned_frames():
    old = {}
    for name, frame, entry, _, source, _, manifest, items, folder in case_frames():
        cases = [dict(row["case"], focus_m=row["case"]["nominal_focus_m"],
                      focus_index=int(row["case"]["nominal_focus_m"] != 1.4))
                 for row in items if row["case"]["protocol"] == "P2" and row["case"]["fstop"] == 1.4]
        old[name.split("-")[2], frame] = (name, None, entry, source, manifest, folder, cases)
    units = {"dev"+p.name.split("-")[1]: p for p in library_unit_paths("dev")}
    order = (*LEGACY, *PRIORITY, *sorted(units.keys()-set(PRIORITY)))
    number = 0
    for short in order:
        for frame in (1, 30, 60) if short in WHY else (30,):
            if short in LEGACY:
                name, unit, entry, source, manifest, folder, cases = old[short, frame]
            else:
                unit = units[short]
                name, entry, source, manifest, folder = unit.name, None, None, None, None
                cases = [c for c in development_cases(unit) if c["frame"] == frame and c["fstop"] == 1.4]
            cases = sorted(cases, key=lambda c: c["focus_index"])
            if [c["focus_index"] for c in cases] != [0, 1]:
                raise ValueError(f"Expected a near/far pair for {name} f{frame:03d}")
            files = []
            for case, label in zip(cases, ("near", "far")):
                number += 1
                files.append(dict(file=f"{number:03d}_{short}_f{frame:03d}_{label}.png", short=short,
                                  scene=name, frame=frame, focus=label, case=case["id"],
                                  why=WHY.get(short, "新开发集全覆盖，固定中间帧")))
            yield dict(name=name, unit=unit, entry=entry, source=source, manifest=manifest, folder=folder,
                       frame=frame, cases=cases, files=files)


def prepare_geometry(net, x, views, memory, frame, cases, subjects, cache=None):
    if cache is None:
        prediction = net(x, protect_residual=True, render_heads=False)
        packed = None
    else:
        # Reuse the original 1c EMA output and its matching candidate pack.
        packed = ovg.load(cache)
        prediction = dict(depth1=x["K"].new_tensor(packed["target_depth"])[None])
    depth = prediction["depth1"][0]
    z = depth.cpu().numpy()
    K = x["K"][0]
    focuses = [p2_focus(z, subject) for subject in subjects]
    bands = [depth_band(z, float(K[0, 0]), f, .05/(2*case["fstop"]))[0] if f is not None else None
             for case, f in zip(cases, focuses)]
    active = [band for band in bands if band is not None]
    store = None
    if active:
        if packed is None:
            anchor = next(row for row in views if row["frame"] == frame)
            packed, _ = query_input_512(views, dict(anchor, depth=z), np.logical_or.reduce(active),
                                       8, 262144, memory)
        store = FrameQueries(views, packed)
    return dict(depth=depth, prediction=prediction, visible=fixed_visible_surfels(prediction, x),
                focuses=focuses, bands=bands, store=store)


def query_layer(query, rig, K, frame, case, index):
    focus = rig["focuses"][index]
    band = rig["bands"][index]
    row = dict(frame=frame, input="gallery-frame", case=dict(case, focus_m=focus))
    hidden = learned_hidden(query, rig["store"], row, dict(size=512), K, band)
    return apply_footprint_order(hidden, rig["depth"], K)[0]


def aperture_remaining(visible, K, focus, offsets):
    parts = (raster(visible, K, focus, part, 512, 512, return_alpha=True)[1].sum(0)
             for part in offsets.split(4))
    return (1-sum(parts)/len(offsets)).clamp(0, 1).cpu().numpy()


def largest_error_crop(rgb, target, mask):
    err = np.square(rgb.astype(np.float64)-target).sum(-1)*mask
    ii = cv2.integral(err)
    size = 96
    total = ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]
    y, x = np.unravel_index(total.argmax(), total.shape)
    return [int(x), int(y), int(x)+size, int(y)+size]


def scene_views(plan):
    unit = plan["unit"]
    if unit:
        return [view_from_observation(observation(unit, f, aligned=True), f) for f in ovg.TRAIN_FRAMES], None
    return original_views(plan["source"], plan["manifest"]), query_memory(plan["source"], plan["manifest"]["frames"])


def frame_reference(plan, sample=None):
    unit, frame, cases = (plan[k] for k in ("unit", "frame", "cases"))
    item = ("B", plan["name"], ovg.TRAIN_FRAMES.index(frame))
    x, y = aligned_sample(item, "cuda") if sample is None else sample
    K = x["K"][0].cpu().numpy()
    ref = ovg.b_edge_reference(item, y)
    subjects = [near_subject(ref["gt"], ref["valid"], K, c["focus_m"]) for c in cases]
    if unit:
        masks = [new_regions(unit, c, ref["gt"], ref["valid"], K) for c in cases]
    else:
        masks = [legacy_masks(geometry_root(plan["name"]), c) for c in cases]
    return x, ref, (masks[0]["in_focus"], masks[1]["recoverable"]), subjects


@torch.inference_mode()
def frame_inputs(plan, views, memory, models, old_blank):
    x, ref, masks, subjects = frame_reference(plan)
    unit, frame, cases = (plan[k] for k in ("unit", "frame", "cases"))
    old = None
    if not (unit and old_blank):
        cache = plan["folder"]/"query/1c-aligned-EMA.npz" if plan["folder"] else None
        old = prepare_geometry(models["S1"], x, views, memory, frame, cases, subjects, cache=cache)
    new = prepare_geometry(models["S3"], x, views, memory, frame, cases, subjects)
    return x, ref, masks, old, new


@torch.inference_mode()
def render_case(plan, state, queries, index, old_blank):
    x, ref, masks, old, new = state
    case, K, frame = plan["cases"][index], x["K"][0], plan["frame"]
    unit, name = plan["unit"], plan["name"]
    new_focus = new["focuses"][index]
    if new_focus is None:
        raise ValueError("P2 无有效名义主体")
    mask = masks[index]
    if not mask.any():
        raise ValueError("评分掩码为空")
    offsets = aperture_disk(CONFIG["native_eval_rays"], .05/(2*case["fstop"]), device=K.device, dtype=K.dtype)
    gt = optical_target(unit, name, case)
    fill = f2_for(unit, plan["entry"], frame, x, new_focus, offsets)
    focuses = {"GT": float(case["focus_m"]), "input": None}
    images = {"input": x["rgb"][0].permute(1, 2, 0)}
    if old is not None:
        focus = old["focuses"][index]
        if focus is None:
            raise ValueError("S1 的 P2 无有效主体")
        hidden = query_layer(queries["old"], old, K, frame, case, index)
        old_fill = f2_for(unit, plan["entry"], frame, x, focus, offsets)
        images["S1"] = render_offsets(merge_layers(old["visible"], hidden), K, focus, offsets, old_fill)
        focuses["S1"] = focus
    else:
        focuses["S1"] = None
    for key, version in (("S2", "old"), ("S3", "new")):
        hidden = query_layer(queries[version], new, K, frame, case, index)
        images[key] = render_offsets(merge_layers(new["visible"], hidden), K, new_focus, offsets, fill)
    images["S4"] = render_offsets(new["visible"], K, new_focus, offsets, fill)
    hidden, _ = oracle_hidden(peeled_truth(unit, name, frame), ref, new["depth"], K, new_focus, case["fstop"])
    images["S5"] = render_offsets(merge_layers(new["visible"], hidden), K, new_focus, offsets, fill)
    focuses.update({k: new_focus for k in ("S2", "S3", "S4", "S5")})
    scores = {k: pooled([region_errors(rgb, gt, mask)]) for k, rgb in images.items()}
    scores["GT"] = pooled([dict(pixels=int(mask.sum()), sse=0.)])
    if old is None:
        scores["S1"] = dict(pixels=0, sse=None, psnr_db=None)
    rgb = {key: display(v.cpu().numpy()) for key, v in images.items()}
    rgb["GT"] = gt
    box = largest_error_crop(rgb["S3"], gt, mask)
    lens = k16(unit, name, frame, case)["aperture_xy"]
    if lens.shape != (16, 2):
        raise ValueError(f"Expected 16 K16 pupil offsets, got {lens.shape}")
    remaining = aperture_remaining(new["visible"], K, new_focus, K.new_tensor(lens))
    row = dict(plan["files"][index], region=("near_in_focus", "far_recoverable")[index],
               crop_xyxy=box, crop_support=int(mask[box[1]:box[3], box[0]:box[2]].sum()),
               region_scores=scores, actual_focus_m=focuses, S1_blank=old_blank if old is None else None)
    return row, rgb, remaining


def font(size):
    return ImageFont.truetype(FONT, size, index=2)


def rgb_image(data):
    return Image.fromarray(np.rint(np.clip(data, 0, 1)*255).astype(np.uint8))


def scalar_image(value, maximum, color):
    values = np.rint(255*np.clip(value/maximum, 0, 1)).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.applyColorMap(values, color), cv2.COLOR_BGR2RGB))


def score_text(score):
    return "—" if score["psnr_db"] is None else f'{score["psnr_db"]:.2f}'


def draw_panel(path, row, images, masks, transmission, *, columns=NAMES, tile=TILE,
               diagnostic_subtitle="K16 光线平均 · 0–1", diagnostic_images=None, cached_columns=None):
    title_y, text_y, top_y = 14, 57, 105
    zoom_y, error_y = top_y+tile+42, top_y+2*tile+84
    diagnostics = transmission if diagnostic_images is None else diagnostic_images
    count = len(columns)+len(diagnostics)
    width = 2*LEFT+count*tile+(count-1)*GAP
    pic = Image.new("RGB", (width, error_y+tile+20), (18, 23, 31))
    draw = ImageDraw.Draw(pic)
    text = (229, 234, 242)
    muted = (164, 178, 193)
    topic = "近焦焦内" if row["focus"] == "near" else "远焦可恢复"
    draw.text((LEFT, title_y), f'{row["short"]}  /  f{row["frame"]:03d}  /  {topic}  ·  P2 f/1.4',
              fill=text, font=font(30))
    legend_x = width-648
    draw.text((legend_x, 24), "青色 · 近焦焦内", fill=NEAR_MASK, font=font(19))
    draw.text((legend_x+198, 24), "粉色 · 远焦可恢复", fill=FAR_MASK, font=font(19))
    draw.text((legend_x+409, 24), "黄色 · 共用裁剪框", fill=(255, 220, 62), font=font(19))
    for i, (key, label) in enumerate(columns.items()):
        x = LEFT+i*(tile+GAP)
        draw.text((x, text_y), label, fill=text, font=font(19))
        score = score_text(row["region_scores"][key])
        focus = row["actual_focus_m"][key]
        distance = "" if focus is None else f"   ·   {focus:.3f} m"
        draw.text((x, text_y+22), f"{score} dB{distance}" if score != "—" else "无结果",
                  fill=muted, font=font(16))
        if cached_columns and key in cached_columns:
            for y, contents in zip((top_y,zoom_y,error_y),cached_columns[key]):
                pic.paste(contents.resize((tile,tile),Image.Resampling.LANCZOS),(x,y))
            continue
        if key not in images:
            for y in (top_y, zoom_y, error_y):
                draw.rectangle((x, y, x+tile-1, y+tile-1), fill=(30, 38, 48))
            continue
        array = images[key]
        whole = rgb_image(array)
        if key == "GT":
            outlines = np.rint(array*255).astype(np.uint8)
            # Draw contours at display size so fine contours survive downsampling.
            outlines = cv2.resize(outlines, (tile, tile), interpolation=cv2.INTER_AREA)
            for mask, color in zip(masks, (NEAR_MASK, FAR_MASK)):
                small = cv2.resize(mask.astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST_EXACT)
                contours, _ = cv2.findContours(small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(outlines, contours, -1, color, 1)
            whole = Image.fromarray(outlines)
            frame = ImageDraw.Draw(whole)
            frame.rectangle(tuple(round(v*tile/512) for v in row["crop_xyxy"]), outline=(255, 220, 62), width=2)
        else:
            whole = whole.resize((tile, tile), Image.Resampling.LANCZOS)
        pic.paste(whole, (x, top_y))
        pic.paste(rgb_image(array).crop(row["crop_xyxy"]).resize((tile, tile), Image.Resampling.NEAREST),
                  (x, zoom_y))
        err = np.abs(array-images["GT"]).mean(-1)
        pic.paste(scalar_image(err, .25, ERROR_SCALE).resize((tile, tile), Image.Resampling.NEAREST),
                  (x, error_y))
    for i, (title, values) in enumerate(diagnostics.items(), len(columns)):
        x = LEFT+i*(tile+GAP)
        draw.text((x, text_y), title, fill=text, font=font(19))
        draw.text((x, text_y+22), diagnostic_subtitle, fill=muted, font=font(16))
        diagnostic = scalar_image(values, 1., TRANS_SCALE) if diagnostic_images is None else rgb_image(values)
        pic.paste(diagnostic.resize((tile, tile), Image.Resampling.NEAREST), (x, top_y))
    x = LEFT+min(len(columns),count-1)*(tile+GAP)
    bars=[(error_y+12, "显示 RGB 平均绝对误差", .25, ERROR_SCALE)]
    if transmission:
        bars.insert(0,(top_y+tile+8, "", 1., TRANS_SCALE))
    for y, label, maximum, color in bars:
        if label:
            draw.text((x, y), label, fill=text, font=font(18))
            y += 30
        bar = np.broadcast_to(np.linspace(0, maximum, tile), (12, tile))
        pic.paste(scalar_image(bar, maximum, color), (x, y))
        draw.text((x, y+13), "0", fill=muted, font=font(16))
        draw.text((x+tile, y+13), f"{maximum:g}", fill=muted, font=font(16), anchor="rt")
    draw.text((LEFT, zoom_y-30), "同框放大 · 96 × 96 px", fill=text, font=font(19))
    draw.text((LEFT, error_y-30), "整幅误差热图 · 共用色标 0–0.25", fill=text, font=font(19))
    temporary = path.with_suffix(".part")
    pic.save(temporary, format="PNG", compress_level=3)
    temporary.replace(path)


def overview(out, rows, focus, *, columns=NAMES, keys=("GT", "S1", "S3", "S4", "S5"), tile=TILE):
    selected = [row for row in rows if row["frame"] == 30 and row["focus"] == focus]
    size, step, left, top = 128, 140, 150, 82
    pic = Image.new("RGB", (left+len(keys)*step+12, top+len(selected)*step+12), (18, 23, 31))
    draw = ImageDraw.Draw(pic)
    label = "近焦" if focus == "near" else "远焦"
    draw.text((16, 8), f"{label}总览 · f030 · P2 f/1.4", font=font(27), fill="white")
    for i, key in enumerate(keys):
        draw.text((left+i*step, 49), key if key != "GT" else "GT · Cycles", font=font(19), fill="white")
    for i, row in enumerate(selected):
        y = top+i*step
        draw.text((16, y+32), row["short"], font=font(21), fill="white")
        if row.get("error"):
            draw.text((left, y+32), row["error"], font=font(17), fill=(164, 178, 193))
            continue
        with Image.open(out/row["file"]) as image:
            for j, key in enumerate(keys):
                x = LEFT+list(columns).index(key)*(tile+GAP)
                zoom_y = 105+tile+42
                crop = image.crop((x, zoom_y, x+tile, zoom_y+tile))
                pic.paste(crop.resize((size, size), Image.Resampling.NEAREST), (left+j*step, y))
    name = f"000_总览_{label}.png"
    pic.save(out/name, compress_level=3)
    return name


def indexes(out, rows):
    overviews = [overview(out, rows, focus) for focus in ("near", "far")]
    guide = ("P2，f/1.4。各方法用自身几何对同一 GT 主体重新定焦，F2 来自同一缓存。"
             "S1–S3 用各自 Query，S4 只保留 F2，S5 按阶段 2 H 的足迹次序放置 GT 隐藏层。"
             "原三场景的 S1 沿用原 1c EMA 几何与候选缓存，新开发场景直接用 1c EMA 推理几何。")
    md = ["# 旧 1c 与新 1b 的 P2 观感对照", "", guide, "",
          "所有分数为当前行指定区域的 PSNR（dB）。近焦使用焦内，远焦使用可恢复区。", "",
          "| 文件 | 场景 | 帧 | 焦点 | 选择理由 | S1 | S2 | S3 | S4 | S5 |",
          "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    html = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<title>旧 1c 与新 1b 的 P2 观感对照</title><style>',
            'body{margin:2rem auto;padding:0 1rem;max-width:110rem;background:#12171f;color:#edf2f9;font-family:"Noto Sans CJK SC",sans-serif}',
            '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,28rem),1fr));gap:1.1rem}',
            'figure{margin:0;background:#202a36;border-radius:8px;overflow:hidden}img{display:block;width:100%;height:auto}',
            'figcaption{padding:.75rem}small{color:#bfcbda}a{color:#77e4dc}nav{margin:1rem 0}nav a{margin-right:1rem}',
            '</style><h1>旧 1c 与新 1b 的 P2 观感对照</h1>', f'<p>{guide}</p><nav>',
            '<a href="索引.md">Markdown 索引</a><a href="metrics.json">评分、裁剪框与焦距</a>']
    for name in overviews:
        url = quote(name)
        focus = "近焦" if "近焦" in name else "远焦"
        md.append(f"| [{name}]({url}) | 全部场景 | f030 | {focus} | 同框放大总览 | — | — | — | — | — |")
        html.append(f'<a href="{url}">{focus}总览</a>')
    html.append("</nav><main class='grid'>")
    for row in rows:
        if row.get("error"):
            continue
        name, url = row["file"], quote(row["file"])
        focus = "近焦焦内" if row["focus"] == "near" else "远焦可恢复"
        scores = [score_text(row["region_scores"][f"S{i}"]) for i in range(1, 6)]
        md.append(f'| [{name}]({url}) | {row["scene"]} | f{row["frame"]:03d} | {focus} | '
                  f'{row["why"]} | {" | ".join(scores)} |')
        label = f'{row["short"]} · f{row["frame"]:03d} · {focus}'
        values = "　".join(f"S{i} {v}" for i, v in enumerate(scores, 1))
        html.append(f'<figure><a href="{url}"><img loading="lazy" src="{url}" alt="{label}"></a>'
                    f'<figcaption>{label}<br><small>{values} dB</small></figcaption></figure>')
    html.append("</main>")
    errors = [row for row in rows if row.get("error")]
    md.extend(["", "## 跳过的 case", ""])
    if errors:
        html.append("<h2>跳过的 case</h2><ul>")
        for row in errors:
            error = row["error"].replace("\n", " ")
            md.append(f'- {row["file"]}，{error}')
            html.append(f'<li>{escape(row["file"])}，{escape(error)}</li>')
        html.append("</ul>")
    else:
        md.append("无。")
    (out/"索引.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    (out/"index.html").write_text("\n".join(html)+"\n", encoding="utf-8")


def run(output, pilot, old_blank):
    configuration()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(6*GIB/total, 0)
    fixed_inputs()
    torch.backends.cuda.matmul.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=True)
    path = output/"metrics.json"
    started = time.monotonic()
    plan = list(planned_frames())
    results = load(path) or dict(started_at=stamp(), seconds=0., planned_cases=sum(len(a["files"]) for a in plan),
                                 scope="P2 f/1.4, 64 aperture rays; same per-frame F2 source and fixed GT scoring masks",
                                 S1_geometry="original 1c EMA depth and candidate cache on legacy scenes; direct 1c EMA inference on new development scenes",
                                 crop="96x96, maximizes masked S3 display-RGB SSE; xyxy, exclusive right/bottom",
                                 heatmap="mean absolute display-RGB error, shared 0–0.25 color scale",
                                 transmittance="S3 mean of 1 minus accumulated visible alpha over the 16 K16 pupil offsets",
                                 perfect_psnr_cap_db=120, rows=[])
    paths = dict(old_1c=LEGACY_CHECKPOINT, new_1b=INITIAL, old_Query=OLD_QUERY, new_Query=QUERY)
    receipt = {k: dict(path=str(v), sha256=sha256(v)) for k, v in paths.items()}
    if "checkpoints" in results and results["checkpoints"] != receipt:
        raise ValueError("Existing gallery metrics refer to different checkpoints")
    results.update(checkpoints=receipt, S1_development_blank_reason=old_blank)
    previous = results["seconds"]
    rows = {r["file"]: r for r in results["rows"]}
    models = dict(S1=load_model(LEGACY_CHECKPOINT, "ema").eval(),
                  S3=load_paired_checkpoint(INITIAL, "ema").eval())
    queries = {"old":query_net(OLD_QUERY), "new":query_net(QUERY)}
    scene, views, memory = None, None, None

    def publish():
        results["rows"] = sorted(rows.values(),key=lambda r:r["file"])
        results["seconds"] = previous+time.monotonic()-started
        results["completed_cases"] = sum(not r.get("error") for r in rows.values())
        results["skipped_cases"] = sum(bool(r.get("error")) for r in rows.values())
        results["peak_allocated_MiB"] = max(results.get("peak_allocated_MiB", 0.),
                                            torch.cuda.max_memory_allocated()/2**20)
        write_json(path, results)

    for task in plan:
        if pilot and (task["files"][0]["short"] not in ("122104", "dev000") or task["frame"] != 30):
            continue
        pending = [i for i, item in enumerate(task["files"]) if item["file"] not in rows]
        if not pending:
            continue
        del_state = None
        reserve(output, GIB)
        try:
            if scene != task["name"]:
                views, memory = scene_views(task)
                scene = task["name"]
            del_state = frame_inputs(task, views, memory, models, old_blank)
        except Exception as error:
            for i in pending:
                item = task["files"][i]
                rows[item["file"]] = dict(item, error=f"{type(error).__name__}: {error}")
                print("VISUAL_SKIP", item["file"], rows[item["file"]]["error"], flush=True)
            torch.cuda.empty_cache()
        if del_state is not None:
            for i in pending:
                item = task["files"][i]
                start = time.monotonic()
                try:
                    row, images, remaining = render_case(task, del_state, queries, i, old_blank)
                    draw_panel(output/row["file"], row, images, del_state[2], {"S3 · 可见层剩余透射率": remaining})
                    rows[row["file"]] = dict(row, seconds=time.monotonic()-start)
                    print("VISUAL_DONE", row["file"], round(rows[row["file"]]["seconds"], 2), flush=True)
                except Exception as error:
                    rows[item["file"]] = dict(item, error=f"{type(error).__name__}: {error}")
                    print("VISUAL_SKIP", item["file"], rows[item["file"]]["error"], flush=True)
                    torch.cuda.empty_cache()
        publish()
    if not pilot:
        results["completed_at"] = stamp()
        indexes(output, results["rows"])
        publish()
    print("VISUAL_TOTAL", results["completed_cases"], results["skipped_cases"], round(results["seconds"], 2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--legacy-only-s1-reason")
    args = parser.parse_args()
    run(args.output.resolve(), args.pilot, args.legacy_only_s1_reason)
