"""Summarize the matched E0 replay, focus attribution, and fixed panels."""
from collections import defaultdict
from html import escape
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

import route3_v2data_module2_infer as ref
import route3_v2data_stage1_visualize as gallery
from route3_v2data_B_batch import sha256
from route3_v2data_E0_rerender import (DIAGNOSTIC_ARMS,E0,HIGHLIGHT,METHODS,RUN,STAGE3C,TITLES,
                                      core_summary,stage3c_snapshot,state)
from route3_v2data_E0_reselect import RECORD as SELECTION,RECORDS,fractions
from route3_v2data_E_labels import load,save_json

RECORD=RECORDS/"route3-v2data-E0-rerender-2026-11-26.json"
CONVENTIONS=("P","G","V")
AREAS=("near_in_focus","far_recoverable")
FOCUS_SCENES=("dev003","dev004","dev005")
SPECIAL=("dev001","dev007","dev009","dev010","dev011")


def packs(folder):
    return {str(p.relative_to(folder/"frames")):load(p) for p in sorted((folder/"frames").glob("*/*.json"))}


def case_key(row):
    return row["scene"],row["case"]


def join(include_new):
    sources={name:packs(RUN/name) for name in ("P-base","fixed-base",*(("rest",) if include_new else ()))}
    index=sources["P-base"]
    if len(index)!=54 or any(v.keys()!=index.keys() for v in sources.values()):
        raise ValueError("Expected the same 54 individual frames in every replay phase")
    result=[]
    arms=METHODS if include_new else METHODS[:9]
    for name in index:
        entry={"rows":{},"relative":name}
        for focus in CONVENTIONS:
            group=sources["P-base" if focus=="P" else "fixed-base"][name]["rows"][focus]
            current=[dict(r,methods=dict(r["methods"])) for r in group]
            if include_new:
                extra={case_key(r):r for r in sources["rest"][name]["rows"][focus]}
                for row in current:
                    other=extra[case_key(row)]
                    if row["status"]!=other["status"]:
                        raise ValueError("The reselected geometry changed the evaluation mask")
                    row["methods"].update(other["methods"])
            for row in current:
                if row["status"]=="evaluated" and tuple(row["methods"])!=arms:
                    raise ValueError("Incomplete arm set in a scored case")
            entry["rows"][focus]=current
        result.append(entry)
    return result,arms,sources


def metric(row,arm,region):
    return row[arm][region]["psnr_db"]


def focus_split(summaries,arms):
    analysis={}
    key="far_recoverable"
    for convention in ("G","V"):
        result={}
        for scene,p in summaries["P"]["development"]["per_scene"].items():
            slug=ref.title(scene)
            if slug not in FOCUS_SCENES:
                continue
            q=summaries[convention]["development"]["per_scene"][scene]
            result[slug]={}
            for arm in arms:
                own_p,own_q=(metric(point,arm,key) for point in (p,q))
                base_p,base_q=(metric(point,"V0_pred",key) for point in (p,q))
                rendering=own_q-base_q
                focus=(own_p-own_q)-(base_p-base_q)
                total=own_p-base_p
                if not np.isclose(total,rendering+focus,rtol=0,atol=1e-10):
                    raise ValueError("Focus and rendering deltas do not sum")
                result[slug][arm]=dict(P=own_p,fixed=own_q,paired_total=total,rendering=rendering,
                                       focusing=focus,arm_focus_shift=own_p-own_q,V0_focus_shift=base_p-base_q)
        analysis[convention]=result
    return analysis


def far_split(parts):
    groups=defaultdict(list)
    for p in parts:
        for row in p["rows"]["G"]:
            if row["status"]=="evaluated" and row["focus_index"]==1:
                groups[row["scene"]].append(row)
    result={}
    for scene,rows in groups.items():
        result[scene]={}
        for arm in DIAGNOSTIC_ARMS:
            pieces=[r["methods"][arm]["far_diagnostic"] for r in rows]
            rays=sum(p["rays"] for p in pieces)
            off,on=(ref.pooled([p[k] for p in pieces]) for k in ("off","on"))
            if off["pixels"]!=on["pixels"]:
                raise ValueError("The on/off far sample support differs")
            result[scene][arm]=dict(rays=rays,mean_visible_residual=sum(p["residual_sum"] for p in pieces)/rays if rays else None,
                hidden_hit_ratio=sum(p["hidden_hits"] for p in pieces)/rays if rays else None,
                hidden_weight_ge_0p001_fraction=sum(p["hidden_significant"] for p in pieces)/rays if rays else None,
                F2=off,with_GT=on,GT_minus_F2=on["psnr_db"]-off["psnr_db"] if rays else None)
    return result


def slope_split(summaries,include_new):
    pairs=(("b","V0_pred"),("E_gt","E"),("a","V0_GT"),*((("E_B_gt","E_B"),) if include_new else ()))
    result={}
    for convention in ("P","G"):
        result[convention]={}
        for group in ("development","thin_structure","dev008","legacy"):
            block=summaries[convention][group]
            result[convention][group]={
                f"{high}-{low}":{k:v for k,v in ref.compare(block["per_scene"],low,high,block["methods"],
                                             include_absolute_near=True).items() if k in AREAS}
                for high,low in pairs}
    return result


def subject_split(sources,include_new):
    return {mode:sorted([p["subject"] for p in sources[folder].values() if p["subject"]],key=lambda v:v["frame"])
            for mode,folder in (("E-H2@2k","P-base"),*((("E_B","rest"),) if include_new else ()))}


def rgba_mask(image,mask):
    pixels=np.asarray(image).copy()
    outline=mask&~cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    pixels[outline]=(253,234,188)
    return Image.fromarray(pixels)


def draw_subjects(subjects, *, base_key="E-H2@2k", other_key="E_B"):
    stems=[]
    root=RUN/"figures"
    for index,old in enumerate(subjects[base_key]):
        with np.load(RUN/old["panel_data"]) as p:
            mask=p["subject"]
            gt=p["GT central axial depth (m)"]
            pieces={"OVG ρ₁":p["OVG rho1 (m^-1)"],
                    "GT 中心逆深度":np.divide(1.,gt,out=np.zeros_like(gt),where=gt>0),
                    f"{base_key} ρc":p["E rho_c (m^-1)"]}
        if other_key in subjects:
            b=subjects[other_key][index]
            if b["frame"]!=old["frame"]:
                raise ValueError("Unmatched P2 subject frames")
            with np.load(RUN/b["panel_data"]) as p:
                if not np.array_equal(mask,p["subject"]):
                    raise ValueError("P2 subject mask differs between E candidates")
                pieces[f"{other_key} ρc"]=p["E rho_c (m^-1)"]
        shared=max(float(np.quantile(v,.99)) for v in pieces.values())
        canvas=Image.new("RGB",(292*len(pieces),325),(18,22,32))
        draw=ImageDraw.Draw(canvas)
        draw.text((6,6),f"dev003 f{old['frame']:03d} 远焦 P2 主体 · 白线 · 共同逆深度 0–{shared:.3f} m⁻¹",
                  font=gallery.font(17),fill="white")
        for i,(label,value) in enumerate(pieces.items()):
            draw.text((i*292+4,37),label,font=gallery.font(14),fill="white")
            image=rgba_mask(gallery.scalar_image(value,shared,cv2.COLORMAP_VIRIDIS),mask)
            canvas.paste(image.resize((275,275),Image.Resampling.NEAREST),(i*292+4,58))
        path=root/f"dev003-f{old['frame']:03d}-P2-subject.jpg"
        canvas.save(path,quality=94)
        stems.append(str(path.relative_to(RUN)))
    return stems


def optical_panel(name,focus,index,row,pictures,arms,*,far_f2=False,far_reference=None):
    target=pictures[f"{index}/target"]
    edge=pictures[f"{index}/edge"]
    zone=pictures.get("rod_gap",target)
    truth=pictures[f"{index}/truth"].astype(np.float32)
    base=pictures[f"{focus}/{index}/V0_pred"].astype(np.float32)
    choices={a:pictures[f"{focus}/{index}/{a}"].astype(np.float32) for a in arms}
    witness=max(choices,key=lambda a:float(np.square(choices[a]-base)[target].sum()))
    detail=ref.crop_for(choices[witness],truth,zone)
    boundary=ref.crop_for(choices[witness],truth,edge)
    width=173
    tiles=("整图","杆 / 缝放大","边缘放大","误差 0–0.25","可见层逆深度","撕裂／误连／误断")
    if far_reference is not None:
        label,reference_images,reference_scores=far_reference
        tiles+=tuple(f"{label} {name}" for name in ("整图","同框边缘","误差 0–0.25"))
    canvas=Image.new("RGB",(width*(len(arms)+1),70+(width+18)*len(tiles)),(18,22,32))
    draw=ImageDraw.Draw(canvas)
    lens = "近焦" if index==0 else "远焦，仅 F2" if far_f2 else "远焦"
    if far_reference is not None:
        lens=f"远焦，F2／{label}"
    draw.text((6,4),f"{name} f030 · {focus} · {lens} · 局部评分掩码",
              font=gallery.font(18),fill="white")
    if name=="dev010":
        draw.text((6,28),"近焦无有效主体，本场只列远焦。",font=gallery.font(14),fill="white")
    hi=max(float(np.quantile(pictures[f"depth/{arm}"],.99)) for arm in arms)
    region=AREAS[index]
    for i,arm in enumerate(("Cycles",*arms)):
        x=i*width
        title="Cycles · ∞" if not i else f"{i} {TITLES[i-1]} · "+fmt(ref.score_psnr(row["methods"][arm]["optical"][region]),2)
        if far_reference is not None and i:
            reference=fmt(ref.score_psnr(reference_scores[arm]),2)
            title+=f"／{reference}"
        draw.text((x+3,50),title,font=gallery.font(13),fill="white")
        now=truth if i==0 else choices[arm]
        whole=gallery.rgb_image(now)
        outlined=whole.copy()
        overlay=ImageDraw.Draw(outlined)
        overlay.rectangle(detail,outline=(255,185,65),width=2)
        overlay.rectangle(boundary,outline=(85,208,242),width=2)
        error=gallery.scalar_image(np.abs(now-truth).mean(-1),.25,cv2.COLORMAP_INFERNO)
        depth=gallery.scalar_image(pictures["depth/V0_GT" if not i else f"depth/{arm}"].astype(np.float32),
                                   max(hi,.01),cv2.COLORMAP_VIRIDIS)
        marked=pictures.get(f"tear/{arm}")
        if marked is not None:
            shade=truth*.46
            active=marked.any(-1)
            shade[active]=marked[active]/255.
            marked=gallery.rgb_image(shade).crop(boundary)
        layers=(outlined,whole.crop(detail),whole.crop(boundary),error,depth,marked)
        if far_reference is not None:
            rgb=truth if not i else reference_images[arm].astype(np.float32)
            second=gallery.rgb_image(rgb)
            boxes=second.copy()
            draw2=ImageDraw.Draw(boxes)
            draw2.rectangle(detail,outline=(255,185,65),width=2)
            draw2.rectangle(boundary,outline=(85,208,242),width=2)
            second_error=gallery.scalar_image(np.abs(rgb-truth).mean(-1),.25,cv2.COLORMAP_INFERNO)
            layers+=(boxes,second.crop(boundary),second_error)
        for j,p in enumerate(layers):
            y=70+(width+18)*j
            if i==0:
                draw.text((4,y+1),tiles[j],font=gallery.font(12),fill="white")
            if p is not None:
                canvas.paste(p.resize((width-4,width-4),Image.Resampling.NEAREST if j in (1,2,5) else Image.Resampling.BOX),(x+2,y+16))
    dst=RUN/"figures"/f"{name}-f030-{focus}-focus{index}.jpg"
    canvas.save(dst,quality=93)
    with Image.open(dst) as check:
        check.verify()
    return str(dst.relative_to(RUN))


def fmt(value,n=3):
    return "—" if value is None else f"{value:.{n}f}"


def visual_summary(parts,arms):
    paths=[]
    notes=defaultdict(list)
    for frame in parts:
        key=frame["relative"]
        name=ref.title(frame["rows"]["P"][0]["scene"])
        if name not in HIGHLIGHT or frame["rows"]["P"][0]["frame"]!=30:
            continue
        pictures={}
        for folder in ("P-base","fixed-base",*(("rest",) if len(arms)>9 else ())):
            with np.load(RUN/folder/"frames"/Path(key).with_suffix(".npz")) as packed:
                pictures.update(packed)
        for focus in ("P","G"):
            for row in frame["rows"][focus]:
                index=row["focus_index"]
                if row["status"]!="evaluated":
                    if focus=="P":
                        notes[name].append(dict(status="unscored",region="近焦",reason="无有效 P2 主体"))
                    continue
                path=optical_panel(name,focus,index,row,pictures,arms)
                paths.append(path)
                region=AREAS[index]
                comparisons=[]
                for arm in arms[1:]:
                    for m in (region,"boundary_2px"):
                        before,after=(ref.score_psnr(row["methods"][a]["optical"][m]) for a in ("V0_pred",arm))
                        if before is not None and after is not None:
                            comparisons.append((after-before,arm,m))
                if comparisons:
                    delta,arm,region=max(comparisons,key=lambda p:abs(p[0]))
                    notes[name].append(dict(convention=focus,focus_index=index,arm=arm,detail=region,
                                            score_delta_db=delta,panel=path))
    return paths,dict(notes)


def draw_main_table(summaries,arms):
    columns=("档 / 方法","P 近焦","P 远焦","G 近焦","G 远焦","V 近焦","V 远焦")
    w,h=162,36
    img=Image.new("RGB",(w*len(columns)+82,(len(arms)+2)*h+35),(18,22,32))
    draw=ImageDraw.Draw(img)
    draw.text((9,5),"新开发集 · 合并 PSNR / 相对 V0 逐场景配对中位 · dB",font=gallery.font(22),fill="white")
    for i,k in enumerate(columns):
        draw.text((i*w+(82 if i else 0)+9,37),k,font=gallery.font(16),fill="white")
    for j,arm in enumerate(arms):
        y=67+j*h
        if j%2==0:
            draw.rectangle((0,y-3,img.width-1,y+h-4),fill=(36,43,55))
        draw.text((9,y),f"{j+1} {TITLES[j]}",font=gallery.font(15),fill="white")
        for i,(focus,area) in enumerate((f,a) for f in CONVENTIONS for a in AREAS):
            g=summaries[focus]["development"]
            value=g["methods"][arm][area]["psnr_db"]
            delta=g["paired"].get(arm,{}).get(area,{}).get("median_db")
            draw.text(((i+1)*w+90,y),f"{fmt(value,2)} / {fmt(delta,2)}",font=gallery.font(16),fill="white")
    path=RUN/"figures"/"开发集总表.png"
    img.save(path)
    return str(path.relative_to(RUN))


def stage3c_overlap():
    before=load(RUN/"3c-before.json")
    snapshots={k:load(RUN/f"3c-{k}.json") for k in ("P-start","fixed-start","rest-start","render-end")}
    rows={}
    for line in (STAGE3C/"steps.jsonl").read_text().splitlines():
        try:
            item=json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[item["step"]]=item["elapsed"]
    step0=before["step"]
    baseline=max(1,step0-500)
    early=(rows[step0]-rows[baseline])/(step0-baseline)
    points={"tear_start":dict(progress=before),**snapshots}
    pairs={}
    old=None
    for k,entry in points.items():
        if not entry or "step" not in entry.get("progress",{}):
            continue
        a=entry["progress"]["step"]
        if old and a>old[1]:
            heading,start=old
            pairs[f"{heading}_to_{k}"]=dict(steps=a-start,mean_train_seconds_per_step=(rows[a]-rows[start])/(a-start))
        old=k,a
    return dict(before_window=[baseline,step0],before_mean_train_seconds_per_step=early,
                snapshots=points,during=pairs,comparison="Observed 3c train time; official checkpoint-evaluation pauses are outside its accumulated train timer.")


def summarize_candidate():
    archived=load(SELECTION)
    chosen=archived["selected"]
    mode,step=chosen["mode"],chosen["step"]
    if (mode,step)==("E-H2",2000):
        return archived
    directory=E0/mode
    depth_file=directory/f"evaluation-{step:06d}-depth/report.json"
    depth=load(depth_file)
    tear_file=directory/f"tear-{step:06d}/training.json"
    tear=load(tear_file)
    if not depth or not tear:
        raise FileNotFoundError("Missing the reselected E_B depth or tear result")
    guard=depth["guards"]["development"]
    limits=load(RECORDS/"route3-v2data-stage3-config-2026-11-24.json")["evaluation"]["relative_geometry_limits"]
    values={k:dict(OVG=guard["OVG"][k],E=guard["E"][k],relative_change=guard["E"][k]/guard["OVG"][k]-1,
                   limit=limit,passed=guard["E"][k]<=guard["OVG"][k]*(1+limit)) for k,limit in limits.items()}
    d=depth["cohorts"]["development"]
    archived["development_depth"]=dict(errors=d["errors"],normal_selected_back=fractions(d,"normal","selected_back"),
                                       guards=values,panels=depth["panels"],source_sha256=sha256(depth_file))
    archived["retrained_tear"]=dict(selected=tear["selected"]["step"],training_steps=tear["config"]["steps"],
                                   selected_assessment=tear["selected"]["assessment"]["at_half"],
                                   rank=tear["selected"]["rank"],protocol=tear["config"],source_sha256=sha256(tear_file))
    save_json(SELECTION,archived)
    return archived


def make_markdown(record,candidate):
    arms=tuple(record["methods"])
    labels=dict(zip(METHODS,TITLES))
    lines=["# E0 统一重渲与远焦拆分","",
           "P 使用各档的 P2 焦距，G 使用 GT 中心几何焦距，V 使用 V0 预测几何焦距。PSNR 合并按 SSE ÷ (3 × 像素数)。","",
           "## 规则 B 重选","",
           f"E_B 为 {candidate['selected']['mode']} {candidate['selected']['step']:,} 步 EMA，"
           f"杆 p50 比门槛低 {candidate['selected_margin_percent']:.2f}%。","",
           "| 配置 | EMA 步数 | 期 | 杆 p50 / p95 | 缝 p50 | NLL | 杆选前，有效／无效 | 正常选后，有效／无效 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in candidate["candidates"]:
        def fraction(key):
            values=c[key]
            return " / ".join("—" if values[k] is None else f"{values[k]:.1%}" for k in ("mv_valid","mv_invalid"))
        lines.append(f"| {c['mode']} | {c['step']} | {c['phase']} | {fmt(c['rod_p50'],5)} / {fmt(c['rod_p95'],5)} | "
                     f"{fmt(c['gap_p50'],5)} | {fmt(c['nll'])} | {fraction('rod_selected_front')} | "
                     f"{fraction('normal_selected_back')} |")
    if "development_depth" in candidate:
        d=candidate["development_depth"]
        lines.extend(["","### E_B 新开发集深度与几何护栏","",
                      "| 区域 | OVG p50 / p95 | E_B p50 / p95 |","|---|---:|---:|"])
        for k in ("rod","gap","normal","boundary"):
            v,e=(d["errors"][f"{k}/{s}"] for s in ("OVG","decoded"))
            lines.append(f"| {k} | {fmt(v['p50'])} / {fmt(v['p95'])} | {fmt(e['p50'])} / {fmt(e['p95'])} |")
        valid,invalid=(d["normal_selected_back"][k] for k in ("mv_valid","mv_invalid"))
        lines.extend(["",f"正常区选后分量，多视角有效 {valid:.1%}，无效 {invalid:.1%}。","",
                      "| 护栏 | OVG | E_B | 变化 | 是否通过 |","|---|---:|---:|---:|---|"])
        for k,v in d["guards"].items():
            lines.append(f"| {k} | {fmt(v['OVG'],5)} | {fmt(v['E'],5)} | {v['relative_change']:+.2%} | "
                         f"{'通过' if v['passed'] else '超限'} |")
        t=candidate["retrained_tear"]
        lines.extend(["",f"撕裂头训练 {t['training_steps']:,} 步，留出集选中第 {t['selected']:,} 步 EMA。"])
    lines.extend(["","## 合并画质与逐场景配对",""])
    for focus in CONVENTIONS:
        for scope,summary in record["results"][focus].items():
            lines.extend([f"### {focus} · {scope}","",
                "| 档 | 近焦 | 远焦 | 边界 | 全图 | 近／远配对中位 | 近焦 GT−F2 | 零光圈可见／合成 | T≥0.1 光线／正向 SSE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for i,arm in enumerate(arms,1):
                v=summary["methods"][arm]
                q=summary["paired"].get(arm,{})
                columns=[fmt(v[key]["psnr_db"]) for key in ref.REGIONS]
                columns.append(" / ".join(fmt(q.get(k,{}).get("median_db")) for k in AREAS))
                columns.append(fmt(v["near_gt_minus_f2"]["psnr_db"]))
                columns.append(" / ".join(f'{v[k]["passed_cases"]}/{v[k]["scored_cases"]}' for k in ref.FLOORS))
                bins=v["transmission"]["positive_sse_bins"]
                high=v["transmission"]["high_fraction"]
                columns.append(("—" if high is None else f"{high:.2%}")+" / "+fmt(bins.get("T>=0.1",0.)))
                lines.append("| "+" | ".join([f"{i} {labels[arm]}",*columns])+" |")
            lines.append("")
    lines.extend(["## 远焦固定焦距拆分","",
                  "总变化是 P 下相对 V0 的差。渲染项为同一固定焦距下的配对差，焦距项为两档从固定焦距回到各自 P2 焦距的差值之差。","",
                  "| 固定焦距 | 场景 | 档 | 总变化 | 渲染项 | 焦距项 |","|---|---|---|---:|---:|---:|"])
    for focus,scenes in record["far_focus_vs_rendering"].items():
        for scene,group in scenes.items():
            for arm,v in group.items():
                lines.append(f"| {focus} | {scene} | {labels[arm]} | {fmt(v['paired_total'])} | {fmt(v['rendering'])} | {fmt(v['focusing'])} |")
    lines.extend(["","## 可见层残余与隐藏层命中（G，K64）","",
        "| 场景 | 档 | 光线数 | 可见层平均残余透射 | 隐藏层命中 | F2 dB | GT−F2 dB |",
        "|---|---|---:|---:|---:|---:|---:|"])
    for focus_scene in (True,False):
        for scene,view in record["far_visible_hidden"]["by_scene"].items():
            slug=ref.title(scene)
            if (slug in SPECIAL)!=focus_scene:
                continue
            for arm,v in view.items():
                lines.append(f"| {slug}{'（重点）' if focus_scene else ''} | {labels[arm]} | {v['rays']} | "
                             f"{fmt(v['mean_visible_residual'],6)} | {fmt(v['hidden_hit_ratio'],6)} | "
                             f"{fmt(v['F2']['psnr_db'])} | {fmt(v['GT_minus_F2'])} |")
    lines.extend(["","## 同几何 GT 撕裂插值增量","",
                  "| 焦距 | 子集 | 增量 | 区域 | 合并差 | 配对中位 | 正／负 |","|---|---|---|---|---:|---:|---:|"])
    for focus,scopes in record["GT_tear_interpolation"].items():
        for scope,pairs in scopes.items():
            for pair,area in pairs.items():
                for region,v in area.items():
                    lines.append(f"| {focus} | {scope} | {pair} | {region} | {fmt(v['pooled_delta_db'])} | "
                                 f"{fmt(v['median_db'])} | {v['positive']}/{v['negative']} |")
    lines.extend(["","## dev003 远焦 P2 主体","",
                  "| 模型 | 帧 | 主体像素 | OVG ρ₁ | E ρc | GT 中心深度 m | E 深度 m | 前／后分量 | ρ₂／DS／三角化 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for mode,probes in record["dev003_subject"].items():
        for p in probes:
            m=p["medians"]
            vals=p["beta_source"]
            lines.append(f"| {mode} | {p['frame']} | {p['pixels']} | {fmt(m['OVG rho1 (m^-1)'])} | "
                f"{fmt(m['E rho_c (m^-1)'])} | {fmt(m['GT central axial depth (m)'])} | "
                f"{fmt(m['E axial depth (m)'])} | {p['selected_front']}/{p['selected_back']} | "
                f"{vals['rho2']}/{vals['DepthSplat']}/{vals['triangulation']} |")
    lines.extend(["","## 固定面板观察","",
                  "差异以评分掩码内相对同焦距 V0 的最大绝对 PSNR 差标示。橙框为杆／缝，蓝框为边缘。撕裂图中绿线为撕裂，红、蓝分别为误连、误断，黄线标记与预测深度台阶相关的配对。",""])
    for scene,notes in record["panel_observations"].items():
        active=[r for r in notes if r.get("panel")]
        if active:
            top=max(active,key=lambda r:abs(r["score_delta_db"]))
            lines.append(f"- {scene}。最大区域差异出现在 {top['convention']} 的 {labels[top['arm']]}，"
                         f"{top['detail']} 相对 V0 为 {top['score_delta_db']:+.3f} dB。 [面板]({top['panel']})")
        if any(r.get("status")=="unscored" for r in notes):
            lines.append(f"- {scene}。近焦无有效主体，保留远焦面板。")
    return "\n".join(lines)+"\n"


def summary():
    info=load(RUN/"dispatch.json")
    if info["phase"] not in ("ready-to-summarize","complete"):
        raise RuntimeError("Replays must finish their fixed-focus safeguards before collection")
    include_new=info["include_new"]
    parts,arms,sources=join(include_new)
    tables={k:core_summary(parts,arms,k) for k in CONVENTIONS}
    for focus,groups in tables.items():
        if (groups["development"]["scored"],groups["legacy"]["scored"])!=(87,18):
            raise ValueError(f"Unexpected {focus} development/legacy coverage")
    root=RUN/"figures"
    root.mkdir(parents=True,exist_ok=True)
    reports=summarize_candidate()
    subjects=subject_split(sources,include_new)
    if any([p["frame"] for p in rows]!=[1,30,60] for rows in subjects.values()):
        raise ValueError("The dev003 subject needs exactly three frames for each checkpoint")
    detail,notes=visual_summary(parts,arms)
    if len(detail)!=38:
        raise ValueError(f"Expected 38 scored P/G optical figures, received {len(detail)}")
    subject_panels=draw_subjects(subjects)
    overall=draw_main_table(tables,arms)
    result=dict(methods={a:dict(arm=i,title=TITLES[i-1]) for i,a in enumerate(arms,1)},
                revised_E_B=reports["selected"],test_cases_used=0,checked_P=load(RUN/"P-self-check.json"),
                paired_baseline="V0_pred in the same focus convention",
                rendering="Uniform corrected FP16 pathway; K64 rendering, K16 nominal-focus near transmission diagnostic. "
                          "Identical numerical focal distances reuse a render inside the same run.",
                results={convention:{k:{a:b for a,b in g.items() if a!="per_scene"} for k,g in table.items()}
                         for convention,table in tables.items()},
                far_focus_vs_rendering=focus_split(tables,[a for a in ("E","E_learned","E_B","E_B_learned") if a in arms]),
                far_visible_hidden=dict(protocol="K64 at fixed GT focus. Hidden hit means a ray receives positive GT substitute compositing weight.",
                                        highlighted=SPECIAL,by_scene=far_split(parts)),
                GT_tear_interpolation=slope_split(tables,include_new),dev003_subject=subjects,
                panel_observations=notes,figures=dict(optical=detail,subjects=subject_panels,development_table=overall),
                sharing_3c=stage3c_overlap())
    save_json(RECORD,result)
    (RUN/"汇总.md").write_text(make_markdown(result,reports))
    paths=[overall,*subject_panels,*detail]
    (RUN/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>E0 统一重渲</title>'
          '<h1>E0 统一重渲</h1><p>dev010 的近焦无有效主体，只展示远焦。</p>'
          '<p><a href="汇总.md">汇总与拆分</a></p>\n'+"\n".join(
            f'<figure><a href="{escape(p)}"><img src="{escape(p)}" width="900" loading="lazy"></a>'
            f'<figcaption>{escape(Path(p).stem)}</figcaption></figure>' for p in paths))
    state("complete",include_new=include_new,record=str(RECORD))
    print(f"E0 rerender result: {len(detail)} optical figures, {len(subject_panels)} subject figures",flush=True)


if __name__=="__main__":
    summary()
