"""CPU-only exploration preparation. Visual decisions are deliberately left to the user.

Run prepare BEFORE analyze. Materials never use A-line ranking. No model imports.
Existing multi-scene JSON containers are filtered immediately by exploration IDs;
held-out metric fields are neither selected, exported, inspected nor ranked.

Use the existing any2bokeh environment. Entry order: prepare; analyze --workers 4
and materials (both CPU only); verify. Scene JSONs and complete material entries
are reused on resume. Preserve and inspect any incomplete .partial.mp4 before
restarting that material item. Outputs are derived; original files stay intact.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import glob
import html
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PENDING = "待用户复核"
VISUAL = ["视觉是否异常", "是否闪烁", "是否有halo/颜色污染/细节损坏", "视觉自然程度", "人工严重程度", "是否属于感知失败", "是否视觉复核通过"]


def load(path):
    return json.loads((ROOT / path).read_text())


def save(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def csv_save(path, rows):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    f = io.StringIO(newline="")
    w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
    w.writeheader()
    for r in rows:
        w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    content = f.getvalue().encode('utf-8-sig')
    if path.exists():
        assert path.read_bytes() == content, f'Preserve inconsistent prior CSV: {path}'
    else:
        with path.open('xb') as target:
            target.write(content)


def now():
    return datetime.now(timezone.utc).isoformat()


def record(c, suffix):
    return c["record_prefix"] + suffix


def input_paths(c, row, kind):
    sid = row["scene_id"]
    if row["dataset"] == "Synthetic":
        sub = {"rgb": "rgb", "alpha": "alpha", "disparity": "disparity", "gt": "gt_mpib_public_srgb"}[kind]
        folder = ROOT / f"outputs/any2bokeh-mpib-public-proxy-200-20260928/clips/{sid}/{sub}"
        return sorted(folder.glob("*.png"))
    assert kind != "gt", "DAVIS has no paired bokeh GT"
    derived = load(c["davis_manifest"])["derived"]
    sub = "mask" if kind == "alpha" else kind
    return sorted((ROOT / derived / "eval" / sid / sub).glob("*.jpg" if kind == "rgb" else "*.png"))


def predictions(c, row, method):
    assert row["split"] == "exploration", "Held-out output access is prohibited"
    if row["dataset"] == "Synthetic":
        folder = ROOT / c["methods"][method]["synthetic"].format(scene=row["scene_id"])
    else:
        assert method != "magicbokeh"
        folder = ROOT / "outputs/davis-baselines-20260929" / method / "predictions" / row["scene_id"]
    return sorted(folder.glob("*.png"))


def image(path, gray=False, mask=False):
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    assert a is not None, path
    if a.shape[:2] != (576, 1024):
        a = cv2.resize(a, (1024, 576), interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR)
    if not gray:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    assert a.dtype == np.uint8 and np.isfinite(a).all()
    return a


def provenance(c):
    scenes = load(c["synthetic_config"])["scenes"]
    canonical_paths = defaultdict(set)
    path_ids = defaultdict(set)
    rows = []
    for s in scenes:
        meta=load(f"outputs/any2bokeh-mpib-public-proxy-200-20260928/clips/{s['id']}/metadata.json")
        assert meta['source']==s and len(meta['frames'])==25, 'Input metadata differs from source manifest'
        bg = Path(s["background_file"])
        bgid = "BG-20k:" + bg.stem
        assert s["background_source_id"] == bgid
        sources = [(bgid, s["background_file"], "background")]
        for o in s["objects"]:
            p = Path(o["rgb"])
            if o["kind"] == "VM":
                # Original movie only: segment start/stride have no effect on identity.
                sid = "VideoMatte240K:" + p.parent.parent.name + ":" + p.stem
                assert p.suffix == ".mp4"
            else:
                sid = "AM-2k:" + p.stem
            assert sid == o["source_id"], (sid, o["source_id"])
            sources.append((sid, o["rgb"], "foreground"))
        for sid, p, role in sources:
            assert (ROOT / p).is_file(), p
            canonical_paths[sid].add(str((ROOT / p).resolve()))
            path_ids[str((ROOT / p).resolve())].add(sid)
        rows.append(dict(dataset="Synthetic", scene_id=s["id"], K=s["K"],
                         sources=sorted(set(x[0] for x in sources)),
                         foreground_sources=sorted(set(x[0] for x in sources if x[2] == "foreground")),
                         background_sources=[bgid]))
    assert all(len(v) == 1 for v in canonical_paths.values())
    assert all(len(v) == 1 for v in path_ids.values()), "Source aliases require reconciliation"
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, r in enumerate(rows):
        for sid in r["sources"]:
            if sid in seen:
                parent[find(i)] = find(seen[sid])
            else:
                seen[sid] = i
    components = defaultdict(list)
    for i, r in enumerate(rows):
        components[find(i)].append(r["scene_id"])
    groups = sorted((sorted(v) for v in components.values()), key=lambda v: v[0])
    gid_by_scene = {sid: f"SYN-G{i:03d}" for i, g in enumerate(groups) for sid in g}
    target = round(len(rows) * (1 - c["exploration_fraction"]))
    lookup = {r["scene_id"]: r for r in rows}
    target_K = {k: sum(r["K"] == k for r in rows) * (1 - c["exploration_fraction"]) for k in [4, 8, 16, 32]}
    rng = random.Random(c["seed"])
    best = None
    # Seeded subset-sum: optimize count first, K imbalance second. No result data.
    for trial in range(c["split_trials"]):
        order = list(range(len(groups)))
        rng.shuffle(order)
        dp = {0: ()}
        for i in order:
            n = len(groups[i])
            for total, subset in sorted(list(dp.items()), reverse=True):
                if total + n <= target and total + n not in dp:
                    dp[total + n] = subset + (i,)
        total = max(dp)
        subset = dp[total]
        ids = {sid for i in subset for sid in groups[i]}
        counts = Counter(lookup[s]["K"] for s in ids)
        score = (abs(total - target), sum(abs(counts[k] - target_K[k]) for k in target_K))
        if best is None or score < best[0]:
            best = (score, ids, trial)
    for r in rows:
        r.update(source_group=gid_by_scene[r["scene_id"]], split="review" if r["scene_id"] in best[1] else "exploration")
    davis = load(c["davis_manifest"])
    ids = sorted(s["id"] for s in davis["sequences"])
    random.Random(c["seed"] + 1).shuffle(ids)
    review = set(ids[:round(len(ids) * (1 - c["exploration_fraction"]))])
    for s in davis["sequences"]:
        rows.append(dict(dataset="DAVIS", scene_id=s["id"], K=s["K"],
                         source_group="DAVIS:" + s["id"], sources=["DAVIS:" + s["id"]],
                         foreground_sources=[], background_sources=[],
                         original_DAVIS_split=s["split"],
                         split="review" if s["id"] in review else "exploration"))
    overlap = {}
    for dataset in ["Synthetic", "DAVIS"]:
        rr = [r for r in rows if r["dataset"] == dataset]
        e = {s for r in rr if r["split"] == "exploration" for s in r["sources"]}
        v = {s for r in rr if r["split"] == "review" for s in r["sources"]}
        assert not e & v
        overlap[dataset] = dict(overlap_sources=sorted(e & v), exploration_scenes=sum(r["split"] == "exploration" for r in rr),
                                review_scenes=sum(r["split"] == "review" for r in rr),
                                exploration_source_groups=len({r["source_group"] for r in rr if r["split"] == "exploration"}),
                                review_source_groups=len({r["source_group"] for r in rr if r["split"] == "review"}),
                                K_counts={sp: dict(Counter(r["K"] for r in rr if r["split"] == sp)) for sp in ["exploration", "review"]})
    return dict(created_utc=now(), seed=c["seed"], metric_values_used=False, model_outputs_used=False,
                split_rule="Synthetic seeded connected-component subset-sum; minimum count deviation then K L1 imbalance; first tie retained. DAVIS sorted source videos shuffled with seed+1, first30% held for review.",
                selected_trial=best[2], component_sizes=sorted(map(len, groups), reverse=True),
                canonical_sources=len(canonical_paths), raw_asset_alias_check=True,
                source_normalization="Original raw file stems; VideoMatte original split/movie, ignoring start/stride; AM/BG original asset, fg/mask are one source. Connected components across foreground AND background, irrespective of role.",
                limits="Existing development materials, not a new blind test. Held-out results remain unopened for discovery.",
                overlap_check=overlap, rows=rows)


def proxy(c, r):
    assert r["split"] == "exploration"
    masks = input_paths(c, r, "alpha")
    rgb = input_paths(c, r, "rgb")
    assert len(masks) == len(rgb) == 25
    area, edge, texture, changes = [], [], [], []
    prev = None
    for t in c["keyframes"]:
        a = image(masks[t], gray=True, mask=r["dataset"] == "DAVIS")
        fg = (a > 0 if r["dataset"] == "DAVIS" else a >= 128).astype(np.uint8)
        boundary = cv2.morphologyEx(fg, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        area.append(float(fg.mean()))
        edge.append(float(boundary.sum() / max(1, fg.sum())))
        im = image(rgb[t]).astype(np.float32) / 255
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        texture.append(float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F)))))
        if prev is not None:
            changes.append(float(np.mean(np.abs(im - prev))))
        prev = im
    return dict(dataset=r["dataset"], scene_id=r["scene_id"], source_group=r["source_group"], K=r["K"],
                foreground_fraction_proxy=float(np.mean(area)), boundary_complexity_proxy=float(np.mean(edge)),
                input_texture_proxy=float(np.mean(texture)), input_6frame_change_proxy=float(np.mean(changes)),
                semantic_labels="待用户确认")


def prepare(c):
    split = provenance(c)
    save(record(c, "-source-split.json"), split)
    csv_save(record(c, "-source-split.csv"), split["rows"])
    print("SOURCE_SPLIT_SAVED", split["overlap_check"], flush=True)
    cv2.setNumThreads(1)
    proxies = [proxy(c, r) for r in split["rows"] if r["split"] == "exploration"]
    csv_save(record(c, "-input-proxies.csv"), proxies)
    selected, used_groups = [], set()
    rng = random.Random(c["seed"] + 2)

    def add(candidates, reason, key=None):
        pool = [p for p in candidates if p["source_group"] not in used_groups]
        assert pool, (reason, "insufficient distinct groups")
        pool = sorted(pool, key=lambda p: p["scene_id"])
        item = rng.choice(pool) if key is None else min(pool, key=key)
        used_groups.add(item["source_group"])
        selected.append(dict(item, sample_id=f"B{len(selected)+1:02d}", sampling_channel="B-independent-input-only", sampling_reason=reason,
                             split="exploration", metric_or_prediction_used=False, **{k: PENDING for k in VISUAL}))

    for k in [4, 8, 16, 32]:
        pool = [p for p in proxies if p["dataset"] == "Synthetic" and p["K"] == k]
        median_a = np.median([p["foreground_fraction_proxy"] for p in pool])
        median_b = np.median([p["boundary_complexity_proxy"] for p in pool])
        add(pool, "ordinary-proxy central area/boundary (not human ordinary label)", lambda p: abs(p["foreground_fraction_proxy"]-median_a)+abs(p["boundary_complexity_proxy"]-median_b))
        add(pool, "high boundary-complexity proxy", lambda p: -p["boundary_complexity_proxy"])
        add(pool, "small foreground proxy" if k in [4, 16] else "large foreground proxy", lambda p: p["foreground_fraction_proxy"] * (1 if k in [4, 16] else -1))
        add(pool, "seeded random control")
    pool = [p for p in proxies if p["dataset"] == "DAVIS"]
    for reason, key in [("small foreground proxy", lambda p: p["foreground_fraction_proxy"]), ("large foreground proxy", lambda p: -p["foreground_fraction_proxy"]),
                        ("high boundary-complexity proxy", lambda p: -p["boundary_complexity_proxy"]),
                        ("ordinary-proxy median area (not human ordinary label)", lambda p: abs(p["foreground_fraction_proxy"]-np.median([q["foreground_fraction_proxy"] for q in pool])))]:
        add(pool, reason, key)
    for _ in range(4):
        add(pool, "seeded random control")
    # This persisted file is the precondition for loading any model metric values.
    save(record(c, "-B-samples.json"), dict(saved_utc=now(), seed=c["seed"]+2, order="saved before A metric access/ranking", rows=selected,
         coverage={ds: dict(n=sum(r["dataset"]==ds for r in selected), source_groups=len({r["source_group"] for r in selected if r["dataset"]==ds}),
                    K_counts=dict(Counter(r["K"] for r in selected if r["dataset"]==ds))) for ds in ["Synthetic", "DAVIS"]},
         missing_semantic_coverage="Hair, transparency, thin structures, occlusion, highlights etc. have not been human confirmed. No paper semantic labels asserted; no new data generated.",
         proxy_definition="Mean of input/reference-only frames0/6/12/18/24: foreground threshold, 3x3 gradient pixels / foreground pixels, mean |Laplacian RGB-gray|, mean absolute RGB change between sampled frames. Not visual difficulty or motion truth."))
    csv_save(record(c, "-B-samples.csv"), selected)
    blank = []
    for s in selected:
        for package in ["B1", "B2"]:
            blank.append(dict(dataset=s["dataset"], scene_id=s["scene_id"], source_group=s["source_group"], split=s["split"], sample_id=s["sample_id"], package=package,
                              frame_id_or_transition_id="", ROI_xywh="", actual_view_mode="", user_observation="", severity=PENDING, confidence=PENDING,
                              normal_motion_or_reasonable_defocus=PENDING, similar_in_input_or_GT=PENDING, normal_control="", notes="", **{k:PENDING for k in VISUAL}))
    csv_save(record(c, "-human-review-blank.csv"), blank)
    print("B_INDEPENDENT_LIST_SAVED", len(selected), "NO_METRIC_OR_PREDICTION_READ", flush=True)


def regions(alpha, disparity, dataset):
    a = alpha if dataset == "Synthetic" else (alpha > 0).astype(np.uint8) * 255
    binary = (a >= 128).astype(np.uint8)
    edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(1-edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) if edge.any() else np.full(a.shape, np.inf)
    q = disparity.astype(np.int16)
    jumps = np.zeros_like(q)
    diff = np.abs(q[1:]-q[:-1]); jumps[1:] = np.maximum(jumps[1:], diff); jumps[:-1] = np.maximum(jumps[:-1], diff)
    diff = np.abs(q[:,1:]-q[:,:-1]); jumps[:,1:] = np.maximum(jumps[:,1:], diff); jumps[:,:-1] = np.maximum(jumps[:,:-1], diff)
    return {"global": np.ones(a.shape, bool), "foreground": a>=230, "background": a<=25, "soft_alpha": (a>25)&(a<230),
            "boundary_0_2": distance<=2, "boundary_2_8": (distance>2)&(distance<=8), "boundary_8_24": (distance>8)&(distance<=24),
            "boundary_gt24": distance>24, "disparity_discontinuity": cv2.dilate((jumps>=4).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)}


def cpu_scene(config_path, row):
    c = load(config_path)
    cv2.setNumThreads(1)
    dest = Path(c["output"]) / "A-CPU" / row["dataset"] / (row["scene_id"] + ".json")
    if (ROOT/dest).exists():
        return load(dest)
    assert row["split"] == "exploration"
    ds = row["dataset"]
    methods = [m for m in c["methods"] if ds=="Synthetic" or m!="magicbokeh"]
    pp = {m: predictions(c,row,m) for m in methods}
    refs = input_paths(c,row,"gt" if ds=="Synthetic" else "rgb")
    masks = input_paths(c,row,"alpha"); disparities = input_paths(c,row,"disparity")
    assert all(len(p)==25 for p in [refs,masks,disparities]+list(pp.values()))
    assert len({tuple(p.stem.split('_zf_')[0] for p in paths) for paths in [refs,masks,disparities]+list(pp.values())})==1, "Frame alignment"
    stats = {m: defaultdict(lambda: np.zeros(5,np.float64)) for m in methods}
    frames = {m: [] for m in methods}; transitions = {m: [] for m in methods}
    previous = {}
    previous_reg = None
    for t in range(25):
        ref = image(refs[t]).astype(np.float32)/255
        reg = regions(image(masks[t],True,ds=="DAVIS"), image(disparities[t],True),ds)
        current = {}
        for m in methods:
            pred = image(pp[m][t]).astype(np.float32)/255
            error = pred-ref
            ae = np.mean(np.abs(error),axis=2); se=np.mean(error*error,axis=2)
            current[m] = error
            frames[m].append(dict(frame_id=t,global_MAE=float(ae.mean()),global_MSE=float(se.mean()),boundary_MAE=float(ae[reg['boundary_0_2']].mean()) if reg['boundary_0_2'].any() else None))
            te = np.mean(np.abs(error-previous[m]),axis=2) if t else None
            for name,mask in reg.items():
                s=stats[m][name];s[0]+=float(ae[mask].sum(dtype=np.float64));s[1]+=float(se[mask].sum(dtype=np.float64));s[2]+=int(mask.sum())
                if t:
                    temporal_mask=mask|previous_reg[name];s[3]+=float(te[temporal_mask].sum(dtype=np.float64));s[4]+=int(temporal_mask.sum())
            if t:
                transitions[m].append(dict(transition_id=f'{t-1:02d}->{t:02d}',time_change_MAE=float(te.mean()),boundary_time_change_MAE=float(te[reg['boundary_0_2']|previous_reg['boundary_0_2']].mean()) if (reg['boundary_0_2']|previous_reg['boundary_0_2']).any() else None))
        previous=current;previous_reg=reg
    result=dict(dataset=ds,scene_id=row['scene_id'],source_group=row['source_group'],K=row['K'],split='exploration',
                reference='MPIB-public proxy GT' if ds=='Synthetic' else 'input RGB; modification, not paired GT error',methods={})
    for m in methods:
        result['methods'][m] = dict(regions={name:dict(MAE=float(s[0]/s[2]) if s[2] else None,MSE=float(s[1]/s[2]) if s[2] else None,
                    pixels=int(s[2]),time_change_MAE=float(s[3]/s[4]) if s[4] else None,transition_pixels=int(s[4])) for name,s in stats[m].items()},
                    frames=frames[m],transitions=transitions[m])
    save(dest,result)
    print('CPU_SCENE_COMPLETE',ds,row['scene_id'],flush=True)
    return result


def existing_metrics(c, split):
    assert (ROOT / record(c,'-B-samples.json')).is_file(), 'Save independent B sample list BEFORE metric access'
    allowed={ds:{r['scene_id'] for r in split['rows'] if r['dataset']==ds and r['split']=='exploration'} for ds in ['Synthetic','DAVIS']}
    out=[]
    keys=['PSNR','PSNR_clip_mse','SSIM','LPIPS','FD','RM','RM_author_public_2transitions','PSNR_TorchMetrics','SSIM_TorchMetrics']
    for m, spec in c['methods'].items():
        found={}
        for f in sorted(glob.glob(str(ROOT/spec['metrics']))):
            # Existing containers include both splits. Held-out rows are discarded by ID;
            # their model-valued fields are never copied or used. No new inference.
            for r in json.loads(Path(f).read_text())['rows']:
                if r['scene'] not in allowed['Synthetic']:
                    continue
                assert r['scene'] not in found
                found[r['scene']]=dict(dataset='Synthetic',scene_id=r['scene'],method=m,**{k:r.get(k) for k in keys},metric_file=str(Path(f).relative_to(ROOT)))
        assert set(found)==allowed['Synthetic'],(m,len(found))
        out.extend(found.values())
    for m in c['methods']:
        if m=='magicbokeh':continue
        path=f'outputs/davis-baselines-20260929/{m}/evaluation/summary.json'
        rows=load(path)['rows']
        selected=[dict(dataset='DAVIS',scene_id=r['id'],method=m,**{k:r[k] for k in ['VEPI_public_mp4','VEPI_png','codec_delta']},metric_file=path) for r in rows if r['id'] in allowed['DAVIS']]
        assert len(selected)==len(allowed['DAVIS']);out.extend(selected)
    return out


def analyze(c, config_path, workers):
    split=load(record(c,'-source-split.json'))
    metrics=existing_metrics(c,split)
    csv_save(record(c,'-A-existing-metrics.csv'),metrics)
    rows=[r for r in split['rows'] if r['split']=='exploration']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(cpu_scene,config_path,r) for r in rows]
        results=[f.result() for f in as_completed(jobs)]
    results.sort(key=lambda r:(r['dataset'],r['scene_id']))
    summarize(c,results,metrics)


def summarize(c,results,metrics):
    proxies={(p['dataset'],p['scene_id']):p for p in csv.DictReader((ROOT/record(c,'-input-proxies.csv')).open(encoding='utf-8-sig'))}
    metric_map={(p['dataset'],p['scene_id'],p['method']):p for p in metrics}
    numerical=[];flat=[]
    for ds in ['Synthetic','DAVIS']:
        subset=[r for r in results if r['dataset']==ds]
        for K in sorted({r['K'] for r in subset}):
            rr=[r for r in subset if r['K']==K]
            def value(r,method,region,key):return r['methods'][method]['regions'][region][key]
            mses=[value(r,'any2bokeh','global','MSE') for r in rr]
            boundaries=[value(r,'any2bokeh','boundary_0_2','MAE') for r in rr]
            temporal=[value(r,'any2bokeh','global','time_change_MAE') for r in rr]
            thresholds=dict(mse90=float(np.quantile(mses,.9)),mse50=float(np.median(mses)),mse25=float(np.quantile(mses,.25)),mse75=float(np.quantile(mses,.75)),
                            boundary90=float(np.quantile([v for v in boundaries if v is not None],.9)),boundary50=float(np.median([v for v in boundaries if v is not None])),temporal90=float(np.quantile(temporal,.9)))
            if ds=='DAVIS':
                scores=[metric_map[(ds,r['scene_id'],'any2bokeh')]['VEPI_public_mp4'] for r in rr]
                deltas=[abs(metric_map[(ds,r['scene_id'],'any2bokeh')]['codec_delta']) for r in rr]
                thresholds.update(vepi10=float(np.quantile(scores,.1)),vepi25=float(np.quantile(scores,.25)),vepi75=float(np.quantile(scores,.75)),codec90=float(np.quantile(deltas,.9)))
            for r in rr:
                mse=value(r,'any2bokeh','global','MSE');b=value(r,'any2bokeh','boundary_0_2','MAE');bm=value(r,'mpib','boundary_0_2','MAE');tm=value(r,'any2bokeh','global','time_change_MAE')
                flags=[]
                if ds=='Synthetic':
                    if mse>=thresholds['mse90'] or tm>=thresholds['temporal90']:flags.append('absolute_proxy_error_candidate')
                    if b is not None and bm is not None and b>1.10*bm and b-bm>=1/255:flags.append('relative_boundary_candidate_vs_fixed_MPIB')
                    if b is not None and mse<=thresholds['mse50'] and b>=thresholds['boundary90']:flags.append('global_normal_local_candidate')
                    normal=thresholds['mse25']<=mse<=thresholds['mse75'] and b is not None and b<=thresholds['boundary50']
                else:
                    mr=metric_map[(ds,r['scene_id'],'any2bokeh')];cmp=metric_map[(ds,r['scene_id'],'mpib')]
                    if mr['VEPI_public_mp4']<=thresholds['vepi10']:flags.append('low_foreground_edge_VEPI_candidate')
                    if mr['VEPI_public_mp4']<=cmp['VEPI_public_mp4']-.02:flags.append('relative_foreground_edge_candidate_vs_fixed_MPIB')
                    if abs(mr['codec_delta'])>=thresholds['codec90']:flags.append('codec_sensitivity_candidate_not_model_failure')
                    normal=thresholds['vepi25']<=mr['VEPI_public_mp4']<=thresholds['vepi75']
                numerical.append(dict(dataset=ds,scene_id=r['scene_id'],source_group=r['source_group'],split='exploration',K=K,channel='A-numerical',
                    numerical_candidate_flags=flags,normal_numeric_control_eligible=normal,global_MSE=mse,boundary_MAE=b,MPIB_boundary_MAE=bm,
                    BokehMe_boundary_MAE=value(r,'bokehme','boundary_0_2','MAE'),time_change_MAE=tm,thresholds=thresholds,
                    existing_ATB_metrics=metric_map[(ds,r['scene_id'],'any2bokeh')], **{k:PENDING for k in VISUAL}))
        for r in subset:
            for m,obs in r['methods'].items():
                for region,vals in obs['regions'].items():
                    flat.append(dict(dataset=ds,scene_id=r['scene_id'],source_group=r['source_group'],K=r['K'],method=m,region=region,**vals))
    selected=[]
    for ds in ['Synthetic','DAVIS']:
        for category in sorted({f for r in numerical if r['dataset']==ds for f in r['numerical_candidate_flags']}):
            candidates=[r for r in numerical if r['dataset']==ds and category in r['numerical_candidate_flags']]
            if 'VEPI' in category or 'foreground_edge' in category:key=lambda r:r['existing_ATB_metrics']['VEPI_public_mp4']
            elif 'codec' in category:key=lambda r:-abs(r['existing_ATB_metrics']['codec_delta'])
            elif 'relative' in category:key=lambda r:-(r['boundary_MAE']-r['MPIB_boundary_MAE'])
            elif 'local' in category:key=lambda r:-r['boundary_MAE']
            else:key=lambda r:-r['global_MSE']
            groups=set()
            for r in sorted(candidates,key=lambda r:(key(r),r['scene_id'])):
                if r['source_group'] in groups:continue
                groups.add(r['source_group']);selected.append(dict(r,selection_category=category))
                if len(groups)==3:break
    proxykeys=['foreground_fraction_proxy','boundary_complexity_proxy','input_texture_proxy','input_6frame_change_proxy']
    controls=[]
    for candidate in selected:
        pool=[r for r in numerical if r['dataset']==candidate['dataset'] and r['K']==candidate['K'] and r['normal_numeric_control_eligible'] and r['source_group']!=candidate['source_group']]
        if not pool:
            controls.append(dict(dataset=candidate['dataset'],candidate_scene=candidate['scene_id'],selection_category=candidate['selection_category'],control_scene=None,reason='No other-source-group control met the predeclared numeric rule; no fallback relabeling'))
            continue
        peers=[p for p in proxies.values() if p['dataset']==candidate['dataset'] and int(p['K'])==candidate['K']]
        scales={k:max(float(np.std([float(p[k]) for p in peers])),1e-8) for k in proxykeys}
        base=proxies[(candidate['dataset'],candidate['scene_id'])]
        def distance(r):
            p=proxies[(r['dataset'],r['scene_id'])]
            return sum(((float(p[k])-float(base[k]))/scales[k])**2 for k in proxykeys)
        control=min(pool,key=lambda r:(distance(r),r['scene_id']))
        controls.append(dict(dataset=candidate['dataset'],candidate_scene=candidate['scene_id'],selection_category=candidate['selection_category'],control_scene=control['scene_id'],control_source_group=control['source_group'],K=control['K'],proxy_distance_squared=float(distance(control)),control_global_MSE=control['global_MSE'],control_boundary_MAE=control['boundary_MAE'],视觉是否异常=PENDING))
    # Scene-first aggregate; independent source groups are separately weighted equally.
    aggregates=[]
    for ds in ['Synthetic','DAVIS']:
        for m in c['methods']:
            if ds=='DAVIS' and m=='magicbokeh':continue
            for region in sorted({r['region'] for r in flat}):
                rr=[r for r in flat if r['dataset']==ds and r['method']==m and r['region']==region]
                group=defaultdict(list)
                for r in rr:
                    if r['MAE'] is not None:group[r['source_group']].append(r['MAE'])
                vals=[r['MAE'] for r in rr if r['MAE'] is not None]
                aggregates.append(dict(dataset=ds,method=m,region=region,valid_scenes=len(vals),empty_scenes=len(rr)-len(vals),source_groups=len(group),
                    pixel_observations=sum(r['pixels'] for r in rr),scene_equal_MAE=float(np.mean(vals)) if vals else None,
                    source_group_equal_MAE=float(np.mean([np.mean(v) for v in group.values()])) if group else None))
    csv_save(record(c,'-A-regions.csv'),flat)
    csv_save(record(c,'-A-region-aggregates.csv'),aggregates)
    csv_save(record(c,'-A-numeric-candidates.csv'),selected)
    csv_save(record(c,'-A-normal-controls.csv'),controls)
    save(record(c,'-A-analysis.json'),dict(completed_utc=now(),status='CPU_ANALYSIS_COMPLETE_NOT_VISUAL_FAILURES',
         samples_saved_before_metric_access=load(record(c,'-B-samples.json'))['saved_utc'],
         rules=c['A_rules_fixed_before_metrics'],region_protocol=c['region_protocol'],
         existing_metric_definitions=load('outputs/any2bokeh-mpib-public-proxy-200-20260928/metrics-00.json')['metrics'],
         metric_limits=['ATB per-clip TorchMetrics PSNR/SSIM and author2-pair RM unavailable: null, not synthesized; common comparisons use same skimage PSNR/SSIM and RM24.',
                        'Global existing RM24 and new CPU time_change_MAE use the same unwarped frame-difference expression; boundary bins are new diagnostics. They do not establish flicker.',
                        'FVD is dataset-level and never used for per-video ranking.',
                        'DAVIS uses input modification diagnostics, not paired bokeh error; no motion compensation.',
                        'Shared-source scenes are not independent repeats; no significance or visual conclusions.'],
         candidates=selected,normal_controls=controls,all_exploration_flags=numerical,scene_results=[str(Path(c['output'])/'A-CPU'/r['dataset']/(r['scene_id']+'.json')) for r in results],aggregates=aggregates))
    print('A_CPU_ANALYSIS_COMPLETE',len(results),'selected',len(selected),flush=True)


def roi_from_input_mask(mask):
    """Fixed crop selected using input mask only, never model output or A error."""
    edge=cv2.morphologyEx((mask>=128).astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    candidates=[(int(edge[y:y+144,x:x+256].sum()),x,y) for y in range(0,576-144+1,48) for x in range(0,1024-256+1,64)]
    _,x,y=max(candidates,key=lambda p:(p[0],-abs(p[1]-384)-abs(p[2]-216),-p[2],-p[1]))
    return [x,y,256,144]


def grid(panels, labels, t, cols, tile=(1024,576), crop=None):
    width,height=tile
    rows=(len(labels)+cols-1)//cols
    canvas=Image.new('RGB',(cols*width,rows*(height+24)),(248,250,253));draw=ImageDraw.Draw(canvas)
    for i,(paths,label) in enumerate(zip(panels,labels)):
        a=image(paths[t]);im=Image.fromarray(a)
        if crop:
            x,y,w,h=crop;im=im.crop((x,y,x+w,y+h))
        if im.size!=tile:im=im.resize(tile,Image.Resampling.BILINEAR)
        x=(i%cols)*width;y=(i//cols)*(height+24)
        canvas.paste(im,(x,y+24));draw.text((x+8,y+5),f'{label} | frame {t:02d}/24',fill=(15,40,80))
    return canvas


def video_encode(c,path,panels,labels,cols):
    if path.exists():return
    assert not path.with_suffix('.partial.mp4').exists(),f'Inspect preserved incomplete video: {path}'
    sample=grid(panels,labels,0,cols)
    w,h=sample.size;d=c['display'];tmp=path.with_suffix('.partial.mp4')
    args=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
          '-r',str(d['fps']),'-i','pipe:0','-an','-c:v',d['codec'],'-preset',d['preset'],'-crf',str(d['crf']),
          '-pix_fmt',d['pixel_format'],'-threads',str(d['threads']),'-movflags','+faststart',str(tmp)]
    with path.with_suffix('.encode.log').open('ab') as log:
        proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        try:
            for t in range(25):proc.stdin.write(grid(panels,labels,t,cols).tobytes())
            proc.stdin.close();code=proc.wait()
        except BaseException:
            proc.stdin.close();proc.wait();raise
        assert code==0,(code,path)
    tmp.rename(path)


def contact_sheet(path,panels,labels,cols,keyframes,crop=None):
    if path.exists():return
    tile=(512,288) if cols==2 else (384,216)
    sheets=[grid(panels,labels,t,cols,tile,crop) for t in keyframes]
    canvas=Image.new('RGB',(sheets[0].width,sum(im.height for im in sheets)),'white')
    offset=0
    for im in sheets:canvas.paste(im,(0,offset));offset+=im.height
    canvas.save(path)


def link_originals(folder,paths):
    folder.mkdir(parents=True,exist_ok=True)
    for t,p in enumerate(paths):
        dest=folder/f'{t:05d}.png'
        if not dest.exists():dest.symlink_to(os.path.relpath(p,folder))
        assert dest.resolve()==p.resolve()


def write_html(path,text):
    if path.exists():
        assert path.read_text()==text, f'Preserve edited or inconsistent page: {path}'
    else:
        path.write_text(text)


STYLE="""<style>body{font-family:system-ui,sans-serif;max-width:1300px;margin:30px auto;padding:0 20px;color:#173657;background:#f6f9ff}article{background:white;padding:20px;margin:22px 0;border:1px solid #b8cfea;border-radius:10px}video,img{width:100%;height:auto}a{color:#125bb8}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #b8cfea}.note{background:#e7f0ff;padding:16px}input,textarea,select{font:inherit;max-width:100%}textarea{width:100%;min-height:65px}button{font:inherit;padding:9px;margin:6px}</style>"""


def review_form(samples):
    """Local browser form stores only user-entered observations, never auto labels."""
    payload=json.dumps([{k:s[k] for k in ['sample_id','dataset','scene_id','source_group','split']} for s in samples],ensure_ascii=False)
    fields=['frame_id_or_transition_id','ROI_xywh','actual_view_mode','user_observation','severity','confidence','normal_motion_or_reasonable_defocus','similar_in_input_or_GT','normal_control','notes']+VISUAL
    labels=['帧号或transition','ROI x,y,w,h','实际观看方式：原速/慢放/逐帧/接触图','用户观察到的现象','严重程度','置信程度','是否正常运动或合理虚化','输入/GT是否已有类似现象','正常对照','补充说明']+VISUAL
    js="""<script>
const samples=SAMPLES,fields=FIELDS,labels=LABELS; const storeKey='atb-failure-B1-user-20261009';
let saved=JSON.parse(localStorage.getItem(storeKey)||'[]');
function add(row){const section=document.createElement('article'); const base=row||{...samples[0],package:'B1'};
const sel=document.createElement('select');samples.forEach(s=>{let o=document.createElement('option');o.value=s.sample_id;o.textContent=s.sample_id+' '+s.dataset+' '+s.scene_id;o.selected=s.sample_id===base.sample_id;sel.append(o)});section.append(sel);section.dataset.kind='review';
fields.forEach((k,i)=>{const label=document.createElement('label');label.textContent=labels[i];const t=document.createElement('textarea');t.dataset.key=k;t.value=base[k]||(['severity','confidence','normal_motion_or_reasonable_defocus','similar_in_input_or_GT'].includes(k)||i>=10?'待用户复核':'');label.append(t);section.append(label)});document.querySelector('#forms').append(section)}
function rows(){return [...document.querySelectorAll('[data-kind=review]')].map(s=>{let row={...samples.find(v=>v.sample_id===s.querySelector('select').value),package:'B1'};s.querySelectorAll('textarea').forEach(t=>row[t.dataset.key]=t.value);return row})}
function saveLocal(){localStorage.setItem(storeKey,JSON.stringify(rows()));document.querySelector('#status').textContent='已保存在当前浏览器；请另导出文件保存。'}
function exportData(){saveLocal();let b=new Blob([JSON.stringify(rows(),null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='ATB-B1-user-review.json';a.click();URL.revokeObjectURL(a.href)}
document.addEventListener('input',()=>{localStorage.setItem(storeKey,JSON.stringify(rows()))});
(saved.length?saved:samples.map(s=>({...s,package:'B1'}))).forEach(add);
</script>""".replace('SAMPLES',payload).replace('FIELDS',json.dumps(fields,ensure_ascii=False)).replace('LABELS',json.dumps(labels,ensure_ascii=False))
    return '<!doctype html><meta charset="utf-8"><title>B1用户复核表</title>'+STYLE+'<h1>B1用户复核表</h1><p>所有视觉结论由用户填写；可为同一scene新增多条ROI/帧记录。填写后请导出JSON回传；未填写项保持待用户复核。浏览器本地保存不等于项目已收到反馈。</p><button onclick="add()">新增记录</button><button onclick="exportData()">导出用户JSON</button><span id="status"></span><div id="forms"></div>'+js


def materials(c):
    samples=load(record(c,'-B-samples.json'))['rows'];out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(c['seed']+3);mapping={}
    for ds in ['Synthetic','DAVIS']:
        methods=[m for m in c['methods'] if ds=='Synthetic' or m!='magicbokeh'];rng.shuffle(methods)
        mapping[ds]={f'P{i+1:02d}':m for i,m in enumerate(methods)}
    mapfile=record(c,'-B2-panel-mapping.json')
    value=dict(seed=c['seed']+3,mapping=mapping,permissions={m:p['permission'] for m,p in c['methods'].items()},
               note='Separate from B1 index. Panel numbers anonymize names, not a claim of perceptual blinding; B1 already reveals ATB.')
    if not (ROOT/mapfile).exists():save(mapfile,value)
    else:assert load(mapfile)==value
    items={'B1':[],'B2':[]};manifest=[]
    for s in samples:
        ds=s['dataset'];sid=s['scene_id'];sample=s['sample_id'];paths=input_paths(c,s,'rgb');assert len(paths)==25
        # Shared lossless display inputs; DAVIS resizing follows frozen frame-method RGB resize.
        rgb=out/'display-input'/sample;rgb.mkdir(parents=True,exist_ok=True)
        if ds=='Synthetic':link_originals(rgb,paths)
        else:
            for t,p in enumerate(paths):
                dest=rgb/f'{t:05d}.png'
                if not dest.exists():Image.fromarray(image(p)).save(dest)
        rgbpaths=sorted(rgb.glob('*.png'))
        roi=roi_from_input_mask(image(input_paths(c,s,'alpha')[12],True,ds=='DAVIS'))
        for package in ['B1','B2']:
            folder=out/package/sample;folder.mkdir(parents=True,exist_ok=True)
            complete=folder/'complete.json'
            if package=='B1':
                labels=['Input RGB','Frozen ATB'];panels=[rgbpaths,predictions(c,s,'any2bokeh')];cols=2
            else:
                labels=['Input RGB'];panels=[rgbpaths]
                if ds=='Synthetic':labels+=['Proxy GT'];panels+=[input_paths(c,s,'gt')]
                for label,method in mapping[ds].items():labels.append(label);panels.append(predictions(c,s,method))
                cols=3
            assert all(len(p)==25 for p in panels)
            for label,p in zip(labels,panels):link_originals(folder/'originals'/label.replace(' ','_'),p)
            video_encode(c,folder/'video.mp4',panels,labels,cols)
            contact_sheet(folder/'contact.png',panels,labels,cols,c['keyframes'])
            if package=='B2':contact_sheet(folder/'ROI-contact.png',panels,labels,cols,c['keyframes'],roi)
            title=f'{sample} · {ds} · {sid} · K={s["K"]}'
            original_links=' '.join(f'<a href="originals/{label.replace(" ","_")}/00012.png">{html.escape(label)} frame12 PNG</a>' for label in labels)
            all_links='<details><summary>全部25帧PNG核对入口</summary>'+''.join('<p>'+html.escape(label)+': '+ ' '.join(f'<a href="originals/{label.replace(" ","_")}/{t:05d}.png">{t:02d}</a>' for t in range(25))+'</p>' for label in labels)+'</details>'
            body=f'<!doctype html><meta charset="utf-8"><title>{title}</title>'+STYLE+f'<h1>{title}</h1><p>来源组：{s["source_group"]}；探索组。语义标签待用户确认；全部视觉判断待用户复核。</p><p>20fps，25帧，1×播放；每个完整视频面板1024×576，无插帧。浏览器缩放显示，可下载原文件或查看PNG。</p><video controls preload="metadata" src="video.mp4"></video><p><a href="video.mp4" download>下载连续25帧视频</a> · <a href="contact.png">0/6/12/18/24接触图</a></p><img loading="lazy" src="contact.png">'
            if package=='B2':body+=f'<p>统一局部ROI={roi}（x,y,w,h），仅按输入mask边界代理选取；所有方法同位置同范围、双线性放大。它不是预设失败位置。</p><a href="ROI-contact.png">统一ROI接触图</a><p>{"Synthetic GT为MPIB-public proxy。" if ds=="Synthetic" else "DAVIS无paired bokeh GT，不显示GT。"}</p>'
            body+=f'<p>疑似压缩伪影请核对无损PNG：{original_links}</p>'+all_links
            write_html(folder/'index.html',body)
            entry=dict(sample_id=sample,dataset=ds,scene_id=sid,source_group=s['source_group'],split='exploration',package=package,
                       index=str((folder/'index.html').relative_to(ROOT)),video=str((folder/'video.mp4').relative_to(ROOT)),contact=str((folder/'contact.png').relative_to(ROOT)),
                       frames=25,keyframes=c['keyframes'],labels=labels,panel_size=[1024,576],video_size=[cols*1024,((len(labels)+cols-1)//cols)*600],ROI_xywh=roi if package=='B2' else None,
                       raw_rgb_paths=[str(p.relative_to(ROOT)) for p in paths],display_RGB_resize='none' if ds=='Synthetic' else 'cv2.INTER_LINEAR to1024x576; derived PNG8, original JPEG unchanged',
                       prediction_source_paths=[[str(p.relative_to(ROOT)) for p in group] for group in panels[1:]],visual_review=PENDING)
            if not complete.exists():save(complete,entry)
            else:assert load(complete)==entry
            manifest.append(entry)
            items[package].append(f'<article><h2><a href="{sample}/index.html">{title}</a></h2><p>来源组 {s["source_group"]}；视觉判断：待用户复核</p></article>')
            print('MATERIAL_COMPLETE',package,sample,ds,sid,flush=True)
    for package in ['B1','B2']:
        prefix='<p class="note">请先独立填写B1观察；本页不显示数值排名或失败标签。<a href="review.html">打开空白人工复核表</a>。完成后导出用户记录交回，再进入后续对照。</p>' if package=='B1' else '<p class="note">B2对照复核材料：等待用户完成B1记录后再启用。面板编号与映射分开保存；此处不提供任何自动视觉结论。</p>'
        page=f'<!doctype html><meta charset="utf-8"><title>{package}观看索引</title>'+STYLE+f'<h1>{package}观看索引</h1>'+prefix+'<p>固定独立样本顺序。20fps，25帧，每面板1024×576；H.264 CRF18 medium yuv420p，ffmpeg7.0.2，CPU2线程；视频为派生展示，不替换原预测。接触图双线性缩小。没有慢放包。DAVIS沿用冻结MP420fps；Synthetic为25个timeline样本的声明展示帧率，不声称原VideoMatte素材时间速率。</p>'+''.join(items[package])
        write_html(out/package/'index.html',page)
    write_html(out/'B1/review.html',review_form(samples))
    dest=record(c,'-materials.json')
    if not (ROOT/dest).exists():save(dest,dict(completed_utc=now(),display=c['display'],ffmpeg=imageio_ffmpeg.get_ffmpeg_version(),items=manifest,B1_index=str(Path(c['output'])/'B1/index.html'),
         B2_index=str(Path(c['output'])/'B2/index.html'),B2_release='Wait for user B1 marks; do not prompt B2 viewing before then',mapping=mapfile,human_review_status=PENDING))
    print('ALL_MATERIALS_COMPLETE',len(manifest),flush=True)


def verify(c):
    split=load(record(c,'-source-split.json'));b=load(record(c,'-B-samples.json'));a=load(record(c,'-A-analysis.json'));m=load(record(c,'-materials.json'))
    explore={(r['dataset'],r['scene_id']) for r in split['rows'] if r['split']=='exploration'}
    review={(r['dataset'],r['scene_id']) for r in split['rows'] if r['split']=='review'}
    assert not explore&review
    assert all(not v['overlap_sources'] for v in split['overlap_check'].values())
    assert all((r['dataset'],r['scene_id']) in explore and all(r[k]==PENDING for k in VISUAL) for r in b['rows'])
    assert len({r['source_group'] for r in b['rows']})==len(b['rows'])
    assert len(a['all_exploration_flags'])==len(explore)
    assert all((r['dataset'],r['scene_id']) in explore and all(r[k]==PENDING for k in VISUAL) for r in a['all_exploration_flags'])
    assert b['saved_utc']==a['samples_saved_before_metric_access']
    for s in load(c['synthetic_config'])['scenes']:
        meta=load(f"outputs/any2bokeh-mpib-public-proxy-200-20260928/clips/{s['id']}/metadata.json")
        assert meta['source']==s and len(meta['frames'])==25
    videos=[];rm_deltas=[];pooled_mse_relative_deltas=[];prediction_headers=0;scene_observations=[];reduction_cases=[]
    em=list(csv.DictReader((ROOT/record(c,'-A-existing-metrics.csv')).open(encoding='utf-8-sig')))
    existing={(r['dataset'],r['scene_id'],r['method']):r for r in em}
    for path in a['scene_results']:
        r=load(path);assert (r['dataset'],r['scene_id']) in explore
        scene_observations.append(r)
        for method in r['methods']:
            pp=predictions(c,dict(r,split='exploration'),method)
            assert [p.stem for p in pp]==[f'{t:05d}' for t in range(25)]
            for p in pp:
                with Image.open(p) as im:
                    assert im.size==(1024,576) and im.mode=='RGB',p
                prediction_headers+=1
        for method,o in r['methods'].items():
            assert len(o['frames'])==25 and len(o['transitions'])==24
            assert o['regions']['global']['pixels']==25*576*1024 and o['regions']['global']['transition_pixels']==24*576*1024
            assert sum(o['regions'][name]['pixels'] for name in ['foreground','background','soft_alpha'])==o['regions']['global']['pixels']
            assert sum(o['regions'][name]['pixels'] for name in ['boundary_0_2','boundary_2_8','boundary_8_24','boundary_gt24'])==o['regions']['global']['pixels']
            assert {name:obs['pixels'] for name,obs in o['regions'].items()}=={name:obs['pixels'] for name,obs in r['methods']['any2bokeh']['regions'].items()}
            if r['dataset']=='Synthetic':
                delta=abs(o['regions']['global']['time_change_MAE']-float(existing[('Synthetic',r['scene_id'],method)]['RM']))
                assert delta<1e-7,(r['scene_id'],method,delta);rm_deltas.append(delta)
                old_mse=10**(-float(existing[('Synthetic',r['scene_id'],method)]['PSNR_clip_mse'])/10)
                relative=abs(o['regions']['global']['MSE']-old_mse)/old_mse
                assert np.isfinite(relative)
                if relative>=3e-6:
                    reduction_cases.append((r,method,old_mse,relative))
                pooled_mse_relative_deltas.append(relative)
    # The initial 3e-6 consistency check failed on4 entries. Reproduce those
    # original float32 reductions, rather than silently relaxing that check.
    audit_path=ROOT/record(c,'-CPU-reduction-audit.json')
    audit=load(audit_path) if audit_path.exists() else {}
    if 'reproduced_initial_check_failures' not in audit:
        audited=[]
        for r,method,old,relative in reduction_cases:
            pred=np.stack([image(p).astype(np.float32)/255 for p in predictions(c,r,method)])
            ref=np.stack([image(p).astype(np.float32)/255 for p in input_paths(c,r,'gt')])
            sq=(pred-ref)**2
            literal=float(np.mean(sq));acc64=float(np.mean(sq,dtype=np.float64))
            regional=r['methods'][method]['regions']['global']['MSE']
            assert abs(literal-old)/old<1e-12,(r['scene_id'],method,'Original expression not reproduced')
            assert abs(regional-acc64)/acc64<1e-7,(r['scene_id'],method,'Regional accumulator mismatch')
            audited.append(dict(scene_id=r['scene_id'],method=method,existing_pooled_MSE=old,literal_float32=literal,
                float64_accumulator=acc64,regional_pipeline=regional,relative_difference=relative))
            del pred,ref,sq
        audit['reproduced_initial_check_failures']=audited
        audit['interpretation']='Existing float32 whole-array mean is reproduced; new CPU region sums use float64 accumulation. Distinct reporting arithmetic, no original metric replacement or candidate-rule change.'
        with audit_path.open('w') as f:json.dump(audit,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    assert {(r['scene_id'],method) for r,method,_,_ in reduction_cases}=={(r['scene_id'],r['method']) for r in audit['reproduced_initial_check_failures']}
    for item in m['items']:
        assert (item['dataset'],item['scene_id']) in explore
        cap=cv2.VideoCapture(str(ROOT/item['video']));count=0;size=None
        assert abs(cap.get(cv2.CAP_PROP_FPS)-c['display']['fps'])<1e-6
        while True:
            ok,frame=cap.read()
            if not ok:break
            size=[frame.shape[1],frame.shape[0]];assert size==item['video_size'] and np.isfinite(frame).all();count+=1
        cap.release();assert count==25
        with Image.open(ROOT/item['contact']) as im:im.verify()
        videos.append(dict(package=item['package'],sample_id=item['sample_id'],frames=count,size=size,fps=c['display']['fps'],finite=True))
    blank=list(csv.DictReader((ROOT/record(c,'-human-review-blank.csv')).open(encoding='utf-8-sig')))
    assert len(blank)==2*len(b['rows']) and all(all(r[k]==PENDING for k in VISUAL) for r in blank)
    b1page=(ROOT/c['output']/'B1/index.html').read_text()
    assert all(x not in b1page for x in ['PSNR','LPIPS','A-numerical','A-analysis','B2/index','P01','Proxy GT'])
    # These are reporting summaries of already persisted CPU scene statistics;
    # no frame is reprocessed and no selection rule is changed.
    extra_aggregates=[]
    for ds in ['Synthetic','DAVIS']:
        rr=[r for r in scene_observations if r['dataset']==ds]
        for method in rr[0]['methods']:
            for region in rr[0]['methods'][method]['regions']:
                for key in ['MSE','time_change_MAE']:
                    group=defaultdict(list);values=[];empty=0;pixel_observations=0
                    for r in rr:
                        obs=r['methods'][method]['regions'][region];value=obs[key]
                        pixel_observations+=obs['transition_pixels' if key=='time_change_MAE' else 'pixels']
                        if value is None:empty+=1;continue
                        values.append(value);group[r['source_group']].append(value)
                    extra_aggregates.append(dict(dataset=ds,method=method,region=region,diagnostic=key,
                        valid_scenes=len(values),empty_scenes=empty,source_groups=len(group),pixel_observations=pixel_observations,
                        scene_equal_mean=float(np.mean(values)) if values else None,
                        source_group_equal_mean=float(np.mean([np.mean(v) for v in group.values()])) if group else None))
    a['additional_scene_and_source_group_aggregates']=extra_aggregates
    a['CPU_arithmetic_note']=f'PNG8->float32 RGB/255; per-pixel channel mean float32; regional sums/ratios float64. Existing PSNR/RM are preserved as stored float32 whole-array reductions. Largest pooled-MSE relative discrepancy: {max(pooled_mse_relative_deltas):.10g}; all {len(reduction_cases)} initial3e-6 check failures reproduce the literal old expression and agree with float64 CPU accumulation. Not a selection-rule change; see CPU-reduction-audit.json and preserved failed verify.log.'
    analysis_path=ROOT/record(c,'-A-analysis.json')
    tmp=analysis_path.with_suffix('.reporting.tmp')
    with tmp.open('x') as f:json.dump(a,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    tmp.replace(analysis_path)
    import PIL
    cpu_model=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    result=dict(checked_utc=now(),status='探索材料已准备，等待用户视觉复核',source_disjoint=True,exploration_scenes=len(explore),review_scenes=len(review),
        B_samples=len(b['rows']),materials=len(videos),videos=videos,CPU_scene_checks=True,max_existing_RM24_difference=max(rm_deltas),
        prediction_PNG_headers_checked=prediction_headers,max_pooled_MSE_relative_difference=max(pooled_mse_relative_deltas),
        float32_reduction_audit=record(c,'-CPU-reduction-audit.json'),initial_check_failures_reproduced=len(reduction_cases),
        user_visual_review_completed=False,visual_fields_all_pending=True,heldout_used_for_analysis_or_materials=False,
        no_model_inference=True,no_new_neural_metrics=True,formal_tables_and_PPT_unchanged=True,
        environment=dict(python=sys.version.split()[0],numpy=np.__version__,opencv=cv2.__version__,Pillow=PIL.__version__,
            imageio_ffmpeg=imageio_ffmpeg.__version__,ffmpeg=imageio_ffmpeg.get_ffmpeg_version(),CPU=cpu_model,
            CPU_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),memory_limit_bytes=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
            analysis_workers=4,opencv_threads_per_worker=1,encoding_threads=c['display']['threads'],GPU_used=False,
            starting_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),
        missing_items=['User B1 visual observations and semantic scene labels pending; B2 viewing deferred until B1 marks.',
                       'ATB per-clip public TorchMetrics PSNR/SSIM and author-first2 RM not available; left null, no new inference or estimates.',
                       'DAVIS paired bokeh GT and motion compensation unavailable; modification diagnostics cannot establish flicker.',
                       'Paper-attributed semantic difficulty coverage not asserted; input proxies are not human labels.',
                       'MagicBokeh DAVIS remains paused and absent; no fabricated panel.'],
        user_next='Open B1 only, record actual view mode, frame/transition, ROI and observations; export user marks before B2 access or rule freezing.')
    save(record(c,'-verification.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='videos'},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','analyze','materials','verify'])
    p.add_argument('--config',default='configs/any2bokeh-failure-analysis-20261009.json')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();os.chdir(ROOT);c=load(a.config);cv2.setNumThreads(1)
    if a.command=='prepare':prepare(c)
    elif a.command=='analyze':analyze(c,a.config,a.workers)
    elif a.command=='materials':materials(c)
    else:verify(c)


if __name__=='__main__':main()
