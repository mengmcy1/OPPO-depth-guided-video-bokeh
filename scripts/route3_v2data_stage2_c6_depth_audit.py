"""Reproduce C6 pinhole scores and stratify mixed-boundary error."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

import route3_v2data_stage1_visualize as gallery
import route3_v2data_stage2_train as stage2
import route3_v2data_stage2_visualize_split5 as split5
from route3_upper_bound_evaluate import display
from route3_v2data_B_batch import GIB, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_stage1_evaluate import p2_focus, pooled
from route3_v2data_stage2_eval import f2_for
from src.layered_aperture import merge_layers
from src.ovg_optics import raster


LABELS = {
    "predicted_depth_ratio": ("<1.00", "1.00–<1.03", "≥1.03"),
    "background_depth_ratio": ("<1.00", "1.00–<1.03", "≥1.03"),
    "gt_alpha": ("(0,0.1]", "(0.1,0.5]", "(0.5,0.9]", "(0.9,1)"),
    "split_neighbors4": ("孤立", "有拆分邻点"),
}


def depth_bins(depth, front):
    return np.where(depth < front, 0, np.where(depth < 1.03*front, 1, 2))


def adjacent(mask):
    neighbors = np.zeros_like(mask)
    neighbors[1:] |= mask[:-1]
    neighbors[:-1] |= mask[1:]
    neighbors[:,1:] |= mask[:,:-1]
    neighbors[:,:-1] |= mask[:,1:]
    return neighbors


def groups(geom, first, labels, split):
    front = geom["depth1"][0].cpu().numpy()
    back = geom["depth2"][0].cpu().numpy()
    alpha = labels["alpha"][0].cpu().numpy()
    if not (np.isfinite(front[split]).all() and np.isfinite(back[split]).all()
            and np.isfinite(first["depth"][split]).all() and (front[split]>0).all()):
        raise ValueError("Invalid depth at a C6 split pixel")
    return {
        "predicted_depth_ratio": depth_bins(back, front)[split],
        "background_depth_ratio": depth_bins(first["depth"], front)[split],
        "gt_alpha": np.searchsorted([.1,.5,.9], alpha[split], side="left"),
        "split_neighbors4": adjacent(split)[split].astype(np.int8),
    }


def summarize(values):
    if not values:
        raise ValueError("No measurements from development cases")
    sse = np.concatenate([v[1] for v in values])
    linear = np.concatenate([v[2] for v in values])
    total = float(sse.sum())
    result = {}
    for key, labels in LABELS.items():
        bucket = np.concatenate([v[0][key] for v in values])
        result[key] = []
        for i,label in enumerate(labels):
            mask = bucket==i
            positive = float(sse[mask].sum())
            p50,p95 = np.quantile(linear[mask], [.5,.95]).tolist() if mask.any() else (None,None)
            result[key].append(dict(group=label,pixels=int(mask.sum()),sse=positive,
                                    sse_fraction=positive/total if total else 0.,p50=p50,p95=p95))
        assert sum(v["pixels"] for v in result[key])==len(sse)
        np.testing.assert_allclose(sum(v["sse"] for v in result[key]),total,rtol=1e-12,atol=1e-12)
    return dict(split_pixels=len(sse),split_sse=total,groups=result)


@torch.inference_mode()
def replay(output, source):
    stage2.configuration()
    torch.cuda.set_per_process_memory_fraction(6*GIB/torch.cuda.get_device_properties(0).total_memory,0)
    torch.backends.cuda.matmul.allow_tf32=False
    stage2.fixed_inputs()
    cp=stage2.RUN/"2a/step-010000.pt"
    ref=load(source/"opacity-calibration/report.json")
    assert sha256(cp)==ref["checkpoint_sha256"]
    original=load(source/"metrics.json")
    originals={r["file"]:r for r in original["rows"] if not r.get("error")}
    model=stage2.Model(stage2.RUN)
    model.heads.load_state_dict(torch.load(cp,map_location="cpu",weights_only=True)["ema"])
    model.heads.eval()
    values=[]
    rows=[]
    skips=[dict(file=r["file"],error=r["error"]) for r in original["rows"] if r.get("error")]
    audits={}
    started=time.monotonic()
    for plan in gallery.planned_frames():
        active=[(i,originals[r["file"]]) for i,r in enumerate(plan["files"]) if r["file"] in originals]
        if not active:
            continue
        try:
            state=split5.frame_setup(model,plan,"2a",only_c6=True)
            x,y,g,_,_,subjects,frames,_,_,split,rgb,_,mixture=state
            K=x["K"][0]
            pinhole=display(rgb)
            first=split5.first_observation(model,plan,g)
            grouping=groups(g,first,y,split)
        except Exception as error:
            skips.extend(dict(file=r["file"],error=str(error)) for _,r in active)
            continue
        for i,r in active:
            try:
                case=plan["cases"][i]
                focus=r["actual_focus_m"]["S3"]
                computed=p2_focus(g["depth1"][0].cpu().numpy(),subjects[i])
                if computed is None or abs(focus-computed)>1e-6:
                    raise ValueError("Actual P2 focus differs from the calibrated baseline")
                offset=K.new_zeros((1,2))
                f2=f2_for(plan["unit"],plan["entry"],plan["frame"],x,focus,offset)
                hidden=model.second(split5.item_for(plan),g["depth1"][0].cpu().numpy(),focus,case["fstop"])
                zero=raster(merge_layers(frames["C6"],hidden),K,focus,offset,512,512,f2)[0].cpu().numpy()
                display_error=display(zero).astype(np.float64)-pinhole.astype(np.float64)
                pixel_sse=np.square(display_error).sum(-1)
                difference=np.abs(zero[split].astype(np.float64)-mixture[split]).max(axis=-1)
                if r["file"] in split5.PINHOLE_CASES:
                    audits[r["file"]]=split5.pinhole_audit(output,r,zero,mixture,split)
                if plan["unit"]:
                    values.append((grouping,pixel_sse[split],difference))
                rows.append(dict(file=r["file"],scene=plan["name"],split_pixels=int(split.sum()),
                                 split_sse=float(pixel_sse[split].sum()),sse=float(pixel_sse.sum()),
                                 pixels=int(pixel_sse.size)))
                print("DEPTH_AUDIT",r["file"],round(rows[-1]["split_sse"],5),flush=True)
            except Exception as error:
                skips.append(dict(file=r["file"],error=str(error)))
                print("DEPTH_AUDIT_SKIP",r["file"],str(error),flush=True)
        del state
    result=summarize(values)
    totals={}
    for group in ("development","legacy","all"):
        scope=rows if group=="all" else [r for r in rows if r["scene"].startswith("development-")==(group=="development")]
        cases=[pooled([r]) for r in scope]
        totals[group]=dict(**pooled(scope),cases=len(cases),pass_60=sum(r["psnr_db"]>=60 for r in cases))
    assert totals["development"]["cases"]+totals["legacy"]["cases"]==totals["all"]["cases"]
    result.update(checkpoint_sha256=sha256(cp),totals=totals,audits=audits,cases=rows,skipped=skips,
                  seconds=time.monotonic()-started,
                  scope="All scored C6 cases from the visual set. The development strata use split pixels and count each focus case separately.",
                  sse_definition="RGB SSE relative to the pinhole input in display space. Fractions use SSE on all development C6 split pixels.",
                  difference_definition="Per-pixel maximum absolute linear RGB channel difference from GT alpha*F+(1-alpha)*B.")
    pat=result["groups"]["predicted_depth_ratio"]
    used=result["groups"]["background_depth_ratio"]
    result["under_1_03_predicted_sse_fraction"]=sum(r["sse_fraction"] for r in pat[:2])
    result["under_1_03_used_sse_fraction"]=sum(r["sse_fraction"] for r in used[:2])
    result["run_depth_fix"]=result["under_1_03_predicted_sse_fraction"]>=.5
    result["full_pinhole_pass"]=totals["all"]["psnr_db"]>=60
    result["audit_p95_pass"]=set(audits)==split5.PINHOLE_CASES and all(r["p95"]<.01 for r in audits.values())
    write_json(output/"report.json",result)
    print("DEPTH_RESULT",result["under_1_03_predicted_sse_fraction"],result["under_1_03_used_sse_fraction"],
          result["run_depth_fix"],totals,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=gallery.OUT)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    replay(args.output.resolve(),args.source.resolve())
