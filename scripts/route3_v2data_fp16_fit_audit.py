"""Compare archived and corrected native fits and selectively replay Stage 3a."""
import argparse
import ast
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt,uniform_filter

import route3_ovg_data as ovg
from route3_ovg_1b_optical_data import b_oracle,place
from route3_s2a_candidates import local_placement
from route3_ovg_subpixel import save_json as write_json
from route3_v2data_B_batch import ROOT,sha256
from route3_v2data_B_plan import load
from route3_v2data_stage1_evaluate import load_paired_checkpoint
from route3_v2data_stage2_train import INITIAL,sample
from route3_v2data_stage3a import checkpoint,checkpoint_model,load_parents
from route3_v2data_stage3a_eval import frames
from route3_v2data_stage3b_eval import START_GT
import route3_v2data_module2_infer as module2

RUN=ROOT/"outputs/route3-v2data-fp16-fit-2026-11-26-01"
ARCHIVE="1f61930"
STAGE3A=ROOT/"outputs/route3-v2data-stage3a-2026-11-24-02"
FIRST=ROOT/"outputs/route3-v2data-stage3a-2026-11-24-01"
STAGE3B=ROOT/"outputs/route3-v2data-stage3b-2026-11-25-01"
START_FIXED=START_GT.with_name("cases-fp16-fit-corrected.json")
PENDING=START_GT.with_name("fp16-fit-correction.pending")


@lru_cache(maxsize=1)
def archived_local_placement():
    source="scripts/route3_s2a_candidates.py"
    raw=subprocess.check_output(["git","show",f"{ARCHIVE}:{source}"],cwd=ROOT,text=True)
    tree=ast.parse(raw)
    definition=next(f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=="local_placement")
    space=dict(np=np,uniform_filter=uniform_filter,distance_transform_edt=distance_transform_edt)
    exec(compile(ast.Module(body=[definition],type_ignores=[]),source,"exec"),space)
    return space["local_placement"]


def gpu_budget():
    """Allow the GT-reference producer to finish while a cold evaluator waits."""
    if PENDING.is_file() and not START_FIXED.is_file():
        used=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                                          text=True).strip())
        if used>22*1024:
            torch.cuda.empty_cache()
            while used>22*1024:
                time.sleep(5)
                used=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"],text=True).strip())
    else:
        module2.yield_gpu()


def background_input(run,unit,name,frame,case,ref):
    if unit is None:
        return b_oracle(name,frame)
    path=run/"dev-hidden-parents"/name/(case["id"]+"-optics.npz")
    if not path.is_file():
        # The optics already archived an empty parent set for these cases.
        path=STAGE3A/"dev-hidden-parents"/name/(case["id"]+"-optics.npz")
    parents=load_parents(path)
    background=ref["valid"]&~np.isin(ref["object_id"],ref["foreground"])
    return parents,ref["gt"],background


def fits(gt,depth,background):
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        before=archived_local_placement()(gt,depth,background,return_valid=True)
    after=local_placement(gt,depth,background,return_valid=True)
    if not np.isfinite(after[0]).all() or not np.isfinite(after[1]).all():
        raise ValueError("Corrected local fit has nonfinite values")
    return before,after


def compare(parents,old,new):
    uv=parents["uv"]
    x,y=np.floor(np.where(np.isfinite(uv),uv,-1.)).astype(int).T
    h,w=new[0].shape
    inside=(x>=0)&(x<w)&(y>=0)&(y<h)&np.isfinite(uv).all(1)
    x,y=x.clip(0,w-1),y.clip(0,h-1)
    expected=parents["recoverable"]&inside
    def passed(fit):
        a,b,_,v=fit
        with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
            inverse=a[y,x]/parents["depth"]+b[y,x]
        supported=expected&v[y,x]
        good=supported&np.isfinite(inverse)&(inverse>1e-4)
        return inverse,supported,good
    old_inv,old_support,a=passed(old)
    new_inv,new_support,b=passed(new)
    common=a&b
    with np.errstate(over="ignore",invalid="ignore"):
        delta=np.abs(1/old_inv[common]-1/new_inv[common])
    changed=int(np.count_nonzero(delta))
    return dict(eligible=int(parents["recoverable"].sum()),unsupported=int((expected&~new_support).sum()),
                old_fit_nonfinite_pixels=int((~np.isfinite(old[0])|~np.isfinite(old[1])).sum()),
                old_candidate_nonfinite=int((old_support&~np.isfinite(old_inv)).sum()),
                new_candidate_nonfinite=int((new_support&~np.isfinite(new_inv)).sum()),
                old_placed=int(a.sum()),new_placed=int(b.sum()),recovered=int((b&~a).sum()),
                lost=int((a&~b).sum()),changed_shared_depths=changed,
                max_depth_difference_m=float(delta.max(initial=0.)),
                needs_rescore=bool((a!=b).any() or changed))


def summary_counts(rows):
    return {scope:dict(cases=len(own),rescore=sum(r["needs_rescore"] for r in own),
        overflow=sum(r["old_candidate_nonfinite"]>0 for r in own),
        fit_nonfinite=sum(r["old_fit_nonfinite_pixels"]>0 for r in own),
        placement_changed=sum(r["old_placed"]!=r["new_placed"] for r in own),
        recovered=sum(r["recovered"] for r in own),lost=sum(r["lost"] for r in own))
        for scope in ("development","legacy","training")
        for own in ([r for r in rows if r["cohort"]==scope],) if own}


def verification(name,case,parents,gt,bg,depth,K,report):
    import route3_ovg_1b_optical_data as optical
    before=optical.local_placement
    try:
        optical.local_placement=archived_local_placement()
        with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
            old,old_stats=place(parents,gt,bg,depth,K,512)
    finally:
        optical.local_placement=before
    fixed,new_stats=place(parents,gt,bg,depth,K,512)
    if [old_stats["placed"],new_stats["placed"]]!=[report["old_placed"],report["new_placed"]]:
        raise ValueError("Audit does not match actual placement")
    equal=old is not None and fixed is not None and all(
        torch.equal(getattr(old,key),getattr(fixed,key)) for key in old.__dataclass_fields__)
    if not report["needs_rescore"] and not equal:
        raise ValueError("Unchanged placement does not match elementwise")
    if name.startswith("development-005") and case["id"]=="f001-focus1-N1.4":
        if old_stats["placed"]!=0 or new_stats["placed"]!=13191 or fixed is None:
            raise ValueError("The registered regression case did not recover")
    return dict(scene=name,case=case["id"],eligible=new_stats["eligible"],old_placed=old_stats["placed"],
                new_placed=new_stats["placed"],all_cloud_fields_bitwise_equal=equal)


def replay_3a(cloud, inputs, unit,name,frame,entry,case,ref,labels,row,lpips_model):
    from route3_upper_bound_evaluate import score
    from route3_v2data_stage3a_eval import render
    from route3_v2data_stage1_evaluate import new_regions
    from route3_v2data_stage2_eval import bands, f2_for, geometry_root, legacy_masks, optical_target
    from src.aperture_renderer import aperture_disk
    from src.c1_bokehme import region_sse
    from src.layered_aperture import merge_layers
    import route3_v2data_stage1_visualize as gallery
    visible,hidden=cloud
    K=inputs["K"][0]
    index=case["focus_index"]
    focus=row["focus_m"]
    truth=optical_target(unit,name,case)
    k=K.cpu().numpy()
    regions=new_regions(unit,case,ref["gt"],ref["valid"],k) if unit else legacy_masks(geometry_root(name),case)
    area="near_in_focus" if index==0 else "far_recoverable"
    full=np.ones(ref["gt"].shape,bool)
    masks=dict(full_image=full,boundary_2px=bands(ref,labels)[0])
    masks[area]=regions["in_focus"] if index==0 else regions["recoverable"]
    pupil=aperture_disk(64,.05/(2*1.4),device=K.device,dtype=K.dtype)
    center=K.new_zeros((1,2))
    scene=merge_layers(visible,hidden)
    rgb=render(scene,K,focus,pupil,f2_for(unit,entry,frame,inputs,focus,pupil))
    zero=render(scene,K,focus,center,f2_for(unit,entry,frame,inputs,focus,center))
    pinhole=gallery.display(inputs["rgb"][0].permute(1,2,0).cpu().numpy())
    updated=score(rgb,truth,masks,lpips_model)
    updated["zero_combined"]=region_sse(zero,pinhole,full)
    actual=row["scores"]["stage3"]["metrics"]
    for key in updated:
        if updated[key]["pixels"]!=actual[key]["pixels"]:
            raise ValueError(f"Changed mask support on {name}/{case['id']}/{key}")
    actual.update(updated)


@torch.inference_mode()
def stage3a(step,weights,run=STAGE3A,*,scan_only=False):
    from route3_ldr_u1r import model as lpips_model
    from route3_v2data_stage3a import gpu_setup
    from src.ovg_optics import fixed_visible_surfels
    gpu_setup(dict(training=dict(max_process_gpu_gib=3 if scan_only else 6)))
    began=time.monotonic()
    registered=run/"evaluation"/f"step-{step:06d}"/weights
    source=registered/"cases.json"
    out=RUN/("scan" if scan_only else "stage3a")/f"step-{step:06d}"/weights
    original=load(source)
    old={(r["scene"],r["case"]):r for rows in original.values() for r in rows}
    corrected=deepcopy(original)
    new={(r["scene"],r["case"]):r for rows in corrected.values() for r in rows}
    ck=checkpoint(run,step) if step else INITIAL
    model=(checkpoint_model(ck,weights) if step else load_paired_checkpoint(INITIAL,"ema")).eval()
    lp=None if scan_only else lpips_model()
    reports,checked=[],[]
    completed=0
    for cohort,name,frame,unit,entry,cases in frames():
        cached=load(out/"frames"/name/f"f{frame:03d}.json")
        if cached:
            reports.extend(cached["reports"])
            checked.extend(cached["verified"])
            for row in cached["updated"]:
                new[row["scene"],row["case"]].update(row)
            completed+=1
            continue
        gpu_budget()
        item=("B",name,ovg.TRAIN_FRAMES.index(frame))
        inputs,labels=sample(item)
        pred=model(inputs,protect_residual=True,render_heads=False)
        depth=pred["depth1"][0]
        z=depth.cpu().numpy()
        ref=ovg.b_edge_reference(item,labels)
        K=inputs["K"][0]
        previous=None
        visible=None
        per_frame,validated,updated=[],[],[]
        for case in cases:
            ident=(name,case["id"])
            rem=new[ident]
            if not rem["scores"]:
                continue
            parents,gt,bg=background_input(run,unit,name,frame,case,ref)
            if previous is None:
                previous=fits(gt,z,bg)
            delta=compare(parents,*previous)
            delta.update(scene=name,frame=frame,case=case["id"],focus_index=case["focus_index"],cohort=cohort,dtype=str(gt.dtype))
            archived=old[ident]["gt_hidden"]
            if (archived["placed"],archived["eligible"])!=(delta["old_placed"],delta["eligible"]):
                raise ValueError(f"Old native placement mismatch {name}/{case['id']}: {archived} vs {delta}")
            if not scan_only and step==0 and ((name.startswith("development-005") and case["id"]=="f001-focus1-N1.4") or
                            (cohort=="legacy" and frame==30 and case["focus_index"]==1)):
                validated.append(verification(name,case,parents,gt,bg,depth,K,delta))
            if delta["needs_rescore"] and not scan_only:
                gpu_budget()
                if visible is None:
                    visible=fixed_visible_surfels(pred,inputs)
                hidden,receipt=place(parents,gt,bg,depth,K,512)
                if receipt["placed"]!=delta["new_placed"]:
                    raise ValueError("Corrected audit and renderer disagree on placement")
                replay_3a((visible,hidden),inputs,unit,name,frame,entry,case,ref,labels,rem,lp)
                rem["gt_hidden"]=receipt
                updated.append(rem)
            per_frame.append(delta)
        reports.extend(per_frame)
        checked.extend(validated)
        write_json(out/"frames"/name/f"f{frame:03d}.json",dict(reports=per_frame,verified=validated,updated=updated))
        completed+=1
        print("FP16_SCAN" if scan_only else "FP16_STAGE3A",step,weights,completed,name,frame,
              sum(d["needs_rescore"] for d in per_frame),flush=True)
    if not scan_only and not step and len(checked)!=4:
        raise ValueError(f"Expected the regression case and 3 unchanged optical cases, got {len(checked)}")
    if not scan_only:
        write_json(out/"cases.json",corrected)
    write_json(out/"audit.json",dict(step=step,weights=weights,checkpoint_sha256=sha256(ck),
              summary=summary_counts(reports),cases=reports,verification=checked,seconds=time.monotonic()-began))
    if not scan_only and not step and run==STAGE3A:
        write_json(START_FIXED,corrected)
        done=PENDING.with_name("fp16-fit-correction.ready")
        if PENDING.is_file():
            PENDING.replace(done)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--weights",choices=("ema","model"),default="ema")
    p.add_argument("--scan",action="store_true")
    a=p.parse_args()
    stage3a(a.step,a.weights,scan_only=a.scan)
