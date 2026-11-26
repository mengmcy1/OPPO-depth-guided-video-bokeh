"""Run matched development depth and optical diagnostics for registered E0 checkpoints."""
import argparse
from collections import defaultdict
import time

import numpy as np
from scipy import ndimage
import torch

import route3_ovg_data as ovg
import route3_v2data_module2_infer as module2
from route3_ovg_1c import aligned_input
from route3_v2data_B_inputs import to_srgb8
from route3_v2data_E_cache import LAYERS,load_encoder,write_arrays
from route3_v2data_E_labels import load,save_json,unit_paths
from route3_v2data_E_panels import crop,save_depth
from route3_v2data_E0_data import RUN,auxiliary_path,dataset,mapped,windows_for_mask
from route3_v2data_E0_train import (GPU_LIMIT,add_diagnostics,diagnostic_summary,frame_batch,
                                   gpu_gate,new_diagnostics,predict)
from route3_v2data_E0_tears import INPUT_CHANNELS,features,load_model
from route3_v2data_E0_windows import encode
from route3_v2data_stage1_evaluate import new_regions
from route3_v2data_stage2_eval import frames,geometry_root,legacy_masks
from route3_v2data_stage2_train import sample
from route3_v2data_stage3a import gpu_setup
from route3_v2data_stage3a_eval import guard_report
from src.ovg_connected_depth import rule_tears
from src.ovg_e import FEATURE_MODES,fixed_channels,rear_base
from src.ovg_edge_statistics import EdgeStatistics
from src.ovg_net import residual_attenuation
from src.ovg_tear_head import TearHead

FULL=("V0_pred","E","E_rule","E_learned","E_gt","V0_GT")
SINGLE=("V0_pred","E")
FIXED=("dev003","dev004","dev005","dev008","dev000","dev011")
LABELS=dict(V0_pred="预测 V0",E="E 单独",E_rule="E + 规则撕裂",
            E_learned="E + 重训撕裂头",E_gt="E + GT 撕裂",V0_GT="V0 GT")
GUARDS=("inverse_depth_rmse","same_surface_step","near_subject_coc_mae","GT_occlusion_crossing_abs_p95")


class EInputs:
    def __init__(self, mode, *, cache_run=RUN):
        self.mode=mode
        self.rows=load(cache_run/"data.json")["rows"]
        self.lookup={(r["scene"],r["frame"]):i for i,r in enumerate(self.rows)}
        self.encoder=None

    def native(self,name,frame):
        return frame_batch(self.rows,[(self.lookup[name,frame],0,0)],self.mode,512)

    def observations(self,name,frame,x,forecast):
        if (name,frame) in self.lookup:
            return self.native(name,frame)
        depth,confidence=(x[k] for k in ("prior_depth","prior_conf"))
        valid=torch.isfinite(depth)&(depth>0)&(confidence>0)
        rho=torch.where(valid,depth.clamp_min(1e-8).reciprocal(),0.)
        conf=torch.where(valid,confidence,0.)
        first,second=forecast["inverse1"][:,None],forecast["inverse2"][:,None]
        profile,mask=fixed_channels(x["rgb"],first,second,rho,conf)
        beta,source,mv=rear_base(first,second,rho,conf)
        result=dict(features=profile,mask=mask,phi=forecast["features"].half(),rho1=first,beta=beta,
                    attenuation=residual_attenuation(depth,confidence)[:,None],mv_source=source,mv_valid=mv)
        if self.mode!="E-L":
            if self.encoder is None:
                self.encoder=load_encoder()
            rgb=to_srgb8(x["rgb"][0].permute(1,2,0).cpu().numpy())
            image,_=self.encoder.image2tensor(np.ascontiguousarray(rgb[...,::-1]),518)
            global_context=torch.cat(self.encoder.pretrained.get_intermediate_layers(image,LAYERS,reshape=True),1).half()
            origins=windows_for_mask(mask[0,0].cpu().numpy()) if self.mode=="E-H2" else []
            local=[]
            if self.mode=="E-H2":
                tokens=torch.from_numpy(encode(self.encoder,rgb,origins)).cuda()
                local.append((tokens,torch.as_tensor(origins,device="cuda").reshape(-1,2)))
            result["context"]=(global_context,torch.zeros((1,2),device="cuda"),local)
        return result


def source_depth(inputs,out):
    return {"OVG":inputs["rho1"][0,0].cpu().numpy(),"E":out["decoded"][0,0].cpu().numpy()}


def legacy_targets(tgt,ref):
    device=tgt["depth1"].device
    z=torch.as_tensor(ref["gt"],device=device,dtype=torch.float32)[None,None]
    vs=torch.as_tensor(ref["valid"],device=device)[None,None] & torch.isfinite(z) & (z>0)
    inv=torch.where(vs,z.clamp_min(1e-8).reciprocal(),0.)
    zone=torch.full_like(inv,2,dtype=torch.uint8)
    boundary=torch.as_tensor(ref["boundary"],device=device)[None,None]
    loss_zone=torch.where(boundary,3,zone)
    return dict(visible=vs,zone=zone,loss_zone=loss_zone,center=inv,rod_reference=torch.zeros_like(inv))


def project_temporal(source,target,before,after):
    z0,z1=(r["depth_axial_m"].astype(np.float64) for r in (source,target))
    known=(z0>0)&source["center_visible"]
    yy,xx=np.nonzero(known)
    pixel=np.c_[xx+.5,yy+.5,np.ones(xx.shape)]
    local=(pixel@np.linalg.inv(source["K"]).T)*z0[yy,xx,None]
    camera=np.linalg.inv(target["camera_c2w"])@source["camera_c2w"]
    moved=local@camera[:3,:3].T+camera[:3,3]
    uv=moved@target["K"].T
    front=moved[:,2]>0
    xy=np.divide(uv[:,:2],uv[:,2:],where=front[:,None],out=np.full_like(uv[:,:2],-1))
    xi,yi=np.floor(xy).astype(np.int64).T
    inside=front&(xi>=0)&(xi<512)&(yi>=0)&(yi<512)
    xi,yi=xi[inside],yi[inside]
    yy,xx=yy[inside],xx[inside]
    expected=moved[inside,2]
    consistent=target["center_visible"][yi,xi]&np.isclose(z1[yi,xi],expected,rtol=.01,atol=.002)
    if "object_id" in source and "object_id" in target:
        consistent&=source["object_id"][yy,xx]==target["object_id"][yi,xi]
    flips=(before[yy,xx]!=after[yi,xi])&consistent
    return int(flips.sum()),int(consistent.sum())


def temporal_report(network,inputs,geometry):
    plan=dataset()
    paths=unit_paths(plan)
    result=[]
    pairs=((29,30),(30,31))
    total=counts=cases=0
    for name in plan["development"]:
        unit=paths[name]
        files={f:unit/"labels"/f"f{f:03d}-pinhole.npz" for f in (29,30,31)}
        missing=[f for f,p in files.items() if not p.is_file()]
        planned=[p for p in pairs if not set(p)&set(missing)]
        row=dict(scene=name,missing_GT_center_frames=missing,pairs=[])
        for a,b in planned:
            frames=[]
            for f in (a,b):
                if (unit/"observations/priors"/f"f{f:03d}.npz").is_file():
                    frames.append(f)
            if len(frames)!=2:
                row["pairs"].append(dict(frames=[a,b],status="missing observed priors"))
                continue
            data=[]
            for f in (a,b):
                gpu_gate()
                other=sorted((q for q in ovg.TRAIN_FRAMES if q!=f),key=lambda q:(abs(q-f),q))[:2]
                obs=ovg.input_record(aligned_input(name,f),[aligned_input(name,q) for q in other])
                x={k:torch.as_tensor(v[None].copy(),device="cuda") for k,v in obs.items()}
                forecast=geometry(x,protect_residual=True,render_heads=False,return_features=True)
                cached=inputs.observations(name,f,x,forecast)
                selected=predict(network,cached)["selected_back"][0,0].cpu().numpy()
                with np.load(files[f]) as gt:
                    data.append((dict(gt),selected))
            flips,valid=project_temporal(data[0][0],data[1][0],data[0][1],data[1][1])
            row["pairs"].append(dict(frames=[a,b],changes=flips,correspondences=valid))
            total+=flips
            counts+=valid
            cases+=1
        result.append(row)
    return dict(frame_pairs=pairs,evaluated_pairs=cases,scenes=result,changes=total,
                correspondences=counts,ratio=total/counts if counts else None)


def panel_depth(root,name,x,out):
    def unpack(t):
        return t[0,0].cpu().numpy()
    fields=dict(center=unpack(x["center"]),rho1=unpack(x["rho1"]),E_decoded=unpack(out["decoded"]),
                E_front=unpack(out["means"][:,:1].exp()),E_back=unpack(out["means"][:,1:].exp()),
                w=unpack(out["log_mass"][:,1:].exp()),selected_back=unpack(out["selected_back"]),
                zone=unpack(x["zone"]),mv_source=unpack(x["mv_source"]),gate=unpack(out["gate"]))
    z=fields["zone"]
    support=unpack(x["visible"])
    boxes=(crop((z==0)&support if (z==0).any() else (unpack(x["loss_zone"])==3)&support),
           crop((z==1)&support if (z==1).any() else (z==2)&support))
    output=root/"depth-panels"
    output.mkdir(parents=True,exist_ok=True)
    panel=save_depth(module2.title(name),{"rgb":x["features"][0,:3].cpu().numpy()},fields,boxes,2.,output=output)
    write_arrays(panel.with_suffix(".npz"),dict(fields,visible=support,boxes=np.asarray(boxes)))
    return str(panel)


def add_guard_stats(rows,edges,coc,method,name,depth,target,ref,inputs,unit,cases):
    z=target["depth1"][0].cpu().numpy().astype(np.float32)
    valid=ref["valid"]&np.isfinite(z)&(z>0)
    disparity=np.zeros_like(depth,dtype=np.float64)
    disparity[valid]=(depth[valid].astype(np.float64)-1/z[valid].astype(np.float64))
    rows.append(dict(sse=float(np.square(disparity[valid]).sum()),pixels=int(valid.sum())))
    edges.add(name,method,1/depth,ref,valid)
    k=inputs["K"][0].cpu().numpy()
    for c in cases:
        if c["focus_index"]:
            continue
        regions=(new_regions(unit,c,ref["gt"],ref["valid"],k) if unit else legacy_masks(geometry_root(name),c))
        area=regions["in_focus"] & valid & np.isin(ref["object_id"],ref["foreground"])
        nominal=c["focus_m"]
        coef=float(k[0,0])*.05/(2*c["fstop"])*nominal/(nominal-.05)
        coc.append(dict(coc_guard=dict(pixels=int(area.sum()),absolute_error_sum=float(np.abs(disparity[area]).sum()*coef))))


@torch.inference_mode()
def depth_report(root,network,inputs,smoke=False):
    stats=defaultdict(new_diagnostics)
    edges=defaultdict(EdgeStatistics)
    rows,coc=defaultdict(list),defaultdict(list)
    panels=[]
    geometry=module2.load_geometry()
    for cohort,name,frameno,unit,entry,cases in frames():
        if smoke and (module2.title(name) not in ("dev005","122104") or frameno!=30):
            continue
        gpu_gate()
        item=("B",name,ovg.TRAIN_FRAMES.index(frameno))
        x,target=sample(item)
        ref=ovg.b_edge_reference(item,target)
        if unit:
            cached=inputs.native(name,frameno)
        else:
            forecast=geometry(x,protect_residual=True,render_heads=False,return_features=True)
            cached=inputs.observations(name,frameno,x,forecast)
            cached.update(legacy_targets(target,ref))
        current=predict(network,cached)
        add_diagnostics(stats[cohort],current,cached)
        label=module2.title(name)
        if label in ("dev003","dev004","dev005","dev008"):
            add_diagnostics(stats["dev008" if label=="dev008" else "thin_structure"],current,cached)
        for method,depth in source_depth(cached,current).items():
            key=cohort,method
            add_guard_stats(rows[key],edges[key],coc[key],method,name,depth,target,ref,x,unit,cases)
        if unit and frameno==30 and label in FIXED:
            panels.append(panel_depth(root,name,cached,current))
        print("E0_DEPTH",label,frameno,flush=True)
    receipts={}
    if not smoke:
        for cohort,reference in stats.items():
            if cohort not in ("development","legacy"):
                continue
            g={method:guard_report(rows[cohort,method],coc[cohort,method],edges[cohort,method])
               for method in ("OVG","E")}
            receipts[cohort]=dict(g,delta={k:g["E"][k]-g["OVG"][k] for k in GUARDS})
    return dict(cohorts={k:diagnostic_summary(v) for k,v in stats.items()},guards=receipts,
                temporal=temporal_report(network,inputs,geometry),panels=panels)


def optical_adapter(network,inputs,arms,tear):
    def adapter(name,frame,unit,x,forecast,label):
        cached=inputs.observations(name,frame,x,forecast)
        out=predict(network,cached)
        depth=out["decoded"][0,0]
        if not torch.isfinite(depth).all() or not (depth>0).all():
            raise FloatingPointError(f"Invalid decoded E geometry for {name}/{frame}")
        field={}
        if "E_rule" in arms:
            field["E_rule"]=rule_tears(depth.cpu().numpy(),.10)
        if "E_gt" in arms:
            field["E_gt"]=label
        if "E_learned" in arms:
            feature=features(out,x["rgb"]).half().float()
            field["E_learned"]=(tear(feature).sigmoid()[0]>.5).cpu().numpy().astype(np.int8)
        return depth.reciprocal(),tuple(a for a in arms if a.startswith("E")),field
    return adapter


def panel_optical(name,frame,path,row,pictures,truth,rgb,masks,overlays,arms):
    panel=module2.panel(path,row,pictures,truth,rgb,masks,overlays,arms)
    if row["focus_index"]==1:
        zone=mapped(auxiliary_path(name,frame),"zone")[0]
        rods=zone==0
        if rods.any():
            support=ndimage.binary_dilation(rods,iterations=2)
            def contrast(image):
                gray=image.mean(2)
                length=np.hypot(ndimage.sobel(gray,axis=0),ndimage.sobel(gray,axis=1))/8
                return float(np.mean(length[support]))
            panel["rod_border_gradient"]={k:contrast(v) for k,v in dict(pictures,Cycles=truth,RGB=rgb).items()}
    return panel


@torch.inference_mode()
def optics(root,network,inputs,arms,tear=None,smoke=False):
    geometry=module2.load_geometry()
    adapter=optical_adapter(network,inputs,arms,tear)
    module2.ARM_NAMES.update(LABELS)
    parts=[]
    skipped=[]
    for cohort,name,frame,unit,entry,cases in frames():
        shown=module2.title(name)
        if smoke and (shown not in ("dev005","122104") or frame!=30):
            continue
        if not (module2.CACHE/name/f"f{frame:03d}.npz").is_file():
            if cohort=="legacy":
                skipped.append(dict(scene=name,frame=frame,reason="center-aligned midpoint tear labels unavailable"))
                continue
            raise FileNotFoundError(f"Missing E0 tear reference labels for {name}/{frame}")
        cache=root/"frames"/name/f"f{frame:03d}.json"
        if cache.is_file():
            part=load(cache)
        else:
            gpu_gate()
            def draw(*args):
                return panel_optical(name,frame,*args) if shown in FIXED and frame==30 else {}
            part=module2.frame_results(geometry,cohort,name,frame,unit,entry,cases,root,arms,
                                       geometry_override=adapter,panel_writer=draw)
            save_json(cache,part)
        parts.append(part)
        print("E0_OPTICS",shown,frame,flush=True)
    pairing={a:"V0_pred" for a in arms if a!="V0_pred"}
    result=module2.summarize(parts,arms,pairings=pairing,include_absolute_near=True)
    special={scope:module2.acc_stats([p for p in parts if p["rows"] and p["rows"][0]["cohort"]==scope])
             for scope in ("development","legacy")}
    return dict(summary=result,by_case=[r for p in parts for r in p["rows"]],skipped_legacy=skipped,
                tear_audit=special,compatibility=dict(GT_substitute="FP32 masked reciprocal, 6cff9f9",
                    optical="P2 f/1.4 K64; separate native center focusing for OVG, E, GT; K16 probes",
                    pooled_psnr="SSE/(3*pixels)"))


@torch.inference_mode()
def evaluate(mode,step,kind,smoke=False):
    case=RUN/mode/f"evaluation-{step:06d}-{kind}{'-smoke' if smoke else ''}"
    out=case/"report.json"
    if out.is_file():
        return load(out)
    gpu_setup(dict(training=dict(max_process_gpu_gib=GPU_LIMIT)))
    waited=gpu_gate()
    start=time.monotonic()
    before=module2.competitor_state()
    network,_=load_model(mode,step)
    case.mkdir(parents=True,exist_ok=True)
    inputs=EInputs(mode)
    if kind=="depth":
        result=depth_report(case,network,inputs,smoke)
    else:
        tear=None
        arms=SINGLE if kind=="E-only" else FULL
        if kind=="full":
            tear=TearHead(INPUT_CHANNELS).cuda().eval()
            archive=RUN/mode/f"tear-{step:06d}/selected.pt"
            tear.load_state_dict(torch.load(archive,weights_only=True,map_location="cpu")["ema"])
        result=optics(case,network,inputs,arms,tear,smoke)
    record=dict(result,mode=mode,step=step,kind=kind,wall_seconds=time.monotonic()-start,
                initial_gpu_wait_seconds=waited,competitor_before=before,competitor_after=module2.competitor_state())
    save_json(out,record)
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=FEATURE_MODES)
    p.add_argument("step",type=int)
    p.add_argument("kind",choices=("depth","E-only","full"))
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    evaluate(a.mode,a.step,a.kind,a.smoke)
