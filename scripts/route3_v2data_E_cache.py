"""Check native OVG forwards, report rear anchors, and cache independent full-frame features."""
import argparse
from collections import Counter
import os
import shutil
import sys
import time

import numpy as np
import torch

import route3_ovg_data as ovg
from route3_v2data_B_batch import GIB, sha256
from route3_v2data_B_inputs import observed_rgb,to_srgb8
from route3_v2data_E_labels import REGIONS, ROOT, RUN, core_labels, load, save_json, stats, unit_paths
from route3_v2data_module2_infer import competitor_state, yield_gpu
from route3_v2data_stage2_train import INITIAL
from route3_v2data_stage3a import gpu_setup
from route3_v2data_stage3b import load_geometry
from route3_ovg_1c import aligned_sample
from src.ovg_e import rear_base
from src.ovg_net import residual_attenuation

REVISION = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
LAYERS = (2,5,8,11)
DA_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
DA_WEIGHTS = ROOT/"models/e-da2-small/depth_anything_v2_vits.pth"
DA_VENDOR = ROOT.parent/"route3-cache"/"vendor"/f"Depth-Anything-V2-{REVISION}"
LIMIT = 4.5


def setup():
    gpu_setup(dict(training=dict(max_process_gpu_gib=LIMIT)))


def cache_path(kind, name, frame):
    return RUN/"cache"/kind/name/f"f{frame:03d}.npz"


def write_arrays(path, arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".part")
    with temp.open("wb") as stream:
        np.savez(stream,**arrays)
    os.replace(temp,path)


def cache_arrays(path):
    with np.load(path) as a:
        return {key:a[key] for key in a.files}


def on_device(array):
    return torch.as_tensor(array,device="cuda")


@torch.inference_mode()
def online_ovg(network, name, frame):
    item=("B",name,ovg.TRAIN_FRAMES.index(frame))
    x,_=aligned_sample(item,"cuda")
    p=network(x,protect_residual=True,render_heads=False,return_features=True)
    depth,confidence=(x[k] for k in ("prior_depth","prior_conf"))
    valid = torch.isfinite(depth)&(depth>0)&(confidence>0)
    prior=torch.where(valid,depth.clamp_min(1e-8).reciprocal(),0.)
    values=dict(rho1=p["inverse1"][0],rho2=p["inverse2"][0],phi_full=p["features"][0],
                prior=prior[0],confidence=torch.where(valid,confidence,0.)[0],
                rgb=x["rgb"][0],attenuation=residual_attenuation(depth,confidence)[0])
    if (values["phi_full"].shape!=(64,512,512) or any(not t.isfinite().all() for t in values.values())
            or (values["rho1"]<=0).any() or (values["rho2"]<=0).any()):
        raise ValueError(f"{name}/f{frame:03d} invalid native OVG prediction")
    return {k:v.cpu().numpy() for k,v in values.items()}


def packed(values):
    return {k:v.astype(np.float16) if k=="phi_full" else v for k,v in values.items()}


def beta_arrays(arrays, margin):
    rho1,rho2=[on_device(arrays[k])[None,None] for k in ("rho1","rho2")]
    priors,confidence=[on_device(arrays[k])[None] for k in ("prior","confidence")]
    beta,source,valid=rear_base(rho1,rho2,priors,confidence,margin)
    return tuple(t[0,0].cpu().numpy() for t in (beta.exp(),source,valid))


def new_regions():
    return {key:dict(counts=Counter(),errors=[]) for key in REGIONS}


@torch.inference_mode()
def audit_ovg():
    inventory=load(RUN/"inventory.json")
    label_report=load(RUN/"labels.json")
    frames=label_report["frames"]
    margin=label_report["m"]
    paths=unit_paths(inventory)
    begin=time.monotonic()
    before=competitor_state()
    setup()
    waited=yield_gpu()
    digest=sha256(INITIAL)
    if not digest.startswith("dfb1d7e5"):
        raise ValueError("Wrong frozen OVG EMA")
    receipt=RUN/"ovg-source.json"
    if receipt.is_file() and load(receipt)["geometry_sha256"]!=digest:
        raise ValueError("Mismatched geometry provenance on restart")
    save_json(receipt,dict(geometry_sha256=digest,protect_residual=True,raw_depth_dtype="float32"))
    network=load_geometry()
    datasets=dict(geometry=set(inventory["geometry"]), optical=set(inventory["optical"]),
                  development=set(inventory["development"]))
    for name in inventory["development"]:
        digits=name.split("-")[1]
        if digits in ("003","004","005","008"):
            datasets[f"dev{digits}"]={name}
    report={scope:new_regions() for scope in datasets}
    counts=Counter()
    for i,row in enumerate(frames):
        if i%8==0:
            waited+=yield_gpu()
        name,frame=row["scene"],row["frame"]
        fn=cache_path("ovg",name,frame)
        if fn.is_file():
            arrays=cache_arrays(fn)
        else:
            if i%16==0 and shutil.disk_usage(RUN).free<(60+48*(len(frames)-i)/1024)*GIB:
                raise OSError("Insufficient disk reserve for OVG and encoder cache")
            arrays=packed(online_ovg(network,name,frame))
            write_arrays(fn,arrays)
        y=core_labels(paths[name],frame,inventory["metadata"][name])
        beta,source,valid=beta_arrays(arrays,margin)
        for zone,key in enumerate(REGIONS):
            own=y["visible"]&(y["zone"]==zone)
            reference=y["rod_reference"] if zone==0 else y["center"]
            used=own&(source>0)
            known=used&(reference>0)
            errors=(beta-reference)[known]
            tally=dict(pixels=int(own.sum()),eligible_multiview=int((own&valid).sum()),
                       DepthSplat=int((used&(source==1)).sum()),triangulation=int((used&(source==2)).sum()),
                       triggered=int(used.sum()),reference_covered=int(known.sum()))
            if zone==0:
                tally[y["rod_reference_source"]+"_pixels"]=int(known.sum())
            for scope,subset in datasets.items():
                if name in subset:
                    dst=report[scope][key]
                    dst["counts"].update(tally)
                    dst["errors"].append(errors)
        counts["frames"]+=1
        counts["bytes"]+=fn.stat().st_size
        if (i+1)%30==0 or i+1==len(frames):
            print("E_OVG",i+1,len(frames),name,flush=True)
    for sess in report.values():
        for r in sess.values():
            c=r.pop("counts")
            r["counts"]=dict(c)
            r["triggered_fraction"]=c["triggered"]/max(1,c["pixels"])
            r["error_inv_m"]=stats(r.pop("errors"))
    save_json(RUN/"ovg-audit.json",dict(geometry_sha256=digest,margin=margin,stats=report,
        counts=dict(counts),forward_usable={k:len(v) for k,v in datasets.items()},
        elapsed_seconds=time.monotonic()-begin,gpu_wait_seconds=waited,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,competitor_start=before,competitor_end=competitor_state()))


def load_encoder():
    if not DA_WEIGHTS.is_file() or not DA_VENDOR.is_dir():
        raise FileNotFoundError("Depth Anything V2 Small official source and weights are required")
    sys.path.insert(0,str(DA_VENDOR))
    from depth_anything_v2.dpt import DepthAnythingV2
    model=DepthAnythingV2(encoder="vits",features=64,out_channels=[48,96,192,384])
    model.load_state_dict(torch.load(DA_WEIGHTS,map_location="cpu",weights_only=True),strict=True)
    if tuple(model.intermediate_layer_idx["vits"])!=LAYERS:
        raise ValueError("Unexpected Small encoder layer registration")
    return model.eval().requires_grad_(False).cuda()


@torch.inference_mode()
def online_encoder(model,unit,frame):
    rgb=to_srgb8(observed_rgb(unit,frame))
    image,_=model.image2tensor(np.ascontiguousarray(rgb[...,::-1]),518)
    if image.shape!=(1,3,518,518):
        raise ValueError(f"Wrong encoder input shape for {unit.name}/{frame}")
    maps=model.pretrained.get_intermediate_layers(image,LAYERS,reshape=True)
    result={f"block_{layer}":t[0].cpu().numpy() for layer,t in zip(LAYERS,maps)}
    if len(result)!=4 or any(a.shape!=(384,37,37) or not np.isfinite(a).all() for a in result.values()):
        raise ValueError(f"Invalid native encoder tokens for {unit.name}/{frame}")
    return result


def compare(raw, cached):
    if raw.keys()!=cached.keys():
        raise ValueError("Cache has different fields from the online pass")
    result={}
    for k,online in raw.items():
        stored=cached[k]
        rounded=online.astype(stored.dtype)
        if online.shape!=stored.shape or not np.array_equal(rounded,stored):
            raise ValueError(f"Failed elementwise cache check: {k}")
        result[k]=dict(elements=stored.size,stored_dtype=str(stored.dtype),exact_after_storage_cast=True,
                       max_abs_after_cast=0.,max_abs_online=float(np.max(np.abs(stored.astype(np.float32)-online))))
    return result


@torch.inference_mode()
def build_encoder():
    if not (RUN/"ovg-audit.json").is_file():
        raise FileNotFoundError("Finish the label and multiview anchor audit first")
    inventory=load(RUN/"inventory.json")
    paths=unit_paths(inventory)
    frames=load(RUN/"labels.json")["frames"]
    began=time.monotonic()
    before=competitor_state()
    setup()
    waited=yield_gpu()
    sha=sha256(DA_WEIGHTS)
    if sha!=DA_SHA256:
        raise ValueError("Weights checksum differs from the official Apache-2.0 ViT-S release")
    receipt=RUN/"encoder-source.json"
    if receipt.is_file() and load(receipt)["weights_sha256"]!=sha:
        raise ValueError("Encoder weights changed during caching")
    source=dict(source_revision=REVISION,weights_sha256=sha,
        source="https://github.com/DepthAnything/Depth-Anything-V2",
        weights="https://huggingface.co/depth-anything/Depth-Anything-V2-Small",
        license="Apache-2.0",layers_zero_based=list(LAYERS),input_size=518,patch_size=14)
    save_json(receipt,source)
    model=load_encoder()
    sizes=[]
    for i,row in enumerate(frames):
        if i%8==0:
            waited+=yield_gpu()
        name,frame=row["scene"],row["frame"]
        path=cache_path("da2-small",name,frame)
        if not path.is_file():
            if i%16==0 and shutil.disk_usage(RUN).free<(60+5*(len(frames)-i)/1024)*GIB:
                raise OSError("Insufficient disk reserve for ViT-S cache")
            values=online_encoder(model,paths[name],frame)
            write_arrays(path,{k:v.astype(np.float16) for k,v in values.items()})
        sizes.append(path.stat().st_size)
        if (i+1)%45==0 or i+1==len(frames):
            print("E_DA2",i+1,len(frames),flush=True)
    probes=[(inventory["overfit"][0],1),(inventory["holdout"][0],30),
            (next(n for n in inventory["development"] if n.split("-")[1]=="005"),60)]
    checks=[]
    for name,frame in probes:
        waited+=yield_gpu()
        values=online_encoder(model,paths[name],frame)
        checks.append(dict(scene=name,frame=frame,encoder=compare(values,cache_arrays(cache_path("da2-small",name,frame)))))
    del model
    torch.cuda.empty_cache()
    network=load_geometry()
    for row in checks:
        waited+=yield_gpu()
        name,frame=row["scene"],row["frame"]
        row["ovg"]=compare(online_ovg(network,name,frame),cache_arrays(cache_path("ovg",name,frame)))
        print("E_CACHE_EQUIVALENCE",name,frame,flush=True)
    save_json(RUN/"cache-report.json",dict(encoder=source,encoder_bytes=sum(sizes),encoder_per_frame_bytes=min(sizes),
        encoder_per_frame_bytes_max=max(sizes),ovg=load(RUN/"ovg-audit.json")["counts"],checks=checks,
        elapsed_seconds=time.monotonic()-began,gpu_wait_seconds=waited,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,competitor_start=before,competitor_end=competitor_state(),
        coordinates="Native pixel x,y have centers x+.5,y+.5. Full 512 px token maps or 518 px model-input maps are sampled at normalized grid 2*(coordinate+.5)/512-1, align_corners=False, border padding, after disk float16 cast. 518/14=37 patches along each dimension. An integer native crop of the full-resolution tensor equals its exact slice. No color jitter or flips."))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("ovg","encoder"))
    args=parser.parse_args()
    (audit_ovg if args.phase=="ovg" else build_encoder)()
