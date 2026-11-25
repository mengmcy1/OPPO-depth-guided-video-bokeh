"""Read frozen full-frame features and materialize a shared native-crop plan."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import os
import shutil
import struct
import zipfile

import numpy as np
from scipy import ndimage
import torch

from route3_v2data_B_batch import GIB, sha256
from route3_v2data_E_cache import LAYERS, cache_path, write_arrays
from route3_v2data_E_labels import FRAMES, ROOT, RUN as PREP, SEED, core_labels, load, save_json, unit_paths
from route3_v2data_stage2_train import INITIAL
from src.ovg_e import LOCAL_SIDE, fixed_channels, rear_base

RUN = ROOT/"outputs/route3-v2data-E0-2026-11-26-01"
STEPS = 20_000
BATCH = 8
SIDE = 256
HOLDOUT = (61,64,72,89,111,140,145,146,170,173)
ANCHORS = ("rod","gap","mv_valid","mv_invalid")
INPUTS = ("features","phi","mask","rho1","beta","attenuation")
CONFIG = ROOT/"03_实验记录/records/route3-v2data-E0-config-2026-11-26.json"
ENCODER_KEYS = tuple(f"block_{layer}" for layer in LAYERS)


def auxiliary_path(name,frame):
    return RUN/"native"/name/f"f{frame:03d}.npz"


def local_path(name,frame):
    return RUN/"windows"/name/f"f{frame:03d}.json"


@lru_cache(maxsize=192)
def mapped(path,key):
    """Map one array in the existing uncompressed, np.savez-produced cache."""
    with zipfile.ZipFile(path) as archive:
        item = archive.getinfo(key+".npy")
    if item.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"Cannot map a compressed cache: {path}")
    with open(path,"rb") as stream:
        stream.seek(item.header_offset)
        header = stream.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError(f"Invalid ZIP member in {path}")
        name_len,extra_len = struct.unpack_from("<HH",header,26)
        stream.seek(name_len+extra_len,os.SEEK_CUR)
        version = np.lib.format.read_magic(stream)
        reader = {(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}[version]
        shape,fortran,dtype = reader(stream)
        offset = stream.tell()
    return np.memmap(path,mode="r",offset=offset,shape=shape,dtype=dtype,order="F" if fortran else "C")


def dataset():
    index = load(PREP/"inventory.json")
    holdout = tuple(int(n.split("-")[1]) for n in index["holdout"])
    if holdout != HOLDOUT or len(index["geometry"])!=295 or len(index["development"])!=15:
        raise ValueError("B inventory disagrees with the frozen E0 splits")
    source = load(PREP/"ovg-source.json")
    geometry = sha256(INITIAL)
    if not geometry.startswith("dfb1d7e5") or geometry not in source.values():
        raise ValueError("Frozen OVG geometry does not match the prepared feature cache")
    checks = load(PREP/"cache-report.json")["checks"]
    if len(checks)!=3 or not all(checks):
        raise ValueError("Expected the three paired online checks from E preparation")
    return index


def build_aux(job):
    name,frame,unit,meta = job
    path = auxiliary_path(name,frame)
    if path.is_file():
        return load(path.with_suffix(".json"))
    raw = cache_path("ovg",name,frame)
    with np.load(raw) as arrays:
        values={k:torch.from_numpy(arrays[k])[None] for k in arrays.files if k!="phi_full"}
    rho1,rho2 = (values[k][:,None] for k in ("rho1","rho2"))
    beta,source,mv = rear_base(rho1,rho2,values["prior"],values["confidence"])
    features,mask = fixed_channels(values["rgb"],rho1,rho2,values["prior"],values["confidence"])
    y = core_labels(unit,frame,meta)
    loss_zone = np.where((y["zone"]==2)&y["depth_boundary"],3,y["zone"]).astype(np.uint8)
    arrays = {k:t[0].numpy() for k,t in dict(features=features,mask=mask.bool(),beta=beta,
                        mv_source=source.byte(),mv_valid=mv,rho1=rho1,
                        attenuation=values["attenuation"][:,None]).items()}
    arrays.update({k:y[k][None] for k in ("center","front","back","visible","valid_dual","zone","rod_reference")})
    arrays["loss_zone"]=loss_zone[None]
    coefficient = float(y["K"][0,0]) * (.05/1.4)
    rng = np.random.default_rng(SEED+frame+100*int(name.split("-")[1]))
    avail = [y["visible"]&(y["zone"]==z) for z in (0,1)]
    avail.extend(y["visible"]&s for s in (arrays["mv_valid"][0],~arrays["mv_valid"][0]))
    coords={}
    for key,accepted in zip(ANCHORS,avail):
        pixels=np.flatnonzero(accepted)
        n=min(len(pixels),256)
        coords[key]=rng.choice(pixels,n,replace=False).tolist()
    summary=dict(scene=name,frame=frame,CoC_K=coefficient,anchors=coords)
    path.parent.mkdir(parents=True,exist_ok=True)
    save_json(path.with_suffix(".json"),summary)
    write_arrays(path,arrays)
    return summary


def native_jobs(index):
    paths=unit_paths(index)
    for name in (*index["geometry"], *index["development"]):
        for frame in FRAMES:
            unit=paths[name]
            if not cache_path("da2-small",name,frame).is_file() or not cache_path("ovg",name,frame).is_file():
                raise FileNotFoundError(f"Missing frozen cache for {name}/f{frame:03d}")
            yield name,frame,unit,index["metadata"][name]


def worker_setup():
    torch.set_num_threads(1)


def prepare():
    index=dataset()
    RUN.mkdir(parents=True,exist_ok=True)
    receipt=RUN/"data.json"
    if receipt.is_file():
        return load(receipt)
    inputs=list(native_jobs(index))
    if shutil.disk_usage(RUN).free<(60+len(inputs)*.025)*GIB:
        raise OSError("Insufficient disk reserve for E0 cache")
    rows=[]
    with ProcessPoolExecutor(max_workers=min(4,max(1,len(os.sched_getaffinity(0))//2)),
                             initializer=worker_setup) as executor:
        for i,row in enumerate(executor.map(build_aux,inputs,chunksize=2),1):
            rows.append(row)
            if i%50==0 or i==len(inputs):
                print("E0_DATA",i,len(inputs),flush=True)
    held=set(index["holdout"])
    train=[i for i,r in enumerate(rows) if r["scene"] not in held and r["scene"].startswith("training-")]
    draws=np.empty((STEPS,BATCH,3),np.int32)
    rng=np.random.default_rng(SEED)
    for j,key in enumerate(ANCHORS):
        pool=[i for i in train if rows[i]["anchors"][key]]
        ids=rng.choice(pool,(STEPS,2))
        pixels=np.asarray([rng.choice(rows[i]["anchors"][key]) for i in ids.ravel()]).reshape(STEPS,2)
        y,x=np.divmod(pixels,512)
        ox=rng.integers(np.maximum(0,x-SIDE+1),np.minimum(x,512-SIDE)+1)
        oy=rng.integers(np.maximum(0,y-SIDE+1),np.minimum(y,512-SIDE)+1)
        draws[:,2*j:2*j+2]=np.stack((ids,ox,oy),-1)
    order=RUN/"crop-order.npy"
    np.save(order,draws)
    result=dict(rows=[{k:v for k,v in r.items() if k!="anchors"} for r in rows],
                crop_order_sha256=sha256(order),steps=STEPS,crops_per_step=BATCH,
                training_frames=len(train),holdout_frames=len(index["holdout"])*len(FRAMES),
                development_frames=len(index["development"])*len(FRAMES),
                preparation_cache_sha256=sha256(PREP/"cache-report.json"))
    save_json(receipt,result)
    return result


def windows_for_mask(mask):
    """Cover connected regions with overlapping windows on the original pixel canvas."""
    masks,n=ndimage.label(mask,structure=np.ones((3,3),np.uint8))
    def starts(lo,hi):
        lower = max(0,min(512-LOCAL_SIDE,lo))
        upper = max(0,min(512-LOCAL_SIDE,hi-LOCAL_SIDE))
        return np.linspace(lower,max(lower,upper),
                           max(1,int(np.ceil((hi-lo-LOCAL_SIDE)/84))+1)).round().astype(int)
    windows=set()
    for bounds in ndimage.find_objects(masks,n):
        if bounds is None:
            continue
        ys,xs=bounds
        if ys.stop-ys.start<=LOCAL_SIDE and xs.stop-xs.start<=LOCAL_SIDE:
            cx=(xs.start+xs.stop-LOCAL_SIDE)//2
            cy=(ys.start+ys.stop-LOCAL_SIDE)//2
            windows.add((max(0,min(512-LOCAL_SIDE,cx)),max(0,min(512-LOCAL_SIDE,cy))))
        else:
            windows.update((int(x),int(y)) for y in starts(ys.start,ys.stop) for x in starts(xs.start,xs.stop)
                           if mask[y:y+LOCAL_SIDE,x:x+LOCAL_SIDE].any())
    # Any entirely covered window can be dropped without changing masked support.
    remaining=mask.copy()
    selected=[]
    while remaining.any():
        if not windows:
            raise ValueError("A hard-mask pixel has no H2 window")
        x,y=max(windows,key=lambda p:(int(remaining[p[1]:p[1]+LOCAL_SIDE,p[0]:p[0]+LOCAL_SIDE].sum()),-p[1],-p[0]))
        selected.append((x,y))
        remaining[y:y+LOCAL_SIDE,x:x+LOCAL_SIDE]=False
        windows.remove((x,y))
    return selected


def prepared_crop(row,x=0,y=0,side=512,*,labels=True):
    name,frame=row["scene"],row["frame"]
    path=auxiliary_path(name,frame)
    fields=(*INPUTS,"mv_valid","mv_source","zone","loss_zone","center","front","back","visible",
            "valid_dual","rod_reference") if labels else INPUTS
    values={}
    for key in fields:
        a=mapped(cache_path("ovg",name,frame),"phi_full") if key=="phi" else mapped(path,key)
        values[key]=a[...,y:y+side,x:x+side]
    return values


def encoder_tokens(name,frame):
    with np.load(cache_path("da2-small",name,frame)) as tokens:
        return np.concatenate([tokens[k] for k in ENCODER_KEYS])


def window_tokens(name,frame,x,y,side):
    path=local_path(name,frame)
    origins=np.asarray(load(path)["origins"],np.int32).reshape(-1,2)
    lo=np.array([x,y])
    subset=((origins+LOCAL_SIDE>lo)&(origins<lo+side)).all(1)
    if not subset.any():
        return np.empty((0,4*384,32,32),np.float16),origins[subset]
    stored=np.load(path.with_suffix(".npy"),mmap_mode="r")
    return stored[subset],origins[subset]


if __name__=="__main__":
    prepare()
