"""Cache frozen Depth Anything V2 tokens for masked 112-pixel E-H2 windows."""
import argparse
import os
import shutil
import time

import numpy as np
import torch

from route3_v2data_B_batch import GIB,sha256
from route3_v2data_B_inputs import observed_rgb,to_srgb8
from route3_v2data_E_cache import DA_SHA256,DA_WEIGHTS,LAYERS,load_encoder
from route3_v2data_E_labels import load,save_json,unit_paths
from route3_v2data_E0_data import RUN,auxiliary_path,dataset,local_path,mapped,windows_for_mask
from route3_v2data_E0_train import gpu_gate
from route3_v2data_module2_infer import competitor_state
from route3_v2data_stage3a import gpu_setup
from src.ovg_e import ENCODER_CHANNELS,LOCAL_SIDE

WINDOW_BATCH = 8
LIMIT = 4.
TOKEN_SIDE = 32
WINDOW_BYTES = ENCODER_CHANNELS*TOKEN_SIDE*TOKEN_SIDE*2


@torch.inference_mode()
def encode(model,rgb,origins):
    if not origins:
        return np.empty((0,ENCODER_CHANNELS,TOKEN_SIDE,TOKEN_SIDE),np.float16)
    chunks=[]
    for j in range(0,len(origins),WINDOW_BATCH):
        tiles=[]
        for x,y in origins[j:j+WINDOW_BATCH]:
            crop=np.ascontiguousarray(rgb[y:y+LOCAL_SIDE,x:x+LOCAL_SIDE,::-1])
            tensor,_=model.image2tensor(crop,448)
            if tensor.shape!=(1,3,448,448):
                raise ValueError("The masked ViT window is not 448 by 448")
            tiles.append(tensor)
        values=model.pretrained.get_intermediate_layers(torch.cat(tiles),LAYERS,reshape=True)
        result=torch.cat(values,1)
        if result.shape[1:]!=(ENCODER_CHANNELS,TOKEN_SIDE,TOKEN_SIDE) or not torch.isfinite(result).all():
            raise ValueError("The window cache contains invalid ViT-S tokens")
        chunks.append(result.half().cpu().numpy())
    return np.concatenate(chunks)


def build(rows):
    plan=dataset()
    paths=unit_paths(plan)
    gpu_setup(dict(training=dict(max_process_gpu_gib=LIMIT)))
    waited=gpu_gate(LIMIT)
    start=time.monotonic()
    previous=competitor_state()
    if sha256(DA_WEIGHTS)!=DA_SHA256:
        raise ValueError("The prepared ViT-S weights have changed")
    model=load_encoder()
    count=size=0
    for j,row in enumerate(rows,1):
        name,frame=row["scene"],row["frame"]
        path=local_path(name,frame)
        data=path.with_suffix(".npy")
        if not path.is_file():
            waited+=gpu_gate(LIMIT)
            mask=mapped(auxiliary_path(name,frame),"mask")[0]
            origins=windows_for_mask(mask)
            if shutil.disk_usage(RUN).free<60*GIB+len(origins)*WINDOW_BYTES:
                raise OSError(f"Insufficient disk reserve for H2 window frame {name}/{frame}")
            rgb=to_srgb8(observed_rgb(paths[name],frame))
            values=encode(model,rgb,origins)
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=data.with_suffix(".part")
            with temp.open("wb") as stream:
                np.save(stream,values)
            os.replace(temp,data)
            save_json(path,dict(origins=origins,hard_mask_pixels=int(mask.sum())))
        count+=len(load(path)["origins"])
        size+=data.stat().st_size
        if j%25==0 or j==len(rows):
            print("E0_H2_WINDOWS",j,len(rows),count,size,flush=True)
    checks=[]
    for i in sorted({0,len(rows)//2,len(rows)-1}):
        row=rows[i]
        name,frame=row["scene"],row["frame"]
        path=local_path(name,frame)
        origins=load(path)["origins"]
        rgb=to_srgb8(observed_rgb(paths[name],frame))
        waited+=gpu_gate(LIMIT)
        online=encode(model,rgb,origins)
        stored=np.load(path.with_suffix(".npy"))
        if not np.array_equal(online,stored):
            raise ValueError(f"Online and cached H2 features differ for {name}/f{frame:03d}")
        checks.append(dict(scene=name,frame=frame,windows=len(origins),exact_after_fp16=True))
    return dict(frames=len(rows),windows=count,bytes=size,checks=checks,
                seconds=time.monotonic()-start,gpu_wait_seconds=waited,competitor_before=previous,
                competitor_after=competitor_state(),peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualification",action="store_true")
    arg=p.parse_args()
    data=load(RUN/"data.json")
    rows=data["rows"]
    if arg.qualification:
        draws=np.load(RUN/"crop-order.npy",mmap_mode="r")
        rows=[rows[i] for i in sorted(set(int(i) for i in draws[0,:,0]))]
    receipt=RUN/("window-qualification.json" if arg.qualification else "window-cache.json")
    if not receipt.is_file():
        save_json(receipt,build(rows))
