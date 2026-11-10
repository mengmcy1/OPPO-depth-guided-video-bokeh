"""LaMa residual-color lookup from explicitly declared image/mask inputs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.c1_regions import linear_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    entries=json.loads(a.manifest.read_text())['items'];a.output=a.output.resolve();a.record=a.record.resolve()
    if a.record.exists():raise FileExistsError(a.record)
    a.output.mkdir(parents=True,exist_ok=False)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0 required')
    subprocess.run(['nvidia-smi'],check=True)
    import torch
    import cv2
    torch.set_num_threads(4);cv2.setNumThreads(1);os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    repo=ROOT.parent/'DrBokeh';os.chdir(repo);sys.path.insert(0,str(repo))
    spec=importlib.util.spec_from_file_location('dual_layer_LaMa',repo/'app/Render/rgb_inpainting.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.RGB_Inpainting_Inference();rows=[]
    for x in entries:
        d=np.load(ROOT/x['input_npz'] if not Path(x['input_npz']).is_absolute() else x['input_npz']);rgb=d['rgb_srgb'];mask=d['fill_mask']
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic();prediction=model.inference(rgb,mask[...,None].astype(np.float32));torch.cuda.synchronize();seconds=time.monotonic()-start
        if prediction is None or not np.isfinite(prediction).all():raise RuntimeError('LaMa failed')
        output=np.where(mask[...,None],prediction.clip(0,1),rgb).astype(np.float32);dest=a.output/x['id'];dest.mkdir()
        np.savez_compressed(dest/'fill.npz',fill_rgb_srgb=output,fill_rgb_linear=linear_rgb(output),fill_mask=mask)
        cv2.imwrite(str(dest/'fill.png'),np.uint8(output[...,::-1]*255))
        rows.append(dict(id=x['id'],input_npz=x['input_npz'],seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,output=str(dest/'fill.npz'),
            uses_GT_segmentation=x.get('uses_GT_segmentation',False),uses_true_hidden_depth=x.get('uses_true_hidden_depth',False),outside_mask_max_error=float(abs(output[~mask]-rgb[~mask]).max()) if (~mask).any() else 0.))
        print('LAMA_FILL',x['id'],seconds,flush=True)
    a.record.write_text(json.dumps(dict(completed=True,rows=rows,lookup_only=True,composition='only remaining transmission after visible+hidden; no opaque precover',training=False),indent=2)+'\n')


if __name__=='__main__':main()
