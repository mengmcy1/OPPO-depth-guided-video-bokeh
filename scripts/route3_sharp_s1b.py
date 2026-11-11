"""S1b output-only anchoring; all products use a fresh experiment directory."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1 import OUT as S1,scene as original_scene,FRAMES
from route3_upper_bound_evaluate import BASE,OUT as UPPER,RECORDS
from src.sharp_anchor import anchor
OUT=ROOT/'outputs/route3-sharp-s1b-2026-11-11-01'
PREFIX='route3-sharp-s1b-2026-11-11'
CRITERIA=RECORDS/'route3-sharp-s1b-criteria-2026-11-11.json'


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def anchor_native(c):
    root=OUT/'anchor/native';root.mkdir(parents=True,exist_ok=True)
    prior=json.loads((S1/'prepared.json').read_text())['rows']
    for frame in [1,30,60]:
        entry=next(r for r in prior if r['scale_label']=='native' and r['frame']==frame)
        d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');fill=torch.tensor(np.load(S1/'fill'/entry['id']/'fill.npz')['fill_rgb_linear'],device='cuda')
        target=torch.tensor(d['rgb_srgb'],device='cuda');base=original_scene(frame,'full')
        for name,op in [('A_color',False),('A_color_opacity',True)]:
            if (OUT/'stop.json').exists():
                stopped=json.loads((OUT/'stop.json').read_text())
                if stopped['stage']!='calibration' or not (OUT/'native-continuation-authorized.json').exists():
                    raise RuntimeError('S1B_STOP another authorized stage triggered stop; do not launch new fits')
            torch.manual_seed(2026111121+frame);folder=root/f'f{frame:03d}'/name
            if (folder/'result.json').exists():
                prior_result=json.loads((folder/'result.json').read_text())
                assert prior_result['completed'] and prior_result['completed_updates']==1000 and Path(prior_result['attributes']).exists()
                print('ANCHOR_REUSE_COMPLETED',frame,name,flush=True);continue
            folder.mkdir(parents=True,exist_ok=False)
            def monitor(value):write(folder/'monitor.json',value)
            fitted,result=anchor(base,K,fill,target,op,monitor)
            torch.save(dict(colors=fitted.colors.detach().cpu(),opacities=fitted.opacities.detach().cpu(),completed_updates=result['completed_updates']),folder/'attributes-final.pt')
            result.update(frame=frame,variant=name,geometry='native',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                attributes=str(folder/'attributes-final.pt'),source_Gaussians=str(S1/'gaussians'/f'f{frame:03d}.pt'),input_RGB=entry['input_npz'],fill=str(S1/'fill'/entry['id']/'fill.npz'))
            write(folder/'result.json',result);write(RECORDS/(PREFIX+f'-anchor-native-f{frame:03d}-{name}.json'),result)
            if not result['completed']:
                write(OUT/'stop.json',dict(stage='anchor_native',frame=frame,variant=name,reason=result['stop_reason'],completed_updates=result['completed_updates']))
                raise RuntimeError('S1B_STOP '+str(result['stop_reason']))
            print('ANCHOR_DONE',frame,name,result['rows'][0]['input_PSNR_dB'],result['rows'][-1]['input_PSNR_dB'],flush=True)
        del base;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['anchor-native'],required=True);a=p.parse_args()
    c=json.loads(CRITERIA.read_text());assert c['frozen'];torch.set_num_threads(8)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit authorized GPU0')
    subprocess.run(['nvidia-smi'],check=True)
    if a.mode=='anchor-native':anchor_native(c)


if __name__=='__main__':main()
