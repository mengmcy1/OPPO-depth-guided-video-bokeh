"""Check a CUDA launch-only correction against already saved official preflight outputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baselines.drbokeh_cuda import load_grid2d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',type=Path,default=Path('outputs/drbokeh-preflight-20260928'))
    p.add_argument('--output',type=Path,default=Path('outputs/drbokeh-launch-probe-20260928'))
    p.add_argument('--only',default='repro_000,repro_001,repro_003,repro_004,repro_007,repro_008')
    p.add_argument('--frames',default='0')
    p.add_argument('--all-available',action='store_true')
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    module=load_grid2d()
    sys.path.insert(0,str(ROOT.parent/'DrBokeh/app/Render/DScatter'))
    import GPU_scatter
    from scatter import Multi_Layer_Renderer
    original=GPU_scatter.scatter_cuda
    protocol=json.loads((args.preflight/'protocol.json').read_text())
    renderer=Multi_Layer_Renderer(protocol['lens']).cuda().eval()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=True)
    report=args.output/'result.json';assert not report.exists()
    rows=[]
    paths=sorted((args.preflight/'clips').glob('*/diagnostics/*.npz')) if args.all_available else [args.preflight/'clips'/sid/'diagnostics'/f'{int(t):05d}.npz' for sid in args.only.split(',') for t in args.frames.split(',')]
    with torch.no_grad():
        for path in paths:
            sid=path.parent.parent.name;t=int(path.stem)
            record=args.preflight/'timing'/sid/f'{t:05d}.json'
            if not record.exists():continue
            audit=json.loads(record.read_text());g=audit['geometry']
            with np.load(path) as saved:
                layers=[saved['fg_rgbad_render'],saved['bg_rgbad_render']]
                expected=saved['final_float_RGB']
                GPU_scatter.scatter_cuda=module
                torch.cuda.synchronize();start=time.perf_counter()
                value=renderer.inference(layers,g['K_Dr'],g['focus_prime'])**(1/protocol['gamma'])-protocol['offset']
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                delta=np.abs(value-expected)
                png_diff=np.abs(np.round(np.clip(value,0,1)*255).astype(np.int16)-np.round(np.clip(expected,0,1)*255).astype(np.int16))
                row=dict(scene=sid,frame=t,seconds_grid2d=seconds,max_abs=float(delta.max()),MAE=float(delta.mean()),
                         PNG8_max_abs=int(png_diff.max()),bitwise_equal=bool(np.array_equal(value,expected)))
                rows.append(row)
                print('CHECK',row,flush=True)
    GPU_scatter.scatter_cuda=original
    result=dict(status='LAUNCH_ONLY_EQUIVALENCE_PROBE',source_commit=protocol['author_commit'],
                original_grid='6 blocks of 512x1 threads at 1024x576, lens71',
                new_grid='2304 blocks of 16x16 threads, one thread per output pixel',
                scope='Only occlusion-aware forward launch changed; per-pixel loops, arithmetic, gamma, layers, and controls unchanged. Concurrent timing is diagnostic only.',
                all_float_bitwise_equal=bool(rows and all(x['bitwise_equal'] for x in rows)),rows=rows)
    with report.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('COMPLETE',len(rows),'all bitwise equal',result['all_float_bitwise_equal'],flush=True)


if __name__=='__main__':main()
