"""One-frame full official demo resource probe, not a frozen benchmark adapter."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/drbokeh-concurrency-probe-20260928'))
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    run = (ROOT/args.output).resolve()
    run.mkdir(parents=True,exist_ok=True)
    assert not (run/'capacity.json').exists()
    clips = ROOT/'outputs/any2bokeh-mpib-public-proxy-200-20260928/clips'
    scene = next(x for x in json.loads((clips/'config.json').read_text())['scenes'] if x['id']=='repro_000')
    src = clips/scene['id']
    path = sorted((src/'disparity').glob('*.png'))[0]
    disparity = np.asarray(Image.open(path)).astype(np.float32)/255
    focus = float(path.stem.split('_zf_')[1])
    lo,span = float(disparity.min()),float(disparity.max())-float(disparity.min())
    assert span > 0 and scene['K']==32
    spec = dict(rgb=str(src/'rgb/00000.png'),disp=str(run/'disparity.npz'),scene=scene['id'],frame=0,
                K=scene['K'],focus_original=focus,focus_native_normalized=(focus-lo)/span,disparity_span=span,
                scope='Single-frame capacity probe using official demo normalization; not frozen Track A adapter or benchmark score')
    if (run/'probe-input.json').exists():
        assert json.loads((run/'probe-input.json').read_text())==spec
        assert np.array_equal(np.load(run/'disparity.npz')['data'],disparity)
    else:
        with (run/'disparity.npz').open('xb') as f:
            np.savez(f,data=disparity)
        with (run/'probe-input.json').open('x') as f:
            json.dump(spec,f,indent=2)
            f.write('\n')
    repo = ROOT.parent/'DrBokeh'
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    os.chdir(repo)
    sys.path.insert(0,str(repo))
    sys.argv = [str(repo/'app/Render/Inference.py'),'--rgb',spec['rgb'],'--disp',spec['disp'],
                '-K',str(spec['K']),'--focal',str(spec['focus_native_normalized']),
                '--gamma','2.2','--lens','71','--ofile',str(run/'prediction.png')]
    # Supplying disparity avoids a new monocular-depth prediction. Alpha is inferred
    # from RGB by LDF; no GT alpha, original foreground, or hidden background is used.
    namespace = runpy.run_path(sys.argv[0],run_name='__main__')
    torch.cuda.synchronize()
    value = namespace['ret']['bokeh']
    assert value.shape == (576,1024,3) and np.isfinite(value).all()
    with Image.open(run/'prediction.png') as im:
        assert im.size == (1024,576)
        im.verify()
    result = dict(status='FULL_OFFICIAL_SINGLE_FRAME_CAPACITY_PROBE_PASSED',input=spec,
                  author_commit=commit,gamma=2.2,lens=71,highlight=False,
                  elapsed_seconds_including_model_loading_and_PNG=time.monotonic()-start,
                  torch_peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                  torch_peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                  output_shape=list(value.shape),output_range=[float(value.min()),float(value.max())],
                  pipeline='LDF predicted alpha + official RGBAD2layers (LaMa RGB/depth inpainting) + two-layer Dr.Bokeh renderer; supplied disparity',
                  limits=['Concurrent BokehMe work; not standardized efficiency.',
                          'PyTorch memory excludes some CUDA context/extension overhead.',
                          'Native demo initializes MiDaS on import but does not run it with supplied disparity.',
                          'Single-frame probe does not establish worst-case memory or freeze geometry/color protocol.'])
    with (run/'capacity.json').open('x') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
