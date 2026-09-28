"""Compare the adapter with the unmodified author demo before JPEG encoding."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baselines.bokehme import setup, defocus_input, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('outputs/bokehme-adapter-recovery-20260928'))
    args = parser.parse_args()
    run = (ROOT/args.run).resolve()
    report = run/'demo-equivalence.json'
    assert not report.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    protocol = json.loads((run/'protocol.json').read_text())
    source = ROOT/protocol['input']/'repro_004'
    cfg = json.loads((ROOT/protocol['input']/'config.json').read_text())
    K = next(s['K'] for s in cfg['scenes'] if s['id'] == source.name)
    dp = sorted((source/'disparity').glob('*.png'))[0]
    focus = float(dp.stem.split('_zf_')[1])
    rgb = cv2.cvtColor(cv2.imread(str(source/'rgb/00000.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    disp = cv2.imread(str(dp),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
    torch.set_num_threads(4); cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    repo = ROOT.parent/'BokehMe'
    pipeline, demo_args, nets, _ = setup(repo,ROOT/'models/BokehMe',protocol)
    results = {}
    for candidate in protocol['candidates']:
        control, stats = defocus_input(disp,focus,K,candidate,protocol['defocus_scale'])
        adapter,_ = predict(pipeline,demo_args,nets,rgb,control)
        span = stats['span']
        native_focus = (focus-stats['disparity_range'][0])/span
        native_K = K*span if candidate == 'shared_defocus' else K
        # The demo ALWAYS normalizes disparity and expects already-normalized focus.
        # K*span here only proves equivalence with A; it does not change A's dataset K.
        sys.argv = [str(repo/'demo.py'),'--image_path',str(source/'rgb/00000.png'),
                    '--disp_path',str(dp),'--K',str(native_K),'--disp_focus',str(native_focus),
                    '--gamma','2.2','--save_dir',str(run/'official-demo'/candidate),
                    '--arnet_checkpoint_path',str(ROOT/'models/BokehMe/arnet.pth'),
                    '--iunet_checkpoint_path',str(ROOT/'models/BokehMe/iunet.pth')]
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        namespace = runpy.run_path(str(repo/'demo.py'),run_name='__main__')
        native = namespace['bokeh_pred']
        difference = np.abs(adapter-native)
        png_difference = np.abs(np.round(adapter*255).astype(np.int16)-np.round(native*255).astype(np.int16))
        values = dict(float_max_abs=float(difference.max()),float_MAE=float(difference.mean()),
                      PNG8_max_abs=int(png_difference.max()),PNG8_different_values=int(np.count_nonzero(png_difference)),
                      native_K=native_K,native_normalized_focus=native_focus)
        for key,tolerance in protocol['demo_equivalence_tolerance'].items():
            assert values[key] <= tolerance, (candidate,key,values[key],tolerance)
        results[candidate] = values
        del namespace
    with report.open('x') as f:
        json.dump(dict(scene='repro_004',frame=0,passed=True,results=results,
                       tolerance=protocol['demo_equivalence_tolerance'],
                       note='Full unmodified demo via runpy, comparing returned float RGB before JPEG. Native K rescaled only to express A in normalized coordinates.'),f,indent=2)
        f.write('\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    main()
