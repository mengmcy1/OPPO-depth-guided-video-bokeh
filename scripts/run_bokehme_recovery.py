"""Run both declared BokehMe defocus conventions through the full official pipeline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from baselines.bokehme import setup, defocus_input, predict


def save_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('configs/bokehme-adapter-recovery.json'))
    p.add_argument('--output', type=Path, default=Path('outputs/bokehme-adapter-recovery-20260928'))
    p.add_argument('--only', default='')
    p.add_argument('--max-frames', type=int, default=25)
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    protocol = json.loads((ROOT/args.protocol).read_text())
    repo = ROOT.parent/'BokehMe'
    head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert head == protocol['author_commit']
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--','demo.py','neural_renderer.py','classical_renderer'],text=True)
    clips = ROOT/protocol['input']
    cfg = json.loads((clips/'config.json').read_text())
    assert 1 <= args.max_frames <= 25
    selected = set(args.only.split(',')) if args.only else {x['id'] for x in cfg['scenes']}
    assert selected <= {x['id'] for x in cfg['scenes']}
    args.output.mkdir(parents=True,exist_ok=True)
    saved = args.output/'protocol.json'
    if saved.exists():
        assert json.loads(saved.read_text()) == protocol
    else:
        save_json(saved,protocol)
    torch.set_num_threads(4); cv2.setNumThreads(4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    pipeline, demo_args, nets, counts = setup(repo,ROOT/'models/BokehMe',protocol)
    calls = {key:0 for key in nets}
    def hook(key):
        def record(*unused):
            calls[key] += 1
        return record
    for key,net in nets.items():
        net.register_forward_hook(hook(key))
    print('CHECKPOINTS_STRICTLY_LOADED',counts,flush=True)
    for scene in cfg['scenes']:
        sid = scene['id']
        if sid not in selected:
            continue
        src = clips/sid
        meta = json.loads((src/'metadata.json').read_text())
        timing = args.output/'timing'/sid
        timing.mkdir(parents=True,exist_ok=True)
        dirs = {c:args.output/c/'clips'/sid/'pred_bokehme' for c in protocol['candidates']}
        for d in dirs.values():
            d.mkdir(parents=True,exist_ok=True)
        if (timing/'complete.json').exists():
            assert json.loads((timing/'complete.json').read_text())['frames'] == 25
            assert all(len(list(d.glob('*.png'))) == 25 for d in dirs.values())
            print('REUSE',sid,flush=True)
            continue
        depths = sorted((src/'disparity').glob('*.png'))
        assert len(depths) == 25
        rows = []
        for t,dp in enumerate(depths[:args.max_frames]):
            assert dp.name.startswith(f'{t:05d}_zf_')
            record = timing/f'{t:05d}.json'
            if record.exists():
                assert all((d/f'{t:05d}.png').exists() for d in dirs.values())
                rows.append(json.loads(record.read_text()))
                continue
            rgb = cv2.cvtColor(cv2.imread(str(src/'rgb'/f'{t:05d}.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
            disp = cv2.imread(str(dp),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            focus = float(dp.stem.split('_zf_')[1])
            assert abs(focus-meta['frames'][t]['focus_disparity']) <= 5.1e-9
            assert rgb.shape == (576,1024,3) and disp.shape == (576,1024)
            row = dict(frame=t,K=scene['K'],candidates={})
            for candidate,directory in dirs.items():
                item_path = timing/f'{t:05d}_{candidate}.json'
                output_path = directory/f'{t:05d}.png'
                if item_path.exists():
                    assert output_path.exists()
                    row['candidates'][candidate] = json.loads(item_path.read_text())
                    continue
                defocus,stats = defocus_input(disp,focus,scene['K'],candidate,demo_args.defocus_scale)
                for key in calls:
                    calls[key] = 0
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
                value,ranges = predict(pipeline,demo_args,nets,rgb,defocus)
                torch.cuda.synchronize()
                stats.update(seconds=time.perf_counter()-start,calls=dict(calls),returned_ranges=ranges,
                             peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
                assert calls['classical'] == calls['arnet'] == 1 and calls['iunet'] >= 1
                quantized = np.uint8(np.round(value*255))
                if output_path.exists():
                    # A crash after PNG write cannot justify silently overwriting a nondeterministic scatter result.
                    previous = np.asarray(Image.open(output_path)).astype(np.int16)
                    assert np.max(np.abs(previous-quantized.astype(np.int16))) <= 1
                    stats['resumed_existing_PNG_max_abs'] = int(np.max(np.abs(previous-quantized.astype(np.int16))))
                else:
                    with output_path.open('xb') as f:
                        Image.fromarray(quantized).save(f,format='PNG')
                save_json(item_path,stats)
                row['candidates'][candidate] = stats
            save_json(record,row)
            rows.append(row)
            print('FRAME',sid,t,'seconds',round(sum(x['seconds'] for x in row['candidates'].values()),3),flush=True)
        if len(rows) == 25:
            summary = dict(scene=sid,frames=25,parameters=counts,candidates={})
            for c in dirs:
                entries = [r['candidates'][c] for r in rows]
                summary['candidates'][c] = dict(seconds=sum(x['seconds'] for x in entries),
                    peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in entries))
            save_json(timing/'complete.json',summary)
            print('DONE',sid,flush=True)


if __name__ == '__main__':
    main()
