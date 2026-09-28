"""Check BokehDiff at the frozen input resolution without changing the benchmark or source code."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baselines.bokehdiff import OfficialBokehDiff


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('configs/bokehdiff-track-a-preflight.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only', default='repro_000')
    p.add_argument('--frames', default='0')
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    protocol = json.loads((ROOT / args.protocol).read_text())
    source = ROOT.parent / 'BokehDiff'
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == protocol['author_commit']
    assert not subprocess.check_output(['git', '-C', str(source), 'diff', '--', 'inference_hf.py', 'dataset.py', 'PISA_attn_processor.py', 'custom_diffusers'], text=True)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        assert json.loads((out / 'protocol.json').read_text()) == protocol
    else:
        write(out / 'protocol.json', protocol)
    clips = ROOT / protocol['input']
    cfg = json.loads((clips / 'config.json').read_text())
    chosen = set(args.only.split(','))
    assert chosen <= set(protocol['preflight_scenes'])
    frames = [int(x) for x in args.frames.split(',')]
    assert frames and len(frames) == len(set(frames)) and all(0 <= x < 25 for x in frames)
    torch.set_num_threads(4)
    model = OfficialBokehDiff(source, ROOT / 'models/BokehDiff')
    print('LOADED', model.weight_loading, flush=True)
    rows = []
    for scene in cfg['scenes']:
        sid = scene['id']
        if sid not in chosen:
            continue
        src = clips / sid
        meta = json.loads((src / 'metadata.json').read_text())
        dest = out / 'clips' / sid / 'pred_bokehdiff'
        dest.mkdir(parents=True, exist_ok=True)
        timing = out / 'timing' / sid
        timing.mkdir(parents=True, exist_ok=True)
        depths = sorted((src / 'disparity').glob('*.png'))
        assert len(depths) == 25
        for t in frames:
            record = timing / f'{t:05d}.json'
            if record.exists():
                assert (dest / f'{t:05d}.png').exists()
                rows.append(json.loads(record.read_text()))
                continue
            path = depths[t]
            assert path.name.startswith(f'{t:05d}_zf_')
            focus = float(path.stem.split('_zf_')[1])
            assert abs(focus - meta['frames'][t]['focus_disparity']) <= 5.1e-9
            rgb = np.asarray(Image.open(src / 'rgb' / f'{t:05d}.png')).astype(np.float32) / 255
            d = np.asarray(Image.open(path)).astype(np.float32) / 255
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.monotonic()
            image, audit = model.predict(rgb, d, focus, scene['K'], protocol['seed_per_frame'])
            torch.cuda.synchronize()
            audit.update(scene=sid, frame=t, seconds=time.monotonic() - start,
                         peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                         peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30)
            with (dest / f'{t:05d}.png').open('xb') as f:
                image.save(f, format='PNG')
            write(record, audit)
            rows.append(audit)
            print('FRAME', sid, t, audit, flush=True)
    write(out / 'result.json', dict(status='CAPACITY_AND_INPUT_ADAPTER_PROBE_COMPLETE',
          scope='No paper-score selection; no GT or oracle inputs; not a frozen 200-video adapter',
          weight_loading=model.weight_loading, protocol=protocol, rows=rows))
    print('COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
