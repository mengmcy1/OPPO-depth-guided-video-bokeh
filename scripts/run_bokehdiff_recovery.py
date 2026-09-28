"""Run one BokehDiff geometry candidate with a continuous, resumable global RNG stream."""
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
from accelerate.utils import set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baselines.bokehdiff import OfficialBokehDiff, map_controls


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=['shared_geometry', 'normalized'], required=True)
    parser.add_argument('--stage', choices=['preflight', 'full'], default='preflight')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    protocol = json.loads((ROOT / 'configs/bokehdiff-adapter-recovery.json').read_text())
    source = ROOT.parent / 'BokehDiff'
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == protocol['author_commit']
    assert not subprocess.check_output(['git', '-C', str(source), 'diff', '--', 'inference_hf.py', 'dataset.py', 'PISA_attn_processor.py', 'custom_diffusers'], text=True)
    if args.stage == 'full':
        acceptance = json.loads((ROOT / '03_实验记录/records/bokehdiff-recovery-preflight-20260928.json').read_text())
        assert acceptance['status'] == 'PREFLIGHT_PASSED' and acceptance['protocol'] == protocol
    clips = ROOT / protocol['input']
    cfg = json.loads((clips / 'config.json').read_text())
    scenes = [s for s in cfg['scenes'] if args.stage == 'full' or s['id'] in protocol['preflight_scenes']]
    out = ROOT / args.output / args.candidate
    out.mkdir(parents=True, exist_ok=True)
    run_protocol = dict(protocol=protocol, candidate=args.candidate, stage=args.stage, scene_ids=[s['id'] for s in scenes])
    if (out / 'protocol.json').exists():
        assert json.loads((out / 'protocol.json').read_text()) == run_protocol
    else:
        write(out / 'protocol.json', run_protocol)
    set_seed(protocol['seed'])  # Once, before model loading, as in the public entry point.
    torch.set_num_threads(4)
    model = OfficialBokehDiff(source, ROOT / 'models/BokehDiff')
    if not (out / 'model-loading.json').exists():
        write(out / 'model-loading.json', model.weight_loading)
    # Resume only a contiguous prefix. Restore its RNG snapshot; never skip random draws.
    prefix = 0
    while prefix < len(scenes) and (out / 'timing' / scenes[prefix]['id'] / 'complete.json').exists():
        sid = scenes[prefix]['id']
        assert len(list((out / 'clips' / sid / 'pred_bokehdiff').glob('*.png'))) == 25
        prefix += 1
    assert not any((out / 'timing' / s['id'] / 'complete.json').exists() for s in scenes[prefix:])
    if prefix:
        state = torch.load(out / 'timing' / scenes[prefix - 1]['id'] / 'rng-after.pt', weights_only=True)
        assert state['next_scene_index'] == prefix
        torch.set_rng_state(state['cpu'])
        torch.cuda.set_rng_state(state['cuda'], device=0)
    print('LOADED', args.candidate, 'resume_scene', prefix, 'PISA', len(model.weight_loading['PISA_processors']), flush=True)
    for index, scene in enumerate(scenes[prefix:], prefix):
        sid = scene['id']
        src = clips / sid
        pred = out / 'clips' / sid / 'pred_bokehdiff'
        timing = out / 'timing' / sid
        pred.mkdir(parents=True, exist_ok=True)
        timing.mkdir(parents=True, exist_ok=True)
        meta = json.loads((src / 'metadata.json').read_text())
        paths = sorted((src / 'disparity').glob('*.png'))
        assert len(paths) == 25
        rows = []
        for t, path in enumerate(paths):
            assert path.name.startswith(f'{t:05d}_zf_')
            focus = float(path.stem.split('_zf_')[1])
            assert abs(focus - meta['frames'][t]['focus_disparity']) <= 5.1e-9
            rgb = np.asarray(Image.open(src / 'rgb' / f'{t:05d}.png')).astype(np.float32) / 255
            d = np.asarray(Image.open(path)).astype(np.float64) / 255
            mapped, fn, geometry = map_controls(d, focus, scene['K'], args.candidate)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.monotonic()
            image, audit = model.predict(rgb, mapped, fn, scene['K'], protocol['seed'],
                                         reset_global_seed=False, quantization='round_png8')
            torch.cuda.synchronize()
            audit.update(scene=sid, frame=t, geometry=geometry, seconds=time.monotonic() - start,
                         peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30)
            assert abs(audit['defocus_condition_range'][1] - geometry['max_defocus_condition']) < 1e-5
            dest, record = pred / f'{t:05d}.png', timing / f'{t:05d}.json'
            if dest.exists():
                with Image.open(dest) as old:
                    assert np.array_equal(np.asarray(old), np.asarray(image)), 'Partial-scene replay changed PNG; preserve and investigate'
            else:
                with dest.open('xb') as f:
                    image.save(f, format='PNG')
            if record.exists():
                audit = json.loads(record.read_text())
                assert audit['geometry'] == geometry and not audit['reset_global_seed']
            else:
                write(record, audit)
            rows.append(audit)
            print('FRAME', sid, t, round(audit['seconds'], 3), flush=True)
        state = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(0), next_scene_index=index + 1)
        checkpoint = timing / 'rng-after.pt'
        if checkpoint.exists():
            old = torch.load(checkpoint, weights_only=True)
            assert old['next_scene_index'] == state['next_scene_index']
            assert torch.equal(old['cpu'], state['cpu']) and torch.equal(old['cuda'], state['cuda'])
        else:
            with checkpoint.open('xb') as f:
                torch.save(state, f)
        write(timing / 'complete.json', dict(scene=sid, frames=25,
              peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in rows)))
        print('DONE', sid, flush=True)
    if not (out / 'complete.json').exists():
        write(out / 'complete.json', dict(status='CANDIDATE_INFERENCE_COMPLETE', stage=args.stage,
              scenes=len(scenes), frames=25 * len(scenes), candidate=args.candidate))
    print('ALL_FRAMES_COMPLETE', args.candidate, flush=True)


if __name__ == '__main__':
    main()
