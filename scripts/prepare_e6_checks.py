"""Build static noise and black/white/checker/occlusion controls from E6 clips."""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synthetic.common import srgb_to_linear, linear_to_srgb


def copy_layers(source, dest, meta):
    (dest / 'layers').mkdir(parents=True)
    for file in set(f for frame in meta['frames'] for f in frame['foreground_files']):
        shutil.copy2(source / file, dest / file)
    shutil.copy2(source / 'layers/background.png', dest / 'layers/background.png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    for kind in ['noise', 'edges']:
        (a.output / kind).mkdir()
        (a.output / kind / 'config.json').write_text(json.dumps({'cycles_fstop_scale': 1.0}))
    # Freeze the strongest foreground defocus case; only rendering seed changes in time.
    source = a.input / 'vm_11'
    meta = json.loads((source / 'metadata.json').read_text())
    first = meta['frames'][0]
    meta['frames'] = [dict(copy.deepcopy(first), index=i) for i in range(25)]
    meta['role'] = 'static_sampling_noise_control'
    dest = a.output / 'noise/static_vm11'
    copy_layers(source, dest, meta)
    (dest / 'metadata.json').write_text(json.dumps(meta, indent=2))
    # Continuous dynamic counterpart uses real consecutive frames, not 0/12/24 subsampling.
    source = a.input / 'vm_02'
    meta = json.loads((source / 'metadata.json').read_text())
    dest = a.output / 'noise/dynamic_vm02'
    copy_layers(source, dest, meta)
    (dest / 'metadata.json').write_text(json.dumps(meta, indent=2))
    for source_id in ['vm_02', 'am_05']:
        source = a.input / source_id
        original = json.loads((source / 'metadata.json').read_text())
        for background in ['black', 'white', 'checker']:
            for reverse in ([False, True] if source_id == 'am_05' else [False]):
                meta = copy.deepcopy(original)
                meta['frames'] = [dict(meta['frames'][0], index=0)]
                frame = meta['frames'][0]
                if reverse:
                    # Reverse which fixed screen-space texture is nearer, without moving it.
                    frame['foreground_files'].reverse()
                    frame['transforms'].reverse()
                name = f'{source_id}_{background}' + ('_reverse' if reverse else '')
                dest = a.output / 'edges' / name
                copy_layers(source, dest, meta)
                W, H = meta['config']['width'], meta['config']['height']
                if background == 'checker':
                    yy, xx = np.indices((H, W))
                    bg = np.repeat((((yy//32 + xx//32) % 2)*255)[..., None], 3, axis=2).astype(np.uint8)
                else:
                    bg = np.full((H, W, 3), 255 if background == 'white' else 0, np.uint8)
                Image.fromarray(np.pad(bg, ((64, 64), (64, 64), (0, 0)), mode='reflect')).save(dest / 'layers/background.png')
                rgb = srgb_to_linear(bg.astype(np.float32)/255)
                for file in reversed(frame['foreground_files']):
                    q = np.array(Image.open(dest / file)).astype(np.float32)/255
                    rgb = srgb_to_linear(q[..., :3])*q[..., 3:] + rgb*(1-q[..., 3:])
                (dest / 'rgb').mkdir()
                Image.fromarray(np.uint8(np.round(linear_to_srgb(rgb)*255))).save(dest / 'rgb/00000.png')
                meta['role'] = 'edge_control'
                meta['edge_control'] = dict(background=background, reverse_depth_order=reverse,
                    note='Visual and pinhole check; bokeh naturally extends beyond sharp alpha')
                (dest / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print('25 static + 25 dynamic noise frames; 9 edge controls', flush=True)


if __name__ == '__main__':
    main()
