"""Generate E6 planar inputs from the explicit source-level development manifest."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synthetic.common import srgb_to_linear, linear_to_srgb


def save_rgb(path, rgb):
    Image.fromarray(np.uint8(np.round(linear_to_srgb(rgb) * 255))).save(path)


def generate(cfg, scene, out, count):
    W, H = cfg['width'], cfg['height']
    folder = out / scene['id']
    for name in ['rgb', 'disparity', 'depth', 'alpha', 'mask', 'layers']:
        (folder / name).mkdir(parents=True, exist_ok=False)
    bg = np.array(Image.open(ROOT / scene['background_file']).convert('RGB'))
    h, w = bg.shape[:2]
    scale = max(W / w, H / h)
    bg = cv2.resize(bg, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    y, x = (bg.shape[0] - H) // 2, (bg.shape[1] - W) // 2
    bg = bg[y:y+H, x:x+W]
    Image.fromarray(np.pad(bg, ((64, 64), (64, 64), (0, 0)), mode='reflect')).save(folder / 'layers/background.png')
    background = srgb_to_linear(bg.astype(np.float32) / 255)
    dynamic = 'source_video' in scene
    if dynamic:
        assert scene['frame_stride'] == 1, 'Only consecutive source frames are implemented'
    caps, assets = [], []
    if dynamic:
        caps = [cv2.VideoCapture(str(ROOT / scene[key])) for key in ['source_video', 'source_alpha']]
        assert all(c.isOpened() for c in caps)
        for c in caps:
            c.set(cv2.CAP_PROP_POS_FRAMES, scene['start_frame'])
    else:
        for file in scene['foreground_files']:
            color = np.array(Image.open(ROOT / file).convert('RGB'))
            alpha = np.array(Image.open(ROOT / file.replace('/fg/', '/mask/')).convert('L'))
            assert not any(edge.any() for edge in [alpha[0],alpha[-1],alpha[:,0],alpha[:,-1]]), 'AM source is truncated at image boundary; select a complete foreground'
            assets.append((color, alpha))
    config = {key: cfg[key] for key in ['width', 'height', 'seed', 'focal_length_mm', 'sensor_width_mm']}
    config.update(K=scene['K'], background_depth_m=scene['background_depth_m'])
    meta = dict(config=config, source=scene, frames=[], role='E6_development_only',
                geometry='dynamic layered planar synthesis',
                color_space='linear-light composition, sRGB quantized PNG textures',
                disparity_rule='inverse axial meters, no scene normalization, alpha >0.5 visible layer; PNG8 is adapter only',
                dynamic_layout='cover_canvas_centered; source-frame cuts stay at/outside canvas boundary' if dynamic else None,
                opencv=cv2.__version__)
    assert scene['foreground_depths_m'] == sorted(scene['foreground_depths_m'])
    assert max(scene['foreground_depths_m']) < scene['background_depth_m']
    selected = set(np.linspace(0, cfg['frames']-1, count, dtype=int).tolist())
    try:
        for t in range(cfg['frames']):
            if dynamic:
                decoded = [cap.read() for cap in caps]
                assert all(ok for ok, _ in decoded), (scene['id'], t)
                fg, alpha = [im for _, im in decoded]
                assert fg.shape == alpha.shape
                assets = [(cv2.cvtColor(fg, cv2.COLOR_BGR2RGB), alpha[:, :, 0])]
            if t not in selected:
                continue
            assert len(assets) == len(scene['foreground_depths_m'])
            layers, transforms = [], []
            index = len(meta['frames'])
            for li, ((color, alpha), z) in enumerate(zip(assets, scene['foreground_depths_m'])):
                color = srgb_to_linear(color.astype(np.float32) / 255)
                alpha = alpha.astype(np.float32) / 255
                h, w = alpha.shape
                # A cropped source person must not acquire a floating rectangular cut edge.
                # Cover the canvas so every source-frame boundary is outside/on the output edge.
                scale = max(W / w, H / h) if dynamic else min(420 / h, 650 / w) * (0.8 if li else 1)
                phase = 2 * np.pi * t / (cfg['frames']-1)
                angle = 0 if dynamic else scene['motion']['rotation_amplitude_degrees'] * np.sin(phase)
                dx = 0 if dynamic else scene['motion']['translation_x_amplitude_px'] * np.sin(phase)
                dy = 0 if dynamic else scene['motion']['translation_y_amplitude_px'] * np.cos(phase)
                matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
                matrix[:, 2] += np.array([W * (.5 + .13 * li) + dx, H * .5 + dy]) - np.array([w/2, h/2])
                aa = cv2.warpAffine(alpha, matrix, (W, H), flags=cv2.INTER_LINEAR)
                premul = cv2.warpAffine(color * alpha[..., None], matrix, (W, H), flags=cv2.INTER_LINEAR)
                cc = np.divide(premul, aa[..., None], out=np.zeros_like(premul), where=aa[..., None] > 1e-6)
                file = f'layers/{index:05d}_fg{li}.png'
                rgba = np.dstack((linear_to_srgb(cc), aa))
                Image.fromarray(np.uint8(np.round(np.clip(rgba, 0, 1)*255))).save(folder / file)
                q = np.array(Image.open(folder / file)).astype(np.float32)/255
                layers.append((srgb_to_linear(q[:, :, :3]), q[:, :, 3], z, file))
                transforms.append(matrix.tolist())
            rgb = background.copy()
            disp = np.full((H, W), 1 / scene['background_depth_m'], np.float32)
            alpha = np.zeros((H, W), np.float32)
            for cc, aa, z, _ in reversed(layers):
                rgb = cc * aa[..., None] + rgb * (1-aa[..., None])
                disp = np.where(aa > .5, 1/z, disp)
                alpha = aa + alpha * (1-aa)
            focus = 1 / scene['focus_distance_m']
            assert np.isfinite(rgb).all() and (disp > 0).all() and (disp <= 1).all()
            save_rgb(folder / 'rgb' / f'{index:05d}.png', rgb)
            Image.fromarray(np.uint8(np.round(disp*255))).save(folder / 'disparity' / f'{index:05d}_zf_{focus:.8f}.png')
            Image.fromarray(np.uint8(np.round(alpha*255))).save(folder / 'alpha' / f'{index:05d}.png')
            Image.fromarray(np.uint8(alpha > .5)*255).save(folder / 'mask' / f'{index:05d}.png')
            np.save(folder / 'depth' / f'{index:05d}.npy', 1/disp)
            meta['frames'].append(dict(index=index, source_frame=scene.get('start_frame', 0)+t,
                timeline_frame=t, focus_distance_m=scene['focus_distance_m'], focus_disparity=focus,
                foreground_files=[layer[3] for layer in layers], foreground_depths_m=scene['foreground_depths_m'],
                transforms=transforms, layer_order='front_to_back'))
    finally:
        for cap in caps:
            cap.release()
    (folder / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print(scene['id'], len(meta['frames']), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT / 'configs/renderer-e6.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only', default='')
    p.add_argument('--frames', type=int, default=25)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    assert 1 <= a.frames <= cfg['frames']
    chosen = set(a.only.split(',')) if a.only else {s['id'] for s in cfg['scenes']}
    assert chosen <= {s['id'] for s in cfg['scenes']}
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'config.json').write_text(json.dumps(cfg, indent=2))
    (a.output / 'render_settings.json').write_text(json.dumps({'fstop_scale': cfg['mapping']['primary_scale']}))
    for scene in cfg['scenes']:
        if scene['id'] in chosen:
            generate(cfg, scene, a.output, a.frames)


if __name__ == '__main__':
    main()
