"""Prepare E6 calibration inputs and inspect candidate videos; no bulk rendering."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/renderer-e6.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    W, H = cfg['width'], cfg['height']
    calibration = out / 'calibration'
    calibration.mkdir()
    (calibration / 'config.json').write_text(json.dumps({'cycles_fstop_scale': 1.0}))
    for case in cfg['calibration_cases']:
        folder = calibration / case['id']
        (folder / 'layers').mkdir(parents=True)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, W // 2:] = 255
        Image.fromarray(np.pad(rgb, ((64, 64), (64, 64), (0, 0)), mode='reflect')).save(folder / 'layers/background.png')
        c = {k: cfg[k] for k in ['width', 'height', 'seed', 'focal_length_mm', 'sensor_width_mm']}
        c.update(K=case['K'], background_depth_m=case['object_depth_m'])
        frame = dict(index=0, foreground_files=[], foreground_depths_m=[],
                     focus_distance_m=case['focus_distance_m'], focus_disparity=1 / case['focus_distance_m'])
        meta = dict(config=c, frames=[frame], role='independent_calibration',
                    nominal_radius_px=case['K'] * abs(1 / case['object_depth_m'] - 1 / case['focus_distance_m']))
        (folder / 'metadata.json').write_text(json.dumps(meta, indent=2))

    previews = out / 'candidate_previews'
    previews.mkdir()
    report = []
    sheet = Image.new('RGB', (960, 12 * 204), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, scene in enumerate(s for s in cfg['scenes'] if 'source_video' in s):
        paths = [ROOT / scene[k] for k in ['source_video', 'source_alpha']]
        caps = [cv2.VideoCapture(str(p)) for p in paths]
        assert all(c.isOpened() for c in caps), paths
        properties = [{name: c.get(key) for name, key in [('width', cv2.CAP_PROP_FRAME_WIDTH), ('height', cv2.CAP_PROP_FRAME_HEIGHT), ('fps', cv2.CAP_PROP_FPS), ('frames', cv2.CAP_PROP_FRAME_COUNT)]} for c in caps]
        assert properties[0] == properties[1], properties
        assert scene['start_frame'] + cfg['frames'] <= properties[0]['frames']
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene['start_frame'])
        previous = None
        delta = []
        frames = []
        for t in range(cfg['frames']):
            decoded = [cap.read() for cap in caps]
            assert all(ok for ok, _ in decoded), (scene['id'], t)
            fg, alpha = [im for _, im in decoded]
            assert fg.shape == alpha.shape
            small = cv2.resize(alpha[:, :, 0], (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            assert small.max() > 0, (scene['id'], t)
            if previous is not None:
                delta.append(float(np.abs(small - previous).mean()))
            previous = small
            if t in [0, 12, 24]:
                color = cv2.cvtColor(cv2.resize(fg, (320, 180)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                # Screening preview only, not the linear-light renderer input.
                yy, xx = np.indices(small.shape)
                checker = (0.3 + 0.4 * ((xx // 12 + yy // 12) % 2))[..., None]
                image = Image.fromarray(np.uint8(np.clip(color * small[..., None] + checker * (1-small[..., None]), 0, 1)*255))
                image.save(previews / f'{scene["id"]}_{t:02d}.png')
                sheet.paste(image, (len(frames) * 320, i * 204 + 24))
                frames.append(t)
        for cap in caps:
            cap.release()
        draw.text((4, i * 204 + 4), f'{scene["id"]} source {paths[0].name}, frames 30/42/54', fill='black')
        report.append(dict(scene=scene['id'], video=scene['source_video'], properties=properties[0], decoded_pairs=cfg['frames'], mean_alpha_change=float(np.mean(delta)), coverage='pending visual review'))
        print(scene['id'], properties[0], 'alpha_delta', np.mean(delta), flush=True)
    sheet.save(previews / 'contact_sheet.jpg')
    (out / 'input_preflight.json').write_text(json.dumps({'opencv': cv2.__version__, 'videos': report, 'calibration_cases': len(cfg['calibration_cases'])}, indent=2))


if __name__ == '__main__':
    main()
