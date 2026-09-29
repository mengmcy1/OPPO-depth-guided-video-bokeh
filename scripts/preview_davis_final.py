"""Fixed DAVIS visual set: cross-method keyframes and chronological all25-frame sheets."""
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['deeplens', 'drbokeh', 'bokehme', 'mpib', 'bokehdiff', 'any2bokeh']


def main():
    config = json.loads((ROOT / 'configs/davis-baselines.json').read_text())
    data = json.loads((ROOT / config['data_protocol']).read_text())
    base = ROOT / config['output']; out = base / 'final-review'; out.mkdir(exist_ok=True)
    for name in config['preflight_sequences']:
        images = {}
        for method in ['input'] + METHODS:
            if method == 'input':
                paths = sorted((ROOT / data['derived'] / 'eval' / name / 'rgb').glob('*.jpg'))
                images[method] = [Image.fromarray(cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (1024, 576), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)) for p in paths]
            else:
                paths = sorted((base / method / 'predictions' / name).glob('*.png'))
                images[method] = [Image.open(p).convert('RGB') for p in paths]
            assert len(paths) == 25
            sheet = Image.new('RGB', (1280, 840), 'white'); draw = ImageDraw.Draw(sheet)
            for t, im in enumerate(images[method]):
                x = t % 5 * 256; y = t // 5 * 168
                sheet.paste(im.resize((256, 144)), (x, y + 24))
                draw.text((x + 4, y + 4), f'{name} {method} t{t:02d}', fill='black')
            dest = out / f'{name}-{method}-all25.jpg'
            if not dest.exists(): sheet.save(dest, quality=94)
        sheet = Image.new('RGB', (224 * 7, 150 * 5), 'white'); draw = ImageDraw.Draw(sheet)
        for j, t in enumerate(config['keyframes']):
            for k, method in enumerate(['input'] + METHODS):
                sheet.paste(images[method][t].resize((224, 126)), (k * 224, j * 150 + 24))
                draw.text((k * 224 + 3, j * 150 + 3), f'{name} t{t} {method}', fill='black')
        dest = out / f'{name}-comparison.jpg'
        if not dest.exists(): sheet.save(dest, quality=94)
        print('FIXED_REVIEW_SHEETS', name, flush=True)


if __name__ == '__main__': main()
