"""Full official MPIB demo pipeline; two predeclared gamma outputs per frame."""
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
import torch.nn.functional as F
import yaml
from munch import DefaultMunch

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def setup():
    import cupy
    if not hasattr(cupy, 'int'):
        cupy.int = np.int32
    if not hasattr(cupy.cuda, 'compile_with_cache'):
        cupy.cuda.compile_with_cache = lambda source: cupy.RawModule(code=source)
    sys.path.insert(0, str(ROOT.parent / 'MPIB'))
    from models import AUPNet, MPINet, FFCResNetGenerator, MPIBokehRenderer
    cfg = DefaultMunch.fromDict(yaml.safe_load((ROOT / 'configs/mpib-inference.yaml').read_text()), 'BokehConfig')
    nets, counts = {}, {}
    for key, cls in [('inpnet', FFCResNetGenerator), ('mpinet', MPINet), ('aupnet', AUPNet)]:
        net = cls(**getattr(cfg, key))
        state = torch.load(ROOT / f'models/MPIB/{key}.pth', map_location='cpu', weights_only=False)['model']
        net.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        nets[key] = torch.nn.DataParallel(net.cuda().eval())
        counts[key] = sum(p.numel() for p in net.parameters())
    nets['renderer'] = torch.nn.DataParallel(MPIBokehRenderer(**cfg.renderer).cuda().eval())
    return nets, counts


@torch.no_grad()
def predict_layers(image, disp, K, nets):
    # Statements follow official demo.py, keeping its mask/inpainting/MPI/upsampling settings.
    from utils import resize, dilate, mask_generator
    inpnet, mpinet, aupnet = (nets[k] for k in ('inpnet', 'mpinet', 'aupnet'))
    inpaint_size = 1024
    disp_re = resize(disp, short_size=inpaint_size, ensure_multiple_of=8)
    gy = cv2.GaussianBlur(cv2.Sobel(disp_re, ddepth=-1, dx=0, dy=1), (7, 7), 0)
    gx = cv2.GaussianBlur(cv2.Sobel(disp_re, ddepth=-1, dx=1, dy=0), (7, 7), 0)
    grad = np.stack((gx, gy), axis=-1)
    mask = (np.abs(grad).sum(axis=-1, keepdims=True) > .2).astype(np.float32)
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    for i in range(1, labels.max() + 1):
        if (labels == i).sum() < 1 / 10000 * labels.shape[0] * labels.shape[1]:
            mask[labels == i] = 0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
    disp = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0).cuda()
    grad = torch.from_numpy(grad).permute(2, 0, 1).unsqueeze(0).cuda()
    mask = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).cuda()
    _, _, H, W = image.shape
    image_re = resize(image, short_size=inpaint_size, ensure_multiple_of=8)
    mask = mask_generator(grad, mask, iteration=int(2 * K * min(1, inpaint_size / min(H, W))), dilate_kernel_size=5)
    inpaint_bg = inpnet(image_re * (1 - mask), mask)
    mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=True)
    inpaint_bg = F.interpolate(inpaint_bg, size=(H, W), mode='bilinear', align_corners=True)
    bg = mask * inpaint_bg + (1 - mask) * image
    base_size = min(512, int(min(H, W) * 30 / K))
    image_re = resize(image, short_size=base_size, ensure_multiple_of=32)
    disp_re = resize(disp, short_size=base_size, ensure_multiple_of=32)
    weight, alpha, _ = mpinet(image_re, disp_re)
    weight = dilate(weight[0], kernel_size=5).unsqueeze(0)
    weight = F.interpolate(weight[0], size=(H, W), mode='bilinear', align_corners=True).unsqueeze(0)
    rgb = weight * bg.unsqueeze(1) + (1 - weight) * image.unsqueeze(1)
    scale = max(H / image_re.shape[2], 1)
    for _ in range(int(np.log2(scale))):
        image_re = F.interpolate(image, size=(image_re.shape[2]*2, image_re.shape[3]*2), mode='bilinear', align_corners=True)
        disp_re = F.interpolate(disp, size=(disp_re.shape[2]*2, disp_re.shape[3]*2), mode='bilinear', align_corners=True)
        alpha = aupnet(image_re.clamp(0, 1), disp_re, alpha)
    image_re = resize(image, short_size=min(H, W), ensure_multiple_of=2)
    disp_re = resize(disp, short_size=min(H, W), ensure_multiple_of=2)
    alpha = F.interpolate(alpha[0], size=(image_re.shape[2]//2, image_re.shape[3]//2), mode='bilinear', align_corners=True).unsqueeze(0)
    alpha = aupnet(image_re.clamp(0, 1), disp_re, alpha)
    alpha = F.interpolate(alpha[0], size=(H, W), mode='bilinear', align_corners=True).unsqueeze(0)
    assert rgb.shape == (1, 32, 3, 576, 1024) and alpha.shape == (1, 32, 1, 576, 1024)
    assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
    return rgb, alpha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/mpib-adapter-recovery-20260928'))
    p.add_argument('--only', default='')
    p.add_argument('--max-frames', type=int, default=25)
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    protocol = json.loads((ROOT / 'configs/mpib-adapter-recovery.json').read_text())
    head = subprocess.check_output(['git', '-C', str(ROOT.parent/'MPIB'), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == protocol['author_commit']
    clips = ROOT / protocol['input']
    cfg = json.loads((clips/'config.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        save_json(path, protocol)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    nets, counts = setup()
    calls = {k: 0 for k in ('inpnet', 'mpinet', 'aupnet')}
    def hook(key):
        def record(*unused):
            calls[key] += 1
        return record
    for key in calls:
        nets[key].register_forward_hook(hook(key))
    print('CHECKPOINTS_STRICTLY_LOADED', counts, flush=True)
    for scene in cfg['scenes']:
        sid = scene['id']
        if args.only and sid not in args.only.split(','):
            continue
        src = clips/sid
        meta = json.loads((src/'metadata.json').read_text())
        dirs = {g: args.output/f'gamma_{g:g}'/'clips'/sid/'pred_mpib' for g in protocol['gamma_candidates']}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        logdir = args.output/'timing'/sid
        logdir.mkdir(parents=True, exist_ok=True)
        complete = logdir/'complete.json'
        if complete.exists():
            assert json.loads(complete.read_text())['frames'] == 25
            assert all(len(list(d.glob('*.png'))) == 25 for d in dirs.values())
            print('REUSE', sid, flush=True)
            continue
        rows = []
        for t, dp in enumerate(sorted((src/'disparity').glob('*.png'))[:args.max_frames]):
            frame_record = logdir/f'{t:05d}.json'
            if frame_record.exists():
                assert all((d/f'{t:05d}.png').exists() for d in dirs.values())
                rows.append(json.loads(frame_record.read_text()))
                continue
            rgb = cv2.cvtColor(cv2.imread(str(src/'rgb'/f'{t:05d}.png')), cv2.COLOR_BGR2RGB).astype(np.float32)/255
            disp = cv2.imread(str(dp), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255
            focus = float(dp.stem.split('_zf_')[1])
            assert abs(focus-meta['frames'][t]['focus_disparity']) <= 5.1e-9
            assert rgb.shape == (576,1024,3) and disp.shape == (576,1024) and 0 <= disp.min() <= disp.max() <= 1
            for key in calls:
                calls[key] = 0
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
            layers, alpha = predict_layers(rgb, disp, scene['K'], nets)
            torch.cuda.synchronize()
            row = dict(frame=t, K=scene['K'], focus=focus, disparity_range=[float(disp.min()),float(disp.max())], shared_layer_seconds=time.perf_counter()-start, calls=dict(calls), gamma={})
            assert all(v > 0 for v in calls.values())
            for gamma, directory in dirs.items():
                start = time.perf_counter()
                with torch.no_grad():
                    value = nets['renderer'](layers, alpha, scene['K'], focus, gamma, norm=True, increase_focus_range=True)
                torch.cuda.synchronize()
                assert value.shape == (1,3,576,1024) and torch.isfinite(value).all()
                x = value[0].permute(1,2,0).cpu().numpy()
                row['gamma'][str(gamma)] = dict(render_seconds=time.perf_counter()-start, range=[float(x.min()),float(x.max())], clipped_fraction=float(((x<0)|(x>1)).mean()))
                out = directory/f'{t:05d}.png'
                quantized = np.uint8(np.round(np.clip(x,0,1)*255))
                if out.exists():
                    assert np.array_equal(np.asarray(Image.open(out)), quantized), 'Interrupted frame differs; preserve output and investigate'
                else:
                    Image.fromarray(quantized).save(out)
            row['peak_allocated_GiB'] = torch.cuda.max_memory_allocated()/2**30
            save_json(frame_record, row)
            rows.append(row)
            del layers, alpha, value
            print('FRAME',sid,t,'seconds',round(row['shared_layer_seconds']+sum(x['render_seconds'] for x in row['gamma'].values()),3),flush=True)
        if len(rows) == 25:
            save_json(complete, dict(scene=sid,frames=25,parameters=counts,shared_layer_seconds=sum(x['shared_layer_seconds'] for x in rows),peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in rows),scope='Shared layers for two gamma candidates; no temporal cache; not standardized efficiency'))
            print('DONE',sid,flush=True)


if __name__ == '__main__':
    main()
