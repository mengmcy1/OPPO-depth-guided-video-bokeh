"""1b′ fixed visible primitive, protected depth, and rendering guardrails."""
import argparse
import os
import shutil
import subprocess
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import numpy as np
import torch

from route3_ovg_1b_common import CONFIG, OUT as DATA_RUN, ROOT, read
from route3_ovg_1b_optical_data import background_samples, b_oracle, place, synthetic_optical
from route3_ovg_1b_train import load_model, train, training_objective
import route3_ovg_data as data
from route3_ovg_subpixel import save_json
from route3_upper_bound_evaluate import display
from src.dual_layer import depth_band
from src.layered_aperture import merge_layers
from src.ovg_net import protect_inverse_depth
from src.ovg_optics import fixed_visible_surfels, raster

PROFILE_PATH = 'configs/route3-ovg-1b-prime-2026-11-18.json'
PROFILE = read(ROOT/PROFILE_PATH)
OUT = ROOT/PROFILE['output']


def guard():
    OUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(OUT).free < 60*2**30:
        raise RuntimeError('Data disk available space is below 60 GiB')


def psnr(sse, pixels):
    return float(-10 * np.log10(max(sse/(3*pixels), 1e-12)))


def zero_score(image, target):
    diff = display(image.detach().cpu().numpy()).astype(np.float64) - target
    sse = float(np.square(diff).sum())
    pixels = target.shape[0]*target.shape[1]
    return dict(sse=sse, pixels=pixels, psnr_db=psnr(sse,pixels))


def probe_cases():
    spec = PROFILE['zero_aperture']['validation_probe']
    for name in read(data.PREVIOUS/'split.json')['heldout']:
        source = data.PREVIOUS/'geometry/original'/name
        entry = next(e for e in read(source/'prepared.json')['rows'] if e['frame']==spec['frame'])
        manifest = read(data.PREVIOUS/'evaluation-input/original'/name/'512-13/manifest.json')
        case = next(r['case'] for r in manifest['rows']
                    if all(r['case'][k]==v for k,v in spec.items()))
        yield name,entry,case


@torch.inference_mode()
def zero_monitor(net, one_visible=False, *, sample=data.sample, method_focus=False):
    zero = PROFILE['zero_aperture']
    scores = []
    for name,entry,case in probe_cases():
        item = ('B', name, data.TRAIN_FRAMES.index(entry['frame']))
        x,_ = sample(item,'cuda')
        pred = net(x, protect_residual=True, render_heads=False)
        visible = fixed_visible_surfels(pred,x)
        size = tuple(pred['depth1'].shape[-2:])
        if len(visible.means)!=size[0]*size[1]*25:
            raise ValueError('Fixed primitive expected every pixel depth and 25 surfels per pixel')
        obs = data.load(entry['input_npz'], ('rgb_srgb','depth'))
        K = x['K'][0]
        center = K.new_zeros((1,2))
        focus = case['focus_m']
        sole = raster(visible,K,focus,center,*size)[0]
        vis_stats = zero_score(sole, obs['rgb_srgb'])
        row = dict(scene=name, frame=entry['frame'], visible_gaussians=len(visible.means),
                   visible_only=vis_stats, visible_below_floor=vis_stats['psnr_db']<zero['visible_only_min_db'])
        if one_visible:
            return row
        parents,gt,background = b_oracle(name,entry['frame'])
        band_depth = obs['depth']
        if method_focus:
            from route3_depth_interface import subject_mask
            from route3_ovg_1b_optical_data import geometry_root
            target = data.load(geometry_root(name)/'geometry'/f'f{entry["frame"]:03d}-center.npz')
            subject = subject_mask(target, case['nominal_focus_m'])
            band_depth = pred['depth1'][0].cpu().numpy()
            focus = float(np.median(band_depth[subject & (band_depth > 0)]))
            row['focus_distance_m'] = focus
        support = depth_band(band_depth, float(K[0,0]), focus, .05/(2*case['fstop']))[0]
        uv = np.floor(parents['uv']).astype(int).clip(0,511)
        parents = dict(parents, recoverable=parents['recoverable'] & support[uv[:,1],uv[:,0]])
        hidden,_ = place(parents,gt,background,pred['depth1'][0],K,512)
        F2 = background_samples(dict(tiers='B',entry=entry),K,focus,center,size)
        merged = raster(merge_layers(visible,hidden),K,focus,center,*size,F2)[0]
        row['with_GT_hidden_and_F2'] = zero_score(merged,obs['rgb_srgb'])
        row['merged_below_floor'] = row['with_GT_hidden_and_F2']['psnr_db']<zero['merged_with_f2_min_db']
        scores.append(row)
    pooled={}
    for arm in ('visible_only','with_GT_hidden_and_F2'):
        pooled[arm]=psnr(sum(r[arm]['sse'] for r in scores),sum(r[arm]['pixels'] for r in scores))
    return dict(frames=scores,pooled_psnr_db=pooled,thresholds=zero)


def check():
    net = load_model().train()
    net.footprint.requires_grad_(False)
    rng = np.random.default_rng(CONFIG['seed'])
    generator = torch.Generator(device='cuda').manual_seed(CONFIG['seed'])
    item = next(i for i in data.inventory('train')['B'] if synthetic_optical(i))
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    loss,terms,counts,hidden = training_objective(net,item,rng,generator,fixed=True)
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    peak=torch.cuda.max_memory_allocated()/2**20
    if not {'occ','zero','bokeh'} <= terms.keys() or not hidden.get('placed'):
        raise RuntimeError('Fixed B batch is missing optical supervision or its hidden substitute')
    if 'normal' in terms or net.geometry.weight.grad[2:5].abs().any() or any(
            p.grad is not None for p in net.footprint.parameters()):
        raise RuntimeError('A disabled rendering head received a gradient')
    # Pixelwise check of the unprotected, trusted, and agreeing branches.
    z=torch.tensor([[[[2.,0.,2.02]],[[2.,2.,2.]],[[0.,0.,0.]]]])
    confidence=torch.tensor([[[[.8,0.,.8]],[[.59,.6,.6]],[[0.,0.,0.]]]])
    fused=torch.ones(1,1,3)
    gate=protect_inverse_depth(fused,2*fused,z,confidence)
    torch.testing.assert_close(gate,torch.tensor([[[2.,1.20,1.05]]]))
    visible = zero_monitor(net.eval(),one_visible=True)
    report = dict(passed=not visible['visible_below_floor'], sample=list(item),
                  loss=float(loss.detach()),grad_norm=float(grad),terms={k:float(v.detach()) for k,v in terms.items()},
                  counts=counts,hidden_placement=hidden,visible_only=visible,batch_seconds=elapsed,
                  peak_allocated_MiB=peak,optimizer_updates=0,
                  gate_expected_residual_fraction=[1.,.20,.05])
    save_json(OUT/'Q1/check.json',report)
    print('PRIME_CHECK',report,flush=True)
    if not report['passed']:
        raise RuntimeError('Q1 visible-only zero aperture is below 50 dB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job',choices=['check','train'])
    args=parser.parse_args()
    guard()
    data.OUT = DATA_RUN
    subprocess.run(['nvidia-smi'],check=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the assigned device 0')
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision('high')
    if args.job=='check':
        check()
    else:
        train(output=OUT,profile=PROFILE,monitor=zero_monitor)
