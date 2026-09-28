"""Diagnose public PISA randomness and d=1 sampling, without changing its computation."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baselines.bokehdiff import OfficialBokehDiff, map_controls


def difference(a, b):
    delta = np.abs(a - b)
    png = np.abs(np.round(a * 255).astype(np.int16) - np.round(b * 255).astype(np.int16))
    return dict(max_abs=float(delta.max()), MAE=float(delta.mean()), PNG8_max_abs=int(png.max()),
                PNG8_different_fraction=float((png != 0).mean()))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    out = ROOT / 'outputs/bokehdiff-stochastic-review-20260928'
    out.mkdir(exist_ok=True)
    assert not (out / 'result.json').exists()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = OfficialBokehDiff(ROOT.parent / 'BokehDiff', ROOT / 'models/BokehDiff')
    # Scalar constant-plane example only: public pairwise broadcasting uses two pixel depths.
    # For a d=1 source column, the final occlusion comparison checks sampled disparity > 1.
    ps = torch.linspace(.1, .9, 7, device='cuda')
    ds = torch.tensor([0., .2, .5, .999, 1.], device='cuda')
    dp = ds[:, None] * ps + (1 - ps)
    native = (1 - dp) * ds[:, None] / (dp * (1 - ds[:, None]))
    limit = ps * ds[:, None] / dp
    assert torch.isnan(native[-1]).all() and torch.isfinite(limit).all()
    sampled = F.grid_sample(torch.ones(1, 1, 3, 3, device='cuda'),
                            torch.full((1, 1, 1, 2), float('nan'), device='cuda'), align_corners=False)
    boundary = dict(native_d1_nan_count=int(torch.isnan(native[-1]).sum()),
        analytic_limit_d1=limit[-1].cpu().tolist(), native_nan_grid_sample_finite=bool(torch.isfinite(sampled).all()),
        native_sample_gt_one=bool((sampled > 1).any()),
        finite_domain_factor_max_abs=float((native[:-1] - limit[:-1]).abs().max()),
        interpretation='Scalar limit is only a constant-plane illustration, not a rewrite of the pairwise public formula. In actual attention, a d=1 source column is compared against threshold 1. Native grid_sample handling and downstream occlusion are measured below; finite RGB alone is insufficient. No global geometry equivalence is claimed.')
    grid_rows = []
    original_grid = F.grid_sample
    original_attn = model.module.AttnProcessorDistReciprocal.__call__.__globals__['customized_scaled_dot_product_attention']
    attn_rows = []

    def grid_probe(value, grid, *args, **kwargs):
        result = original_grid(value, grid, *args, **kwargs)
        invalid = ~torch.isfinite(grid).all(-1)
        grid_rows.append(dict(grid_points=invalid.numel(), invalid_points=int(invalid.sum()),
            sampled_nonfinite=int((~torch.isfinite(result)).sum()), input_min=float(value.min()), input_max=float(value.max())))
        return result

    def attention_probe(*args, **kwargs):
        occ = kwargs['occ_map']
        assert torch.isfinite(occ).all()
        # disp_coc has shape (batch,2,positions); occ uses source pixels on its final axis.
        d1 = args[4][:, 0] == 1
        selected = occ[:, 0].transpose(-1, -2)[d1]
        attn_rows.append(dict(source_d1=int(d1.sum()), occ_at_d1_max=float(selected.max()) if selected.numel() else None,
                              dropout_p=kwargs['dropout_p'], train_argument=kwargs['train']))
        return original_attn(*args, **kwargs)

    F.grid_sample = grid_probe
    model.module.AttnProcessorDistReciprocal.__call__.__globals__['customized_scaled_dot_product_attention'] = attention_probe
    src = ROOT / 'outputs/any2bokeh-mpib-public-proxy-200-20260928/clips/repro_003'
    rgb = np.asarray(Image.open(src / 'rgb/00000.png')).astype(np.float32) / 255
    path = sorted((src / 'disparity').glob('*.png'))[0]
    d = np.asarray(Image.open(path)).astype(np.float64) / 255
    focus = float(path.stem.split('_zf_')[1])
    results = {}
    for candidate in ('shared_geometry', 'normalized'):
        mapped, fn, geometry = map_controls(d, focus, 16, candidate)
        start_grid, start_attn = len(grid_rows), len(attn_rows)

        def run(seed, reset):
            if reset:
                torch.manual_seed(seed)
            image, audit = model.predict(rgb, mapped, fn, 16, seed, reset_global_seed=False, quantization='round_png8')
            return image, model.output_float.copy(), audit

        image, first, audit = run(42, True)
        image.save(out / f'{candidate}-seed42.png')
        _, advancing, _ = run(42, False)
        _, repeat, _ = run(42, True)
        image43, second_seed, _ = run(43, True)
        image43.save(out / f'{candidate}-seed43.png')
        assert np.max(np.abs(first - repeat)) <= 1e-6
        torch.manual_seed(42)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda() * 2 - 1
        dt = torch.from_numpy(mapped)[None, None].cuda()
        control = torch.cat([dt, (dt - fn).abs() * (16 / 10)], dim=1)
        model.module.log_validation(model.pipeline, 42, 499, SimpleNamespace(device=torch.device('cuda:0')),
                                    aif_image=tensor, disp_coc=control)
        official = model.output_float.copy()
        assert np.array_equal(first, official)
        gs, ats = grid_rows[start_grid:], attn_rows[start_attn:]
        results[candidate] = dict(geometry=geometry, input_output=audit, seed42_repeat=difference(first, repeat),
            seed42_vs43=difference(first, second_seed), advancing_global_RNG=difference(first, advancing),
            adapter_vs_public_prequant_float=difference(first, official),
            grid_sample_calls=len(gs), grid_points=sum(x['grid_points'] for x in gs),
            invalid_grid_points=sum(x['invalid_points'] for x in gs),
            sampled_nonfinite=sum(x['sampled_nonfinite'] for x in gs),
            sampled_field_range=[min(x['input_min'] for x in gs), max(x['input_max'] for x in gs)],
            PISA_calls=len(ats), source_d1=sum(x['source_d1'] for x in ats),
            max_occlusion_at_d1=max((x['occ_at_d1_max'] for x in ats if x['occ_at_d1_max'] is not None), default=None),
            native_dropout_p=sorted(set(x['dropout_p'] for x in ats)))
        print(candidate, results[candidate], flush=True)
    F.grid_sample = original_grid
    model.module.AttnProcessorDistReciprocal.__call__.__globals__['customized_scaled_dot_product_attention'] = original_attn
    report = dict(status='STOCHASTIC_AND_NATIVE_BOUNDARY_DIAGNOSTICS_COMPLETE', boundary=boundary,
        scope='Fixed repro_003 frame0 only. 42/43 are sensitivity checks, never seed selection. No weights, PISA computation, or inputs clamped.',
        seed_probe_scope='Activation RNG reset after the same loaded weights for isolated sensitivity; recovery seeds once before model load.',
        candidates=results, loading=model.weight_loading)
    with (out / 'result.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
