"""Section 13.13 paired continuation, changing only the observed SHARP prior."""
import argparse
from collections import Counter
from functools import lru_cache, partial
import time

import numpy as np
import torch

import route3_ovg_data as data
from route3_ovg_1b_common import CONFIG, OUT as DATA_RUN
from route3_ovg_1b_prime import OUT as PRIME, PROFILE, zero_monitor
from route3_ovg_1b_train import train, training_pool
from route3_ovg_refocus import gpu_setup
from route3_ovg_subpixel import save_json, save_npz
from route3_sharp_s1b_calibrate import fit_inverse, inverse_stats
from route3_v2data_B_inputs import formal_unit, observation

OUT = data.ROOT/'outputs/route3-ovg-1c-2026-11-19-01'
INITIAL = PRIME/'checkpoints/step-030000.pt'
SEED = CONFIG['seed']
HIGH = .6
CONSISTENT = .03


def fit(name, frame, row, *, scene_seed=None):
    sharp, ds = row['depth'][:2]
    valid = (row['confidence'][0] > 0) & np.isfinite(sharp) & (sharp > 0)
    common = valid & (row['confidence'][1] >= HIGH) & np.isfinite(ds) & (ds > 0)
    pixels = np.flatnonzero(common)
    rng = np.random.default_rng([SEED, int(name.split('-')[2]) if scene_seed is None else scene_seed, frame])
    pixels = rng.permutation(pixels)
    n_validation = len(pixels) // 5
    heldout, train_pixels = pixels[:n_validation], pixels[n_validation:]
    source, target = sharp.ravel(), ds.ravel()
    a, b, reason = 1., 0., None
    report = dict(scene=name, frame=frame, common_pixels=len(pixels), fit_pixels=len(train_pixels),
                  holdout_pixels=len(heldout))
    if n_validation:
        report['holdout_before'] = inverse_stats(source[heldout], target[heldout])
    if len(train_pixels) < 2000:
        reason = 'insufficient_fit_pixels'
    else:
        low, median, high = np.percentile(1./target[train_pixels], [10, 50, 90])
        report['relative_inverse_p10_p90_span'] = float((high-low)/median)
        if high-low < .2*median:
            reason = 'insufficient_inverse_depth_span'
        else:
            try:
                fitted = fit_inverse(source[train_pixels], target[train_pixels], sharp[valid])
            except RuntimeError as error:
                reason = 'numerical_fit_failure'
                report['fit_error'] = str(error)
            else:
                a, b = fitted['a'], fitted['b']
                inverse = a/sharp[valid].astype(np.float64)+b
                before = report['holdout_before']
                after = inverse_stats(source[heldout], target[heldout], a, b)
                report.update(fit=fitted, holdout_after_proposed=after)
                if np.any(inverse <= 0):
                    reason = 'nonpositive_inverse_depth'
                elif after['inverse_RMSE'] > before['inverse_RMSE'] or after['sparse_depth_AbsRel'] > before['sparse_depth_AbsRel']:
                    reason = 'heldout_error_increased'
    mapped = sharp.copy()
    if reason is None:
        mapped[valid] = 1./(a/sharp[valid].astype(np.float64)+b)
    else:
        a, b = 1., 0.
    report.update(accepted=reason is None, fallback=reason, slope=a, intercept=b)
    if n_validation:
        report['holdout_after_applied'] = inverse_stats(source[heldout], target[heldout], a, b)
    base = 1./target[pixels]
    for label, z in (('before', sharp), ('after', mapped)):
        report['consistent_3pct_'+label] = int((abs(1./z.ravel()[pixels]-base) <= CONSISTENT*base).sum())
    return mapped, report


def aligned_path(name, frame):
    return OUT/'C1-alignment'/name/f'f{frame:03d}.npz'


@lru_cache(maxsize=48)
def aligned_input(name, frame):
    unit = formal_unit(name)
    if unit:
        return observation(unit, frame, aligned=True)
    row = data.b_input(name, frame)
    depth = row['depth'].copy()
    depth[0] = data.load(aligned_path(name, frame), ('sharp',))['sharp']
    return dict(row, depth=depth)


aligned_sample = partial(data.sample, b_loader=aligned_input)


def check_inputs(rows):
    """Audit raw observations, then the two dependent SHARP tensors."""
    accepted = {(r['scene'],r['frame']) for r in rows if r['accepted']}
    def participant_count(item):
        _, name, index = item
        positions = [index, *data.neighbors(index,len(data.TRAIN_FRAMES))]
        return sum((name,data.TRAIN_FRAMES[j]) in accepted for j in positions)
    eligible = [x for x in data.inventory('dev')['B'] if (x[1],data.TRAIN_FRAMES[x[2]]) in accepted]
    item = max(eligible,key=participant_count)
    _, name, index = item
    frames = [data.TRAIN_FRAMES[j] for j in [index, *data.neighbors(index, len(data.TRAIN_FRAMES))]]
    native = [data.b_input(name, frame) for frame in frames]
    aligned = [aligned_input(name, frame) for frame in frames]
    for before, after in zip(native, aligned):
        for key in before:
            np.testing.assert_array_equal(before[key][1:] if key == 'depth' else before[key],
                                          after[key][1:] if key == 'depth' else after[key], err_msg=key)
    control, y0 = data.sample(item, 'cpu')
    train_input, y1 = aligned_sample(item, 'cpu')
    restored = [dict(after, depth=np.concatenate((before['depth'][:1], after['depth'][1:])))
                for before, after in zip(native, aligned)]
    equivalent = data.input_record(restored[0], restored[1:])
    evaluation = data.input_record(aligned[0], aligned[1:])
    changed = {}
    for key in control:
        torch.testing.assert_close(control[key], torch.as_tensor(equivalent[key][None]), atol=0, rtol=0)
        torch.testing.assert_close(train_input[key], torch.as_tensor(evaluation[key][None]), atol=0, rtol=0)
        if key not in ('prior_depth', 'neighbor_depth'):
            torch.testing.assert_close(control[key], train_input[key], atol=0, rtol=0)
        changed[key] = int(torch.count_nonzero(control[key] != train_input[key]))
    for key in y0:
        torch.testing.assert_close(y0[key], y1[key], atol=0, rtol=0)
    if not changed['prior_depth'] or not changed['neighbor_depth']:
        raise ValueError('Alignment audit did not exercise both current and neighbor SHARP')
    return dict(passed=True, scene=name, frames=frames, changed_elements=changed,
                evaluation_entry_equal=True, invariant_when_SHARP_restored=True,
                raw_non_SHARP_observations_equal=True, untouched_target_labels_equal=True,
                derived_multiview_input='neighbor_depth rebuilds the existing neighbor fusion with its own mapped SHARP')


def prepare():
    start = time.monotonic()
    pool = training_pool()
    data_split = data.read(data.PREVIOUS/'split.json')
    rows = []
    for split, names in (('train',data_split['train']),('dev',data_split['heldout'])):
        for name in names:
            for frame in data.TRAIN_FRAMES:
                sharp, report = fit(name, frame, data.b_input(name, frame))
                save_npz(aligned_path(name, frame), sharp=sharp)
                rows.append(dict(report, split=split))
            print('1C_ALIGNMENT',name,Counter(r['fallback'] or 'accepted' for r in rows if r['scene']==name),flush=True)
    save_json(OUT/'C1.json',dict(rows=rows, fixed_check=check_inputs(rows), seed=SEED,
        untouched_tiers={t:len(pool[t]) for t in ('C','G','R')}, seconds=time.monotonic()-start,
        relative_error='S1b sparse_depth_AbsRel on heldout DepthSplat',
        alignment_supervision='jointly valid observed priors, no GT'))
    # Freeze the online input inventory for both arms.
    save_json(OUT/'C2-inventory.json', pool)


def paired_training():
    assert data.read(OUT/'C1.json')['fixed_check']['passed']
    pool = {tier:[tuple(row) for row in rows] for tier,rows in data.read(OUT/'C2-inventory.json').items()}
    for arm, sample in (('control',data.sample),('aligned',aligned_sample)):
        dest = OUT/'train'/arm
        dest.mkdir(parents=True,exist_ok=True)
        if (dest/'training.json').is_file():
            continue
        torch.cuda.reset_peak_memory_stats()
        train(output=dest, profile=dict(PROFILE, experiment='1c', arm=arm), sample=sample,
              monitor=partial(zero_monitor, sample=sample, method_focus=True),
              initialization=INITIAL, steps=5000, eval_steps=[1000,2000,3000,4000,5000], fixed_pool=pool)
    runs = {arm:data.read(OUT/'train'/arm/'training.json') for arm in ('control','aligned')}
    assert runs['control']['counts'] == runs['aligned']['counts']
    assert runs['control']['batch_sequence_sha256'] == runs['aligned']['batch_sequence_sha256']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare','train'))
    args = parser.parse_args()
    data.OUT = DATA_RUN
    if args.operation == 'prepare':
        prepare()
    else:
        gpu_setup()
        paired_training()
