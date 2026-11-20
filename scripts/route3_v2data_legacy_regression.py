"""Replay the archived 1c EMA geometry comparison on its original three scenes."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('XFORMERS_DISABLED', '1')

import cv2
import torch

import route3_ovg_data as data
from route3_ovg_1b_train import evaluate, load_model
from route3_ovg_1c import aligned_sample, DATA_RUN, OUT
from route3_ovg_subpixel import save_json


def regression():
    torch.set_num_threads(5)
    cv2.setNumThreads(1)
    data.OUT = DATA_RUN
    checkpoint = OUT/'train/aligned/checkpoints/step-005000.pt'
    reference = OUT/'train/aligned/validation/step-005000.json'
    old = data.read(reference)['EMA']
    items = data.inventory('dev')['B']
    if len(items) != 39 or len({x[1] for x in items}) != 3:
        raise ValueError('Archived comparison requires thirteen frames from each original scene')
    net = load_model(checkpoint,'ema',device='cpu')
    new = evaluate(net,5000,fixed=True,compare_gate_off=True,sample=aligned_sample,dev_items=items,device='cpu')
    key = ('scene','subset','method','focus_m')
    before, after = ({tuple(r[k] for k in key):r for r in table['rows']} for table in (old,new))
    if before.keys() != after.keys():
        raise ValueError('Archived comparison rows changed')
    tolerance = dict(front_inverse_mae=.0001, front_CoC_mae_px=.002,
                     boundary_front_inverse_mae=.0001,boundary_front_CoC_mae_px=.002)
    maxima = dict.fromkeys(tolerance,0.)
    by_scene = []
    for identity, original in before.items():
        rerun = after[identity]
        if any(original[k]!=rerun[k] for k in ('front_n','edge_front_n')):
            raise ValueError(f'Archived comparison support changed for {identity}')
        for field in tolerance:
            if original[field] is not None:
                maxima[field] = max(maxima[field],abs(original[field]-rerun[field]))
        if identity[1:]==('any_prior','OVG-Net',1.4):
            field = 'boundary_front_CoC_mae_px'
            by_scene.append(dict(scene=identity[0],metric=field,original=original[field],rerun=rerun[field],
                                 abs_difference=abs(original[field]-rerun[field])))
    return dict(checkpoint=str(checkpoint.relative_to(data.ROOT)),reference=str(reference.relative_to(data.ROOT)),
                device='cpu',passed=all(maxima[k]<=threshold for k,threshold in tolerance.items()),
                scenes=3,frames=len(items),matched_rows=len(before),max_absolute_differences=maxima,
                tolerances=tolerance,key_rows=by_scene)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    result = regression()
    save_json(args.output,result)
    print(result,flush=True)
    raise SystemExit(0 if result['passed'] else 1)
