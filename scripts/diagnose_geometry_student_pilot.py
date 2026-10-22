"""Post-hoc CPU diagnostics; no additional fitting on dev or training retries."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'configs/ours-student-pilot-v1-20261023.json').read_text())
M=json.loads((ROOT/C['data_manifest']).read_text());OUT=ROOT/C['output']


def main():
    train=[]
    for s in M['summary']:
        if s['split']!='pilot_train':continue
        row=next(r for r in M['frames'] if r['sequence']==s['sequence'])
        with np.load(ROOT/row['vda_npz']) as z:d=z['disparity_normalized']
        with np.load(ROOT/row['pair_npz']) as z:ref=z['reference'];valid=z['supervision_domain']
        train.append((ref-d)[valid])
    bias=float(np.median(np.concatenate(train)));rows=[];keep=[]
    for name in ['cave_2','market_2']:
        row=next(r for r in M['frames'] if r['sequence']==name)
        with np.load(ROOT/row['vda_npz']) as z:d=z['disparity_normalized']
        with np.load(ROOT/row['pair_npz']) as z:ref=z['reference'];valid=z['supervision_domain'];near=z['near_reference']
        constant=np.clip(d+bias,0,1)
        for variant in C['variants']:
            pred=np.load(OUT/variant/'predictions'/name/'natural.npz')['prediction']
            change=(pred-d)[valid]
            rows.append(dict(scene=name,variant=variant,model_change_mean=float(change.mean(dtype=np.float64)),model_change_std=float(change.std(dtype=np.float64)),
                model_change_quantiles=list(map(float,np.quantile(change,[.01,.1,.5,.9,.99]))),
                model_mae=float(np.abs(pred-ref)[valid].mean(dtype=np.float64)),constant_bias_mae=float(np.abs(constant-ref)[valid].mean(dtype=np.float64)),
                model_constant_difference_MAE=float(np.abs(pred-constant)[valid].mean(dtype=np.float64)),
                model_near_MAE=float(np.abs(pred-ref)[near].mean(dtype=np.float64)),constant_near_MAE=float(np.abs(constant-ref)[near].mean(dtype=np.float64))))
            eps=M['config']['keep_epsilon'];baseline=np.abs(d-ref);corrected=np.abs(pred-ref);delta=np.abs(pred-d)
            harm=near&(corrected>baseline)
            keep.append(dict(scene=name,variant=variant,near_pixels=int(near.sum()),harm_fraction=float(harm[near].mean()),
                change_within_deadzone_fraction=float((delta<=eps)[near].mean()),
                harm_inside_deadzone_fraction_of_near=float((harm&(delta<=eps))[near].mean()),
                clipped_output_keep_penalty_mean=float(np.maximum(delta-eps,0)[near].mean(dtype=np.float64)),
                interpretation='clipped-output diagnostic of the frozen near-reference mask; not a counterfactual loss ablation or sole-cause proof'))
    result=dict(scope='post-hoc CPU control; median residual fitted on training pixels only, no dev tuning or new training',train_bias=bias,rows=rows,keep_diagnostics=keep)
    path=ROOT/(C['record_prefix']+'-bias-diagnostic.json')
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(train_bias=bias,rows=rows,keep_diagnostics=keep),indent=2))


if __name__=='__main__':main()
