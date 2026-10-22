"""Construct candidate geometry labels in fixed VDA coordinates. CPU only."""
import json
import sys
from pathlib import Path
import numpy as np
from prepare_ours_geometry_pilot import depth_read, write_json

ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'configs/ours-sintel-supervision-pilot-v1.json').read_text())
BASE=json.loads((ROOT/C['input_config']).read_text())


def main():
    evidence=json.loads((ROOT/'03_实验记录/records/ours-geometry-pilot-stereo-convention-20261022.json').read_text())
    assert len(evidence['rows'])==6
    assert all(r['axial_abs_px_p95']<.016 and r['axial_abs_px_median']<r['ray_abs_px_median'] for r in evidence['rows']), 'Resolve depth convention first'
    out=ROOT/C['output_root']
    assert not out.exists(), 'Preserve existing derived labels'
    rows=[]
    train_residual_sample=[]
    for s in BASE['sequences']:
        name=s['sequence'];raw=ROOT/BASE['raw_root']/'training'
        z=np.stack([depth_read(raw/'depth'/name/f'frame_{t:04d}.dpt') for t in s['frames']])
        with np.load(ROOT/C['vda_root']/name/'prediction.npz') as d:
            inp=d['disparity_normalized']
        low,high=C['label_domain_m']
        valid=np.isfinite(z)&(z>low)&(z<high)
        inv=np.zeros_like(z);np.divide(1,z,out=inv,where=valid)
        x=inv[valid].astype(np.float64);y=inp[valid].astype(np.float64)
        vx=np.mean((x-x.mean())**2);assert vx>1e-12
        a=float(np.mean((x-x.mean())*(y-y.mean()))/vx);b=float(y.mean()-a*x.mean());assert a>0
        target=(a*inv+b).astype(np.float32);target[~valid]=0
        assert np.isfinite(target).all()
        residual=(target-inp)[valid]
        dest=out/name;dest.mkdir(parents=True)
        np.savez_compressed(dest/'reference.npz',reference_normalized=target,supervision_domain=valid)
        per_frame=[]
        for i,t in enumerate(s['frames']):
            m=valid[i];e=(target[i]-inp[i])[m]
            per_frame.append(dict(frame=t,valid_pixels=int(m.sum()),mae=float(np.abs(e).mean()),
                                  target_min=float(target[i][m].min()),target_max=float(target[i][m].max())))
        row=dict(sequence=name,split=s['split'],frames=s['frames'],scale=a,shift=b,
                 valid_fraction=float(valid.mean()),absolute_residual_quantiles=dict(zip(['p50','p90','p95','p99'],map(float,np.quantile(np.abs(residual),[.5,.9,.95,.99])))),
                 target_outside_0_1_fraction=float(((target[valid]<0)|(target[valid]>1)).mean()),
                 frame_records=per_frame,reference=str((dest/'reference.npz').relative_to(ROOT)),
                 input=str((ROOT/C['vda_root']/name/'prediction.npz').relative_to(ROOT)),
                 rgb_directory=str((raw/'clean'/name).relative_to(ROOT)))
        rows.append(row)
        if s['split']=='pilot_train':
            # Deterministic diagnostic subsample, never used for fitting labels.
            train_residual_sample.append(residual[::97])
        with np.load(dest/'reference.npz') as d:
            assert np.array_equal(d['reference_normalized'],target) and np.array_equal(d['supervision_domain'],valid)
        print('PAIRED',name,'valid',round(row['valid_fraction'],4),'MAE',round(float(np.abs(residual).mean()),4),flush=True)
    samples=np.concatenate(train_residual_sample)
    result=dict(status='150_NATURAL_GEOMETRY_PAIRS_PREPARED_CANDIDATE_PROTOCOL',config=C,rows=rows,
                python=sys.version, numpy=np.__version__,
                train_only_signed_residual_quantiles=dict(zip(['p01','p05','p50','p95','p99'],map(float,np.quantile(samples,[.01,.05,.5,.95,.99])))),
                residual_sample_stride=97,train_frames=100,dev_frames=50,source_family_overlap=0,
                inference_runs_this_stage=0,student_training_runs=0,perturbed_pairs=0,
                next='Review reference mapping/out-of-range targets and border quality, then freeze perturbation and keep-loss rules; no training yet.')
    write_json(out/'manifest.json',result);write_json(ROOT/C['record'],result)


if __name__=='__main__': main()
