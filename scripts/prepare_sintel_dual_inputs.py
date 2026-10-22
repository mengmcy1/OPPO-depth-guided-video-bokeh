"""CPU-only Sintel geometry pilot: freeze train-only rules, build/check pairs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.synthetic.geometry_correction_data import (bounded_affine, map_reference, perturb,
    frame_signs, GeometryPairs, add_residual)
from prepare_ours_geometry_pilot import depth_read, write_json

BASE_PATH=ROOT/'configs/ours-geometry-data-pilot-20261022.json'
BASE=json.loads(BASE_PATH.read_text())
CONFIG=ROOT/'configs/ours-sintel-dual-input-v2.json'
PREFIX=ROOT/'03_实验记录/records/ours-sintel-dual-input-v2-20261022'
OUT=ROOT/'data/processed/ours-geometry-pilot-20261022/dual-input-v2'
VDA=ROOT/BASE['derived_root']/'vda-native'


def load_sequence(s):
    raw=ROOT/BASE['raw_root']/'training/depth'/s['sequence']
    z=np.stack([depth_read(raw/f'frame_{t:04d}.dpt') for t in s['frames']])
    valid=np.isfinite(z)&(z>.001)&(z<70)
    inv=np.zeros_like(z);np.divide(1,z,out=inv,where=valid)
    with np.load(VDA/s['sequence']/'prediction.npz') as f:d=f['disparity_normalized']
    assert d.shape==z.shape==(25,436,1024)
    return z,inv,valid,d


def compare_one(inv,valid,d):
    free,bounded=bounded_affine(inv[valid],d[valid])
    a=map_reference(inv,valid,free);b=map_reference(inv,valid,bounded)
    row=dict(free=free,bounded=bounded,
             free_outside=float(((a[valid]<0)|(a[valid]>1)).mean()),bounded_outside=float(((b[valid]<0)|(b[valid]>1)).mean()),
             free_mse=float(np.mean((a[valid]-d[valid])**2)),bounded_mse=float(np.mean((b[valid]-d[valid])**2)),
             label_change_mean=float(np.mean(b[valid]-a[valid])),label_change_mae=float(np.abs(b[valid]-a[valid]).mean()),
             scale_ratio=bounded['scale']/free['scale'])
    return row,b


def freeze():
    assert not CONFIG.exists(), 'Do not revise frozen rules after dev inspection'
    rows=[];samples=[]
    for s in BASE['sequences']:
        if s['split']!='pilot_train':continue
        _,inv,valid,d=load_sequence(s)
        row,reference=compare_one(inv,valid,d)
        row['sequence']=s['sequence'];rows.append(row)
        samples.append(np.abs(reference-d)[valid][::97])
    q25,q50,q90=map(float,np.quantile(np.concatenate(samples),[.25,.5,.9]))
    assert q25>0 and q90>q50>0
    comparison=dict(rows=rows,train_residual_quantiles=dict(p25=q25,p50=q50,p90=q90),residual_sample_stride=97,
                    selection_uses='four pilot_train families only; no dev pixels',
                    decision='bounded positive sequence-level affine: compatible 0..1 support without per-pixel reference clipping',
                    cost='Global contrast and fitting-error changes retained; no claim bounded mapping is physically more accurate.')
    write_json(Path(str(PREFIX)+'-train-selection.json'),comparison)
    config=dict(version=2,stage='data protocol pilot, not model training',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source_config=str(BASE_PATH.relative_to(ROOT)),output_root=str(OUT.relative_to(ROOT)),
        mapping='d_ref=a/Z+b; constrained sequence OLS, 0<=min target<=max target<=1, positive slope; preserve raw float VDA',
        mapping_support='all finite pixels with 0.001<Z<70 across 25 frames; each clip independently fits labels only',
        selection_record=str(Path(str(PREFIX)+'-train-selection.json').relative_to(ROOT)),
        input='RGB float32 0..1 (3 channels) + actual normalized float VDA input (1 channel)',
        model_output='one unconstrained signed residual; preclamp output = actual input + residual',
        output_range='geometry loss on preclamp float output; deployment geometry clips 0..1 and reports saturation, never renormalizes',
        atb_interface='provisional: clip float disparity to [0,1], quantize once floor(255*d); original fixed focus/K; CoC transplant only; NOT executed or claimed verified',
        geometry_validity='range/finite domain, not official confidence; invalid labels zero-filled only for storage and excluded from all losses',
        boundary_diagnostic='3x3 inverse-depth relative range >0.05; dilate one pixel (5x5 total support). Include in geometry loss; report separately; not semantic annotation.',
        keep_rule='S=valid & distance>=2 pixels from invalid/image border & outside diagnostic boundary & abs(original_d-reference)<=epsilon',
        keep_epsilon=q25,keep_meaning='train residual p25 near-reference subset, not calibrated reliable/correct region',
        keep_loss='mean_S relu(abs(preclamp_output - actual_input) - epsilon); perturbed branch removes perturbation footprint from S; reference lies in zero-penalty band',
        geometry_loss='mean_valid abs(preclamp_output-reference); weighting between losses still unset',
        perturbation=dict(seed=20261022,amplitude_range=[q50,q90],amplitude_basis='train only absolute residual p50..p90, stride97',
             area_fraction_range=[.01,.05],axis_ratio_range=[.5,2.],flat_radius=.75,max_attempts=64,
             shape='axis-aligned ellipse; raised-cosine taper in outer quarter of radius',
             fractions_per_clip=dict(zero=.2,positive=.4,negative=.4),
             placement='random geometry-independent location; entire footprint must lie in supervision domain; failed placement becomes documented no-op',
             clipping='clip d+signed_amplitude*weight to [0,1]; report clipping and actual delta; no renormalization',
             temporal='independent per-frame samples for framewise Student; no temporal-coherence claim'),
        budget='same total optimizer steps, batch size and base-frame schedule for natural-only/perturbed-only/joint; joint 50/50 within same budget; perturb branch includes 20% no-op',
        split_policy='same six reserved source families; no new split or DAVIS/heldout access',
        focus_K=None,teacher=False,training=False,limitations=['range validity does not resolve transparency/anti-aliasing GT errors','fitting absorbs scale/shift; not independent metric-depth accuracy','pilot is 100 train/50 dev frames, not final corpus'])
    write_json(CONFIG,config)
    print('FROZEN before dev processing',config['frozen_at_utc'],flush=True)


def regions(inv,valid,d,ref,epsilon):
    core=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    high=cv2.dilate(inv,np.ones((3,3),np.uint8))
    low=cv2.erode(np.where(valid,inv,1e30),np.ones((3,3),np.uint8))
    edge=valid & ((high-low)/np.maximum(high,1e-8)>.05)
    edge=cv2.dilate(edge.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&valid
    near=core & ~edge & (np.abs(d-ref)<=epsilon)
    return edge,near


def preview(s,c,d,ref,altered,masks,valid):
    sheet=Image.new('RGB',(6*256,3*143),'white');draw=ImageDraw.Draw(sheet)
    titles=['RGB','VDA [0,1]','perturbed [0,1]','reference [0,1]','perturb footprint','supervision domain']
    raw=ROOT/BASE['raw_root']/'training/clean'/s['sequence']
    for row,i in enumerate([0,12,24]):
        with Image.open(raw/f"frame_{s['frames'][i]:04d}.png") as im:rgb=im.copy()
        arrays=[rgb,Image.fromarray(np.uint8(d[i]*255)).convert('RGB'),Image.fromarray(np.uint8(altered[i]*255)).convert('RGB'),
                Image.fromarray(np.uint8(ref[i]*255)).convert('RGB'),Image.fromarray(masks[i].astype(np.uint8)*255).convert('RGB'),Image.fromarray(valid[i].astype(np.uint8)*255).convert('RGB')]
        for col,(title,im) in enumerate(zip(titles,arrays)):
            draw.text((col*256+2,row*143+3),f"f{s['frames'][i]:04d} {title}",fill='black')
            sheet.paste(im.resize((256,109),resample=Image.Resampling.NEAREST),(col*256,row*143+26))
    sheet.save(OUT/s['sequence']/'contact.png')


def build():
    c=json.loads(CONFIG.read_text());assert not OUT.exists(), 'Preserve previous generated data'
    frozen_sha=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    OUT.mkdir(parents=True);rows=[];frames=[]
    train_cmp=json.loads(Path(str(PREFIX)+'-train-selection.json').read_text())
    for s in BASE['sequences']:
        name=s['sequence'];_,inv,valid,d=load_sequence(s)
        comparison,ref=compare_one(inv,valid,d)
        assert np.all((ref[valid]>=0)&(ref[valid]<=1))
        if s['split']=='pilot_train':
            prior=next(x for x in train_cmp['rows'] if x['sequence']==name)
            assert comparison['bounded']==prior['bounded'], 'Training fit changed after freeze'
        signs=frame_signs(c['perturbation']['seed'],name,s['frames'])
        altered=[];masks=[];weights=[];edges=[];near=[];frame_stats=[]
        for i,t in enumerate(s['frames']):
            changed,mask,weight,info=perturb(d[i],valid[i],name,t,signs[t],c['perturbation'])
            edge,normal=regions(inv[i],valid[i],d[i],ref[i],c['keep_epsilon'])
            altered.append(changed);masks.append(mask);weights.append(weight);edges.append(edge);near.append(normal)
            delta=np.abs(changed-d[i])
            stats=dict(frame=t,valid_pixels=int(valid[i].sum()),footprint_pixels=int(mask.sum()),footprint_fraction=float(mask.mean()),
                       signed_delta_mean=float((changed-d[i])[mask].mean()) if mask.any() else 0.,
                       mean_abs_delta_in_footprint=float(delta[mask].mean()) if mask.any() else 0.,
                       saturation_fraction_in_footprint=info.get('clipped_pixels',0)/max(int(mask.sum()),1),
                       near_reference_pixels=int(normal.sum()),boundary_pixels=int(edge.sum()),**info)
            before=np.abs(d[i]-ref[i]);after=np.abs(changed-ref[i])
            stats.update(reference_mae_before_in_footprint=float(before[mask].mean()) if mask.any() else None,
                         reference_mae_after_in_footprint=float(after[mask].mean()) if mask.any() else None,
                         error_increased_pixels=int((mask & (after>before)).sum()),
                         error_decreased_pixels=int((mask & (after<before)).sum()))
            frame_stats.append(stats)
            frames.append(dict(sequence=name,source_id=s['source_family'],split=s['split'],frame=t,array_index=i,
                               rgb=str(Path(BASE['raw_root'])/'training/clean'/name/f'frame_{t:04d}.png'),
                               vda_npz=str((VDA/name/'prediction.npz').relative_to(ROOT)),
                               pair_npz=str((OUT/name/'pairs.npz').relative_to(ROOT)),statistics=stats))
        altered=np.stack(altered);masks=np.stack(masks);weights=np.stack(weights);edges=np.stack(edges);near=np.stack(near)
        dest=OUT/name;dest.mkdir()
        np.savez_compressed(dest/'pairs.npz',perturbed=altered,reference=ref,supervision_domain=valid,
                            perturbation_mask=masks,perturbation_weight=weights,near_reference=near,boundary_diagnostic=edges)
        assert hashlib.sha256(CONFIG.read_bytes()).hexdigest()==frozen_sha
        row=dict(sequence=name,split=s['split'],frames=25,valid_fraction=float(valid.mean()),
                 scale=comparison['bounded']['scale'],shift=comparison['bounded']['shift'],
                 target_min=float(ref[valid].min()),target_max=float(ref[valid].max()),target_outside=0,
                 mean_footprint_fraction=float(masks.mean()),zero_frames=sum(x['status']!='applied' for x in frame_stats),
                 positive_frames=sum(x['status']=='applied' and x['requested_sign']>0 for x in frame_stats),
                 negative_frames=sum(x['status']=='applied' and x['requested_sign']<0 for x in frame_stats),
                 placement_failures=sum(x['status']=='no_valid_ellipse' for x in frame_stats),
                 clipped_pixel_fraction_in_footprint=sum(x.get('clipped_pixels',0) for x in frame_stats)/max(int(masks.sum()),1),
                 changed_pixels=int((altered!=d).sum()),near_reference_fraction=float(near.mean()),
                 requested_amplitude_min=min(x['amplitude'] for x in frame_stats if x['status']=='applied'),
                 requested_amplitude_max=max(x['amplitude'] for x in frame_stats if x['status']=='applied'),
                 actual_mean_abs_delta_in_footprint=float(np.abs(altered-d)[masks].mean()),
                 clipped_pixel_fraction_in_image=sum(x.get('clipped_pixels',0) for x in frame_stats)/masks.size,
                 error_increased_fraction_in_footprint=sum(x['error_increased_pixels'] for x in frame_stats)/max(int(masks.sum()),1),
                 error_decreased_fraction_in_footprint=sum(x['error_decreased_pixels'] for x in frame_stats)/max(int(masks.sum()),1),
                 mapping_comparison=comparison)
        rows.append(row);preview(s,c,d,ref,altered,masks,valid)
        print('BUILT',name,'signs',row['zero_frames'],row['positive_frames'],row['negative_frames'],'clipped',round(row['clipped_pixel_fraction_in_footprint'],5),flush=True)
    manifest=dict(config=c,config_sha256=frozen_sha,frames=frames,summary=rows,numpy=np.__version__,opencv=cv2.__version__,python=sys.version)
    write_json(OUT/'manifest.json',manifest)
    write_json(Path(str(PREFIX)+'-summary.json'),{k:v for k,v in manifest.items() if k!='frames'})
    write_json(Path(str(PREFIX)+'-frames.json'),frames)
    fields=[k for k in rows[0] if k!='mapping_comparison']
    with Path(str(PREFIX)+'-summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)


def solver_checks():
    # Known exact affine, negative lower endpoint and sign-reversed failure cases.
    x=np.linspace(.1,2,503);y=.3*x+.1
    _,p=bounded_affine(x,y);assert np.allclose([p['scale'],p['shift']],[.3,.1],atol=1e-10)
    y=np.clip(.8*x-.4,0,1)
    _,p=bounded_affine(x,y)
    v=np.array([p['endpoint_low'],p['endpoint_high']]);t=(x-x.min())/np.ptp(x)
    score=np.mean((v[0]*(1-t)+v[1]*t-y)**2)
    for lo in np.linspace(0,1,101):
        for hi in np.linspace(lo,1,101):
            assert score <= np.mean((lo*(1-t)+hi*t-y)**2)+1e-12
    try:bounded_affine(x,-x)
    except AssertionError:pass
    else:raise AssertionError('Negative slope accepted')


def check():
    solver_checks();c=json.loads(CONFIG.read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest()==manifest['config_sha256']
    families={k:{r['source_id'] for r in manifest['frames'] if r['split']==k} for k in ['pilot_train','pilot_dev']}
    assert families['pilot_train'].isdisjoint(families['pilot_dev'])
    checked=0;no_ops=0;changed_frames=0;residual_witnesses=0
    for split,expected in [('pilot_train',100),('pilot_dev',50)]:
        ds=GeometryPairs(ROOT,str((OUT/'manifest.json').relative_to(ROOT)),split);assert len(ds)==expected
        for i in range(len(ds)):
            a,b,row=ds.pair(i)
            assert a['input'].shape==b['input'].shape==(4,436,1024)
            assert a['input'].dtype==b['input'].dtype==a['target'].dtype==np.float32
            assert a['target'] is b['target'] and a['valid'] is b['valid']
            assert np.isfinite(a['input']).all() and np.isfinite(b['input']).all() and np.isfinite(a['target']).all()
            assert np.array_equal(a['input'][:3],b['input'][:3])
            d=a['input'][3];changed,mask,weight,info=perturb(d,a['valid'],row['sequence'],row['frame'],row['statistics']['requested_sign'],c['perturbation'])
            assert np.array_equal(changed,b['input'][3]) and info=={k:row['statistics'][k] for k in info}
            j=row['array_index']
            assert np.array_equal(mask,ds._paired['perturbation_mask'][j]) and np.array_equal(weight,ds._paired['perturbation_weight'][j])
            assert np.array_equal(d[~mask],changed[~mask]) and not mask[~a['valid']].any()
            assert 0<=changed.min()<=changed.max()<=1
            assert (a['target'][a['valid']]>=0).all() and (a['target'][a['valid']]<=1).all()
            assert np.array_equal(add_residual(changed,np.zeros_like(changed)),changed)
            residual=a['target']-changed
            assert np.allclose(add_residual(changed,residual)[a['valid']],a['target'][a['valid']],atol=1e-7,rtol=0)
            if np.any(changed!=d):
                changed_frames+=1
                assert not np.allclose(add_residual(d,residual)[a['valid']],a['target'][a['valid']],atol=1e-7,rtol=0)
                residual_witnesses+=1
            else:no_ops+=1
            # At the GT target, keep dead zone cannot fight geometry supervision.
            for branch in (a,b):
                penalty=np.maximum(np.abs(branch['target']-branch['input'][3])-c['keep_epsilon'],0)
                assert not (penalty[branch['keep']]>1e-7).any()
            checked+=1
    result=dict(status='CPU_PAIR_AND_LOSS_COMPATIBILITY_CHECKS_PASSED',frames=checked,branch_samples=2*checked,
                train_frames=100,dev_frames=50,source_family_overlap=0,no_op_frames=no_ops,perturbed_frames=changed_frames,
                exact_replay=True,identical_GT_per_pair=True,unchanged_RGB=True,unchanged_outside_perturbation=True,
                invalid_region_perturbation_pixels=0,residual_actual_input_witnesses=residual_witnesses,
                target_has_zero_keep_penalty=True,bounded_solver_analytic_and_grid_checks=True,
                model_inference=0,student_training=0,config_sha256=manifest['config_sha256'])
    write_json(Path(str(PREFIX)+'-checks.json'),result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','build','check']);a=p.parse_args()
    globals()[a.phase]()
