"""Repack target-conditioned Query caches from the original source observations."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path
import time

import numpy as np

from route3_ovg_refocus import OUT, case_frames, guard, valid_depth
from route3_ovg_1b_common import read
from route3_ovg_1b_prime_evaluate import A, OLD, OLD_FIXED
from route3_s2a_data import query_input_512, query_memory, views_for
from route3_ovg_subpixel import save_json, save_npz
from src.dual_layer import depth_band


def method_band(depth, K, row):
    aperture=.05/(2*row['aperture'])
    return depth_band(np.where(valid_depth(depth),depth,1e4),K[0,0],row['实际对焦距离_m'],aperture)[0]


def original_views(source, manifest):
    """Verify immutable images, depths and cameras against the original Query context."""
    descriptor=dict(model=source/'models/SHARP',calibration=source/'models/calibration/SHARP',
                    confidence=source/'models/calibration')
    views=views_for(descriptor,manifest['frames'],512)
    with np.load(manifest['context']) as frozen:
        if not np.array_equal(frozen['frames'],[v['frame'] for v in views]):
            raise RuntimeError('Original Query context frame identities differ')
        images,depths,poses,K=(frozen[k] for k in ('images','depth','c2w','K'))
        for slot, obs in enumerate(views):
            if not all(np.array_equal(obs[k],expected,equal_nan=True) for k,expected in (
                    ('rgb_linear',images[slot]),('depth',depths[slot]),('c2w',poses[slot]),('K',K))):
                raise RuntimeError(f'Frozen source observation differs for frame {obs["frame"]}')
    return views


def cache_path(folder, method):
    return folder/'query'/f'{OLD if method==OLD_FIXED else method}.npz'


def coverage(row,band,old_pixels,old_candidate,new_candidate,new_input):
    flat=band.ravel()
    old_count=int((flat&old_candidate).sum())
    new_count=int((flat&new_candidate).sum())
    count=int(flat.sum())
    n_in_band=flat[new_input['query_pixel']]
    if int(n_in_band.sum())!=count:
        raise RuntimeError('Query cache does not index every pixel of the case band')
    return dict(row,band_pixels=count,
        old_cached_query_positions=int((flat&old_pixels).sum()),
        old_positions_with_candidate=old_count,
        new_positions_with_candidate=new_count,
        old_candidate_coverage=old_count/count if count else None,
        new_candidate_coverage=new_count/count if count else None,
        supplemented_positions=int((flat&new_candidate&~old_candidate).sum()),
        supplemented_outside_old_cached_positions=int((flat&new_candidate&~old_pixels).sum()),
        previously_covered_now_filtered=int((flat&old_candidate&~new_candidate).sum()),
        new_positions_without_candidate=count-new_count,
        valid_candidate_slots=int(new_input['candidate_valid'][n_in_band].sum()))


def prepare_scene(frames):
    name,_,_,_,source,_,manifest,_,_=frames[0]
    views=original_views(source,manifest)
    memory=query_memory(source,manifest['frames'])
    reports=[]
    for _,frame,_,_,_,_,_,case_entries,folder in frames:
        guard()
        path=folder/'R2.json'
        if path.exists():
            reports.extend(read(path)['rows'])
            continue
        start=time.monotonic()
        r1=read(folder/'R1.json')['rows']
        target=next(v for v in views if v['frame']==frame)
        K=target['K']
        old_file=Path(case_entries[0]['input'])
        with np.load(old_file) as orig:
            old_pixels=np.zeros(512**2,bool)
            old_candidate=np.zeros_like(old_pixels)
            old_pixels[orig['query_pixel']]=True
            old_candidate[orig['query_pixel'][orig['candidate_valid'].any(-1)]]=True
            old_case_control=None
            # Verify the factored builder against one unchanged 512/13 row.
            if '122105' in name and frame==30:
                rebuilt,_=query_input_512(views,dict(target,depth=orig['target_depth']),orig['band'],
                                          manifest['M'],manifest['query_cap'],memory)
                for key in orig.files:
                    np.testing.assert_array_equal(rebuilt[key],orig[key],err_msg=f'Original Query cache: {key}')
                old_case_control=dict(fields=len(orig.files),pixels=len(orig['query_pixel']),exact=True)
                del rebuilt
        rows=[]
        band_masks={}
        with np.load(folder/'visible-depths.npz') as maps:
            grouped=defaultdict(list)
            for row in r1:
                if not row['失败']:
                    group=OLD if row['method']==OLD_FIXED else row['method']
                    grouped[group].append(row)
            for method,method_rows in grouped.items():
                depth=maps[method]
                case_bands={}
                union=np.zeros((512,512),bool)
                for row in method_rows:
                    key=(row['protocol'],row['id'])
                    if key not in case_bands:
                        band=method_band(depth,K,row)
                        case_bands[key]=band
                        union|=band
                    band_masks['__'.join((row['method'],*key))]=case_bands[key]
                if method==A:
                    for row in method_rows:
                        rows.append(dict(row,band_pixels=int(case_bands[row['protocol'],row['id']].sum()),
                                         query_applicable=False))
                    continue
                target_for_method=dict(target,depth=np.where(valid_depth(depth),depth,1e4))
                inp,audit=query_input_512(views,target_for_method,union,manifest['M'],manifest['query_cap'],memory)
                new_candidate=np.zeros(512**2,bool)
                new_candidate[inp['query_pixel'][inp['candidate_valid'].any(-1)]]=True
                inp_file=cache_path(folder,method)
                save_npz(inp_file,**inp)
                for row in method_rows:
                    measured=coverage(row,case_bands[row['protocol'],row['id']],old_pixels,
                                      old_candidate,new_candidate,inp)
                    rows.append(dict(measured,query_applicable=True,cache_path=str(inp_file.relative_to(OUT)),
                                     cache_union_pixels=audit['full_band_queries']))
                del inp
        save_npz(folder/'R2-bands.npz',**band_masks)
        save_json(path,dict(rows=rows,old_cache_control=old_case_control,seconds=time.monotonic()-start,
                            original_context=manifest['context'],
                            source_observations_identical=True,weights='frozen original footprint Query'))
        reports.extend(rows)
        print('REFOCUS_R2',name,frame,len(rows),
              sum(r.get('supplemented_positions',0) for r in rows),flush=True)
    return reports


def prepare():
    start=time.monotonic()
    assert len(read(OUT/'R1.json')['fixed_check'])==3
    groups=[list(group) for _,group in groupby(case_frames(),key=lambda row:row[0])]
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        rows=[row for group in pool.map(prepare_scene,groups) for row in group]
    save_json(OUT/'R2.json',dict(rows=rows,seconds=time.monotonic()-start,
        coverage_definition='new method band pixels with at least one valid source candidate divided by band pixels',
        supplementation_definition='positions inside the new band without an old cached candidate that now have a candidate',
        on_new_depth='candidate backside test and packed target depth both use the evaluated method geometry',
        source_context='original 512/13 images, depths, poses and appearance observations',
        frozen_Query='a4dfb0e V1 footprint step 020000',
        training_or_hidden_labels_used_in_Query_cache=False))


if __name__=='__main__':
    prepare()
