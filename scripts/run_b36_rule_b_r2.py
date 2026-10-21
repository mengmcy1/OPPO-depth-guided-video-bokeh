"""One approved, manually assisted B36 tail-CoC control; reuse native worker.

The original inference function is imported unchanged. Preparation verifies the
reused A/R1 provenance and saves common input-defined ROIs BEFORE R2 inference.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F

import run_b36_rule_b as old
from check_atb_mpi_invariance import author_imports, digest, tensor_record, KEYS
import prepare_b36_rule_b as b
from analyze_b36_rule_b import bbox, rectangle, link

ROOT=b.ROOT
CFG=ROOT/'configs/any2bokeh-B36-rule-b-R2-20261021.json'
read,save=old.read,old.save


def bit_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def function_ast(source,name):
    return ast.dump(next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)


def prepare(c,out):
    approval=read(ROOT/c['approval_record'])
    assert approval['neck_reference_confirmed'] and approval['late_target_confirmed'] and approval['new_inference_authorized']
    assert not out.exists();out.mkdir(parents=True)
    priorc=read(ROOT/c['prior_config']);prior=ROOT/priorc['output'];info=read(prior/'prepared.json')
    prior_record=read(ROOT/(priorc['record_prefix']+'.json'))
    assert info['config']==priorc
    # Compare actual inference/loading functions, including the saved executed source.
    actual=Path(old.__file__).read_text();executed=(ROOT/prior_record['executed_script_copy']).read_text()
    assert function_ast(actual,'worker')==function_ast(executed,'worker')
    helper_path=ROOT/'scripts/check_atb_mpi_invariance.py'
    historic_helper=subprocess.check_output(['git','show','384c03e:scripts/check_atb_mpi_invariance.py'],cwd=ROOT,text=True)
    for name in ['load_model','author_imports']:
        assert function_ast(helper_path.read_text(),name)==function_ast(historic_helper,name)
    p,_,_,_,Dataset,collate=author_imports()
    historic_cfg=json.loads(subprocess.check_output(['git','show','384c03e:configs/davis-baselines.json'],cwd=ROOT,text=True))['any2bokeh']
    assert p==historic_cfg
    saved=torch.load(prior/'inputs.pt',weights_only=True,map_location='cpu')
    csvpath=ROOT/'outputs/any2bokeh-failure-analysis-20261009/B36-causal-diagnosis-20261015/inputs/native/input.csv'
    fresh=collate([Dataset(str(csvpath),sample_size=(576,1024))[0]])
    for key in KEYS:
        assert bit_equal(fresh[key],saved['base'][key]) and digest(fresh[key])==info['original'][key]['sha256']
    del fresh
    base=saved['base'];r1=saved['rule_b_coc'];r2=r1.clone();valid=saved['valid'];groups=info['groups']
    runs={v:dict(runtime=read(prior/v/'runtime.json'),events=read(prior/v/'events.json'),complete=read(prior/v/'complete.json')) for v in priorc['conditions']}
    for v,r in runs.items():
        expected={k:digest(r1 if k=='coc' and v=='rule_b' else base[k]) for k in KEYS}
        assert r['runtime']['input_sha256']==expected and r['runtime']['author_commit']==p['author_commit']
        assert r['runtime']['initial_latent_cache'] is None and r['runtime']['initial_encoder_cache_absent']
        assert [unit['output_frames'] for unit in r['events']['units']]==[8,8,9]
    rng_path=ROOT/runs['native']['runtime']['RNG_source'];state=torch.load(rng_path,weights_only=True)
    assert {k:digest(state[k]) for k in ['cpu','cuda']}==runs['native']['runtime']['entry_RNG']
    assert runs['native_repeat']['runtime']['entry_RNG']==runs['rule_b']['runtime']['entry_RNG']==runs['native']['runtime']['entry_RNG']
    assert prior_record['comparisons']['native_repeat']['different_float_bit_patterns']==0
    # Check checkpoint provenance without inventing a historical checksum.
    launched=datetime.fromisoformat(prior_record['launch']['UTC']).timestamp()
    checkpoint_files=[x for x in (ROOT/p['checkpoint']).rglob('*') if x.is_file()]
    assert checkpoint_files and all(x.stat().st_mtime<launched for x in checkpoint_files)
    provenance=dict(author_commit=p['author_commit'],author_checkout_clean=True,
        fresh_original_loader_inputs_bitwise_equal=True,worker_AST_matches_executed_R1=True,
        loading_functions_and_frozen_method_config_match_R1_commit=True,
        prior_runtimes_inputs_and_RNG_verified=True,scene_entry_RNG_sha256={k:digest(state[k]) for k in ['cpu','cuda']},
        checkpoints=dict(path=p['checkpoint'],files=len(checkpoint_files),all_mtime_before_A_R1_run=True,
            limitation='No historical full weight checksums exist in the prior record; reuse also relies on unchanged frozen paths/config and older file mtimes.'),
        prior_output=priorc['output'],prior_record=priorc['record_prefix']+'.json',
        prior_runtime_versions=runs['native']['runtime'],new_GPU_inference_conditions=1)
    tailcfg=read(ROOT/c['tail_config']);proposal=read(ROOT/c['tail_proposal']);tail=ROOT/tailcfg['output']
    selection=read(ROOT/priorc['selection_config']);selection_rows=read(ROOT/priorc['selection_record'])['rows']
    input_rows=[]
    for t in range(25):
        slots=[[g,j] for g,ids in enumerate(groups) for j,source in enumerate(ids) if source==t]
        q,u,focus=b.original_geometry(selection,t)
        if t>=21:
            row=next(x for x in proposal['frames'] if x['frame_id']==t)
            mask=np.asarray(Image.open(tail/row['target_mask']))!=0
            reference=b.reference(selection,dict(localization_valid=True,missing_reason='',native_xywh=row['reference_native_xywh']),q,u)[1]
            assert reference*255==row['reference_q'] and mask.sum()==row['pending_target_pixels'] and not mask[~u].any()
            assert mask.sum()/u.sum()<=.25
            qt=torch.from_numpy(q.astype(np.float32));newq=qt.clone();newq[torch.from_numpy(mask)]=row['reference_q']
            value=F.normalize(F.resize(((newq-focus*255).abs()/255).repeat(3,1,1),(576,1024)),[.5]*3,[.5]*3)
            original=F.normalize(F.resize(((qt-focus*255).abs()/255).repeat(3,1,1),(576,1024)),[.5]*3,[.5]*3)
            assert all(bit_equal(original,base['coc'][g,j]) for g,j in slots)
            for g,j in slots:r2[g,j]=value
            delta=value-original;projected=cv2.resize(u.astype(np.uint8),(1024,576),interpolation=cv2.INTER_NEAREST)>0
            data=dict(reference_q=row['reference_q'],reference_native_xywh=row['reference_native_xywh'],
                target_native_pixels=int(mask.sum()),target_fraction_subject=float(mask.sum()/u.sum()),
                condition_changed_elements=int(torch.count_nonzero(delta)),condition_max_abs_change=float(delta.abs().max()),
                native_background_changed_pixels=0,
                resized_condition_changed_pixels_outside_nearest_subject=int(((delta[0].numpy()!=0)&~projected).sum()),
                nominal_replacement_K_abs_d_minus_focus=16*abs(reference-focus))
        else:
            data=dict(reference_q=selection_rows[t]['reference_q'],target_native_pixels=selection_rows[t]['selected_area'],
                condition_unchanged_from_R1_bitwise=all(bit_equal(r2[g,j],r1[g,j]) for g,j in slots))
            assert data['condition_unchanged_from_R1_bitwise']
        assert all(bit_equal(r2[g,j],r2[slots[0][0],slots[0][1]]) for g,j in slots)
        input_rows.append(dict(frame=t,focus=focus,K=16,slots=slots,R2_vs_R1_condition_changed=t>=21,
            CoC_sha256=digest(r2[slots[0][0],slots[0][1]]),**data))
    assert bit_equal(r2[~valid],r1[~valid]) and bit_equal(r2[~valid],base['coc'][~valid])
    # Common supports are prepared without reading any new output.
    roi=read(prior/'regions.json');common=[];oldsource=ROOT/selection['output']/'B36'
    for t in range(25):
        q,u,focus=b.original_geometry(selection,t);small=lambda m:cv2.resize(m.astype(np.uint8),(1024,576),interpolation=cv2.INTER_NEAREST)>0
        previous={r['region']:np.asarray(Image.open(prior/r['path']))!=0 for r in roi['frames'][t]['regions'] if r['region'] in ['head_high_side','neck','body_unselected','head_context','water_near','water_far']}
        target=previous['head_high_side'] if t<=20 else small(np.asarray(Image.open(tail/next(x['target_mask'] for x in proposal['frames'] if x['frame_id']==t)))!=0)
        neck=previous['neck'] if t<=20 else None
        if t>=21:
            box=next(x['reference_native_xywh'] for x in proposal['frames'] if x['frame_id']==t)
            native_neck=np.zeros_like(u);x,y,w,h=box;native_neck[y:y+h,x:x+w]=True;neck=small(native_neck&u)
        sets=dict(target_head=target,neck=neck,body_non_target=previous['body_unselected']&~target,
            head_context=previous['head_context'],water_near=previous['water_near'],water_far=previous['water_far'],background_all=~small(u))
        folder=out/'regions'/f'{t:05d}';folder.mkdir(parents=True)
        entries=[]
        for name,mask in sets.items():
            path=folder/(name+'.png');Image.fromarray(np.uint8(mask)*255).save(path)
            entries.append(dict(region=name,path=str(path.relative_to(out)),pixels=int(mask.sum()),display_bbox=bbox(mask)))
        common.append(dict(frame_id=t,head_context_xywh=roi['frames'][t]['head_context_xywh'],regions=entries))
    # Worker sees precisely the same base dictionary and candidate slot layout.
    torch.save(dict(base=base,rule_b_coc=r2,valid=valid),out/'inputs.pt')
    save(out/'prepared.json',dict(config=c,approval=approval,created_UTC=datetime.now(timezone.utc).isoformat(),
        groups=groups,input_rows=input_rows,provenance=provenance,
        original={k:tensor_record(v) for k,v in base.items()},rule_b_coc=tensor_record(r2),
        source0_20_CoC_bitwise_equal_R1=True,padding_bitwise_equal_R1_and_A=True,
        worker_key_rule_b_means_R2=True))
    save(out/'regions.json',dict(policy=c['common_regions_before_output'],frames=common,prepared_before_R2=True))
    for name,source in [('native',prior/'native'),('native_repeat',prior/'native_repeat'),('revision1',prior/'rule_b')]:link(out/name,source)
    print('PREPARED_R2; common regions frozen; reused A/R1 provenance passed',flush=True)


def run(c,out):
    assert read(out/'prepared.json')['config']==c
    if (out/'rule_b/complete.json').exists():print('REUSE_COMPLETED_R2',flush=True);return
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy; leave other processes untouched'
    with (out/'R2-inference.log').open('x') as log:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('SINGLE_R2_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run','worker']);args=parser.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1);c=read(CFG);out=ROOT/c['output']
    if args.stage=='worker':old.worker(c,out,'rule_b')
    else:globals()[args.stage](c,out)
