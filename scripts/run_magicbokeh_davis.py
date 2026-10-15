"""DAVIS-only target adapter around frozen MagicBokeh; reuse existing VEPI entry.

inventory is CPU-only and can precede approval. All rendering requires the explicit
approved configuration. Per-frame deterministic seeds allow completed frames and
the original eight preflight clips to be reused without changing a random stream.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from baselines.magicbokeh_focus import safe_target_region, medoid_point, control_diagnostics
from prepare_davis_v2 import write_new


def read(path): return json.loads((ROOT/path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()


def manifest(c):
    data=read(c['data_protocol']);source=read(str(Path(data['derived'])/'source-manifest.json'))
    names=sorted(s['id'] for s in source['sequences']);first=c['preflight_sequences']
    assert first==read(c['baseline_configuration'])['preflight_sequences'] and len(names)==50
    return data, first+[s for s in names if s not in first]


def inputs(data,scene,t):
    base=ROOT/data['derived']/'eval'/scene
    raw_rgb=cv2.imread(str(base/'rgb'/f'{t:05d}.jpg'));assert raw_rgb is not None
    labels=np.asarray(Image.open(ROOT/data['source']/'Annotations'/data['resolution_source']/scene/f'{t:05d}.png'))
    assert labels.dtype==np.uint8 and labels.shape==raw_rgb.shape[:2]
    prepared=np.asarray(Image.open(base/'mask'/f'{t:05d}.png'))
    assert np.array_equal(prepared,(labels!=0).astype(np.uint8)*255)
    display_rgb=cv2.cvtColor(cv2.resize(raw_rgb,(1024,576),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
    resized_labels=cv2.resize(labels,(1024,576),interpolation=cv2.INTER_NEAREST)
    alpha=(resized_labels!=0).astype(np.float32)
    region,distance,target=safe_target_region(alpha)
    if region is None:raise ValueError(f'focus_point_unavailable: {scene}/{t}; no substitute or omission authorized')
    return Image.fromarray(display_rgb),resized_labels,alpha,region,distance,target,labels


def inventory(c):
    data,names=manifest(c);out=ROOT/c['output'];dest=out/'target-inventory.json'
    if dest.exists():
        value=json.loads(dest.read_text());assert value['target_definition']==c['target'];print('REUSE_INVENTORY',flush=True);return value
    rows=[]
    for i,scene in enumerate(names):
        for t in range(25):
            image,labels,alpha,region,distance,target,original=inputs(data,scene,t)
            rows.append(dict(scene=scene,scene_index=i,frame=t,seed=42+25*i+t,native_size=[original.shape[1],original.shape[0]],display_size=list(image.size),
                native_label_counts={str(int(v)):int((original==v).sum()) for v in np.unique(original)},
                display_label_counts={str(int(v)):int((labels==v).sum()) for v in np.unique(labels)},
                selected_component_label_counts={str(int(v)):int(((labels==v)&region).sum()) for v in np.unique(labels) if v},target=target))
        print('TARGET_INVENTORY',scene,flush=True)
    value=dict(status='INPUT_INVENTORY_COMPLETE_NO_MODEL_OUTPUTS',created_utc=now(),target_definition=c['target'],scene_order=names,
        scenes=50,frames=1250,unavailable=[],rows=rows,scope='Read input annotation/RGB only; no output viewing, model or metric execution')
    write_new(dest,value);write_new(ROOT/'03_实验记录/records/magicbokeh-davis-target-inventory-20261015.json',value)
    return value


def require_approval(c):
    assert c['status']=='FROZEN_USER_APPROVED_DAVIS_UNION_TARGET_B', 'DAVIS target proposal awaits explicit user approval'
    assert c['authorization'] and c['K']==16 and c['prompt']=='high quality,'


def render(c,depth_model,adapter,scene,index,t,data):
    import torch
    image,labels,alpha,region,distance,target,original=inputs(data,scene,t)
    d=depth_model.infer(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))
    assert d.shape==(576,1024) and d.dtype==np.float32 and np.isfinite(d).all() and d.min()>=0 and d.max()<=1
    q=(d*255).astype(np.uint8);point,medoid=medoid_point(region,distance,q);x,y=point;focus=float(q[y,x]/255)
    assert labels[y,x]!=0 and region[y,x]
    prepared=adapter.prepare(image,Image.fromarray(q,mode='L'),focus,16)
    assert prepared['positions']==[(0,0),(0,256),(0,512),(64,0),(64,256),(64,512)]
    native_depth=adapter.module.tensor_transforms(Image.fromarray(q,mode='L')).unsqueeze(0).cuda()
    raw_condition=(native_depth-focus).abs()*.5;native_focused=raw_condition<=.1
    # Verify control tiles in the actual CUDA arithmetic, not by a CPU approximation.
    expected_controls,positions=adapter.module.extract_patches(raw_condition)
    assert positions==prepared['positions'] and torch.equal(prepared['controls'],expected_controls*2-1)
    assert torch.equal(prepared['focused'],(1-(expected_controls>.1).float())*2-1)
    diag=control_diagnostics(alpha,q,focus,16,native_focused=native_focused[0,0].cpu().numpy())
    own_focused=native_focused[0,0].cpu().numpy()
    by_id={str(int(v)):dict(pixels=int((labels==v).sum()),focused_fraction=float(own_focused[labels==v].mean())) for v in np.unique(labels) if v}
    seed=42+25*index+t;torch.manual_seed(seed)
    with torch.no_grad():result=adapter.forward_cuda(prepared,prompt=c['prompt'])
    assert result.shape==(1,3,576,1024) and torch.isfinite(result).all()
    output=adapter.to_pil(result,image);pixels=np.asarray(output)
    assert output.mode=='RGB' and output.size==(1024,576) and pixels.dtype==np.uint8 and np.isfinite(pixels).all()
    # A constant output is reported rather than silently re-rendered or excluded.
    audit=dict(scene=scene,scene_index=index,frame=t,K=16,seed=seed,actual_torch_initial_seed=torch.initial_seed(),
        target=target,click_xy=point,click_label_ID=int(labels[y,x]),inside_target=True,inside_selected_component=True,medoid=medoid,
        component_label_ID_counts={str(int(v)):int(((labels==v)&region).sum()) for v in np.unique(labels) if v},
        focus=focus,own_depth_at_click_PNG8=int(q[y,x]),focus_source='own DAdepth PNG8 at click; no VDA scalar',
        selected_focus_roundtrip_equal=focus==float(q[y,x]/255),**diag,per_ID_focus_coverage=by_id,
        native_source_size=[original.shape[1],original.shape[0]],output_size=list(output.size),output_mode=output.mode,
        raw_float_shape=list(result.shape),raw_float_range=[float(result.min()),float(result.max())],PNG_range=[int(pixels.min()),int(pixels.max())],
        constant_output=bool(pixels.min()==pixels.max()),finite=True,native_tiles=6,tile_positions=positions,
        condition_range=[float(raw_condition.min()),float(raw_condition.max())],condition_tiles_exactly_match_native=True,
        focused_mask_range=[int(native_focused.min()),int(native_focused.max())],native_depth_range=[float(d.min()),float(d.max())],
        target_mask_supplied_to_generator=False,temporal_propagation=False)
    return output,d,q,audit


def check(c,names,stage):
    require_approval(c);data,order=manifest(c);out=ROOT/c['output'];rows=[];constant=[]
    for scene in names:
        folder=out/'predictions'/scene;files=sorted(folder.glob('*.png'))
        assert [p.name for p in files]==[f'{t:05d}.png' for t in range(25)],scene
        done=json.loads((out/'audit'/scene/'complete.json').read_text());assert done['frames']==25
        ids=set();components=[]
        for t,path in enumerate(files):
            r=json.loads((out/'audit'/scene/f'{t:05d}.json').read_text());image,labels,alpha,region,distance,target,_=inputs(data,scene,t)
            d=np.load(out/'geometry'/scene/f'{t:05d}.npy');q=np.asarray(Image.open(out/'geometry'/scene/f'{t:05d}.png'))
            assert d.shape==q.shape==(576,1024) and np.isfinite(d).all() and np.array_equal(q,(d*255).astype(np.uint8))
            point,medoid=medoid_point(region,distance,q);x,y=point
            assert point==r['click_xy'] and labels[y,x]!=0 and region[y,x]
            assert r['target']==target and r['medoid']==medoid
            assert r['scene_index']==order.index(scene) and r['seed']==42+25*order.index(scene)+t==r['actual_torch_initial_seed']
            assert r['focus']==q[y,x]/255 and r['K']==16 and r['native_tiles']==6 and r['finite'] and r['condition_tiles_exactly_match_native']
            assert not r['target_mask_supplied_to_generator']
            with Image.open(path) as im:assert im.size==(1024,576) and im.mode=='RGB';im.verify()
            pixels=np.asarray(Image.open(path));assert [int(pixels.min()),int(pixels.max())]==r['PNG_range']
            assert np.isfinite(r['raw_float_range']).all()
            ids.add(r['click_label_ID']);components.append(r['target']['component_pixels'])
            if r['constant_output']:constant.append([scene,t])
        rows.append(dict(scene=scene,frames=25,selected_click_IDs=sorted(ids),component_pixel_range=[min(components),max(components)]))
    report=dict(status='PROGRAM_IMPLEMENTATION_CHECKS_PASSED',stage=stage,checked_utc=now(),scenes=len(names),frames=25*len(names),
        constant_outputs=constant,missing_frames=[],target_point_depth_control_seed_checks=True,PNG_RGB_size_order_finite_checks=True,
        full_native_pipeline_unchanged=True,no_perceptual_verdict=True,rows=rows)
    if constant:print('CONSTANT_NATIVE_OUTPUTS_RETAINED_REQUIRING_CAUSE_REVIEW',constant,flush=True)
    dest=out/f'{stage}-integrity.json'
    if not dest.exists():write_new(dest,report)
    print('IMPLEMENTATION_CHECKS_PASSED',stage,len(names)*25,flush=True);return report


def infer(c,stage):
    require_approval(c);data,order=manifest(c);out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    write_new(out/'protocol.json',dict(configuration=c,data_protocol=data,scene_order=order))
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    inventory(c)
    if stage=='full':check(c,order[:8],'preflight')
    from run_magicbokeh_track_a import load_models
    depth_model,adapter=load_models();import torch
    runtime=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,
        GPU=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability()),model_loader='run_magicbokeh_track_a.load_models unchanged',
        author_commit=subprocess.check_output(['git','-C',str(ROOT.parent/'MagicBokeh'),'rev-parse','HEAD'],text=True).strip(),
        initial_seed=42,precision='unchanged DAdepth native FP32 and generator FP16',
        generator_training_flags={k:module.training for k,module in adapter.model.named_children()})
    assert runtime['author_commit'].startswith(read(c['synthetic_frozen_reference'])['author_commit'])
    if not (out/'runtime.json').exists():write_new(out/'runtime.json',runtime)
    for index,scene in enumerate(order if stage=='full' else order[:8]):
        done=out/'audit'/scene/'complete.json'
        if done.exists():print('REUSE_SCENE',scene,flush=True);continue
        start=time.monotonic()
        for t in range(25):
            pred=out/'predictions'/scene/f'{t:05d}.png';record=out/'audit'/scene/f'{t:05d}.json';native=out/'geometry'/scene/f'{t:05d}.npy';qfile=native.with_suffix('.png')
            if record.exists():assert pred.exists() and native.exists() and qfile.exists();continue
            if any(p.exists() for p in [pred,native,qfile]):raise FileExistsError(f'Incomplete saved frame {scene}/{t}; preserve and inspect')
            result,d,q,row=render(c,depth_model,adapter,scene,index,t,data)
            for p in [pred,record,native]:p.parent.mkdir(parents=True,exist_ok=True)
            result.save(pred);np.save(native,d);Image.fromarray(q,mode='L').save(qfile);write_new(record,row)
            print('FRAME_DONE',scene,t,'seed',row['seed'],'click',row['click_xy'],'focus',row['focus'],flush=True)
        write_new(done,dict(scene=scene,index=index,frames=25,completed_utc=now(),seconds=time.monotonic()-start,finite=True))
        print('SCENE_DONE',scene,index,flush=True)
    return check(c,order if stage=='full' else order[:8],stage)


def summarize(c):
    data,order=manifest(c);out=ROOT/c['output'];integrity=read(str(out.relative_to(ROOT)/'full-integrity.json'));scores=read(str(out.relative_to(ROOT)/'evaluation/summary.json'))
    assert integrity['frames']==scores['frames']==1250 and scores['videos']==50
    assert {r['id'] for r in scores['rows']}==set(order) and all(r['frames']==25 for r in scores['rows'])
    old=read('03_实验记录/records/davis-six-method-50-20260929.json')
    comparison={m:{key:r[key] for key in ['videos','frames','VEPI_public_mp4','VEPI_png','codec_delta']} for m,r in old['methods'].items()}
    comparison['magicbokeh']={key:scores[key] for key in ['videos','frames','VEPI_public_mp4','VEPI_png','codec_delta']}
    if (ROOT/c['record']).exists():
        previous=read(c['record']);assert previous['configuration']==c and previous['VEPI_public_mp4']==scores['VEPI_public_mp4']
        print('REUSE_COMPLETE_RESULT',flush=True);return
    result=dict(scores)
    result.update(completed_utc=now(),
        configuration=c,runtime=read(str(out.relative_to(ROOT)/'runtime.json')),preflight_reused_videos=8,new_remaining_videos=42,
        integrity=integrity,comparison=comparison,
        limits=['Annotation-guided own-depth single-click baseline; not automatic autofocus or shared VDA geometry',
                'Union plus largest component can select one of several instances or switch components; coverage is recorded, not tuned',
                'VEPI is foreground-edge preservation, not overall bokeh or temporal quality; no new visual judgments',
                'Full50 quantitative supplement does not authorize held-out failure discovery or tuning'])
    result['status']='MAGICBOKEH_DAVIS_50_QUANTITATIVE_COMPLETE_USER_VISUAL_REVIEW_NOT_PERFORMED'
    write_new(ROOT/c['record'],result);print('MAGICBOKEH_DAVIS_RESULT',scores['VEPI_public_mp4'],scores['VEPI_png'],scores['codec_delta'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['inventory','preflight','full','all','check','summarize'])
    parser.add_argument('--config',default='configs/magicbokeh-davis-frozen.json');a=parser.parse_args();c=read(a.config);cv2.setNumThreads(2)
    if a.stage=='inventory':inventory(c);return
    require_approval(c)
    if a.stage=='summarize':summarize(c);return
    if a.stage=='check':data,names=manifest(c);check(c,names,'full');return
    if a.stage=='all':
        infer(c,'preflight');infer(c,'full')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        subprocess.run([sys.executable,'scripts/evaluate_davis_vepi.py','--config',c['data_protocol'],'--predictions',str(Path(c['output'])/'predictions'),
            '--output',str(Path(c['output'])/'evaluation')],cwd=ROOT,env=env,check=True)
        summarize(c)
    else:infer(c,a.stage)


if __name__=='__main__':main()
