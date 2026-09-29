"""DAVIS native inputs with frozen baseline adapters and resumable scene prefixes."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text())==value,str(path)
    else:
        with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def save_rgb(path,value):
    assert value.shape==(576,1024,3) and np.isfinite(value).all()
    q=np.uint8(np.round(np.clip(value,0,1)*255))
    assert q.max()>q.min(), 'Degenerate constant/black frame: preserve and investigate'
    if path.exists():assert np.array_equal(np.asarray(Image.open(path)),q),'Interrupted output differs; preserve and investigate'
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:Image.fromarray(q).save(f,format='PNG')
    return dict(shape=list(q.shape),raw_range=[float(value.min()),float(value.max())],
        clip_fraction=float(((value<0)|(value>1)).mean()),PNG_range=[int(q.min()),int(q.max())])


def inputs(c,name):
    root=ROOT/c['derived'];folder=root/'eval'/name
    rgb=sorted((folder/'rgb').glob('*.jpg'));disp=sorted((folder/'disparity').glob('*.png'))
    metadata=json.loads((root/'sequences'/name/'geometry.json').read_text())
    assert len(rgb)==len(disp)==25
    for t,(r,d,m) in enumerate(zip(rgb,disp,metadata['frames'][:25])):
        assert r.stem==f'{t:05d}' and d.name.startswith(f'{t:05d}_zf_') and m['frame']==t
        assert d.name==m['file'] and float(d.stem.split('_zf_')[1])==m['focus']
    return folder,rgb,disp,metadata


def save_rng(path,next_index):
    state=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0),next_scene_index=next_index)
    if path.exists():
        prior=torch.load(path,weights_only=True)
        assert prior['next_scene_index']==next_index and torch.equal(prior['cpu'],state['cpu']) and torch.equal(prior['cuda'],state['cuda'])
    else:
        with path.open('xb') as f:torch.save(state,f)


def restore_prefix(out,order):
    prefix=0
    for name in order:
        if not (out/'audit'/name/'complete.json').exists():break
        assert len(list((out/'predictions'/name).glob('*.png')))==25
        prefix+=1
    assert not any((out/'audit'/n/'complete.json').exists() for n in order[prefix:])
    if prefix:
        state=torch.load(out/'audit'/order[prefix-1]/'rng-after.pt',weights_only=True)
        assert state['next_scene_index']==prefix
        torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'],device=0)
    return prefix


def audit_atb_batch(ds,batch,folder,meta,pipeline_class):
    import torchvision.transforms.functional as F
    from utils.dataset import gen_mpi_mask
    files=sorted(p.name for p in (folder/'disparity').glob('*.png'))
    groups,_=ds.group_frames_with_overlap(files,lambda p,o:.5*(1+math.cos(math.pi*p/o)))
    ids=[[int(Path(n).stem.split('_')[0]) for n in g] for g in groups]
    assert ids==[list(range(s,min(s+8,25))) for s in [0,4,8,12,16,20]]
    errors=dict(RGB=0.,CoC=0.,MPI=0.);expected=[]
    for t,n in enumerate(files):
        q=torch.from_numpy(np.asarray(Image.open(folder/'disparity'/n)).astype(np.float32))
        focus=float(Path(n).stem.split('_zf_')[1]);assert focus==meta['frames'][t]['focus']
        rgb=torch.from_numpy(np.asarray(Image.open(folder/'rgb'/f'{t:05d}.jpg').convert('RGB')).copy()).permute(2,0,1).float()/255
        coc=(q-focus*255).abs()/255
        mpi=gen_mpi_mask(q/255,focus,alpha=1.5)
        expected.append((ds.pixel_transforms(rgb),ds.pixel_transforms(coc.repeat(3,1,1)),F.resize(mpi[None],(576,1024))[0]))
    weight_sums=np.zeros(25)
    for g,group in enumerate(ids):
        for j,t in enumerate(group):
            for key,actual,exp in [('RGB',batch['pixel_values'][g,j],expected[t][0]),('CoC',batch['coc'][g,j],expected[t][1]),('MPI',batch['mpi_mask'][g,j],expected[t][2])]:
                err=float((actual-exp).abs().max());errors[key]=max(errors[key],err);assert err<=2e-6,(key,g,j,err)
            assert float(batch['k'][g,j])==16
            weight_sums[t]+=float(batch['weight'][g,j])
        if len(group)<8:
            for k in ['pixel_values','coc','k','mpi_mask','weight']:assert torch.count_nonzero(batch[k][g,len(group):])==0
    assert np.max(np.abs(weight_sums-1))<1e-6
    assert all(torch.isfinite(batch[k]).all() for k in ['pixel_values','coc','k','mpi_mask','weight'])
    # Execute the actual public decoder's overlap/padding assembly on frame IDs.
    # An identity VAE isolates indexing without adding a neural inference test.
    from types import SimpleNamespace
    class IdentityVAE:
        config=SimpleNamespace(scaling_factor=1)
        def forward(self,x,num_frames=None,start_frame=0):return SimpleNamespace(sample=x)
        decode=forward
    fake=SimpleNamespace(vae=IdentityVAE());cache=None;assembled=[];chunk_ids=[]
    index_latents=torch.zeros(6,8,1,1,1)
    for g,group in enumerate(ids):index_latents[g,:len(group),0,0,0]=torch.tensor(group)
    for i in range(0,6,2):
        result,cache=pipeline_class.decode_latents(fake,index_latents[i:i+2].clone(),8,8,batch['weight'][i:i+2],4,cache,i==4)
        values=result[0,0,:,0,0].tolist();assembled.extend(values);chunk_ids.append([int(round(v)) for v in values])
    assert np.max(np.abs(np.asarray(assembled)-np.arange(25)))<3e-6
    assert chunk_ids==[list(range(8)),list(range(8,16)),list(range(16,25))]
    return dict(input_groups=ids,padded_positions=[[5,5],[5,6],[5,7]],decoder_source_indices=chunk_ids,
        author_decoder_identity_VAE_index_check=True,max_input_errors=errors,
        max_weight_sum_error=float(np.max(np.abs(weight_sums-1))),
        batch_shapes={k:list(batch[k].shape) for k in ['pixel_values','coc','k','mpi_mask','weight']},
        focus_values=[x['focus'] for x in meta['frames'][:25]],K=16,
        inference_inputs=['original RGB JPEG','prepared VDA PNG8','filename focus','K16'],
        GT_mask_argument_supplied=False)


def run_any2bokeh(p,c,out,order,names):
    repo=ROOT.parent/'any-to-bokeh';assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==p['any2bokeh']['author_commit']
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--','utils/dataset.py','pipelines/any2bokeh_pipe.py'],text=True)
    os.environ.setdefault('HF_HOME',str(ROOT/'models/huggingface'));sys.path.insert(0,str(repo))
    from models.unet import UNetSpatioTemporalConditionModel
    from models.vae import AutoencoderKLTemporalDecoder
    from pipelines.any2bokeh_pipe import StableVideoDiffusionPipeline
    from utils.dataset import BokehCocMPIDatasetWholeVid,collate_fn_val_coc_mpi_whole_video
    torch.set_num_threads(4);torch.manual_seed(0)
    cp=ROOT/p['any2bokeh']['checkpoint']
    unet=UNetSpatioTemporalConditionModel.from_pretrained(str(cp),subfolder='unet',torch_dtype=torch.float16,low_cpu_mem_usage=False,local_files_only=True)
    vae=AutoencoderKLTemporalDecoder.from_pretrained(str(cp/'vae'),torch_dtype=torch.float16,local_files_only=True)
    pipe=StableVideoDiffusionPipeline.from_pretrained('stabilityai/stable-video-diffusion-img2vid-xt',unet=unet,vae=vae,
        torch_dtype=torch.float16,variant='fp16',low_cpu_mem_usage=False,local_files_only=True).to('cuda:0')
    pipe.set_progress_bar_config(disable=True)
    prefix=restore_prefix(out,order)
    print('ATB_LOADED_RESUME_INDEX',prefix,flush=True)
    for index,name in enumerate(order):
        if name not in names or index<prefix:continue
        folder,_,_,meta=inputs(c,name);audit=out/'audit'/name;audit.mkdir(parents=True,exist_ok=True)
        csvpath=audit/'input.csv';rows=[['aif_folder','disp_folder','k'],[str(folder/'rgb'),str(folder/'disparity'),'16']]
        if csvpath.exists():assert list(csv.reader(csvpath.open()))==rows
        else:
            with csvpath.open('x') as f:csv.writer(f).writerows(rows)
        ds=BokehCocMPIDatasetWholeVid(str(csvpath),sample_size=(576,1024))
        batch=collate_fn_val_coc_mpi_whole_video([ds[0]])
        check=audit_atb_batch(ds,batch,folder,meta,StableVideoDiffusionPipeline)
        write(audit/'input.json',check)
        cache=None;frames=[];timings=[];torch.cuda.reset_peak_memory_stats()
        for i in range(0,batch['coc'].shape[0],2):
            tensors={k:batch[k][i:i+2].to('cuda',dtype=torch.float32 if k in ['pixel_values','coc'] else torch.float16)
                     for k in ['pixel_values','coc','k','mpi_mask','weight']}
            torch.cuda.synchronize();start=time.monotonic()
            with torch.inference_mode():
                result,cache=pipe(tensors['coc'],tensors['pixel_values'],input_k=tensors['k'],mpi_mask=tensors['mpi_mask'],
                    weight=tensors['weight'],latent_cache=cache,is_last_batch=(i==4),overlap_frame=4,height=576,width=1024,
                    num_frames=8,decode_chunk_size=8,motion_bucket_id=127,fps=7,noise_aug_strength=0.,num_inference_steps=1,
                    generator=torch.Generator(device='cpu').manual_seed(0),batch_size=tensors['weight'].shape[0],output_type='pt')
            torch.cuda.synchronize();assert torch.isfinite(result[0]).all()
            assert len(result[0])==len(check['decoder_source_indices'][i//2])
            timings.append(time.monotonic()-start);frames.extend(result[0].detach().cpu())
            print('ATB_CHUNK',name,i//2,len(result[0]),round(timings[-1],2),flush=True)
        assert len(frames)==25
        rows=[]
        for t,frame in enumerate(frames):
            row=save_rgb(out/'predictions'/name/f'{t:05d}.png',frame.permute(1,2,0).numpy());row.update(frame=t,focus=meta['frames'][t]['focus'],K=16)
            rows.append(row)
        save_rng(audit/'rng-after.pt',index+1)
        write(audit/'complete.json',dict(scene=name,frames=25,seconds=sum(timings),chunk_seconds=timings,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,outputs=rows,finite=True,
            rng_stream='global torch0 plus public per-call CPU generator0; continued across ordered scenes'))
        print('DONE',name,25,round(sum(timings),2),flush=True)
        del frames,batch,ds,result,cache,tensors


def run_frame_method(method,p,c,out,order,names):
    protocol=json.loads((ROOT/p['frame_methods'][method]).read_text())
    repo=ROOT.parent/{'mpib':'MPIB','bokehme':'BokehMe','drbokeh':'DrBokeh','bokehdiff':'BokehDiff'}[method]
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==protocol['author_commit']
    paths={'mpib':['demo.py','models','utils'],
           'bokehme':['demo.py','neural_renderer.py','classical_renderer'],
           'drbokeh':['app/Render','app/cuda-src'],
           'bokehdiff':['inference_hf.py','dataset.py','PISA_attn_processor.py','custom_diffusers']}[method]
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--',*paths],text=True)
    write(out/'model-protocol-reference.json',protocol)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if method=='bokehdiff':
        from accelerate.utils import set_seed
        from baselines.bokehdiff import OfficialBokehDiff,map_controls
        set_seed(42)
        model=OfficialBokehDiff(repo,ROOT/'models/BokehDiff')
        write(out/'model-loading.json',model.weight_loading)
    else:
        torch.manual_seed(0)
        if method=='mpib':
            from run_mpib_recovery import setup,predict_layers
            nets,counts=setup();write(out/'model-loading.json',dict(strict=True,parameters=counts))
        elif method=='bokehme':
            from baselines.bokehme import setup,defocus_input,predict
            pipeline,demo_args,nets,counts=setup(repo,ROOT/'models/BokehMe',protocol)
            write(out/'model-loading.json',dict(strict=True,parameters=counts))
        else:
            from baselines.drbokeh import OfficialDrBokeh,map_geometry
            model=OfficialDrBokeh(repo,protocol)
            write(out/'model-loading.json',dict(protocol='Full-public LDF+LaMa; no alpha_override',
                gamma=protocol['gamma'],lens=protocol['lens'],cuda_launch=protocol['cuda_launch']))
    prefix=restore_prefix(out,order)
    print('MODEL_LOADED',method,'resume_index',prefix,flush=True)
    for index,name in enumerate(order):
        if name not in names or index<prefix:continue
        folder,rgbs,disps,meta=inputs(c,name);auditdir=out/'audit'/name;auditdir.mkdir(parents=True,exist_ok=True)
        rows=[]
        for t,(rp,dp) in enumerate(zip(rgbs,disps)):
            rgb=cv2.cvtColor(cv2.resize(cv2.imread(str(rp)),(1024,576),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB).astype(np.float32)/255
            q=cv2.imread(str(dp),cv2.IMREAD_UNCHANGED);assert q.dtype==np.uint8 and q.ndim==2
            d=cv2.resize(q.astype(np.float64)/255,(1024,576),interpolation=cv2.INTER_LINEAR)
            focus=float(dp.stem.split('_zf_')[1]);assert focus==meta['frames'][t]['focus']
            assert rgb.shape==(576,1024,3) and d.shape==(576,1024) and np.isfinite(d).all()
            if method in ('mpib','bokehme'):d=d.astype(np.float32)
            lo,hi=float(d.min()),float(d.max());span=hi-lo;assert span>0
            base=dict(scene=name,frame=t,K=16,focus=focus,source_RGB=str(rp.relative_to(ROOT)),
                source_disparity=str(dp.relative_to(ROOT)),resized_disparity_range=[lo,hi],span=span,
                GT_mask_argument_supplied=False,model_inputs=['RGB','VDA disparity','focus','K16'])
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
            if method=='mpib':
                with torch.inference_mode():
                    layers,alpha=predict_layers(rgb,d,16,nets)
                    result=nets['renderer'](layers,alpha,16,focus,2.2,norm=True,increase_focus_range=True)
                assert result.shape==(1,3,576,1024) and torch.isfinite(result).all()
                value=result[0].permute(1,2,0).detach().cpu().numpy()
                base.update(gamma=2.2,disparity_normalization=False)
                del layers,alpha,result
            elif method=='bokehme':
                control,g=defocus_input(d,focus,16,'normalized',demo_args.defocus_scale)
                value,ranges=predict(pipeline,demo_args,nets,rgb,control)
                base.update(geometry=g,effective_K=16/g['span'],gamma=2.2,returned_ranges=ranges)
            elif method=='drbokeh':
                dn,fn,kn,g=map_geometry(d,focus,16)
                assert g['signed_CoC_error_float64']<=protocol['mapping_tolerance']['float64_max_abs']
                assert g['signed_CoC_error_float32']<=protocol['mapping_tolerance']['float32_max_abs']
                value,checks,arrays=model.predict(rgb,dn,fn,kn)
                assert checks['calls']['LDF']==1 and checks['calls']['MiDaS']==0
                assert all(v['radial_tail_mass_at_max']<=protocol['kernel_check']['omitted_radial_scatter_weight_mass_tolerance'] for v in checks['actual_layers'].values())
                base.update(geometry=g,full_public=checks)
                if name in p['preflight_sequences'] or t in p['keyframes']:
                    path=out/'diagnostics'/name/f'{t:05d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                    if not path.exists():
                        with path.open('xb') as f:np.savez_compressed(f,**arrays)
                    base['diagnostics']=str(path.relative_to(ROOT))
                del arrays
            else:
                dn,fn,g=map_controls(d,focus,16,'normalized')
                image,checks=model.predict(rgb,dn,fn,16,42,reset_global_seed=False,quantization='round_png8')
                value=model.output_float
                assert checks['reset_global_seed'] is False and checks['seed']==42
                assert abs(checks['defocus_condition_range'][1]-g['max_defocus_condition'])<1e-5
                base.update(geometry=g,effective_K=16/g['span'],bokehdiff=checks)
            torch.cuda.synchronize();base.update(seconds=time.monotonic()-start,finite=True,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
            base['output']=save_rgb(out/'predictions'/name/f'{t:05d}.png',value)
            path=auditdir/f'{t:05d}.json'
            if path.exists():
                old=json.loads(path.read_text())
                assert old['focus']==base['focus'] and old['K']==16 and old['frame']==t
                base=old
            else:write(path,base)
            rows.append(base)
            print('FRAME',method,name,t,round(base['seconds'],2),flush=True)
        save_rng(auditdir/'rng-after.pt',index+1)
        write(auditdir/'complete.json',dict(scene=name,frames=25,finite=True,seconds=sum(r['seconds'] for r in rows),
            peak_allocated_GiB=max(r['peak_allocated_GiB'] for r in rows)))
        print('DONE',method,name,flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--method',required=True,choices=['any2bokeh','mpib','bokehme','drbokeh','bokehdiff'])
    a.add_argument('--stage',choices=['preflight','full'],default='preflight');args=a.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    p=json.loads((ROOT/'configs/davis-baselines.json').read_text());c=json.loads((ROOT/p['data_protocol']).read_text())
    manifest=json.loads((ROOT/c['derived']/'source-manifest.json').read_text());all_ids=sorted(r['id'] for r in manifest['sequences'])
    order=p['preflight_sequences']+[s for s in all_ids if s not in p['preflight_sequences']]
    assert len(order)==len(set(order))==50
    names=order[:8] if args.stage=='preflight' else order
    if args.stage=='full':
        gate=json.loads((ROOT/'03_实验记录/records/davis-baseline-preflight-20260929.json').read_text())
        assert gate['status']=='ALL_FIVE_PREFLIGHTS_PASSED'
    out=ROOT/p['output']/args.method
    out.mkdir(parents=True,exist_ok=True)
    # A separately scheduled method may finish before the original supervisor
    # reaches it. Serialize that method and reuse its complete outputs.
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        write(out/'protocol.json',dict(configuration=p,data_protocol=c,scene_order=order,method=args.method))
        complete=all((out/'audit'/n/'complete.json').exists() and
                     len(list((out/'predictions'/n).glob('*.png')))==25 for n in names)
        if complete:print('REUSE_COMPLETE_METHOD',args.method,len(names),flush=True)
        elif args.method=='any2bokeh':run_any2bokeh(p,c,out,order,names)
        else:run_frame_method(args.method,p,c,out,order,names)
    print('METHOD_INFERENCE_COMPLETE',args.method,args.stage,flush=True)


if __name__=='__main__':main()
