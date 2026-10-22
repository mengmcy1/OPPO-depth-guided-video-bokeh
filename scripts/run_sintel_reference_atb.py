"""Small paired aligned-reference CoC diagnostic. Native author model unchanged."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from check_atb_mpi_invariance import author_imports,load_model,digest,rng_state,KEYS

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs/ours-sintel-reference-atb-20261023.json'
C=json.loads(CFG.read_text());OUT=ROOT/C['output']
BASE=json.loads((ROOT/C['source_config']).read_text())
PAIR=json.loads((ROOT/C['label_config']).read_text())


def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def prepare():
    assert not OUT.exists()
    _,_,_,_,Dataset,collate=author_imports()
    assert C['samples']==[s['sequence'] for s in BASE['sequences'] if s['split']=='pilot_train'][:2]
    prepared=[]
    for name in C['samples']:
        s=next(x for x in BASE['sequences'] if x['sequence']==name)
        dest=OUT/'scenes'/name;dest.mkdir(parents=True)
        with np.load(ROOT/BASE['derived_root']/'vda-native'/name/'prediction.npz') as z:q=z['disparity_q8']
        with np.load(ROOT/PAIR['output_root']/name/'pairs.npz') as z:ref,valid=z['reference'],z['supervision_domain']
        assert q.shape==ref.shape==valid.shape==(25,436,1024)
        qr=q.copy();qr[valid]=np.floor(ref[valid]*255).astype(np.uint8)
        focus=float(f'{q[0,202:234,496:528].mean()/255:.5f}')
        for sub in ['rgb','A-disparity','R-disparity','regions']:(dest/sub).mkdir()
        regions=[];coc_a=[];coc_r=[]
        for i,t in enumerate(s['frames']):
            rgb=np.asarray(Image.open(ROOT/BASE['raw_root']/'training/clean'/name/f'frame_{t:04d}.png'))
            Image.fromarray(np.pad(rgb,((6,6),(0,0),(0,0)),mode='edge')).save(dest/'rgb'/f'{i:05d}.png')
            qa=np.pad(q[i],((6,6),(0,0)),mode='edge');rr=qa.copy();rr[6:-6]=qr[i]
            for label,arr in [('A',qa),('R',rr)]:Image.fromarray(arr).save(dest/(label+'-disparity')/f'{i:05d}_zf_{focus:.5f}.png')
            ca=np.abs(q[i].astype(np.float64)/255-focus)*16;cr=np.abs(qr[i].astype(np.float64)/255-focus)*16
            support=valid[i].copy();support[:8]=False;support[-8:]=False;support[:,:8]=False;support[:,-8:]=False
            masks=dict(near_focus=support&(cr<=.5),middle=support&(cr>.5)&(cr<=2),far_defocus=support&(cr>2),
                       reduce_defocus=support&(ca-cr>=.5),increase_defocus=support&(cr-ca>=.5),all_valid=support)
            np.savez_compressed(dest/'regions'/f'{i:05d}.npz',**masks)
            regions.append(dict(frame=i,source_frame=t,regions={k:int(v.sum()) for k,v in masks.items()},
                                original_center_nominal_coc=float(np.median(ca[202:234,496:528])),
                                reference_center_nominal_coc=float(np.median(cr[202:234,496:528]))))
            coc_a.append(ca);coc_r.append(cr)
        datasets={};batches={}
        for label in ['A','R']:
            cp=dest/(label+'-input.csv')
            with cp.open('x',newline='') as f:
                w=csv.writer(f,lineterminator='\n');w.writerow(['aif_folder','disp_folder','k']);w.writerow([dest/'rgb',dest/(label+'-disparity'),16])
            ds=Dataset(str(cp),sample_size=(448,1024));datasets[label]=ds;batches[label]={k:v for k,v in collate([ds[0]]).items() if k in KEYS}
        a,r=batches['A'],batches['R'];groups=[list(range(t,min(t+8,25))) for t in [0,4,8,12,16,20]]
        real=torch.zeros((6,8),dtype=torch.bool)
        for g,ids in enumerate(groups):
            real[g,:len(ids)]=True
            for j,i in enumerate(ids):
                for label,source in [('A',q),('R',qr)]:
                    base_q=np.pad(q[i],((6,6),(0,0)),mode='edge')
                    if label=='R':base_q[6:-6]=source[i]
                    expected=(torch.from_numpy(base_q.astype(np.float32))-focus*255).abs().div(255).repeat(3,1,1)
                    expected=datasets[label].pixel_transforms(expected)
                    assert torch.equal(expected,batches[label]['coc'][g,j])
        assert int(real.sum())==45 and int((~real).sum())==3
        for key in ['pixel_values','k','weight']:assert torch.equal(a[key],r[key])
        assert torch.equal(a['coc'][~real],r['coc'][~real])
        # R MPI from its loader is deliberately discarded; all non-CoC inputs use A.
        torch.save(dict(base=a,reference_coc=r['coc'],real=real),dest/'inputs.pt')
        info=dict(scene=name,focus=focus,K=16,groups=groups,real_slots=45,padding_slots=3,
                  base_hashes={k:digest(v) for k,v in a.items()},R_coc_hash=digest(r['coc']),
                  regions=regions,reference_source='dual-input-v2 bounded sequence fit',
                  reference_MPI_discarded=True,GT_only_in_R=True,nonCoC_inputs_unchanged=True)
        save(dest/'prepared.json',info);prepared.append(info)
        print('PREPARED',name,'focus',focus,flush=True)
    save(OUT/'prepared.json',dict(config=C,scenes=prepared))
    save(ROOT/(C['record_prefix']+'-prepared.json'),dict(config=C,scenes=prepared))


def worker(scene,variant):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    dest=OUT/'scenes'/scene;run=dest/variant;assert not run.exists();run.mkdir()
    pipe,p=load_model();assert p['author_commit']==C['author_commit']
    loaded=torch.load(dest/'inputs.pt',map_location='cpu',weights_only=True)
    base=loaded['base'];batch=dict(base)
    if variant=='R':batch['coc']=loaded['reference_coc']
    for k in ['pixel_values','k','mpi_mask','weight']:assert torch.equal(batch[k],base[k])
    assert torch.equal(batch['coc'][~loaded['real']],base['coc'][~loaded['real']])
    assert not hasattr(pipe.vae.encoder,'residual_down_blocks')
    torch.manual_seed(C['seed']);torch.cuda.manual_seed_all(C['seed']);cache=None
    entry=rng_state();events=[];active={}
    from models.vae import DiagonalGaussianDistribution
    encode=pipe.vae.encode;sample=DiagonalGaussianDistribution.sample
    def encode_observer(*args,**kwargs):
        active['branch']='RGB' if kwargs.get('cache_flag',False) else 'CoC'
        return encode(*args,**kwargs)
    def sample_observer(dist,*args,**kwargs):
        before=rng_state();value=sample(dist,*args,**kwargs)
        events.append(dict(unit=active['unit'],branch=active['branch'],shape=list(value.shape),rng_before=before,rng_after=rng_state(),latent_sha256=digest(value)))
        return value
    pipe.vae.encode=encode_observer;DiagonalGaussianDistribution.sample=sample_observer
    save(run/'runtime.json',dict(python=sys.version,torch=torch.__version__,GPU=torch.cuda.get_device_name(0),seed=C['seed'],
                               entry_rng=entry,base_hashes={k:digest(v) for k,v in batch.items()},config=C,
                               matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32))
    arr=np.lib.format.open_memmap(run/'float.npy',mode='w+',dtype=np.float32,shape=(25,436,1024,3))
    (run/'png').mkdir();offset=0;torch.cuda.reset_peak_memory_stats()
    for unit,g in enumerate([0,2,4]):
        active['unit']=unit
        keys={k:batch[k][g:g+2].to('cuda',dtype=torch.float32 if k in ['pixel_values','coc'] else torch.float16) for k in KEYS}
        with torch.inference_mode():
            value,cache=pipe(keys['coc'],keys['pixel_values'],input_k=keys['k'],mpi_mask=keys['mpi_mask'],weight=keys['weight'],latent_cache=cache,
                  is_last_batch=g==4,overlap_frame=4,height=448,width=1024,num_frames=8,decode_chunk_size=8,motion_bucket_id=127,fps=7,
                  noise_aug_strength=0.,num_inference_steps=1,generator=torch.Generator(device='cpu').manual_seed(0),batch_size=2,output_type='pt')
        assert tuple(value[0].shape)==((9 if g==4 else 8),3,448,1024) and torch.isfinite(value[0]).all()
        x=value[0][:,:,6:-6,:].detach().cpu().permute(0,2,3,1).numpy();arr[offset:offset+len(x)]=x
        for j,im in enumerate(x):Image.fromarray(np.uint8(np.round(np.clip(im,0,1)*255))).save(run/'png'/f'{offset+j:05d}.png')
        offset+=len(x);print('CHUNK',scene,variant,unit,offset,flush=True)
        del value,keys,x
    assert offset==25 and len(events)==6;arr.flush()
    save(run/'complete.json',dict(frames=offset,finite=True,events=events,end_rng=rng_state(),peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def run():
    assert subprocess.check_output(['git','show','HEAD:'+str(CFG.relative_to(ROOT))],cwd=ROOT)==CFG.read_bytes()
    assert json.loads((OUT/'prepared.json').read_text())['config']==C
    for scene in C['samples']:
        for variant in C['variants']:
            target=OUT/'scenes'/scene/variant
            if (target/'complete.json').exists():continue
            assert not target.exists(),'Investigate partial worker, do not overwrite'
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy; do not terminate other tasks'
            log=OUT/f'{scene}-{variant}.log'
            with log.open('x') as f:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--scene',scene,'--variant',variant],env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            print('DONE',scene,variant,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run','worker']);parser.add_argument('--scene',choices=C['samples']);parser.add_argument('--variant',choices=C['variants']);a=parser.parse_args()
    torch.set_num_threads(4)
    worker(a.scene,a.variant) if a.phase=='worker' else globals()[a.phase]()
