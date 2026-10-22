"""Fixed-endpoint pilot report, curves, dev previews and checkpoint reload check."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
C=json.loads((ROOT/'configs/ours-student-pilot-v1-20261023.json').read_text());OUT=ROOT/C['output'];PREFIX=ROOT/C['record_prefix']


def save(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def average(rows,key):
    vals=[r[key] for r in rows if r[key] is not None]
    return float(np.mean(vals)) if vals else None


def summarize():
    complete={v:json.loads((OUT/v/'complete.json').read_text()) for v in C['variants']}
    assert len({r['init_sha256'] for r in complete.values()})==len({r['schedule_sha256'] for r in complete.values()})==1
    assert all(r['steps']==2000 and r['exposures']==8000 for r in complete.values())
    assert complete['natural']['branch_counts']==[8000,0] and complete['perturbed']['branch_counts']==[0,8000] and complete['joint']['branch_counts']==[4000,4000]
    table=[];per_scene=[];curves=[]
    for variant in C['variants']:
        final=json.loads((OUT/variant/'eval-02000.json').read_text())
        for row in final['per_scene']:per_scene.append(dict(variant=variant,**row))
        for split in ['pilot_train','pilot_dev']:
            for branch in ['natural','perturbed']:
                for region in sorted({r['region'] for r in final['per_scene']}):
                    rs=[r for r in final['per_scene'] if r['split']==split and r['branch']==branch and r['region']==region]
                    row=dict(variant=variant,split=split,branch=branch,region=region,source_count=sum(r['pixels']>0 for r in rs))
                    for field in ['baseline_mae','mae','preclamp_mae','mean_change','outside_0_1_fraction']:row[field]=average(rs,field)
                    row['MAE_relative_change_percent']=(row['mae']/row['baseline_mae']-1)*100 if row['baseline_mae'] else None
                    table.append(row)
        for step in C['evaluation_steps']:
            e=json.loads((OUT/variant/f'eval-{step:05d}.json').read_text())
            for branch in ['natural','perturbed']:
                rs=[r for r in e['per_scene'] if r['split']=='pilot_dev' and r['branch']==branch and r['region']=='valid']
                curves.append(dict(variant=variant,step=step,branch=branch,baseline_mae=average(rs,'baseline_mae'),mae=average(rs,'mae')))
    primary=[r for r in table if r['split']=='pilot_dev' and r['branch']=='natural' and r['region']=='valid']
    result=dict(status='THREE_EQUAL_BUDGET_RUNS_COMPLETE_FIXED_FINAL_ENDPOINT',config=C,primary=primary,regions=table,per_scene=per_scene,
                runs=complete,curves=curves,aggregation='pixels within each source, then equal weight across sources; two dev sources, not independent final test',
                caveats=['One initialization seed; no statistical-significance or general-method claim.',
                         'GT targets are bounded affine-aligned labels; no post-student alignment or metric-depth accuracy claim.',
                         '150 source frames remain100 train/50 dev; 8000 exposures per run are not8000 independent examples.',
                         'No ATB outputs, real-domain adaptation or DAVIS evaluation in this training experiment.',
                         'Near-reference subset is a proxy for small original geometric error, not calibrated confidence.'])
    save(Path(str(PREFIX)+'-results.json'),result)
    for name,rows in [('-summary.csv',table),('-per-scene.csv',per_scene),('-curves.csv',curves)]:
        with Path(str(PREFIX)+name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(primary,indent=2),flush=True)
    print('NEAR REFERENCE',json.dumps([r for r in table if r['split']=='pilot_dev' and r['branch']=='natural' and r['region']=='near_reference'],indent=2),flush=True)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads(Path(str(PREFIX)+'-results.json').read_text());dest=OUT/'report';dest.mkdir()
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,branch in zip(axes,['natural','perturbed']):
        for variant in C['variants']:
            rs=[r for r in result['curves'] if r['variant']==variant and r['branch']==branch]
            ax.plot([r['step'] for r in rs],[r['mae'] for r in rs],marker='o',label=variant)
        ax.axhline(rs[0]['baseline_mae'],linestyle='--',color='black',label='unchanged input')
        ax.set(title=f'Dev: {branch} input',xlabel='Training steps',ylabel='Aligned-geometry MAE');ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Two pilot dev sources, one seed; fixed final endpoint = 2000 steps')
    fig.savefig(dest/'learning-curves.png',dpi=160);fig.savefig(dest/'learning-curves.pdf');plt.close(fig)
    manifest=json.loads((ROOT/C['data_manifest']).read_text())
    for name in ['cave_2','market_2']:
        rows=[r for r in manifest['frames'] if r['sequence']==name]
        with np.load(ROOT/rows[0]['vda_npz']) as z:d=z['disparity_normalized']
        with np.load(ROOT/rows[0]['pair_npz']) as z:ref=z['reference'];valid=z['supervision_domain']
        predictions={v:np.load(OUT/v/'predictions'/name/'natural.npz')['prediction'] for v in C['variants']}
        sheet=Image.new('RGB',(6*256,3*145),'white');draw=ImageDraw.Draw(sheet)
        labels=['RGB','VDA','fixed reference','natural trained','perturbed trained','joint trained']
        for rr,i in enumerate([0,12,24]):
            rgb=Image.open(ROOT/rows[i]['rgb']).convert('RGB')
            images=[rgb]
            for ar in [d[i],ref[i]]+[predictions[v][i] for v in C['variants']]:
                im=np.repeat(np.uint8(np.clip(ar,0,1)*255)[:,:,None],3,2)
                # Gray unsupported reference is a mask visualization, never supervised depth0.
                if len(images)==2:im[~valid[i]]=[200,80,200]
                images.append(Image.fromarray(im))
            for col,(label,im) in enumerate(zip(labels,images)):
                x,y=col*256,rr*145;draw.text((x+3,y+3),f'f{i+1:04d} {label}',fill='black');sheet.paste(im.resize((256,109)),(x,y+25))
        sheet.save(dest/(name+'-geometry-contact.png'))
    print('FIGURES',dest,flush=True)


def verify():
    import torch
    from src.geometry_student import GeometryResidualUNet,predict
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    m=json.loads((ROOT/C['data_manifest']).read_text());checks=[]
    for variant in C['variants']:
        cp=torch.load(OUT/variant/'final.pt',map_location='cpu',weights_only=False)
        assert cp['config']==C and cp['step']==2000 and cp['variant']==variant
        assert all(float(s['step'])==2000 for s in cp['optimizer']['state'].values())
        model=GeometryResidualUNet().cuda().eval();model.load_state_dict(cp['model'],strict=True)
        for name in ['cave_2','market_2']:
            rows=[r for r in m['frames'] if r['sequence']==name]
            with np.load(ROOT/rows[0]['vda_npz']) as z:natural=z['disparity_normalized']
            with np.load(ROOT/rows[0]['pair_npz']) as z:perturbed=z['perturbed']
            for branch,data in [('natural',natural),('perturbed',perturbed)]:
                stored=np.load(OUT/variant/'predictions'/name/(branch+'.npz'))['prediction']
                assert stored.shape==(25,436,1024) and np.isfinite(stored).all() and stored.min()>=0 and stored.max()<=1
                for i in [0,24]:
                    rgb=np.asarray(Image.open(ROOT/rows[i]['rgb']),dtype=np.float32).transpose(2,0,1)/255.
                    x=torch.from_numpy(np.concatenate([rgb,data[i][None]],0))[None].cuda()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=predict(model,x).clamp(0,1)[0,0].cpu().numpy()
                    assert np.array_equal(prediction,stored[i]),(variant,name,branch,i)
                    checks.append(dict(variant=variant,scene=name,branch=branch,frame=i,checkpoint_reload_bitwise_equal=True))
        del model,cp;torch.cuda.empty_cache()
    save(Path(str(PREFIX)+'-checkpoint-check.json'),dict(status='ALL_FINAL_CHECKPOINTS_RELOAD_AND_REPRODUCE',optimizer_steps=2000,checked_frames=checks,
                                                     final_dev_prediction_arrays_checked=12,training_updates_in_this_check=0))
    print('CHECKPOINT_RELOAD_PASS',len(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['summarize','figures','verify']);a=p.parse_args();globals()[a.phase]()
