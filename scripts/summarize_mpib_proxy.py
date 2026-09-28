"""Validate all proxy artifacts and collect overall/K-group results."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
cfg=json.loads((a.run/'clips/config.json').read_text());assert len(cfg['scenes'])==200
rows=[]
for i in range(20):rows.extend(json.loads((a.run/f'metrics-{i:02d}.json').read_text())['rows'])
assert len(rows)==200 and {r['scene'] for r in rows}=={s['id'] for s in cfg['scenes']}
checks=[]
for scene in cfg['scenes']:
    folder=a.run/'clips'/scene['id'];m=json.loads((folder/'metadata.json').read_text());assert m['source']==scene and len(m['frames'])==25
    timing=json.loads((folder/'gt_mpib_public_srgb/timing.json').read_text());assert timing['commit']==cfg['renderer']['commit'] and timing['samples_per_side']==101 and timing['frames']==25
    inp=json.loads((folder/'pred_any2bokeh/input_check.json').read_text());assert inp['frames']==25 and inp['K']==scene['K']
    for sub in ['rgb','gt_mpib_public_srgb','pred_any2bokeh']:
        files=sorted((folder/sub).glob('*.png'));assert [x.stem for x in files]==[f'{i:05d}' for i in range(25)]
        for file in files:
            with Image.open(file) as image:assert image.size==(1024,576) and image.mode=='RGB';image.verify()
    checks.append({'scene':scene['id'],'K':scene['K'],'input_max_coc_quantization_error_px':inp['max_coc_radius_quantization_error_px']})
keys=['PSNR','PSNR_clip_mse','SSIM','LPIPS','FD','RM'];kmap={s['id']:s['K'] for s in cfg['scenes']}
for row in rows:row['K']=kmap[row['scene']]
def mean(rs):return {k:float(np.mean([r[k] for r in rs])) for k in keys}
report={'status':'FULL_200_MPIB_PUBLIC_PROXY_PIPELINE_COMPLETE','name':cfg['name'],'config':'configs/any2bokeh-mpib-public-proxy.json','output_root':str(a.run),'frames_each_rgb_gt_prediction':5000,'mean':mean(rows),'by_K':{str(k):{'scenes':sum(r['K']==k for r in rows),'mean':mean([r for r in rows if r['K']==k])} for k in [4,8,16,32]},'rows':rows,'output_checks':checks,'renderer':cfg['renderer'],'metrics':cfg['metrics'],'paper_reference':{'source':'https://arxiv.org/html/2505.21593v2#S4.T1','PSNR':38.899,'SSIM':.974,'FD':.431,'RM':.007,'LPIPS':None,'scope':'External reference; not same test samples/renderer/evaluation details'},'differences':cfg['differences'],'limits':['Natural train foregrounds are fixed composition assets, not physical color GT','Not strict held-out; no assertion that author checkpoint never saw materials','MPIB public kernel not proven identical to private author kernel','Raw PNG scored, not author JPEG/MP4 chain','RM evaluates all24 transitions; public metrics code derives count from channel dimension before flattening and may evaluate only2 for B,C,T,H,W input','No FVD/VFID or standardized efficiency claim'],'inference':'Official checkpoint,8-frame windows,overlap4,batch2,num_steps1,seed0,full1024x576; unmodified author model code','timing_scope':'Concurrent renderer/inference/metrics; not standardized performance'}
# Mutually exclusive composition groups; mixed scenes are not counted twice.
def composition(scene):
    kinds={obj['kind'] for obj in scene['objects']}
    return 'AM_only' if kinds=={'AM'} else ('VideoMatte_only' if kinds=={'VM'} else 'mixed_AM_VideoMatte')
compositions={scene['id']:composition(scene) for scene in cfg['scenes']}
report['by_foreground_composition']={}
for group in sorted(set(compositions.values())):
    selected=[r for r in rows if compositions[r['scene']]==group]
    report['by_foreground_composition'][group]={'scenes':len(selected),'mean':mean(selected)}
report['limits'].append('Foreground composition groups are mutually exclusive descriptive statistics; repeated source assets and differing K/focus/content prevent independent or causal difficulty claims.')
resume=a.run/'resume-after-disconnect.json'
report['resume_record']=json.loads(resume.read_text()) if resume.exists() else None
review=a.run/'visual-review.json'
report['visual_review']=json.loads(review.read_text()) if review.exists() else None
with a.report.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps({'mean':report['mean'],'by_K':report['by_K']},indent=2))
