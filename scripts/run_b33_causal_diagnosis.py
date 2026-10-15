"""Fixed B33 sparse RGB rope geometry intervention and focus sensitivity control."""
from pathlib import Path
import json,sys,csv,math,argparse
import numpy as np
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_b36_causality import infer,metrics,save

def prepare(c,out):
 assert not (out/'prepared.json').exists();out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
 base=ROOT/'data/processed/davis2016-v2/eval/car-turn';g=json.loads((ROOT/'data/processed/davis2016-v2/sequences/car-turn/geometry.json').read_text());t=22;row=g['frames'][t];q=np.asarray(Image.open(base/'disparity'/row['file']));d=q.astype(float)/255;rgb=cv2.cvtColor(cv2.imread(str(base/'rgb'/f'{t:05d}.jpg')),cv2.COLOR_BGR2RGB);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
 loc=json.loads((ROOT/'03_实验记录/records/any2bokeh-B33-B36-diagnostics-20261014-localization.json').read_text())['rows'];points={r['correspondence_id']:r for r in loc if r['frame_id']==22};st=json.loads((ROOT/'03_实验记录/records/any2bokeh-B33-B36-diagnostics-20261014-B33-input-statistics.json').read_text())['rows']
 for k in ['R1','R2','R4','P-white','P-dark']:assert points[k]['accepted']
 centers=np.array([points[k]['native_center'] for k in ['R1','R2','R4']]);slope,bias=np.polyfit(centers[:,0],centers[:,1],1);xL=points['P-white']['native_center'][0];xR=points['P-dark']['native_center'][0]
 anchors={k:next(r['disparity_p10_p50_p90'][1] for r in st if r['frame_id']==22 and r['correspondence_id']==k and r['region_type']=='post_control') for k in ['P-white','P-dark']}
 # Single-frame segmentation candidates entirely from RGB; do not inspect disparity to pick pixels.
 mask=np.zeros_like(q,bool);trace=[]
 for x in range(math.ceil(xL+12),math.floor(xR-12)):
  y0=int(round(slope*x+bias));candidates=[]
  for off in [-2,-1,0,1,2]:
   y=y0+off;value=gray[y,x]-.5*(gray[y-8,x]+gray[y+8,x]);candidates.append((value,-abs(off),off,y))
  score,_,off,y=max(candidates)
  if score>=10:mask[y-1:y+2,x]=True
  trace.append(dict(native_x=x,native_y=y,contrast=float(score),accepted=bool(score>=10)))
 ys,xs=np.where(mask);assert len(xs)>0
 ref=anchors['P-white']+(xs-xL)/(xR-xL)*(anchors['P-dark']-anchors['P-white']);target=np.round(ref*255).astype(np.int16);delta=target-q[ys,xs].astype(np.int16);qh=q.copy();qh[ys,xs]=np.clip(target,0,255)
 control=np.zeros_like(mask);control[ys+c['control_translation_native_y'],xs]=True;assert not (mask&control).any();qc=q.copy();raw=q[ys+16,xs].astype(np.int16)+delta;qc[ys+16,xs]=np.clip(raw,0,255)
 car=np.asarray(Image.open(base/'mask/00022.png'))>0;assert not (car&(mask|control)).any()
 f1=g['frames'][1]['focus'];inputrows=[]
 for variant in c['variants']:
  dest=out/'inputs'/variant
  for folder in ['rgb','disparity']:(dest/folder).mkdir(parents=True,exist_ok=True)
  for i in range(25):
   rp=base/'rgb'/f'{i:05d}.jpg';rgb_link=dest/'rgb'/rp.name
   if rgb_link.exists():assert rgb_link.resolve()==rp.resolve()
   else:rgb_link.symlink_to(rp.resolve())
   dp=base/'disparity'/g['frames'][i]['file'];fn=f'00022_zf_{f1:.5f}.png' if i==22 and variant=='focus_hold_frame1' else dp.name
   target_file=dest/'disparity'/fn
   if i==22 and variant in ['rope_post_reference','translated_grass_delta']:
    modified=qh if variant=='rope_post_reference' else qc
    if target_file.exists():assert np.array_equal(np.asarray(Image.open(target_file)),modified)
    else:Image.fromarray(modified).save(target_file)
   elif target_file.exists():assert target_file.resolve()==dp.resolve()
   else:target_file.symlink_to(dp.resolve())
  with (dest/'input.csv').open('w') as f:csv.writer(f).writerows([['aif_folder','disp_folder','k'],[str(dest/'rgb'),str(dest/'disparity'),16]])
 Image.fromarray(mask.astype(np.uint8)*255).save(out/'rope-mask-native.png');Image.fromarray(control.astype(np.uint8)*255).save(out/'grass-control-native.png')
 overlay=rgb.copy();overlay[mask]=[0,255,255];overlay[control]=[255,0,255];Image.fromarray(overlay).save(out/'input-mask-overlay-native.png')
 rois={k:[round(points[k]['native_center'][0]/1.875)-24,round(points[k]['native_center'][1]/1.875)-20,48,40] for k in ['R1','R2','R4']};rois.update(car=[750,285,100,100],grass=[560,290,60,30],road=[560,445,110,55])
 info=dict(config=c,anchors=anchors,post_x_native=[xL,xR],guide_slope=slope,guide_intercept=bias,trace=trace,intervention_pixels=int(mask.sum()),control_pixels=int(control.sum()),clipped_control_pixels=int(((raw<0)|(raw>255)).sum()),source_focus=row['focus'],held_focus=f1,delta_q_p10_p50_p90=np.quantile(delta,[.1,.5,.9]).tolist(),rois=rois)
 # Input control changes within fixed existing R1/R2/R4 candidate bands; not new selection.
 from prepare_atb_b33_b36_diagnostics import band_pixels
 for k in ['R1','R2','R4']:
  r=points[k];xx,yy,_=band_pixels(r['native_center'],r['tangent'],0,21,3);old=d[yy,xx];new=qh[yy,xx]/255
  inputrows.append(dict(id=k,original_d_mean=float(old.mean()),reference_d_mean=float(new.mean()),original_nominal_mean=float((16*abs(old-row['focus'])).mean()),reference_nominal_mean=float((16*abs(new-row['focus'])).mean()),modified_fraction=float((qh[yy,xx]!=q[yy,xx]).mean())))
 info['candidate_input_changes']=inputrows;save(out/'prepared.json',info);print('PREPARED',info['intervention_pixels'],info['clipped_control_pixels'],inputrows,flush=True)

def analyze(c,out):
 info=json.loads((out/'prepared.json').read_text());t=22;rgb=np.asarray(Image.open(ROOT/'outputs/any2bokeh-failure-analysis-20261009/B1-batch2-20261014/B33/originals/Input_RGB/00022.png'));native=np.asarray(Image.open(out/'predictions/native/00022.png'));rows=[]
 for v in c['variants']:
  a=np.asarray(Image.open(out/'predictions'/v/'00022.png'))
  for name,box in info['rois'].items():rows.append(dict(variant=v,region=name,vs_input=metrics(a,rgb,box),vs_native=metrics(a,native,box)))
 result=dict(config=c,input_audit=info,runs=[json.loads((out/f'{v}-run.json').read_text()) for v in c['variants']],region_metrics=rows,limits=['Gradient/MAE response is not perceptual accuracy.','Linearly interpolated post predictions are only a diagnostic geometric reference.','Sparse RGB mask may miss the rope or include mixed pixels; line consistency not depth truth.','Single-frame intervention, original focus unchanged except separate focus control.'])
 save(ROOT/c['record'],result)
 names={'Input':'原图','native':'原版','native_repeat':'原版重复','rope_post_reference':'仅细绳采用两端立柱几何参考','translated_grass_delta':'同幅度邻近草地对照','focus_hold_frame1':'仅焦点暂用第1帧值'}
 html=['<!doctype html><meta charset="utf-8"><title>B33栏杆几何诊断</title><style>body{font-family:system-ui;background:#f6f9ff;margin:24px}.row{display:flex;flex-wrap:wrap;gap:12px}figure{margin:4px}img{image-rendering:pixelated}</style><h1>B33第22帧局部输入诊断</h1><p>原RGB、K和RNG固定，保留25帧调用。两端立柱插值只是近似参考，非真实深度。对照焦点固定为第1帧属于独立敏感性实验，不是推荐对焦方式。新输出质量由用户判断。</p><p><a href="input-mask-overlay-native.png">原图上的干预像素：青=细绳候选，粉=草地对照</a></p>']
 for name,box in [('圈选整段',[505,280,218,155])]+list(info['rois'].items())[:3]:
  html.append(f'<h2>{name}</h2><div class="row">');x,y,w,h=box
  for v in ['Input']+c['variants']:
   a=rgb if v=='Input' else np.asarray(Image.open(out/'predictions'/v/'00022.png'));fn=f'{v}-{name}.png';Image.fromarray(a[y:y+h,x:x+w]).save(out/fn);html.append(f'<figure><figcaption>{names[v]}</figcaption><img src="{fn}" width="{w*2}"></figure>')
  html.append('</div>')
 (out/'index.html').write_text(''.join(html))
 for r in rows:
  if r['region'] in ['R1','R2','R4']:print(r,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','analyze']);a=p.parse_args();c=json.loads((ROOT/'configs/any2bokeh-B33-causal-diagnosis-20261015.json').read_text());globals()[a.stage](c,ROOT/c['output'])
