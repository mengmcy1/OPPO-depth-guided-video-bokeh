"""B36 fixed single-frame intervention in native25-frame context; existing outputs read-only."""
import argparse,csv,json,math,os,sys,time,subprocess,gc
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs/any2bokeh-B36-causal-diagnosis-20261015.json'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def region(a,box):
 x,y,w,h=box;return a[y:y+h,x:x+w]
def nativebox(a,box):
 x,y,w,h=box;return a[math.floor(y*1.875):math.ceil((y+h)*1.875),math.floor(x*1.875):math.ceil((x+w)*1.875)]
def stats(a):return {'count':int(a.size),'p10_p50_p90':np.quantile(a,[.1,.5,.9]).tolist(),'mean':float(a.mean())}
def prepare(c,out):
 assert not (out/'prepared.json').exists();out.mkdir(parents=True,exist_ok=True)
 source=ROOT/'data/processed/davis2016-v2/eval'/c['scene'];g=read(ROOT/'data/processed/davis2016-v2/sequences'/c['scene']/'geometry.json');f=g['frames'][0];q=np.asarray(Image.open(source/'disparity'/f['file']));m=np.asarray(Image.open(source/'mask/00000.png'))>0;d=q.astype(float)/255
 poly=np.zeros_like(q);pts=np.round(np.array(c['head_polygon_display'])*1.875).astype(np.int32);cv2.fillPoly(poly,[pts],1);head=(poly>0)&m
 target=int(round(float(np.median(nativebox(q,[655,223,20,26])))));delta=target-q[head].astype(np.int16)
 ys,xs=np.where(head);shift=round(c['water_translation_display'][0]*1.875);water=np.zeros_like(head);water[ys,xs+shift]=True;assert not (head&water).any()
 qh=q.copy();qh[head]=target;qw=q.copy();raw=q[ys,xs+shift].astype(np.int16)+delta;qw[ys,xs+shift]=np.clip(raw,0,255)
 fh=float(f"{float(np.median(nativebox(d,[664,163,20,22]))):.5f}")
 checks=[]
 for t in range(25):
  row=g['frames'][t];dq=np.asarray(Image.open(source/'disparity'/row['file']),dtype=float)/255;mask=np.asarray(Image.open(source/'mask'/f'{t:05d}.png'))>0
  rawann=np.asarray(Image.open(ROOT/'data/raw/DAVIS-2017/DAVIS/Annotations/Full-Resolution'/c['scene']/f'{t:05d}.png'))
  assert np.array_equal(rawann!=0,mask);assert round(float(dq[mask].mean()),5)==row['focus']
  er=cv2.erode(mask.astype(np.uint8),np.ones((9,9),np.uint8))>0
  checks.append(dict(frame=t,focus=row['focus'],union_pixels=int(mask.sum()),union_d=stats(dq[mask]),eroded_median=float(np.median(dq[er])),focus_minus_median=float(row['focus']-np.median(dq[mask])),q_step_defocus=16/255))
 for v in c['variants']:
  dest=out/'inputs'/v;dest.mkdir(parents=True,exist_ok=True)
  for folder in ['rgb','disparity']:(dest/folder).mkdir(exist_ok=True)
  for t in range(25):
   rgb=source/'rgb'/f'{t:05d}.jpg';(dest/'rgb'/rgb.name).symlink_to(rgb.resolve())
   dp=source/'disparity'/g['frames'][t]['file']
   name=f'00000_zf_{fh:.5f}.png' if t==0 and v=='focus_to_head' else dp.name
   if t==0 and v in ['head_to_neck','translated_water_delta']:Image.fromarray(qh if v=='head_to_neck' else qw).save(dest/'disparity'/name)
   else:(dest/'disparity'/name).symlink_to(dp.resolve())
  with (dest/'input.csv').open('w') as file:csv.writer(file).writerows([['aif_folder','disp_folder','k'],[str(dest/'rgb'),str(dest/'disparity'),16]])
 Image.fromarray(np.uint8(head)*255).save(out/'head-intervention-native.png');Image.fromarray(np.uint8(water)*255).save(out/'water-control-native.png')
 # Input-selected ROIs; no candidate-output-based selection.
 rois={'head':[664,163,20,22],'bill':[700,218,12,8],'neck':[655,223,20,26],'body':[490,255,50,24],'water_near_head':[745,170,35,30],'far_water':[90,85,150,100]}
 info=dict(config=c,target_q=target,head_focus_diagnostic=fh,original_focus=f['focus'],head_pixels=int(head.sum()),water_pixels=int(water.sum()),water_annotation_overlap=int((water&m).sum()),water_clipped_pixels=int(((raw<0)|(raw>255)).sum()),head_delta_q=stats(delta),original_head_d=stats(d[head]),rois=rois,input_checks=checks,conditions='Only frame0 is modified; no actual focus recomputation except named focus sensitivity control.')
 audits={}
 for method in ['bokehme','bokehdiff','mpib','drbokeh']:
  file=ROOT/'outputs/davis-baselines-20260929'/method/'audit'/c['scene']/'00000.json';a=read(file);audits[method]={k:a[k] for k in ['focus','K','span','effective_K','geometry','gamma'] if k in a}
 info['existing_adapter_audits_frame0']=audits;save(out/'prepared.json',info);print('PREPARED',target,fh,'water_clip',info['water_clipped_pixels'],flush=True)

def infer(c,out):
 assert (out/'prepared.json').is_file(), 'Complete CPU preparation before inference'
 import torch
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';torch.set_num_threads(4);cv2.setNumThreads(1)
 repo=ROOT.parent/'any-to-bokeh';p=read(ROOT/'configs/davis-baselines.json');assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==p['any2bokeh']['author_commit']
 assert not subprocess.check_output(['git','-C',str(repo),'diff','--','utils/dataset.py','pipelines/any2bokeh_pipe.py'],text=True)
 os.environ.setdefault('HF_HOME',str(ROOT/'models/huggingface'));sys.path.insert(0,str(repo));torch.manual_seed(0)
 from models.unet import UNetSpatioTemporalConditionModel
 from models.vae import AutoencoderKLTemporalDecoder
 from pipelines.any2bokeh_pipe import StableVideoDiffusionPipeline
 from utils.dataset import BokehCocMPIDatasetWholeVid,collate_fn_val_coc_mpi_whole_video
 cp=ROOT/p['any2bokeh']['checkpoint'];unet=UNetSpatioTemporalConditionModel.from_pretrained(str(cp),subfolder='unet',torch_dtype=torch.float16,low_cpu_mem_usage=False,local_files_only=True)
 vae=AutoencoderKLTemporalDecoder.from_pretrained(str(cp/'vae'),torch_dtype=torch.float16,local_files_only=True)
 pipe=StableVideoDiffusionPipeline.from_pretrained('stabilityai/stable-video-diffusion-img2vid-xt',unet=unet,vae=vae,torch_dtype=torch.float16,variant='fp16',low_cpu_mem_usage=False,local_files_only=True).to('cuda:0');pipe.set_progress_bar_config(disable=True)
 old=ROOT/'outputs/davis-baselines-20260929/any2bokeh';order=read(old/'protocol.json')['scene_order'];idx=order.index(c['scene']);prior=order[idx-1];rng=torch.load(old/'audit'/prior/'rng-after.pt',weights_only=True);assert rng['next_scene_index']==idx
 runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),saved_rng_predecessor=prior,scene_index=idx,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,author_commit=p['any2bokeh']['author_commit'])
 save(out/'runtime.json',runtime);baseline=None;reference_end=None
 for variant in c['variants']:
  dest=out/'predictions'/variant;assert not dest.exists();dest.mkdir(parents=True)
  torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],device=0)
  ds=BokehCocMPIDatasetWholeVid(str(out/'inputs'/variant/'input.csv'),sample_size=(576,1024));batch=collate_fn_val_coc_mpi_whole_video([ds[0]])
  # No augmentation/generator change; assert only intended source frame condition changes.
  if variant=='native':basebatch={k:v.clone() for k,v in batch.items() if torch.is_tensor(v)}
  diffs={k:float((batch[k]-basebatch[k]).abs().max()) for k in ['pixel_values','coc','mpi_mask','k','weight']}
  for k in ['pixel_values','k','weight']:assert diffs[k]==0
  for k in ['coc','mpi_mask']:
   err=(batch[k]-basebatch[k]).abs()
   for group,start_frame in enumerate(range(0,24,4)):
    if start_frame<=c['source_frame']<min(start_frame+8,25):err[group,c['source_frame']-start_frame]=0
   assert torch.count_nonzero(err)==0
  cache=None;frames=[];torch.cuda.reset_peak_memory_stats();start=time.monotonic()
  for i in range(0,6,2):
   tensors={k:batch[k][i:i+2].to('cuda',dtype=torch.float32 if k in ['pixel_values','coc'] else torch.float16) for k in ['pixel_values','coc','k','mpi_mask','weight']}
   with torch.inference_mode():
    result,cache=pipe(tensors['coc'],tensors['pixel_values'],input_k=tensors['k'],mpi_mask=tensors['mpi_mask'],weight=tensors['weight'],latent_cache=cache,is_last_batch=i==4,overlap_frame=4,height=576,width=1024,num_frames=8,decode_chunk_size=8,motion_bucket_id=127,fps=7,noise_aug_strength=0.,num_inference_steps=1,generator=torch.Generator(device='cpu').manual_seed(0),batch_size=tensors['weight'].shape[0],output_type='pt')
   assert torch.isfinite(result[0]).all();frames.extend(result[0].detach().cpu());print('CHUNK',variant,i//2,len(frames),flush=True)
  assert len(frames)==25;arr=torch.stack(frames).permute(0,2,3,1).numpy();q=np.uint8(np.round(np.clip(arr,0,1)*255));np.save(out/f'{variant}-frame0-float.npy',arr[0])
  for t in range(25):Image.fromarray(q[t]).save(dest/f'{t:05d}.png')
  end=(torch.get_rng_state(),torch.cuda.get_rng_state(0));rr=dict(variant=variant,seconds=time.monotonic()-start,frames=25,finite=True,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,condition_differences=diffs)
  if variant=='native':
   baseline=q.copy();reference_end=end;original=np.stack([np.asarray(Image.open(old/'predictions'/c['scene']/f'{t:05d}.png')) for t in range(25)]);delta=abs(q.astype(float)-original.astype(float));rr['legacy_replay']={'max_byte_error':float(delta.max()),'mean_byte_error':float(delta.mean()),'different_bytes':int(np.count_nonzero(delta))};assert delta.mean()<.1,'Native replay drift too large to proceed'
  else:
   assert torch.equal(end[0],reference_end[0]) and torch.equal(end[1],reference_end[1]);rr['end_RNG_matches_native']=True;delta=abs(q.astype(float)-baseline.astype(float));rr['change_per_frame_MAE_RGB8']=delta.mean(axis=(1,2,3)).tolist();rr['max_byte_change']=float(delta.max())
   if variant=='native_repeat':rr['repeat_different_bytes']=int(np.count_nonzero(delta));assert delta.mean()<.02
  save(out/f'{variant}-run.json',rr);print('DONE',variant,rr,flush=True)
  del frames,batch,ds,result,cache,tensors,arr,q;gc.collect()

def metrics(a,b,box):
 x=region(a,box).astype(np.float64)/255;y=region(b,box).astype(np.float64)/255
 def gray(v):return v[:,:,0]*.299+v[:,:,1]*.587+v[:,:,2]*.114
 def energy(v):
  z=gray(v);gx=cv2.Sobel(z,cv2.CV_64F,1,0,ksize=3)[2:-2,2:-2];gy=cv2.Sobel(z,cv2.CV_64F,0,1,ksize=3)[2:-2,2:-2];lp=cv2.Laplacian(z,cv2.CV_64F)[2:-2,2:-2];return float(np.mean(gx*gx+gy*gy)),float(np.mean(lp*lp))
 eg,el=energy(x);rg,rl=energy(y)
 return dict(MAE_vs_reference_RGB8=float(np.abs(x-y).mean()*255),gradient_energy_ratio=eg/max(rg,1e-15),laplacian_energy_ratio=el/max(rl,1e-15))

def extended_cpu_audit(c):
 cfg=read(ROOT/'configs/any2bokeh-B33-B36-diagnostics-20261014.json')['B36'];rows=[];deep=[];controls=[]
 import torch
 import torchvision.transforms.functional as F
 torch.set_num_threads(4)
 base=ROOT/'data/processed/davis2016-v2/eval'/c['scene']
 for t in cfg['keyframes']:
  rgb=np.asarray(Image.open(ROOT/'outputs/any2bokeh-failure-analysis-20261009/B1-batch2-20261014/B36/originals/Input_RGB'/f'{t:05d}.png'))
  rois={k:v for k,v in cfg['ROIs_display_xywh'][str(t)].items() if k!='head_context'}
  for method in ['any2bokeh','mpib','bokehme','drbokeh','bokehdiff','deeplens']:
   a=np.asarray(Image.open(ROOT/'outputs/davis-baselines-20260929'/method/'predictions'/c['scene']/f'{t:05d}.png'))
   for name,box in rois.items():rows.append(dict(frame=t,method=method,region=name,**metrics(a,rgb,box)))
  dp=next((base/'disparity').glob(f'{t:05d}_zf_*.png'));q=np.asarray(Image.open(dp));focus=float(dp.stem.split('_zf_')[1]);d=q.astype(float)/255
  # Match author CoC order: float32 q -> abs(q -255*focus)/255 -> resize.
  coc=(torch.from_numpy(q.astype(np.float32))-focus*255).abs()/255
  actual=F.resize(coc[None],(576,1024))[0].numpy()*16
  cvd=cv2.resize(d,(1024,576),interpolation=cv2.INTER_LINEAR);span=float(cvd.max()-cvd.min())
  for name,box in rois.items():
   controls.append(dict(frame=t,region=name,native_nominal_median=float(np.median(16*abs(nativebox(d,box)-focus))),author_resized_CoC_times_K_median=float(np.median(region(actual,box))),normalized_adapter_nominal_median=float(np.median(16*abs(region(cvd,box)-focus)/span)),span=span))
  db=ROOT/'outputs/davis-baselines-20260929/deeplens/native'/c['scene'];da=read(db/f'{t:05d}.json');dd=np.load(db/'depth'/f'{t:05d}.npy').squeeze();H,W=dd.shape
  for name,(x,y,w,h) in rois.items():
   v=dd[math.floor(y*H/576):math.ceil((y+h)*H/576),math.floor(x*W/1024):math.ceil((x+w)*W/1024)]
   deep.append(dict(frame=t,region=name,low_resolution=[W,H],depth_median=float(np.median(v)),focus_depth=da['focus_depth'],absolute_difference_median=float(np.median(abs(v-da['focus_depth']))),native_aperture_times_difference_median=float(np.median(.5*abs(v-da['focus_depth'])))))
 return dict(five_keyframe_existing_output_metrics=rows,actual_condition_resize_checks=controls,deeplens_own_geometry=deep,limits=['DeepLens native condition units are not interchangeable with K*VDA disparity.','Fixed small ROI gradient ratios are not global background blur measurements or perceptual rankings.','Prior bill ROIs12/18 contain annotation background; kept for traceability, excluded from pure-bill claims.'])

def analyze(c,out):
 info=read(out/'prepared.json');rgb=np.asarray(Image.open(ROOT/'outputs/any2bokeh-failure-analysis-20261009/B1-batch2-20261014/B36/originals/Input_RGB/00000.png'));native=np.asarray(Image.open(out/'predictions/native/00000.png'))
 rows=[]
 for v in c['variants']:
  a=np.asarray(Image.open(out/'predictions'/v/'00000.png'))
  for name,box in info['rois'].items():rows.append(dict(method=v,region=name,vs_input=metrics(a,rgb,box),vs_native=metrics(a,native,box)))
 for method in ['any2bokeh','mpib','bokehme','drbokeh','bokehdiff','deeplens']:
  a=np.asarray(Image.open(ROOT/'outputs/davis-baselines-20260929'/method/'predictions/mallard-water/00000.png'))
  for name,box in info['rois'].items():rows.append(dict(method='frozen_'+method,region=name,vs_input=metrics(a,rgb,box)))
 result=dict(config=c,input_audit=info,runtime=read(out/'runtime.json'),runs=[read(out/f'{v}-run.json') for v in c['variants']],region_metrics=rows,extended_CPU_audit=extended_cpu_audit(c),metric_limits='Input-relative errors and edge energy measure photometric change/structure response, not bokeh accuracy or user quality. Reconstruction may sharpen/hallucinate; one intervened frame.')
 result['diagnosis']={
  'scope':'5 variants x25 output frames; interventions only on source frame0 of one exploration video. GT mask only constructs a diagnostic region, not an implementable new baseline.',
  'supported':['Original frozen focus matches native annotation-union mean on all25 frames; source annotation agrees with prepared mask.',
   'Actual author resize preserves the large head-vs-focus offset at the five checked keyframes; no sign of head defocus created solely by display resizing.',
   'Paired head-to-neck geometry intervention changes head output toward original RGB and raises local gradient energy; identical signed q perturbation translated to water produces almost no head response.',
   'Global focus shift is a different intervention: head response changes together with neck and background.',
   'BokehDiff normalized input control is larger than shared-domain nominal control, despite its higher measured head gradient energy; output differences cannot be explained by a smaller numerical control alone.'],
  'limits':['No metric depth or paired bokeh truth: neck-reference intervention is not a validated physical correction.',
   'CoC and MPI are both rebuilt after disparity change; their individual contributions are not isolated.',
   'The water control matches area and signed q changes, not original defocus, texture, or semantic context.',
   'New output aesthetics remain for user review; edge energy and input residuals are not quality scores.',
   'Only frame0 is intervened; changes in frames1–7 do not establish temporal quality. No claimed across-source generalization.',
   'DeepLens head is near its own focus at frame0, but its own head-focus differences vary across keyframes; do not generalize the frame0 explanation to all25 frames.']}
 save(ROOT/c['record'],result)
 flat=[]
 for r in rows:
  flat.append(dict(method=r['method'],region=r['region'],input_MAE_RGB8=r['vs_input']['MAE_vs_reference_RGB8'],input_gradient_ratio=r['vs_input']['gradient_energy_ratio'],input_laplacian_ratio=r['vs_input']['laplacian_energy_ratio'],change_vs_native_MAE_RGB8=r.get('vs_native',{}).get('MAE_vs_reference_RGB8')))
 csvpath=(ROOT/c['record']).with_suffix('.csv')
 with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
 # Compact local comparison for user; only input-selected fixed frame0 and same crop.
 html=['<!doctype html><meta charset="utf-8"><title>B36局部干预诊断</title><style>body{font-family:system-ui;background:#f6f9ff;margin:24px} .row{display:flex;flex-wrap:wrap;gap:14px}img{image-rendering:pixelated}figure{margin:4px} </style><h1>B36第0帧：局部几何与焦点诊断</h1><p>原RGB及原focus/K/RNG固定，局部干预只作用第0帧。focus_to_head为单独焦点敏感性对照。诊断值不是几何真值；用户判断视觉变化。未修改正式预测。</p>']
 names={'Input':'原始输入','native':'原版重放','native_repeat':'原版重复','head_to_neck':'仅头部视差改为颈部参考值','translated_water_delta':'同幅度水面干预对照','focus_to_head':'仅焦点改到头部参考值'}
 html.append('<p>局部头部修改保持原focus不变；颈部参考值来自原输入，非真实深度。水面对照匹配像素数及视差改变量，但不匹配原离焦强度。五次都保留25帧时序调用，图示为第0帧。</p><table border="1" cellpadding="6"><tr><th>条件</th><th>头部对原图MAE（0–255）</th><th>头部梯度能量比（输入=1）</th><th>头部变化MAE</th><th>邻近水面变化MAE</th></tr>')
 for v in c['variants']:
  h=next(r for r in rows if r['method']==v and r['region']=='head');b=next(r for r in rows if r['method']==v and r['region']=='water_near_head')
  html.append(f'<tr><td>{names[v]}</td><td>{h["vs_input"]["MAE_vs_reference_RGB8"]:.3f}</td><td>{h["vs_input"]["gradient_energy_ratio"]:.3f}</td><td>{h["vs_native"]["MAE_vs_reference_RGB8"]:.3f}</td><td>{b["vs_native"]["MAE_vs_reference_RGB8"]:.3f}</td></tr>')
 html.append('</table><p>MAE与梯度比只衡量变化/结构响应，不是虚化准确率；不能按它们直接决定视觉最好。</p>')
 for title,box in [('头部与喙',[632,128,128,160]),('完整画面',[0,0,1024,576])]:
  html.append(f'<h2>{title}</h2><div class="row">')
  for v in ['Input']+c['variants']:
   a=rgb if v=='Input' else np.asarray(Image.open(out/'predictions'/v/'00000.png'));im=Image.fromarray(region(a,box));fn=f'{v}-{box[2]}.png';im.save(out/fn);html.append(f'<figure><figcaption>{names[v]}</figcaption><a href="{fn}"><img src="{fn}" width="{256 if box[2]<1024 else 512}"></a></figure>')
  html.append('</div>')
 (out/'index.html').write_text(''.join(html));print('RESULT',ROOT/c['record'],flush=True)
 for r in rows:
  if r['region'] in ['head','neck','water_near_head']:print(r,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','infer','analyze']);args=a.parse_args();c=read(CFG);out=ROOT/c['output'];cv2.setNumThreads(1);globals()[args.stage](c,out)
