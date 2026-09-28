"""Exploratory renderer comparison on aligned PNG frames, not paper-score reproduction."""
import argparse,json,sys,os,time
from pathlib import Path
import cv2,numpy as np,torch
from skimage.metrics import structural_similarity
root=Path(__file__).resolve().parents[1];os.environ.setdefault('TORCH_HOME',str(root/'models/metrics'))
import lpips
from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--only',default='');p.add_argument('--cycles',default='cycles_optix_s1024_f0.945244');p.add_argument('--references',choices=['both','cycles'],default='both');p.add_argument('--per-frame',action='store_true');p.add_argument('--scope',default='');p.add_argument('--reference-dir');p.add_argument('--reference-label',default='Custom reference');p.add_argument('--report',type=Path);p.add_argument('--prediction-root',type=Path);p.add_argument('--prediction-dir',default='pred_any2bokeh');p.add_argument('--author-rm',action='store_true');p.add_argument('--torchmetrics',action='store_true');a=p.parse_args();report_path=a.report or a.input/'metrics.json'
sys.path.insert(0,str(root/'src'))
from synthetic.any2bokeh_input import discover
folders=discover(a.input,a.only)
references=[('Dr.Bokeh','bokeh_dr'),('Blender Cycles',a.cycles)] if a.references=='both' else [('Blender Cycles',a.cycles)]
if a.reference_dir:references=[(a.reference_label,a.reference_dir)]
if report_path.exists():raise FileExistsError(f'Use a new report path: {report_path}')
torch.set_num_threads(4)
perceptual=lpips.LPIPS(net='squeeze').eval().cuda();weights=Raft_Large_Weights.DEFAULT;flow=raft_large(weights=weights).eval().cuda();preprocess=weights.transforms()
if a.torchmetrics:
 from evaluate_author_public_metrics import public_image_functions
 metric_functions=public_image_functions(root.parent/'any-to-bokeh');metric_functions['tqdm']=lambda x,**kwargs:x
def load(folder):
 imgs=[]
 for p in sorted(folder.glob('*.png')):
  if not p.stem.isdigit():continue
  x=cv2.imread(str(p),cv2.IMREAD_UNCHANGED);scale=65535 if x.dtype==np.uint16 else 255;imgs.append(x[:,:,::-1].astype(np.float32)/scale)
 assert len(imgs)==25,(folder,len(imgs));return np.stack(imgs)
def tensor(x):return torch.from_numpy(x.copy()).permute(0,3,1,2).cuda()
rows=[]
with torch.inference_mode():
 for folder in folders:
  pred=load((a.prediction_root/folder.name if a.prediction_root else folder)/a.prediction_dir)
  for label,sub in references:
   gt=load(folder/sub);ps=[];ss=[];lp=[];fd=[];flow_details=[]
   for j in range(len(pred)):
    ps.append(float(-10*np.log10(max(np.mean((pred[j]-gt[j])**2),1e-12))))
    ss.append(float(structural_similarity(pred[j],gt[j],channel_axis=-1,data_range=1.,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)))
    lp.append(float(perceptual(tensor(pred[j:j+1])*2-1,tensor(gt[j:j+1])*2-1).item()))
   for j in range(1,len(pred)):
    x=tensor(np.stack([pred[j-1],gt[j-1]]));y=tensor(np.stack([pred[j],gt[j]]));x,y=preprocess(x,y);f=flow(x,y)[-1];fd.append(float((f[0]-f[1]).abs().mean().item()))
    if a.per_frame:flow_details.append({'from':j-1,'to':j,'FD':fd[-1],'mean_flow_magnitude_prediction_px':float(f[0].norm(dim=0).mean().item()),'mean_flow_magnitude_reference_px':float(f[1].norm(dim=0).mean().item())})
   rm=float(np.mean(np.abs(np.diff(pred,axis=0)-np.diff(gt,axis=0))))
   row=dict(scene=folder.name,renderer=label,PSNR=float(np.mean(ps)),PSNR_clip_mse=float(-10*np.log10(max(float(np.mean((pred-gt)**2)),1e-12))),SSIM=float(np.mean(ss)),LPIPS=float(np.mean(lp)),FD=float(np.mean(fd)),RM=rm,FVD=None)
   if a.author_rm:row['RM_author_public_2transitions']=float(np.mean(np.abs(np.diff(pred[:3],axis=0)-np.diff(gt[:3],axis=0))))
   if a.torchmetrics:
    pair=[(tensor(gt).permute(1,0,2,3).unsqueeze(0),tensor(pred).permute(1,0,2,3).unsqueeze(0))]
    previous_tf32=torch.backends.cudnn.allow_tf32;torch.backends.cudnn.allow_tf32=False
    for name in ('psnr','ssim'):row[name.upper()+'_TorchMetrics']=float(metric_functions[name](pair))
    torch.backends.cudnn.allow_tf32=previous_tf32
    del pair
   if a.per_frame:row['per_frame']={'PSNR':ps,'SSIM':ss,'LPIPS':lp,'flow_transitions':flow_details,'temporal_difference_error':np.abs(np.diff(pred,axis=0)-np.diff(gt,axis=0)).mean(axis=(1,2,3)).tolist()}
   rows.append(row);print(json.dumps(row),flush=True)
report={'scope':f'{len(folders)} development scenes; prediction/reference distances, not renderer accuracy or official test scores','metrics':{'PSNR':'mean frame PSNR, RGB sRGB [0,1]','SSIM':'Gaussian sigma1.5, RGB, skimage population covariance','LPIPS':'squeeze v0.1, RGB [-1,1]','FD':'torchvision RAFT-large C_T_SKHT_V2, full resolution, mean L1 flow difference over 24 transitions','RM':'mean absolute difference of predicted and reference frame differences over all 24 transitions','FVD':'not computed: distribution-estimator protocol not configured for this run'},'rows':rows,'mean':{}}
if a.scope:report['scope']=a.scope
report['metrics']['PSNR_clip_mse']='PSNR from pooled MSE over the 25-frame clip; separate from mean frame PSNR'
for label,_ in references:
 report['mean'][label]={k:float(np.mean([r[k] for r in rows if r['renderer']==label])) for k in ['PSNR','PSNR_clip_mse','SSIM','LPIPS','FD','RM']}
 if a.author_rm:report['mean'][label]['RM_author_public_2transitions']=float(np.mean([r['RM_author_public_2transitions'] for r in rows if r['renderer']==label]))
 if a.torchmetrics:
  for key in ('PSNR_TorchMetrics','SSIM_TorchMetrics'):report['mean'][label][key]=float(np.mean([r[key] for r in rows if r['renderer']==label]))
if a.author_rm:report['metrics']['RM_author_public_2transitions']='Equivalent to public temporal_consistency B,C,T,H,W dimensionality behavior at batch=1: first two transitions only; diagnostic, not replacement for RM'
if a.torchmetrics:report['metrics']['TorchMetrics']='Unmodified author public functions per clip; TF32 disabled as in Any-to-Bokeh crosscheck; scene values aggregated in float64 instead of public float32 accumulator; version 1.9.0'
if a.prediction_root:report['prediction_root']=str(a.prediction_root)
report['prediction_dir']=a.prediction_dir
report_path.write_text(json.dumps(report,indent=2));print(json.dumps(report['mean']),flush=True)
