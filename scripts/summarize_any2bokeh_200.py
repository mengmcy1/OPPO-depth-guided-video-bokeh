"""Collect a completed 200-video reproduction, rejecting missing outputs or scores."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
cfg=json.loads((a.run/'clips/config.json').read_text());assert len(cfg['scenes'])==200
rows=[]
for i in range(20):rows.extend(json.loads((a.run/f'metrics-{i:02d}.json').read_text())['rows'])
assert len(rows)==200 and {r['scene'] for r in rows}=={s['id'] for s in cfg['scenes']}
checks=[]
for scene in cfg['scenes']:
 folder=a.run/'clips'/scene['id'];m=json.loads((folder/'metadata.json').read_text());assert m['source']==scene and len(m['frames'])==25
 timing=json.loads((folder/'cycles_optix_s1024_f1/timing.json').read_text());assert timing['samples']==1024 and timing['compute_device_type']=='OPTIX' and timing['per_frame_layer_depths']
 expected=(cfg['focal_length_mm']/1000)**2*cfg['width']/(2*(cfg['sensor_width_mm']/1000)*scene['K']);assert abs(timing['aperture_fstop']-expected)<1e-5
 inp=json.loads((folder/'pred_any2bokeh/input_check.json').read_text());assert inp['frames']==25 and inp['K']==scene['K']
 for sub in ['rgb','cycles_optix_s1024_f1','pred_any2bokeh']:
  files=sorted((folder/sub).glob('*.png'));assert [x.stem for x in files]==[f'{i:05d}' for i in range(25)]
  for file in files:
   with Image.open(file) as image:assert image.size==(1024,576);image.verify()
 checks.append({'scene':scene['id'],'actual_aperture_fstop':timing['aperture_fstop'],'input_max_coc_quantization_error_px':inp['max_coc_radius_quantization_error_px']})
keys=['PSNR','PSNR_clip_mse','SSIM','LPIPS','FD','RM']
report={'status':'FULL_200_PIPELINE_COMPLETE_NOT_OFFICIAL_TEST_OR_FAIR_BENCHMARK','name':cfg['name'],'config':'configs/any2bokeh-reproduction-200.json','output_root':str(a.run),'frames_each_rgb_gt_prediction':5000,'mean':{k:float(np.mean([r[k] for r in rows])) for k in keys},'rows':rows,'output_checks':checks,'paper_reference':{'source':'https://arxiv.org/html/2505.21593v2#S4.T1','PSNR':38.899,'SSIM':.974,'FD':.431,'RM':.007,'LPIPS':None,'scope':'External reference only; different data, renderer and evaluation details'},'differences':cfg['differences'],'metric_limits':['Mean frame PSNR and mean clip-pooled-MSE PSNR both reported; not assumed identical to author aggregation','SSIM uses skimage Gaussian sigma1.5 population covariance; author uses TorchMetrics','LPIPS squeeze v0.1 extra diagnostic; not provided in paper Table1','FD uses all24 transitions and torchvision RAFT-large default C_T_SKHT_V2','RM uses all24 transitions; public metrics code derives transition count from channel dimension before flattening, so literal code may evaluate only2 transitions for B,C,T,H,W input','Raw PNG scored, not author JPEG/MP4 chain','No FVD/VFID or formal efficiency comparison requested/executed'],'inference':'Official checkpoint, 8-frame windows, overlap4, batch2, num_steps1, seed0, full1024x576; unmodified author model code','timing_scope':'Generation/render/inference/metrics may overlap; not standardized efficiency scores'}
with a.report.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(report['mean'],indent=2))
