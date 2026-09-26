"""Full 25-frame streaming inference on each pilot scene; preserve float RGB/CoC encoding."""
import argparse,os,sys,json,csv,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--only',default='');p.add_argument('--preflight-only',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1];pilot=a.input.resolve()
sys.path.insert(0,str(root/'src'))
from synthetic.any2bokeh_input import discover,validate,check_batch
folders=discover(pilot,a.only)
checks=[validate(folder) for folder in folders]
os.environ.setdefault('HF_HOME',str(root/'models/huggingface'));sys.path.insert(0,str(root.parent/'any-to-bokeh'))
import torch
from models.unet import UNetSpatioTemporalConditionModel
from models.vae import AutoencoderKLTemporalDecoder
from pipelines.any2bokeh_pipe import StableVideoDiffusionPipeline
from utils.dataset import BokehCocMPIDatasetWholeVid,collate_fn_val_coc_mpi_whole_video
from PIL import Image
import numpy as np
torch.set_num_threads(4);torch.manual_seed(0)
if a.preflight_only:
 import tempfile
 for folder,check in zip(folders,checks):
  with tempfile.TemporaryDirectory(prefix='any2bokeh-preflight-') as temp:
   csvpath=Path(temp)/'input.csv'
   with csvpath.open('w') as f:w=csv.writer(f);w.writerow(['aif_folder','disp_folder','k']);w.writerow([folder/'rgb',folder/'disparity',check['K']])
   ds=BokehCocMPIDatasetWholeVid(str(csvpath),sample_size=(check['resolution'][1],check['resolution'][0]));batch=collate_fn_val_coc_mpi_whole_video([ds[0]])
   check['batch_shapes']=check_batch(ds,batch,folder)
   del batch
 with (pilot/'any2bokeh_preflight.json').open('x') as f:json.dump(checks,f,indent=2)
 print(json.dumps(checks),flush=True)
 sys.exit(0)
checkpoint=root/'models/any-to-bokeh/checkpoints'
unet=UNetSpatioTemporalConditionModel.from_pretrained(str(checkpoint),subfolder='unet',torch_dtype=torch.float16,low_cpu_mem_usage=False,local_files_only=True)
vae=AutoencoderKLTemporalDecoder.from_pretrained(str(checkpoint/'vae'),torch_dtype=torch.float16,local_files_only=True)
pipe=StableVideoDiffusionPipeline.from_pretrained('stabilityai/stable-video-diffusion-img2vid-xt',unet=unet,vae=vae,torch_dtype=torch.float16,variant='fp16',low_cpu_mem_usage=False,local_files_only=True).to('cuda:0');pipe.set_progress_bar_config(disable=True)
for folder,check in zip(folders,checks):
 meta=json.loads((folder/'metadata.json').read_text());c=meta['config'];out=folder/'pred_any2bokeh';out.mkdir(exist_ok=False)
 csvpath=out/'input.csv'
 with csvpath.open('w') as f: w=csv.writer(f);w.writerow(['aif_folder','disp_folder','k']);w.writerow([folder/'rgb',folder/'disparity',c['K']])
 ds=BokehCocMPIDatasetWholeVid(str(csvpath),sample_size=(c['height'],c['width']));batch=collate_fn_val_coc_mpi_whole_video([ds[0]])
 check['batch_shapes']=check_batch(ds,batch,folder)
 (out/'input_check.json').write_text(json.dumps(check,indent=2))
 cache=None;frames=[];times=[];torch.cuda.reset_peak_memory_stats()
 for i in range(0,batch['coc'].shape[0],2):
  inputs={k:batch[k][i:i+2].to('cuda',dtype=torch.float32 if k in ['pixel_values','coc'] else torch.float16) for k in ['pixel_values','coc','k','mpi_mask','weight']}
  last=i+2>=batch['coc'].shape[0];torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():res,cache=pipe(inputs['coc'],inputs['pixel_values'],input_k=inputs['k'],mpi_mask=inputs['mpi_mask'],weight=inputs['weight'],latent_cache=cache,is_last_batch=last,overlap_frame=4,height=c['height'],width=c['width'],num_frames=8,decode_chunk_size=8,motion_bucket_id=127,fps=7,noise_aug_strength=0.,num_inference_steps=1,generator=torch.Generator(device='cpu').manual_seed(0),batch_size=inputs['weight'].shape[0],output_type='pt')
  torch.cuda.synchronize();times.append(time.perf_counter()-start);assert torch.isfinite(res[0]).all();frames.extend(res[0].detach().cpu())
 assert len(frames)==len(meta['frames']),(folder,len(frames))
 for i,frame in enumerate(frames):Image.fromarray(np.uint8(np.round(frame.clamp(0,1).permute(1,2,0).numpy()*255))).save(out/f'{i:05d}.png')
 (out/'timing.json').write_text(json.dumps({'seconds':times,'total_seconds':sum(times),'frames':len(frames),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'note':'pilot only; not repeated standardized efficiency benchmark'},indent=2));print(folder.name,len(frames),sum(times),flush=True)
