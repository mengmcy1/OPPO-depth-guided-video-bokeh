"""Single-scene fit then S2a. Only the light fusion module is optimized.

No Cycles image, LDR reference or scoring-region mask is read by training.
The final fixed step is used; heldout monitoring never selects checkpoints.
"""
import argparse,functools,json,math,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.light_candidate_fusion import LightCandidateFusion
from src.light_candidate_supervision import pack_targets,objectives,direct_metrics,summarize
from src.dual_layer import depth_band
from route3_s2a_screen import OUT,RECORDS,write
CONFIG=ROOT/'configs/route3-network-s2a-2026-11-12.json'

class Store:
    def __init__(self):self.contexts={}
    @functools.lru_cache(None)
    def manifest(self,scene,kind='256-5'):
        return json.loads((OUT/'learning-data'/scene/kind/'manifest.json').read_text())
    @functools.lru_cache(None)
    def inputs(self,path):
        with np.load(path) as f:return {k:f[k] for k in f.files}
    @functools.lru_cache(None)
    def targets(self,path,input_path):
        with np.load(path) as f:return pack_targets(f,self.inputs(input_path))
    @functools.lru_cache(None)
    def active(self,input_path,focus,fstop):
        d=self.inputs(input_path);mask=depth_band(d['target_depth'],d['target_K'][0,0],focus,.05/(2*fstop))[0];return np.flatnonzero(mask.ravel()[d['query_pixel']])
    def context(self,manifest):
        path=manifest['context']
        if path not in self.contexts:
            with np.load(path) as d:self.contexts[path]=dict(images=torch.tensor(d['images'],device='cuda').permute(0,3,1,2).contiguous(),K=torch.tensor(d['K'],device='cuda',dtype=torch.float32),c2w=torch.tensor(d['c2w'],device='cuda',dtype=torch.float32))
        return self.contexts[path]
    def forward(self,net,row,indices,manifest):
        d=self.inputs(row['input']);c=row['case'];size=manifest['size'];ctx=self.context(manifest);pix=d['query_pixel'][indices];condition=np.c_[d['query_xy'][indices]/size,np.full(len(indices),2/c['focus_m']),np.full(len(indices),50*.05/(2*c['fstop'])),d['target_depth'].ravel()[pix]/5,np.full(len(indices),row['frame']/60)].astype(np.float32)
        tensor=lambda key,dtype:torch.tensor(d[key][indices],device='cuda',dtype=dtype)
        p=net(tensor('features',torch.float32),tensor('candidate_valid',torch.bool),tensor('source_slots',torch.long),tensor('source_xy',torch.float32),tensor('source_depth',torch.float32),torch.tensor(condition,device='cuda'),ctx,torch.tensor(d['target_c2w'],device='cuda',dtype=torch.float32))
        return p,tensor('candidate_valid',torch.bool),ctx
    def target_batch(self,row,indices):
        raw=self.targets(row['target'],row['input']);return {k:torch.tensor(v[indices],device='cuda',dtype=torch.bool if v.dtype==bool else torch.float32) for k,v in raw.items()}


def evaluate_row(net,store,row,manifest,batch=2048):
    indices=store.active(row['input'],row['case']['focus_m'],row['case']['fstop']);stats=None;loss_sum=0.;net.eval()
    with torch.no_grad():
        for offset in range(0,len(indices),batch):
            ids=indices[offset:offset+batch];p,valid,ctx=store.forward(net,row,ids,manifest);t=store.target_batch(row,ids);loss,_=objectives(p,t,valid);v=direct_metrics(p,t,valid,ctx['K']);stats=v if stats is None else {k:stats[k]+v[k] for k in v};loss_sum+=float(loss)*len(ids)
    return dict(loss=loss_sum/max(len(indices),1),metrics=summarize(stats) if stats is not None else None,active_queries=len(indices))


def checkpoint(path,net,opt,step,rng,metadata):
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+'.part');torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),**metadata),temp);temp.replace(path)


def monitor_record(scene,row,value):return dict(scene=scene,id=row['case']['id'],**value)


def train(mode):
    if not (OUT/'formal-data-ready.json').exists() or not (OUT/'derivative-data-ready.json').exists():raise RuntimeError('Formal data and derived5-frame supervision must complete before training')
    if (OUT/'STOP.json').exists():raise RuntimeError('A prior stop condition is pending')
    cfg=json.loads(CONFIG.read_text());split=json.loads((OUT/'split.json').read_text());assert len(split['train'])==12 and len(split['heldout'])==4 and not set(split['train'])&set(split['heldout']);assert not set(split['train'])&{'B2-high','pilot-thin','pilot-multilayer'}
    setting=cfg['light_fusion'];record=RECORDS/f'route3-network-s2a-2026-11-12-{mode}.json';dest=OUT/mode;dest.mkdir(exist_ok=True)
    if record.exists() and json.loads(record.read_text()).get('completed'):return
    if mode=='S2a':
        previous=json.loads((RECORDS/'route3-network-s2a-2026-11-12-overfit.json').read_text())
        if not previous.get('passed'):raise RuntimeError('Single-scene fit did not pass;do not enlarge training')
    seed=setting['seeds'][mode];torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed);torch.cuda.reset_peak_memory_stats();net=LightCandidateFusion(setting['encoder_width'],setting['source_axial_depth_delta_bound_fraction'],setting['linear_RGB_residual_bound']).cuda();opt=torch.optim.Adam(net.parameters(),lr=setting['learning_rate']);steps=setting['overfit_max_steps'] if mode=='overfit' else setting['S2a_steps'];store=Store();first=split['train'][0];manifest=store.manifest(first);row=next(r for r in manifest['rows'] if r['frame']==30 and r['case']['nominal_focus_m']==5 and r['case']['fstop']==1.4);initial=evaluate_row(net,store,row,manifest) if mode=='overfit' else None;curve=[];validation=[];start=time.monotonic();first_step=0
    meta=dict(mode=mode,config=cfg,split=split,seed=seed,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),trainable_parameters=sum(p.numel() for p in net.parameters()),network_pretrained=False)
    snapshots=sorted(dest.glob('step-*.pt'))
    if snapshots:
        state=torch.load(snapshots[-1],map_location='cuda',weights_only=True);assert state['mode']==mode and state['seed']==seed and state['split']['train']==split['train'];net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);first_step=state['step'];rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        if (dest/'progress.json').exists():old=json.loads((dest/'progress.json').read_text());curve=old['curve'];validation=old['validation'];initial=old.get('initial',initial)
    for step in range(first_step+1,steps+1):
        if mode=='overfit':current_manifest=manifest;current=row;scene=first
        elif step<=setting['GT_warmup_steps']:
            current_manifest=store.manifest(first,'256-5-GTwarmup');current=current_manifest['rows'][int(rng.integers(len(current_manifest['rows'])))];scene=first
        else:
            scene=split['train'][int(rng.integers(12))];current_manifest=store.manifest(scene);current=current_manifest['rows'][int(rng.integers(len(current_manifest['rows'])))];
        active=store.active(current['input'],current['case']['focus_m'],current['case']['fstop'])
        if len(active)==0:raise RuntimeError('No active query in training case '+scene+current['case']['id'])
        indices=rng.choice(active,size=setting['batch_queries'],replace=len(active)<setting['batch_queries']);net.train();opt.param_groups[0]['lr']=setting['final_learning_rate']+(setting['learning_rate']-setting['final_learning_rate'])*.5*(1+math.cos(math.pi*(step-1)/steps));opt.zero_grad(set_to_none=True);p,valid,ctx=store.forward(net,current,indices,current_manifest);t=store.target_batch(current,indices);loss,terms=objectives(p,t,valid)
        if not bool(torch.isfinite(loss)):write(OUT/'STOP.json',dict(reason='nonfinite optimization',mode=mode,step=step));raise RuntimeError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%100==0:
            values={k:float(v.detach()) for k,v in terms.items()};curve.append(dict(step=step,scene=scene,id=current['case']['id'],GT_geometry_warmup=(mode=='S2a' and step<=setting['GT_warmup_steps']),seconds=time.monotonic()-start,lr=opt.param_groups[0]['lr'],**values));print('LIGHT_TRAIN',mode,step,values,flush=True)
        frequency=setting['overfit_monitor_steps'] if mode=='overfit' else setting['eval_every_steps']
        if step%frequency==0 or step==steps:
            if mode=='overfit':validation.append(dict(step=step,rows=[monitor_record(first,row,evaluate_row(net,store,row,manifest))]))
            else:
                monitors=[]
                for name in split['heldout']:
                    mm=store.manifest(name);rr=next(r for r in mm['rows'] if r['frame']==30 and r['case']['nominal_focus_m']==5 and r['case']['fstop']==1.4);monitors.append(monitor_record(name,rr,evaluate_row(net,store,rr,mm)))
                validation.append(dict(step=step,rows=monitors))
            checkpoint(dest/f'step-{step:06d}.pt',net,opt,step,rng,meta);write(dest/'progress.json',dict(initial=initial,curve=curve,validation=validation,step=step,checkpoints_for_resume_only=True,heldout_not_used_for_checkpoint_selection=True))
    final=evaluate_row(net,store,row,manifest) if mode=='overfit' else None;passed=None
    if mode=='overfit':
        m=final['metrics'];passed=bool(m is not None and m['matched_heads']>0 and final['loss']<=.5*initial['loss'] and m['selection_correct_rate']>=.9 and m['matched_ray_relative_depth_L1']<=.05 and m['matched_ray_linear_RGB_MAE']<=.05)
    result=dict(completed=True,passed=passed,initial=initial,final=final,curve=curve,validation=validation,steps=steps,checkpoint=str(dest/f'step-{steps:06d}.pt'),seconds=time.monotonic()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,environment=dict(python=platform.python_version(),torch=str(torch.__version__),GPU=torch.cuda.get_device_name(0),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']),training_data_only=split['train'],GT_geometry_steps=0 if mode=='overfit' else setting['GT_warmup_steps'],GT_Cycles_or_LDR_reference_or_score_masks_read=False,overfit_scope='full active queries of first training scene frame30 farfocus f1.4',overfit_engineering_check='loss <= halfinitial;selected-GT-correct>=90%;matched-ray relative depth and linearRGBL1 <=.05;not a paper success gate',**meta);write(record,result)
    if mode=='overfit' and not passed:write(OUT/'STOP.json',dict(reason='single-scene cannot fit under recorded engineering check',record=str(record)));return


def evaluate_direct():
    setting=json.loads(CONFIG.read_text())['light_fusion'];state=torch.load(OUT/'S2a'/f'step-{setting["S2a_steps"]:06d}.pt',map_location='cuda',weights_only=True);net=LightCandidateFusion(setting['encoder_width'],setting['source_axial_depth_delta_bound_fraction'],setting['linear_RGB_residual_bound']).cuda();net.load_state_dict(state['model']);store=Store();split=json.loads((OUT/'split.json').read_text());rows=[]
    for scene in split['heldout']+['B2-high','pilot-thin','pilot-multilayer']:
        (OUT/'inference'/scene).mkdir(parents=True,exist_ok=True);manifest=store.manifest(scene);summed=None
        for row in manifest['rows']:
            if row['case'].get('training_focus_only'):continue
            value=evaluate_row(net,store,row,manifest);d=store.inputs(row['input']);rows.append(dict(scene=scene,id=row['case']['id'],target_calibration_applied=bool(d['target_calibration_applied']),**value))
            if value['metrics'] is None:continue
            stats={k:v for k,v in value['metrics'].items() if k in ['TP','FP','FN','TN','selected_correct','matched_heads','matched_ray_weight','weighted_relative_depth_sum','weighted_linear_RGB_MAE_sum','weighted_placement_pixel_error_sum','false_activated_queries','unneeded_queries','activated_needed_queries','needed_queries','candidate_correct_available_queries','queries']};summed=stats if summed is None else {k:summed[k]+stats[k] for k in stats}
        write(OUT/'inference'/scene/'direct-metrics.json',dict(scene=scene,pooled=summarize(summed) if summed is not None else None,rows=[r for r in rows if r['scene']==scene],configuration='256/5;GT matched-ray labels;qualityflags preserved'))
    write(RECORDS/'route3-network-s2a-2026-11-12-direct-evaluation.json',dict(rows=rows,configuration='256/5;only final fixed checkpoint',GT_poses_E1=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['overfit','S2a','direct'],required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Explicit allocated GPU0 required')
    subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.mode=='direct':evaluate_direct()
    else:train(a.mode)

if __name__=='__main__':main()
