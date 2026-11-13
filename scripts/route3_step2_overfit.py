"""User-authorized O only: fixed single-scene minibatch, no heldout split gate."""
import json,sys,time,subprocess,math,functools
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_step2_diagnostic import C,OUT,REC,PREFIX,write,save
from route3_s2a_learn import Store,evaluate_row
from src.light_candidate_fusion import LightCandidateFusion
from src.light_candidate_supervision import objectives,direct_metrics,summarize
from route3_s2a_data import descriptor,input_cache_512
from route3_s2a_predict import visible_cloud,baseline_path
from route3_sharp_s1 import raster
from src.aperture_renderer import GaussianScene
from src.layered_aperture import merge_layers
from src.analytic_aperture import sampled
from src.dual_layer import depth_band
from route3_s2a_proxy import display

class OStore(Store):
    @functools.lru_cache(None)
    def manifest(self,scene,kind='256-5'):
        if kind=='512-13':return json.loads((OUT/'cache'/scene/kind/'manifest.json').read_text())
        return super().manifest(scene,kind)

def evaluate_subset(net,store,row,manifest,indices):
    net.eval()
    with torch.no_grad():
        p,v,ctx=store.forward(net,row,indices,manifest);t=store.target_batch(row,indices);loss,terms=objectives(p,t,v);metrics=summarize(direct_metrics(p,t,v,ctx['K']))
    return dict(loss=float(loss),terms={k:float(v) for k,v in terms.items()},metrics=metrics,queries=len(indices))

def render_case(net,store,name):
    import route3_s2a_data as data
    desc=descriptor(name);folder=OUT/'cache'/name/'512-13';folder.mkdir(parents=True,exist_ok=True)
    if not(folder/'manifest.json').exists():input_cache_512(name,desc,folder)
    manifest=store.manifest(name,'512-13');entry=next(e for e in desc['entries'] if e['frame']==30);d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');E=torch.eye(4,device='cuda');cloud=visible_cloud(desc,entry);fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda');rows=[]
    with torch.no_grad():_,depth,alpha=raster(cloud,K)
    own=np.where(alpha.cpu().numpy()>=.1,depth.cpu().numpy(),d['depth'])
    for case in entry['cases']:
        if case['fstop']!=1.4:continue
        row=next(r for r in manifest['rows'] if r['case']['id']==case['id']);indices=store.active(row['input'],case['focus_m'],case['fstop']);parts={k:[] for k in ['means_camera','colors','opacities']}
        with torch.no_grad():
            for begin in range(0,len(indices),2048):
                p,_,_=store.forward(net,row,indices[begin:begin+2048],manifest)
                for key in parts:parts[key].append(p[key].cpu().numpy())
        val={k:np.concatenate(v).reshape(-1,3 if k!='opacities' else 1) for k,v in parts.items()};means=val['means_camera'];colors=val['colors'];opacity=val['opacities'].ravel();p=means@d['K'].T;uv=p[:,:2]/np.maximum(p[:,2:],1e-8);xy=uv.astype(int);x=np.clip(xy[:,0],0,511);y=np.clip(xy[:,1],0,511);inside=(means[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<512);keep=inside&(means[:,2]>own[y,x]+np.maximum(.005,.02*own[y,x]))&(opacity>1e-4)&np.isfinite(means).all(-1)
        xyz=torch.tensor(means[keep],device='cuda');sig=xyz[:,2]/K[0,0]*.7;quat=torch.zeros(len(xyz),4,device='cuda');quat[:,0]=1;hidden=GaussianScene(xyz,quat,torch.stack([sig,sig,sig*.01],-1),torch.tensor(opacity[keep],device='cuda'),torch.tensor(colors[keep],device='cuda'),torch.ones(len(xyz),device='cuda',dtype=torch.long));band=depth_band(d['depth'],K[0,0].item(),case['focus_m'],.05/2.8)[0]
        with torch.no_grad():result=sampled(merge_layers(cloud,hidden),K,E,512,512,case['focus_m'],.05/2.8,64,fill,bg,far_plane=1e5)
        base=np.load(baseline_path(name,case['id']));rgb=np.where(band[...,None],result['rgb_linear'].cpu().numpy(),base['rgb_linear']);path=OUT/'overfit'/('render-'+case['id']+'.npz');save(path,prediction_srgb=display(rgb),rgb_linear=rgb);rows.append(dict(**case,path=str(path),training_resolution=256,training_context=5,inference_resolution=512,inference_context=13,rendered_hidden_gaussians=int(keep.sum()),scope='descriptive;only fixed1024 queries/frame30/farfocus optimized'))
    write(OUT/'overfit/render.json',dict(rows=rows));return rows

def main():
    subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(4);cfg=C['O'];folder=OUT/'overfit';folder.mkdir(exist_ok=True);assert not(folder/'final.pt').exists();torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);name=cfg['scene'];store=OStore();manifest=store.manifest(name);row=next(r for r in manifest['rows'] if r['frame']==30 and r['case']['nominal_focus_m']==5. and r['case']['fstop']==1.4);active=store.active(row['input'],row['case']['focus_m'],1.4);indices=rng.choice(active,min(cfg['fixed_training_queries'],len(active)),replace=False);np.save(folder/'fixed-training-query-indices.npy',indices);net=LightCandidateFusion().cuda();opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate']);initial=evaluate_subset(net,store,row,manifest,indices);curve=[];monitors=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats();stopped=None
    for step in range(1,cfg['steps']+1):
        net.train();opt.param_groups[0]['lr']=cfg['final_learning_rate']+(cfg['learning_rate']-cfg['final_learning_rate'])*.5*(1+math.cos(math.pi*(step-1)/cfg['steps']));opt.zero_grad(set_to_none=True);p,v,_=store.forward(net,row,indices,manifest);t=store.target_batch(row,indices);loss,terms=objectives(p,t,v)
        if not bool(torch.isfinite(loss)):stopped='nonfinite loss';break
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step==1 or step%100==0:curve.append(dict(step=step,seconds=time.monotonic()-start,**{k:float(v.detach()) for k,v in terms.items()}));print('O_TRAIN',step,curve[-1],flush=True)
        if step%1000==0:monitors.append(dict(step=step,**evaluate_subset(net,store,row,manifest,indices)));write(folder/'progress.json',dict(initial=initial,curve=curve,monitors=monitors))
    final=evaluate_subset(net,store,row,manifest,indices);m=final['metrics'];passed=stopped is None and final['loss']<=.5*initial['loss'] and m['matched_heads']>0 and m['selection_correct_rate']>=.9 and m['matched_ray_relative_depth_L1']<=.05 and m['matched_ray_linear_RGB_MAE']<=.05
    torch.save(dict(model=net.state_dict(),steps=step,config=cfg,source_manifest=manifest,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),folder/'final.pt');result=dict(completed=stopped is None,passed=bool(passed),stop_reason=stopped,steps=step,initial=initial,final=final,full_query_diagnostic=evaluate_row(net,store,row,manifest),curve=curve,monitors=monitors,seconds=time.monotonic()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,config=cfg,checkpoint=str(folder/'final.pt'),method_conclusion=False,no_heldout_evaluation=True,no_model_or_configuration_selection=True,GT_only_direct_training_labels=True,GPU_overlap_with_D_recorded=True)
    write(folder/'result.json',result);write(REC/(PREFIX+'-O.json'),result)
    # Rendering remains part of the explicitly requested O report, even if
    # its engineering fit fails. No broader training or generator follows.
    render_case(net,store,name)
    if not passed:write(OUT/'STOP-O.json',dict(reason='single-scene fixed-minibatch engineering fit failed',result=str(folder/'result.json')))
    print('O_COMPLETE',passed,final,flush=True)

if __name__=='__main__':main()
