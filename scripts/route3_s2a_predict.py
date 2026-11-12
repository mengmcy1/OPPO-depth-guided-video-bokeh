"""Final S2a model at512/13, same fixed K64/lens/fill as the baselines.

GT hidden content and scoring masks are absent from the prediction path.
Original B2/pilot visible clouds and baseline predictions are read-only.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_s2a_screen import OUT,RECORDS,write
from route3_s2a_data import descriptor,save_npz
from route3_s2a_learn import Store
from src.light_candidate_fusion import LightCandidateFusion
from src.aperture_renderer import GaussianScene
from src.layered_aperture import merge_layers
from src.analytic_aperture import sampled
from src.dual_layer import depth_band
from route3_sharp_s1 import raster
from route3_upper_bound_evaluate import display
CONFIG=ROOT/'configs/route3-network-s2a-2026-11-12.json'
PILOT=ROOT/'outputs/route3-network-pilot-2026-11-12-01'


def baseline_path(name,cid):
    if name in ['B2-high','pilot-thin','pilot-multilayer']:
        r=json.loads((RECORDS/'route3-network-pilot-2026-11-12-baseline-Cycles.json').read_text());return r['predictions'][name][cid]['A_color']
    return str(OUT/name/'predictions'/(cid+'-A_color.npz'))


def visible_cloud(desc,entry):
    if desc['source_root'] is None:
        from route3_sharp_s1c_apply import anchored
        return anchored(entry,'A_color')
    return GaussianScene(**torch.load(desc['source_root']/'visible'/f'f{entry["frame"]:03d}.pt',map_location='cuda',weights_only=True))


def predict(name,net,store):
    desc=descriptor(name);dest=OUT/'inference'/name;dest.mkdir(parents=True,exist_ok=True);record=dest/'predictions.json';rows=json.loads(record.read_text())['rows'] if record.exists() else [];done={r['id'] for r in rows};manifest=store.manifest(name,'512-13')
    for entry in desc['entries']:
        cloud=visible_cloud(desc,entry);d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda',dtype=torch.float32);E=torch.eye(4,device='cuda');fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda')
        with torch.no_grad():_,z,alpha=raster(cloud,K)
        own=np.where(alpha.cpu().numpy()>=.1,z.cpu().numpy(),d['depth'])
        for c in entry['cases']:
            if c['id'] in done:continue
            row=next(r for r in manifest['rows'] if r['case']['id']==c['id']);indices=store.active(row['input'],c['focus_m'],c['fstop']);npinput=store.inputs(row['input']);parts={k:[] for k in ['means_camera','colors','opacities','q','weights']};torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for off in range(0,len(indices),2048):
                    ix=indices[off:off+2048];p,_,_=store.forward(net,row,ix,manifest)
                    for key in ['means_camera','colors','opacities']:parts[key].append(p[key].cpu().numpy())
                    parts['q'].append(p['q_logits'].sigmoid().cpu().numpy());parts['weights'].append(p['weights'].cpu().numpy())
            torch.cuda.synchronize();neural_seconds=time.monotonic()-start
            if len(indices):values={k:np.concatenate(v) for k,v in parts.items()};means=values['means_camera'].reshape(-1,3);colors=values['colors'].reshape(-1,3);opacity=values['opacities'].reshape(-1)
            else:values={};means=np.empty((0,3),np.float32);colors=np.empty((0,3),np.float32);opacity=np.empty(0,np.float32)
            projected=means@d['K'].T;uv=projected[:,:2]/np.maximum(projected[:,2:],1e-8);x=np.clip(np.floor(uv[:,0]).astype(int),0,511);y=np.clip(np.floor(uv[:,1]).astype(int),0,511);inside=(means[:,2]>1e-5)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<512);front=means[:,2]<=own[y,x]+np.maximum(.005,.02*own[y,x]);keep=inside&~front&np.isfinite(means).all(-1)&np.isfinite(colors).all(-1)&(opacity>1e-4)
            # Existing R1 center-view depth consistency, after learned correction.
            # Confidence q never multiplies alpha. No GT is used in this filter.
            hid=None
            if keep.any():
                xyz=torch.tensor(means[keep],device='cuda');sig=xyz[:,2]/K[0,0]*.7;scales=torch.stack([sig,sig,sig*.01],-1);quat=torch.zeros((len(xyz),4),device='cuda');quat[:,0]=1;hid=GaussianScene(xyz,quat,scales,torch.tensor(opacity[keep],device='cuda'),torch.tensor(colors[keep],device='cuda'),torch.ones(len(xyz),device='cuda',dtype=torch.long))
            band=depth_band(d['depth'],d['K'][0,0],c['focus_m'],.05/(2*c['fstop']))[0];base=np.load(baseline_path(name,c['id']));base_alpha=base['alpha'] if 'alpha' in base.files else None
            if base_alpha is None:
                with torch.no_grad():base_render=sampled(cloud,K,E,512,512,c['focus_m'],.05/(2*c['fstop']),64,fill,bg,far_plane=1e5)
                base_alpha=base_render['alpha'].cpu().numpy();del base_render
            torch.cuda.synchronize();start=time.monotonic()
            if hid is not None:
                with torch.no_grad():p=sampled(merge_layers(cloud,hid),K,E,512,512,c['focus_m'],.05/(2*c['fstop']),64,fill,bg,far_plane=1e5,return_fill_contribution=True)
                rgb=np.where(band[...,None],p['rgb_linear'].cpu().numpy(),base['rgb_linear']);a=np.where(band,p['alpha'].cpu().numpy(),base_alpha);del p
            else:rgb=base['rgb_linear'];a=base_alpha
            torch.cuda.synchronize();render_seconds=time.monotonic()-start;path=dest/(c['id']+'-LightFusion.npz');save_npz(path,dict(prediction_srgb=display(rgb),rgb_linear=rgb,alpha=a));parent_path=dest/(c['id']+'-parents.npz')
            if len(indices):save_npz(parent_path,dict(means_camera=values['means_camera'],colors_linear=values['colors'],opacities=values['opacities'],input_query_indices=indices,source_weights=values['weights'].astype(np.float16),q=values['q'].astype(np.float16),render_keep=keep.reshape(-1,2)))
            rows.append(dict(**c,scene=name,path=str(path),method='LightFusion',neural_seconds=neural_seconds,render_seconds=render_seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,queried_positions=len(indices),kept_hidden_Gaussians=int(keep.sum()),front_filtered_Gaussians=int((front&inside&(opacity>.1)).sum()),active_predicted_Gaussians=int((opacity>.1).sum()),source_input=str(row['input']),parent_outputs=str(parent_path),source_RGB_reference_count=13,training_only_256_5=True));write(record,dict(rows=rows,K=64,query_cap=262144,forward_chunk=2048,hidden_per_query=2,sigma_pixels=.7,q_multiplies_alpha=False,post_visibility_filter='same S1dR1 ownrenderdepth+max5mm2%;input-only',GT_hidden_not_read=True,source_provenance='candidate refs in input;source_weights+q savedfp16 fordiagnostic;means/colors/opacity float32;Gaussian source_frames remains layercode1'))
            print('S2A_PREDICT',name,c['id'],int(keep.sum()),flush=True);del hid
        del cloud;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Explicit GPU0 required')
    subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;setting=json.loads(CONFIG.read_text())['light_fusion'];r=json.loads((RECORDS/'route3-network-s2a-2026-11-12-S2a.json').read_text());assert r['completed'] and r['steps']==setting['S2a_steps'];state=torch.load(r['checkpoint'],map_location='cuda',weights_only=True);net=LightCandidateFusion(setting['encoder_width'],setting['source_axial_depth_delta_bound_fraction'],setting['linear_RGB_residual_bound']).cuda();net.load_state_dict(state['model']);net.eval();store=Store();split=json.loads((OUT/'split.json').read_text());names=[a.scene] if a.scene else split['heldout']+['B2-high','pilot-thin','pilot-multilayer']
    for name in names:predict(name,net,store)

if __name__=='__main__':main()
