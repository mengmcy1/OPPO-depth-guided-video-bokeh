"""Direct-supervision light fusion and the fixed O1/F2 render comparison."""
import argparse
import fcntl
import math
import time

from route3_g3_r14 import C, LOG, OUT, TRUTH, qa, read, write
from route3_g3_r14_data import BASE
import numpy as np
import torch
import route3_s2a_learn as legacy
from route3_s2a_learn import Store, evaluate_row
from route3_depth_interface import subject_mask
from route3_g3_geometry import aligned_hidden, fit_and_map, load, zero_parts
from route3_ldr_u1r import model as lpips
from route3_s2a_proxy import read as read_exr
from route3_upper_bound_evaluate import display, score
from src.aperture_renderer import GaussianScene
from src.dual_layer import depth_band
from src.hidden_retrieval_rules import aggregate
from src.layered_aperture import merge_layers, pixel_layer
from src.light_candidate_fusion import LightCandidateFusion
from src.light_candidate_supervision import objectives
from src.ordered_layers import order_hidden, ray_samples, render_fallbacks

SETTINGS = C['light_fusion']


def train_network(root, dest, scenes, monitors, single=False, augmentations=(),
                  balance_activation=False, require_descent=False):
    legacy.OUT = root
    store = Store()
    seed = SETTINGS['seeds']['overfit' if single else 'S2a']
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = LightCandidateFusion().cuda()
    optimizer = torch.optim.Adam(net.parameters(),lr=SETTINGS['learning_rate'])
    steps = SETTINGS['overfit_max_steps' if single else 'S2a_steps']
    manifests = {name:store.manifest(name) for name in scenes+monitors}

    def far(m):
        return next(r for r in m['rows'] if r['frame']==30 and r['case']['nominal_focus_m']==5 and r['case']['fstop']==1.4)

    def active(row):
        return store.active(row['input'],row['case']['focus_m'],row['case']['fstop'])

    first = manifests[scenes[0]]
    rows = [(name,r,m) for name in scenes for m in [manifests[name]]
            for r in ([far(m)] if single else m['rows']) if len(active(r))]
    if not single:
        for variant in augmentations:
            for name in scenes:
                m = read(variant/'learning-data'/name/'256-5/manifest.json')
                rows.extend((name,r,m) for r in m['rows'] if len(active(r)))
    if not rows:
        raise ValueError('No eligible queries for direct supervision')
    warmup = [] if single else [(scenes[0], r, m) for m in [store.manifest(scenes[0],'256-5-GTwarmup')]
                                for r in m['rows'] if len(active(r))]
    dest.mkdir(parents=True,exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    init = evaluate_row(net,store,far(first),first) if single else None
    probe = lambda: [evaluate_row(net,store,far(manifests[n]),manifests[n],
                      balance_activation=balance_activation)['loss'] for n in scenes]
    train_probe = [dict(step=0,losses=probe())] if require_descent else []
    curve, validation = [], []
    start = time.monotonic()
    period = SETTINGS['overfit_monitor_steps' if single else 'eval_every_steps']
    first_step = 1
    progress_file = dest/'progress.json'
    if progress_file.exists():
        resume = read(progress_file)
        step = resume['step']
        state = torch.load(dest/f'step-{step:06d}.pt',weights_only=True,map_location='cpu')
        assert scenes == state['scenes'] and seed == state['seed']
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        init, curve, validation = resume['initial'], resume['curve'], resume['validation']
        train_probe = resume.get('train_probe',train_probe)
        first_step = step+1
    for step in range(first_step,steps+1):
        eligible = warmup if warmup and step<=SETTINGS['GT_warmup_steps'] else rows
        name,row,m = eligible[int(rng.integers(len(eligible)))]
        indices = active(row)
        b = SETTINGS['batch_queries']
        indexes = rng.choice(indices,size=b,replace=len(indices)<b)
        decay = .5*(1+math.cos(math.pi*(step-1)/steps))
        optimizer.param_groups[0]['lr'] = SETTINGS['final_learning_rate'] + (SETTINGS['learning_rate']-SETTINGS['final_learning_rate'])*decay
        net.train()
        optimizer.zero_grad(set_to_none=True)
        pred,valid,_ = store.forward(net,row,indexes,m)
        loss,terms = objectives(pred,store.target_batch(row,indexes),valid,balance_activation)
        if not bool(torch.isfinite(loss)):
            write(dest/'optimization-failure.json',dict(step=step,reason='nonfinite direct loss'))
            raise RuntimeError('Nonfinite direct loss')
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
        except RuntimeError as error:
            write(dest/'optimization-failure.json',dict(step=step,reason=str(error)))
            raise
        optimizer.step()
        if step==1 or step%100==0:
            curve.append(dict(step=step,**{key:float(val.detach()) for key,val in terms.items()}))
            print('TRAIN',name,step,curve[-1]['total'],flush=True)
        if step%period==0 or step==steps:
            review = scenes if single else monitors
            validation.append(dict(step=step,rows=[dict(scene=n,**evaluate_row(net,store,far(manifests[n]),manifests[n],balance_activation=balance_activation)) for n in review]))
            if require_descent:
                train_probe.append(dict(step=step,losses=probe()))
            legacy.checkpoint(dest/f'step-{step:06d}.pt',net,optimizer,step,rng,dict(scenes=scenes,seed=seed))
            write(dest/'progress.json',dict(initial=init,curve=curve,validation=validation,train_probe=train_probe,step=step))
            if require_descent and step>=2*period:
                initial_loss=np.mean(train_probe[0]['losses'])
                if all(np.mean(row['losses'])>=initial_loss for row in train_probe[1:]):
                    write(dest/'optimization-failure.json',dict(step=step,reason='fixed training probes have not decreased from initialization',train_probe=train_probe))
                    raise RuntimeError('S2a fixed training probes did not decrease')
    final = validation[-1]['rows'][0] if single else None
    m = final['metrics'] if final else None
    passed = None if not single else bool(m and m['matched_heads'] and final['loss']<=.5*init['loss']
              and m['selection_correct_rate']>=.9 and m['matched_ray_relative_depth_L1']<=.05
              and m['matched_ray_linear_RGB_MAE']<=.05)
    write(dest/'training.json',dict(initial=init,final=final,passed=passed,curve=curve,validation=validation,
          trainable_parameters=sum(p.numel() for p in net.parameters()),contexts='256/5',
          scenes=scenes,seed=seed,seconds=time.monotonic()-start,steps=steps,augmented_roots=[str(p) for p in augmentations],
          peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,GT_warmup_steps=0 if single else SETTINGS['GT_warmup_steps'],
          balance_activation=balance_activation,train_probe=train_probe))
    return net


def learned_hidden(net,store,row,manifest,K,band):
    c = row['case']
    ix = store.active(row['input'],c['focus_m'],c['fstop'])
    parts = {k:[] for k in ['means_camera','colors','opacities']}
    with torch.inference_mode():
        for start in range(0,len(ix),2048):
            p,_,_ = store.forward(net,row,ix[start:start+2048],manifest)
            for k in parts:
                parts[k].append(p[k].reshape(-1,3) if k!='opacities' else p[k].reshape(-1))
    if not len(ix):
        return None
    xyz,rgb,opacity = [torch.cat(parts[k]) for k in parts]
    projected = xyz@K.T
    uv = projected[:,:2]/projected[:,2:].clamp_min(1e-8)
    pixel = uv.floor().long().clamp(0,511)
    valid = (xyz[:,2]>0)&(uv>=0).all(-1)&(uv<512).all(-1)&(opacity>1e-4)
    valid &= torch.as_tensor(band,device='cuda')[pixel[:,1],pixel[:,0]]
    xyz,rgb,opacity = xyz[valid],rgb[valid],opacity[valid]
    if not len(xyz):
        return None
    sigma = xyz[:,2]/K[0,0]*.7
    scale = torch.stack([sigma,sigma,sigma*.01],dim=-1)
    quat = torch.zeros((len(xyz),4),device='cuda')
    quat[:,0] = 1
    return GaussianScene(xyz,quat,scale,opacity,rgb,torch.ones_like(opacity,dtype=torch.long))


def render_scene(name,net,root,dest,single=False):
    from route3_s2a_baselines import scene
    from route3_b2_oracle_3d import sparse_pixel_layer
    from route3_local_replacement import PROFILE
    from route3_s2a_evaluate import masks
    legacy.OUT = root
    store = Store()
    manifest = store.manifest(name,'512-13')
    by_id = {r['case']['id']:r for r in manifest['rows']}
    folder = root/name
    results = []
    metric = lpips()
    eye = torch.eye(4,device='cuda')
    tensor = lambda x:torch.as_tensor(x,device='cuda')
    for entry in read(folder/'prepared.json')['rows']:
        f = entry['frame']
        cases = [c for c in entry['cases'] if not single or (f==30 and c['nominal_focus_m']==5 and c['fstop']==1.4)]
        if not cases:
            continue
        d = load(entry['input_npz'])
        depth,K = tensor(d['depth']),tensor(d['K'])
        visible = pixel_layer(tensor(d['rgb_linear']),depth,K,eye)
        a_src = BASE/name
        a_entry = next(e for e in read(a_src/'prepared.json')['rows'] if e['frame']==f)
        a_data = load(a_entry['input_npz'])
        a_maps = load(a_src/'A-depth'/f'f{f:03d}.npz')
        A = scene(a_src/'visible'/f'f{f:03d}.pt')
        fill = tensor(load(a_entry['fill'])['fill_rgb_linear'])
        bg = tensor(a_data['background_depth'])
        parents = load(folder/'oracle'/f'f{f:03d}.npz')
        g = load(folder/'geometry'/f'f{f:03d}-center.npz')
        fg = np.load(folder/'scene-geometry.npz')['foreground_object_ids']
        aa, bb, supported, mapped, valid, _ = fit_and_map(g,d['depth'],fg,parents)
        align = dict(parent_depth_aligned=mapped,parent_valid=valid,local_supported=supported)
        p = load(entry['pool'])
        retrieval,_ = aggregate(p,2,a_maps['depth'])
        dom = (a_maps['cluster_fraction']>=.9)&(a_maps['contribution_alpha']>1e-6)&(a_maps['cluster_depth']>0)
        with torch.inference_mode():
            r,a = zero_parts(A,K)
            a_zero = r+(1-a[...,None])*fill
        for c in cases:
            cid = c['id']
            out = dest/cid
            out.mkdir(parents=True,exist_ok=True)
            focus,aperture = c['focus_m'],.05/(2*c['fstop'])
            band = depth_band(d['depth'],d['K'][0,0],focus,aperture)[0]
            if c['nominal_focus_m']==5:
                nominal = 5.
            else:
                nominal = c['nominal_focus_m']
            good = subject_mask(g,nominal)&dom
            if not good.any():
                raise ValueError(f'A P2 subject unavailable for {name} {cid}')
            focus_a = float(np.median(a_maps['cluster_depth'][good]))
            h,_ = aligned_hidden(parents,align,band,K)
            h,_ = order_hidden(h,depth,K,'O1')
            learned = learned_hidden(net,store,by_id[cid],manifest,K,band)
            learned,_ = order_hidden(learned,depth,K,'O1')
            uv = retrieval['uv'].astype(int).clip(0,511)
            a_band = depth_band(a_maps['depth'],d['K'][0,0],focus_a,aperture)[0]
            select = a_band[uv[:,1],uv[:,0]]
            ret = sparse_pixel_layer(retrieval,select,K,eye,PROFILE,'cuda') if select.any() else None
            with torch.inference_mode():
                samples = ray_samples(A,K,focus,aperture,64,fill,bg)
                a_samples = samples if focus==focus_a else ray_samples(A,K,focus_a,aperture,64,fill,bg)
            predictions = dict(A=a_samples.mean(0).cpu().numpy())
            zeros = dict(A=a_zero.cpu().numpy())
            render_groups = [('D0',visible,None,focus,samples),('D1_Oracle',visible,h,focus,samples),
                             ('learned',visible,learned,focus,samples),('R1_R2',A,ret,focus_a,a_samples)]
            for label,vis,hid,zf,back in render_groups:
                with torch.inference_mode():
                    cloud = merge_layers(vis,hid)
                    p = render_fallbacks(cloud,K,zf,aperture,64,fill,bg,back)
                    r,a = zero_parts(cloud,K)
                    if label=='R1_R2':
                        predictions[label] = np.where(a_band[...,None],p['F1'].cpu().numpy(),predictions['A'])
                        zeros[label] = np.where(a_band[...,None],(r+(1-a[...,None])*fill).cpu().numpy(),zeros['A'])
                    else:
                        predictions[label] = p['F2'].cpu().numpy()
                        zeros[label] = (r+(1-a[...,None])*a_zero).cpu().numpy()
            truth = display(read_exr(TRUTH/name/'dof'/(cid+'.exr')))
            regions=masks(TRUTH/name,dict(c,frame=f))
            all_regions={k:regions[k] for k in ['full_image','in_focus','band','recoverable']}
            all_regions['invalid_geometry']=~load(folder/'models/calibration/SHARP'/f'f{f:03d}.npz')['dominant_valid']
            for label,image in predictions.items():
                pred = display(image)
                zero = display(zeros[label])
                np.savez_compressed(out/(label+'.npz'),prediction_srgb=pred,zero_srgb=zero)
                zr = float(-10*np.log10(max(np.mean((zero.astype(np.float64)-d['rgb_srgb'])**2),1e-30)))
                for region,r in score(pred,truth,all_regions,metric).items():
                    results.append(dict(scene=name,frame=f,id=cid,method=label,region=region,zero_PSNR=zr,
                                        focus_m=focus_a if label in ['A','R1_R2'] else focus,**r))
            write(dest/'render.json',dict(rows=results,K=64,cycles=True))
            print('TRAIN_RENDER',name,cid,flush=True)
    return results


def run_single(name,stage):
    dest = OUT/'overfit'/name
    if stage=='fit':
        train_network(BASE,dest,[name],[],single=True)
    elif stage=='render':
        net = LightCandidateFusion().cuda()
        state = torch.load(dest/f'step-{SETTINGS["overfit_max_steps"]:06d}.pt',weights_only=True)
        net.load_state_dict(state['model'])
        render_scene(name,net,BASE,dest,single=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('scene')
    p.add_argument('stage',choices=['fit','render'])
    p.add_argument('--overlap-fit',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4)
    if a.overlap_fit and a.stage!='fit':
        p.error('--overlap-fit only applies to minibatch fitting')
    lock_path = LOG/('gpu0-fit.lock' if a.overlap_fit else 'gpu0.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        qa.guard(True)
        if a.overlap_fit:
            free,_ = torch.cuda.mem_get_info()
            if free < 12*2**30:
                raise RuntimeError('Minibatch fit GPU budget no longer available')
            torch.cuda.set_per_process_memory_fraction(.15)
        run_single(a.scene,a.stage)
