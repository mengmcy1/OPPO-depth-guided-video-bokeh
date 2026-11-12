"""Heldout-only LDR reference and unchanged C2 ProPainter layer baseline.

Two independent Blender job lists can share GPU0 with recorded concurrency;
no training or method scoring is launched here.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_s2a_screen import OUT,RECORDS,BLENDER,PY,LOG,run,write
from route3_pilot_plan import disk

def ldr():
    split=json.loads((OUT/'split.json').read_text());jobs=[]
    for name in split['heldout']:
        s=OUT/name
        for entry in json.loads((s/'prepared.json').read_text())['rows']:
            for case in entry['cases']:
                for tag,seed in [('main',20261112),('noise',20261113)]:
                    if tag=='noise' and not(entry['frame']==30 and case['fstop']==1.4):continue
                    for k,xy in enumerate(disk(64,.05/(2*case['fstop']))):jobs.append(dict(scene=name,blend=str(s/'scene.blend'),size=512,id=name+'-'+case['id']+f'-{tag}-k{k:02d}',frame=entry['frame'],spp=64,seed=seed,dof=False,focus=case['nominal_focus_m'],fstop=case['fstop'],aperture_xy=xy.tolist(),reference_tag=tag,output=str(s/'LDR-subviews'/(case['id']+f'-{tag}-k{k:02d}.exr'))))
    chunks=[[],[]]
    for i,name in enumerate(split['heldout']):chunks[i%2]+=[j for j in jobs if j['scene']==name]
    processes=[];handles=[];start=time.monotonic();subprocess.run(['nvidia-smi'],check=True)
    import shutil
    free=shutil.disk_usage(ROOT).free/2**30
    if free-20<60:write(OUT/'STOP.json',dict(reason='heldout LDR projection below60GiB',free_GiB=free));return
    for i,group in enumerate(chunks):
        path=OUT/f'heldout-ldr-worker{i}-jobs.json'
        if path.exists():assert json.loads(path.read_text())==group
        else:write(path,group)
        handle=(LOG/f'heldout-ldr-worker{i}.log').open('a');handles.append(handle);cmd=[BLENDER,'-b','-t','12','--python','scripts/route3_s2a_blender.py','--','--jobs',str(path)];processes.append(subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT))
    for p in processes:
        if p.wait()!=0:raise RuntimeError('heldout LDR Blender job failed;outputs retained')
    for h in handles:h.close()
    allrows=[]
    for i,group in enumerate(chunks):
        rows=json.loads((OUT/f'heldout-ldr-worker{i}-jobs-results.json').read_text());assert len(rows)==len(group);allrows+=rows
    write(OUT/'heldout-ldr-render.json',dict(rows=allrows,wall_seconds=time.monotonic()-start,GPU0_concurrency=2,container_CPU_quota_cores=20,Blender_threads_per_process=12,K=64,spp=64,GT_only=True,former_spp_and_seeds_preserved=True))
    from route3_pilot_fields import read
    from route3_upper_bound_evaluate import display,score
    from route3_ldr_u1r import model
    import torch
    torch.set_num_threads(6);net=model();refs=[];noise=[]
    for name in split['heldout']:
        s=OUT/name;dest=s/'LDR-reference';dest.mkdir(exist_ok=True)
        from route3_s2a_evaluate import masks
        for entry in json.loads((s/'prepared.json').read_text())['rows']:
            for c in entry['cases']:
                pair={}
                for tag in ['main','noise']:
                    subset=[j for j in allrows if j['scene']==name and j['frame']==entry['frame'] and j['focus']==c['nominal_focus_m'] and j['fstop']==c['fstop'] and j['reference_tag']==tag]
                    if not subset:continue
                    assert len(subset)==64
                    linear=sum(read(j['output'])[...,:3].astype(np.float64).clip(0,1) for j in subset)/64;rgb=display(linear);p=dest/(c['id']+'-'+tag+'.npz')
                    if not p.exists():np.savez_compressed(p,reference_srgb=rgb,rgb_linear=linear.astype(np.float32))
                    pair[tag]=rgb
                if 'noise' in pair:noise.append(dict(scene=name,id=c['id'],metrics=score(pair['main'],pair['noise'],masks(s,c),net)))
                refs.append(dict(scene=name,id=c['id'],path=str(dest/(c['id']+'-main.npz'))))
    write(RECORDS/'route3-network-s2a-2026-11-12-LDR-references.json',dict(rows=refs,noise=noise,formula='standard_sRGB(mean64 clip(scene-linear EXRSubview,0,1))',spp=64,K=64,auxiliary_reference=True,independent_seed_noise_reported=True))

def c2():
    import cv2,torch
    from route3_pilot_baselines import scene
    from route3_sharp_s1 import raster
    from route3_upper_bound_evaluate import display
    from src.layered_aperture import merge_layers
    from src.analytic_aperture import sampled
    from src.dual_layer import depth_band
    from src.c1_regions import linear_rgb
    from route3_b2_oracle_3d import sparse_pixel_layer
    profile=json.loads((RECORDS/'route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile'];rows=[];torch.set_num_threads(6);cv2.setNumThreads(1)
    for name in json.loads((OUT/'split.json').read_text())['heldout']:
        s=OUT/name;run([PY,'scripts/route3_pilot_propainter.py','--scene',name],name+'-C2',True)
        for e in json.loads((s/'prepared.json').read_text())['rows']:
            d=np.load(e['input_npz']);K=torch.tensor(d['K'],device='cuda');E=torch.eye(4,device='cuda');cloud=scene(s/'visible'/f'f{e["frame"]:03d}.pt');fill=torch.tensor(np.load(e['fill'])['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda');idx=[1,6,11,16,21,26,30,35,40,45,50,55,60].index(e['frame']);image=cv2.imread(str(OUT/'C2'/name/'results/input/frames'/f'{idx:04d}.png'))[...,::-1].astype(np.float32)/255;mask=cv2.imread(str(OUT/'C2'/name/'mask'/f'{idx:04d}.png'),0)>0;mask=cv2.dilate(mask.astype(np.uint8),np.ones((9,9),np.uint8))>0
            with torch.inference_mode():_,z,alpha=raster(cloud,K)
            own=np.where(alpha.cpu().numpy()>=.1,z.cpu().numpy(),d['depth']);yy,xx=np.nonzero(mask&(d['background_depth']>own+np.maximum(.005,.02*own)));parent=dict(uv=np.c_[xx+.5,yy+.5].astype(np.float32),depth=d['background_depth'][yy,xx],colors_linear=linear_rgb(image)[yy,xx])
            for c in e['cases']:
                path=s/'predictions'/(c['id']+'-C2_ProPainter.npz')
                if not path.exists():
                    band=depth_band(d['depth'],K[0,0].item(),c['focus_m'],.05/(2*c['fstop']))[0];uv=parent['uv'];hid=sparse_pixel_layer(parent,band[uv[:,1].astype(int),uv[:,0].astype(int)],K,E,profile,'cuda');base=np.load(s/'predictions'/(c['id']+'-A_color.npz'))
                    with torch.inference_mode():p=sampled(merge_layers(cloud,hid),K,E,512,512,c['focus_m'],.05/(2*c['fstop']),64,fill,bg,far_plane=1e5)
                    rgb=np.where(band[...,None],p['rgb_linear'].cpu().numpy(),base['rgb_linear']);np.savez_compressed(path,prediction_srgb=display(rgb),rgb_linear=rgb);del hid,p
                rows.append(dict(scene=name,**c,method='C2_ProPainter',path=str(path)))
            del cloud;torch.cuda.empty_cache()
    write(OUT/'heldout-C2.json',dict(rows=rows,configuration='unchanged pilot;depth<3m mask;official dilation4;S1d bg extension;R1behindownvisible;K64;LaMa residual',source_code_and_weights_unmodified=True,not_a_general_claim_about_ProPainter=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['ldr','c2'],required=True);a=p.parse_args();globals()[a.mode]()
