"""Write explicit U0 evaluation and U1 train/heldout lens positions before use."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.aperture_renderer import aperture_disk


def main():
    out=ROOT/'outputs/route3-upper-bound-2026-11-11-01'
    if (out/'view-plan.json').exists():raise FileExistsError('Plan already exists')
    base=ROOT/'outputs/route3-b2only-2026-11-10-01'
    scene=ROOT/'outputs/route3-a2b-2026-11-10-01/blender-b2/B2-high'
    K=np.array([[512*50/36,0,256],[0,512*50/36,256],[0,0,1.]])
    configs={'K':K.tolist(),'blend':str(scene/'B2-high.blend')}
    jobs=[];noise=[];training=[];positions=[]
    radius=.05/(2*1.4)
    fixed=aperture_disk(64,1.,device='cpu',dtype=torch.float64).numpy()
    forbidden=np.concatenate([fixed*radius,fixed*.05/(2*2.)])
    rng=np.random.default_rng(2026111101)
    for frame in [1,30,60]:
        for focus in [1.4,5.]:
            for stop in [1.4,2.]:
                case=f'B2-high-f{frame:03d}-N{stop:g}-F{focus:g}'
                for k,xy in enumerate(fixed*.05/(2*stop)):
                    job=dict(key=f'{case}-k{k:02d}',frame=frame,focus_m=focus,fstop=stop,
                        xy_m=xy.tolist(),sample=k,spp=1024,seed=2026111102,role='U0-fixed64')
                    jobs.append(job)
                    if (frame,focus,stop) in [(1,1.4,1.4),(30,5.,2.)]:
                        noise.append(dict(job,key=job['key']+'-noise',seed=2026111103,role='U0-noise-independent'))
        for index in range(80):
            u,v=rng.random(2);xy=radius*np.sqrt(u)*np.array([np.cos(2*np.pi*v),np.sin(2*np.pi*v)])
            if np.min(np.linalg.norm(forbidden-xy,axis=-1))<1e-7:raise ValueError('Evaluation-position collision')
            role='U1-train' if index<64 else 'U1-heldout'
            job=dict(key=f'f{frame:03d}-{role}-{index:02d}',frame=frame,focus_m=None,
                xy_m=xy.tolist(),sample=index,spp=1024,seed=2026111104,role=role)
            training.append(job);positions.append(job)
        row=next(r for r in json.loads((base/'inputs/B2-high/pinhole/timings.json').read_text()) if r['frame']==frame)
        positions.append(dict(key=f'f{frame:03d}-center',frame=frame,focus_m=None,xy_m=[0.,0.],role='U1-center',
            existing_exr=str(base/'inputs/B2-high/pinhole'/(row['key']+'.exr'))))
    groups={}
    for job in jobs:groups.setdefault(job['key'].rsplit('-k',1)[0],[]).append(job)
    plan=dict(K=K.tolist(),evaluation_groups=groups,independent_noise_jobs=noise,U1_positions=positions,
        evaluation_positions_used_in_U1=False,one_fit_per_frame=True,seed=2026111101)
    (out/'view-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'U1-positions.json').write_text(json.dumps(dict(K=K.tolist(),positions=positions,seed=2026111101),indent=2)+'\n')
    for name,listing,dest in [('U0-full',jobs,'U0-subviews'),('U0-noise',noise,'U0-noise'),
                              ('U0-pilot',[jobs[0],jobs[31],jobs[383]],'U0-subviews'),
                              ('U1-supervision',training,'U1-supervision')]:
        (out/(name+'-jobs.json')).write_text(json.dumps(dict(configs,output=str(out/dest),jobs=listing),indent=2)+'\n')
    print(json.dumps(dict(U0=len(jobs),noise=len(noise),U1_new=len(training),U1_center_reused=3)))


if __name__=='__main__':main()
