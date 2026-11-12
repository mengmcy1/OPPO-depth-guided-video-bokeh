"""Recreate the two-pilot render job lists from the tracked configuration.

Order: build -> jobs -> CPU center/aperture geometry -> peel; ldr is optional
for training scenes and required for the pilot benchmark's LDR reference.
This planner never launches inference/training and refuses differing job lists.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]


def save(path,value):
    text=json.dumps(value,indent=2)+'\n'
    if path.exists():
        if json.loads(path.read_text())!=value:raise FileExistsError('Use a fresh output root; existing plan differs: '+str(path))
        return
    path.write_text(text)


def disk(K,radius):
    i=np.arange(K//2);r=np.sqrt((i+.5)/(K//2));theta=i*math.pi*(3-math.sqrt(5));xy=np.c_[r*np.cos(theta),r*np.sin(theta)];return np.r_[xy,-xy]*radius


def initial(config):
    root=Path(config['output']);plans={name:[] for name in ['preflight','formal','center-data','aperture-corrected']}
    for label in config['scenes']:
        s=root/label;meta=json.loads((s/'scene.json').read_text());common=dict(scene=label,blend=str(s/'scene.blend'),size=512,seed=20261112)
        def job(id,frame,spp,subdir,stem,**kw):return dict(**common,id=label+'-'+id,frame=int(frame),spp=spp,output=str(s/subdir/(stem+'.exr')),**kw)
        plans['preflight'].append(job('preview',30,64,'preflight','center',dof=False,data_passes=True,cryptomatte=True))
        for focus in config['focus_distances_m']:
            for spp,seed in [(512,20261112),(1024,20261112),(1024,20261113)]:
                j=job(f'noise-F{focus}-s{spp}-seed{seed}',30,spp,'preflight',f'N1.4-F{float(focus)}-s{spp}-seed{seed}',dof=True,fstop=1.4,focus=float(focus));j['seed']=seed;plans['preflight'].append(j)
        for name,hidden in [('foreground',[o['name'] for o in meta['objects'] if o['name'] not in meta['foreground_names']]),('background',meta['foreground_names'])]:plans['preflight'].append(job(name,30,256,'preflight',name,dof=False,hide_names=hidden,transparent_world=name=='foreground'))
        for frame in config['input_frames']:
            plans['formal'].append(job(f'input-{frame}',frame,256,'inputs',f'f{frame:03d}',dof=False,data_passes=True,cryptomatte=True))
            plans['center-data'].append(job(f'unfiltered-data-{frame}',frame,1,'input-passes-v2',f'f{frame:03d}',dof=False,data_passes=True,cryptomatte=True))
        for frame in config['representative_frames']:
            for focus in config['focus_distances_m']:
                for stop in config['fstops']:
                    key=f'f{frame:03d}-N{stop:g}-F{focus:g}';plans['formal'].append(job(key,frame,1024,'dof',key,dof=True,fstop=stop,focus=focus))
                    for k,xy in enumerate(disk(16,.05/(2*stop))):plans['aperture-corrected'].append(job('world-'+key+f'-k{k:02d}-camera-fixed',frame,1,'aperture-passes-v2',key+f'-k{k:02d}',dof=False,fstop=stop,focus=focus,aperture_xy=xy.tolist(),data_passes=True))
        for frame in config['continuous_frames']:
            for kind,focus in [('continuous',5.),('rack',1/(1/1.4+(frame-27)/7*(1/5-1/1.4)))]:plans['formal'].append(job(f'{kind}-{frame}',frame,1024,kind,f'f{frame:03d}',dof=True,fstop=1.4,focus=focus))
    return plans


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/route3-network-pilot-2026-11-12.json');p.add_argument('--stage',choices=['build','jobs','peel','ldr'],required=True);a=p.parse_args();c=json.loads(a.config.read_text());root=Path(c['output']).resolve();root.mkdir(parents=True,exist_ok=True)
    if a.stage=='build':
        for label,seed in zip(c['scenes'],c['seeds']):
            if not (root/label/'scene.blend').exists():subprocess.run([c['cycles']['binary'],'-b','--python',str(ROOT/'scripts/route3_pilot_blender.py'),'--','--build',str(root/label),'--label',label,'--seed',str(seed)],check=True)
    elif a.stage=='jobs':
        for name,jobs in initial(c).items():save(root/(name+'-jobs.json'),jobs)
    elif a.stage=='peel':
        jobs=[]
        for label in c['scenes']:
            s=root/label;meta=json.loads((s/'scene.json').read_text());names={o['id']:o['name'] for o in meta['objects']}
            for f,sets in json.loads((s/'geometry/required-exclusions.json').read_text())['frames'].items():
                for ids in sets:
                    key=f'f{int(f):03d}-exclude-'+'-'.join(map(str,ids));jobs.append(dict(id=label+'-'+key,scene=label,blend=str(s/'scene.blend'),frame=int(f),spp=256,size=512,dof=False,hide_names=[names[i] for i in ids],hidden_object_ids=ids,output=str(s/'references'/(key+'.exr'))))
        save(root/'peel-jobs.json',jobs)
    else:
        import torch
        from src.aperture_renderer import aperture_disk
        jobs=[];groups={}
        for label in c['scenes']:
            s=root/label
            for f in c['representative_frames']:
                for focus in c['focus_distances_m']:
                    for stop in c['fstops']:
                        case=f'f{f:03d}-N{stop:g}-F{focus:g}';groups[label+'-'+case]=[]
                        for seed,tag in [(20261112,'main')]+([(20261113,'noise')] if f==30 and stop==1.4 else []):
                            for k,xy in enumerate(aperture_disk(64,.05/(2*stop),device='cpu',dtype=torch.float32).numpy()):
                                j=dict(id=f'{label}-LDR-{case}-{tag}-k{k:02d}',scene=label,blend=str(s/'scene.blend'),frame=f,spp=64,size=512,dof=False,fstop=stop,focus=focus,aperture_xy=xy.tolist(),seed=seed,reference_tag=tag,output=str(s/'LDR-subviews'/case/tag/f'k{k:02d}.exr'));jobs.append(j);groups[label+'-'+case].append(j['id'])
        save(root/'ldr-subviews-jobs.json',jobs)
    print('Plan stage completed:',a.stage)


if __name__=='__main__':main()
