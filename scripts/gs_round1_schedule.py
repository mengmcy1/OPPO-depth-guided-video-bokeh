"""Temporary ordering before the09:00 report; the complete job list is retained."""
from datetime import datetime
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/dof-gs-round1-20261105.json'
REPORT_READY=ROOT/'logs/dof-gs-repro-2026-11-05-round1/priority-report-completed.json'


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def schedule_mode(config,now=None,report_ready=None):
    now=time.time() if now is None else now
    policy=config.get('priority_window',{})
    if not policy.get('enabled'):return 'full'
    cutoff=datetime.fromisoformat(policy['launch_until']).timestamp()
    due=datetime.fromisoformat(config['report_at']).timestamp()
    if now<cutoff:return 'priority'
    if now<due:return 'wrap_up'
    if report_ready is None:report_ready=read(REPORT_READY,{}).get('completed_at',0)>=due
    return 'full' if report_ready else 'awaiting_report'


def job_group(job,config):
    policy=config.get('priority_window',{})
    if job.get('kind')=='author':return 'author'
    if job['dataset']=='deblurnerf' and job['method'] in policy.get('common_methods',[]) and job['scene'] in policy.get('common_scenes',[]):return 'A_common'
    if job['dataset']=='dofnerf-mix' and job['scene'] in policy.get('dofnerf_scenes',[]) and job['method'] in [*policy.get('common_methods',[]),'bokehme']:return 'B_selected'
    return 'after_report'


def secondary_ready(config,jobs,now=None):
    now=time.time() if now is None else now
    policy=config['priority_window'];primary=policy['primary_dofnerf_scene']
    selected=[j for j in jobs if j['dataset']=='dofnerf-mix' and j['scene']==primary and j['method'] in [*policy['common_methods'],'bokehme']]
    if len(selected)!=4 or any(j['status'] not in {'evaluated','failed','blocked'} for j in selected):return False
    observed=[]
    for job in selected:
        if job.get('kind')=='bokehme':continue
        info=read(Path(job.get('output',job['base_output']))/'training-efficiency.json',{})
        if info.get('status')=='trained':observed.append(info['total_training_call_seconds'])
    needed=max(policy['secondary_min_remaining_seconds'],sum(observed)/2+900)
    return datetime.fromisoformat(policy['launch_until']).timestamp()-now>=needed


def job_order(job,config,jobs,now=None,phase='train'):
    mode=schedule_mode(config,now)
    if phase in {'render','evaluate'}:return (-1,0,0)
    if mode=='full':return (job['priority'],0,0)
    if mode!='priority':return None
    policy=config['priority_window'];group=job_group(job,config)
    if group=='author':return (0,-1,0)
    if group=='A_common':return (0,policy['common_scenes'].index(job['scene']),policy['common_methods'].index(job['method']))
    if group=='B_selected':
        primary=policy['primary_dofnerf_scene']
        if job['scene']!=primary and not secondary_ready(config,jobs,now):return None
        return (1,0 if job['scene']==primary else 1,0 if job.get('kind')=='bokehme' else 1)
    return None


def colmap_order(key,config,now=None):
    now=time.time() if now is None else now
    mode=schedule_mode(config,now)
    if mode=='full':return (0 if key.startswith('deblurnerf/') else 1,0)
    if mode!='priority':return None
    policy=config['priority_window'];scene=key.split('/')[-1]
    if key.startswith('deblurnerf/') and scene in policy['common_scenes']:return (0,policy['common_scenes'].index(scene))
    if key.startswith('dofnerf/dense/mix/') and scene in policy['dofnerf_scenes']:
        if scene==policy['primary_dofnerf_scene']:return (1,0)
        remaining=datetime.fromisoformat(policy['launch_until']).timestamp()-now
        if remaining>=policy['secondary_min_remaining_seconds']:return (1,1)
    return None
