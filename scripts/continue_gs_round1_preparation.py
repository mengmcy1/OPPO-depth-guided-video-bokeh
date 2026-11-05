#!/usr/bin/env python3
"""Complete missing real scenes with at most two concurrent COLMAP pipelines."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import prepare_gs_repro as prep
from gs_round1_schedule import read,CONFIG,colmap_order,schedule_mode

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'logs/dof-gs-repro-2026-11-05'
COLMAP=Path('/root/autodl-tmp/tools/colmap-point-visibility/install/bin/colmap')


def update(key,row):
    with (LOG/'colmap-record.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=LOG/'colmap.json'
        data=json.loads(path.read_text()) if path.exists() else {'runs':{}}
        data['runs'][key]=row
        data.update(executable=str(COLMAP),max_parallel_pipelines=2,
                    protocol='modified COLMAP c84269d; real scenes SIMPLE_RADIAL official pipeline; DoF-NeRF provided poses, rectified497x331')
        prep.save(path,data)


def run(item):
    key,source,target,rebuild_sparse=item
    with (LOG/('scene-'+key.replace('/','-')+'.lock')).open('a') as scene_lock:
        fcntl.flock(scene_lock,fcntl.LOCK_EX)
        existing=json.loads((LOG/'colmap.json').read_text()).get('runs',{}).get(key,{})
        if existing.get('status')=='complete':return key,'already_complete'
        row={'source':str(source),'output':str(target),'status':'running','started_at':time.time(),
             'geometry_source':'new SIMPLE_RADIAL reconstruction' if rebuild_sparse else 'provided poses/points, matched by sharp-image names; rectified at497x331'}
        if existing.get('started_at'):row['previous_started_at']=existing['started_at']
        update(key,row)
        try:
            # A coordinator reload must not duplicate an already running native command.
            from gs_round1_resources import alive,attach
            adopted=[]
            for proc in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    argv=proc.read_bytes().split(b'\0');pid=int(proc.parent.name)
                    if argv and argv[0]==str(COLMAP).encode() and any(str(target).encode() in arg for arg in argv) and alive(pid):
                        adopted.append(pid)
                        if any(b'patch_match_stereo'==a or b'feature_extractor'==a or b'exhaustive_matcher'==a for a in argv):attach(pid,4096,'adopted COLMAP')
                except (OSError,ValueError):pass
            if adopted:
                row['adopted_native_commands']=adopted;update(key,row)
                while any(alive(pid) for pid in adopted):time.sleep(3)
                row['timing_note']='previous command continues unchanged; previous stage timings remain in native logs'
            prep.copy_scene(source,target,preserve_sparse=rebuild_sparse)
            logfile=LOG/('colmap-'+key.replace('/','-')+'.log')
            row['sparse']=prep.sparse(target,COLMAP,logfile) if rebuild_sparse else {
                'registered_images':prep.model_images(target/'sparse/0'),'total_images':len(prep.image_files(target/'images')),'fixed_supplied_poses':True}
            row['dense']=prep.dense(target,COLMAP,logfile)
            row['status']='complete' if row['dense']['complete'] and row['sparse']['registered_images']==row['sparse']['total_images'] else 'incomplete'
        except Exception as exc:
            row.update(status='blocked',error=repr(exc),traceback=traceback.format_exc())
        row['finished_at']=time.time();update(key,row)
        return key,row['status']


def main():
    os.environ['GS_COLMAP_TRAINING_OVERLAP']='1'
    os.environ['GS_COLMAP_MAX_PIPELINES']='2'
    LOG.mkdir(parents=True,exist_ok=True)
    with (Path(os.environ['TMPDIR'])/'dof-gs-colmap-pipeline.lock').open('a') as pipeline_lock:
        fcntl.flock(pipeline_lock,fcntl.LOCK_EX)
        jobs=[('deblurnerf/'+name,ROOT/'data/raw/DeblurNeRF/real_defocus_blur'/('defocus'+name),
               ROOT/'data/processed/deblurnerf-dofgaussian'/('defocus'+name),True)
              for name in ['caps','cisco','coral','cupcake','cups','daisy','sausage','seal','tools']]
        marker=LOG/'provided-poses.finished'
        while not marker.exists():time.sleep(5)
        prepared=json.loads(marker.read_text())
        jobs += [('dofnerf/dense/mix/'+name,ROOT/'data/processed/dofnerf-scenes-provided/mix'/name,
               ROOT/'data/processed/dofnerf-dofgaussian-provided/mix'/name,False)
              for name,row in prepared.items() if row.get('status')=='prepared']
        completed=set();active={}
        with ThreadPoolExecutor(max_workers=2) as pool:
            while len(completed)<len(jobs) or active:
                config=read(CONFIG,{})
                runs=read(LOG/'colmap.json',{}).get('runs',{})
                for item in jobs:
                    if runs.get(item[0],{}).get('status')=='complete':completed.add(item[0])
                pending=[item for item in jobs if item[0] not in completed and item[0] not in active.values() and
                         (runs.get(item[0],{}).get('status')=='running' or colmap_order(item[0],config) is not None)]
                pending.sort(key=lambda item:(-1,0) if runs.get(item[0],{}).get('status')=='running' else colmap_order(item[0],config))
                for item in pending[:max(0,2-len(active))]:active[pool.submit(run,item)]=item[0]
                prep.save(LOG/'colmap-schedule-status.json',{'phase':schedule_mode(config),'active':list(active.values()),
                    'deferred':[item[0] for item in jobs if item[0] not in completed and item[0] not in active.values() and colmap_order(item[0],config) is None]})
                if active:
                    finished,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                    for future in finished:
                        key=active.pop(future);completed.add(key);print(future.result(),flush=True)
                elif len(completed)<len(jobs):time.sleep(5)
    prep.save(LOG/'round1-colmap-continuation.finished',{'finished_at':time.time()})


if __name__=='__main__':main()
