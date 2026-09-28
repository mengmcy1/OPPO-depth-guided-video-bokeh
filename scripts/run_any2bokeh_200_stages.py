"""Run fixed reproduction stages as inputs become available; never replace outputs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('stage',choices=['render','inference','metrics']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
run=a.run.resolve();clips=run/'clips';cfg=json.loads((clips/'config.json').read_text());logs=ROOT/'logs'/run.name;logs.mkdir(exist_ok=True)
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
def wait(paths):
    start=time.monotonic()
    while not all(path.exists() for path in paths):
        if time.monotonic()-start>43200:raise TimeoutError('Stage inputs unavailable for 12h')
        time.sleep(10)
def execute(command,name):
    print('START',name,flush=True)
    with (logs/name).open('x') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)
if a.stage=='render':
    for scene in cfg['scenes']:
        folder=clips/scene['id'];wait([folder/'metadata.json'])
        if (folder/'cycles_optix_s1024_f1/timing.json').exists():continue
        execute(['/root/autodl-tmp/tools/blender-4.5.13-linux-x64/blender','-b','-t','4','--python','scripts/render_pilot_cycles.py','--','--input',str(clips),'--only',scene['id']],f"render-{scene['id']}.log")
elif a.stage=='inference':
    wait([clips/s['id']/'metadata.json' for s in cfg['scenes']])
    ids=[s['id'] for s in cfg['scenes'] if not (clips/s['id']/'pred_any2bokeh/timing.json').exists()]
    if ids:execute(['/root/miniconda3/envs/any2bokeh/bin/python','scripts/run_any2bokeh_pilot.py','--input',str(clips),'--only',','.join(ids)],'inference-rest.log')
else:
    for i in range(0,len(cfg['scenes']),10):
        group=cfg['scenes'][i:i+10]
        wait([clips/s['id']/f for s in group for f in ['pred_any2bokeh/timing.json','cycles_optix_s1024_f1/timing.json']])
        execute(['/root/miniconda3/envs/magicbokeh/bin/python','scripts/evaluate_renderer_pilot.py','--input',str(clips),'--only',','.join(s['id'] for s in group),'--references','cycles','--cycles','cycles_optix_s1024_f1','--scope','Reproduced Any-to-Bokeh-style 200-video Test Set; Cycles fallback, non-held-out reproduction, not paper-identical or fair-benchmark scores','--report',str(run/f'metrics-{i//10:02d}.json')],f'metrics-{i//10:02d}.log')
