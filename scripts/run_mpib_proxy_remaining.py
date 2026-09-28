"""Run remaining fixed proxy rendering/inference and evaluate batches as they finish."""
import argparse,json,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--log-dir',type=Path);a=p.parse_args()
run=a.run.resolve();clips=run/'clips';cfg=json.loads((clips/'config.json').read_text());logs=a.log_dir or ROOT/'logs'/run.name;logs.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
def wait(paths):
    start=time.monotonic()
    while not all(path.exists() for path in paths):
        if time.monotonic()-start>43200:raise TimeoutError('Stage inputs unavailable after12h')
        time.sleep(5)
def execute(command,name):
    print('START',name,flush=True)
    with (logs/name).open('x') as f:subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,flush=True)
def render():
    ids=[s['id'] for s in cfg['scenes'] if not (clips/s['id']/'gt_mpib_public_srgb/timing.json').exists()]
    if ids:execute(['/root/miniconda3/envs/magicbokeh/bin/python','scripts/render_mpib_proxy.py','--input',str(clips),'--only',','.join(ids),'--wait-for-inputs'],'render-rest.log')
def inference():
    wait([clips/s['id']/'metadata.json' for s in cfg['scenes']])
    ids=[s['id'] for s in cfg['scenes'] if not (clips/s['id']/'pred_any2bokeh/timing.json').exists()]
    if ids:execute(['/root/miniconda3/envs/any2bokeh/bin/python','scripts/run_any2bokeh_pilot.py','--input',str(clips),'--only',','.join(ids)],'inference-rest.log')
def metrics():
    for i in range(0,200,10):
        group=cfg['scenes'][i:i+10]
        report=run/f'metrics-{i//10:02d}.json'
        if report.exists():
            rows=json.loads(report.read_text())['rows']
            assert len(rows)==len(group) and {r['scene'] for r in rows}=={s['id'] for s in group}
            assert all(r['renderer']=='MPIB-public proxy' for r in rows)
            print('REUSE',report.name,flush=True)
            continue
        wait([clips/s['id']/f for s in group for f in ['pred_any2bokeh/timing.json','gt_mpib_public_srgb/timing.json']])
        execute(['/root/miniconda3/envs/magicbokeh/bin/python','scripts/evaluate_renderer_pilot.py','--input',str(clips),'--only',','.join(s['id'] for s in group),'--reference-dir','gt_mpib_public_srgb','--reference-label','MPIB-public proxy','--scope','200-video non-held-out MPIB-public proxy reproduction; not official test set or future fair benchmark','--report',str(run/f'metrics-{i//10:02d}.json')],f'metrics-{i//10:02d}.log')
with ThreadPoolExecutor(max_workers=3) as pool:
    tasks=[pool.submit(fn) for fn in [render,inference,metrics]]
    for task in tasks:task.result()
execute(['/root/miniconda3/envs/magicbokeh/bin/python','scripts/summarize_mpib_proxy.py','--run',str(run),'--report','03_实验记录/records/any2bokeh-mpib-public-proxy-200-20260928.json'],'summary.log')
