"""Evaluate the single frozen Dr.Bokeh adapter as completed scene batches arrive."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
RUN=Path('outputs/drbokeh-track-a-20260928')
LOG=Path('logs/drbokeh-track-a-20260928')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
protocol=json.loads((RUN/'protocol.json').read_text())
INPUT=Path(protocol['input'])
ids=[s['id'] for s in json.loads((INPUT/'config.json').read_text())['scenes']]
jobs=json.loads((RUN/'server-jobs.json').read_text())['inference']
owners={sid:job['pid'] for job in jobs for sid in job['scenes']}


def run(command,logfile):
    with logfile.open('x') as f:
        print('START',logfile,flush=True)
        subprocess.run([sys.executable,*command],stdout=f,stderr=subprocess.STDOUT,check=True)
        print('DONE',logfile,flush=True)


for batch in range(20):
    chosen=ids[batch*10:(batch+1)*10];start=time.monotonic()
    while True:
        missing=[s for s in chosen if not (RUN/'timing'/s/'complete.json').exists()]
        if not missing:break
        for sid in missing:
            stat=Path(f'/proc/{owners[sid]}/stat')
            if not stat.exists() or stat.read_text().split()[2]=='Z':
                raise RuntimeError(f'Inference owner exited before {sid} completed; preserve outputs and inspect worker log')
        if time.monotonic()-start>14400:raise TimeoutError(f'Batch {batch} not completed after 4 hours')
        time.sleep(10)
    path=RUN/'metrics'/f'batch_{batch:02d}.json';path.parent.mkdir(exist_ok=True)
    if path.exists():
        data=json.loads(path.read_text())
        assert [r['scene'] for r in data['rows']]==chosen and data['prediction_dir']=='pred_drbokeh'
        assert data['prediction_root']==str(RUN/'clips')
        continue
    run(['scripts/evaluate_renderer_pilot.py','--input',str(INPUT),'--only',','.join(chosen),
         '--reference-dir','gt_mpib_public_srgb','--reference-label','MPIB-public proxy',
         '--prediction-root',str(RUN/'clips'),'--prediction-dir','pred_drbokeh','--author-rm','--torchmetrics',
         '--scope','Single frozen Dr.Bokeh Track A; no oracle alpha or hidden layers; not official test',
         '--report',str(path)],LOG/f'metrics-{batch:02d}.log')

for phase in ('vfid','fvd'):
    dest=RUN/'author-public'
    run(['scripts/evaluate_author_public_metrics.py','--clips',str(INPUT),
         '--prediction-root',str(RUN/'clips'),'--prediction-dir','pred_drbokeh',
         '--reference-features','outputs/author-public-metrics-20260928',
         '--output',str(dest),'--phase',phase],LOG/f'{phase}.log')
print('ALL_DRBOKEH_TRACK_A_METRICS_COMPLETE',flush=True)
