"""Evaluate completed MPIB candidates in ten-scene batches while inference proceeds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
RUN = Path('outputs/mpib-adapter-recovery-20260928')
LOG = Path('logs/mpib-adapter-recovery-20260928')
INPUT = Path('outputs/any2bokeh-mpib-public-proxy-200-20260928/clips')
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
ids = [x['id'] for x in json.loads((INPUT/'config.json').read_text())['scenes']]


def run(cmd, path):
    with path.open('x') as log:
        print('START',path,flush=True)
        subprocess.run([sys.executable,*cmd],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',path,flush=True)


for batch in range(20):
    chosen = ids[batch*10:(batch+1)*10]
    start = time.monotonic()
    while not all((RUN/'timing'/sid/'complete.json').exists() for sid in chosen):
        if time.monotonic()-start > 7200:
            raise TimeoutError(f'Inference has not completed batch {batch}')
        time.sleep(10)
    for gamma in ('1','2.2'):
        path = RUN/f'gamma_{gamma}'/'metrics'/f'batch_{batch:02d}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            data=json.loads(path.read_text())
            assert [x['scene'] for x in data['rows']] == chosen
            assert data['prediction_dir'] == 'pred_mpib'
            assert data['prediction_root'] == str(RUN/f'gamma_{gamma}'/'clips')
            continue
        run(['scripts/evaluate_renderer_pilot.py','--input',str(INPUT),'--only',','.join(chosen),
             '--reference-dir','gt_mpib_public_srgb','--reference-label','MPIB-public proxy',
             '--prediction-root',str(RUN/f'gamma_{gamma}'/'clips'),'--prediction-dir','pred_mpib',
             '--author-rm','--torchmetrics','--scope',f'MPIB gamma={gamma} recovery; not official test',
             '--report',str(path)],LOG/f'metrics-gamma{gamma}-{batch:02d}.log')

for gamma in ('1','2.2'):
    for phase in ('vfid','fvd'):
        dest=RUN/f'gamma_{gamma}'/'author-public'
        if (dest/f'{phase}.json').exists():
            saved=json.loads((dest/f'{phase}.json').read_text())
            assert saved['video_count']==200 and saved['protocol']['prediction_root']==str(RUN/f'gamma_{gamma}'/'clips')
            continue
        run(['scripts/evaluate_author_public_metrics.py','--clips',str(INPUT),
             '--prediction-root',str(RUN/f'gamma_{gamma}'/'clips'),'--prediction-dir','pred_mpib',
             '--reference-features','outputs/author-public-metrics-20260928',
             '--output',str(dest),'--phase',phase],LOG/f'{phase}-gamma{gamma}.log')
print('ALL_MPIB_RECOVERY_METRICS_COMPLETE',flush=True)
