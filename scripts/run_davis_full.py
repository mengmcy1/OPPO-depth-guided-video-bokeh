"""Run independent frozen methods; evaluate only after all six1250-frame sets exist."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PY={
 'common':'/root/miniconda3/envs/magicbokeh/bin/python',
 'any2bokeh':'/root/miniconda3/envs/any2bokeh/bin/python',
 'bokehdiff':'/root/autodl-tmp/envs/bokehdiff/bin/python'}
METHODS=['any2bokeh','mpib','bokehme','drbokeh','bokehdiff','deeplens']


def execute(name,cmd):
    logs=ROOT/'logs/davis-baselines-20260929';logs.mkdir(parents=True,exist_ok=True)
    with (logs/f'{name}.log').open('a') as f:
        subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'),
                       stdout=f,stderr=subprocess.STDOUT,check=True)
    print('FINISHED',name,datetime.now(timezone.utc).isoformat(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['inference','evaluate','all'],default='all');a=p.parse_args()
    gate=json.loads((ROOT/'03_实验记录/records/davis-baseline-preflight-20260929.json').read_text())
    assert gate['status']=='ALL_FIVE_PREFLIGHTS_PASSED'
    config=json.loads((ROOT/'configs/davis-baselines.json').read_text());out=ROOT/config['output']
    if a.phase in ('all','inference'):
        def gpu_lane(methods):
            for method in methods:
                execute(method+'-full',[PY.get(method,PY['common']),'scripts/run_davis_baseline.py','--method',method,'--stage','full'])
        with ThreadPoolExecutor(3) as pool:
            jobs=[pool.submit(gpu_lane,['bokehdiff']),
                  pool.submit(gpu_lane,['any2bokeh','mpib','bokehme','drbokeh']),
                  pool.submit(execute,'deeplens-full',[PY['common'],'scripts/run_deeplens_davis.py'])]
            for job in jobs:job.result()
    c=json.loads((ROOT/config['data_protocol']).read_text());names=[s['id'] for s in json.loads((ROOT/c['derived']/'source-manifest.json').read_text())['sequences']]
    for method in METHODS:
        for name in names:
            frames=sorted((out/method/'predictions'/name).glob('*.png'))
            assert [f.name for f in frames]==[f'{i:05d}.png' for i in range(25)],(method,name)
    if a.phase in ('all','evaluate'):
        def evaluate(method):
            execute(method+'-vepi',[PY['common'],'scripts/evaluate_davis_vepi.py',
                '--predictions',str(out/method/'predictions'),'--output',str(out/method/'evaluation')])
        with ThreadPoolExecutor(3) as pool:list(pool.map(evaluate,METHODS))
        results={m:json.loads((out/m/'evaluation/summary.json').read_text()) for m in METHODS}
        result=dict(status='SIX_METHOD_DAVIS_50_METRICS_COMPLETE_FINAL_REVIEW_PENDING',scenes=50,frames_per_method=1250,
            configuration='configs/davis-baselines.json',completed_utc=datetime.now(timezone.utc).isoformat(),
            methods={m:{k:results[m][k] for k in ['videos','frames','VEPI_public_mp4','VEPI_png','codec_delta','codec_delta_range','codec','ffmpeg','rows']} for m in METHODS},
            scope='Reproduced DAVIS2016 first25 protocol, not the private author manifest. VEPI measures edge preservation, not complete bokeh quality; qualitative review required.')
        from prepare_davis_v2 import write_new
        write_new(ROOT/'03_实验记录/records/davis-six-method-50-20260929.json',result)
        print('ALL_SIX_METHOD_METRICS_COMPLETE',flush=True)


if __name__=='__main__':main()
