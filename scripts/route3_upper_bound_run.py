"""Explicit sequential GPU continuation after the authorized Blender queue.

This is the U0/U1/U2 experiment command list, not a general experiment manager.
Every subprocess is checked; a stop/error prevents subsequent GPU work.
"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/route3-upper-bound-2026-11-11-01'
RECORDS=ROOT/'03_实验记录/records'
LOG=ROOT/'logs/route3-upper-bound-2026-11-11'
PYTHON='/root/autodl-tmp/envs/route3/bin/python'
MAGIC='/root/miniconda3/envs/magicbokeh/bin/python'
BOKEH='/root/autodl-tmp/envs/dofgaussian/bin/python'


def run(name,command,gpu=False):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0' if gpu else '',
        TMPDIR='/root/autodl-tmp/route3-cache/tmp',TMP='/root/autodl-tmp/route3-cache/tmp',TEMP='/root/autodl-tmp/route3-cache/tmp',
        HF_HOME='/root/autodl-tmp/.cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TORCH_HOME='/root/autodl-tmp/.cache/torch',XDG_CACHE_HOME='/root/autodl-tmp/.cache',
        TORCH_EXTENSIONS_DIR='/root/autodl-tmp/route3-cache/torch_extensions',
        MPLCONFIGDIR='/root/autodl-tmp/route3-cache/matplotlib',CUPY_CACHE_DIR='/root/autodl-tmp/route3-cache/cupy',
        CUDA_CACHE_PATH='/root/autodl-tmp/.cache/cuda',OPENCV_IO_ENABLE_OPENEXR='1')
    if (OUT/'STOP-EXPERIMENT').exists():raise RuntimeError('Explicit experiment stop file')
    print('START',name,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)
    with (LOG/(name+'.log')).open('w') as log:
        if gpu:subprocess.run(['nvidia-smi'],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',name,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),flush=True)


def main():
    start=time.monotonic()
    while True:
        finished=True
        for name,count in [('U0-subviews',768),('U0-noise',128),('U1-supervision',240)]:
            p=OUT/name/'timings.json'
            try:finished &= p.exists() and len(json.loads(p.read_text()))==count and (p.parent/'projection-checks.json').exists()
            except json.JSONDecodeError:finished=False
        if finished:break
        if time.monotonic()-start>7200:raise TimeoutError('Blender inputs did not finish within2h; no new GPU work')
        if (OUT/'STOP-EXPERIMENT').exists():raise RuntimeError('Explicit experiment stop file')
        time.sleep(15)
    run('U0-average',[PYTHON,'scripts/route3_upper_bound_evaluate.py','--mode','average'])
    run('U0-evaluation',[PYTHON,'scripts/route3_upper_bound_evaluate.py','--mode','evaluate','--manifest',str(OUT/'U0-predictions.json'),'--stage','U0','--record',str(RECORDS/'route3-upper-bound-2026-11-11-U0-evaluation.json')])
    noise=json.loads((RECORDS/'route3-upper-bound-2026-11-11-U0-evaluation.json').read_text())['noise']
    if len(noise)!=2 or not all(x['below_existing_noise'] for x in noise):
        raise RuntimeError('U0 independent seed noise needs review before further work')
    run('U2-fill',[MAGIC,'scripts/route3_dual_layer_fill.py','--manifest',str(OUT/'U2-fill-inputs.json'),'--output',str(OUT/'U2-fill'),'--record',str(RECORDS/'route3-upper-bound-2026-11-11-U2-fill.json')],gpu=True)
    run('U2-O3',[BOKEH,'scripts/route3_upper_bound_u2.py','--mode','bokehme'],gpu=True)
    run('U2-render',[PYTHON,'scripts/route3_upper_bound_u2.py','--mode','render'],gpu=True)
    run('U2-evaluation',[PYTHON,'scripts/route3_upper_bound_evaluate.py','--mode','evaluate','--manifest',str(OUT/'U2-predictions.json'),'--stage','U2','--record',str(RECORDS/'route3-upper-bound-2026-11-11-U2-evaluation.json')])
    run('U2-decisions',[PYTHON,'scripts/route3_upper_bound_decisions.py','--stage','U2'])
    for frame in [1,30,60]:
        run(f'U1-fit-f{frame:03d}',[PYTHON,'scripts/route3_upper_bound_u1.py','--mode','fit','--frame',str(frame)],gpu=True)
    run('U1-render',[PYTHON,'scripts/route3_upper_bound_u1.py','--mode','render'],gpu=True)
    run('U1-evaluation',[PYTHON,'scripts/route3_upper_bound_evaluate.py','--mode','evaluate','--manifest',str(OUT/'U1-predictions.json'),'--stage','U1','--record',str(RECORDS/'route3-upper-bound-2026-11-11-U1-evaluation.json')])
    run('U1-decisions',[PYTHON,'scripts/route3_upper_bound_decisions.py','--stage','U1'])
    print('UPPER_COMPUTE_DONE',flush=True)


if __name__=='__main__':main()
