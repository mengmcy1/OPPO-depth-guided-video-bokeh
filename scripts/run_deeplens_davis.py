"""Extend the accepted native-default DeepLens preflight to all50 DAVIS sequences."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess

from prepare_davis_v2 import write_new,link

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--sequences',nargs='*');a=p.parse_args()
    c=json.loads((ROOT/'configs/davis-eval-v2.json').read_text());protocol=json.loads((ROOT/'configs/davis-baselines.json').read_text())
    gate=json.loads((ROOT/'03_实验记录/records/davis-baseline-preflight-20260929.json').read_text())
    assert gate['status']=='ALL_FIVE_PREFLIGHTS_PASSED' and c['deeplens']['aperture_rule']==.5
    prior=ROOT/c['deeplens']['output'];out=ROOT/protocol['output']/'deeplens'
    manifest=json.loads((ROOT/c['derived']/'source-manifest.json').read_text());names=sorted(r['id'] for r in manifest['sequences'])
    rest=[n for n in names if n not in c['deeplens']['preflight_sequences']]
    if a.worker:
        from preflight_deeplens_davis import worker
        assert a.sequences and set(a.sequences)<=set(rest)
        task=copy.deepcopy(c);task['deeplens']['output']=str((out/'native').relative_to(ROOT));task['deeplens']['aperture_probe']=[.5]
        worker(task,a.sequences,None)
        return
    write_new(out/'protocol.json',dict(data_protocol=c,selected_aperture=.5,display='native-default',reuse_scenes=c['deeplens']['preflight_sequences']))
    for name in c['deeplens']['preflight_sequences']:
        link(out/'native'/name,prior/name)
    native=ROOT.parent/'envs/deeplens-tf1-cpu/bin/python';logs=ROOT/'logs/davis-baselines-20260929'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TF_ENABLE_ONEDNN_OPTS='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    def run(i):
        with (logs/f'deeplens-full-worker-{i}.log').open('a') as f:
            subprocess.run([str(native),str(Path(__file__).resolve()),'--worker','--sequences',*rest[i::4]],
                cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(run,range(4)))
    from PIL import Image
    for name in names:
        source=out/'native'/name/'aperture-0.5'
        assert len(list(source.glob('*.png')))==25
        for t in range(25):
            r=json.loads((out/'native'/name/f'{t:05d}.json').read_text())
            assert r['frame']==t and any(x['aperture']==.5 and x['signed_error']<=1e-6 for x in r['apertures'])
            with Image.open(source/f'{t:05d}.png') as im:assert im.size==(1024,576) and im.mode=='RGB';im.verify()
        link(out/'predictions'/name,source)
    write_new(out/'complete.json',dict(status='FIFTY_SEQUENCES_COMPLETE',scenes=50,frames=1250,aperture=.5,reused_frames=200,new_frames=1050))
    print('DEEPLENS_ALL50_COMPLETE',flush=True)


if __name__=='__main__':main()
