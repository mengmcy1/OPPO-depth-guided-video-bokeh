"""Two CPU workers precompute only already-selected future-scene geometry.

A per-scene flock serializes with the original driver. No extra render,
model score, threshold change, training or deletion is performed.
"""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/route3-network-s2a-2026-11-12-01';LOG=ROOT/'logs/route3-network-s2a-2026-11-12';CPU='/root/autodl-tmp/envs/dofgs/bin/python'

def one(name):
    rows=[];env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    for mode in ['center','aperture','training_focus']:
        if (OUT/'STOP.json').exists():break
        path=LOG/(name+'-prefetch-'+mode+'.log');start=time.monotonic()
        with path.open('a') as h:subprocess.run([CPU,'scripts/route3_s2a_geometry.py','--scene',str(OUT/name),'--mode',mode],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
        rows.append(dict(mode=mode,wall_seconds=time.monotonic()-start))
    return dict(scene=name,rows=rows,CPU_workers=2,BLAS_threads_each=4,GPU_used=False)

def main():
    c=json.loads((ROOT/'configs/route3-network-s2a-2026-11-12.json').read_text());progress=json.loads((OUT/'formal-progress.json').read_text());done={r['scene'] for r in progress['cost']};names=c['scenes'];remaining=[n for n in names if n not in done]
    # Leave the active scene to its existing process (started before flock).
    targets=remaining[1:];results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future={pool.submit(one,n):n for n in targets}
        for f in concurrent.futures.as_completed(future):
            results.append(f.result());(OUT/'geometry-prefetch.json').write_text(json.dumps(dict(rows=results,targets=targets,selected_scenes_only=True,container_CPU_quota=20),indent=2)+'\n');print('PREFETCH_READY',results[-1]['scene'],flush=True)

if __name__=='__main__':main()
