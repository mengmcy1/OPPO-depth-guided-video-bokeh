"""Prepare the selected frontend, then train and score the fixed split."""
import fcntl
from concurrent.futures import ThreadPoolExecutor

from route3_g3_r14 import C, CPU, LOG, OUT, PREFIX, PY, REC, ROOT, TRUTH, qa, read, write
from route3_g3_r14_data import BASE
from route3_g3_r14_report import choose_geometry, get_split, table
from route3_g3_r14_train import SETTINGS, Store, evaluate_row, legacy, render_scene, train_network
from route3_upper_bound_evaluate import pool
from src.light_candidate_fusion import LightCandidateFusion
import torch


def stages(name, version, is_train):
    if is_train:
        for group in range(C['pose_perturbations']['groups']):
            qa.run([PY,'scripts/route3_g3_r14_data.py','pose',name,'--group',str(group),
                    '--version',version],name+f'-pose-{group}')


def finish():
    split = get_split()
    geometry = choose_geometry()
    if not split or not geometry:
        raise RuntimeError('Wait for category checks and T4 metrics before starting S2a')
    categories = ['broad','multilayer']+([] if read(OUT/'split.json')['thin_excluded'] else ['thin'])
    bad = [name for name in categories if not read(OUT/f'{name}-done.json')['passed']]
    cases = split['train']+split['heldout']
    if bad:
        write(OUT/'S2a-blocked.json',dict(reason='required category paused',categories=bad))
        return
    if not all((BASE/n/'training-data-ready.json').exists() for n in cases):
        raise RuntimeError('Training caches are not ready')
    overfit = [read(OUT/'overfit'/n/'training.json') for n in C['scenes'][:2]]
    if any(not d['passed'] for d in overfit):
        write(OUT/'S2a-blocked.json',dict(reason='single-scene direct fit stop',failed=[n for n,d in zip(C['scenes'],overfit) if not d['passed']]))
        return
    version=geometry['default_geometry']
    root=OUT/'geometry'/version
    if not all((root/n/'training-data-ready.json').exists() for n in cases):
        raise RuntimeError('Selected-geometry caches are not ready')
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(stages,n,version,n in split['train']) for n in cases]
        for f in futures:
            f.result()
    qa.run([CPU,'scripts/route3_s2a_data.py','--scene',split['train'][0],'--ordered-root',str(root),
            '--size','256','--GT-warmup'],'warmup')
    dest=OUT/'S2a'
    with (LOG/'gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        qa.guard(True)
        if not (dest/'training.json').exists():
            net=train_network(root,dest,split['train'],split['heldout'],
                              augmentations=[OUT/'geometry'/f'aug-{g}' for g in range(C['pose_perturbations']['groups'])])
        else:
            net=LightCandidateFusion().cuda()
            ckpt=torch.load(dest/f'step-{SETTINGS["S2a_steps"]:06d}.pt',weights_only=True)
            net.load_state_dict(ckpt['model'])
    legacy.OUT=root
    store=Store()
    direct=[]
    scores=[]
    for n in split['heldout']:
        with (LOG/'gpu0.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            qa.guard(True)
            m=store.manifest(n)
            direct.extend(dict(scene=n,case=r['case']['id'],**evaluate_row(net,store,r,m)) for r in m['rows']
                          if not r['case'].get('training_focus_only'))
            scores.extend(render_scene(n,net,root,dest/n))
    groups = [(n,[r for r in scores if r['scene']==n]) for n in split['heldout']]+[('all heldout',scores)]
    metrics=[]
    for name,data in groups:
        for focus in [1.4,5.,None]:
            subset=[r for r in data if focus is None or float(r['id'].rsplit('-F',1)[1])==focus]
            for region in ['full_image','in_focus','band','recoverable','invalid_geometry']:
                methods={method:pool([r for r in subset if r['method']==method and r['region']==region])
                         for method in ['A','D0','D1_Oracle','R1_R2','learned']}
                d0=methods['D0']['psnr']
                oracle=methods['D1_Oracle']['psnr']
                learned=methods['learned']['psnr']
                gain=learned-d0 if d0 is not None else None
                gap=oracle-d0 if d0 is not None else None
                metrics.append(dict(scene=name,focus=focus,region=region,method_metrics=methods,
                      learned_increment_dB=gain,oracle_increment_dB=gap,
                      accepted_recovery_fraction=gain/gap if name!='all heldout' and gap is not None and gap>=.5 else None,
                      insufficient_oracle_gain=name!='all heldout' and (gap is None or gap<.5)))
    result=dict(split=split,default_geometry=version,direct=direct,render_rows=scores,
                evaluation=metrics,reference='Cycles',pooled_by_SSE=True,GT_warmup_steps=SETTINGS['GT_warmup_steps'])
    write(dest/'evaluation.json',result)
    write(REC/f'{PREFIX}-S2a.json',dict(**result,train=read(dest/'training.json')))


if __name__=='__main__':
    torch.set_num_threads(4)
    finish()
