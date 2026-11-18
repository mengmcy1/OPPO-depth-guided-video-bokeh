"""G3 truth and estimated frontends. Each scene writes only its own outputs."""
import argparse
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from route3_g3_r14 import C, CONFIG, CPU, LOG, OUT, PY, ROOT, TRUTH, PREVIOUS, qa, read, reference_files, write

BASE = OUT/'geometry'/'original'
NATIVE = OUT/'geometry'/'native'
FRAMES = C['input_frames']
REPS = C['representative_frames']


def truth_aux(name):
    scene = TRUTH/name
    if (scene/'truth-aux-ready.json').exists():
        return
    assert read(scene/'quality.json')['passed']
    start = time.monotonic()
    # Revision 13's first scene already has verified central and aperture casts.
    previous = PREVIOUS/name/'geometry'
    if name == C['scenes'][0] and (previous/'required-exclusions.json').exists():
        reference_files(previous, scene/'geometry', [p.name for p in previous.glob('*')])
    for mode in ['center', 'aperture', 'training_focus']:
        qa.run([CPU,'scripts/route3_s2a_geometry.py','--scene',str(scene),'--mode',mode],name+'-'+mode)
    req = read(scene/'geometry/required-exclusions.json')['frames']
    names = {o['id']:o['name'] for o in read(scene/'scene.json')['objects']}
    peels = []
    for frame, sets in req.items():
        for ids in sets:
            key = f'f{int(frame):03d}-exclude-'+'-'.join(map(str,ids))
            peels.append(dict(scene=name,id=key,blend=str(scene/'scene.blend'),frame=int(frame),
                              spp=256,size=512,seed=20261112,omit_reserved_passes=True,dof=False,
                              hidden_object_ids=ids,hide_names=[names[i] for i in ids],
                              output=str(scene/'references'/(key+'.exr'))))
    qa.jobs(scene,'peels',peels)
    qa.run([CPU,'scripts/route3_s2a_reference_geometry.py','--scene',str(scene),
            '--peel-manifest',str(scene/'peels-jobs.json')],name+'-peel-geometry')
    qa.run([CPU,'scripts/route3_pilot_oracle.py','--scene',str(scene),
            '--peel-manifest',str(scene/'peels-jobs-results.json')],name+'-oracle')
    write(scene/'truth-aux-ready.json',dict(scene=name,seconds=time.monotonic()-start,
                                          K=16,training_focuses_per_frame=4))
    print('TRUTH_AUX_READY', name, flush=True)


def frontend(name):
    import route3_s2a_models as model
    import fcntl
    folder = NATIVE/name
    if (folder/'frontend-ready.json').exists():
        return
    assert read(TRUTH/name/'quality.json')['passed']
    start = time.monotonic()
    reference_files(TRUTH/name,folder,['inputs','scene-geometry.npz','scene.json'])
    write(folder/'inputs-source.json',dict(truth=str(TRUTH/name),pose='E1'))
    with (LOG/'gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        qa.guard(True)
        model.inference(folder)
        model.calibrate(folder,flow_only=True)
    model.calibrate(folder,cached_flow_only=True)
    write(folder/'frontend-ready.json',dict(scene=name,seconds=time.monotonic()-start))
    print('FRONTEND_READY',name,flush=True)


def dominant(name, version='original', calibration=None, root=BASE):
    """Cache factorized explicit surfaces and a separate, fixed A geometry."""
    import numpy as np
    import torch
    from route3_s2a_baselines import scene
    from src.aperture_renderer import GaussianScene
    from src.visible_depth_interface import dominant_surface, visible_depth_interface
    native = NATIVE/name
    source = Path(calibration) if calibration else native/'models/calibration'
    folder = root/name
    reference_files(TRUTH/name, folder, ['inputs','scene-geometry.npz','scene.json','geometry','oracle','dof'])
    reference_files(TRUTH/name,folder,['inputs'])
    if not (folder/'input-passes-v2').exists():
        (folder/'input-passes-v2').symlink_to((TRUTH/name/'inputs').resolve(), target_is_directory=True)
    reference_files(native/'models',folder/'models',['SHARP','DA3'])
    dest = folder/'models/calibration'
    reference_files(source, dest, [p.name for p in source.iterdir() if p.name != 'SHARP'])
    (dest/'SHARP').mkdir(exist_ok=True)
    fit = {r['frame']:r['models']['SHARP'] for r in read(source/'summary.json')['rows']}
    for f in FRAMES:
        p = dest/'SHARP'/f'f{f:03d}.npz'
        if p.exists():
            continue
        raw = scene(native/'models/SHARP'/f'f{f:03d}.pt')
        d = np.load(source/'SHARP'/f'f{f:03d}.npz')
        z = raw.means[:,2]
        mapped = 1/(fit[f]['a']/z.clamp_min(1e-6)+fit[f]['b']).clamp_min(1e-4)
        ratio = torch.where(z>1e-6,mapped/z.clamp_min(1e-6),torch.ones_like(z))
        cloud = GaussianScene(raw.means*ratio[:,None],raw.quats,raw.scales*ratio[:,None],raw.opacities,raw.colors,raw.source_frames)
        K = torch.tensor(d['K'],device='cuda')
        with torch.inference_mode():
            maps = visible_depth_interface(cloud,K,np.ones((512,512),bool)) if f in REPS else dominant_surface(cloud,K,np.ones((512,512),bool))
        depth = maps['cluster_depth']
        finite = np.isfinite(depth)&(depth>0)
        valid = finite&(maps['cluster_fraction']>=.9)&(maps['contribution_alpha']>1e-6)
        np.savez_compressed(p,depth=np.where(finite,depth,1e4).astype(np.float32),K=d['K'],c2w=d['c2w'],dominant_valid=valid)
        if f in REPS:
            a = folder/'A-depth'; a.mkdir(exist_ok=True)
            np.savez_compressed(a/f'f{f:03d}.npz', **{k:v for k,v in maps.items() if isinstance(v,np.ndarray)})
        print('DOMINANT',name,f,float(valid.mean()),flush=True)
    write(folder/'explicit.json', dict(geometry=version, depth='models/calibration/SHARP/*.npz',
          input_RGB='models/SHARP/*.npz', representation=dict(subdivision=5,sigma_px=.7,opacity=.999),
          order='O1', fallback='F2', factorized=True))


def prepare_labels(name, root, train_only=False):
    import numpy as np
    from route3_s2a_candidates import prepare, local_placement
    folder = root/name
    reference_files(TRUTH/name,folder,['geometry','oracle'])
    prepare(folder, dominant_p2=True)
    dst = folder/'placement';dst.mkdir(exist_ok=True)
    fg = np.load(folder/'scene-geometry.npz')['foreground_object_ids']
    for f in REPS:
        path = dst/f'f{f:03d}.npz'
        if path.exists():
            continue
        g = np.load(folder/'geometry'/f'f{f:03d}-center.npz')
        z = np.load(folder/'models/calibration/SHARP'/f'f{f:03d}.npz')['depth']
        a,b,stats,valid = local_placement(g['depth_axial_m'],z,g['center_visible']&~np.isin(g['object_id'],fg),return_valid=True)
        np.savez_compressed(path,a=a,b=b,local_supported=valid)
        write(path.with_suffix('.json'),dict(stats,order='O1; clamp label depth to 1.03*visible before supervision'))
    qa.run([PY,'-c','import route3_s2a_memory as m;import sys;from pathlib import Path;m.OUT=Path(sys.argv.pop(1));m.main()',
            str(root),'--scene',name],name+'-memory-'+root.name)
    for size in [256] if train_only else [256,512]:
        qa.run([CPU,'scripts/route3_s2a_data.py','--scene',name,'--ordered-root',str(root),'--size',str(size)],
               name+f'-data-{size}-'+root.name)
    write(folder/'training-data-ready.json',dict(scene=name,version=root.name))


def geometry_version(name, version, root, original=None, train_only=False):
    import numpy as np
    from route3_g3_r14_geometry import flow_depth, registered_masks, run_depthsplat
    original = original or BASE
    source = original/name
    folder = root/name
    if (folder/'training-data-ready.json').exists():
        return
    reference_files(source,folder,['scene.json','scene-geometry.npz','inputs','input-passes-v2','geometry','oracle','dof','A-depth'])
    if version=='original':
        reference_files(source,folder,['models'])
    if version!='original':
        native_models = source/'models'
        cal = native_models/'calibration'
        dest = folder/'models/calibration'
        reference_files(native_models,folder/'models',['SHARP','DA3'])
        reference_files(cal,dest,[p.name for p in cal.iterdir() if p.name!='SHARP'])
        (dest/'SHARP').mkdir(exist_ok=True)
        views = {}
        for f in FRAMES:
            current = np.load(cal/'SHARP'/f'f{f:03d}.npz')
            raw = np.load(native_models/'SHARP'/f'f{f:03d}.npz')
            views[f] = {k:current[k] for k in ['depth','K','c2w']}
            views[f]['rgb_linear'] = raw['rgb_linear']
        masks = registered_masks(views)
        neighbors = read(cal/'flow-plan.json')['selected']
        changes = []
        if version.startswith('depthsplat'):
            output = folder/'depthsplat'
            plan = []
            for f,v in views.items():
                chosen = [f]+sorted([g for g in FRAMES if f!=g],key=lambda g:(np.linalg.norm(v['c2w'][:3,3]-views[g]['c2w'][:3,3]),g))[:7]
                plan.append(dict(scene=name,target_frame=f,target_index=0,frames=chosen,
                                 bounds_m=[.5,200.],
                                 images=[str(source/'inputs'/f'f{g:03d}.png') for g in chosen],
                                 K=[views[g]['K'].tolist() for g in chosen], c2w=[views[g]['c2w'].tolist() for g in chosen],
                                 output=str(output/f'f{f:03d}.npz')))
            plan_path=folder/'depthsplat-plan.json'
            write(plan_path,plan)
            run_depthsplat(plan_path,name+'-'+root.name+'-depthsplat')
        for f,v in views.items():
            if version=='flow':
                z,valid = flow_depth(f,views,masks,cal/'flow',neighbors[str(f)])
            else:
                with np.load(output/f'f{f:03d}.npz') as result:
                    z,valid = result['depth'], result['consistent']
            valid &= ~masks[f]
            data = np.load(cal/'SHARP'/f'f{f:03d}.npz')
            np.savez_compressed(dest/'SHARP'/f'f{f:03d}.npz', depth=np.where(valid,z,v['depth']),
                                K=v['K'],c2w=v['c2w'],dominant_valid=(valid|data['dominant_valid']),replaced=valid)
            changes.append(dict(frame=f,replaced_pixels=int(valid.sum()),motion_pixels=int(masks[f].sum())))
        write(folder/'explicit.json',dict(version=version,frames=changes,original=str(source),O1=True,F2=True))
    prepare_labels(name,root,train_only)


def pose_variant(name, group, version):
    from scipy.spatial.transform import Rotation
    import numpy as np
    cfg = C['pose_perturbations']
    native = NATIVE/name
    dest = OUT/'geometry'/'pose-calibration'/f'{group}'/name
    dest.mkdir(parents=True,exist_ok=True)
    base_cal = native/'models/calibration'
    rng = np.random.default_rng(cfg['seed']+100*group+C['scenes'].index(name))
    poses = np.load(native/'scene-geometry.npz')['camera_c2w_opencv'].copy()
    translations = poses[:,:3,3].copy()
    drift = rng.normal(0,cfg['sequence_scale_drift_std'])*np.linspace(0,1,len(poses))
    poses[:,:3,3] = translations[0]+(translations-translations[0])*(1+drift[:,None])
    indices = np.array(FRAMES)-1
    poses[indices,:3,3] += rng.normal(0,cfg['translation_sigma_m'],(len(indices),3))
    rotation = Rotation.from_rotvec(rng.normal(0,np.deg2rad(cfg['rotation_sigma_degrees']),(len(indices),3))).as_matrix()
    poses[indices,:3,:3] = poses[indices,:3,:3] @ rotation
    p = dest/'poses.npz'
    if not p.exists():
        np.savez_compressed(p,c2w=poses)
    calibration = dest/'calibration'
    old = os.environ.copy()
    try:
        os.environ.update(ROUTE3_CALIBRATION_OUTPUT=str(calibration), ROUTE3_CALIBRATION_POSES=str(p),
                          ROUTE3_CALIBRATION_FLOW_PLAN=str(base_cal/'flow-plan.json'),
                          ROUTE3_CALIBRATION_FLOW_CACHE=str(base_cal/'flow'))
        from route3_s2a_models import calibrate
        import fcntl
        root = OUT/'geometry'/f'aug-{group}'
        original = root if version=='original' else OUT/'geometry'/f'aug-{group}-original'
        calibrate(native,cached_flow_only=True)
        with (LOG/'gpu0.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            qa.guard(True)
            dominant(name,'original',calibration,original)
        # The cache describes the same fixed A; only the explicit geometry and
        # candidate coordinate system are augmented in training.
        reference_files(base_cal,calibration,['flow'])
        reference_files(calibration,original/name/'models/calibration',['flow'])
        if version=='original':
            prepare_labels(name,root,train_only=True)
        else:
            geometry_version(name,version,root,original,train_only=True)
    finally:
        for key in ['ROUTE3_CALIBRATION_OUTPUT','ROUTE3_CALIBRATION_POSES',
                    'ROUTE3_CALIBRATION_FLOW_PLAN','ROUTE3_CALIBRATION_FLOW_CACHE']:
            if key in old:
                os.environ[key]=old[key]
            else:
                os.environ.pop(key,None)


def color_anchor(name):
    from route3_s2a_baselines import anchor_visible
    folder = BASE/name
    if (folder/'visible/summary.json').exists():
        return
    if not (folder/'fill-record.json').exists():
        qa.run(['/root/autodl-tmp/envs/conda-magicbokeh/bin/python',
                'scripts/route3_dual_layer_fill.py','--manifest',str(folder/'fill-inputs.json'),
                '--output',str(folder/'fill'),'--record',str(folder/'fill-record.json')],name+'-fill',True)
    import fcntl
    with (LOG/'gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        qa.guard(True)
        anchor_visible(folder)


def run_stage(name, stage):
    qa.run([PY,'scripts/route3_g3_r14_data.py',stage,name],name+'-pipeline-'+stage)


def full_labels(name, version):
    from route3_s2a_candidates import labels
    labels(OUT/'geometry'/version/name,ordered=True)


def temporal(name):
    scene = TRUTH/name
    import fcntl
    with (scene/'.temporal.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        render_temporal(scene,name)


def render_temporal(scene, name):
    ready = scene/'temporal-ready.json'
    if ready.exists():
        return
    spp = read(scene/'quality.json')['final_spp']
    jobs = []
    for frame in C['continuous_frames']:
        alpha = (frame-C['continuous_frames'][0])/(C['continuous_frames'][-1]-C['continuous_frames'][0])
        for group, focus in [('continuous',5.),('focus-rack',1/((1-alpha)/1.4+alpha/5.))]:
            jobs.append(dict(id=f'{group}-{frame}',scene=name,frame=frame,blend=str(scene/'scene.blend'),
                             size=512,spp=spp,seed=20261112,dof=True,fstop=1.4,focus=focus,
                             omit_reserved_passes=True,output=str(scene/group/f'f{frame:03d}.exr')))
    start=time.monotonic()
    qa.jobs(scene,'temporal',jobs)
    write(ready,dict(scene=name,frames=C['continuous_frames'],count=len(jobs),spp=spp,seconds=time.monotonic()-start))


def remaining():
    """Overlap CPU truth preparation with serialized GPU frontend jobs."""
    pending = set(C['scenes'])-set(C['scenes'][:2])
    queued = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        def process(name):
            run_stage(name,'frontend')
            run_stage(name,'dominant')
            run_stage(name,'truth_aux')
            run_stage(name,'labels')
            run_stage(name,'color_anchor')
        while pending or queued:
            qa.guard()
            thin = OUT/'thin-done.json'
            for name in C['scenes']:
                if name not in pending:
                    continue
                category = name.rsplit('-',1)[-1]
                if category=='thin' and not thin.exists():
                    continue
                if category=='thin' and not read(thin)['passed']:
                    pending.remove(name)
                    continue
                p = TRUTH/name/'quality.json'
                if p.exists():
                    pending.remove(name)
                    if read(p)['passed']:
                        queued[name] = pool.submit(process,name)
                elif (OUT/f'{category}-done.json').exists():
                    pending.remove(name)
            for name,future in list(queued.items()):
                if future.done():
                    try:
                        future.result()
                    except Exception as e:
                        write(OUT/'geometry-errors'/f'{name}.json',dict(scene=name,error=str(e)))
                        print('SCENE_PRECOMPUTE_ERROR',name,e,flush=True)
                    del queued[name]
            if pending or queued:
                time.sleep(4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=['truth_aux','frontend','dominant','labels','full_labels','color_anchor','temporal','remaining','variant','pose'])
    p.add_argument('scene',nargs='?')
    p.add_argument('--version',choices=['flow','depthsplat-official','original'])
    p.add_argument('--group',type=int)
    a = p.parse_args()
    qa.torch.set_num_threads(4)
    if a.phase=='dominant':
        import fcntl
        with (LOG/'gpu0.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            qa.guard(True)
            globals()[a.phase](a.scene)
    elif a.phase=='frontend':
        frontend(a.scene)
    elif a.phase=='labels':
        prepare_labels(a.scene,BASE)
    elif a.phase=='full_labels':
        full_labels(a.scene,a.version or 'original')
    elif a.phase=='color_anchor':
        color_anchor(a.scene)
    elif a.phase=='remaining':
        remaining()
    elif a.phase=='temporal':
        temporal(a.scene)
    elif a.phase=='variant':
        geometry_version(a.scene,a.version,OUT/'geometry'/a.version)
    elif a.phase=='pose':
        pose_variant(a.scene,a.group,a.version)
    else:
        truth_aux(a.scene)


if __name__=='__main__':
    main()
