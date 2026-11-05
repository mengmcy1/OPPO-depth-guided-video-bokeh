#!/usr/bin/env python3
"""Simple persistent round-one queue: exclusive cake runs, then memory packing.

Run with nohup after preparing the method scripts. Progress and child process IDs
are saved on the data disk. A dated report is generated without stopping jobs.
"""
import argparse
from collections import defaultdict
import csv
import fcntl
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import re

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
PREFIX = 'dof-gs-repro-2026-11-05-round1'
LOG = ROOT/'logs'/PREFIX
OUT = ROOT/'outputs/dof-gs-repro-2026-11-05'
PREP = ROOT/'logs/dof-gs-repro-2026-11-05'
RECORD = ROOT/'03_实验记录/records'
HOLDS = {'cake': 9, 'caps': 5, 'cisco': 7, 'coral': 10, 'cupcake': 6, 'cups': 8, 'daisy': 8, 'sausage': 6, 'seal': 6, 'tools': 7}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n'); temporary.replace(path)


def read(path, default=None):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return default


def command(arguments, log, environment=None):
    with log.open('a') as stream:
        stream.write('\nCOMMAND '+json.dumps(list(map(str, arguments)))+'\n'); stream.flush()
        return subprocess.Popen(list(map(str, arguments)), cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, env=environment, close_fds=True)


def environment(method):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONNOUSERSITE='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OMP_NUM_THREADS='4', TORCH_HOME=str(WORK/'.cache/torch'),
               HF_HOME=str(WORK/'.cache/huggingface'), TMPDIR=str(WORK/'.cache/tmp'),
               PIP_CACHE_DIR=str(WORK/'.cache/pip'), CONDA_PKGS_DIRS=str(WORK/'.cache/conda-pkgs'),
               CUDA_CACHE_PATH=str(WORK/'.cache/cuda'), MPLCONFIGDIR=str(WORK/'.cache/matplotlib'),
               CUPY_CACHE_DIR=str(WORK/'.cache/cupy'), TRITON_CACHE_DIR=str(WORK/'.cache/triton'),
               XDG_CACHE_HOME=str(WORK/'.cache'))
    prefix = Path('/root/miniconda3/envs/any2bokeh') if method == 'bokehme' else WORK/'envs'/method
    env['CONDA_PREFIX'] = str(prefix); env['PATH'] = str(prefix/'bin')+':'+env['PATH']
    env['CUDA_HOME'] = '/usr/local/cuda-12.8' if method in {'dofgaussian','bokehme'} else str(prefix)
    env['LD_LIBRARY_PATH'] = str(prefix/'lib')+':'+str(Path(env['CUDA_HOME'])/'lib64')+':'+env.get('LD_LIBRARY_PATH', '')
    return env


def gpu_empty():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return not output.strip()


class AdoptedProcess:
    """Observe a surviving child after a queue process is restarted."""
    def __init__(self, job, phase):
        self.pid=job['pid'];self.job=job;self.phase=phase;self.returncode=None
    def poll(self):
        try:
            if Path(f'/proc/{self.pid}/stat').read_text().split()[2]!='Z':return None
        except OSError:pass
        output=Path(self.job['output'])
        marker=output/('model-metadata.json' if self.phase=='train' else 'evaluation.json' if self.phase=='evaluate' else 'rendered/pairs.json')
        self.returncode=0 if marker.exists() else 1
        return self.returncode


def unknown_gpu_memory(active):
    known = {process.pid for process, _, _ in active.values()}
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    return sum(float(row.split(',')[1]) for row in rows.splitlines() if int(row.split(',')[0]) not in known)


def create_jobs(config):
    jobs = []
    for dataset, scenes in [('deblurnerf', config['real_scenes']), ('dofnerf-mix', config['dofnerf_scenes'])]:
        for scene in scenes:
            for method in config['methods']:
                if dataset == 'deblurnerf':
                    source = ROOT/('data/processed/deblurnerf-dofgaussian' if method == 'dofgaussian' else
                                   'data/raw/Deblurring3DGS/deblur_dataset/real_defocus_blur' if method == 'cocogaussian' else
                                   'data/raw/DeblurNeRF/real_defocus_blur')/('defocus'+scene)
                else:
                    source = ROOT/'data/processed'/('dofnerf-dofgaussian' if method == 'dofgaussian' else 'dofnerf-scenes')/'mix'/scene
                jobs.append({'id': method+'-'+dataset+'-'+scene, 'method': method, 'dataset': dataset, 'scene': scene,
                             'hold': HOLDS[scene] if dataset == 'deblurnerf' else 9, 'source': str(source),
                             'base_output': str(OUT/method/dataset/scene), 'priority': 0 if scene == 'cake' and dataset == 'deblurnerf' else 1 if dataset == 'deblurnerf' else 2,
                             'attempt': 0, 'status': 'queued'})
    for scene in config['real_scenes']:
        model = ROOT/'data/raw/DOFGS-author/real_results'/('real_'+scene)
        jobs.append({'id': 'author-'+scene, 'kind': 'author', 'method': 'dofgs', 'dataset': 'author-retrained',
                     'scene': scene, 'hold': HOLDS[scene], 'source': str(ROOT/'data/raw/DeblurNeRF/real_defocus_blur'/('defocus'+scene)),
                     'model': str(model), 'base_output': str(OUT/'dofgs/author-retrained'/scene), 'priority': 0, 'attempt': 0, 'status': 'queued'})
    for scene in config['dofnerf_scenes']:
        jobs.append({'id': 'bokehme-'+scene, 'kind': 'bokehme', 'method': 'bokehme', 'dataset': 'dofnerf-mix',
                     'scene': scene, 'hold': 9, 'source': str(ROOT/'data/processed/dofnerf-scenes/mix'/scene),
                     'base_output': str(OUT/'bokehme/dofnerf-mix'/scene), 'priority': 2, 'attempt': 0, 'status': 'dependency_wait'})
    return jobs


def dependency(job, prep):
    if job.get('kind') == 'bokehme':
        return 'Awaiting DOF-GS reconstructed training views' if not job.get('dofgs_output') or not (Path(job['dofgs_output'])/'rendered/pairs.json').exists() else None
    env = prep['environments'].get(job['method'], {})
    if env.get('rasterization', {}).get('status') != 'passed' or any(v.get('status') != 'passed' for v in env.get('imports', {}).values()):
        return 'Environment self-check/import not passed'
    dataset = 'deblurring3dgs-real' if job['method'] == 'cocogaussian' and job['dataset'] == 'deblurnerf' else 'deblurnerf-real' if job['dataset'] in {'deblurnerf', 'author-retrained'} else 'dofnerf-mix'
    checks = [r for r in prep['splits'] if r['method'] == job['method'] and r['dataset'] == dataset and r['scene'] == job['scene']]
    if len(checks) != 1 or checks[0].get('matches') is not True:
        return 'Official split reader not passed: '+str(checks[0].get('error', '') if checks else 'missing row')
    if job.get('kind') == 'author' and not (Path(job['model'])/'point_cloud').is_dir():
        candidates = [p for p in (ROOT/'data/raw/DOFGS-author/real_results').rglob('real_'+job['scene']) if (p/'point_cloud').is_dir()]
        if len(candidates) == 1: job['model'] = str(candidates[0])
        else: return 'Author retrained checkpoint unavailable'
    if job['dataset'] in {'deblurnerf', 'author-retrained'}:
        scene = prep['manifest'].get('real_scenes', {}).get(job['scene'], {})
        if not scene.get('matches_baseline'): return 'Raw image count/decoding/hold not passed'
        if job['method'] == 'cocogaussian':
            compare = prep['manifest'].get('sha256_comparisons', {}).get(job['scene'], {})
            if not compare.get('hold_matches') or not all(r.get('identical') for r in compare.get('directories', {}).values()):
                return 'CoCo image/hold comparison not passed'
    if job['method'] == 'dofgaussian':
        key = 'deblurnerf/'+job['scene'] if job['dataset'] == 'deblurnerf' else 'dofnerf/dense/mix/'+job['scene']
        if prep['colmap'].get('runs', {}).get(key, {}).get('status') != 'complete':
            return 'Dense reconstruction is incomplete'
        if prep['colmap'].get('visibility_parse_check', {}).get('status') != 'passed':
            return 'Modified visibility parser not passed'
    return None


def preparation_state():
    final = (PREP/'workflow.finished').exists()
    env_record = read(RECORD/'dof-gs-repro-2026-11-05-environments.json', {})
    environments = env_record.get('environments', {}) if final else {
        name: read(PREP/('environment-'+name+'.json'), {}) for name in ['dofgs','dofgaussian','mipsplatting','cocogaussian']}
    return {'environments': environments,
            'splits': read((RECORD/'dof-gs-repro-2026-11-05-split-check.json') if final else PREP/'split-check.json', {}).get('checks', []),
            'manifest': read((RECORD/'dof-gs-repro-2026-11-05-data-manifest.json') if final else PREP/'data-manifest.json', {}),
            'colmap': read((RECORD/'dof-gs-repro-2026-11-05-colmap.json') if final else PREP/'colmap.json', {}), 'final': final,
            'visibility': read(PREP/'visibility-check.json', {})}


def update_throughput_limit(state, job):
    """Reduce future concurrency if a completed batch fails to beat cake throughput."""
    if job['priority'] != 1: return
    efficiency=read(Path(job['output'])/'training-efficiency.json',{})
    cake=next(j for j in state['jobs'] if j['method']==job['method'] and j['dataset']=='deblurnerf' and j['scene']=='cake')
    baseline=read(Path(cake.get('output',cake['base_output']))/'training-efficiency.json',{}).get('total_training_call_seconds')
    if not baseline or efficiency.get('efficiency_valid'):return
    records=state.setdefault('packing_observations',[])
    records.append({'method':job['method'],'scene':job['scene'],'start':job['started_at'],'end':job['trained_at'],
                    'cake_equivalent_seconds':baseline*(1.75 if job['method']=='dofgs' and job['scene']=='coral' else 1),
                    'gpu_utilization':efficiency.get('sampling',{}).get('gaussian',{}).get('mean_gpu_utilization',0)})
    count=state.get('packing_last_count',0)
    if len(records)-count<6 or state['parallel_limit']<=1:return
    batch=records[count:];state['packing_last_count']=len(records)
    speedup=sum(r['cake_equivalent_seconds'] for r in batch)/max(1,max(r['end'] for r in batch)-min(r['start'] for r in batch))
    utilization=sum(r['gpu_utilization'] for r in batch)/len(batch)
    decision={'jobs':len(batch),'cake_normalized_throughput_ratio':speedup,'mean_gpu_utilization':utilization,
              'limit_before':state['parallel_limit'],'note':'scene complexity differs; cake-normalized heuristic, not a formal efficiency measurement'}
    if utilization>=90 and speedup<=1.05:state['parallel_limit']-=1
    decision['limit_after']=state['parallel_limit'];state.setdefault('packing_decisions',[]).append(decision)


def commit_report(files, message):
    def git(*args): return subprocess.check_output(['git', '-c', 'core.quotepath=false', *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    try:
        remote=git('remote','get-url','origin')
        if re.search(r'https?://[^/]*@',remote):raise RuntimeError('Credential-bearing remote rejected')
        git('fetch', 'origin'); branch = git('branch', '--show-current')
        ahead, behind = map(int, git('rev-list', '--left-right', '--count', 'HEAD...origin/'+branch).split())
        if behind:
            if ahead: raise RuntimeError('Remote/local diverged; no automatic history rewriting')
            git('merge', '--ff-only', 'origin/'+branch)
        staged = set(git('diff', '--cached', '--name-only').splitlines())
        if staged-set(files): raise RuntimeError('Unrelated staged edits preserved')
        def without_owned_status(text):
            for marker in ['DOF-GS 前期准备执行状态（2026-11-05）','3DGS 第一轮执行（2026-11-05）']:
                text=re.sub(r'\n\n\*\*'+re.escape(marker)+r'\*\*[^\n]*(?:\n(?!\n)[^\n]*)*','',text)
            return text
        progress_name='03_实验记录/当前实验进度与方法论.md'
        head_progress=subprocess.check_output(['git','show','HEAD:'+progress_name],cwd=ROOT,text=True)
        if without_owned_status((ROOT/progress_name).read_text())!=without_owned_status(head_progress):
            raise RuntimeError('Progress document has unrelated uncommitted edits; preserve them for review')
        for name in files:
            if (ROOT/name).stat().st_size>3*1024**2:raise RuntimeError('Oversized deliverable: '+name)
        git('diff', '--check', '--', *files); git('add', '--', *files); git('diff', '--cached', '--check')
        if not git('diff', '--cached', '--name-only'): return {'status': 'no_changes', 'commit': git('rev-parse', 'HEAD')}
        author, email = git('log', '-1', '--format=%an'), git('log', '-1', '--format=%ae')
        git('-c', 'user.name='+author, '-c', 'user.email='+email, 'commit', '-m', message)
        commit = git('rev-parse', 'HEAD'); git('push', 'origin', branch)
        return {'status': 'pushed', 'commit': commit}
    except Exception as exc:
        return {'status': 'blocked', 'error': str(exc)}


def _summarize(state, config, reason, commit=False):
    rows = []; training = []; author = []; r1 = []; regional = []; calibration = []
    for job in state['jobs']:
        result = read(Path(job.get('output', job['base_output']))/'evaluation.json', {})
        protocol = 'A' if job['dataset'] == 'deblurnerf' else 'r0'
        score = result.get('summary', {}).get(protocol, {}).get('unified', {})
        native = result.get('summary', {}).get(protocol, {}).get('native', {})
        metadata=read(Path(job.get('output',job['base_output']))/'model-metadata.json',{})
        rendering=read(Path(job.get('output',job['base_output']))/'rendered/pairs.json',{})
        rows.append({'method': job['method'], 'dataset': job['dataset'], 'scene': job['scene'], 'status': job['status'], **score,
                     **{'native_'+key: value for key,value in native.items() if key in {'psnr','ssim','lpips'}},
                     **{key:metadata[key] for key in ['gaussians','optimized_parameters','model_bytes','focal_planes_in_optimizer'] if key in metadata},
                     'reason': job.get('reason', ''), 'output': job.get('output', '')})
        if rendering.get('r1_training_check_passed') is False:
            rows[-1]['r1_status']='blocked: own-parameter training-view reproduction check failed; R0 retained'
        if 'ssim' in score:
            rows[-1]['unified_ssim']=score['ssim']
            if config.get('report_primary_ssim')=='native' and 'ssim' in native:
                rows[-1]['ssim']=native['ssim'];rows[-1]['ssim_protocol']='official native'
        for name, value in result.get('summary', {}).items():
            if not name.startswith('r1-'): continue
            r1.append({'method': job['method'], 'scene': job['scene'], 'mode': name[3:], 'views': value['views'], **value['unified']})
            for region, scores in value.get('regions', {}).items():
                regional.append({'method': job['method'], 'scene': job['scene'], 'mode': name[3:], 'region': region, **scores})
        if job.get('kind') == 'bokehme':
            values = read(Path(job.get('output', job['base_output']))/'calibration.json', {})
            if values: calibration.append(values)
        if job['priority'] == 0 and job.get('kind') != 'author':
            efficiency = read(Path(job.get('output', job['base_output']))/'training-efficiency.json', {})
            reference=efficiency
            measured=state.get('exclusive_training_measurements',{}).get(job['method'],{})
            if measured.get('status')=='complete':efficiency=read(Path(measured['output'])/'training-efficiency.json',efficiency)
            colmap=read(PREP/'colmap.json',{}).get('runs',{}).get('deblurnerf/cake',{})
            training.append({'method': job['method'], 'status': job['status'],
                'colmap_seconds':sum(value for stage in ['sparse','dense'] for value in colmap.get(stage,{}).get('seconds',{}).values()) if job['method']=='dofgaussian' and colmap else None,
                'geometry_source':'fresh official COLMAP pipeline' if job['method']=='dofgaussian' else 'provided original sparse model; original reconstruction time unavailable',
                'total_seconds':efficiency.get('total_training_call_seconds'),
                'gaussian_seconds': efficiency.get('gaussian_seconds'), 'depth_prior_seconds': efficiency.get('depth_prior_seconds'),
                'torch_gaussian_peak_bytes': efficiency.get('torch_gaussian_peak_bytes'),
                'torch_depth_prior_peak_bytes':efficiency.get('torch_depth_prior_peak_bytes'),
                'nvidia_depth_prior_peak_mib':efficiency.get('sampling',{}).get('depth_prior',{}).get('nvidia_peak_mib'),
                'nvidia_gaussian_peak_mib': efficiency.get('sampling', {}).get('gaussian', {}).get('nvidia_peak_mib'),
                'gpu_utilization': efficiency.get('sampling', {}).get('gaussian', {}).get('mean_gpu_utilization'), 'efficiency_valid': efficiency.get('efficiency_valid')})
            training[-1].update(measurement_label='exclusive formal' if efficiency.get('efficiency_valid') else 'parallel reference; exclusive measurement pendingP3',
                                initial_cake_training_seconds=reference.get('total_training_call_seconds'),
                                exclusive_supplement=measured)
    for job in state['jobs']:
        if job.get('kind') != 'author': continue
        result = read(Path(job.get('output', job['base_output']))/'evaluation.json', {})
        values=result.get('summary', {}).get('A', {})
        author.append({'scene': job['scene'], 'status': job['status'], **values.get('unified', {}),
                       **{'native_'+key:value for key,value in values.get('native',{}).items()}})
        if 'ssim' in values.get('unified',{}):
            author[-1]['unified_ssim']=values['unified']['ssim']
            if config.get('report_primary_ssim')=='native':author[-1]['ssim']=values.get('native',{}).get('ssim',author[-1]['ssim'])
    averages = []
    for method in config['methods']:
        for dataset in ['deblurnerf', 'dofnerf-mix']:
            completed = [row for row in rows if row['method'] == method and row['dataset'] == dataset and 'psnr' in row]
            row = {'method': method, 'dataset': dataset, 'completed_scenes': len(completed)}
            if completed:
                row.update({metric: sum(r[metric] for r in completed)/len(completed) for metric in ['psnr', 'ssim', 'lpips']})
                for metric in ['native_psnr','native_ssim','native_lpips','unified_ssim']:
                    available=[r[metric] for r in completed if metric in r]
                    if available:row[metric]=sum(available)/len(available)
            averages.append(row)
    r1_averages=[]
    for method in [*config['methods'],'bokehme']:
        for mode in ['fg','bg']:
            selected=[row for row in r1 if row['method']==method and row['mode']==mode]
            if selected:r1_averages.append({'method':method,'mode':mode,'completed_scenes':len(selected),**{key:sum(r[key] for r in selected)/len(selected) for key in ['psnr','ssim','lpips']}})
    author_complete=[row for row in author if 'psnr' in row]
    author_mean={key:sum(row[key] for row in author_complete if key in row)/sum(key in row for row in author_complete)
                 for key in ['psnr','ssim','lpips','native_psnr','native_ssim','unified_ssim'] if any(key in row for row in author_complete)}
    author_comparison={'completed_scenes':len(author_complete),'mean':author_mean,'checkpoint':'author retrained; original paper checkpoint was lost',
                       'paper':config['paper_reference']['dofgs_real_mean'],
                       'protocol_difference':'native SSIM uses skimage on [-1,1]; unified SSIM uses3DGS on[0,1]; native LPIPS unavailable'}
    if len(author_complete)==10:
        author_comparison['native_delta_from_paper']={metric:author_mean['native_'+metric]-config['paper_reference']['dofgs_real_mean'][metric] for metric in ['psnr','ssim']}
        author_comparison['unified_delta_from_paper']={metric:author_mean['unified_ssim' if metric=='ssim' else metric]-config['paper_reference']['dofgs_real_mean'][metric] for metric in ['psnr','ssim']}
    cake_comparison=[]
    for method in config['methods']:
        row=next(r for r in rows if r['method']==method and r['dataset']=='deblurnerf' and r['scene']=='cake')
        reference=config['paper_reference'].get(method+'_cake')
        cake_comparison.append({'method':method,'status':row['status'],'measured':row,'paper':reference,
                                'delta':{k:row[k]-reference[k] for k in ['psnr','ssim','lpips'] if k in row and k in reference} if reference else None})
    estimates={}
    for method in config['methods']:
        cake=next(j for j in state['jobs'] if j['method']==method and j['dataset']=='deblurnerf' and j['scene']=='cake')
        timing=read(Path(cake.get('output',cake['base_output']))/'training-efficiency.json',{})
        seconds=timing.get('total_training_call_seconds') if timing.get('status')=='trained' else None
        remaining=[j for j in state['jobs'] if j['method']==method and not j.get('kind') and j['status'] not in {'evaluated','failed','blocked','trained','rendered','rendering','evaluating'}]
        if seconds:estimates[method]={'remaining_training_jobs':len(remaining),'cake_seconds':seconds,
                                      'estimated_serial_training_seconds':seconds*sum(1.75 if j['scene']=='coral' and method=='dofgs' else 1 for j in remaining)}
    paper_means=[]
    for row in averages:
        if row['dataset']!='deblurnerf':continue
        reference=config['paper_reference'].get(row['method']+'_real_mean',{})
        comparison={'method':row['method'],'completed_scenes':row['completed_scenes'],'measured':row,'paper':reference}
        if row['completed_scenes']==reference.get('scenes'):
            comparison['delta']={key:row[key]-reference[key] for key in ['psnr','ssim','lpips'] if key in row and key in reference}
            comparison['native_delta']={key:row['native_'+key]-reference[key] for key in ['psnr','ssim','lpips'] if 'native_'+key in row and key in reference}
        paper_means.append(comparison)
    cake_table=[{'method':r['method'],'status':r['status'],**{k:r['measured'][k] for k in ['psnr','ssim','lpips','native_psnr','native_ssim','native_lpips'] if k in r['measured']},
                 **{'paper_'+k:v for k,v in (r['paper'] or {}).items() if k in ['psnr','ssim','lpips']},**{'delta_'+k:v for k,v in (r['delta'] or {}).items()}} for r in cake_comparison]
    summary = {'generated_at': datetime.now(timezone.utc).isoformat(), 'reason': reason, 'report_at': config['report_at'],
               'jobs': rows, 'averages_completed_scenes_only': averages, 'cake_training_efficiency': training,
               'author_retrained': author, 'author_paper_comparison':author_comparison,'cake_paper_comparison':cake_comparison,'method_paper_means':paper_means, 'paper_references': config['paper_reference'],
               'R1': r1, 'regional_R1': regional, 'bokehme_calibration': calibration,
               'R1_averages_completed_scenes_only':r1_averages,
               'memory_profiles_mib': state.get('memory_profiles', {}), 'parallel_limit': state.get('parallel_limit', 1),
               'parallel_throughput_decisions':state.get('packing_decisions',[]),
               'ssim_audit':read(RECORD/(PREFIX+'-ssim-audit.json'),{'status':'pending','report_primary':'official native'}),
               'deferred_due_to_colmap':[{'id':j['id'],'status':j['status'],**j['colmap_delay']} for j in state['jobs'] if j.get('colmap_delay')],
               'colmap_progress':read(PREP/'colmap.json',{}).get('runs',{}),
               'gpu_reservations':read(Path(os.environ['TMPDIR'])/'dof-gs-round1-gpu-reservations.json',{}),
               'throughput': state.get('throughput', {}), 'quality_gates': state.get('quality_gates', {}),
               'pending': [r for r in rows if r['status'] not in {'evaluated'}],
               'eta': {'measured_cake_based':estimates,'estimated_remaining_training_lower_bound_seconds':sum(v['estimated_serial_training_seconds'] for v in estimates.values())/max(1,state.get('parallel_limit',1)),
                       'scope':'rough training estimate only; excludes pending data, COLMAP, render/evaluation and P3; scene sizes and concurrent slowdown vary; methods without a completed cake remain unknown'},
               'regions': read(OUT/'regions/regions-approved.json', {'status': 'awaiting_visual_review'}),
               'benchmark': [read(p) for p in LOG.glob('benchmark-*.json') if read(p,{}).get('status') in {'benchmarked','failed'}],
               'self_written': ['official-function training wrapper/measurement and auxiliary serialization', 'R1 renderers', 'unified/native evaluation', 'CPU DA-V2 regions', 'BokehMe training-view calibration', 'CoCo offset-scaled E4']}
    summary['implementation_notes']=[
        'Updated2026-11-05 20:35+08:00: DoF-NeRF supplied poses/points are matched to sharp filenames, scaled to497x331, and radial camera/images rectified into PINHOLE derived views. Predictions/depth/alpha are reprojected to canonical original pixels for unified evaluation and BokehMe; original GT and masks are unchanged.',
        'DoF-NeRF native training view links images_4 as images and uses resolution1 to preserve exact497x331; scaling1326/4 would round height to332. Camera FOV is preserved.',
        'DOF-GS DoF-NeRF uses official cake configuration at40k; the public repository has no per-scene DoF-NeRF configuration. No test-based hyperparameter fitting.',
        'DOF-GS rendering preserves the saved band_limit/kernel_size settings, as official render.py does.',
        'DoF-Gaussian custom R1 uses10*activated_K and gamma2, matching official train.py, rather than render_bokeh function defaults.',
        'CoCo auxiliary kernel/focal-plane state is saved after timed training; camera UID mapping is restored from training metadata. Actual focal_planes optimizer membership is recorded per model without changing official training.',
        'CoCo E4 scales learned position offsets only; scale/rotation heads remain as learned. Zero radius uses the base Gaussian renderer.',
        'Native SSIM on[-1,1] and native Alex LPIPS can differ from unified3DGS SSIM and VGG LPIPS; both are preserved. DOF-GS native metric has no LPIPS.',
        'Author checkpoints are retrained releases, not the original paper checkpoints. Paper comparison is marked incomplete until all10 scenes are evaluated.'
    ]
    summary['implementation_notes'].append('CoCoGaussian supplementary Table5 includes book and omits turtle; the user explicitly requested six validated scenes excluding both, so its published seven-scene DoF-NeRF mean is not directly compared.')
    RECORD.mkdir(parents=True, exist_ok=True)
    write(RECORD/(PREFIX+'-summary.json'), summary)
    for name, table in [('A-and-R0', rows), ('averages', averages), ('training-efficiency', training), ('author-retrained', author), ('cake-paper',cake_table), ('R1',r1), ('R1-averages',r1_averages), ('regional-R1',regional)]:
        fields = sorted(set().union(*(r.keys() for r in table))) if table else ['status']
        with (RECORD/(PREFIX+'-'+name+'.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader(); writer.writerows(table)
    if reason.startswith('stage'):
        for suffix in ['summary.json','A-and-R0.csv','averages.csv','training-efficiency.csv','author-retrained.csv','cake-paper.csv','R1.csv','R1-averages.csv','regional-R1.csv']:
            snapshot=RECORD/(PREFIX+'-stage-'+suffix)
            if not snapshot.exists():shutil.copy2(RECORD/(PREFIX+'-'+suffix),snapshot)
    progress = ROOT/'03_实验记录/当前实验进度与方法论.md'
    text = progress.read_text(); marker = '**3DGS 第一轮执行（2026-11-05）**'
    done = sum(r['status'] == 'evaluated' for r in rows); running = sum(r['status'] in {'training', 'rendering', 'evaluating'} for r in rows)
    paragraph = marker+f'：已完成 {done}/{len(rows)} 个方法×场景，运行中 {running} 个；作者重训即评，COLMAP最多两路与训练并行，总显存预算40GiB、最多4个GPU任务；受并行影响的cake计时标参考，P3补独占。\n阶段汇总见 [round1 summary](records/'+PREFIX+'-summary.json)；SSIM主列保留官方口径，统一3DGS口径及差异核查另列；平均值仅包含实际完成场景，报告时间 '+config['report_at']+'，到点汇总不停止队列。'
    if (RECORD/(PREFIX+'-stage-summary.json')).exists():paragraph+=' 截止时刻快照见 [stage summary](records/'+PREFIX+'-stage-summary.json)。'
    if marker in text:
        start = text.index(marker); end = text.find('\n\n', start); text = text[:start]+paragraph+text[end if end >= 0 else len(text):]
    else:
        start = text.index('**DOF-GS 前期准备执行状态'); end = text.find('\n\n', start); text = text[:end]+'\n\n'+paragraph+text[end:]
    progress.write_text(text)
    if commit:
        files = [str(p.relative_to(ROOT)) for p in sorted(RECORD.glob(PREFIX+'-*'))]
        files += ['03_实验记录/当前实验进度与方法论.md', 'configs/dof-gs-round1-20261105.json']
        files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'scripts').glob('*gs_round1*.py'))]
        files += ['scripts/prepare_dofnerf_scenes.py', 'scripts/prepare_gs_repro.py']
        if (PREP/'workflow.finished').exists() and read(PREP/'git-result.json', {}).get('status') != 'pushed':
            files += [str(p.relative_to(ROOT)) for p in sorted(RECORD.glob('dof-gs-repro-2026-11-05-*')) if not p.name.startswith(PREFIX)]
        result=commit_report(files, '记录3DGS第一轮'+reason+'：实际完成场景与运行队列')
        write(LOG/('git-'+reason+'.json'), result)
        return result


def post_benchmarks(state,config):
    marker=Path(os.environ['TMPDIR'])/'dof-gs-round1-exclusive.pending'
    marker.write_text(str(os.getpid()))
    try:
        for method in config['methods']:
            for scene in config['benchmark_scenes']:
                job=next(j for j in state['jobs'] if j['method']==method and j['dataset']=='deblurnerf' and j['scene']==scene)
                if not job.get('output') or not (Path(job['output'])/'model-metadata.json').exists():continue
                destination=LOG/('benchmark-'+method+'-'+scene+'.json')
                if read(destination,{}).get('status')=='benchmarked':continue
                while not gpu_empty():time.sleep(5)
                bench=dict(job,exclusive=True,model=job['output'],benchmark_output=str(destination))
                path=LOG/('benchmark-job-'+method+'-'+scene+'.json');write(path,bench)
                state['postprocessing']={'phase':'benchmark','method':method,'scene':scene};write(LOG/'queue.json',state)
                process=command([WORK/'envs'/method/'bin/python',ROOT/'scripts/gs_round1_method.py','benchmark','--job',path],
                                LOG/('benchmark-'+method+'-'+scene+'.log'),environment(method))
                code=process.wait()
                if code:
                    process=command([WORK/'envs'/method/'bin/python',ROOT/'scripts/gs_round1_method.py','benchmark','--job',path],
                                    LOG/('benchmark-'+method+'-'+scene+'-retry.log'),environment(method))
                    if process.wait():write(destination,{'status':'failed','method':method,'scene':scene,'reason':'benchmark failed twice; see logs'})
    finally:
        if marker.exists():marker.unlink()
    state['postprocessing']={'phase':'finished'};write(LOG/'queue.json',state)


def post_training_measurements(state,config):
    """Supplement contaminated cake timing only after the quality queue finishes."""
    marker=Path(os.environ['TMPDIR'])/'dof-gs-round1-exclusive.pending'
    continuation=PREP/'round1-colmap-continuation.finished'
    if (PREP/'round1-colmap-continuation.pid').exists():
        while not continuation.exists():time.sleep(10)
    marker.write_text(str(os.getpid()))
    try:
        for method in config['methods']:
            cake=next(j for j in state['jobs'] if j['method']==method and j['dataset']=='deblurnerf' and j['scene']=='cake')
            if not cake.get('output') or not (Path(cake['output'])/'model-metadata.json').exists():continue
            initial=read(Path(cake['output'])/'training-efficiency.json',{})
            if initial.get('efficiency_valid'):continue
            output=Path(cake['base_output'])/'efficiency-exclusive'
            entry=state.setdefault('exclusive_training_measurements',{}).setdefault(method,{'output':str(output),'status':'pending'})
            if (output/'model-metadata.json').exists() and read(output/'training-efficiency.json',{}).get('efficiency_valid'):
                entry['status']='complete';continue
            while not gpu_empty():time.sleep(5)
            job=dict(cake,id='exclusive-efficiency-'+method,output=str(output),exclusive=True,reserved_memory_mib=state['memory_profiles'].get(method,16384))
            path=LOG/('exclusive-efficiency-'+method+'.json');write(path,job)
            entry.update(status='running',started_at=time.time());write(LOG/'queue.json',state)
            process=command([WORK/'envs'/method/'bin/python',ROOT/'scripts/gs_round1_method.py','train','--job',path],LOG/('exclusive-efficiency-'+method+'.log'),environment(method))
            code=process.wait()
            if code:
                retry=output.with_name('efficiency-exclusive-retry');job['output']=str(retry);entry['output']=str(retry);write(path,job)
                process=command([WORK/'envs'/method/'bin/python',ROOT/'scripts/gs_round1_method.py','train','--job',path],LOG/('exclusive-efficiency-'+method+'-retry.log'),environment(method));code=process.wait()
            entry.update(status='complete' if code==0 else 'failed',finished_at=time.time());write(LOG/'queue.json',state)
    finally:
        if marker.exists():marker.unlink()


def summarize(state, config, reason, commit=False):
    with (Path(os.environ['TMPDIR'])/'dof-gs-round1-report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _summarize(state,config,reason,commit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/dof-gs-round1-20261105.json')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--reason', default='stage')
    args = parser.parse_args(); LOG.mkdir(parents=True, exist_ok=True)
    config = read(args.config); state_path = LOG/'queue.json'
    state = read(state_path, {'jobs': create_jobs(config), 'memory_profiles': {}, 'parallel_limit': config['max_parallel'], 'quality_gates': {}, 'started_at': time.time()})
    for job in state['jobs']:
        if job.get('kind')=='author' and config.get('author_replay_immediate'):
            job['allow_gpu_overlap']=True
        if job['method']=='dofgaussian' and job['scene']!='cake' and job['status']=='dependency_wait':
            job.setdefault('colmap_delay',{'reason':'awaiting missing COLMAP reconstruction, continuation requested2026-11-05 20:35+08:00','observed_at':time.time()})
        if job['dataset']=='dofnerf-mix' and job.get('kind')!='bokehme' and job['status'] in {'queued','dependency_wait','blocked'}:
            directory='dofnerf-dofgaussian-provided' if job['method']=='dofgaussian' else 'dofnerf-scenes-provided'
            target=ROOT/'data/processed'/directory/'mix'/job['scene']
            if (ROOT/'data/processed/dofnerf-scenes-provided/mix'/job['scene']/'rectification.json').exists():
                job['source']=str(target)
                job['geometry_protocol']='supplied poses; exact497x331 rectification; evaluation reprojected to original pixels'
    if args.report_only:
        result=summarize(state,config,args.reason,commit=True)
        if result.get('status')=='blocked':raise SystemExit(2)
        return
    instance = (Path(os.environ['TMPDIR'])/'dof-gs-round1-queue.lock').open('a')
    fcntl.flock(instance,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(state_path, state)
    waiting_report_written = False
    while not (PREP/'priority-cake-prep.finished').exists() and not (PREP/'workflow.finished').exists():
        write(LOG/'waiting.json', {'status': 'waiting_for_prep', 'time': time.time(), 'report_at': read(args.config)['report_at']})
        config = read(args.config)
        if time.time() >= datetime.fromisoformat(config['report_at']).timestamp() and not waiting_report_written:
            summarize(state, config, 'stage-waiting-for-prep', commit=True); waiting_report_written = True
        if shutil.disk_usage('/').free < 1024**3: raise RuntimeError('System disk below1GiB')
        time.sleep(15)
    prep = preparation_state()
    if 'visibility_parse_check' not in prep['colmap']: prep['colmap']['visibility_parse_check'] = prep['visibility']
    for job in state['jobs']:
        if job['status'] not in {'queued','dependency_wait','blocked'}:continue
        reason = dependency(job, prep)
        if reason: job.update(status='blocked' if prep['final'] else 'dependency_wait', reason=reason)
    active = {}; cpu = {}; last_summary = 0; report_written = waiting_report_written
    for job in state['jobs']:
        if job['status'] in {'training','rendering','evaluating'} and job.get('pid'):
            phase=job.get('phase','evaluate' if job['status']=='evaluating' else 'train' if job['status']=='training' else 'render')
            process=AdoptedProcess(job,phase)
            if phase=='evaluate':cpu[job['id']]=(process,job)
            else:active[job['id']]=(process,job,phase)
    def finish_gpu(job, phase, code):
        if code:
            job.update(status='retry' if job['attempt'] < 2 and phase in {'train','bokehme'} else
                       'render_retry' if phase == 'render' and job.get('render_attempt', 0) < 2 else 'failed', reason=f'{phase} exit={code}')
            return
        if phase == 'train':
            job['trained_at'] = time.time(); job['status'] = 'trained'
            update_throughput_limit(state,job)
            if job['priority'] == 0:
                data = read(Path(job['output'])/'training-efficiency.json', {})
                interference=read(Path(job['output'])/'timing-interference.json',{})
                if interference:
                    data.update(efficiency_valid=False,measurement_label='parallel reference',interference=interference)
                    write(Path(job['output'])/'training-efficiency.json',data)
                peak = max((r.get('nvidia_peak_mib', 0) for r in data.get('sampling', {}).values()), default=0)
                state['memory_profiles'][job['method']] = math.ceil(peak*1.1+512)
            return
        job['status'] = 'rendered'
        if phase=='bokehme':
            memory=read(Path(job['output'])/'bokehme-efficiency.json',{}).get('nvidia_peak_mib',0)
            state['memory_profiles']['bokehme']=math.ceil(memory*1.1+512)
    while True:
        config = read(args.config); now = time.time()
        prep = preparation_state()
        if 'visibility_parse_check' not in prep['colmap']: prep['colmap']['visibility_parse_check'] = prep['visibility']
        for job in state['jobs']:
            if job.get('kind') == 'bokehme':
                parent = next(j for j in state['jobs'] if j['method']=='dofgs' and j['dataset']=='dofnerf-mix' and j['scene']==job['scene'])
                if parent.get('output'): job['dofgs_output'] = parent['output']
                if parent['status'] in {'blocked','failed'}:
                    job.update(status='blocked',reason='DOF-GS parent '+parent['status']); continue
            if job['status'] not in {'dependency_wait', 'blocked'} or job.get('reason', '').startswith('Cake quality'): continue
            reason = dependency(job, prep)
            if not reason:
                job.update(status='queued', reason='')
                if job.get('colmap_delay'):job['colmap_delay'].setdefault('resolved_at',now)
            elif prep['final']: job.update(status='blocked', reason=reason)
        if shutil.disk_usage('/').free < 1024**3: raise RuntimeError('System disk below1GiB; queue stopped')
        for key, (process, job, phase) in list(active.items()):
            if process.poll() is None: continue
            finish_gpu(job, phase, process.returncode); active.pop(key)
        for key, (process, job) in list(cpu.items()):
            if process.poll() is None: continue
            job['status'] = 'evaluated' if process.returncode == 0 else 'failed'; job['finished_at'] = now
            cpu.pop(key)
            if job['dataset']=='deblurnerf' and job['scene']=='cake' and config['paper_reference'].get(job['method']+'_cake') and process.returncode == 0:
                result = read(Path(job['output'])/'evaluation.json', {})['summary']['A']['unified']['psnr']
                delta = result-config['paper_reference'][job['method']+'_cake']['psnr']
                state['quality_gates'][job['method']] = {'cake_psnr_delta': delta, 'passed': abs(delta) <= config['quality_gate_psnr_db'],
                    'diagnosis': 'Check canonical/native GT, resolution, hold and renderer before expanding if outside threshold'}
            if job.get('kind') == 'author' and job['scene'] == 'cake' and process.returncode == 0:
                reference = read(Path(job['output'])/'evaluation.json', {}).get('summary', {}).get('A', {}).get('unified', {}).get('psnr')
                trained = next(j for j in state['jobs'] if j['method'] == 'dofgs' and j['dataset'] == 'deblurnerf' and j['scene'] == 'cake')
                value = read(Path(trained.get('output', trained['base_output']))/'evaluation.json', {}).get('summary', {}).get('A', {}).get('unified', {}).get('psnr')
                if reference is not None and value is not None:
                    delta = value-reference
                    state['quality_gates']['dofgs'] = {'cake_psnr_delta': delta, 'reference': 'author retrained checkpoint, not original paper checkpoint',
                                                       'passed': abs(delta) <= config['quality_gate_psnr_db']}
        for job in state['jobs']:
            if job['status'] == 'rendered' and len(cpu) < 2:
                destination = Path(job['output'])/'evaluation.json'
                arguments = [WORK/'envs/gs-base-cu118/bin/python', ROOT/'scripts/evaluate_gs_round1.py',
                             '--manifest', Path(job['output'])/'rendered/pairs.json', '--output', destination]
                if (OUT/'regions/regions-approved.json').exists(): arguments += ['--regions', OUT/'regions']
                process = command(arguments, LOG/(job['id']+'-evaluate.log'), environment('gs-base-cu118'))
                cpu[job['id']] = (process, job); job['status'] = 'evaluating';job['phase']='evaluate';job['pid']=process.pid
        unfinished_cake = [j for j in state['jobs'] if j['priority'] == 0 and j['status'] not in {'evaluated', 'failed', 'blocked', 'dependency_wait'}]
        unfinished_A = [j for j in state['jobs'] if j['priority'] == 1 and j['status'] not in {'evaluated', 'failed', 'blocked', 'dependency_wait'}]
        priority = 0 if unfinished_cake else 1 if unfinished_A else 2
        for job in state['jobs']:
            if job['priority'] > 0 and state['quality_gates'].get(job['method'], {}).get('passed') is False and job['status'] == 'queued':
                job.update(status='blocked', reason='Cake quality gate requires diagnosis before expansion')
            cake_failed = any(j['method'] == job['method'] and j['dataset'] == 'deblurnerf' and j['scene'] == 'cake' and j['status'] in {'failed', 'blocked'} for j in state['jobs'])
            if job.get('kind')!='bokehme' and job['priority'] > 0 and cake_failed and not state['memory_profiles'].get(job['method']) and job['status'] == 'queued':
                job.update(status='blocked', reason='No successful exclusive cake memory measurement')
        retries = [j for j in state['jobs'] if j['status'] in {'retry', 'render_retry'}]
        exclusive_pending = Path(os.environ['TMPDIR'])/'dof-gs-round1-exclusive.pending'
        if (unfinished_cake and not config.get('concurrent_preparation')) or retries:
            exclusive_pending.write_text(str(os.getpid()))
        elif exclusive_pending.exists():
            exclusive_pending.unlink()
        if config.get('concurrent_preparation'):
            candidates=[j for j in state['jobs'] if j['status'] in {'queued','trained'} and
                        (j['priority']==0 or j['priority']==1 and state['memory_profiles'].get(j['method']) or priority==2 and j['priority']==2)]
            candidates.sort(key=lambda j:(0 if j.get('kind')=='author' else 1 if j['priority']==0 else 2,j['priority']))
        else:
            candidates = [j for j in state['jobs'] if j['priority'] == priority and j['status'] in {'queued', 'trained'}]
        if priority==2:candidates.sort(key=lambda job:job.get('kind')!='bokehme')
        if retries and not active: candidates = retries[:1]
        if not retries or not active:
            for job in candidates:
                parallel_prep=config.get('concurrent_preparation',False)
                if len(active) >= (1 if retries or priority==0 and not parallel_prep else state['parallel_limit']): break
                if job.get('kind')=='author' and any(j.get('kind')=='author' for _,j,_ in active.values()):continue
                if job['priority']==0 and job.get('kind')!='author' and any(j['priority']==0 and j.get('kind')!='author' for _,j,_ in active.values()):continue
                exclusive = bool(retries) or job['priority']==0 and not parallel_prep
                if exclusive and (active or not gpu_empty()): break
                reserve = state['memory_profiles'].get(job['method'], config['unknown_memory_mib'])
                total = unknown_gpu_memory(active)+sum(state['memory_profiles'].get(j['method'], config['unknown_memory_mib']) for _, j, _ in active.values())
                if not exclusive and total+reserve > config['memory_budget_mib']: continue
                phase = 'bokehme' if job.get('kind')=='bokehme' else 'render' if job['status'] in {'trained', 'render_retry'} or job.get('kind') == 'author' else 'train'
                if phase in {'train','bokehme'}:
                    job['attempt'] += 1; job['output'] = str(Path(job['base_output'])/('attempt-'+str(job['attempt'])))
                    job['exclusive'] = exclusive; job['started_at'] = now
                if phase == 'render':
                    job['exclusive'] = exclusive
                    job['render_attempt'] = job.get('render_attempt', 0)+1
                    if 'output' not in job:
                        job['output'] = str(Path(job['base_output'])/'attempt-1'); job['exclusive'] = exclusive
                jobfile = LOG/(job['id']+'.json'); write(jobfile, job)
                job['reserved_memory_mib']=reserve
                if not exclusive:job['measurement_label']='parallel reference' if job['priority']==0 else 'parallel'
                write(jobfile,job)
                arguments = [Path('/root/miniconda3/envs/any2bokeh/bin/python'),ROOT/'scripts/gs_round1_bokehme.py','--job',jobfile] if phase=='bokehme' else [WORK/'envs'/job['method']/'bin/python', ROOT/'scripts/gs_round1_method.py', phase, '--job', jobfile]
                process = command(arguments,
                                  LOG/(job['id']+'-'+phase+'-'+str(job['attempt'])+'.log'), environment(job['method']))
                active[job['id']] = (process, job, phase); job['status'] = 'training' if phase in {'train','bokehme'} else 'rendering'; job['pid'] = process.pid; job['phase']=phase
        state['throughput'] = {'completed_trainings': sum('trained_at' in j for j in state['jobs']),
                               'elapsed_since_first_training_seconds': now-min([j['started_at'] for j in state['jobs'] if 'started_at' in j] or [now]),
                               'parallel_times_are_reference_only': True}
        write(state_path, state)
        due = datetime.fromisoformat(config['report_at']).timestamp()
        if now >= due and not report_written:
            summarize(state, config, 'stage', commit=True); report_written = True
        elif now-last_summary > 300:
            summarize(state, config, 'running'); last_summary = now
        if not active and not cpu and all(j['status'] in {'evaluated', 'failed', 'blocked'} for j in state['jobs']): break
        time.sleep(10)
    write(LOG/'training-queue.finished', {'finished_at': time.time()})
    summarize(state, config, 'training-complete', commit=True)
    post_training_measurements(state,config)
    post_benchmarks(state,config)
    plot=command([WORK/'envs/dofgs/bin/python',ROOT/'scripts/plot_gs_round1.py'],LOG/'plot.log',environment('dofgs'))
    plot.wait()
    summarize(state,config,'complete',commit=True)
    write(LOG/'round1.finished',{'finished_at':time.time()})


if __name__ == '__main__': main()
