"""B-only follow-up dry runs and phased legacy F2 preparation."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

from route3_v2data_B_batch import blender, run as run_command, sha256
from route3_v2data_B_f2 import cache_optical
from route3_v2data_B_front import RECORD, ROOT, write_json
from route3_v2data_B_inputs import FORMAL_DATA, observation
from route3_v2data_B_optical_pair import stage_optical
from route3_v2data_B_plan import load
from route3_v2data_dryrun import (
    REPORT, SEED, START, fingerprint, load_model, locked_gpu, stage_objective, statistics, summarize,
)
from route3_v2data_loader import TrainingData, legacy
from route3_ovg_subpixel import TRAIN_FRAMES
from src.ovg_appearance import AppearanceHeads


class QualificationData(TrainingData):
    """Use an isolated training qualification without registering it for production."""
    def __init__(self, *units):
        super().__init__()
        self.qualification = {unit.name:unit for unit in units}
        self.b.update(self.qualification)

    def load(self, item, device):
        unit = self.qualification.get(item[1]) if item[0]=='B' else None
        if unit is None:
            return super().load(item, device)
        with patch.object(legacy, 'formal_unit', lambda name:unit if name==unit.name else None):
            return legacy.sample(item, device, b_loader=lambda _,frame:observation(unit,frame,aligned=True))

    def pair_reference(self, item, target):
        unit = self.qualification.get(item[1]) if item[0]=='B' else None
        if unit is None:
            return super().pair_reference(item, target)
        with patch.object(legacy, 'formal_unit', lambda name:unit if name==unit.name else None):
            legacy.b_foreground_reference.cache_clear()
            return super().pair_reference(item, target)


def item_info(item):
    return item[1] if item[0]=='B' else Path(item[1]).parent.name


def run_batches(inventory, plan, *, validate_support=True):
    net = load_model(START,'model',device='cpu').train()
    head = AppearanceHeads(net.coverage).train()
    baseline = fingerprint(net,head)
    rows, peak = [], 0
    for index, (stage,item,spec) in enumerate(plan):
        with locked_gpu(net,head):
            generator = torch.Generator(device='cuda').manual_seed(SEED+index)
            terms,counts,weights = stage_objective(
                net,inventory,item,stage,generator,spec=spec,
                audit_k16=not validate_support, audit_transmission=spec is not None,
            )
            parameters = tuple(value for value in net.parameters() if value.requires_grad)
            losses = statistics(terms,counts,weights,parameters,1.)
            rows.append(dict(stage=stage,tier=item[0],scene=item_info(item),index=item[2],
                             input_frame=TRAIN_FRAMES[item[2]] if item[0]=='B' else item[2],
                             optical=({key:spec[key] for key in ('frame','focus','fstop')} if spec else None),
                             support=counts,losses=losses))
            peak = max(peak,torch.cuda.max_memory_reserved())
            print('B_DRYRUN_P2',stage,item[0],item_info(item),rows[-1]['input_frame'],
                  'occ',counts.get('occ'),'bokeh',counts.get('bokeh'),flush=True)
    unchanged = fingerprint(net,head)==baseline
    failures = [] if unchanged else ['model state changed']
    for row in rows:
        required = {'disp','mv','bimodal','alpha','same_surface'}
        if row['stage']=='1a':
            required.add('normal')
        else:
            required.update(('zero','occ'))
        if row['optical']:
            required.add('bokeh')
        missing = required-row['losses'].keys()
        if missing:
            failures.append(f"{row['scene']}/{row['input_frame']} missing terms {sorted(missing)}")
        if not row['optical'] and 'bokeh' in row['losses']:
            failures.append(f"{row['scene']}/{row['input_frame']} used bokeh without an optical reference")
        if any(not s['finite'] or not s['gradient_tensors'] for s in row['losses'].values()):
            failures.append(f"{row['scene']}/{row['input_frame']} nonfinite or disconnected gradient")
        if row['stage']=='1b' and (row['tier']=='B' or row['optical']) and not (
            row['support'].get('occ_reveal',0)+row['support'].get('occ_block',0)
        ):
            failures.append(f"{row['scene']}/{row['input_frame']} no effective K16 occlusion rays")
        if row['optical'] and not row['support']['bokeh']:
            failures.append(f"{row['scene']}/{row['input_frame']} no effective optical supervision")
    aggregate = {}
    for stage,tier in sorted({(row['stage'],row['tier']) for row in rows}):
        block = summarize([row for row in rows if (row['stage'],row['tier'])==(stage,tier)])
        aggregate[f'{stage}/{tier}'] = block
        for name,loss in block['losses'].items():
            if loss['supervision']==0:
                failures.append(f'{stage}/{tier}/{name} no effective supervision')
    return dict(passed=not failures,failures=failures,optimizer_steps=0,model_state_unchanged=unchanged,
                gpu_peak_reserved_bytes=peak,by_stage=aggregate,batches=rows)


def transmission_report(rows):
    result = {}
    for tier in ('B','C'):
        batches = []
        scene_totals = defaultdict(Counter)
        for row in rows:
            if row['tier']!=tier or not row['optical']:
                continue
            source = row['support']
            counts = {key:source[key] for key in
                      ('bokeh_rays','uncovered_rays','rays_T_ge_0p1','residual_transmittance_sum')}
            scene_totals[row['scene']].update(counts)
            batches.append(dict(scene=row['scene'],frame=row['input_frame'],case=row['optical'],
                                **ratios(counts)))
        fields = ('uncovered_fraction','fraction_T_ge_0p1','mean_T','observed_prior_grid_fallback_fraction')
        scene_rows = [dict(scene=name,**ratios(counts)) for name,counts in scene_totals.items()]
        result[tier] = dict(
            source="legacy A-color/LaMa F2" if tier=='B' else "no observed-prior grid fallback",
            batches=batches, scenes=scene_rows,
            percentiles_by_batch={key:percentiles(batches,key) for key in fields},
            percentiles_by_scene={key:percentiles(scene_rows,key) for key in fields},
        )
    return result


def ratios(counts):
    n = counts['bokeh_rays']
    return dict(rays=n,uncovered_fraction=counts['uncovered_rays']/n,
                fraction_T_ge_0p1=counts['rays_T_ge_0p1']/n,mean_T=counts['residual_transmittance_sum']/n,
                observed_prior_grid_fallback_fraction=0.)


def percentiles(rows, key):
    return dict(zip(('median','p95'),np.quantile([row[key] for row in rows],[.5,.95]).tolist())) if rows else None


def c_controls(inventory, count):
    choices = []
    cases_per_scene = Counter()
    for item in inventory.items['C']:
        unique = item_info(item)
        if cases_per_scene[unique]:
            continue
        specs = inventory.optical_specs(item)
        if not specs:
            continue
        choices.append((item,specs[0]))
        cases_per_scene[unique] += 1
        if len(choices)==count:
            return choices
    raise ValueError('Not enough C optical comparators in the training inventory')


def geometry_probes(unit):
    inventory = QualificationData(unit)
    c_item, _ = c_controls(inventory,1)[0]
    c_native_only = next((key for key in inventory.items['C'] if inventory.k16_specs(key)
                          and not inventory.optical_specs(key)), None)
    cases = [('1b',('B',unit.name,TRAIN_FRAMES.index(6)),None), ('1b',c_native_only or c_item,None)]
    result = run_batches(inventory,cases,validate_support=False)
    result['c_geometry_probe_reference_withheld'] = c_native_only is None
    return result


def qualification_checks(unit, destination, f2_root):
    inventory = QualificationData(unit)
    old_f2 = f2_root/unit.name/'f030'/'manifest.json'
    if not old_f2.is_file():
        raise FileNotFoundError('build the B training qualification F2 cache first')
    optical = destination/'b-f030'
    if not (optical/'labels/optical-audit.json').is_file():
        stage_optical(unit,optical)
        run_command(blender('route3_v2data_B_render.py','--scene',optical,'--optical-only','--frames',30,threads=2),
                    optical/'render.log',uses_gpu=True)
    reference = load(optical/'labels/optical-audit.json')['optical'][0]
    b_spec = dict(tiers='B',unit=unit,frame=30,focus=reference['focus_m'],fstop=reference['fstop'],
                  truth=optical/'labels'/f"{reference['id']}-optics.npz",f2_root=f2_root)
    c_item, c_spec = c_controls(inventory,1)[0]
    cases = [('1b',('B',unit.name,TRAIN_FRAMES.index(30)),b_spec),('1b',c_item,c_spec)]
    result = run_batches(inventory,cases,validate_support=False)
    result['F2_cache_manifest_sha256']=sha256(old_f2)
    result['transmission']=transmission_report(result['batches'])
    return result


def geometry_plan(inventory):
    by_category = defaultdict(list)
    for name,unit in sorted(inventory.b.items()):
        manifest = load(unit/'manifest.json')
        if manifest['generator_version'].endswith('-p2'):
            by_category[manifest['category']].append(name)
    if len(by_category)!=6 or min(map(len,by_category.values()))<2:
        return []
    selected = [name for category in sorted(by_category) for name in by_category[category][:2]]
    return [(stage,('B',selected[i%len(selected)],(i//len(selected)*5+i)%len(TRAIN_FRAMES)),None)
            for stage,count in (('1a',21),('1b',70)) for i in range(count)]


def optical_plan(inventory,names):
    rows = []
    for name in names:
        for frame in (1,30,60):
            item = ('B',name,TRAIN_FRAMES.index(frame))
            for spec in inventory.optical_specs(item):
                rows.append(('1b',item,spec))
    return rows+[('1b',item,spec) for item,spec in c_controls(inventory,len(rows))] if rows else []


def publish(values, checkout):
    path = checkout/'03_实验记录/records'/REPORT.name
    lock = checkout/'.git'
    if checkout==ROOT or not lock.exists():
        raise ValueError('publication must use a separate Git worktree')
    subprocess.run(['git','pull','--rebase','origin','main'],cwd=checkout,check=True)
    original = load(path)
    original.setdefault('b_p2_followup',{}).update(values)
    write_json(path,original)
    subprocess.run(['git','diff','--check'],cwd=checkout,check=True)
    subprocess.run(['git','add','--',str(path.relative_to(checkout))],cwd=checkout,check=True)
    changed = subprocess.run(['git','diff','--cached','--quiet'],cwd=checkout)
    if changed.returncode==1:
        subprocess.run(['git','commit','-m','记录 B v2 p2 空跑和 F2 回退覆盖率',
                        '-m','Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'],cwd=checkout,check=True)
        subprocess.run(['git','pull','--rebase','origin','main'],cwd=checkout,check=True)
        subprocess.run(['git','push','origin','HEAD:main'],cwd=checkout,check=True)
    elif changed.returncode:
        changed.check_returncode()


def result_reference(path):
    return dict(path=str(path.resolve().relative_to(FORMAL_DATA.resolve())),sha256=sha256(path))


def geometry_publication(path):
    result = load(path)
    return dict(**result_reference(path),passed=result["passed"],batches=result["by_stage"],
                failures=result["failures"])


def monitor(output, checkout, production_record=RECORD):
    output.mkdir(parents=True,exist_ok=True)
    created, failed = set(), {}
    first_optical = []
    prior_geometry = output/'geometry.json'
    if prior_geometry.is_file():
        summary = geometry_publication(prior_geometry)
        registered = load(checkout/'03_实验记录/records'/REPORT.name).get('b_p2_followup',{}).get('geometry',{})
        if summary.get('sha256')!=registered.get('sha256'):
            publish({'geometry':summary},checkout)
    while True:
        state = load(FORMAL_DATA/'manifest.json')
        if state.get('generator_patch',{}).get('version','').endswith('-p2'):
            if not (output/'geometry.json').exists():
                inventory = TrainingData()
                plan = geometry_plan(inventory)
                if plan:
                    result = run_batches(inventory,plan)
                    write_json(output/'geometry.json',result)
                    publish({'geometry':geometry_publication(output/'geometry.json')},checkout)
            inventory = TrainingData()
            available = [row['scene_id'] for row in state['optical_units'] if row['scene_id'] in inventory.b]
            if not first_optical:
                first_optical = list(available)
            for name in available:
                if name in created or name in failed:
                    continue
                try:
                    cache_optical(inventory.b[name],FORMAL_DATA/'optical'/name)
                    created.add(name)
                except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
                    try:
                        cache_optical(inventory.b[name],FORMAL_DATA/'optical'/name)
                        created.add(name)
                    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as retry_error:
                        failed[name] = str(retry_error)
                        write_json(output/'F2-failures.json',failed)
            verify_names = [name for name in first_optical if name in created]
            if verify_names and len(verify_names)+sum(name in failed for name in first_optical)==len(first_optical) and not (output/'optical.json').exists():
                result = run_batches(inventory,optical_plan(inventory,verify_names))
                result['transmission'] = transmission_report(result['batches'])
                write_json(output/'optical.json',result)
                publish({'optical':dict(**result_reference(output/'optical.json'),passed=result['passed'],
                                         failures=result['failures'], transmission=result['transmission'])},checkout)
            dispatch = load(production_record)["v2"]["p2_dispatch"]["status"]
            if len(created)==80 or (dispatch in {'completed','completed_with_accepted_geometry_gap','queue_exhausted_with_shortfalls'}
                                      and len(created)+len(failed)==len(available)):
                publish({'F2':dict(ready_scenes=len(created),ready_frames=3*len(created),
                                  failed_scenes=failed,finished_utc=datetime.now(timezone.utc).isoformat())},checkout)
                return
        time.sleep(30)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke-unit',type=Path)
    parser.add_argument('--f2-root',type=Path)
    parser.add_argument('--geometry-probes',action='store_true')
    parser.add_argument('--publish-worktree',type=Path)
    parser.add_argument('--production-record',type=Path,default=RECORD)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        parser.error('assign the shared CUDA device explicitly')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.smoke_unit:
        args.output.mkdir(parents=True,exist_ok=True)
        unit = args.smoke_unit.resolve()
        if args.geometry_probes:
            result = geometry_probes(unit)
            write_json(args.output/'geometry-probes.json',result)
        else:
            if not args.f2_root:
                parser.error('use isolated F2 products for the qualification smoke')
            result = qualification_checks(unit,args.output.resolve(),args.f2_root.resolve())
            write_json(args.output/'optical-probes.json',result)
        raise SystemExit(0 if result['passed'] else 1)
    if not args.publish_worktree:
        parser.error('monitor mode needs its own worktree to publish records')
    monitor(args.output.resolve(),args.publish_worktree.resolve(),args.production_record.resolve())
