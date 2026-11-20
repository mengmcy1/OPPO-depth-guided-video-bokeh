"""Publish checksummed G and R inventories and their small experiment records."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import shutil

from route3_v2data_G_acquire import ATTRIBUTION as G_SOURCE, TARGET as G_TARGET, WINDOW
from route3_v2data_G_common import DATA, GENERATOR, GIB, SPLITS, files_receipt, publish, read, sizes, write
from route3_v2data_R_acquire import ATTRIBUTION as R_SOURCE, TARGET as R_TARGET


def optional(path, fallback=None):
    return read(path) if path.exists() else fallback


def measured(directory):
    identifier = directory.name
    tier_root = directory.parent.parent
    receipt = tier_root/'checksums'/f'{identifier}.json'
    if receipt.exists():
        return read(receipt)
    details = files_receipt(directory)
    details['allocated_bytes'] = sum((directory/row['path']).stat().st_blocks * 512 for row in details['files'])
    write(receipt, details)
    return details


def distribution(rows, field):
    return dict(sorted(Counter(row.get(field) or 'unknown' for row in rows).items()))


def standard_report(tier, units, split_counts, failures, exclusions, target, achieved, attribution):
    root = DATA / tier
    attribution_path = root/'attributions.json'
    write(attribution_path, [row['attribution'] for row in units])
    failure_path = root/'failures-and-exclusions.json'
    write(failure_path, dict(failures=failures, exclusions=exclusions))
    access = optional(root/'access.json', {'status': 'public'})
    total = sizes([row['bytes'] for row in units])
    physical = sizes([row['allocated_bytes'] for row in units])
    overview = dict(tier=tier, generator_version=GENERATOR, target=target, achieved=achieved,
                    difference=max(0, target-achieved), counts=split_counts, verified_units=len(units),
                    storage=total, allocated_storage=physical, available_disk_bytes=shutil.disk_usage(root).free,
                    disk_reserve_bytes=60*GIB, cgroup_limit_bytes=90*GIB, resident_limit_bytes=28*GIB,
                    acquisition_status='finished' if (root/'acquisition-complete.json').exists() else 'incomplete',
                    acquisition_stop=optional(root/'acquisition-stop.json'),
                    gpu=optional(root/'gpu-metrics.json'), gpu_stop=optional(root/'gpu-stop.json'),
                    failures_count=len(failures), exclusions_count=len(exclusions),
                    failure_and_exclusion_list=str(failure_path.relative_to(root)),
                    licenses={attribution['license']: len(units)}, access=access,
                    pending_credentials=[attribution['source']] if access['status']=='pending credentials' else [])
    manifest = dict(**overview, source=attribution, units=units,
                    split_protocol=dict(key='<source>:<asset_id>', hash='int(sha256(key)[:8], 16) % 1000',
                                        train='0–799', development='800–899', test='900–999',
                                        development_use='per-tier training/development gap only',
                                        test_use='held outside train and development'),
                    attributions=dict(path=attribution_path.name, sha256=sha256(attribution_path.read_bytes()).hexdigest()),
                    failure_and_exclusion_sha256=sha256(failure_path.read_bytes()).hexdigest())
    return manifest, overview


def summarize_g():
    root = DATA/'G'
    sources = optional(root/'source-catalog.json', {})
    plan = optional(root/'sampling-plan.json', {})
    quotas = plan.get('quotas', {})
    windows = {}
    failures, exclusions = [], []
    for path in sorted((root/'windows').glob('*/manifest.json')):
        done = path.parent/'prior/summary.json'
        if done.exists():
            meta = read(path)
            windows.setdefault(meta['environment'], []).append((path, meta))
        elif error := optional(path.parent/'prior-error.json'):
            failures.append(dict(window=path.parent.name, **error))
    units = []
    split_counts = {role: dict(environments=0, complete_environments=0, frames=0) for role in SPLITS}
    per_environment = {}
    initial = optional(root/'initial-storage.json')
    for env, source in sources.items():
        quota = quotas.get(env, 0)*WINDOW
        receipt = optional(root/'acquisition'/f'{env}.json')
        selected = windows.get(env, [])
        if receipt:
            failures.extend(dict(environment=env, **error) for error in receipt.get('errors', []))
            if receipt.get('complete'):
                allowed = set(receipt['windows'])
                selected = [(p, m) for p, m in selected if m['scene'] in allowed]
        if not source['archives']:
            exclusions.append(dict(environment=env, reason='No official archive pair in pinned release'))
        if not selected:
            per_environment[env] = dict(frames=0, target_frames=quota, split=source['split'],
                                        environment_type=source['environment_type'], complete=False)
            continue
        paths = [str(p.relative_to(root)) for p, _ in selected]
        files = []
        frames = total_bytes = allocated = 0
        for path, meta in selected:
            sha = measured(path.parent)
            n = len(meta['rows'])
            frames += n
            total_bytes += sha['bytes']
            allocated += sha['allocated_bytes']
            files.append(dict(window=meta['scene'], path=f'checksums/{meta["scene"]}.json', sha256=sha['sha256'], bytes=sha['bytes']))
            if initial is None:
                initial = dict(first_window=meta['scene'], frames=n, bytes=sha['bytes'],
                               bytes_per_frame=sha['bytes']/n, estimated_target_bytes=round(sha['bytes']/n*G_TARGET))
                write(root/'initial-storage.json', initial)
        complete = frames == quota and bool(receipt and receipt.get('complete'))
        row = dict(asset_id=env, key=source['key'], bucket=source['bucket'], split=source['split'],
                   environment_type=source['environment_type'], frames=frames, target_frames=quota,
                   complete=complete, windows=paths, checksums=files, bytes=total_bytes,
                   allocated_bytes=allocated, sha256=sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                   attribution={key: source[key] for key in ('source','asset_id','author','license','terms','terms_version','channel','revision')})
        units.append(row)
        count = split_counts[source['split']]
        count['environments'] += 1
        count['complete_environments'] += complete
        count['frames'] += frames
        per_environment[env] = {key: row[key] for key in ('frames','target_frames','split','environment_type','complete','bytes','allocated_bytes')}
    frames = sum(unit['frames'] for unit in units)
    manifest, summary = standard_report('G', units, split_counts, failures, exclusions, G_TARGET, frames, G_SOURCE)
    extra = dict(official_environment_count=len(sources), per_environment=per_environment,
                 official_environment_type_distribution=distribution(list(sources.values()), 'environment_type'),
                 verified_environment_type_distribution=distribution(units, 'environment_type'),
                 frame_storage=sizes([v['bytes']/v['frames'] for v in units]) if frames else sizes([]),
                 mean_bytes_per_frame=summary['storage']['total_bytes']/frames if frames else None,
                 initial_storage=initial, planned_frames_by_split={role: sum(quotas.get(e, 0)*WINDOW for e, row in sources.items() if row['split']==role)
                                                                  for role in SPLITS},
                 sharp_identity_check=optional(root/'sharp-identity-check.json'),
                 environment_type_labeling='Grouped locally from the published environment names',
                 hub_card_license_tag='BSD-3-Clause; data use follows the publisher’s explicit CC BY 4.0 documentation',
                 upstream_download_scope='Official front RGB, depth and pose for selected windows across all available environments')
    publish('G', dict(manifest, **extra), dict(summary, **extra))


def summarize_r():
    root = DATA/'R'
    source = optional(root/'source-catalog.json', {})
    plan = optional(root/'sampling-plan.json', {})
    old = plan.get('existing', [p.parent.name for p in sorted((root/'windows').glob('*/retained.json'))])
    candidates = plan.get('candidates', [])
    failures, exclusions = [], []
    units = []
    split_counts = {role: dict(scenes=0, frames=0, input_views=0, heldout_views=0, benchmark_scenes=0) for role in SPLITS}
    for identifier in old + candidates:
        path = root/'windows'/identifier/'manifest.json'
        readiness = optional(path.parent/'readiness.json')
        if not readiness or not readiness['passed']:
            acquisition = optional(root/'acquisition'/f'{identifier}.json')
            error = readiness or acquisition
            if error and not error['passed']:
                failures.append(dict(scene=identifier, reason=error.get('reason'), phase='validation' if readiness else 'acquisition'))
            continue
        if len(units) >= R_TARGET:
            exclusions.append(dict(scene=identifier, reason='Completed beyond the planned scene count'))
            continue
        meta = read(path)
        attribution = meta['source']
        receipt = measured(path.parent)
        units.append(dict(asset_id=identifier, **{key: meta[key] for key in ('key','bucket','split','retained')},
                          window=str(path.relative_to(root)), checksum_receipt=f'checksums/{identifier}.json',
                          sha256=receipt['sha256'], bytes=receipt['bytes'], allocated_bytes=receipt['allocated_bytes'],
                          benchmark=attribution['benchmark'], benchmark_hub_folder=attribution['benchmark_hub_folder'],
                          metadata=attribution['metadata'], attribution={k: v for k,v in attribution.items() if k != 'metadata'},
                          **{key: value for key, value in readiness.items() if key != 'scene'}))
        count = split_counts[meta['split']]
        count['scenes'] += 1
        count['frames'] += readiness['frame_count']
        count['input_views'] += readiness['input_count']
        count['heldout_views'] += readiness['heldout_count']
        count['benchmark_scenes'] += attribution['benchmark']
    exclusions.extend(dict(archive=path, reason='Duplicate scene ID in the official archive index')
                      for path in source.get('duplicate_archives', []))
    manifest, summary = standard_report('R', units, split_counts, failures, exclusions, R_TARGET, len(units), R_SOURCE)
    pixel_count = sum(row['pair_pixels'] for row in units)
    retained = sizes([row['bytes'] for row in units if row['retained']])
    finalized_estimate = round(retained['mean_bytes']*R_TARGET) if retained['count'] else None
    metadata = [row['metadata'] for row in units]
    data_fields = ('scene_type','scene_category','boundedness','lighting','reflection','transparency','camera','upstream_batch')
    extra = dict(retained_scenes=sum(row['retained'] for row in units), new_scenes=sum(not row['retained'] for row in units),
                 source_scene_count=len(source.get('scenes', {})),
                 benchmark_official_scene_count=len(source.get('benchmark_ids', [])),
                 benchmark_hub_only_ids=source.get('benchmark_hub_only_ids', []),
                 benchmark_scene_ids=[row['asset_id'] for row in units if row['benchmark']],
                 official_metadata_distribution={key: distribution(metadata, key) for key in data_fields},
                 common_visible_fraction=sum(row['common_pixels'] for row in units)/pixel_count if pixel_count else None,
                 static_common_fraction=sum(row['static_common_pixels'] for row in units)/pixel_count if pixel_count else None,
                 pair_pixels=pixel_count, initial_storage=optional(root/'initial-storage.json'),
                 retained_validated_storage=retained, estimated_target_bytes_after_validation=finalized_estimate,
                 source_resolution='DL3DV-ALL-480P images_8', observed_tensor_resolution=[256, 256])
    publish('R', dict(manifest, **extra), dict(summary, **extra))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=['G', 'R', 'both'], default='both')
    args = parser.parse_args()
    if args.tier in ('G', 'both'):
        summarize_g()
    if args.tier in ('R', 'both'):
        summarize_r()


if __name__ == '__main__':
    main()
