#!/usr/bin/env python3
"""Prepare the approved DoF-NeRF protocol without changing source images.

Example (run in gs-base-cu118):
  python scripts/prepare_dofnerf_scenes.py --raw data/raw/DoF-NeRF/DoF_RealWorld_Data \
    --output data/processed/dofnerf-scenes --summary path/to/dofnerf-scenes.json

Only the seven requested scenes are eligible; extra scenes are reported. A scene
with an ambiguous triplet or invalid sharp reference is recorded and skipped.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

import cv2
import exifread

SCENES = ('amiya', 'camera', 'plant', 'kendo', 'desk', 'shelf', 'turtle')
MODES = {'sharp': 'f11', 'fg': 'f4_obj', 'bg': 'f4_bak'}
EXTENSIONS = {'.png', '.jpg', '.jpeg'}


def disk_check():
    if shutil.disk_usage('/').free < 1024**3:
        raise RuntimeError('STOP: system disk has less than 1 GiB free')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def exif_fields(path):
    with path.open('rb') as stream:
        tags = exifread.process_file(stream, details=True, strict=False)
    fields = {str(k): str(v) for k, v in tags.items()
              if any(s in str(k).lower() for s in ('focus', 'distance', 'fnumber', 'focal', 'lensmodel', 'datetimeoriginal'))
              and 'makernote' != str(k).lower()}
    focus = {k: v for k, v in fields.items()
             if any(s in k.lower() for s in ('focusdistance', 'subjectdistance', 'focus distance'))}
    return {'fields': fields, 'focus_distance': focus or None}


def triplets(scene):
    paths = {mode: sorted(p for p in (scene/'images'/folder).iterdir()
                          if p.suffix.lower() in EXTENSIONS)
             for mode, folder in MODES.items()}
    counts = {mode: len(files) for mode, files in paths.items()}
    if not counts['sharp'] or len(set(counts.values())) != 1:
        raise ValueError(f'Unequal or empty triplet counts: {counts}')
    result = []
    for index, files in enumerate(zip(*(paths[m] for m in MODES))):
        row = dict(zip(MODES, files))
        if len({p.name for p in files}) != 1:
            nums = {mode: re.fullmatch(r'DSC(\d+)', p.stem) for mode, p in row.items()}
            if not all(nums.values()):
                raise ValueError(f'Unknown triplet naming: {row}')
            nums = {mode: int(m.group(1)) for mode, m in nums.items()}
            if not (nums['sharp'] == nums['fg'] + 2 == nums['bg'] + 1):
                # camera contains a discarded shutter frame (08720); rank
                # pairing is still verifiable from filenames and capture times.
                times = []
                for mode in ('fg', 'bg', 'sharp'):
                    with row[mode].open('rb') as stream:
                        tags = exifread.process_file(stream, details=False, stop_tag='DateTimeOriginal')
                    try:
                        times.append(datetime.strptime(str(tags['EXIF DateTimeOriginal']), '%Y:%m:%d %H:%M:%S'))
                    except (KeyError, ValueError):
                        times = []; break
                ordered = nums['fg'] < nums['bg'] < nums['sharp']
                if index+1 < len(paths['fg']):
                    match = re.fullmatch(r'DSC(\d+)', paths['fg'][index+1].stem)
                    ordered = ordered and bool(match) and nums['sharp'] < int(match.group(1))
                verified_time = len(times) == 3 and times[0] < times[1] < times[2] and (times[2]-times[0]).total_seconds() <= 120
                if not ordered or not verified_time:
                    raise ValueError(f'Ambiguous triplet (filename/capture-time checks failed): {row}')
        result.append(row)
    return result


def prepare_scene(scene, output):
    rows = triplets(scene)
    # Verify references before writing any layout for this scene.
    duplicates = []
    for idx, row in enumerate(rows):
        if sha256(row['sharp']) in {sha256(row['fg']), sha256(row['bg'])}:
            duplicates.append(idx)
    if duplicates:
        raise ValueError(f'Sharp references duplicate defocused images at views {duplicates}')
    exif = {mode: [] for mode in MODES}
    source_sizes = Counter()
    for row in rows:
        for mode, path in row.items():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Cannot decode {path}')
            source_sizes[f'{image.shape[1]}x{image.shape[0]}'] += 1
            exif[mode].append(exif_fields(path))
    focus_stats = {}
    for mode, values in exif.items():
        available = [json.dumps(v['focus_distance'], sort_keys=True) for v in values if v['focus_distance']]
        focus_stats[mode] = {'available': len(available), 'unavailable': len(values)-len(available),
                             'distinct_values': dict(Counter(available)),
                             'consistent': len(set(available)) == 1 if available else None,
                             'interpretation': 'unavailable' if not available else 'reported_exif_only'}
    layouts = {}
    for layout in ('mix', 'fg', 'bg'):
        dest = output/layout/scene.name
        manifests = []
        training_index = 0
        for idx, row in enumerate(rows):
            test = idx % 9 == 0
            selected = 'sharp' if test else (('fg' if training_index % 2 == 0 else 'bg') if layout == 'mix' else layout)
            if not test:
                training_index += 1
            filename = f'{idx:03d}.png'
            image = cv2.imread(str(row[selected]), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            sizes = {'images': (width//2, height//2), 'images_4': (497, 331)}
            for directory, size in sizes.items():
                target = dest/directory/filename
                target.parent.mkdir(parents=True, exist_ok=True)
                resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                if target.exists():
                    existing = cv2.imread(str(target), cv2.IMREAD_COLOR)
                    if existing is None or existing.shape != resized.shape or not (existing == resized).all():
                        raise FileExistsError(f'Refusing to overwrite changed output: {target}')
                elif not cv2.imwrite(str(target), resized):
                    raise OSError(f'Cannot write {target}')
            manifests.append({'index': idx, 'role': 'test' if test else 'train',
                              'output_filename': filename, 'selected_mode': selected,
                              'source_files': {mode: str(p.relative_to(scene)) for mode, p in row.items()},
                              'selected_source': str(row[selected].relative_to(scene)),
                              'exif': {mode: exif[mode][idx] for mode in MODES},
                              'sizes': sizes})
        (dest/'hold=9').touch(exist_ok=True)
        manifest = {'scene': scene.name, 'layout': layout, 'hold': 9,
                    'sort': 'lexicographic source filename',
                    'mix_alternation': 'training-view rank, starting with foreground', 'views': manifests}
        (dest/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        layouts[layout] = {'path': str(dest), 'views': len(rows),
                           'train': sum(r['role'] == 'train' for r in manifests),
                           'test': [r['output_filename'] for r in manifests if r['role'] == 'test'],
                           'selected_modes': dict(Counter(r['selected_mode'] for r in manifests)),
                           'manifest': str(dest/'manifest.json'), 'sizes': sizes}
    return {'status': 'prepared', 'triplets': len(rows), 'source_sizes': dict(source_sizes),
            'naming': 'same filename across modes, or ordered DSC triplets; shutter gaps require EXIF time validation',
            'naming_exceptions': [{'index': i, 'files': {mode: p.name for mode, p in row.items()},
                                   'verification': 'ordered filenames, next-view boundary and EXIF capture times'}
                                  for i, row in enumerate(rows) if len({p.name for p in row.values()}) != 1
                                  and int(row['sharp'].stem[3:]) != int(row['fg'].stem[3:])+2],
            'aperture_exif': {mode: dict(Counter(value['fields'].get('EXIF FNumber', 'unavailable') for value in values))
                              for mode, values in exif.items()},
            'provided_geometry': [str(p.relative_to(scene)) for p in scene.rglob('*')
                                  if p.is_file() and (p.suffix in {'.bin', '.npy'} or p.name == 'colmap_output.txt')],
            'provided_geometry_used': False, 'focus_distance': focus_stats, 'layouts': layouts}


def main():
    cv2.setNumThreads(8)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    if args.raw.resolve() == args.output.resolve() or args.raw.resolve() in args.output.resolve().parents:
        parser.error('Output must be outside the raw input tree')
    disk_check()
    actual = sorted(p.name for p in args.raw.iterdir() if p.is_dir())
    report = {'raw': str(args.raw), 'requested_scenes': list(SCENES), 'actual_scenes': actual,
              'scene_count_matches': len(actual) == len(SCENES),
              'extra_scenes': sorted(set(actual)-set(SCENES)), 'scenes': {}}
    for name in SCENES:
        disk_check()
        try:
            report['scenes'][name] = prepare_scene(args.raw/name, args.output)
        except (ValueError, OSError) as exc:
            evidence = {'status': 'blocked', 'error': str(exc), 'source_counts': {}, 'focus_distance': {}}
            hashes = {}
            for mode, directory in MODES.items():
                paths = sorted(p for p in (args.raw/name/'images'/directory).glob('*') if p.suffix.lower() in EXTENSIONS)
                evidence['source_counts'][mode] = len(paths)
                values = [exif_fields(p)['focus_distance'] for p in paths]
                available = [json.dumps(value, sort_keys=True) for value in values if value]
                evidence['focus_distance'][mode] = {'available': len(available), 'unavailable': len(values)-len(available),
                                                     'distinct_values': dict(Counter(available)),
                                                     'consistent': len(set(available)) == 1 if available else None}
                hashes[mode] = {sha256(p): str(p.relative_to(args.raw/name)) for p in paths}
            evidence['sharp_defocus_duplicates'] = [
                {'sharp': path, 'defocused': hashes[mode][digest], 'mode': mode}
                for digest, path in hashes['sharp'].items() for mode in ('fg', 'bg') if digest in hashes[mode]]
            report['scenes'][name] = evidence
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(name, report['scenes'][name]['status'], flush=True)
    disk_check()


if __name__ == '__main__':
    main()
