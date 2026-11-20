"""Acquire deterministic TartanAir V2 windows from all published environments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
import io
from pathlib import PurePosixPath
import re
import resource
import struct
import time
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
import zlib

import cv2
import numpy as np
import requests

import route3_ovg_tartan as tartan
from route3_ovg_r2_tartan import DirectZip
from route3_v2data_G_common import DATA, GENERATOR, ROOT, SpaceError, check_space, read, split, write


OUT = DATA / 'G'
SOURCE = 'TartanAir-V2'
TERMS = 'https://tartanair.org/index.html#license'
ATTRIBUTION = dict(source=SOURCE, author='TartanAir Team, Carnegie Mellon University AirLab',
                   license='CC BY 4.0', terms=TERMS, terms_version='4.0 International',
                   channel=tartan.HOST, revision=tartan.REVISION)
TARGET = 20_000
WINDOW = 8
OLD = ROOT / 'outputs/route3-ovg-1b-2026-11-18-01/G/windows'
TYPES = {
    'built interior': 'AbandonedFactory2 AmericanDiner ArchVizTinyHouseDay ArchVizTinyHouseNight CarWelding CoalMine HQWesternSaloon Hospital House IndustrialHangar Office OldBrickHouseDay OldBrickHouseNight Prison Restaurant RetroOffice Sewerage Slaughter Supermarket',
    'natural landscape': 'BrushifyMoon ForestEnv GreatMarsh Ocean PolarSciFi SeasonalForestAutumn SeasonalForestSpring SeasonalForestSummerNight SeasonalForestWinter SeasonalForestWinterNight ShoreCaves TerrainBlending',
    'urban or residential': 'AbandonedSchool AncientTowns Antiquity3D CyberPunkDowntown Cyberpunk Downtown HongKong JapaneseAlley JapaneseCity MiddleEast ModUrbanCity ModernCityDowntown ModularNeighborhood ModularNeighborhoodIntExt OldScandinavia OldTownFall OldTownNight OldTownSummer OldTownWinter Rome SeasideTown SoulCity VictorianStreet WesternDesertTown',
    'industrial or infrastructure': 'AbandonedCable AbandonedFactory ConstructionSite DesertGasStation FactoryWeather Gascola NordicHarbor OldIndustrialCity UrbanConstruction WaterMillDay WaterMillNight',
    'themed or mixed': 'AmusementPark Apocalyptic CastleFortress CountryHouse EndofTheWorld Fantasy GothicIsland Ruins',
}
ENV_TYPES = {name: category for category, names in TYPES.items() for name in names.split()}


def get_json(url):
    for attempt in range(3):
        try:
            with requests.get(url, timeout=(8, 24)) as response:
                response.raise_for_status()
                return response.json()
        except requests.RequestException as error:
            code = getattr(error.response, 'status_code', None)
            if attempt == 2 or code in (401, 403, 404):
                raise OSError(f'TartanAir public catalog HTTP {code or "timeout"} at {url}') from None
            time.sleep(1 + attempt)


def catalog():
    path = OUT / 'source-catalog.json'
    if path.exists():
        return read(path)
    api = f'https://huggingface.co/api/datasets/{tartan.REPO}'
    info = get_json(api + f'/revision/{tartan.REVISION}?blobs=true')
    entries = {}
    for row in info['siblings']:
        match = re.fullmatch(r'([^/]+)/(Data_easy|Data_hard)/(image|depth)_lcam_front.zip', row['rfilename'])
        if match:
            env, difficulty, modality = match.groups()
            entries.setdefault(env, {}).setdefault(difficulty, {})[modality] = row
    official = get_json(api + f'/tree/{tartan.REVISION}')
    envs = sorted(row['path'] for row in official if row['type'] == 'directory')
    sources = {}
    for env in envs:
        available = entries.get(env, {})
        complete = {d: row for d, row in available.items() if set(row) == {'image', 'depth'}}
        sources[env] = dict(ATTRIBUTION, asset_id=env, **split(SOURCE, env),
                            environment_type=ENV_TYPES.get(env, 'unclassified'), archives=complete)
    write(path, sources)
    return sources


def sample_windows(candidates, count):
    if len(candidates) < count:
        raise ValueError(f'Only {len(candidates)} disjoint windows for quota {count}')
    return [candidates[((2 * i + 1) * len(candidates)) // (2 * count)] for i in range(count)]


def ranges(reader, archive, names):
    """Read nearby ZIP members in one range and verify each member CRC."""
    members = sorted((archive.getinfo(name) for name in names), key=lambda entry: entry.header_offset)
    groups = []
    for entry in members:
        start = entry.header_offset
        end = min(reader.size, start + 30 + len(entry.filename.encode()) + entry.compress_size + 128)
        if groups and start <= groups[-1][1] + 8192 and end - groups[-1][0] < 16 * 2**20:
            groups[-1][1] = end
            groups[-1][2].append(entry)
        else:
            groups.append([start, end, [entry]])
    values = {}
    for start, end, group in groups:
        data = reader.read_at(start, end)
        for entry in group:
            offset = entry.header_offset - start
            if data[offset:offset+4] != b'PK\x03\x04':
                raise OSError('Invalid ZIP local header')
            name_bytes, extra_bytes = struct.unpack_from('<HH', data, offset+26)
            offset += 30 + name_bytes + extra_bytes
            stop = offset + entry.compress_size
            if stop > len(data):
                values[entry.filename] = reader.read_member(entry)
                continue
            body = data[offset:stop]
            if entry.compress_type == ZIP_DEFLATED:
                body = zlib.decompress(body, -15)
            elif entry.compress_type != ZIP_STORED:
                raise ValueError(f'Unsupported ZIP compression {entry.compress_type}')
            if len(body) != entry.file_size or zlib.crc32(body) != entry.CRC:
                raise OSError('ZIP member size or CRC mismatch')
            values[entry.filename] = body
    return values


def open_archive(stack, entry):
    name = entry['rfilename']
    size = entry.get('size')
    if size is None:
        parent = str(PurePosixPath(name).parent)
        tree = get_json(f'https://huggingface.co/api/datasets/{tartan.REPO}/tree/{tartan.REVISION}/{parent}')
        size = next(x['size'] for x in tree if x['path'] == name)
    reader = stack.enter_context(DirectZip(f'{tartan.HOST}/resolve/{tartan.REVISION}/{name}', size))
    archive = stack.enter_context(ZipFile(reader))
    return reader, archive


def enumerate_windows(image, depth):
    rgb = tartan.members(image[1], 'image_lcam_front')
    labels = tartan.members(depth[1], 'depth_lcam_front')
    pose_index = {}
    for reader, archive in (image, depth):
        for entry in archive.infolist():
            parts = PurePosixPath(entry.filename).parts
            if parts[-1] == 'pose_lcam_front.txt':
                trajectory = next((part for part in parts if re.fullmatch(r'P\d+', part)), None)
                if trajectory:
                    pose_index.setdefault(trajectory, (reader, entry))
    choices = []
    for trajectory in sorted(set(rgb) & set(labels) & set(pose_index)):
        shared = set(rgb[trajectory]) & set(labels[trajectory])
        choices.extend((trajectory, frame) for frame in sorted(shared)
                       if frame % WINDOW == 0 and all(frame+i in shared for i in range(WINDOW)))
    return choices, rgb, labels, pose_index


def acquire_window(env, difficulty, item, image, depth, rgb, labels, poses, pose_index, entries):
    trajectory, frame = item
    name = f'{env}-{difficulty}-{trajectory}-{frame:06d}'
    manifest = OUT / 'windows' / name / 'manifest.json'
    if not manifest.exists():
        ids = list(range(frame, frame + WINDOW))
        nominal = f'{env}-{trajectory}-{frame:06d}'
        old = OLD / nominal / 'manifest.json'
        prior = read(old) if old.exists() and difficulty == 'Data_easy' else None
        if prior and [row['frame'] for row in prior['rows']] != ids:
            prior = None
        paths = [rgb[trajectory], labels[trajectory]]
        if prior:
            bodies = [{files[i]: (ROOT/row[k]).read_bytes() for i, row in zip(ids, prior['rows'])}
                      for k, files in zip(('image', 'depth'), paths)]
        else:
            bodies = [ranges(*arc, [files[i] for i in ids]) for arc, files in zip((image, depth), paths)]
        dest = manifest.parent
        dest.mkdir(parents=True, exist_ok=True)
        rows = []
        from scipy.spatial.transform import Rotation
        for i, f in enumerate(ids):
            p = poses[trajectory][f]
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(p[3:]).as_matrix() @ tartan.CV_TO_NED
            pose[:3, 3] = p[:3]
            rgb_path = dest / f'frame-{i:02d}.png'
            label_path = dest / 'labels' / f'depth-{i:02d}.png'
            label_path.parent.mkdir(exist_ok=True)
            rgb_path.write_bytes(bodies[0][paths[0][f]])
            label_path.write_bytes(bodies[1][paths[1][f]])
            z = tartan.depth_from_png(label_path)
            if z.shape != (640, 640) or np.isfinite(z).mean() < .95 or np.median(z) <= 0:
                raise ValueError(f'Invalid packed depth in {name}/{f}')
            rows.append(dict(frame=f, image=str(rgb_path.relative_to(ROOT)),
                             depth=str(label_path.relative_to(ROOT)), c2w=pose.tolist(),
                             K=[[320., 0., 320.], [0., 320., 320.], [0., 0., 1.]]))
        write(manifest, dict(tier='G', scene=name, environment=env, trajectory=trajectory,
                             difficulty=difficulty, rows=rows, generator_version=GENERATOR,
                             depth_format='OpenCV axial metres; little-endian float32 PNG bytes',
                             pose_format='NED source, OpenCV camera-to-world',
                             source_archives=[entry['rfilename'] for entry in entries.values()],
                             pose_member=pose_index[trajectory][1].filename, **split(SOURCE, env)))
    return name


def acquire_environment(source, quota):
    env = source['asset_id']
    receipt = OUT / 'acquisition' / f'{env}.json'
    if receipt.exists() and read(receipt).get('complete'):
        return read(receipt)
    result = dict(environment=env, target_frames=quota * WINDOW, windows=[], errors=[])
    # Try easy trajectories first for continuity with the previous sampler.
    for difficulty, entries in sorted(source['archives'].items()):
        result['windows'] = []
        try:
            with ExitStack() as stack:
                image = open_archive(stack, entries['image'])
                depth = open_archive(stack, entries['depth'])
                choices, rgb, labels, pose_index = enumerate_windows(image, depth)
                selected = sample_windows(choices, quota)
                poses = {}
                for trajectory in sorted({trajectory for trajectory, _ in selected}):
                    reader, entry = pose_index[trajectory]
                    poses[trajectory] = np.loadtxt(io.BytesIO(reader.read_member(entry)), ndmin=2)
                with ThreadPoolExecutor(max_workers=6) as workers:
                    for start in range(0, len(selected), 6):
                        batch = selected[start:start+6]
                        measured = OUT/'initial-storage.json'
                        expected = round(read(measured)['bytes_per_frame'] * WINDOW) if measured.exists() else 0
                        check_space(OUT, len(batch) * max(expected, 32 * 2**20))
                        ready = workers.map(lambda item: acquire_window(
                            env, difficulty, item, image, depth, rgb, labels, poses, pose_index, entries), batch)
                        result['windows'].extend(ready)
                result.update(complete=True, difficulty=difficulty, frames=quota*WINDOW,
                              downloaded_bytes=image[0].bytes_received+depth[0].bytes_received)
                break
        except SpaceError:
            raise
        except (OSError, ValueError, KeyError, IndexError, RuntimeError) as error:
            # Network exceptions may contain signed delivery URLs.
            message = type(error).__name__ if isinstance(error, requests.RequestException) else str(error)
            result['errors'].append(dict(difficulty=difficulty, reason=message))
    if not source['archives']:
        result['errors'].append(dict(reason='No matching official front RGB and depth archive pair'))
    write(receipt, result)
    print('G_ENV', env, result.get('complete', False), len(result['windows']), result['errors'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit-environments', type=int)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    sources = catalog()
    available = [row for row in sources.values() if row['archives']]
    covered = sorted(available, key=lambda row: sha256(row['key'].encode()).hexdigest())
    base, extra = divmod(TARGET // WINDOW, len(covered))
    quotas = {row['asset_id']: base + (i < extra) for i, row in enumerate(covered)}
    write(OUT / 'sampling-plan.json', dict(target_frames=TARGET, window_frames=WINDOW, quotas=quotas,
                                          peak_rss_limit_bytes=28*2**30))
    historical = {read(path)['environment'] for path in OLD.glob('*/manifest.json')}
    ordered = sorted(sources.values(), key=lambda row: (row['asset_id'] not in historical, row['asset_id']))
    with ThreadPoolExecutor(max_workers=2) as group:
        def run(row):
            return acquire_environment(row, quotas.get(row['asset_id'], 0))
        for _ in group.map(run, ordered[:args.limit_environments]):
            pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.limit_environments is None:
        write(OUT/'acquisition-complete.json', dict(environments=len(ordered), peak_rss_kib=rss))
    print('G_ACQUISITION_DONE', rss, flush=True)


if __name__ == '__main__':
    main()
