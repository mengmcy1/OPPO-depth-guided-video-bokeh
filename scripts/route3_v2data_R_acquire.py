"""Stage the retained and new DL3DV-10K 480P windows."""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
from html import unescape
import io
import json
from pathlib import PurePosixPath
import re
import resource
import shutil
from zipfile import BadZipFile, ZipFile

import cv2
from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_url
import numpy as np
import requests

import route3_ovg_1b_dl3dv as dl3dv
from route3_v2data_G_acquire import ranges
from route3_v2data_G_common import DATA, GENERATOR, ROOT, SpaceError, check_space, read, sizes, split, write


OUT = DATA / 'R'
SOURCE = 'DL3DV-10K'
TARGET = 500
LEGACY = ROOT / 'outputs/route3-ovg-1b-2026-11-18-01/R'
DATASET_REVISION = '5902ed6d707cc13a7779907c1e096676f7707971'
METADATA_REVISION = 'cda4445d0d987758c8a66f2ae868ba3ece6d2682'
BENCHMARK_REPO = 'DL3DV/DL3DV-Benchmark'
BENCHMARK_REVISION = '9684e8382278c5e18173c1e72bd246daf2874539'
BENCHMARK_PREVIEW = 'bc80acb6bd79d9352895d4451080b17029303287'
PUBLISHER = f'https://raw.githubusercontent.com/DL3DV-10K/Dataset/{METADATA_REVISION}'
TERMS = f'https://github.com/DL3DV-10K/Dataset/blob/{METADATA_REVISION}/License.md'
ATTRIBUTION = dict(source=SOURCE, author='Lu Ling and the DL3DV-10K Dataset Collection Team',
                   license='CC BY-NC 4.0', terms=TERMS, terms_version=METADATA_REVISION,
                   use='Non-commercial research and education',
                   channel=f'https://huggingface.co/datasets/{dl3dv.REPO}', revision=DATASET_REVISION)
ID = r'[0-9a-f]{64}'


def safe_reason(error):
    if isinstance(error, requests.RequestException):
        status = getattr(error.response, 'status_code', None)
        return f'{type(error).__name__} HTTP {status}' if status else type(error).__name__
    return str(error)


def public(url):
    response = requests.get(url, timeout=(10, 45))
    response.raise_for_status()
    return response.content


def scene_descriptions():
    fields = ('batch', 'boundedness', 'reflection', 'transparency', 'lighting',
              'scene_type', 'scene_category', 'camera')
    page = public(PUBLISHER + '/visualize/index.html').decode()
    details = {}
    for fragment in re.findall(r'<tr>(.*?)</tr>', page, flags=re.S):
        match = re.search(fr'<figcaption>({ID})</figcaption>', fragment)
        if match:
            values = re.findall(r'<td>([^<>]*)</td>', fragment)
            if len(values) == len(fields):
                details[match[1]] = {field: unescape(value).strip() or None for field, value in zip(fields, values)}
    if len(details) < 3000:
        raise ValueError('Official scene preview unexpectedly has fewer than 3,000 annotated scene rows')
    census = public(PUBLISHER + '/cache/DL3DV-valid.csv').decode()
    for row in csv.DictReader(io.StringIO(census)):
        identifier = row['hash']
        details.setdefault(identifier, {}).update(duration_seconds=float(row['duration']) if row['duration'] else None,
                                                   sensitivity=row['sensibility label'] or None)
    return details


def upstream():
    path = OUT / 'source-catalog.json'
    if path.exists():
        return read(path)
    api = HfApi()
    benchmark = api.list_repo_tree(BENCHMARK_REPO, revision=BENCHMARK_REVISION, repo_type='dataset', token=False)
    benchmark_folders = {row.path for row in benchmark if re.fullmatch(ID, row.path)}
    preview = public('https://api.github.com/repos/DL3DV-10K/DL3DV-Benchmark-Preview/git/trees/'
                     + BENCHMARK_PREVIEW + '?recursive=1')
    benchmark_ids = {match[1] for item in json.loads(preview)['tree']
                     if (match := re.fullmatch(fr'imgs/({ID})\.jpg', item['path']))}
    if len(benchmark_ids) != 140 or not benchmark_ids <= benchmark_folders:
        raise ValueError('The 140 official benchmark preview IDs disagree with the Hub listing')
    details = scene_descriptions()
    archived = api.dataset_info(dl3dv.REPO, revision=DATASET_REVISION, timeout=30, token=False).siblings
    entries = {}
    duplicates = []
    for entry in sorted(row.rfilename for row in archived):
        match = re.fullmatch(fr'(\d+K)/({ID})\.zip', entry)
        if match:
            batch, identifier = match.groups()
            if identifier in entries:
                duplicates.append(entry)
                continue
            entries[identifier] = dict(ATTRIBUTION, asset_id=identifier, archive=entry,
                                       **split(SOURCE, identifier), benchmark=identifier in benchmark_ids,
                                       benchmark_hub_folder=identifier in benchmark_folders,
                                       metadata=dict(details.get(identifier, {}), upstream_batch=batch))
    result = dict(scenes=entries, duplicate_archives=duplicates,
                  benchmark_ids=sorted(benchmark_ids), benchmark_revision=BENCHMARK_REVISION,
                  benchmark_preview_revision=BENCHMARK_PREVIEW,
                  benchmark_hub_only_ids=sorted(benchmark_folders - benchmark_ids),
                  metadata_revision=METADATA_REVISION)
    write(path, result)
    return result


def ranked(rows, old):
    groups = defaultdict(list)
    existing = Counter()
    def stratum(source):
        m = source['metadata']
        return (m.get('scene_category') or 'unknown', m.get('lighting') or 'unknown',
                m.get('boundedness') or 'unknown')
    for identifier, source in rows.items():
        if identifier in old:
            existing[stratum(source)] += 1
        elif source['metadata'].get('scene_category'):
            groups[stratum(source)].append(source)
    def order(row):
        return sha256(('route3-v2data-R:' + row['key']).encode()).hexdigest()
    groups = {key: deque(sorted(items, key=order)) for key, items in groups.items()}
    while groups:
        key = min(groups, key=lambda group: (existing[group], group))
        row = groups[key].popleft()
        existing[key] += 1
        if not groups[key]:
            del groups[key]
        yield row
    yield from sorted((r for i, r in rows.items() if i not in old and not r['metadata'].get('scene_category')),
                      key=order)


def retain(source, old):
    identifier = source['asset_id']
    old_dir = LEGACY / 'windows' / identifier
    destination = OUT / 'windows' / identifier
    manifest = destination / 'manifest.json'
    if (destination / 'retained.json').exists():
        return
    files = [path for path in old_dir.rglob('*') if path.is_file() and path.name != 'manifest.json'
             and path.name != 'visibility.json']
    check_space(OUT, sum(path.stat().st_size for path in files) + 2**20)
    for path in files:
        target = destination / path.relative_to(old_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    meta = read(old_dir / 'manifest.json')
    for row in meta['rows']:
        row['image'] = str((destination/'480p'/row['image'].split('/')[-1]).relative_to(ROOT))
    write(manifest, dict(meta, generator_version=GENERATOR, source=source, retained=True,
                         **split(SOURCE, identifier)))
    vis = read(old_dir / 'visibility.json')
    for row in vis['accepted']:
        row['mask'] = str((destination/'static-common'/row['mask'].split('/')[-1]).relative_to(ROOT))
    write(destination / 'visibility.json', vis)
    write(destination / 'retained.json', dict(prior_acquisition=old['transport']))


class RemoteScene(dl3dv.DirectDl3dvZip):
    """Reuse the small ZIP tail for directory and end-record lookups."""

    def __init__(self, metadata, revision, path):
        super().__init__(metadata, revision, path)
        # The configured HTTPS route handles the larger contiguous CDN ranges.
        # The signed delivery URL lives only in memory and is never included in receipts.
        self.session.trust_env = True
        self.tail_start = max(0, self.size - 512 * 1024)
        self.seek(self.tail_start)
        self.tail = super().read()
        self.seek(0)

    def read(self, count=-1):
        start = self.position
        end = self.size if count < 0 else min(start + count, self.size)
        if start >= self.tail_start and end >= start:
            self.position = end
            return self.tail[start - self.tail_start:end - self.tail_start]
        return super().read(count)

    def read_at(self, start, end):
        self.seek(start)
        return self.read(end - start)

    def read_member(self, entry):
        with ZipFile(self) as archive:
            return archive.read(entry.filename)


def stage(reader, source, manifest):
    with ZipFile(reader) as archive:
        names = archive.namelist()
        transforms = [name for name in names if PurePosixPath(name).name == 'transforms.json']
        if len(transforms) != 1:
            raise ValueError('DL3DV archive does not have one transforms.json')
        transforms = transforms[0]
        meta = json.loads(ranges(reader, archive, [transforms])[transforms])
        picked = dl3dv.choose(dl3dv.metadata(meta), dl3dv.CONFIG['dl3dv']['frames_per_window'])
        images = {PurePosixPath(name).name: name for name in names if PurePosixPath(name).parent.name == 'images_8'}
        bodies = ranges(reader, archive, [images[row['filename']] for row in picked])
    destination = manifest.parent/'480p'
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in picked:
        name = row['filename']
        body = bodies[images[name]]
        decoded = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or max(decoded.shape[:2]) != 480:
            raise ValueError('DL3DV image does not have native 480P resolution')
        height, width = decoded.shape[:2]
        src_width, src_height = row['resolution']
        intrinsics = np.array(row['K'])
        intrinsics[:2] *= np.array([width/src_width, height/src_height])[:, None]
        image = destination/name
        image.write_bytes(body)
        rows.append(dict(K=intrinsics.tolist(), c2w=row['c2w'], image=str(image.relative_to(ROOT))))
    heldout = list(range(1, len(rows), 3))
    inputs = [i for i in range(len(rows)) if i not in heldout]
    write(manifest, dict(tier='R', scene=source['asset_id'], rows=rows, inputs=inputs, heldout=heldout,
                         generator_version=GENERATOR, source=source, retained=False,
                         official_repository=dl3dv.REPO, license=source['license'], terms=source['terms'],
                         **split(SOURCE, source['asset_id'])))


def acquisition(source, budget_bytes):
    identifier = source['asset_id']
    destination = OUT / 'windows' / identifier
    manifest = destination / 'manifest.json'
    receipt = OUT / 'acquisition' / f'{identifier}.json'
    if receipt.exists() and read(receipt).get('passed') and manifest.exists():
        return
    check_space(OUT, budget_bytes)
    archive = None
    for attempt in range(2):
        try:
            url = hf_hub_url(dl3dv.REPO, source['archive'], repo_type='dataset', revision=source['revision'])
            metadata = get_hf_file_metadata(url, token=True, timeout=20)
            with RemoteScene(metadata, source['revision'], source['archive']) as archive:
                # The historical reader resolves only authorized official CDN ranges.
                stage(archive, source, manifest)
                write(receipt, dict(passed=True, scene=identifier, downloaded_bytes=archive.bytes_received,
                                    archive_bytes=archive.size, archive=source['archive']))
                return
        except SpaceError:
            raise
        except (OSError, ValueError, KeyError, BadZipFile) as error:
            reason = safe_reason(error)
            if attempt == 0 and isinstance(error, (OSError, requests.RequestException)):
                continue
            write(receipt, dict(passed=False, scene=identifier, reason=reason))
    print('R_ACQUISITION_ERROR', identifier, reason, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=int, default=510, help='New candidates before mask validation')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    old = {row['scene']: row for row in read(LEGACY/'acquisition.json')['scenes'] if row['passed']}
    upstream_data = upstream()
    rows = upstream_data['scenes']
    for identifier, receipt in old.items():
        retain(rows[identifier], receipt)
    retained_sizes = [sum(f.stat().st_size for f in (OUT/'windows'/identifier).rglob('*') if f.is_file())
                      for identifier in old]
    storage = sizes(retained_sizes)
    write(OUT/'initial-storage.json', dict(retained_first_batch=storage,
                                          target_bytes_estimate=round(storage['mean_bytes']*TARGET)))
    check_space(OUT, max(retained_sizes)*8)
    try:
        if not get_token():
            raise ValueError('Hugging Face default token absent')
        HfApi().auth_check(dl3dv.REPO, repo_type='dataset', token=True)
    except (ValueError, requests.RequestException) as error:
        status = getattr(getattr(error, 'response', None), 'status_code', None)
        state = 'pending credentials' if status in (401, 403) or not get_token() else 'access check failed'
        write(OUT/'access.json', dict(status=state, reason=safe_reason(error)))
        return
    write(OUT/'access.json', dict(status='authorized', repository=dl3dv.REPO, revision=DATASET_REVISION))
    candidates = list(ranked(rows, old))[:args.candidates]
    write(OUT/'sampling-plan.json', dict(existing=list(old), candidates=[row['asset_id'] for row in candidates],
                                        target=TARGET, selection='ranked by scene category, lighting and boundedness; deterministic per stratum'))
    def fetch(row):
        if not (OUT/'target-reached.json').exists():
            acquisition(row, max(retained_sizes)*6)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for _ in pool.map(fetch, candidates):
            pass
    write(OUT/'acquisition-complete.json', dict(candidates=len(candidates),
                                              peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__ == '__main__':
    main()
