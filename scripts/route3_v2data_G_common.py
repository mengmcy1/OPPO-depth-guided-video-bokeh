"""Split, storage and locked GPU helpers for the G and R release."""
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DATE = '2026-11-20'
DATA = ROOT / 'data/processed' / f'route3-v2data-{DATE}'
RECORDS = ROOT / '03_实验记录/records'
GPU_LOCK = Path('/root/autodl-tmp/route3-cache/gpu-data.lock')
GIB = 2**30
MEMORY_CAP = 28 * GIB
GENERATOR = 'route3-v2data-gr/1.0'
SPLITS = ('train', 'development', 'test')


class SpaceError(RuntimeError):
    pass


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)


def split(source, identifier):
    key = f'{source}:{identifier}'
    bucket = int(sha256(key.encode()).hexdigest()[:8], 16) % 1000
    return dict(key=key, bucket=bucket, split=SPLITS[int(bucket >= 800) + int(bucket >= 900)])


def check_space(directory, batch_bytes):
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < 60 * GIB + batch_bytes:
        raise SpaceError(f'Available {free} bytes, requires 60 GiB + {batch_bytes} batch bytes')
    with Path('/proc/self/status').open() as stream:
        rss = next(int(line.split()[1]) * 1024 for line in stream if line.startswith('VmRSS:'))
    if rss > MEMORY_CAP:
        raise MemoryError(f'Worker RSS {rss} exceeds the 28 GiB budget')
    return free


def files_receipt(directory):
    files = []
    for path in sorted(directory.rglob('*')):
        if path.is_file() and not path.name.endswith('.part'):
            files.append(dict(path=str(path.relative_to(directory)), bytes=path.stat().st_size,
                              sha256=sha256(path.read_bytes()).hexdigest()))
    return dict(files=files, bytes=sum(row['bytes'] for row in files),
                sha256=sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())


def sizes(values):
    return dict(count=len(values), total_bytes=sum(values), min_bytes=min(values, default=None),
                mean_bytes=sum(values) / len(values) if values else None,
                max_bytes=max(values, default=None))


def publish(tier, manifest, summary):
    path = DATA / tier / 'manifest.json'
    write(path, manifest)
    write(RECORDS / f'route3-v2data-{tier}-{DATE}.json',
          dict(summary, manifest=str(path.relative_to(ROOT)), manifest_sha256=sha256(path.read_bytes()).hexdigest()))


@contextmanager
def locked_cuda():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with GPU_LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(['nvidia-smi'], check=True)
        import torch
        torch.set_num_threads(2)
        free, total = torch.cuda.mem_get_info()
        if free < 24 * GIB:
            raise RuntimeError('Fewer than 24 GiB GPU free after acquiring GPU data lock')
        # Keep headroom for CUDA allocations outside PyTorch's caching allocator.
        torch.cuda.set_per_process_memory_fraction(min(1, 22 * GIB / total))
        torch.cuda.reset_peak_memory_stats()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            print('GPU_PEAK_RESERVED_BYTES', torch.cuda.max_memory_reserved(), flush=True)
