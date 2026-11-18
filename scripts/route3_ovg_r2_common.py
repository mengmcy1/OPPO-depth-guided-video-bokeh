"""Run locations shared by the second-round entry points."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT/'configs/route3-ovg-1a-r2-2026-11-18.json'
CONFIG = json.loads(CONFIG_FILE.read_text())
OUT = ROOT/CONFIG['output']
ROUND1 = ROOT/CONFIG['round1']
LOG = ROOT/'logs/route3-ovg-1a-r2-2026-11-18'
REC = ROOT/'03_实验记录/records'


def guard():
    OUT.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(OUT).free/2**30 < 60:
        raise RuntimeError('Data disk below 60 GiB')


def reference(source, destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        destination.symlink_to(source.resolve(),target_is_directory=source.is_dir())


def pooled_psnr(rows):
    import math

    pixels = sum(n for n, _ in rows)
    return -10*math.log10(max(sum(n*10**(-db/10) for n,db in rows)/pixels,1e-30))
