"""Paths and resource guard for the independent 1b run."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
CONFIG_PATH=ROOT/'configs/route3-ovg-1b-2026-11-18.json'
CONFIG=json.loads(CONFIG_PATH.read_text())
OUT=ROOT/CONFIG['output']
SOURCE=ROOT/CONFIG['geometry_source']
LOG=ROOT/'logs/route3-ovg-1b-2026-11-18'
REC=ROOT/'03_实验记录/records'


def guard():
    OUT.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(OUT).free<60*2**30:
        raise RuntimeError('Data disk available space is below 60 GiB')


def read(path):
    return json.loads(Path(path).read_text())


def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())
