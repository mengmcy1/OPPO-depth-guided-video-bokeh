"""Check native Sintel depth against official stereo disparity (six first frames).
Archive members are range-downloaded with CRC checks, never regenerated.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import zipfile
import zlib

import numpy as np
from PIL import Image
from prepare_ours_geometry_pilot import RangeReader, depth_read, camera_read, write_json

ROOT = Path(__file__).resolve().parents[1]
C = json.loads((ROOT/'configs/ours-geometry-data-pilot-20261022.json').read_text())
RAW = ROOT/C['raw_root']
OUT = ROOT/'data/raw/MPI-Sintel-stereo-convention-check-20261022'
URL = 'https://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-stereo-training-20150305.zip'
SIZE = 2612604481
RECORD = ROOT/'03_实验记录/records/ours-geometry-pilot-stereo-convention-20261022.json'


def download():
    r = RangeReader(URL, SIZE, block=65536)
    z = zipfile.ZipFile(r)
    members = ['README_stereo.txt','sdk/python/sintel_io.py'] + [f"training/disparities/{s['sequence']}/frame_0001.png" for s in C['sequences']]
    r.prefetch([z.getinfo(n) for n in members if not (OUT/n).exists()])
    provenance = []
    for name in members:
        info = z.getinfo(name)
        p = OUT/name
        if p.exists():
            b = p.read_bytes()
        else:
            b = z.read(name)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
        assert zlib.crc32(b) == info.CRC and len(b) == info.file_size
        provenance.append(dict(path=name, bytes=len(b), crc32=info.CRC))
    write_json(OUT/'provenance.json', dict(url=URL, archive_bytes=SIZE, members=provenance))
    print('STEREO_MEMBERS_VERIFIED', len(members), flush=True)


def check():
    spec = importlib.util.spec_from_file_location('stereo_sdk',OUT/'sdk/python/sintel_io.py')
    sdk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdk)
    rows=[]
    for s in C['sequences']:
        stem=f"{s['sequence']}/frame_0001"
        d=depth_read(RAW/f'training/depth/{stem}.dpt').astype(np.float64)
        K,_=camera_read(RAW/f'training/camdata_left/{stem}.cam')
        assert not np.asarray(Image.open(OUT/f'training/disparities/{stem}.png'))[:,:,2].any()
        disp=sdk.disparity_read(str(OUT/f'training/disparities/{stem}.png'))
        assert disp.shape == d.shape
        h,w=d.shape
        y,x=np.mgrid[:h,:w]
        norm=np.sqrt(1+((x-K[0,2])/K[0,0])**2+((y-K[1,2])/K[1,1])**2)
        # Native array as axial Z vs native array as ray distance.
        axial=K[0,0]*0.1/d
        radial=axial*norm
        valid=np.isfinite(d)&(d>0)&(d<1e10)&np.isfinite(disp)&(disp>0)
        # Exclude encoded disparity saturation (official 16-bit R/G encoding, 1/64 pixel precision).
        valid &= disp < (1024-1/64)
        erra=np.abs(axial-disp)[valid];errr=np.abs(radial-disp)[valid]
        rows.append(dict(sequence=s['sequence'],frame=1,fx=float(K[0,0]),baseline_m=.1,
                         pixels=int(valid.sum()), axial_abs_px_median=float(np.median(erra)),
                         axial_abs_px_p95=float(np.quantile(erra,.95)),
                         ray_abs_px_median=float(np.median(errr)),ray_abs_px_p95=float(np.quantile(errr,.95)),
                         axial_fraction_within_0p002px=float((erra<.002).mean())))
    result=dict(status='EMPIRICAL_CONVENTION_CHECK',sources=['https://sintel.is.tue.mpg.de/depth','https://sintel.is.tue.mpg.de/stereo'],
                archive=URL,selection='first frame of each preselected pilot family, no model-based selection',
                SDK_used='official stereo sdk/python/sintel_io.py disparity_read',rows=rows,
                scope='Checks consistency with official stereo derived from known depth; not independent physical GT or all-frame accuracy.',
                raw_unchanged=True)
    result.update(conclusion='Native depth is consistent with axial Z in six checked frames; do not apply ray-distance correction.',
                  tolerance_basis='Official README_stereo.txt: 16-bit R/G encoding causes disparity error below 0.016 pixel; all six p95 are below this bound. SDK additionally reads blue, observed zero.',
                  acceptance='all six axial p95 < 0.016 pixel and axial median < ray hypothesis median',
                  renderer_background_evidence='Blender v2.63 render_result.c SCE_PASS_Z initialization is 10e10; matches observed float32 99999997952. Supporting provenance inference, not exact Sintel exporter version proof.')
    assert all(r['axial_abs_px_p95']<.016 and r['axial_abs_px_median']<r['ray_abs_px_median'] for r in rows)
    write_json(RECORD,result)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['download','check']);a=p.parse_args()
    download() if a.phase=='download' else check()
