"""Prepare the fixed DAVIS source/mask manifest, whole-sequence VDA, and checks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == value, str(path)
        return
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def png(path, value):
    if path.exists():
        assert np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), value), str(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode('.png', value)
    assert ok
    with path.open('xb') as f:
        f.write(buf.tobytes())


def link(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        assert path.resolve() == source.resolve()
    else:
        assert not path.exists()
        path.symlink_to(os.path.relpath(source, path.parent))


def assets(c):
    from PIL import Image
    source = ROOT / c['source']; out = ROOT / c['derived']
    # Data preparation stays fixed when a later baseline-only control is selected.
    snapshot=out/'protocol.json'
    if snapshot.exists():
        prior=json.loads(snapshot.read_text())
        for key in ('source','splits','resolution_source','derived','sequence_count','source_frames',
                    'evaluation_frames_per_sequence','K','output_size','mask','vda'):
            assert prior[key]==c[key], f'Changed data preparation rule: {key}'
    else:
        write_new(snapshot,c)
    rows = []
    for splitfile in c['splits']:
        for name in (source / splitfile).read_text().split():
            rgb = source / 'JPEGImages' / c['resolution_source'] / name
            ann = source / 'Annotations' / c['resolution_source'] / name
            files = sorted(rgb.glob('*.jpg'))
            assert len(files) >= 25
            assert [int(p.stem) for p in files] == list(range(len(files)))
            assert [p.stem for p in sorted(ann.glob('*.png'))] == [p.stem for p in files]
            unique = set(); sizes = set(); empty = []
            for i, p in enumerate(files):
                labels = np.asarray(Image.open(ann / (p.stem + '.png')))
                unique.update(int(v) for v in np.unique(labels))
                w, h = Image.open(p).size
                assert labels.shape == (h, w)
                sizes.add((w, h))
                mask = ((labels != 0) * 255).astype(np.uint8)
                if not mask.any():
                    empty.append(i)
                png(out / 'sequences' / name / 'mask' / (p.stem + '.png'), mask)
                link(out / 'sequences' / name / 'rgb' / p.name, p)
                if i < 25:
                    link(out / 'eval' / name / 'rgb' / p.name, p)
                    link(out / 'eval' / name / 'mask' / (p.stem + '.png'), out / 'sequences' / name / 'mask' / (p.stem + '.png'))
            # Later context frames may have no annotated foreground (object exits).
            # They still enter VDA; only the first25 require a defined focus/ROI.
            assert not [i for i in empty if i < 25], (name, empty)
            assert len(sizes) == 1, (name, sizes)
            # Follow the explicitly selected author single-object loader (!=0).
            # Preserve every source label ID in the manifest, including 255.
            rows.append(dict(id=name, split=Path(splitfile).stem, source_frames=len(files),
                             source_size=list(next(iter(sizes))), label_ids=sorted(unique),
                             context_empty_mask_frames=empty,
                             frame_ids=[f'{i:05d}' for i in range(25)], K=c['K']))
    rows.sort(key=lambda r: r['id'])
    assert len(rows) == c['sequence_count'] == len(set(r['id'] for r in rows))
    assert sum(r['source_frames'] for r in rows) == c['source_frames']
    result = dict(name=c['name'], source=c['source'], derived=c['derived'], sequences=rows,
                  source_frames=c['source_frames'], evaluation_frames=len(rows)*25,
                  mask_values=[0,255], source_unchanged=True)
    write_new(out / 'source-manifest.json', result)
    write_new(ROOT / '03_实验记录/records/davis-v2-source-manifest-20260929.json', result)
    print('ASSETS_COMPLETE', len(rows), c['source_frames'], flush=True)


def vda(c, selected):
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    repo = (ROOT / c['vda']['repository']).resolve()
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip() == c['vda']['commit']
    checkpoint = ROOT / c['vda']['checkpoint']
    assert checkpoint.stat().st_size == c['vda']['checkpoint_bytes'], 'Weight download incomplete'
    sys.path.insert(0, str(repo))
    from video_depth_anything.video_depth import VideoDepthAnything
    torch.set_num_threads(6); cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Same confirmed Blackwell fix already used by scripts/run_magicbokeh.py.
    # xformers0.0.32.post2 otherwise selects a Hopper-only FA3 CUDA kernel.
    fa3_disabled = torch.cuda.get_device_capability(0)[0] >= 12
    if fa3_disabled:
        from xformers.ops.fmha import _set_use_fa3
        _set_use_fa3(False)
    model = VideoDepthAnything(encoder='vitl',features=256,out_channels=[256,512,1024,1024])
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval().to('cuda')
    out = ROOT / c['derived']
    manifest = json.loads((out/'source-manifest.json').read_text())
    write_new(out/'vda-runtime.json',dict(python=sys.version.split()[0],torch=torch.__version__,
        numpy=np.__version__,opencv=cv2.__version__,GPU=torch.cuda.get_device_name(0),
        source_commit=c['vda']['commit'],checkpoint_revision=c['vda']['checkpoint_revision'],
        strict_load=True,tf32=False,input_size=518,fp32=False,blackwell_fa3_disabled=fa3_disabled))
    for row in manifest['sequences']:
        name = row['id']
        if selected and name not in selected: continue
        dest = out/'sequences'/name; done=dest/'geometry.json'
        if done.exists():
            prior=json.loads(done.read_text())
            assert prior['source_frames']==row['source_frames'] and len(list((dest/'disparity').glob('*.png')))==row['source_frames']
            print('REUSE',name,flush=True);continue
        frames=np.stack([cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB) for p in sorted((dest/'rgb').glob('*.jpg'))])
        assert len(frames)==row['source_frames']
        start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        depths,_=model.infer_video_depth(frames,1,input_size=518,device='cuda',fp32=False)
        assert depths.shape==frames.shape[:3] and np.isfinite(depths).all()
        lo,hi=depths.min(),depths.max();assert hi>lo
        details=[]
        for i,d in enumerate(depths):
            q=((d-lo)/(hi-lo)*255).astype(np.uint8)
            mask=cv2.imread(str(dest/'mask'/f'{i:05d}.png'),cv2.IMREAD_GRAYSCALE)
            assert mask.shape==q.shape
            if i < 25:
                assert (mask>128).any()
                raw_focus=float(q[mask>128].mean()/255)
                focus_text=f'{raw_focus:.5f}'; focus=float(focus_text)
                filename=f'{i:05d}_zf_{focus_text}.png'
            else:
                raw_focus=focus=None
                filename=f'{i:05d}.png'
            png(dest/'disparity'/filename,q)
            if i<25: link(out/'eval'/name/'disparity'/filename,dest/'disparity'/filename)
            details.append(dict(frame=i,file=filename,focus=focus,focus_unrounded=raw_focus,
                                quantized_min=int(q.min()),quantized_max=int(q.max())))
        result=dict(id=name,source_frames=len(depths),min=float(lo),max=float(hi),
            seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            finite=True,normalization='per-sequence all original frames',frames=details)
        write_new(done,result)
        print('VDA_DONE',name,len(depths),round(result['seconds'],2),flush=True)
        del frames,depths;torch.cuda.empty_cache()


def check(c):
    out=ROOT/c['derived'];manifest=json.loads((out/'source-manifest.json').read_text());rows=[]
    for row in manifest['sequences']:
        name=row['id'];dest=out/'sequences'/name
        g=json.loads((dest/'geometry.json').read_text()); assert g['max']>g['min']
        assert len(g['frames'])==row['source_frames']
        for r in g['frames']:
            q=cv2.imread(str(dest/'disparity'/r['file']),cv2.IMREAD_UNCHANGED)
            mask=cv2.imread(str(dest/'mask'/f"{r['frame']:05d}.png"),cv2.IMREAD_UNCHANGED)
            assert q.dtype==np.uint8 and q.shape==mask.shape and set(np.unique(mask))<={0,255}
            if r['frame'] < 25:
                assert float(f'{q[mask>128].mean()/255:.5f}')==r['focus']
            else:
                assert r['focus'] is None
        for sub in ('rgb','mask','disparity'):
            assert len(list((out/'eval'/name/sub).iterdir()))==25
        rows.append(dict(id=name,source_frames=g['source_frames'],min=g['min'],max=g['max'],
            seconds=g['seconds'],peak_allocated_bytes=g['peak_allocated_bytes'],
            evaluation_focus=[r['focus'] for r in g['frames'][:25]]))
    result=dict(status='ALL_50_SEQUENCE_GEOMETRY_AND_1250_EVALUATION_FRAMES_CHECKED',
        sequences=50,source_frames=sum(r['source_frames'] for r in rows),evaluation_frames=1250,
        max_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
        rows=rows,scope='Data preparation only; baseline outputs and DeepLens aperture remain separate')
    write_new(out/'geometry-check.json',result)
    write_new(ROOT/'03_实验记录/records/davis-v2-geometry-20260929.json',result)
    print('GEOMETRY_CHECK_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['assets','vda','check','all'])
    p.add_argument('--config',default='configs/davis-eval-v2.json');p.add_argument('--sequences',nargs='*')
    a=p.parse_args();c=json.loads((ROOT/a.config).read_text());cv2.setNumThreads(2)
    if a.phase in ('assets','all'):assets(c)
    if a.phase in ('vda','all'):vda(c,a.sequences)
    if a.phase in ('check','all'):check(c)
