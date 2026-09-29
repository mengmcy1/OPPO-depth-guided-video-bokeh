"""Author-public MP4 VEPI plus explicitly separate lossless PNG diagnostic."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_davis_v2 import write_new,png


def author_module(c):
    repo=(ROOT/c['vepi']['author_repository']).resolve()
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==c['vepi']['author_commit']
    spec=importlib.util.spec_from_file_location('author_vepi',repo/'metrics/edge_batch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def encode(c,frames,path):
    assert len(frames)==25
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True)
    codec=c['vepi']['codec'];tmp=path.with_suffix('.partial.mp4')
    # A failed encoding is never counted as a completed video.
    assert not tmp.exists(), f'Inspect incomplete encoding before resuming: {tmp}'
    with imageio.get_writer(str(tmp),format='FFMPEG',fps=codec['fps'],codec=codec['codec'],
                            quality=codec['quality'],pixelformat=codec['pixelformat'],
                            macro_block_size=codec['macro_block_size'],ffmpeg_params=['-threads','2']) as writer:
        for p in frames:
            bgr=cv2.imread(str(p));assert bgr is not None and bgr.shape==(576,1024,3)
            writer.append_data(bgr[:,:,::-1])
    tmp.rename(path)


def scores(c,name,pred,video,author):
    data=ROOT/c['derived']/'eval'/name
    refs=sorted((data/'rgb').glob('*.jpg'));masks=sorted((data/'mask').glob('*.png'))
    frames=sorted(pred.glob('*.png'))
    assert len(refs)==len(masks)==len(frames)==25
    assert [p.stem for p in refs]==[p.stem for p in masks]==[p.stem for p in frames]
    encode(c,frames,video)
    cap=cv2.VideoCapture(str(video));decoded=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        assert f.shape==(576,1024,3)
        decoded.append(f)
    cap.release();assert len(decoded)==25
    png_values=[];mp4_values=[]
    for ref,mask,prediction,decoded_frame in zip(refs,masks,frames,decoded):
        bgr=cv2.imread(str(prediction));reference=cv2.imread(str(ref));m=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
        assert bgr is not None and reference is not None and m is not None
        reference=cv2.resize(reference,(1024,576),interpolation=cv2.INTER_LINEAR)
        m=cv2.resize(m,(1024,576),interpolation=cv2.INTER_LINEAR)
        m=(cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=1)>127).astype(np.uint8)
        assert m.any()
        edge=author.edge_map(reference)
        png_values.append(float(author.correlation(edge,author.edge_map(bgr),m)))
        mp4_values.append(float(author.correlation(edge,author.edge_map(decoded_frame),m)))
    public,_=author.bokeh_epi_eval_single(str(data/'rgb'),str(video),str(data/'mask'),name,None)
    assert len(public)==25 and np.array_equal(public,np.asarray(mp4_values)), 'Wrapper differs from original MP4 entrance'
    assert np.isfinite(png_values+mp4_values).all()
    return dict(id=name,frames=25,VEPI_public_mp4=float(np.mean(public)),VEPI_png=float(np.mean(png_values)),
        codec_delta=float(np.mean(public)-np.mean(png_values)),frame_mp4=mp4_values,frame_png=png_values,
        unmodified_author_entrance_equal=True)


def run(c,predictions,out,self_test):
    cv2.setNumThreads(2);author=author_module(c)
    assert imageio_ffmpeg.__version__==c['vepi']['codec']['imageio_ffmpeg']
    manifest=json.loads((ROOT/c['derived']/'source-manifest.json').read_text());rows=[]
    for scene in manifest['sequences']:
        name=scene['id'];saved=out/'scores'/f'{name}.json'
        if saved.exists():rows.append(json.loads(saved.read_text()));continue
        pred=predictions/name
        if self_test:
            for p in sorted((ROOT/c['derived']/'eval'/name/'rgb').glob('*.jpg')):
                png(pred/(p.stem+'.png'),cv2.resize(cv2.imread(str(p)),(1024,576),interpolation=cv2.INTER_LINEAR))
        r=scores(c,name,pred,out/'mp4'/f'{name}.mp4',author)
        if self_test:assert r['VEPI_png']==1., 'Identity edge self-comparison must be exactly1'
        write_new(saved,r);rows.append(r)
        print('VEPI_DONE',name,r['VEPI_public_mp4'],r['codec_delta'],flush=True)
    result=dict(status='SELF_TEST_AND_CODEC_DIAGNOSTIC_COMPLETE' if self_test else 'VEPI_COMPLETE',
        videos=len(rows),frames=sum(r['frames'] for r in rows),
        VEPI_public_mp4=float(np.mean([r['VEPI_public_mp4'] for r in rows])),
        VEPI_png=float(np.mean([r['VEPI_png'] for r in rows])),
        codec_delta=float(np.mean([r['codec_delta'] for r in rows])),
        codec_delta_range=[min(r['codec_delta'] for r in rows),max(r['codec_delta'] for r in rows)],
        codec=dict(c['vepi']['codec'],threads=2),ffmpeg=imageio_ffmpeg.get_ffmpeg_version(),
        cv2=cv2.__version__,imageio=__import__('imageio').__version__,
        scope='AIF identity control, not a bokeh baseline result' if self_test else 'Author-public MP4 primary; PNG diagnostic only',
        rows=rows)
    write_new(out/'summary.json',result)
    if self_test:write_new(ROOT/'03_实验记录/records/davis-v2-vepi-codec-selftest-20260929.json',result)
    print('VEPI_COMPLETE',result['VEPI_public_mp4'],result['codec_delta'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/davis-eval-v2.json')
    p.add_argument('--self-test',action='store_true');p.add_argument('--predictions');p.add_argument('--output')
    a=p.parse_args();c=json.loads((ROOT/a.config).read_text())
    out=ROOT/(a.output or 'outputs/davis-v2-codec-selftest-20260929')
    assert a.self_test or (a.output and a.predictions)
    pred=out/'identity-png' if a.self_test else ROOT/a.predictions
    run(c,pred,out,a.self_test)
