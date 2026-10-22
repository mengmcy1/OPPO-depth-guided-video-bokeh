"""Paired numerical checks and local review artifacts; no model/visual grading."""
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.review.any2bokeh_pages import install_assets,viewer_fragment,render_review_form
C=json.loads((ROOT/'configs/ours-sintel-reference-atb-20261023.json').read_text())
BASE=json.loads((ROOT/C['source_config']).read_text());OUT=ROOT/C['output'];PREFIX=ROOT/C['record_prefix']


def write_json(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def gradient(im):
    gray=cv2.cvtColor(im.astype(np.float32),cv2.COLOR_RGB2GRAY)
    return cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)**2+cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)**2


def analyze():
    summaries=[];stats=[];perframe=[]
    for name in C['samples']:
        scene=OUT/'scenes'/name
        complete={v:json.loads((scene/v/'complete.json').read_text()) for v in C['variants']}
        runtime={v:json.loads((scene/v/'runtime.json').read_text()) for v in C['variants']}
        a=np.load(scene/'A/float.npy',mmap_mode='r');rep=np.load(scene/'A_repeat/float.npy',mmap_mode='r');r=np.load(scene/'R/float.npy',mmap_mode='r')
        assert a.shape==rep.shape==r.shape==(25,436,1024,3)
        for v in C['variants']:
            assert complete[v]['frames']==25 and len(complete[v]['events'])==6
            assert runtime[v]['entry_rng']==runtime['A']['entry_rng'] and complete[v]['end_rng']==complete['A']['end_rng']
            for aa,bb in zip(complete['A']['events'],complete[v]['events']):
                for key in ['unit','branch','shape','rng_before','rng_after']:assert aa[key]==bb[key]
                if aa['branch']=='RGB':assert aa['latent_sha256']==bb['latent_sha256']
            for key in ['pixel_values','k','mpi_mask','weight']:assert runtime[v]['base_hashes'][key]==runtime['A']['base_hashes'][key]
        bit_diff=0;byte_diff=0;response_elements=0;response_max=0.;response_sum=0.;byte_response_sum=0;byte_response_max=0
        for i in range(25):
            assert np.isfinite(a[i]).all() and np.isfinite(rep[i]).all() and np.isfinite(r[i]).all()
            bd=int(np.count_nonzero(a[i].view(np.uint32)!=rep[i].view(np.uint32)));bit_diff+=bd
            images={v:np.asarray(Image.open(scene/v/'png'/f'{i:05d}.png')) for v in C['variants']}
            assert all(im.shape==(436,1024,3) and im.dtype==np.uint8 for im in images.values())
            byte_diff+=int(np.count_nonzero(images['A']!=images['A_repeat']))
            delta=np.abs(r[i]-a[i]);response_elements+=int(np.count_nonzero(delta));response_max=max(response_max,float(delta.max()));response_sum+=float(delta.sum(dtype=np.float64))
            bytes_delta=np.abs(images['R'].astype(np.int16)-images['A'].astype(np.int16));byte_response_sum+=int(bytes_delta.sum());byte_response_max=max(byte_response_max,int(bytes_delta.max()))
            rgb=np.asarray(Image.open(ROOT/BASE['raw_root']/'training/clean'/name/f'frame_{i+1:04d}.png'))
            grads={v:gradient(im) for v,im in images.items() if v!='A_repeat'};grad_rgb=gradient(rgb)
            with np.load(scene/'regions'/f'{i:05d}.npz') as masks:
                for region in masks.files:
                    mask=masks[region];pixels=int(mask.sum())
                    row=dict(scene=name,frame=i,source_frame=i+1,region=region,pixels=pixels,
                             A_input_MAE=None,R_input_MAE=None,A_gradient_energy_ratio=None,R_gradient_energy_ratio=None,
                             input_gradient_energy=None,A_gradient_energy=None,R_gradient_energy=None,R_A_RGB8_MAE=None)
                    if pixels:
                        denominator=float(grad_rgb[mask].mean(dtype=np.float64));row['input_gradient_energy']=denominator
                        for v in ['A','R']:
                            row[v+'_input_MAE']=float(np.abs(images[v].astype(np.float32)-rgb.astype(np.float32))[mask].mean(dtype=np.float64))
                            e=float(grads[v][mask].mean(dtype=np.float64));row[v+'_gradient_energy']=e
                            row[v+'_gradient_energy_ratio']=e/denominator if denominator>1e-8 else None
                        row['R_A_RGB8_MAE']=float(bytes_delta[mask].mean(dtype=np.float64))
                    stats.append(row)
            perframe.append(dict(scene=name,frame=i,float_max_diff=float(delta.max()),float_mean_abs_diff=float(delta.mean(dtype=np.float64)),
                                 RGB8_max_diff=int(bytes_delta.max()),RGB8_mean_abs_diff=float(bytes_delta.mean())))
        assert bit_diff==0 and byte_diff==0,'Reproducibility failed; do not attribute candidate response'
        assert response_elements>0,'Reference CoC did not produce a response'
        summaries.append(dict(scene=name,frames=25,focus=json.loads((scene/'prepared.json').read_text())['focus'],
                              A_repeat_float_bit_differences=bit_diff,A_repeat_RGB8_byte_differences=byte_diff,
                              R_A_float_changed_elements=response_elements,R_A_float_max_diff=response_max,R_A_float_MAE=response_sum/a.size,
                              R_A_RGB8_MAE=byte_response_sum/a.size,R_A_RGB8_max_diff=byte_response_max,
                              posterior_RNG_boundaries_equal=True,RGB_latents_equal=True,
                              peak_allocated_GiB=max(x['peak_allocated_bytes'] for x in complete.values())/2**30))
    aggregated=[]
    for name in C['samples']:
        for region in sorted({s['region'] for s in stats}):
            selected=[s for s in stats if s['scene']==name and s['region']==region and s['pixels']]
            total=sum(s['pixels'] for s in selected)
            row=dict(scene=name,region=region,pixels_across_frames=total,nonempty_frames=len(selected))
            for key in ['A_input_MAE','R_input_MAE','input_gradient_energy','A_gradient_energy','R_gradient_energy','R_A_RGB8_MAE']:
                row[key]=sum(s[key]*s['pixels'] for s in selected)/total if total else None
            for v in ['A','R']:
                row[v+'_gradient_energy_ratio']=row[v+'_gradient_energy']/row['input_gradient_energy'] if total and row['input_gradient_energy']>1e-8 else None
            aggregated.append(row)
    result=dict(status='PAIRED_REFERENCE_CONDITION_RESPONSE_VERIFIED_VISUAL_REVIEW_PENDING',config=C,scenes=summaries,regions=aggregated,
                aggregation='pixel-weighted across25 frames; gradient ratios are ratio of summed region energies; empty regions null',
                metric_units='RGB8 MAE in 0..255 channel units; gradient energy squared Sobel(grayRGB8); float RGB in0..1',
                limits=['Input RGB is sharp-input reference, not Bokeh GT; these diagnostics do not prove perceptual benefit.',
                        'Fixed input-based center focus does not define an entire semantic subject; corrected geometry may change which parts are near focus.',
                        'Reference has fitted scale/shift and quantization; not physical ground-truth bokeh or strict upper bound.',
                        'No motion-compensated temporal metric; no automatic flicker conclusions.'])
    write_json(Path(str(PREFIX)+'-results.json'),result)
    for suffix,rows in [('-regions.csv',stats),('-frames.csv',perframe),('-regions-summary.csv',aggregated)]:
        with Path(str(PREFIX)+suffix).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(summaries,indent=2),flush=True)


def link(dst,src):
    assert src.exists();dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_symlink():
        assert dst.resolve()==src.resolve()
    else:
        assert not dst.exists()
        dst.symlink_to(os.path.relpath(src,dst.parent))


def review():
    results=json.loads(Path(str(PREFIX)+'-results.json').read_text());assert all(r['A_repeat_float_bit_differences']==0 for r in results['scenes'])
    import imageio_ffmpeg
    install_assets(OUT);samples=[]
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for index,name in enumerate(C['samples'],1):
        dest=OUT/'view'/name;dest.mkdir(parents=True,exist_ok=True)
        panels=[dict(label='输入 RGB',prefix='RGB/'),dict(label='原 VDA 条件 A',prefix='A/'),dict(label='参考条件 R',prefix='R/')]
        for i in range(25):
            link(dest/'RGB'/f'{i:05d}.png',ROOT/BASE['raw_root']/'training/clean'/name/f'frame_{i+1:04d}.png')
            for label in ['A','R']:link(dest/label/f'{i:05d}.png',OUT/'scenes'/name/label/'png'/f'{i:05d}.png')
        cmd=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n']
        for label in ['RGB','A','R']:cmd+=['-framerate','20','-i',str(dest/label/'%05d.png')]
        cmd+=['-filter_complex','[0:v][1:v][2:v]hstack=inputs=3[v]','-map','[v]','-frames:v','25','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart','-threads','4',str(dest/'comparison.mp4')]
        subprocess.run(cmd,check=True)
        sheet=Image.new('RGB',(3*340,5*170),'white');draw=ImageDraw.Draw(sheet)
        for row,i in enumerate([0,6,12,18,24]):
            for col,label in enumerate(['RGB','A','R']):
                im=Image.open(dest/label/f'{i:05d}.png');sheet.paste(im.resize((340,145)),(col*340,row*170+22));draw.text((col*340+4,row*170+3),f'{label} frame{i} / source{i+1}',fill='black')
        sheet.save(dest/'contact.png')
        focus=next(r['focus'] for r in results['scenes'] if r['scene']==name)
        item=dict(sample_id=f'S{index:02d}',dataset='Sintel',scene_id=name,source_group=name.rsplit('_',1)[0],split='pilot_train',package='B1',panels=panels,review_url='../../review.html',frame_jumps=[0,6,12,18,24])
        fragment=viewer_fragment(item);assert fragment.count('"height": 576')==1;fragment=fragment.replace('"height": 576','"height": 436')
        page=f'''<!doctype html><meta charset="utf-8"><title>{name}参考条件对照</title>
<style>body{{font-family:Arial,sans-serif;background:#f5f8ff;margin:24px;color:#153956}}video{{width:100%}}a{{color:#075aa0}}</style>
<h1>{name}：原条件与参考条件对照</h1>
<p><a href="../../index.html">返回索引</a> · <a href="../../review.html">填写观察</a> · <a href="contact.png">五关键帧接触图</a></p>
<p>左：输入RGB；中：原VDA条件A；右：参考条件R。两组权重、RGB、MPI、K16及随机流相同，仅CoC变化。原版重复运行逐位一致。</p>
<p>焦点{focus:.5f}来自首帧中心32×32区域的原VDA均值，整段固定；不代表语义主体全部应清晰。参考修正后，该中心区域也可能不再处于同一焦平面。</p>
<p>原1024×436，上下各补6像素后推理，再裁回原图；PNG原像素观看。页面帧0对应源frame_0001。25帧视频20fps仅供观看；循环24→0不是连续变化。</p>
<p>建议先1×看整体，再0.25×或逐帧比较头发/轮廓、近焦细节、背景虚化及边缘。可以记录更好、更差、无明显差别或无法判断；参考条件不是虚化真值。</p>
<video controls preload="metadata" src="comparison.mp4"></video>{fragment}'''
        (dest/'index.html').write_text(page)
        samples.append(item)
    notice='本页仅记录用户视觉观察。RGB不是配对Bokeh真值；数值差异不能自动证明画质提升。原版重复已逐位一致。'
    form=render_review_form(samples,package='B1',store_key='ours-sintel-reference-20261023',title='Sintel参考条件诊断反馈',notice=notice)
    form=form.replace('../review-tools/','review-tools/')
    (OUT/'review.html').write_text(form)
    index='<!doctype html><meta charset="utf-8"><title>Sintel参考条件诊断</title><style>body{font-family:Arial;margin:32px;background:#f5f8ff;color:#153956}li{margin:20px}</style><h1>Sintel参考条件诊断</h1><p>两段、每段25帧。每段先看原RGB/原条件/参考条件同步视频，再逐帧或放大局部。这里只做条件对照，尚未训练修正器。</p><ol>'
    for item in samples:index+=f'<li><a href="view/{item["scene_id"]}/">{item["sample_id"]} · {item["scene_id"]}</a></li>'
    index+='</ol><p><a href="review.html">填写并导出观察记录</a></p><p>关注：近焦细节是否更好、合理背景虚化是否保留、边缘是否异常、跨帧是否稳定。没有差别或变差也应保留。</p>'
    (OUT/'index.html').write_text(index)
    # Reuse the already-running local review service; never restart unrelated services.
    link(ROOT/'outputs/any2bokeh-failure-analysis-20261009'/OUT.name,OUT)
    print('REVIEW_READY',OUT,flush=True)


def condition_audit():
    # Post-hoc interpretation only: never chooses parameters or additional runs.
    rows=[]
    for name in C['samples']:
        root=OUT/'scenes'/name;info=json.loads((root/'prepared.json').read_text());f=info['focus'];aa=[];rr=[]
        for i in range(25):
            a=np.asarray(Image.open(next((root/'A-disparity').glob(f'{i:05d}_*.png'))))[6:-6].astype(float)/255
            r=np.asarray(Image.open(next((root/'R-disparity').glob(f'{i:05d}_*.png'))))[6:-6].astype(float)/255
            with np.load(root/'regions'/f'{i:05d}.npz') as z:mask=z['near_focus']
            aa.append((16*abs(a-f))[mask]);rr.append((16*abs(r-f))[mask])
        a=np.concatenate(aa);r=np.concatenate(rr)
        rows.append(dict(scene=name,region='near_focus',pixels=len(a),A_nominal_mean=float(a.mean()),R_nominal_mean=float(r.mean()),
                         R_greater_fraction=float((r>a).mean()),R_less_fraction=float((r<a).mean())))
    result=dict(scope='post-hoc input-only interpretation of the predeclared near-focus bin; no parameter selection or new inference',rows=rows)
    path=Path(str(PREFIX)+'-condition-audit.json')
    if path.exists():
        old=json.loads(path.read_text());assert old['scope']==result['scope']
        for x,y in zip(old['rows'],result['rows']):
            assert x.keys()==y.keys()
            for k in x:
                assert np.isclose(x[k],y[k],rtol=1e-12,atol=1e-12) if isinstance(x[k],float) else x[k]==y[k]
    else:write_json(path,result)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['analyze','review','condition_audit']);args=p.parse_args();globals()[args.phase]()
