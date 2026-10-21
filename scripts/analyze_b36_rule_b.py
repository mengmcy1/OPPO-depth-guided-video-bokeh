"""Input-defined B36 regions, CPU paired response statistics and review material."""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw

from run_b36_rule_b import ROOT,CFG,read,save
from prepare_b36_rule_b import original_geometry,input_rgb,native_box,display_box,match
from check_atb_mpi_invariance import compare


def csv_write(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def bbox(mask):
    ys,xs=np.where(mask)
    return [int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)] if xs.size else None


def rectangle(box,shape=(576,1024)):
    mask=np.zeros(shape,bool)
    if box is not None:
        x,y,w,h=box;x0,y0=max(0,x),max(0,y);x1,y1=min(shape[1],x+w),min(shape[0],y+h)
        mask[y0:y1,x0:x1]=True
    return mask


def regions(c,out):
    selection=read(ROOT/c['selection_config']);sr=read(ROOT/c['selection_record']);source=ROOT/selection['output']/'B36'
    original=read(ROOT/selection['reference']['anchor_config'])['B36'];policy=selection['reference'];anchors=original['keyframes']
    grays={t:cv2.cvtColor(input_rgb(selection,t),cv2.COLOR_RGB2GRAY) for t in anchors}
    bounds={t:bbox(original_geometry(selection,t)[1]) for t in anchors}
    initial=bounds[0];frames=[]
    for t in range(25):
        q,union,_=original_geometry(selection,t);bb=bbox(union);anchor=min(anchors,key=lambda v:(abs(v-t),v))
        selected=np.asarray(Image.open(source/'native/Selected_mask'/f'{t:05d}.png'))!=0
        neck=np.asarray(Image.open(source/'native/Reference_mask'/f'{t:05d}.png'))!=0
        r=sr['rows'][t]['reference_disparity']
        high=selected&(q.astype(np.float32)/255>r) if r is not None else np.zeros_like(union)
        low=selected&~high
        small=lambda a:cv2.resize(a.astype(np.uint8),(1024,576),interpolation=cv2.INTER_NEAREST)>0
        fg,sel=small(union),small(selected)
        sets=[('head_high_side',small(high),'approved selected high disparity support; head proxy',None)]
        n,labels,stats,_=cv2.connectedComponentsWithStats(low.astype(np.uint8),connectivity=8)
        ids=sorted(range(1,n),key=lambda i:(stats[i,1],stats[i,0],i))
        for j,i in enumerate(ids):sets.append((f'low_component_{j+1:02d}',small(labels==i),'one low-side component; frame-local ID, not tracked',int(stats[i,4])))
        if not ids:sets.append(('low_component_none',np.zeros((576,1024),bool),'no approved low-side component',0))
        sets.append(('neck',small(neck),'saved original reference mask; invalid21 empty',int(neck.sum())))
        # RGB-only body localization, independent of the neck match and predictions.
        b=native_box(original['ROIs_display_xywh'][str(anchor)]['body_core']);x,y,w,h=b;reason='';checks={}
        if t not in anchors:
            current=cv2.cvtColor(input_rgb(selection,t),cv2.COLOR_RGB2GRAY)
            target,score,margin=match(current,grays[anchor][y:y+h,x:x+w],b,policy['search_radius_native_xy'])
            xx,yy,ww,hh=target;back,reverse,_=match(grays[anchor],current[yy:yy+hh,xx:xx+ww],target,policy['search_radius_native_xy'])
            error=float(np.hypot(back[0]-x,back[1]-y));checks=dict(NCC=score,peak_margin=margin,roundtrip_error_native=error)
            if score<policy['min_NCC'] or margin<policy['min_peak_margin_outside_3_native_pixels'] or error>policy['max_roundtrip_error_native_pixels']:reason='body RGB localization failed fixed checks'
            b=target
        body_box=display_box(b) if not reason else None
        sets.append(('body_unselected',rectangle(body_box)&fg&~sel,'input-RGB body rectangle intersect union excluding selected pixels',None))
        context=original['ROIs_display_xywh'][str(anchor)]['head_context'].copy()
        context[0]+=int(round((bb[0]-bounds[anchor][0])/1.875));context[1]+=int(round((bb[1]-bounds[anchor][1])/1.875))
        sets.append(('head_context',rectangle(context),'supplementary input-defined context; includes neck/background',None))
        water=[745+int(round((bb[0]-initial[0])/1.875)),170+int(round((bb[1]-initial[1])/1.875)),35,30]
        sets.extend([('water_near',rectangle(water)&~fg,'subject-relative water sampling window, not material tracking',None),
                     ('water_far',rectangle([90,85,150,100])&~fg,'fixed historical water ROI',None)])
        folder=out/'regions'/f'{t:05d}';folder.mkdir(parents=True)
        info=[]
        for name,mask,definition,native_count in sets:
            Image.fromarray(np.uint8(mask)*255).save(folder/f'{name}.png')
            info.append(dict(region=name,pixels=int(mask.sum()),native_pixels=native_count,display_bbox=bbox(mask),definition=definition,
                path=str((folder/f'{name}.png').relative_to(out))))
        frames.append(dict(frame_id=t,head_context_xywh=context,body_display_xywh=body_box,body_checks=checks,body_missing_reason=reason,regions=info))
    value=dict(created_UTC=datetime.now(timezone.utc).isoformat(),policy=c['analysis_policy'],frames=frames,
        definition_source='Only original RGB, annotations, prior boxes and approved masks. No new predictions read.',
        limitation='All empty or failed supports remain empty; no changes to intervention rules or frame21 reference.')
    save(out/'regions.json',value);save(ROOT/(c['record_prefix']+'-regions.json'),value)
    print('REGIONS',sum(len(x['regions']) for x in frames),'body missing',[x['frame_id'] for x in frames if x['body_missing_reason']],flush=True)


def energy(rgb):
    x=rgb.astype(np.float64)/255;gray=x[...,0]*.299+x[...,1]*.587+x[...,2]*.114
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    return gx*gx+gy*gy


def local_metrics(pred,rgb,mask,gradient_mask,pred_energy,ref_energy):
    if not mask.any():return dict(RGB8_MAE=None,gradient_energy_ratio=None,gradient_reason='empty region')
    mae=float(np.abs(pred.astype(float)-rgb.astype(float))[mask].mean())
    denominator=float(ref_energy[gradient_mask].mean()) if gradient_mask.any() else None
    ratio=None if denominator is None or denominator<=1e-15 else float(pred_energy[gradient_mask].mean()/denominator)
    return dict(RGB8_MAE=mae,gradient_energy_ratio=ratio,gradient_reason='' if ratio is not None else 'empty eroded support or zero input gradient')


def analyze(c,out):
    reg=read(out/'regions.json');prepared=read(out/'prepared.json')
    allruns={v:dict(runtime=read(out/v/'runtime.json'),events=read(out/v/'events.json'),complete=read(out/v/'complete.json')) for v in c['conditions']}
    a=allruns['native'];execution=[]
    for name,r in allruns.items():
        assert r['runtime']['entry_RNG']==a['runtime']['entry_RNG'] and r['complete']['end_RNG']==a['complete']['end_RNG']
        assert len(r['events']['posterior'])==6
        for v,base in zip(r['events']['posterior'],a['events']['posterior']):
            for k in ['unit','branch','shape','rng_before','rng_after','generator']:assert v[k]==base[k]
        assert [x['output_frames'] for x in r['events']['units']]==[8,8,9]
        for v,base in zip(r['events']['units'],a['events']['units']):
            assert v['groups']==base['groups'] and v['rng_before']==base['rng_before'] and v['rng_after']==base['rng_after']
            assert v['CPU_generator_before']==v['CPU_generator_after']==base['CPU_generator_before']
            for k in ['pixel_values','k','mpi_mask','weight']:assert v['input_sha256'][k]==base['input_sha256'][k]
        execution.append(dict(condition=name,entry_and_end_RNG_equal=True,posterior_order_and_RNG_equal=True,
            fixed_GPU_inputs_equal=True,first_latent_cache_none=r['events']['units'][0]['latent_cache_in'] is None,
            encoder_cache_initially_absent=r['runtime']['initial_encoder_cache_absent'],pid=r['runtime']['pid']))
    comparisons={v:compare(out,v) for v in ['native_repeat','rule_b']}
    floats={v:np.load(out/v/'float.npy',mmap_mode='r') for v in c['conditions']}
    selection=read(ROOT/c['selection_config']);source=ROOT/selection['output']/'B36/originals/Input_RGB'
    rows=[];images={}
    for t in range(25):
        rgb=np.asarray(Image.open(source/f'{t:05d}.png'));pred={v:np.asarray(Image.open(out/v/'png'/f'{t:05d}.png')) for v in c['conditions']}
        images[t]=dict(Input=rgb,**pred);energies={v:energy(x) for v,x in pred.items()};ref_energy=energy(rgb)
        for variant in ['native_repeat','rule_b']:
            delta_bytes=np.abs(pred[variant].astype(np.int16)-pred['native'].astype(np.int16))
            comparisons[variant]['per_frame'][t].update(RGB8_mean_abs_diff=float(delta_bytes.mean()),RGB8_max_abs_diff=int(delta_bytes.max()))
        for region in reg['frames'][t]['regions']:
            mask=np.asarray(Image.open(out/region['path']))!=0;gm=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
            row=dict(frame_id=t,region=region['region'],region_mask_path=region['path'],region_display_bbox=region['display_bbox'],
                pixels=int(mask.sum()),gradient_pixels=int(gm.sum()),input_CoC_changed=t<=20)
            for variant,label in c['condition_labels'].items():
                row.update({label+'_'+k:v for k,v in local_metrics(pred[variant],rgb,mask,gm,energies[variant],ref_energy).items()})
            for variant,label in [('native_repeat','repeat_vs_A'),('rule_b','R_vs_A')]:
                diff=np.abs(floats[variant][t].astype(np.float64)-floats['native'][t].astype(np.float64))
                row[label+'_float_max']=float(diff[mask].max()) if mask.any() else None
                row[label+'_float_MAE']=float(diff[mask].mean()) if mask.any() else None
                row[label+'_RGB8_MAE']=float(np.abs(pred[variant].astype(float)-pred['native'].astype(float))[mask].mean()) if mask.any() else None
                row[label+'_RGB8_different_bytes']=int(np.count_nonzero(pred[variant][mask]!=pred['native'][mask])) if mask.any() else None
            rows.append(row)
    temporal=[]
    for t in [18,19,20,21,22]:
        for name in ['head_high_side','head_context']:
            paths=[next(r['path'] for r in reg['frames'][i]['regions'] if r['region']==name) for i in [t,t+1]]
            mask=(np.asarray(Image.open(out/paths[0]))!=0)|(np.asarray(Image.open(out/paths[1]))!=0)
            row=dict(transition=f'{t}->{t+1}',region=name+'_pair_union',pixels=int(mask.sum()),motion_compensated=False)
            changes={v:images[t+1][v].astype(float)-images[t][v].astype(float) for v in ['Input']+c['conditions']}
            for variant,label in c['condition_labels'].items():
                row[label+'_temporal_RGB8_MAE']=float(np.abs(changes[variant])[mask].mean()) if mask.any() else None
                row[label+'_input_relative_change_error']=float(np.abs(changes[variant]-changes['Input'])[mask].mean()) if mask.any() else None
            for variant,label in [('native_repeat','repeat_vs_A'),('rule_b','R_vs_A')]:
                row[label+'_temporal_response_MAE']=float(np.abs(changes[variant]-changes['native'])[mask].mean()) if mask.any() else None
            temporal.append(row)
    csv_write(ROOT/(c['record_prefix']+'-regions.csv'),rows);csv_write(ROOT/(c['record_prefix']+'-temporal.csv'),temporal)
    skipped=[dict(frame=t,rule_b=comparisons['rule_b']['per_frame'][t],repeat_noise=comparisons['native_repeat']['per_frame'][t],
        CoC_input_bitwise_unchanged=True) for t in [21,22,23,24]]
    result=dict(config=c,prepared=prepared,region_definition_record=c['record_prefix']+'-regions.json',execution_checks=execution,runs=allruns,comparisons=comparisons,
        per_frame_region_metrics=rows,temporal_region_change=temporal,unchanged_condition_frames=skipped,
        status='配对推理与CPU统计完成，等待用户视觉复核',user_visual_review_completed=False,
        limits=['Only B36 exploration source; no paired bokeh GT. Input-relative MAE/edge ratios are diagnostic, not quality scores.',
            'High-side selected support is empty21-24; its metrics are N/A. Supplementary head-context window is a separate mixed region, never substituted under the same label.',
            'Low-side components are reported separately with frame-local IDs; no cross-region averaging.',
            'Uncompensated temporal changes include camera/object motion, mask support and model response; not flicker.',
            'Unmodified CoC on21-24 does not guarantee unchanged outputs under the original overlapping temporal execution.'])
    save(ROOT/(c['record_prefix']+'.json'),result);save(out/'analysis.json',result)
    print('ANALYSIS',len(rows),'region rows',len(temporal),'temporal rows',flush=True)
    for k,v in comparisons.items():print('COMPARE',k,{key:val for key,val in v.items() if key!='per_frame'},flush=True)


def link(p,source):
    p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists();p.symlink_to(os.path.relpath(source,p.parent))


def materials(c,out):
    import imageio_ffmpeg
    sys.path.insert(0,str(ROOT/'src'))
    from review.any2bokeh_pages import viewer_fragment,render_review_form,payload
    result=read(out/'analysis.json');reg=read(out/'regions.json');selection=read(ROOT/c['selection_config'])
    old=ROOT/selection['output']/'B36/originals/Input_RGB';page=out/'B36';assert not page.exists();page.mkdir()
    labels=['Input_RGB','A','R']
    for label in labels:(page/'originals'/label).mkdir(parents=True)
    frames=out/'video-frames';frames.mkdir()
    full=Image.new('RGB',(3*384,len(c['keyframes'])*240),'white')
    close=Image.new('RGB',(3*256,len(c['keyframes'])*344),'white')
    tracks=[]
    for name,title in [('head_context','头部上下文，含颈部/背景'),('head_high_side','高视差选区bbox，21–24为空'),('neck','颈部原参考'),('body_unselected','身体非选区'),('water_near','邻近水面'),('water_far','远处水面')]:
        tracks.append(dict(id=name,name=title,xywh_by_frame={str(t):(reg['frames'][t]['head_context_xywh'] if name=='head_context' else next(x['display_bbox'] for x in reg['frames'][t]['regions'] if x['region']==name)) for t in range(25)}))
    low_names=sorted({x['region'] for f in reg['frames'] for x in f['regions'] if x['region'].startswith('low_component_') and x['region']!='low_component_none'})
    for name in low_names:
        tracks.append(dict(id=name,name='低侧帧内编号，非跨帧对应：'+name,
            xywh_by_frame={str(t):next((x['display_bbox'] for x in reg['frames'][t]['regions'] if x['region']==name),None) for t in range(25)}))
    for t in range(25):
        paths=[old/f'{t:05d}.png',out/'native/png'/f'{t:05d}.png',out/'rule_b/png'/f'{t:05d}.png']
        box=reg['frames'][t]['head_context_xywh'];x,y,w,h=box
        movie=Image.new('RGB',(3072,608),'#12345a');draw=ImageDraw.Draw(movie)
        for j,(label,path) in enumerate(zip(labels,paths)):
            link(page/'originals'/label/f'{t:05d}.png',path)
            with Image.open(path) as im:
                movie.paste(im,(j*1024,32));draw.text((j*1024+8,8),f'{label} | frame{t:02d} | '+('CoC changed' if t<=20 else 'CoC native; temporal context may respond'),fill='white')
                if t in c['keyframes']:
                    row=c['keyframes'].index(t);full.paste(im.resize((384,216),Image.Resampling.BILINEAR),(j*384,row*240+24))
                    ImageDraw.Draw(full).text((j*384+8,row*240+5),f'{label} frame{t:02d}',fill='black')
                    crop=im.crop((x,y,x+w,y+h)).resize((256,320),Image.Resampling.NEAREST)
                    close.paste(crop,(j*256,row*344+24));ImageDraw.Draw(close).text((j*256+8,row*344+5),f'{label} frame{t:02d}',fill='black')
        movie.save(frames/f'{t:05d}.png')
    full.save(page/'contact-full.png');close.save(page/'contact-head.png')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-framerate','20','-i',str(frames/'%05d.png'),
        '-frames:v','25','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','2',str(page/'RGB-A-R.mp4')],check=True)
    item=dict(sample_id='B36',dataset='DAVIS',scene_id=c['scene'],source_group='DAVIS:mallard-water',split='exploration',
        package='B36-rule-b-output',review_url='../review.html',initial_frame=20,frame_jumps=[0,8,16,19,20,21,22,24],
        panels=[dict(label=label,prefix=f'originals/{label}/') for label in labels],tracked_rois=tracks)
    text='''<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>B36规则修正：A/R配对输出</title>
<style>body{font-family:system-ui;background:#f6f9ff;margin:24px}video{width:min(100%,1450px)}table{border-collapse:collapse}td,th{border:1px solid #aac0d6;padding:5px}#atb-viewer .atb-panel-grid{grid-template-columns:repeat(3,minmax(0,1fr))}</style>
<h1>B36 revision1：原RGB／A原版／R规则修正</h1>
<p>仅B36单一探索来源。0–20帧移植已认可CoC，21–24原条件不变，原MPI/RGB/focus/K及原生8/8/9流程保留。未改条件不意味着输出必定不变：重叠窗口、时序UNet/解码及缓存接续均保留。新输出观感由用户判断，当前仅完成程序与CPU统计。</p>
<p>A-repeat对照用于测量本机重复差异，完整结果见JSON/CSV。所有区域分别统计，低侧小块使用帧内编号，不能将同编号当成跨帧同一块。高侧选区21–24为空，相关指标不适用；头部上下文是另一个含颈部/背景的窗口，不能混称纯头部。</p>
<p>请重点观看19/20/21/22，判断鸭头、身体/边界/背景和时序观感；未运动补偿的差分只称区域变化，不称闪烁。20fps展示视频25帧/1.25秒，libx264 CRF18/yuv420p；0.25×仅减播放速度，无插帧。原PNG用于精确逐帧和压缩疑点核对。</p>
<p><a href="../review.html">填写新输出视觉反馈（独立key）</a> · <a href="../regions.csv">各区域逐帧CSV</a> · <a href="../temporal.csv">区域时序变化CSV</a> · <a href="../result.json">完整JSON及逐帧float/RGB8响应</a></p>
<video controls preload="metadata" src="RGB-A-R.mp4"></video>'''
    text+=viewer_fragment(item)
    text+='<h2>固定关键帧接触图：0/8/16/20/21/24</h2><p>头部上下文裁剪统一最近邻放大，三面板同帧同坐标；框定位来自原输入，未按输出调整。</p><a href="contact-head.png"><img src="contact-head.png" style="max-width:100%"></a><details><summary>完整画面对照接触图</summary><a href="contact-full.png"><img src="contact-full.png" style="max-width:100%"></a></details>'
    def number(x):return '不适用' if x is None else f'{x:.6f}' if isinstance(x,float) else str(x)
    fields=['frame_id','region','pixels','gradient_pixels','A_RGB8_MAE','R_RGB8_MAE','A_gradient_energy_ratio','R_gradient_energy_ratio','R_vs_A_RGB8_MAE','repeat_vs_A_RGB8_MAE']
    text+='<details><summary>各区域逐帧统计（不是质量分数）</summary><p>Sobel3×3在全图计算后只用3×3腐蚀后的区域支持，避免mask人为边缘；输入梯度为0或支持为空时不填比值。不能与旧矩形crop口径直接混用。</p><table><tr>'+''.join('<th>'+k+'</th>' for k in fields)+'</tr>'
    for row in result['per_frame_region_metrics']:
        text+='<tr>'+''.join('<td>'+('<a href="../'+row['region_mask_path']+'">'+row[k]+'</a>' if k=='region' else number(row[k]))+'</td>' for k in fields)+'</tr>'
    text+='</table></details><h2>18→23头部附近区域变化</h2><p>每对使用两帧区域的图像坐标并集，没有运动补偿；头部上下文包含颈部/背景。高侧选区两帧均空时留空。</p><table>'
    tf=['transition','region','pixels','A_temporal_RGB8_MAE','R_temporal_RGB8_MAE','A_input_relative_change_error','R_input_relative_change_error','R_vs_A_temporal_response_MAE','repeat_vs_A_temporal_response_MAE']
    text+='<tr>'+''.join('<th>'+k+'</th>' for k in tf)+'</tr>'
    for row in result['temporal_region_change']:text+='<tr>'+''.join('<td>'+number(row[k])+'</td>' for k in tf)+'</tr>'
    text+='</table>'
    (page/'index.html').write_text(text);save(out/'viewer-item.json',item)
    form=render_review_form([item],package='B1',store_key=c['review_storage_key'],title='B36规则修正：用户视觉复核',
        notice='整体选区已获认可；这不等于新输出质量通过。请检查鸭头持续性、低侧小块、身体、边界、背景及20→21附近观感。视觉结论由用户填写。')
    start=form.index('<script id="atb-review-config" type="application/json">')+len('<script id="atb-review-config" type="application/json">');end=form.index('</script>',start)
    fc=json.loads(form[start:end]);fc['package']=item['package'];form=form[:start]+payload(fc)+form[end:]
    (out/'review.html').write_text(form)
    for suffix,target in [('-regions.csv','regions.csv'),('-temporal.csv','temporal.csv'),('.json','result.json')]:link(out/target,ROOT/(c['record_prefix']+suffix))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>B36：配对推理完成，等待用户视觉复核</h1><a href="B36/">打开原RGB/A/R同步25帧</a> · <a href="review.html">视觉复核表</a> · <a href="regions.csv">区域逐帧CSV</a> · <a href="temporal.csv">时序区域变化CSV</a>')
    save(ROOT/(c['record_prefix']+'-user-review-blank.json'),dict(sample_id='B36',scene_id=c['scene'],status='待用户视觉复核',
        user_visual_review_completed=False,store_key=c['review_storage_key'],observation='',severity='',naturalness='',temporal_quality=''))
    print('MATERIALS',page,flush=True)


def verify(c,out):
    import imageio.v2 as imageio
    import torch
    result=read(out/'analysis.json');reg=read(out/'regions.json');prepared=read(out/'prepared.json')
    inputs=torch.load(out/'inputs.pt',weights_only=True,map_location='cpu');base=inputs['base'];candidate=inputs['rule_b_coc']
    for t in range(25):
        copies=[candidate[g,j] for g,ids in enumerate(prepared['groups']) for j,source in enumerate(ids) if source==t]
        assert all(torch.equal(copies[0],value) for value in copies)
        if t>=21:
            for g,ids in enumerate(prepared['groups']):
                for j,source in enumerate(ids):
                    if source==t:assert torch.equal(candidate[g,j],base['coc'][g,j])
    assert torch.equal(candidate[~inputs['valid']],base['coc'][~inputs['valid']])
    assert (5,0) in [tuple(slot) for slot in prepared['input_rows'][20]['slots']]
    assert float(base['weight'][5,0])==0 and not torch.equal(candidate[5,0],base['coc'][5,0])
    for condition in c['conditions']:
        arr=np.load(out/condition/'float.npy',mmap_mode='r');assert arr.shape==(25,576,1024,3) and arr.dtype==np.float32 and np.isfinite(arr).all()
        pngs=sorted((out/condition/'png').glob('*.png'));assert len(pngs)==25
        for t,path in enumerate(pngs):
            with Image.open(path) as im:
                assert im.mode=='RGB' and im.size==(1024,576)
                assert np.array_equal(np.asarray(im),np.uint8(np.round(np.clip(arr[t],0,1)*255)))
    for row in result['per_frame_region_metrics']:
        mask=np.asarray(Image.open(out/row['region_mask_path']))!=0
        assert row['pixels']==int(mask.sum()) and row['region_display_bbox']==bbox(mask)
        if not mask.any():assert row['A_RGB8_MAE'] is None and row['R_RGB8_MAE'] is None
        assert row['pixels']>=row['gradient_pixels']
    video=imageio.get_reader(out/'B36/RGB-A-R.mp4',format='FFMPEG');meta=video.get_meta_data();count=0
    for frame in video:assert frame.shape==(608,3072,3);count+=1
    video.close();assert count==25 and meta['fps']==20
    selection=read(ROOT/c['selection_config']);sr=read(ROOT/c['selection_record'])
    assert selection['candidate_rule']==sr['config']['candidate_rule']==prepared['selection_rule']
    check=dict(checked_UTC=datetime.now(timezone.utc).isoformat(),frames=25,prediction_PNGs=75,conditions=3,
        output_float_shape=[25,576,1024,3],all_float_finite=True,PNG_roundtrip_equal=True,
        revision1_unchanged=True,all_real_overlap_copies_equal=True,real_zero_weight_source20_slot_modified=True,
        skipped_source21_24_CoC_unchanged=True,padding_unchanged=True,
        region_rows=len(result['per_frame_region_metrics']),temporal_rows=len(result['temporal_region_change']),
        native_repeat_float_bitwise_equal=result['comparisons']['native_repeat']['different_float_bit_patterns']==0,
        video=dict(frames=25,fps=20,size=[3072,608]),inference_started=True,neural_inference_runs=3,
        user_visual_review_completed=False)
    save(out/'program-checks.json',check);print('PROGRAM_CHECKS',json.dumps(check),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['regions','analyze','materials','verify']);args=parser.parse_args()
    cv2.setNumThreads(1);c=read(CFG);out=ROOT/c['output'];globals()[args.stage](c,out)
