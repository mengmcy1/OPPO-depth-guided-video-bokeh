"""CPU response diagnostics and existing-viewer materials for frozen V2.

All supports were saved before predictions. Five visual judgements remain
separate and pending user review. Input similarity is not bokeh quality.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch

from run_subject_coc_v2 import ROOT,CFG,geometry,condition,read,save,now
from analyze_b36_rule_b import energy,local_metrics,csv_write,link
from check_atb_mpi_invariance import digest
from check_atb_mpi_invariance import author_imports
from stat_subject_coc import masks_for


def numeric_delta(a,b,mask):
    if not mask.any():return dict(RGB8_MAE=None,RGB8_max_channel_diff=None,RGB8_different_bytes=None,first_peak_xy=None)
    d=np.abs(b.astype(np.int16)-a.astype(np.int16));maximum=int(d[mask].max());pixel=d.max(axis=2)
    y,x=np.argwhere(mask&(pixel==maximum))[0]
    return dict(RGB8_MAE=float(d[mask].mean()),RGB8_max_channel_diff=maximum,
        RGB8_different_bytes=int(np.count_nonzero(d[mask])),first_peak_xy=[int(x),int(y)] if maximum else None)


def analyze(c,out):
    assert (out/'inference-complete.json').exists();rows=[];temporal=[];b36=[];execution=[];input_summary=[]
    _,_,_,_,Dataset,_=author_imports()
    for sample in c['samples']:
        scene=sample['scene_id'];dest=out/'scenes'/scene;prepared=read(dest/'prepared.json');regions=read(dest/'regions.json')
        rt=read(dest/'V2/runtime.json');events=read(dest/'V2/events.json');complete=read(dest/'V2/complete.json')
        for label_id in ['union']+list(map(str,prepared['label_ids'])):
            selected=[r for r in prepared['input_rows'] if r['label_id']==label_id and r['grid']=='native' and r['region']=='whole']
            valid=[r for r in selected if r['limited_fraction_subject'] is not None]
            input_summary.append(dict(scene_id=scene,sample_id=sample['sample_id'],label_id=label_id,
                limited_native_fraction_clip_median=None if not valid else float(np.median([r['limited_fraction_subject'] for r in valid])),
                limited_native_pixels_min=min(r['limited_native_pixels'] for r in selected),limited_native_pixels_max=max(r['limited_native_pixels'] for r in selected),
                outside_union_condition_changed_min=min(r['resized_union_outside_changed_pixels'] for r in prepared['frame_checks']),
                outside_union_condition_changed_max=max(r['resized_union_outside_changed_pixels'] for r in prepared['frame_checks']),
                native_union_after_max=max(r['native_union_after_max'] for r in prepared['frame_checks']),
                actual_display_union_after_max=max(r['actual_union_after_max'] for r in prepared['frame_checks'])))
        expected={k:v['sha256'] for k,v in prepared['original'].items()};expected['coc']=prepared['candidate_coc']['sha256'];assert rt['input_sha256']==expected
        assert len(events['posterior'])==6 and [x['output_frames'] for x in events['units']]==[8,8,9]
        for unit in events['units']:assert unit['CPU_generator_before']==unit['CPU_generator_after']
        a_rt=read(dest/'A/runtime.json') if (dest/'A/runtime.json').exists() else None
        if a_rt is not None:
            assert a_rt['entry_RNG']==rt['entry_RNG'] and read(dest/'A/complete.json')['end_RNG']==complete['end_RNG']
            a_events=read(dest/'A/events.json')
            for a,b in zip(a_events['posterior'],events['posterior']):
                for k in ['unit','branch','shape','rng_before','rng_after']:assert a[k]==b[k]
                if b['branch']=='RGB':assert a.get('sha256',a.get('sample_sha256'))==b['sha256']
            for a,b in zip(a_events['units'],events['units']):
                for k in ['pixel_values','k','mpi_mask','weight']:assert a['input_sha256'][k]==b['input_sha256'][k]
        else:
            frozen_state=torch.load(ROOT/'outputs/davis-baselines-20260929/any2bokeh/audit'/scene/'rng-after.pt',weights_only=True,map_location='cpu')
            assert complete['end_RNG']=={k:digest(frozen_state[k]) for k in ['cpu','cuda']}
        fa=np.load(dest/'A/float.npy',mmap_mode='r') if (dest/'A/float.npy').exists() else None
        fv=np.load(dest/'V2/float.npy',mmap_mode='r');assert fv.shape==(25,576,1024,3) and np.isfinite(fv).all()
        # Explain strict>1 fractions at the exact cap without changing any
        # reported threshold. Reconstruct pre-Normalize with the SAME committed
        # input helper; do not infer it by inverting Normalize.
        ds=Dataset(str(ROOT/'outputs/davis-baselines-20260929/any2bokeh/audit'/scene/'input.csv'),sample_size=(576,1024))
        bundle=torch.load(dest/'inputs.pt',weights_only=True,map_location='cpu');interior_peaks=[]
        for tt in range(25):
            q,lab,f=geometry(scene,tt);d=q.astype(np.float64)/255;new=d.copy();new[lab!=0]=np.clip(d[lab!=0],f-1/16,f+1/16)
            normalized,actual=condition(ds,new*255,f)
            g,j=next((g,j) for g,ids in enumerate(prepared['groups']) for j,source in enumerate(ids) if source==tt)
            assert digest(normalized)==digest(bundle['candidate_coc'][g,j])
            inside=masks_for(lab,'union')['actual_condition']['interior']
            interior_peaks.append(float(actual[inside].max()) if inside.any() else None)
        for item in input_summary:
            if item['scene_id']==scene:item['actual_interior_after_max']=max(v for v in interior_peaks if v is not None)
        del bundle,ds
        images={};scene_rows=[]
        for t in range(25):
            rgb=cv2.cvtColor(cv2.imread(str(ROOT/'data/processed/davis2016-v2/eval'/scene/'rgb'/f'{t:05d}.jpg')),cv2.COLOR_BGR2RGB)
            rgb=cv2.resize(rgb,(1024,576),interpolation=cv2.INTER_LINEAR)
            im=dict(A=np.asarray(Image.open(dest/'A/png'/f'{t:05d}.png')),V2=np.asarray(Image.open(dest/'V2/png'/f'{t:05d}.png')))
            assert np.array_equal(im['V2'],np.uint8(np.round(np.clip(fv[t],0,1)*255)))
            if scene=='mallard-water':im['R2']=np.asarray(Image.open(ROOT/'outputs/any2bokeh-failure-analysis-20261009/B36-rule-b-R2-20261021/rule_b/png'/f'{t:05d}.png'))
            images[t]=dict(Input=rgb,**im);en={k:energy(v) for k,v in im.items()};ref=energy(rgb)
            for region in regions['frames'][t]['regions']:
                mask=np.asarray(Image.open(dest/region['path']))!=0;gm=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
                row=dict(scene_id=scene,sample_id=sample['sample_id'],frame_id=t,label_id=region['label_id'],region=region['region'],
                    pixels=int(mask.sum()),gradient_pixels=int(gm.sum()),mask_path=str((dest/region['path']).relative_to(out)))
                for label in ['A','V2']+(['R2'] if scene=='mallard-water' else []):
                    row.update({label+'_'+k:v for k,v in local_metrics(im[label],rgb,mask,gm,en[label],ref).items()})
                row.update({'V2_vs_A_'+k:v for k,v in numeric_delta(im['A'],im['V2'],mask).items()})
                row['A_float_available']=fa is not None
                df=None if fa is None or not mask.any() else np.abs(fv[t].astype(np.float64)-fa[t].astype(np.float64))[mask]
                row['V2_vs_A_float_MAE']=None if df is None else float(df.mean());row['V2_vs_A_float_max']=None if df is None else float(df.max())
                row['float_limit']='' if fa is not None else 'Frozen A has PNG only; no surrogate float reconstruction.'
                if scene=='mallard-water' and region['region']=='B36_head_target':
                    denom=row['A_RGB8_MAE']-row['R2_RGB8_MAE'];ed=row['R2_gradient_energy_ratio']-row['A_gradient_energy_ratio']
                    row['MAE_gap_fill_vs_R2']=None if abs(denom)<=1e-12 else (row['A_RGB8_MAE']-row['V2_RGB8_MAE'])/denom
                    row['edge_gap_fill_vs_R2']=None if abs(ed)<=1e-12 else (row['V2_gradient_energy_ratio']-row['A_gradient_energy_ratio'])/ed
                    b36.append(row.copy())
                rows.append(row);scene_rows.append(row)
        for t in range(24):
            for label_id in ['union']+list(map(str,prepared['label_ids'])):
                for name in ['limited','remainder','exterior_ring_2_6','far_background']:
                    support=[]
                    for tt in [t,t+1]:
                        r=next(x for x in regions['frames'][tt]['regions'] if x['label_id']==label_id and x['region']==name)
                        support.append(np.asarray(Image.open(dest/r['path']))!=0)
                    mask=support[0]|support[1];x={k:images[t+1][k].astype(float)-images[t][k].astype(float) for k in ['Input','A','V2']}
                    temporal.append(dict(scene_id=scene,sample_id=sample['sample_id'],transition=f'{t}->{t+1}',label_id=label_id,region=name,
                        pixels=int(mask.sum()),motion_compensated=False,
                        A_region_change=None if not mask.any() else float(np.abs(x['A'])[mask].mean()),
                        V2_region_change=None if not mask.any() else float(np.abs(x['V2'])[mask].mean()),
                        V2_minus_A_region_change=None if not mask.any() else float(np.abs(x['V2']-x['A'])[mask].mean())))
        execution.append(dict(scene_id=scene,frames=25,author_commit=rt['author_commit'],A_source=str((dest/'A').resolve().relative_to(ROOT)),
            A_float_available=fa is not None,input_hashes_verified=True,posterior_calls=6,entry_end_RNG_verified=True,
            fixed_inputs_preserved=True,all25_finite=True,instance_count=len(prepared['label_ids']),native_label_ids=prepared['label_ids']))
        print('ANALYZED',scene,len(scene_rows),'region rows',flush=True)
    summary=[]
    for sample in c['samples']:
        sr=[r for r in rows if r['scene_id']==sample['scene_id']]
        for key in sorted({(r['label_id'],r['region']) for r in sr}):
            group=[r for r in sr if (r['label_id'],r['region'])==key];valid=[r for r in group if r['pixels']]
            n=sum(r['pixels'] for r in valid);peak=max(valid,key=lambda r:(r['V2_vs_A_RGB8_max_channel_diff'],-r['frame_id'])) if valid else None
            summary.append(dict(scene_id=sample['scene_id'],sample_id=sample['sample_id'],label_id=key[0],region=key[1],
                valid_frames=len(valid),total_pixels=n,pixel_weighted_V2_vs_A_RGB8_MAE=None if not n else sum(r['V2_vs_A_RGB8_MAE']*r['pixels'] for r in valid)/n,
                clip_max_RGB8_channel_diff=None if peak is None else peak['V2_vs_A_RGB8_max_channel_diff'],
                peak_frame=None if peak is None else peak['frame_id'],first_peak_xy=None if peak is None else peak['V2_vs_A_first_peak_xy']))
    prefix=c['record_prefix'];csv_write(ROOT/(prefix+'-regions.csv'),rows);csv_write(ROOT/(prefix+'-summary.csv'),summary);csv_write(ROOT/(prefix+'-temporal.csv'),temporal);csv_write(ROOT/(prefix+'-B36-R2.csv'),b36)
    result=dict(status='统计与配对推理完成，等待用户视觉复核',config=c,created_UTC=now(),freeze=read(out/'freeze.json'),launch=read(out/'launch.json'),
        inference=read(out/'inference-complete.json'),A_reproduction=read(out/'A-reproduction.json'),execution=execution,
        region_summary=summary,input_summary=input_summary,B36_R2_input_relative_comparison=b36,
        B36_gap_description='Unclipped input-relative MAE/gradient gap filling against R2, not a perceptual-quality recovery percentage.',
        cap_precision_note='All strict>1 fractions remain unmodified. Native mathematical cap is1. Actual interior can have FP32 excess AND larger resize mixing because a3-pixel erosion of nearest-projected labels does not ensure all native filter support is inside union. actual_interior_after_max is reconstructed BEFORE Normalize and checked against input bytes; do not relabel thresholds or equate either grid to the other.',
        user_visual_review_completed=False,limitations=['Only6 exploration sources; no held-out videos or new neural metrics.',
            'DAVIS GT union provides extra spatial condition permission beyond scalar focus, not predicted-mask deployment.',
            'V2 is subject clear-protection, not depth correction or Ours. Multi-ID union can flatten genuine depth.',
            'Closer to input RGB may just mean less defocus. Five human criteria remain separately pending.',
            'Temporal regions are uncompensated pair unions, not flicker. Shared far-background repeated across IDs is not independent evidence.'])
    save(ROOT/(prefix+'.json'),result);save(out/'analysis.json',result)


def crop_fit(im,box,size=(384,240)):
    x,y,w,h=box;tile=im.crop((max(0,x-8),max(0,y-8),min(1024,x+w+8),min(576,y+h+8)))
    ratio=min(size[0]/tile.width,size[1]/tile.height);tile=tile.resize((max(1,round(tile.width*ratio)),max(1,round(tile.height*ratio))),Image.Resampling.NEAREST)
    canvas=Image.new('RGB',size,'white');canvas.paste(tile,((size[0]-tile.width)//2,(size[1]-tile.height)//2));return canvas


def materials(c,out):
    import imageio_ffmpeg
    sys.path.insert(0,str(ROOT/'src'))
    from review.any2bokeh_pages import viewer_fragment,render_review_form,payload
    items=[];keys=c['keyframes'];ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for sample in c['samples']:
        scene=sample['scene_id'];dest=out/'scenes'/scene;reg=read(dest/'regions.json');page=out/sample['sample_id'];page.mkdir()
        names=['Input_RGB','A']+(['R2'] if scene=='mallard-water' else [])+['V2'];n=len(names)
        full=Image.new('RGB',(n*384,len(keys)*240),'white');close=Image.new('RGB',(n*384,len(keys)*264),'white');video=dest/'video-frames';video.mkdir()
        for t in range(25):
            rgb=cv2.cvtColor(cv2.imread(str(ROOT/'data/processed/davis2016-v2/eval'/scene/'rgb'/f'{t:05d}.jpg')),cv2.COLOR_BGR2RGB);rgb=cv2.resize(rgb,(1024,576),interpolation=cv2.INTER_LINEAR)
            rgbpath=page/'originals/Input_RGB'/f'{t:05d}.png';rgbpath.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(rgbpath)
            paths=dict(Input_RGB=rgbpath,A=dest/'A/png'/f'{t:05d}.png',V2=dest/'V2/png'/f'{t:05d}.png')
            if 'R2' in names:paths['R2']=ROOT/'outputs/any2bokeh-failure-analysis-20261009/B36-rule-b-R2-20261021/rule_b/png'/f'{t:05d}.png'
            movie=Image.new('RGB',(n*1024,608),'#12345a');draw=ImageDraw.Draw(movie)
            for j,name in enumerate(names):
                if name!='Input_RGB':link(page/'originals'/name/f'{t:05d}.png',paths[name])
                with Image.open(paths[name]) as im:
                    movie.paste(im,(j*1024,32));draw.text((j*1024+8,8),f'{sample["sample_id"]} {scene} | {name} | f{t:02d} | '+('GT-union CoC clamp tau1' if name=='V2' else 'assisted diagnosis' if name=='R2' else ''),fill='white')
                    if t in keys:
                        i=keys.index(t);full.paste(im.resize((384,216),Image.Resampling.BILINEAR),(j*384,i*240+24));ImageDraw.Draw(full).text((j*384+5,i*240+4),f'{name} f{t}',fill='black')
                        close.paste(crop_fit(im,reg['frames'][t]['subject_bbox']),(j*384,i*264+24));ImageDraw.Draw(close).text((j*384+5,i*264+4),f'{name} f{t} subject-input bbox',fill='black')
            movie.save(video/f'{t:05d}.png')
        full.save(page/'contact-full.png');close.save(page/'contact-subject.png')
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-framerate','20','-i',str(video/'%05d.png'),'-frames:v','25','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','2',str(page/'comparison.mp4')],check=True)
        tracks=[dict(id='subject',name='原标注主体union框（输入定义）',xywh_by_frame={str(t):reg['frames'][t]['subject_bbox'] for t in range(25)})]
        for name in ['limited','remainder','exterior_ring_2_6']+(['B36_head_target'] if scene=='mallard-water' else []):
            tracks.append(dict(id=name,name=name+'：输入定义，空支持不插补',xywh_by_frame={str(t):next(r['display_bbox'] for r in reg['frames'][t]['regions'] if r['label_id']=='union' and r['region']==name) for t in range(25)}))
        item=dict(sample_id=sample['sample_id'],scene_id=scene,dataset='DAVIS',source_group='DAVIS:'+scene,split='exploration',package='subject-coc-clamp-v2',review_url='../review.html',initial_frame=0,frame_jumps=keys,tracked_rois=tracks,panels=[dict(label=name,prefix=f'originals/{name}/') for name in names])
        items.append(item)
        text='<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>主体清晰保护V2 '+scene+'</title><style>body{font-family:system-ui;margin:24px;background:#f6f9ff}video,img{max-width:100%}#atb-viewer .atb-panel-grid{grid-template-columns:repeat('+str(n)+',minmax(0,1fr))}</style>'
        text+='<h1>'+sample['sample_id']+' / '+scene+'：主体内离焦约束V2</h1><p>诊断用固定tau=1，原GT非零ID union逐像素约束CoC；mask权限从仅求焦点标量扩大到逐像素条件，不是预测mask部署、深度修正或Ours。全部25帧同一规则，原focus/K、MPI和模型流程不变。该来源'+str(sample['instance_count'])+'个标注ID，union限幅可能压平真实纵深。</p>'
        text+='<p>请分别判断主体是否更清楚、整体景深是否自然、背景虚化是否保持、主体边缘过渡、跨帧是否稳定。主体更接近原RGB可能只是虚化变少，不能单独证明更好。所有视觉判断待用户填写；统计选样仍属探索材料。</p><p>限幅发生在原网格，随后原生resize/antialias可能令显示边界条件超过1并让union外条件变化，不能宣称输出背景不变；逐帧/ID前后分布及变化像素见<a href="../inputs.csv">输入CSV</a>。</p><p>视频25帧/20fps，CRF18/yuv420p，仅派生观看；0.25×不插帧。准确定位用原PNG1024×576。接触图主体框仅依据输入union，统一最近邻缩放、不锐化；100%保留源像素。24→0是循环，不是连续transition。</p><p><a href="../">六来源索引与双口径排名</a> · <a href="../review.html">五项独立反馈</a> · <a href="../regions.csv">逐帧区域CSV</a> · <a href="../summary.csv">每来源/ID区域汇总</a></p><video controls preload="metadata" src="comparison.mp4"></video>'
        text+=viewer_fragment(item)+'<h2>主体局部：固定0/6/12/18/24</h2><a href="contact-subject.png"><img src="contact-subject.png"></a><details><summary>完整画面接触图</summary><a href="contact-full.png"><img src="contact-full.png"></a></details>'
        (page/'index.html').write_text(text)
    save(out/'viewer-items.json',items)
    form=render_review_form(items,package='B1',store_key=c['review_storage_key'],title='主体清晰保护V2：五项独立用户复核',notice='六段探索样本，GT union辅助；主体清晰与整体景深自然度分开，不能合成一个总评。更接近输入RGB也可能只是虚化变少。')
    marker='<script id="atb-review-config" type="application/json">';start=form.index(marker)+len(marker);end=form.index('</script>',start);fc=json.loads(form[start:end])
    fc['package']='subject-coc-clamp-v2';fc['visual']=c['review_fields'];fc['fields']=['frame_id_or_transition_id','ROI_xywh','actual_view_mode']+c['review_fields']+['notes','actual_playback_rate'];fc['labels']=['帧号或transition','ROI x,y,w,h（显示网格）','实际观看方式','主体是否更清楚','整体景深是否自然，是否不合理压平主体纵深','背景虚化是否保持','主体边缘清晰—模糊过渡是否突变或光晕','跨帧是否稳定','补充说明','实际播放速度']
    (out/'review.html').write_text(form[:start]+payload(fc)+form[end:])
    save(ROOT/(c['record_prefix']+'-user-review-blank.json'),[dict(sample_id=i['sample_id'],scene_id=i['scene_id'],source_group=i['source_group'],split='exploration',**{k:'待用户复核' for k in c['review_fields']}) for i in items])
    stats=read(ROOT/'03_实验记录/records/any2bokeh-subject-coc-statistics-20261021.json');native={r['scene_id']:r for r in stats['native_ranking']}
    index='<!doctype html><meta charset="utf-8"><title>主体清晰保护V2：六来源探索复核</title><style>body{font-family:system-ui;margin:24px;background:#f6f9ff}td,th{border:1px solid #abc;padding:6px}table{border-collapse:collapse}</style><h1>固定tau=1主体内离焦约束：六来源探索</h1><p>统计与配对推理完成，等待用户视觉复核。使用GT union逐像素约束条件，不是深度错误判定、预测mask部署或Ours；多实例可能压平真实纵深。无来源隔离复核视频访问，无相关/显著性分析。</p><ul>'
    for i in items:index+='<li><a href="'+i['sample_id']+'/">'+i['sample_id']+' '+i['scene_id']+'</a></li>'
    index+='</ul><a href="review.html">五项独立反馈</a> · <a href="summary.csv">区域汇总</a> · <a href="B36-R2.csv">B36 A/R2/V2及输入相对差距比例</a><p>B36比例仅为相对RGB的MAE/梯度数值差距填补，不是感知恢复百分比。完整逐帧数据与空区域原因均保留。</p><h2>35段探索输入排名：实际条件为主，原网格独立对照</h2><p>原VDA q/255不是米制深度。内部为3显示像素腐蚀；实际条件abs后作者bilinear antialias，再取Normalize前CoC×K；原网格直接K|q/255−focus|。片段值为逐帧占比中位数。正常与问题样本未按预期排序，未换指标或阈值。</p><table><tr><th>实际条件排名</th><th>来源/标注</th><th>实例数</th><th>union内部&gt;1片段中位数</th><th>最高帧/占比</th><th>原网格排名/占比</th></tr>'
    for r in stats['primary_ranking']:
        nr=native[r['scene_id']];index+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in [r['rank'],r['scene_id']+' '+r['user_report'],r['instance_count'],f'{r["union_interior_gt1_clip_median"]:.3%}',f'{r["highest_frame"]} / {r["highest_fraction"]:.3%}',f'{nr["rank"]} / {nr["union_interior_gt1_clip_median"]:.3%}'])+'</tr>'
    index+='</table><p>统计选样train：全榜第4、剩余合格池第1，4个ID；清单先于输出查看保存。B1/B2、固定8段、专项页排除依据见selection.json。原网格例外kite-walk/soccerball1600×900，其余1920×1080；不新增缩放。</p>'
    (out/'index.html').write_text(index)
    for suffix,name in [('-regions.csv','regions.csv'),('-summary.csv','summary.csv'),('-temporal.csv','temporal.csv'),('-B36-R2.csv','B36-R2.csv'),('-inputs.csv','inputs.csv'),('.json','result.json')]:link(out/name,ROOT/(c['record_prefix']+suffix))
    link(out/'selection.json',ROOT/c['selection_record'])
    print('MATERIALS_ALL_SIX_COMPLETE',flush=True)


def verify(c,out):
    import imageio.v2 as imageio
    checked=0
    for sample in c['samples']:
        scene=sample['scene_id'];dest=out/'scenes'/scene;prep=read(dest/'prepared.json');bundle=torch.load(dest/'inputs.pt',weights_only=True,map_location='cpu')
        for key in ['pixel_values','k','mpi_mask','weight']:assert digest(bundle['base'][key])==prep['original'][key]['sha256']
        assert torch.equal(bundle['candidate_coc'][~bundle['valid']],bundle['base']['coc'][~bundle['valid']])
        value=np.load(dest/'V2/float.npy',mmap_mode='r');assert value.shape==(25,576,1024,3) and np.isfinite(value).all()
        assert len(list((dest/'V2/png').glob('*.png')))==25
        for t in range(25):
            with Image.open(dest/'V2/png'/f'{t:05d}.png') as im:assert im.mode=='RGB' and im.size==(1024,576) and np.array_equal(np.asarray(im),np.uint8(np.round(np.clip(value[t],0,1)*255)));checked+=1
        reader=imageio.get_reader(out/sample['sample_id']/'comparison.mp4',format='FFMPEG');meta=reader.get_meta_data();n=0
        width=4096 if scene=='mallard-water' else 3072
        for frame in reader:assert frame.shape==(608,width,3);n+=1
        reader.close();assert n==25 and meta['fps']==20
        assert all(f['native_union_after_max']<=1+1e-12 for f in prep['frame_checks'])
        del bundle
    check=dict(UTC=now(),scenes=6,V2_frames=150,new_prediction_PNGs_verified=checked,all_finite=True,
        native_cap_tau1_verified=True,padding_and_fixed_inputs_verified=True,config_frozen_before_GPU=read(out/'freeze.json'),
        all_videos_frames=25,fps=20,all_visual_fields='待用户复核',user_visual_review_completed=False)
    save(out/'program-checks.json',check);print('PROGRAM_CHECKS_PASS',flush=True)


def input_mix_audit(c,out):
    """Describe each source's maximal display-interior input, not a new rule."""
    _,_,_,_,Dataset,_=author_imports();audit=[]
    for sample in c['samples']:
        scene=sample['scene_id'];ds=Dataset(str(ROOT/'outputs/davis-baselines-20260929/any2bokeh/audit'/scene/'input.csv'),sample_size=(576,1024));best=None
        for t in range(25):
            q,label,f=geometry(scene,t);d=q.astype(np.float64)/255;clamped=d.copy();clamped[label!=0]=np.clip(d[label!=0],f-1/16,f+1/16)
            _,actual=condition(ds,clamped*255,f);inside=masks_for(label,'union')['actual_condition']['interior']
            if not inside.any():continue
            peak=float(actual[inside].max());y,x=np.argwhere(inside&(actual==peak))[0]
            if best is None or peak>best['peak']:best=dict(peak=peak,frame=t,x=int(x),y=int(y))
        t=best['frame'];q,label,f=geometry(scene,t);union=label!=0;d=q.astype(np.float64)/255;d[union]=np.clip(d[union],f-1/16,f+1/16)
        raw=(torch.from_numpy((d*255).astype(np.float32))-f*255).abs()/255
        def resized(a):
            value=a.repeat(3,1,1)
            for transform in ds.pixel_transforms.transforms[:-1]:value=transform(value)
            return value[0].numpy()
        inside_contribution=resized(raw*torch.from_numpy(union))*16
        outside_contribution=resized(raw*torch.from_numpy(~union))*16
        outside_weight=resized(torch.from_numpy((~union).astype(np.float32)))
        y,x=best['y'],best['x'];a=float(inside_contribution[y,x]);b=float(outside_contribution[y,x])
        assert abs(a+b-best['peak'])<2e-5
        audit.append(dict(scene_id=scene,sample_id=sample['sample_id'],frame_id=t,display_xy=[x,y],
            actual_interior_peak=best['peak'],native_FP32_inside_max=float((raw[torch.from_numpy(union)]*16).max()),
            outside_union_filter_weight=float(outside_weight[y,x]),inside_union_contribution=a,outside_union_contribution=b,
            linear_sum_error=abs(a+b-best['peak']),
            scope='Post-hoc input-only linear-resize contribution at per-source maximal interior point. No ROI/ranking/threshold/condition/output change; not an anatomical or true-depth verdict.'))
    csv_write(ROOT/(c['record_prefix']+'-input-mixing.csv'),audit)
    for path in [out/'analysis.json',ROOT/(c['record_prefix']+'.json')]:
        r=read(path);r['input_mixing_audit']=audit
        r['cap_precision_note']='Native mathematical cap1 holds. Strict>1 fractions retained. Nearest-projected display interior can still include native outside-union filter support; input-only linear contribution audit documents maximal points. Small FP32 excess is separately visible in native_FP32_inside_max. No grid interchange or threshold tuning.'
        path.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    link(out/'input-mixing.csv',ROOT/(c['record_prefix']+'-input-mixing.csv'))
    p=out/'index.html';p.write_text(p.read_text()+'<h2>显示内部与原网格滤波支持：CPU输入补充</h2><p>主体原网格数学上限1成立，但最近邻显示mask腐蚀3像素后，也未必排除原网格微小空隙/边缘对resize的贡献。逐来源最大点使用原缩放算子拆分原union内外贡献；不自动认定标注或深度错误，不改变任何排名/阈值/输出。<a href="input-mixing.csv">标量证据CSV</a></p>')
    print('INPUT_MIX_AUDIT',json.dumps(audit,ensure_ascii=False),flush=True)


def report(c,out):
    """Append static result tables; no new images, routes or review logic."""
    result=read(out/'analysis.json');page=out/'index.html';text=page.read_text();assert 'id="numeric-results"' not in text
    extra='<section id="numeric-results"><h2>V2−A：各来源共同union区域</h2><p>每格为RGB8平均绝对差／最大通道差；平均值按25帧有效区域像素加权，0–255单位。各ID另列在CSV，空支持保留不适用。数值变化不等于视觉改善。</p><table><tr><th>来源</th><th>ID数</th><th>被限幅主体</th><th>主体其余</th><th>外环2–6px</th><th>远背景&gt;32px</th></tr>'
    for sample in c['samples']:
        group={r['region']:r for r in result['region_summary'] if r['scene_id']==sample['scene_id'] and r['label_id']=='union'}
        cells=[]
        for name in ['limited','remainder','exterior_ring_2_6','far_background']:
            r=group[name];cells.append('不适用' if not r['valid_frames'] else f'{r["pixel_weighted_V2_vs_A_RGB8_MAE"]:.4f} / {r["clip_max_RGB8_channel_diff"]}')
        extra+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in [sample['sample_id']+' '+sample['scene_id'],sample['instance_count']]+cells)+'</tr>'
    reproduction=result['A_reproduction'];different=sum(x['RGB8_different_bytes'] for x in reproduction['per_frame']);maximum=max(x['RGB8_max_abs_diff'] for x in reproduction['per_frame']);mean=float(np.mean([x['RGB8_MAE'] for x in reproduction['per_frame']]))
    extra+='</table><p>B18原版复现：与原冻结PNG不同字节'+str(different)+'，最大通道差'+str(maximum)+'，平均绝对差'+f'{mean:.8f}'+'；已采用当前环境A配对，B18复现复用、B36已有A复用、另4来源重跑A。原冻结输出保留，差异未计入V2收益。</p>'
    extra+='<h2>B36共同头部区域：A／R2／V2</h2><p>以下比例只量化相对输入RGB的统计差距，不是清晰度或感知质量恢复百分比；全25帧见<a href="B36-R2.csv">CSV</a>。</p><table><tr><th>帧</th><th>A对RGB MAE</th><th>R2对RGB MAE</th><th>V2对RGB MAE</th><th>A/R2/V2梯度比</th><th>MAE差距填补</th></tr>'
    for row in result['B36_R2_input_relative_comparison']:
        if row['frame_id'] not in c['keyframes']:continue
        v=[row['frame_id']]+[f'{row[k+"_RGB8_MAE"]:.4f}' for k in ['A','R2','V2']]+[' / '.join(f'{row[k+"_gradient_energy_ratio"]:.4f}' for k in ['A','R2','V2']),f'{row["MAE_gap_fill_vs_R2"]:.2%}']
        extra+='<tr>'+''.join('<td>'+str(x)+'</td>' for x in v)+'</tr>'
    ratios=result['B36_R2_input_relative_comparison'];m=float(np.median([r['MAE_gap_fill_vs_R2'] for r in ratios]));e=float(np.median([r['edge_gap_fill_vs_R2'] for r in ratios]));extra+='</table><p>25帧逐帧MAE差距填补比例中位数'+f'{m:.2%}'+'，梯度差距填补比例中位数'+f'{e:.2%}'+'。不据此代填视觉通过。</p></section>'
    page.write_text(text+extra)
    for p in [out/'analysis.json',ROOT/(c['record_prefix']+'.json')]:
        r=read(p);r['B36_R2_gap_clip_summary']=dict(method='median of25 per-frame un-clipped input-relative gap ratios',MAE=m,gradient=e,perceptual_quality_fraction=False)
        p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('STATIC_RESULT_TABLES_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['analyze','materials','verify','input_mix_audit','report']);a=p.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(4);c=read(CFG);globals()[a.stage](c,ROOT/c['output'])
