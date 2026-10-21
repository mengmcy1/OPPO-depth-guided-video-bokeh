"""CPU response analysis and four-panel review for one assisted B36 R2.

No visual verdicts: input-relative metrics and uncompensated region changes
are diagnostics. All conditions use the same pre-inference region masks.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from run_b36_rule_b_r2 import ROOT, CFG, read, save, bit_equal
from analyze_b36_rule_b import csv_write, energy, local_metrics, bbox, link
from check_atb_mpi_invariance import digest


def compare_runs(a_path,b_path,label):
    a=np.load(a_path/'float.npy',mmap_mode='r');b=np.load(b_path/'float.npy',mmap_mode='r')
    assert a.shape==b.shape==(25,576,1024,3) and a.dtype==b.dtype==np.float32
    rows=[]
    for t in range(25):
        assert np.isfinite(a[t]).all() and np.isfinite(b[t]).all()
        d=np.abs(b[t].astype(np.float64)-a[t].astype(np.float64))
        pa=np.asarray(Image.open(a_path/'png'/f'{t:05d}.png'));pb=np.asarray(Image.open(b_path/'png'/f'{t:05d}.png'))
        assert np.array_equal(pb,np.uint8(np.round(np.clip(b[t],0,1)*255)))
        byte=np.abs(pb.astype(np.int16)-pa.astype(np.int16))
        rows.append(dict(frame_id=t,float_max_abs_diff=float(d.max()),float_mean_abs_diff=float(d.mean()),
            different_float_elements=int(np.count_nonzero(a[t]!=b[t])),
            different_float_bit_patterns=int(np.count_nonzero(a[t].view(np.uint32)!=b[t].view(np.uint32))),
            RGB8_different_bytes=int(np.count_nonzero(pa!=pb)),RGB8_MAE=float(byte.mean()),RGB8_max_abs_diff=int(byte.max())))
    return dict(comparison=label,float_max_abs_diff=max(r['float_max_abs_diff'] for r in rows),
        float_mean_abs_diff=float(np.mean([r['float_mean_abs_diff'] for r in rows])),
        different_float_elements=sum(r['different_float_elements'] for r in rows),
        different_float_bit_patterns=sum(r['different_float_bit_patterns'] for r in rows),
        RGB8_different_bytes=sum(r['RGB8_different_bytes'] for r in rows),
        affected_float_frames=[r['frame_id'] for r in rows if r['different_float_bit_patterns']],
        affected_RGB8_frames=[r['frame_id'] for r in rows if r['RGB8_different_bytes']],per_frame=rows)


def analyze(c,out):
    prep=read(out/'prepared.json');reg=read(out/'regions.json')
    assert reg['prepared_before_R2']
    runs={label:dict(runtime=read(out/folder/'runtime.json'),events=read(out/folder/'events.json'),complete=read(out/folder/'complete.json'))
          for label,folder in [('A','native'),('A_repeat','native_repeat'),('R1','revision1'),('R2','rule_b')]}
    base=runs['A'];r2=runs['R2'];r1=runs['R1']
    expected_inputs={k:v['sha256'] for k,v in prep['original'].items()}
    expected_inputs['coc']=prep['rule_b_coc']['sha256']
    assert r2['runtime']['input_sha256']==expected_inputs
    for k in ['python','torch','cuda','device','author_commit','imports','allocator_environment','entry_RNG','RNG_source','initial_encoder_cache_absent','initial_latent_cache']:
        assert r2['runtime'][k]==base['runtime'][k],k
    assert r2['complete']['end_RNG']==base['complete']['end_RNG']
    assert [x['output_frames'] for x in r2['events']['units']]==[8,8,9]
    assert len(r2['events']['posterior'])==6
    for actual,old in zip(r2['events']['posterior'],base['events']['posterior']):
        for k in ['unit','branch','shape','rng_before','rng_after','generator']:assert actual[k]==old[k]
        if actual['branch']=='RGB':assert actual['sample_sha256']==old['sample_sha256']
    for i,(unit,old) in enumerate(zip(r2['events']['units'],r1['events']['units'])):
        for k in ['groups','rng_before','rng_after','CPU_generator_before','CPU_generator_after']:assert unit[k]==old[k]
        assert unit['CPU_generator_before']==unit['CPU_generator_after']
        for k in ['pixel_values','k','mpi_mask','weight']:assert unit['input_sha256'][k]==old['input_sha256'][k]
        if i<2:
            assert unit['input_sha256']['coc']==old['input_sha256']['coc']
            assert unit['latent_cache_in']==old['latent_cache_in'] and unit['latent_cache_out']==old['latent_cache_out']
    comps=dict(repeat_vs_A=compare_runs(out/'native',out/'native_repeat','A-repeat vs A'),
        R1_vs_A=compare_runs(out/'native',out/'revision1','R1 vs A'),
        R2_vs_A=compare_runs(out/'native',out/'rule_b','R2 vs A'),
        R2_vs_R1=compare_runs(out/'revision1',out/'rule_b','R2 vs R1'))
    assert comps['repeat_vs_A']['different_float_bit_patterns']==0
    priorc=read(ROOT/c['prior_config']);prior_result=read(ROOT/(priorc['record_prefix']+'.json'))
    for old_key,new_key in [('native_repeat','repeat_vs_A'),('rule_b','R1_vs_A')]:
        for key in ['float_max_abs_diff','float_mean_abs_diff','different_float_elements','different_float_bit_patterns','RGB8_different_bytes']:
            assert comps[new_key][key]==prior_result['comparisons'][old_key][key]
    folders=dict(A='native',A_repeat='native_repeat',R1='revision1',R2='rule_b')
    floats={k:np.load(out/v/'float.npy',mmap_mode='r') for k,v in folders.items()}
    selection=read(ROOT/priorc['selection_config']);rgbsource=ROOT/selection['output']/'B36/originals/Input_RGB'
    rows=[];images={}
    for t in range(25):
        rgb=np.asarray(Image.open(rgbsource/f'{t:05d}.png'))
        imgs={k:np.asarray(Image.open(out/v/'png'/f'{t:05d}.png')) for k,v in folders.items()}
        images[t]=dict(Input=rgb,**imgs)
        energies={k:energy(x) for k,x in imgs.items()};reference=energy(rgb)
        delta=np.abs(floats['R2'][t].astype(np.float64)-floats['R1'][t].astype(np.float64))
        noise=np.abs(floats['A_repeat'][t].astype(np.float64)-floats['A'][t].astype(np.float64))
        for region in reg['frames'][t]['regions']:
            mask=np.asarray(Image.open(out/region['path']))!=0
            gm=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
            row=dict(frame_id=t,region=region['region'],region_mask_path=region['path'],region_display_bbox=region['display_bbox'],
                pixels=int(mask.sum()),gradient_pixels=int(gm.sum()),R2_vs_R1_source_CoC_changed=t>=21)
            for label in ['A','R1','R2']:
                row.update({label+'_'+k:v for k,v in local_metrics(imgs[label],rgb,mask,gm,energies[label],reference).items()})
            for name,diff,p1,p0 in [('R2_vs_R1',delta,imgs['R2'],imgs['R1']),('repeat_vs_A',noise,imgs['A_repeat'],imgs['A'])]:
                row.update({name+'_float_max':float(diff[mask].max()) if mask.any() else None,
                    name+'_float_MAE':float(diff[mask].mean()) if mask.any() else None,
                    name+'_RGB8_MAE':float(np.abs(p1.astype(float)-p0.astype(float))[mask].mean()) if mask.any() else None,
                    name+'_RGB8_different_bytes':int(np.count_nonzero(p1[mask]!=p0[mask])) if mask.any() else None})
            rows.append(row)
    temporal=[]
    for t in range(18,24):
        for region in ['target_head','head_context']:
            masks=[np.asarray(Image.open(out/next(r['path'] for r in reg['frames'][tt]['regions'] if r['region']==region)))!=0 for tt in [t,t+1]]
            mask=masks[0]|masks[1]
            changes={k:images[t+1][k].astype(float)-images[t][k].astype(float) for k in ['Input','A','A_repeat','R1','R2']}
            row=dict(transition=f'{t}->{t+1}',region=region+'_pair_union',pixels=int(mask.sum()),motion_compensated=False)
            for label in ['A','R1','R2']:
                row[label+'_temporal_RGB8_MAE']=float(np.abs(changes[label])[mask].mean()) if mask.any() else None
                row[label+'_input_relative_change_error']=float(np.abs(changes[label]-changes['Input'])[mask].mean()) if mask.any() else None
            row['R2_vs_R1_temporal_response_MAE']=float(np.abs(changes['R2']-changes['R1'])[mask].mean()) if mask.any() else None
            row['repeat_vs_A_temporal_response_MAE']=float(np.abs(changes['A_repeat']-changes['A'])[mask].mean()) if mask.any() else None
            temporal.append(row)
    result=dict(config=c,status='人工辅助末段R2推理及CPU响应统计完成，等待用户视觉复核',
        prepared=prep,launch=read(out/'launch.json'),runs=runs,comparisons=comps,
        execution_checks=dict(A_R1_provenance_verified=True,only_one_new_inference=True,
            environment_imports_allocator_same=True,runtime_inputs_match_prepared_bytes=True,entry_end_CPU_CUDA_RNG_equal=True,
            posterior_order_shapes_RNG_equal=True,RGB_posterior_samples_bitwise_equal=True,
            per_call_CPU_generator_unchanged=True,initial_cache_same=True,
            first_two_units_R1_inputs_and_latent_caches_bitwise_equal=True,fixed_GPU_inputs_equal=True),
        per_frame_region_metrics=rows,temporal_region_change=temporal,
        user_visual_review_completed=False,visual_findings='待用户复核',
        limits=['B36 one exploration source only; manually assisted diagnostic, not automatic rule success.',
            'Input-defined target support shared by every condition; head_context is a separate mixed rectangle.',
            'DAVIS has no paired bokeh GT; RGB8 MAE and gradient response are not bokeh quality scores.',
            'No motion compensation; temporal numbers are region change, not flicker.',
            'Only21-24 CoC changed vs R1; all25 output frames compared without assuming earlier output invariance.'])
    csv_write(ROOT/(c['record_prefix']+'-regions.csv'),rows)
    csv_write(ROOT/(c['record_prefix']+'-temporal.csv'),temporal)
    csv_write(ROOT/(c['record_prefix']+'-response.csv'),comps['R2_vs_R1']['per_frame'])
    save(ROOT/(c['record_prefix']+'.json'),result);save(out/'analysis.json',result)
    save(ROOT/(c['record_prefix']+'-regions.json'),reg)
    for label,co in comps.items():print(label,{k:v for k,v in co.items() if k!='per_frame'},flush=True)


def materials(c,out):
    import imageio_ffmpeg
    sys.path.insert(0,str(ROOT/'src'))
    from review.any2bokeh_pages import viewer_fragment,render_review_form,payload
    result=read(out/'analysis.json');reg=read(out/'regions.json');priorc=read(ROOT/c['prior_config'])
    selection=read(ROOT/priorc['selection_config']);oldrgb=ROOT/selection['output']/'B36/originals/Input_RGB'
    page=out/'B36';assert not page.exists();page.mkdir()
    panels=['Input_RGB','A','R1','R2'];paths=[oldrgb,out/'native/png',out/'revision1/png',out/'rule_b/png']
    frames=out/'video-frames';frames.mkdir()
    full=Image.new('RGB',(4*384,len(c['keyframes'])*240),'white')
    close=Image.new('RGB',(4*256,len(c['keyframes'])*344),'white')
    for t in range(25):
        movie=Image.new('RGB',(4096,608),'#12345a');draw=ImageDraw.Draw(movie)
        x,y,w,h=reg['frames'][t]['head_context_xywh']
        for j,(label,source) in enumerate(zip(panels,paths)):
            link(page/'originals'/label/f'{t:05d}.png',source/f'{t:05d}.png')
            with Image.open(source/f'{t:05d}.png') as im:
                movie.paste(im,(j*1024,32));draw.text((j*1024+8,8),f'{label} | frame{t:02d} | '+('human-assisted tail diagnosis' if label=='R2' else 'original' if label=='A' else 'revision1' if label=='R1' else 'input RGB'),fill='white')
                if t in c['keyframes']:
                    row=c['keyframes'].index(t);full.paste(im.resize((384,216),Image.Resampling.BILINEAR),(j*384,row*240+24))
                    ImageDraw.Draw(full).text((j*384+8,row*240+5),f'{label} f{t:02d}',fill='black')
                    close.paste(im.crop((x,y,x+w,y+h)).resize((256,320),Image.Resampling.NEAREST),(j*256,row*344+24))
                    ImageDraw.Draw(close).text((j*256+8,row*344+5),f'{label} f{t:02d}',fill='black')
        movie.save(frames/f'{t:05d}.png')
    full.save(page/'contact-full.png');close.save(page/'contact-head.png')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-framerate','20','-i',str(frames/'%05d.png'),
        '-frames:v','25','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','2',str(page/'RGB-A-R1-R2.mp4')],check=True)
    tracks=[dict(id=name,name=title,xywh_by_frame={str(t):next(r['display_bbox'] for r in reg['frames'][t]['regions'] if r['region']==name) for t in range(25)})
        for name,title in [('target_head','共同输入目标区域（各组相同）'),('head_context','头部上下文（含颈部/背景）'),('neck','同帧颈部参考'),('body_non_target','身体核心非目标'),('water_near','邻近水面'),('water_far','远处水面')]]
    item=dict(sample_id='B36',dataset='DAVIS',scene_id=c['scene'],source_group='DAVIS:mallard-water',split='exploration',
        package='B36-rule-b-R2-output',review_url='../review.html',initial_frame=20,frame_jumps=c['keyframes'],
        panels=[dict(label=x,prefix=f'originals/{x}/') for x in panels],tracked_rois=tracks)
    text='''<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>B36人工辅助末段R2：四组同步对照</title>
<style>body{font-family:system-ui;background:#f6f9ff;margin:24px}video,img{max-width:100%}td,th{border:1px solid #aac0d6;padding:5px}table{border-collapse:collapse}#atb-viewer .atb-panel-grid{grid-template-columns:repeat(4,minmax(0,1fr))}</style>
<h1>B36 人工辅助末段诊断：原RGB／A／R1／R2</h1>
<p>A原版、R1及A-repeat经来源核对后复用。本轮只有一次新R2推理。源0–20 CoC与R1逐位一致；仅21–24使用用户确认的同帧颈部参考和粉色局部目标，按作者原顺序构造CoC。原MPI/RGB/focus/K、权重精度、8/8/9窗口、padding、缓存及入口RNG不变。人工辅助对照不能称原规则自动成功。</p>
<p>请先以1×观看整段，再以0.25×及PNG逐帧重点检查18–24：末段鸭头是否持续清楚、20→21是否自然、边界有无异常、前段和背景是否受到影响。新输出的清晰度、自然度、严重程度均等待用户判断。只改末段条件仍可能影响较早输出，完整25帧均已比较。</p>
<p>同帧四面板坐标一致，ROI预设与手动框选同步；头部目标与含颈部/背景的上下文框分开。视频25帧/20fps、1.25秒，libx264 CRF18/yuv420p，慢放只改播放速度，不插帧。PNG1024×576不修图；裁剪最近邻，100%为原输出网格。24→0循环不是连续transition。</p>
<p><a href="../review.html">填写R2视觉反馈（独立保存）</a> · <a href="../regions.csv">统一区域逐帧CSV</a> · <a href="../response.csv">全25帧R2−R1响应</a> · <a href="../temporal.csv">时序区域变化</a> · <a href="../result.json">来源/随机状态/完整记录</a></p>
<video controls preload="metadata" src="RGB-A-R1-R2.mp4"></video>'''
    text+=viewer_fragment(item)
    text+='<h2>固定关键帧：0/8/16及18–24</h2><p>头部上下文同帧同坐标最近邻放大，包含混合上下文，不能当纯目标mask。</p><a href="contact-head.png"><img src="contact-head.png"></a><details><summary>完整画面对照接触图</summary><a href="contact-full.png"><img src="contact-full.png"></a></details>'
    def number(x):return '不适用' if x is None else f'{x:.6f}' if isinstance(x,float) else str(x)
    fields=['frame_id','region','pixels','A_RGB8_MAE','R1_RGB8_MAE','R2_RGB8_MAE','A_gradient_energy_ratio','R1_gradient_energy_ratio','R2_gradient_energy_ratio','R2_vs_R1_RGB8_MAE','repeat_vs_A_RGB8_MAE']
    text+='<details><summary>共同输入区域的逐帧数值（不是虚化质量分数）</summary><p>先于新输出固定区域；三组共用，避免R1空区对R2非空区。梯度在全图Sobel3×3计算、3×3腐蚀支持求比值。DAVIS无paired bokeh GT，不以相对RGB的MAE替代视觉判断。</p><table><tr>'+''.join('<th>'+x+'</th>' for x in fields)+'</tr>'
    for row in result['per_frame_region_metrics']:text+='<tr>'+''.join('<td>'+number(row[k])+'</td>' for k in fields)+'</tr>'
    text+='</table></details><h2>18→24区域变化</h2><p>每对使用共同目标或混合上下文的图像坐标并集；无运动补偿，不能称闪烁。</p><table>'
    tf=['transition','region','pixels','R1_temporal_RGB8_MAE','R2_temporal_RGB8_MAE','R1_input_relative_change_error','R2_input_relative_change_error','R2_vs_R1_temporal_response_MAE','repeat_vs_A_temporal_response_MAE']
    text+='<tr>'+''.join('<th>'+x+'</th>' for x in tf)+'</tr>'
    for row in result['temporal_region_change']:text+='<tr>'+''.join('<td>'+number(row[k])+'</td>' for k in tf)+'</tr>'
    text+='</table>'
    (page/'index.html').write_text(text);save(out/'viewer-item.json',item)
    form=render_review_form([item],package='B1',store_key=c['review_storage_key'],title='B36人工辅助末段R2：用户视觉复核',
        notice='参考与目标已确认；新输出质量尚未确认。请评价末段持续性、前段、边界和背景；保留无法判断/正常对照。')
    marker='<script id="atb-review-config" type="application/json">';start=form.index(marker)+len(marker);end=form.index('</script>',start)
    fc=json.loads(form[start:end]);fc['package']=item['package'];(out/'review.html').write_text(form[:start]+payload(fc)+form[end:])
    for suffix,name in [('-regions.csv','regions.csv'),('-response.csv','response.csv'),('-temporal.csv','temporal.csv'),('.json','result.json')]:link(out/name,ROOT/(c['record_prefix']+suffix))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>B36人工辅助末段R2完成，等待用户视觉复核</h1><a href="B36/">打开原RGB/A/R1/R2同步25帧</a> · <a href="review.html">R2视觉反馈</a>')
    save(ROOT/(c['record_prefix']+'-user-review-blank.json'),dict(scene_id=c['scene'],sample_id='B36',status='待用户视觉复核',
        review_storage_key=c['review_storage_key'],user_visual_review_completed=False,
        late_head_continuity='待用户复核',earlier_frames='待用户复核',boundary='待用户复核',background='待用户复核',naturalness='待用户复核'))
    print('MATERIALS_COMPLETE',flush=True)


def verify(c,out):
    import imageio.v2 as imageio
    inputs=torch.load(out/'inputs.pt',weights_only=True,map_location='cpu')
    priorc=read(ROOT/c['prior_config']);old=torch.load(ROOT/priorc['output']/'inputs.pt',weights_only=True,map_location='cpu')
    prepared=read(out/'prepared.json');candidate=inputs['rule_b_coc']
    for k in inputs['base']:assert bit_equal(inputs['base'][k],old['base'][k])
    for g,ids in enumerate(prepared['groups']):
        for j,t in enumerate(ids):
            assert (not bit_equal(candidate[g,j],old['rule_b_coc'][g,j]))==(t>=21)
    assert bit_equal(candidate[~inputs['valid']],old['rule_b_coc'][~inputs['valid']])
    result=read(out/'analysis.json');reg=read(out/'regions.json')
    for folder in ['native','native_repeat','revision1','rule_b']:
        arr=np.load(out/folder/'float.npy',mmap_mode='r');assert arr.shape==(25,576,1024,3) and np.isfinite(arr).all()
        assert len(list((out/folder/'png').glob('*.png')))==25
        for t in range(25):
            with Image.open(out/folder/'png'/f'{t:05d}.png') as im:
                assert im.mode=='RGB' and im.size==(1024,576)
                assert np.array_equal(np.asarray(im),np.uint8(np.round(np.clip(arr[t],0,1)*255)))
    for row in result['per_frame_region_metrics']:
        mask=np.asarray(Image.open(out/row['region_mask_path']))!=0
        assert int(mask.sum())==row['pixels'] and bbox(mask)==row['region_display_bbox']
        assert row['repeat_vs_A_RGB8_MAE']==0
    video=imageio.get_reader(out/'B36/RGB-A-R1-R2.mp4',format='FFMPEG');meta=video.get_meta_data();count=0
    for frame in video:assert frame.shape==(608,4096,3);count+=1
    video.close();assert count==25 and meta['fps']==20
    # Descriptive localization of response already measured on the frozen
    # background support. This does not add/change an evaluation or treatment ROI.
    peaks=[]
    for t in range(25):
        support=next(r for r in reg['frames'][t]['regions'] if r['region']=='background_all')
        mask=np.asarray(Image.open(out/support['path']))!=0
        a=np.asarray(Image.open(out/'revision1/png'/f'{t:05d}.png')).astype(np.int16)
        b=np.asarray(Image.open(out/'rule_b/png'/f'{t:05d}.png')).astype(np.int16)
        delta=np.abs(b-a);pixelmax=delta.max(axis=2);maximum=int(pixelmax[mask].max())
        y,x=np.argwhere(mask & (pixelmax==maximum))[0]
        distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        peaks.append(dict(frame_id=t,RGB8_max_channel_diff=maximum,
            first_peak_display_xy=[int(x),int(y)] if maximum else None,
            peak_distance_to_projected_subject_pixels=float(distance[y,x]) if maximum else None,
            RGB8_MAE=float(delta[mask].mean()),
            definition='Post-inference descriptive argmax on unchanged common background support; ties row-major. No new ROI or quality label.'))
    csv_write(ROOT/(c['record_prefix']+'-background-peaks.csv'),peaks)
    result['background_response_localization']=peaks
    result['background_peak_localization_limit']='Distance is to original nearest-projected DAVIS union at1024x576, not true foreground geometry or proof of halo. Descriptive CPU supplement after outputs, no rule/region change.'
    for path in [out/'analysis.json',ROOT/(c['record_prefix']+'.json')]:
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    link(out/'background-peaks.csv',ROOT/(c['record_prefix']+'-background-peaks.csv'))
    page=out/'B36/index.html'
    text='<h2>全背景响应峰值定位（CPU补充）</h2><p>沿用推理前固定的背景mask；逐帧取RGB8最大通道差，并列取row-major。只是已测变化的位置描述，不新增ROI，不作halo或感知失败判断。距离是到最近邻投影DAVIS主体的显示像素距离，不是真实几何。</p><a href="../background-peaks.csv">峰值CSV</a><table><tr><th>帧</th><th>R2−R1背景MAE</th><th>最大RGB8通道差</th><th>首个峰坐标</th><th>距投影主体/像素</th></tr>'
    for row in peaks:
        text+='<tr>'+''.join('<td>'+str(row[k])+'</td>' for k in ['frame_id','RGB8_MAE','RGB8_max_channel_diff','first_peak_display_xy','peak_distance_to_projected_subject_pixels'])+'</tr>'
    page.write_text(page.read_text()+text+'</table>')
    save(out/'program-checks.json',dict(checked_UTC=datetime.now(timezone.utc).isoformat(),frames=25,
        new_inference_runs=1,reused_conditions=['A','A-repeat','R1'],new_prediction_PNGs=25,
        comparison_PNGs=100,all_finite=True,PNG_roundtrip_equal=True,
        source0_20_CoC_bitwise_equal_R1=True,only_source21_24_conditions_differ=True,
        padding_unchanged=True,all25_output_frames_compared=True,
        common_regions_frozen_before_R2=True,region_rows=len(result['per_frame_region_metrics']),temporal_rows=len(result['temporal_region_change']),
        repeat_float_bitwise_equal=True,video=dict(frames=25,fps=20,size=[4096,608]),
        inference_started=True,user_visual_review_completed=False,label='人工辅助末段诊断'))
    print('PROGRAM_CHECKS_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['analyze','materials','verify']);args=parser.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(4);c=read(CFG);out=ROOT/c['output'];globals()[args.stage](c,out)
