"""Input-only correspondence, frozen-output viewing materials, and CPU plots.

locate saves RGB choices before materials reads disparity. plots uses the already
installed base Matplotlib environment; other commands use the unchanged ATB env.
"""
import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from review.any2bokeh_pages import viewer_fragment, render_review_form, VISUAL


def now(): return datetime.now(timezone.utc).isoformat()
def load(path): return json.loads((ROOT / path).read_text())
def save(path, value):
    p = ROOT / path; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_save(path, rows):
    p = ROOT / path; p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n'); w.writeheader()
        for row in rows: w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k,v in row.items()})


def write_new(path, value):
    p = ROOT / path; p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): assert p.read_text() == value, f'Preserve existing {p}'
    else: p.write_text(value)


def link(path, source):
    p = ROOT / path; s = ROOT / source; p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): assert p.resolve() == s.resolve()
    else: p.symlink_to(os.path.relpath(s, p.parent))


def png(path, array):
    p = ROOT / path; p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists(): Image.fromarray(array).save(p)


def rgb(scene, frame):
    p = ROOT / f'data/processed/davis2016-v2/eval/{scene}/rgb/{frame:05d}.jpg'
    a = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    assert a.shape == (1080,1920,3); return a


def gray(a): return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
def prefix(c, suffix): return c['record_prefix'] + suffix
def quant(a):
    if not len(a): return [None]*3
    return np.quantile(np.asarray(a, np.float64), [.1,.5,.9]).tolist()


def track(reference, current, center, policy):
    half = policy['patch_size']//2; x,y = np.round(center).astype(int)
    template = reference[y-half:y+half+1,x-half:x+half+1]
    rx,ry = policy['search_xy_radius']; x0,y0 = x-half-rx,y-half-ry
    search = current[y0:y+half+ry+1,x0:x+half+rx+1]
    score = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _,best,_,pos = cv2.minMaxLoc(score); px,py = pos
    other = score.copy(); other[max(0,py-3):py+4,max(0,px-3):px+4] = -1
    margin = best-float(other.max()); target = [int(x0+px+half),int(y0+py+half)]
    tx,ty = target; reverse_template = current[ty-half:ty+half+1,tx-half:tx+half+1]
    reverse = cv2.matchTemplate(reference[y0:y+half+ry+1,x0:x+half+rx+1], reverse_template, cv2.TM_CCOEFF_NORMED)
    _,reverse_score,_,back = cv2.minMaxLoc(reverse)
    reverse_error = float(np.hypot(x0+back[0]+half-x,y0+back[1]+half-y))
    reasons=[]
    if best < policy['min_NCC']: reasons.append('NCC below frozen threshold')
    if margin < policy['min_peak_margin_outside_3px']: reasons.append('ambiguous RGB registration peak')
    if reverse_error > policy['max_reverse_error_native']: reasons.append('reverse registration failed')
    return target, dict(NCC=best, peak_margin=margin, reverse_NCC=reverse_score, reverse_error_native=reverse_error), reasons


def band_pixels(center, tangent, offset, length, width):
    tangent=np.asarray(tangent,np.float64); tangent/=np.linalg.norm(tangent)
    normal=np.array([-tangent[1],tangent[0]]); center=np.asarray(center)+normal*offset
    corners=np.array([center+tangent*a+normal*b for a,b in [(-length/2,-width/2),(length/2,-width/2),(length/2,width/2),(-length/2,width/2)]])
    x0,y0=np.floor(corners.min(axis=0)-1).astype(int);x1,y1=np.ceil(corners.max(axis=0)+1).astype(int)
    xs,ys=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1));v=np.stack([xs-center[0],ys-center[1]],axis=-1)
    take=(np.abs(v@tangent)<=length/2)&(np.abs(v@normal)<width/2)
    return xs[take],ys[take],corners.tolist()


def contrast(current, center, tangent, c):
    length=c['B33']['band_length_native'];width=c['B33']['band_width_native'];gap=c['B33']['grass_normal_offset_native']
    core_x,core_y,_=band_pixels(center,tangent,0,length,width)
    side1_x,side1_y,_=band_pixels(center,tangent,gap,length,width);side2_x,side2_y,_=band_pixels(center,tangent,-gap,length,width)
    return float(current[core_y,core_x].mean()-(current[side1_y,side1_x].mean()+current[side2_y,side2_x].mean())/2)


def locate(c):
    target=prefix(c,'-localization.json')
    if (ROOT/target).exists(): print('REUSE_LOCALIZATION',flush=True); return
    split=load('03_实验记录/records/any2bokeh-failure-analysis-20261009-source-split.json')['rows']
    assert all(any(r['scene_id']==c[k]['scene_id'] and r['split']=='exploration' for r in split) for k in ['B33','B36'])
    # This durable manifest precedes all new disparity reading.
    selection_path=prefix(c,'-RGB-selection.json')
    if (ROOT/selection_path).exists():assert load(selection_path)['selection']==c
    else:save(selection_path,dict(saved_utc=now(),source_commit=c['source_commit'],selection=c,
         selected_from='Unannotated RGB only; blue image used for object localization',new_disparity_values_read=False))
    marked=[]
    for file,panel,t in c['B33']['marked_files']:
        a=np.asarray(Image.open(ROOT/file).convert('RGB'))
        source=ROOT/f'outputs/any2bokeh-failure-analysis-20261009/B1-batch2-20261014/B33/originals/{panel}/{t:05d}.png'
        original=np.asarray(Image.open(source).convert('RGB')); assert a.shape==original.shape==(576,1024,3)
        diff=np.any(a!=original,axis=-1); hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
        blue=diff&(hsv[:,:,0]>=85)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>=100)&(hsv[:,:,2]>=100)
        outside=int((diff&~(cv2.dilate(blue.astype(np.uint8),np.ones((7,7),np.uint8))>0)).sum());assert outside==0
        ys,xs=np.where(diff);marked.append(dict(file=file,panel=panel,frame_id=t,size=[1024,576],changed_pixels=int(diff.sum()),
            changed_bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],outside_blue_marker_dilation_mismatches=outside,
            scope='Pixel/frame correspondence only; no new perceptual verdict'))
    reference=gray(rgb('car-turn',0)); rows=[]
    for t in range(25):
        current=gray(rgb('car-turn',t))
        for p in c['B33']['rope_points']+c['B33']['post_controls']:
            is_rope=p['id'].startswith('R'); initial=p.get('native_center')
            if initial is None: x,y,w,h=p['native_xywh'];initial=[x+w/2,y+h/2]
            center,checks,reasons=track(reference,current,initial,c['B33']['tracking'])
            if is_rope:
                tangent=[1,p['slope']];normal=np.array([-p['slope'],1]);normal/=np.linalg.norm(normal)
                candidates=[(contrast(current,np.array(center)+normal*off,tangent,c),off) for off in c['B33']['tracking']['normal_refinement_offsets_native']]
                score,off=max(candidates,key=lambda pair:(pair[0],-abs(pair[1])))
                center=(np.array(center)+normal*off).tolist();checks.update(RGB_rope_contrast=score,normal_refinement_native=off)
                if score<c['B33']['tracking']['min_rope_RGB_contrast']: reasons.append('rope too weak to localize by RGB contrast')
            else:tangent=None
            rows.append(dict(scene_id='car-turn',frame_id=t,correspondence_id=p['id'],accepted=not reasons,
                native_center=center if not reasons else None,candidate_native_center=center,RGB_registration_checks=checks,
                localization_reliability='RGB context registration passed; exact rope material-point and pure-pixel identity unverified' if not reasons else 'UNAVAILABLE',
                missing_reason='; '.join(reasons),tangent=tangent,is_rope=is_rope))
        print('RGB_LOCALIZATION_FRAME',t,flush=True)
    save(target,dict(saved_utc=now(),new_disparity_values_read=False,marked_image_checks=marked,policy=c['B33']['tracking'],rows=rows,
        accepted_counts={p['id']:sum(r['accepted'] for r in rows if r['correspondence_id']==p['id']) for p in c['B33']['rope_points']+c['B33']['post_controls']},
        limits=['Context contains adjacent grass; registration is not proof of independent rope material-point identity',
                'RGB contrast confirms a bright narrow-line candidate, not pristine rope segmentation','No invalid point is interpolated or replaced']))
    csv_save(prefix(c,'-localization.csv'),rows)
    print('RGB_LOCALIZATION_SAVED_BEFORE_NEW_DISPARITY',flush=True)


def geometry(scene,t):
    base=ROOT/f'data/processed/davis2016-v2/eval/{scene}'; files=list((base/'disparity').glob(f'{t:05d}_zf_*.png'));assert len(files)==1
    q=np.asarray(Image.open(files[0]));focus=float(files[0].stem.split('_zf_')[1]);assert q.shape==(1080,1920) and q.dtype==np.uint8
    mask=np.asarray(Image.open(base/'mask'/f'{t:05d}.png'))>128
    assert focus==float(f'{q[mask].mean()/255:.5f}');return q.astype(np.float64)/255,focus,files[0]


def stat(scene,t,cid,kind,xs,ys,polygon,focus,d,reliability,reason,width):
    a=d[ys,xs] if len(xs) else np.array([])
    return dict(scene_id=scene,frame_id=t,correspondence_id=cid,region_type=kind,coordinate_system='native original VDA/RGB grid1920x1080; integer pixel centers',
        native_polygon=polygon,sampling_width_native=width,valid_pixels=int(len(a)),localization_reliability=reliability,
        disparity_p10_p50_p90=quant(a),focus=focus,K=16,signed_d_minus_focus_p10_p50_p90=quant(a-focus),
        nominal_defocus_p10_p50_p90=quant(16*np.abs(a-focus)),missing_reason=reason)


def b33_stats(c,loc):
    rows=[];overlay_polygons={};tracked=[]
    for point in c['B33']['rope_points']+c['B33']['post_controls']:
        tracked.append(dict(id=point['id'],name=point['id']+' RGB定位候选（非用户确认）',xywh_by_frame={}))
    for t in range(25):
        d,focus,_=geometry('car-turn',t);overlay_polygons[str(t)]=[]
        for r in [v for v in loc['rows'] if v['frame_id']==t]:
            cid=r['correspondence_id'];positions=[]
            if r['is_rope']:
                for kind,offset in [('rope_center_candidate',0),('grass_side_A_candidate',8),('grass_side_B_candidate',-8)]:
                    if r['accepted']: xs,ys,poly=band_pixels(r['native_center'],r['tangent'],offset,c['B33']['band_length_native'],c['B33']['band_width_native'])
                    else:xs=ys=np.array([],int);poly=None
                    positions.append(set(zip(xs.tolist(),ys.tolist())))
                    rows.append(stat('car-turn',t,cid,kind,xs,ys,poly,focus,d,r['localization_reliability'],r['missing_reason'],3))
                    if poly:overlay_polygons[str(t)].append(dict(correspondence_id=cid,kind=kind,native_polygon=poly))
                assert not (positions[0]&positions[1] or positions[0]&positions[2] or positions[1]&positions[2])
            else:
                p=next(p for p in c['B33']['post_controls'] if p['id']==cid);x,y,w,h=p['native_xywh']
                if r['accepted']:
                    cx,cy=r['native_center'];x0=int(round(cx-w/2));y0=int(round(cy-h/2));xs,ys=np.meshgrid(np.arange(x0,x0+w),np.arange(y0,y0+h));xs=xs.ravel();ys=ys.ravel();poly=[[x0,y0],[x0+w,y0],[x0+w,y0+h],[x0,y0+h]]
                else:xs=ys=np.array([],int);poly=None
                rows.append(stat('car-turn',t,cid,'post_control',xs,ys,poly,focus,d,r['localization_reliability'],r['missing_reason'],w))
                if poly:overlay_polygons[str(t)].append(dict(correspondence_id=cid,kind='post_control',native_polygon=poly))
            pos=None
            if r['accepted']:
                cx,cy=np.array(r['native_center'])/1.875;pos=[int(round(cx))-24,int(round(cy))-20,48,40]
            next(p for p in tracked if p['id']==cid)['xywh_by_frame'][str(t)]=pos
    comparisons=[]
    for p in c['B33']['rope_points']:
        pairs=[r for r in rows if r['correspondence_id']==p['id']]; by={(r['frame_id'],r['region_type']):r for r in pairs}
        for t in range(25):
            core=by[t,'rope_center_candidate']
            for side in ['grass_side_A_candidate','grass_side_B_candidate']:
                grass=by[t,side];delta=core['disparity_p10_p50_p90'][1]-grass['disparity_p10_p50_p90'][1] if core['valid_pixels'] else None
                comparisons.append(dict(frame_id=t,correspondence_id=p['id'],grass_side=side,rope_minus_grass_median=delta,delta_in_q_steps=None if delta is None else delta*255))
    save(prefix(c,'-B33-input-statistics.json'),dict(calculated_utc=now(),selection_saved_utc=load(prefix(c,'-RGB-selection.json'))['saved_utc'],rows=rows,
        comparisons=comparisons,tracked_rois=tracked,overlay_polygons=overlay_polygons,
        limits=['3-native-pixel candidate band may include grass and mixed pixels; not pure rope truth',
            'Adjacent grass candidates may contain small rocks or other background details',
            'Adjacent grass bands follow the rope registration candidate; no independent grass material-point tracking',
            'Current resolution/localization cannot establish pure rope disparity',
            'These local registration candidates require user correspondence review before a physical interpretation',
            'Nominal values are K*abs(d-focus) on native q/255; not actual ATB condition or measured output blur']))
    csv_save(prefix(c,'-B33-input-statistics.csv'),rows);csv_save(prefix(c,'-B33-rope-grass-differences.csv'),comparisons)
    return rows,tracked,overlay_polygons


def b36_stats(c):
    rows=[];tracked=[dict(id=k,name=name,xywh_by_frame={str(t):None for t in range(25)}) for k,name in [('head','头/喙同尺度上下文'),('neck','颈部对照'),('body','身体对照')]]
    old=load('03_实验记录/records/any2bokeh-failure-analysis-20261014-B36-input-diagnosis.json')['statistics']
    for t in c['B36']['keyframes']:
        d,focus,_=geometry('mallard-water',t);rois=c['B36']['ROIs_display_xywh'][str(t)]
        for kind,box in rois.items():
            if kind=='head_context':continue
            x,y,w,h=box;x0,y0=math.floor(x*1.875),math.floor(y*1.875);x1,y1=math.ceil((x+w)*1.875),math.ceil((y+h)*1.875)
            xs,ys=np.meshgrid(np.arange(x0,x1),np.arange(y0,y1));poly=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
            row=stat('mallard-water',t,kind,kind,xs.ravel(),ys.ravel(),poly,focus,d,'Executor localized on input RGB; user confirmation pending','',w*1.875)
            row['display_xywh']=box
            union=np.asarray(Image.open(ROOT/f'data/processed/davis2016-v2/eval/mallard-water/mask/{t:05d}.png'))>128
            row['annotation_foreground_fraction']=float(union[ys.ravel(),xs.ravel()].mean())
            if t in [0,6]:
                previous=next(r for r in old if r['frame']==t and r['region']==kind)
                assert np.max(np.abs(np.array(previous['d_p10_p50_p90'])-row['disparity_p10_p50_p90']))<1e-12
                assert row['valid_pixels']==previous['pixel_count']
            rows.append(row)
        tracked[0]['xywh_by_frame'][str(t)]=rois['head_context']
        for index,kind,w,h in [(1,'neck_core',64,64),(2,'body_core',96,64)]:
            x,y,rw,rh=rois[kind];tracked[index]['xywh_by_frame'][str(t)]=[int(round(x+rw/2-w/2)),int(round(y+rh/2-h/2)),w,h]
    save(prefix(c,'-B36-input-statistics.json'),dict(calculated_utc=now(),rows=rows,tracked_rois=tracked,old_frame0_6_core_values_reused_and_verified=True,
        limits=['RGB-localized keyframe ROIs are executor-selected; no independent true geometry',
                'Non-keyframe ROI presets are absent; full25 predictions remain available, no ROI interpolation',
                'Existing water checks0/6 retained in prior input-diagnosis; not a universal water-depth conclusion']))
    csv_save(prefix(c,'-B36-input-statistics.csv'),rows);return rows,tracked


def page(c, code, panels, video, tracked, initial, jumps, extra):
    scene=c[code]['scene_id'];folder=Path(c['output'])/code
    item=dict(sample_id=code,dataset='DAVIS',scene_id=scene,source_group='DAVIS:'+scene,split='exploration',package='B1' if code=='B33' else 'B2',
        panels=panels,tracked_rois=tracked,initial_frame=initial,frame_jumps=jumps,review_url=f'../review-{code}.html')
    style='<style>body{font-family:system-ui;background:#f6f9ff;color:#173657;margin:24px}a{color:#1263a0}img{max-width:100%}video{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #a4bddd;padding:6px}</style>'
    body=f'<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>{code}专项诊断</title>{style}<h1>{code} {scene} 专项诊断</h1><p>探索组既有数据。所有新视觉结论由用户填写，程序不判哪个方法更好。</p>{extra}<video controls preload="metadata" src="{video}"></video>'+viewer_fragment(item)
    write_new(folder/'index.html',body)
    sample={k:item[k] for k in ['sample_id','dataset','scene_id','source_group','split']}
    form=render_review_form([sample],item['package'],store_key=c[code]['review_storage_key'],title=code+'专项用户复核',
        notice='独立专项key；既有浏览器内容与聊天原话保留，不自动合并。所有视觉字段待用户填写；方法映射已知，后续观察不是严格盲评。')
    if code=='B36':question='鸭头是否变软？与身体相比是否明显？哪些面板保留更好？是否伴随背景虚化强弱差别？也可记录正常、无明显差异或无法判断。'
    else:question='请先核对R1–R4是否在各帧仍是同一局部绳段、邻近窄带是否合理；比较原RGB与ATB。原授权的两张输出图复核不扩展成本轮自动视觉验收。'
    form=form.replace('<div id="forms">','<p>'+question+'</p><div id="forms">')
    write_new(Path(c['output'])/f'review-{code}.html',form)
    blank=dict(sample,package=item['package'],frame_id_or_transition_id='',ROI_xywh='',actual_view_mode='',user_observation='',
        severity='待用户复核',confidence='待用户复核',actual_playback_rate='',normal_motion_or_reasonable_defocus='待用户复核',
        similar_in_input_or_GT='待用户复核',normal_control='',notes='',**{key:'待用户复核' for key in VISUAL})
    save(prefix(c,f'-{code}-human-review-blank.json'),[blank]);csv_save(prefix(c,f'-{code}-human-review-blank.csv'),[blank])


def materials(c):
    target=prefix(c,'-materials.json')
    if (ROOT/target).exists():print('REUSE_MATERIALS',flush=True);return
    loc=load(prefix(c,'-localization.json'));b33,tracked33,polys=b33_stats(c,loc);b36,tracked36=b36_stats(c)
    from analyze_any2bokeh_failures import video_encode,contact_sheet
    original=load('configs/any2bokeh-failure-analysis-20261009.json');out=Path(c['output']);manifest={}
    mapping=load(c['B36']['anonymous_mapping'])['mapping']['DAVIS']
    for filename in ['disparity-scale.png','defocus-scale.png']:
        link(out/filename,Path('outputs/any2bokeh-failure-analysis-20261009/B19-input-check-20261014')/filename)
    write_new(out/'method-mapping.json',json.dumps(dict(mapping=mapping,user_already_knows_mapping=True,strict_blind_review=False),ensure_ascii=False,indent=2)+'\n')
    for code in ['B33','B36']:
        folder=out/code;scene=c[code]['scene_id'];panels=[dict(label='输入RGB',prefix='originals/RGB/')]
        if code=='B33':panels.append(dict(label='RGB采样叠加（诊断框非用户确认）',prefix='originals/Sampling_overlay/'))
        panels += [dict(label='原网格VDA q/255（固定0–1，NEAREST显示）',prefix='originals/VDA/'),dict(label='原网格名义离焦（固定0–16）',prefix='originals/Nominal/')]
        methods=['any2bokeh'] if code=='B33' else c['B36']['methods']
        for method in methods:panels.append(dict(label='冻结ATB' if code=='B33' else next(k for k,v in mapping.items() if v==method),prefix=f'originals/{method}/'))
        for t in range(25):
            source=f'outputs/any2bokeh-failure-analysis-20261009/B1-batch2-20261014/{code}/originals/Input_RGB/{t:05d}.png';link(folder/'originals/RGB'/f'{t:05d}.png',source)
            d,focus,qp=geometry(scene,t)
            for kind,a,high in [('VDA',d,1),('Nominal',16*np.abs(d-focus),16)]:
                colored=cv2.cvtColor(cv2.applyColorMap(np.uint8(np.round(np.clip(a/high,0,1)*255)),cv2.COLORMAP_VIRIDIS),cv2.COLOR_BGR2RGB)
                png(folder/'originals'/kind/f'{t:05d}.png',cv2.resize(colored,(1024,576),interpolation=cv2.INTER_NEAREST))
                if code=='B33':png(folder/'native'/kind/f'{t:05d}.png',colored)
            link(folder/'native/RGB'/f'{t:05d}.jpg',f'data/processed/davis2016-v2/eval/{scene}/rgb/{t:05d}.jpg')
            link(folder/'native/q'/qp.name,qp.relative_to(ROOT))
            if code=='B33':
                overlay=np.asarray(Image.open(ROOT/source).convert('RGB')).copy()
                for p in polys[str(t)]:
                    pts=np.round(np.array(p['native_polygon'])/1.875).astype(np.int32);color=(0,220,255) if p['kind']=='rope_center_candidate' else (255,80,180) if p['kind']=='grass_side_A_candidate' else (0,230,80) if p['kind']=='grass_side_B_candidate' else (255,170,0)
                    cv2.polylines(overlay,[pts],True,color,1);cv2.putText(overlay,p['correspondence_id'],tuple(pts[0]+[0,-4]),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
                png(folder/'originals/Sampling_overlay'/f'{t:05d}.png',overlay)
            for m in methods:
                src=f'outputs/davis-baselines-20260929/{m}/predictions/{scene}/{t:05d}.png'
                with Image.open(ROOT/src) as im:assert im.size==(1024,576) and im.mode=='RGB'
                link(folder/f'originals/{m}/{t:05d}.png',src)
        if code=='B33':link(folder/'video.mp4','outputs/any2bokeh-failure-analysis-20261009/B1-batch2-20261014/B33/video.mp4')
        else:
            video_panels=[panels[0]]+panels[3:];pp=[sorted((ROOT/folder/p['prefix']).glob('*.png')) for p in video_panels]
            video_encode(original,ROOT/folder/'video.mp4',pp,['Input_RGB']+[p['label'] for p in video_panels[1:]],3)
        paths=[sorted((ROOT/folder/p['prefix']).glob('*.png')) for p in panels]
        keys=[1,22] if code=='B33' else c['B36']['keyframes']
        contact_sheet(ROOT/folder/'keyframes.png',paths,['Input_RGB','RGB_ROIs','VDA','Nominal','ATB'] if code=='B33' else ['Input_RGB','VDA','Nominal']+[p['label'] for p in panels[3:]],3,keys)
        extra='<p><a href="../">专项总索引/统计下载</a> · <a href="keyframes.png">端点/关键帧概览</a></p><p>VDA及名义量先在1920×1080原网格计算，再NEAREST显示；色标固定、裁剪不拉伸。q/255非米制真实深度；名义量不是输出模糊半径或ATB实际条件。RGB显示为原冻结INTER_LINEAR到1024×576。</p>'
        extra+='<p><img src="../disparity-scale.png" alt="固定视差0–1色标"><img src="../defocus-scale.png" alt="固定名义离焦0–16色标"></p><details><summary>全部25帧原网格RGB与VDA q入口（1920×1080）</summary>'
        for t in range(25):
            _,_,qp=geometry(scene,t)
            extra+=f'<p>frame{t}: <a href="native/RGB/{t:05d}.jpg">原RGB</a> · <a href="native/q/{qp.name}">原q</a></p>'
        extra+='</details>'
        if code=='B33':
            extra+='<p>默认帧1；帧1与22是非相邻端点，不证明中间单调变化。R1–R4为RGB局部配准候选，窄带可能混入草地，不能称纯净细绳真值。相邻帧预设ROI随帧变化；失效处留空，不插补。青=绳中心候选，粉/绿=两侧邻近草地候选，橙=独立立柱对照。</p><p>'+''.join(f'<a href="curves/{p["id"]}.png">{p["id"]}独立曲线</a> · ' for p in c['B33']['rope_points']+c['B33']['post_controls'])+'</p>'
            for p in tracked33:
                extra+=f'<h3>{p["id"]} 原RGB与ATB端点局部（1与22，非连续transition）</h3><div style="display:flex;gap:8px;flex-wrap:wrap">'
                for t in [1,22]:
                    box=p['xywh_by_frame'][str(t)]
                    if box is None:extra+=f'<p>frame{t}定位未通过门槛，局部留空</p>';continue
                    x,y,w,h=box
                    for label,key in [('Input_RGB','RGB'),('ATB','any2bokeh')]:
                        im=Image.open(ROOT/folder/f'originals/{key}/{t:05d}.png').crop((x,y,x+w,y+h));dest=folder/f'local/{p["id"]}-{t:05d}-{label}.png';(ROOT/dest).parent.mkdir(parents=True,exist_ok=True);im.save(ROOT/dest)
                        extra+=f'<figure><figcaption>{p["id"]} {label} frame{t} 48×40；统一NN 4×</figcaption><img style="width:192px;image-rendering:pixelated" src="local/{dest.name}"></figure>'
                    # Native input crop, without first reducing to the display grid.
                    r=next(r for r in loc['rows'] if r['frame_id']==t and r['correspondence_id']==p['id']);cx,cy=r['native_center'];a=rgb('car-turn',t);crop=a[int(round(cy))-38:int(round(cy))+38,int(round(cx))-45:int(round(cx))+45]
                    png(folder/f'local/{p["id"]}-{t:05d}-native-RGB.png',crop)
                    extra+=f'<p><a href="local/{p["id"]}-{t:05d}-native-RGB.png">frame{t}原网格RGB局部90×76</a></p>'
                extra+='</div>'
        else:
            extra+='<p>本轮P编号沿用旧映射，用户已知方法名称，不是严格盲评。DeepLens自身DepthNet/native-default，不是共享VDA等价对照；BokehMe/BokehDiff保留冻结Normalized/span，不补偿K；Dr.Bokeh为DAVIS Full-public。共享外部VDA/focus/K不等于内部有效离焦相同。</p><p>头/喙同尺度128×160，颈/身体分别固定64×64和96×64；五关键帧按输入RGB调整坐标，同帧所有方法用完全相同位置。非关键帧无预设跟踪位置，仍可查看全25帧并手动选框。</p>'
            for t in keys:
                extra+=f'<h3>frame{t}局部（所有面板同坐标，统一最近邻2×）</h3>'
                extra+='<table><tr><th>输入诊断区域</th><th>native像素数</th><th>q/255中位数</th><th>focus</th><th>名义离焦中位数</th><th>标注前景占比</th></tr>'
                for row in [row for row in b36 if row['frame_id']==t]:
                    extra+=f'<tr><td>{row["region_type"]}</td><td>{row["valid_pixels"]}</td><td>{row["disparity_p10_p50_p90"][1]:.5f}</td><td>{row["focus"]:.5f}</td><td>{row["nominal_defocus_p10_p50_p90"][1]:.4f}</td><td>{row["annotation_foreground_fraction"]:.3f}</td></tr>'
                extra+='</table><p>喙框可能包含水面（尤其12/18帧），不是纯喙像素真值；不依据统计更换ROI。标注前景不意味着同深度或必须清晰。</p>'
                for tracked in tracked36:
                    x,y,w,h=tracked['xywh_by_frame'][str(t)];extra+=f'<details><summary>{tracked["name"]} xywh={x,y,w,h}</summary><div style="display:flex;flex-wrap:wrap;gap:8px">'
                    for panel in [panels[0]]+panels[3:]:
                        label='Input_RGB' if panel==panels[0] else panel['label'];im=Image.open(ROOT/folder/panel['prefix']/f'{t:05d}.png').crop((x,y,x+w,y+h));dest=folder/f'local/{t:05d}-{tracked["id"]}-{label}.png';(ROOT/dest).parent.mkdir(parents=True,exist_ok=True);im.save(ROOT/dest)
                        extra+=f'<figure><figcaption>{label}</figcaption><a href="local/{dest.name}"><img style="width:{w*2}px;image-rendering:pixelated" src="local/{dest.name}"></a></figure>'
                    extra+='</div></details>'
        page(c,code,panels,'video.mp4',tracked33 if code=='B33' else tracked36,1 if code=='B33' else 0,keys,extra)
        manifest[code]=dict(index=str(folder/'index.html'),panels=panels,frames=25,tracked_rois=tracked33 if code=='B33' else tracked36,video=str(folder/'video.mp4'))
        print('MATERIAL_COMPLETE',code,flush=True)
    index='<!doctype html><meta charset="utf-8"><title>B33/B36诊断材料</title><h1>B33/B36诊断材料</h1><p>诊断材料已准备，等待用户视觉复核。仅探索数据/CPU分析，不进行机制验证或Ours开发。</p><p><a href="B33/">①B33细绳定位/曲线/原输入与ATB</a> · <a href="B36/">②B36鸭头六方法同位置对照</a></p><p><a href="B33-input-statistics.csv">B33输入统计</a> · <a href="localization.csv">RGB定位清单</a> · <a href="B36-input-statistics.csv">B36输入统计</a> · 方法映射单独保存于method-mapping.json</p>'
    for filename,panel,t in c['B33']['marked_files']:link(out/'marked'/f'{panel}-{t:05d}.png',filename)
    index+='<p>用户原圈选图（只用于定位，未改动）：<a href="marked/Input_RGB-00000.png">输入0</a> · <a href="marked/Frozen_ATB-00001.png">ATB1</a> · <a href="marked/Frozen_ATB-00022.png">ATB22</a></p>'
    write_new(out/'index.html',index)
    for suffix,dest in [('-B33-input-statistics.csv','B33-input-statistics.csv'),('-localization.csv','localization.csv'),('-B36-input-statistics.csv','B36-input-statistics.csv')]:link(out/dest,prefix(c,suffix))
    save(target,dict(prepared_utc=now(),items=manifest,storage_keys={k:c[k]['review_storage_key'] for k in ['B33','B36']},old_keys_unchanged=True,
        neural_inference=False,user_visual_review_completed=False,status='诊断材料已准备，等待用户视觉复核'))


def plots(c):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=load(prefix(c,'-B33-input-statistics.json'))['rows'];folder=ROOT/c['output']/'B33/curves';folder.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for p in c['B33']['rope_points']+c['B33']['post_controls']:
        own=[r for r in rows if r['correspondence_id']==p['id']];is_rope=p['id'].startswith('R')
        kinds=['rope_center_candidate','grass_side_A_candidate','grass_side_B_candidate'] if is_rope else ['post_control']
        fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True);summ=dict(correspondence_id=p['id'],sides=[])
        for kind in kinds:
            seq=sorted([r for r in own if r['region_type']==kind],key=lambda r:r['frame_id']);x=[r['frame_id'] for r in seq]
            for ax,key in zip(axes,['disparity_p10_p50_p90','nominal_defocus_p10_p50_p90','signed_d_minus_focus_p10_p50_p90']):
                a=np.array([[np.nan if v is None else v for v in r[key]] for r in seq]);line=ax.plot(x,a[:,1],'.-',label=kind,linewidth=1)[0];ax.fill_between(x,a[:,0],a[:,2],alpha=.12,color=line.get_color())
            if is_rope and kind!='rope_center_candidate':
                core=sorted([r for r in own if r['region_type']=='rope_center_candidate'],key=lambda r:r['frame_id']);valid=[i for i,r in enumerate(seq) if r['valid_pixels'] and core[i]['valid_pixels']]
                a=np.array([core[i]['disparity_p10_p50_p90'][1] for i in valid]);b=np.array([seq[i]['disparity_p10_p50_p90'][1] for i in valid])
                corr=float(np.corrcoef(a,b)[0,1]) if len(valid)>=3 and np.ptp(a)>0 and np.ptp(b)>0 else None
                ca=np.array([core[i]['nominal_defocus_p10_p50_p90'][1] for i in valid]);cb=np.array([seq[i]['nominal_defocus_p10_p50_p90'][1] for i in valid])
                nominal_corr=float(np.corrcoef(ca,cb)[0,1]) if len(valid)>=3 and np.ptp(ca)>0 and np.ptp(cb)>0 else None
                summ['sides'].append(dict(grass_side=kind,common_frames=len(valid),median_absolute_disparity_delta=float(np.median(np.abs(a-b))) if len(valid) else None,
                    max_absolute_disparity_delta=float(np.max(np.abs(a-b))) if len(valid) else None,Pearson_corresponding_frame_medians=corr,
                    Pearson_nominal_defocus_frame_medians=nominal_corr,nominal_correlation_limit='Both nominal curves share the supplied focus; common focus drift can produce correlation without synchronized disparity',
                    limit='Dependent frames of one source; registration/mixed-pixel candidates, no causal or inferential significance claim'))
        axes[0].plot(range(25),[next(r for r in own if r['frame_id']==t)['focus'] for t in range(25)],'k--',label='given focus')
        for ax in axes:
            ax.axvline(1,color='#7c8798',linestyle=':');ax.axvline(22,color='#7c8798',linestyle=':');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
        axes[0].set_ylabel('Native disparity q/255');axes[1].set_ylabel('Nominal K|d-focus|');axes[2].set_ylabel('Signed d-focus');axes[2].set_xlabel('frame (index; no interpolation)')
        axes[0].set_title(p['id']+' | RGB-registered input candidates; missing points break lines; bands p10-p90')
        fig.tight_layout();fig.savefig(folder/(p['id']+'.png'),dpi=150);fig.savefig(folder/(p['id']+'.svg'));plt.close(fig);summaries.append(summ)
    save(prefix(c,'-B33-numeric-summary.json'),dict(created_utc=now(),per_correspondence=summaries,perceptual_conclusions='待用户复核',
        limitations=['Correlation is descriptive, not evidence of depth error or background mixing causation','No pooling of different rope sites or source-independent repeats','No smoothing or missing-position interpolation']))
    print('PER_CORRESPONDENCE_PLOTS_COMPLETE',len(summaries),flush=True)


def verify(c):
    loc=load(prefix(c,'-localization.json'));a=load(prefix(c,'-B33-input-statistics.json'));b=load(prefix(c,'-B36-input-statistics.json'));m=load(prefix(c,'-materials.json'))
    assert len(a['rows'])==350 and len(b['rows'])==20
    assert load(prefix(c,'-RGB-selection.json'))['saved_utc']<a['calculated_utc']
    for row in a['rows']+b['rows']:
        if row['valid_pixels']==0:
            assert row['native_polygon'] is None and row['missing_reason'] and row['disparity_p10_p50_p90']==[None]*3
        else:
            assert row['native_polygon'] and all(0<=v<=1 for v in row['disparity_p10_p50_p90'])
            assert np.isfinite(row['nominal_defocus_p10_p50_p90']).all()
            assert all(0<=x<1920 and 0<=y<1080 for x,y in row['native_polygon'])
    png_count=0;videos=[]
    for code,item in m['items'].items():
        folder=(ROOT/item['index']).parent
        for panel in item['panels']:
            files=sorted((folder/panel['prefix']).glob('*.png'));assert [p.stem for p in files]==[f'{t:05d}' for t in range(25)]
            for p in files:
                with Image.open(p) as im: assert im.size==(1024,576) and im.mode=='RGB';im.verify()
                png_count+=1
        for roi in item['tracked_rois']:
            assert set(roi['xywh_by_frame'])=={str(t) for t in range(25)}
            for box in roi['xywh_by_frame'].values():
                if box is not None:x,y,w,h=box;assert 0<=x and 0<=y and x+w<=1024 and y+h<=576
        cap=cv2.VideoCapture(str(ROOT/item['video']));fps=cap.get(cv2.CAP_PROP_FPS);n=0;size=None
        while True:
            ok,frame=cap.read()
            if not ok:break
            n+=1;size=[frame.shape[1],frame.shape[0]];assert np.isfinite(frame).all()
        cap.release();assert n==25 and fps==20
        videos.append(dict(sample_id=code,frames=n,fps=fps,size=size,finite=True))
        assert all(all(r[k]=='待用户复核' for k in VISUAL) and r['user_observation']=='' for r in load(prefix(c,f'-{code}-human-review-blank.json')))
    curves=ROOT/c['output']/'B33/curves'
    for point in c['B33']['rope_points']+c['B33']['post_controls']:
        assert (curves/(point['id']+'.svg')).is_file()
        with Image.open(curves/(point['id']+'.png')) as im:im.verify()
    record=dict(checked_utc=now(),status='PROGRAM_CHECKS_COMPLETE_BROWSER_PENDING',source_commit=c['source_commit'],
        RGB_marker_correspondence=loc['marked_image_checks'],RGB_selection_before_disparity=True,accepted_positions=loc['accepted_counts'],
        B33_stat_rows=len(a['rows']),B33_missing_stat_rows=sum(r['valid_pixels']==0 for r in a['rows']),B36_stat_rows=len(b['rows']),
        original_frame0_6_B36_values_match=True,PNG_integrity_checks=png_count,decoded_videos=videos,per_correspondence_charts=6,
        source_split='Both videos in original exploration; held-out data not opened',all_perceptual_fields_pending=True,
        failed_initial_localization_log='logs/atb-b33-b36-locate-20261014.log (JSON numpy-int serialization error; fixed without changing selections or thresholds)',
        user_images_preserved_outside_Git=True,neural_inference=False)
    save(prefix(c,'-verification.json'),record);print('PROGRAM_CHECKS_PASS',png_count,videos,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['locate','materials','plots','verify'])
    parser.add_argument('--config',default='configs/any2bokeh-B33-B36-diagnostics-20261014.json');args=parser.parse_args();c=load(args.config)
    if args.command=='plots':plots(c);return
    global cv2
    import cv2
    cv2.setNumThreads(1)
    if args.command=='locate':locate(c)
    elif args.command=='materials':materials(c)
    else:verify(c)


if __name__=='__main__':main()
