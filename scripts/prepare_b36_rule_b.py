"""CPU-only B36 rule-b input localization and distribution inspection.

This entry point deliberately has no model loading or inference command.
Localization precedes disparity inspection; candidate parameters must be fixed
in the config before rendering proposed CoC. User approval is a separate step.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT/'configs/any2bokeh-B36-rule-b-20261021.json'
DISPLAY = (1024,576)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    assert not p.exists(), f'Preserve existing artifact: {p}'
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def record(c, suffix):
    return ROOT/(c['record_prefix']+suffix)


def image_path(c, t):
    return ROOT/'data/processed/davis2016-v2/eval'/c['scene']/'rgb'/f'{t:05d}.jpg'


def input_rgb(c, t):
    a=cv2.imread(str(image_path(c,t)))
    assert a is not None and a.shape==(1080,1920,3)
    return cv2.cvtColor(a,cv2.COLOR_BGR2RGB)


def quantiles(a):
    if not a.size:
        return None
    return dict(p01_p10_p25_p50_p75_p90_p95_p99=np.quantile(a,[.01,.1,.25,.5,.75,.9,.95,.99]).tolist(),
                minimum=float(a.min()),maximum=float(a.max()),pixels=int(a.size))


def native_box(box):
    x,y,w,h=box
    a,b=math.floor(x*1.875),math.floor(y*1.875)
    return [a,b,math.ceil((x+w)*1.875)-a,math.ceil((y+h)*1.875)-b]


def display_box(box):
    if box is None:
        return None
    x,y,w,h=box
    a,b=math.floor(x/1.875),math.floor(y/1.875)
    return [a,b,math.ceil((x+w)/1.875)-a,math.ceil((y+h)/1.875)-b]


def match(gray, template, box, radius):
    x,y,w,h=box;rx,ry=radius
    x0,y0=max(0,x-rx),max(0,y-ry)
    x1,y1=min(gray.shape[1],x+w+rx),min(gray.shape[0],y+h+ry)
    scores=cv2.matchTemplate(gray[y0:y1,x0:x1],template,cv2.TM_CCOEFF_NORMED)
    _,score,_,(px,py)=cv2.minMaxLoc(scores)
    other=scores.copy();other[max(0,py-3):py+4,max(0,px-3):px+4]=-1
    return [x0+px,y0+py,w,h],float(score),float(score-other.max())


def locate(c):
    # Localization never uses candidate parameters, including on a frozen-rule replay.
    old=read(ROOT/c['reference']['anchor_config'])['B36']
    anchors={t:native_box(old['ROIs_display_xywh'][str(t)]['neck_core']) for t in c['reference']['anchor_frames']}
    policy=c['reference'];out=ROOT/c['output'];out.mkdir(parents=True,exist_ok=True)
    refs={t:cv2.cvtColor(input_rgb(c,t),cv2.COLOR_RGB2GRAY) for t in anchors}
    rows=[]
    for t in range(25):
        anchor=min(anchors,key=lambda f:(abs(f-t),f))
        x,y,w,h=anchors[anchor];reason=[]
        if t in anchors:
            box=anchors[t];checks=dict(NCC=None,peak_margin=None,roundtrip_error_native=None)
            method='existing input-RGB neck anchor'
        else:
            gray=cv2.cvtColor(input_rgb(c,t),cv2.COLOR_RGB2GRAY)
            template=refs[anchor][y:y+h,x:x+w]
            box,score,margin=match(gray,template,anchors[anchor],policy['search_radius_native_xy'])
            xx,yy,ww,hh=box
            back,reverse_score,_=match(refs[anchor],gray[yy:yy+hh,xx:xx+ww],box,policy['search_radius_native_xy'])
            error=float(np.hypot(back[0]-x,back[1]-y))
            if score<policy['min_NCC']:reason.append('NCC below fixed0.8')
            if margin<policy['min_peak_margin_outside_3_native_pixels']:reason.append('ambiguous RGB match peak')
            if error>policy['max_roundtrip_error_native_pixels']:reason.append('RGB roundtrip error above3 native pixels')
            checks=dict(NCC=score,peak_margin=margin,reverse_NCC=reverse_score,roundtrip_error_native=error)
            method='nearest-anchor independent native RGB translation'
        rows.append(dict(frame_id=t,anchor_frame=anchor,method=method,attempted_native_xywh=box,
            native_xywh=box if not reason else None,display_xywh=display_box(box) if not reason else None,
            localization_valid=not reason,checks=checks,missing_reason='; '.join(reason)))
    result=dict(created_UTC=now(),scene=c['scene'],policy=policy,anchors_native_xywh=anchors,
        rows=rows,selection_source='RGB only; disparity and model predictions not read',
        visual_judgment='No output quality judgement; executor localization still needs user confirmation.')
    save(record(c,'-localization.json'),result)
    sheet=Image.new('RGB',(5*300,5*210),'white');draw=ImageDraw.Draw(sheet)
    for row in rows:
        t=row['frame_id'];a=Image.fromarray(input_rgb(c,t));x,y,w,h=row['attempted_native_xywh']
        d=ImageDraw.Draw(a);d.rectangle((x,y,x+w,y+h),outline='lime' if row['localization_valid'] else 'red',width=2)
        cx,cy=x+w//2,y+h//2
        a=a.crop((cx-140,cy-90,cx+140,cy+90)).resize((300,193),Image.Resampling.NEAREST)
        left,top=(t%5)*300,(t//5)*210;sheet.paste(a,(left,top+17))
        draw.text((left+3,top+2),f"f{t:02d} anchor{row['anchor_frame']} {'valid' if row['localization_valid'] else 'MISSING'}",fill='black')
    sheet.save(out/'RGB-neck-localization-contact.png')
    print('LOCALIZED',sum(r['localization_valid'] for r in rows),'of25',flush=True)


def original_geometry(c,t):
    folder=ROOT/'data/processed/davis2016-v2/eval'/c['scene']
    geometry=read(ROOT/'data/processed/davis2016-v2/sequences'/c['scene']/'geometry.json')
    row=geometry['frames'][t];assert row['frame']==t
    assert float(Path(row['file']).stem.split('_zf_')[1])==row['focus']
    q=np.asarray(Image.open(folder/'disparity'/row['file']))
    labels=np.asarray(Image.open(ROOT/'data/raw/DAVIS-2017/DAVIS/Annotations/Full-Resolution'/c['scene']/f'{t:05d}.png'))
    prepared=np.asarray(Image.open(folder/'mask'/f'{t:05d}.png'))!=0
    assert q.shape==labels.shape==(1080,1920) and np.array_equal(labels!=0,prepared)
    return q,labels!=0,float(row['focus'])


def reference(c,row,q,union):
    take=np.zeros_like(union);reason=row['missing_reason']
    if row['localization_valid']:
        x,y,w,h=row['native_xywh'];take[y:y+h,x:x+w]=True;take &= union
        count=int(take.sum());fraction=count/(w*h)
        if count<c['reference']['min_valid_native_pixels']:reason='insufficient reference foreground pixels'
        if fraction<c['reference']['min_reference_foreground_fraction']:reason='reference mostly outside original union'
    else:
        count=0;fraction=0.
    value=float(np.median(q[take].astype(np.float32)))/255 if not reason else None
    return take,value,dict(valid=not reason,valid_pixels=count,foreground_fraction=fraction,
        reference_disparity=value,reference_q=None if value is None else value*255,reason=reason)


def distributions(c):
    rows=read(record(c,'-localization.json'))['rows'];result=[]
    for row in rows:
        q,union,focus=original_geometry(c,row['frame_id']);take,r,ref=reference(c,row,q,union)
        d=q.astype(np.float32)/255
        result.append(dict(frame_id=row['frame_id'],reference=ref,focus=focus,subject_pixels=int(union.sum()),
            reference_q_distribution=quantiles(q[take]),subject_d_distribution=quantiles(d[union]),
            subject_abs_d_minus_reference=None if r is None else quantiles(np.abs(d[union]-r)),
            subject_signed_d_minus_reference=None if r is None else quantiles(d[union]-r)))
    save(record(c,'-input-distributions.json'),dict(created_UTC=now(),candidate_rule_selected=False,rows=result,
        scope='Original input geometry only; no candidate masks or model outputs evaluated.'))
    for row in result:
        print(row['frame_id'],row['reference'], 'e_quantiles',None if row['subject_abs_d_minus_reference'] is None else row['subject_abs_d_minus_reference']['p01_p10_p25_p50_p75_p90_p95_p99'],flush=True)


def select(c,q,union,r):
    rule=c['candidate_rule'];raw=np.zeros_like(union);chosen=raw.copy();components=[]
    if r is None:
        return raw,chosen,components,'invalid neck reference'
    raw=union & (np.abs(q.astype(np.float32)/255-r)>rule['absolute_disparity_threshold'])
    n,labels,stats,_=cv2.connectedComponentsWithStats(raw.astype(np.uint8),connectivity=rule['connectivity'])
    ids=[]
    for i in range(1,n):
        x,y,w,h,area=[int(v) for v in stats[i]]
        qualifies=area>=rule['min_component_native_pixels']
        components.append(dict(component=i,native_xywh=[x,y,w,h],area=area,passes_min_area=qualifies))
        if qualifies:ids.append(i)
    eligible=np.isin(labels,ids) if ids else np.zeros_like(union)
    fraction=float(eligible.sum()/union.sum())
    if not raw.any():reason='empty raw candidates'
    elif not ids:reason='all components below64 native pixels'
    elif fraction>rule['max_total_subject_fraction']:reason='eligible total exceeds25% of subject; entire frame skipped'
    else:reason='';chosen=eligible
    return raw,chosen,components,reason


def color(values,minimum,maximum):
    scaled=np.uint8(np.round(np.clip((values-minimum)/(maximum-minimum),0,1)*255))
    return cv2.cvtColor(cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)


def difference_color(v,limit):
    fraction=np.clip(np.abs(v)/limit,0,1)
    a=np.full(v.shape+(3,),255,np.float32)
    a[...,1]=255*(1-fraction)
    a[...,0]=np.where(v<0,255*(1-fraction),255)
    a[...,2]=np.where(v>0,255*(1-fraction),255)
    return np.uint8(np.round(a))


def overlay(rgb,regions):
    a=rgb.copy()
    for take,tint in regions:
        a[take]=np.uint8(np.round(.55*a[take]+.45*np.asarray(tint)))
    return a


def link(p,source):
    p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists()
    p.symlink_to(os.path.relpath(source,p.parent))


def materials(c):
    import torch
    import torchvision.transforms.functional as F
    torch.set_num_threads(4)
    assert c['candidate_rule'] is not None and not c['inference_started'] and not c['user_selection_confirmed']
    localization=read(record(c,'-localization.json'));distribution=read(record(c,'-input-distributions.json'))
    out=ROOT/c['output'];page=out/'B36';assert not page.exists();page.mkdir()
    save(out/'selection-policy-before-materials.json',dict(saved_UTC=now(),config=c,
        note='One input-only candidate proposal; no model output evaluated and no threshold sweep.'))
    split=read(ROOT/'03_实验记录/records/any2bokeh-failure-analysis-20261009-source-split.json')['rows']
    assert any(x['dataset']=='DAVIS' and x['scene_id']==c['scene'] and x['split']=='exploration' for x in split)
    names=['Input_RGB','VDA','Neck_reference','Raw_candidates','Selected_candidates','Original_CoC','Proposed_CoC','CoC_delta']
    for name in names:(page/'originals'/name).mkdir(parents=True)
    for name in ['RGB','Raw_VDA','DAVIS_labels','Reference_mask','Raw_candidate_mask','Selected_mask']:(page/'native'/name).mkdir(parents=True)
    # Compare CPU preprocessing against the already-verified author-loader input.
    old=ROOT/'outputs/any2bokeh-failure-analysis-20261009/B36-MPI-invariance-20261021'
    original= torch.load(old/'prepared-inputs.pt',weights_only=True,map_location='cpu')['base']['coc']
    source_groups=read(old/'prepared.json')['audit']['input_groups']
    positions={t:next((g,j) for g,ids in enumerate(source_groups) for j,source in enumerate(ids) if source==t) for t in range(25)}
    rows=[];selected_previous=None;raw_previous=None;changes=[]
    video_frames=out/'video-frames';video_frames.mkdir()
    sheet=Image.new('RGB',(5*320,5*210),'white');sheet_draw=ImageDraw.Draw(sheet)
    for loc in localization['rows']:
        t=loc['frame_id'];q,union,focus=original_geometry(c,t)
        take,r,ref=reference(c,loc,q,union);raw,chosen,components,reason=select(c,q,union,r)
        if not ref['valid']:reason=ref['reason']
        assert not chosen[~union].any()
        q_new=q.astype(np.float32)
        if chosen.any():q_new[chosen]=np.float32(ref['reference_q'])
        qt=torch.from_numpy(q.astype(np.float32));qn=torch.from_numpy(q_new)
        before=(qt-focus*255).abs()/255;after=(qn-focus*255).abs()/255
        assert torch.equal(before[~torch.from_numpy(chosen)],after[~torch.from_numpy(chosen)])
        # Exact author order: abs on native grid, resize with torchvision default
        # bilinear+antialias, then three-channel Normalize(.5,.5).
        before_cond=F.normalize(F.resize(before.repeat(3,1,1),(576,1024)),[.5]*3,[.5]*3)
        after_cond=F.normalize(F.resize(after.repeat(3,1,1),(576,1024)),[.5]*3,[.5]*3)
        g,j=positions[t];assert torch.equal(before_cond,original[g,j]),f'Author CoC preprocessing drift frame{t}'
        delta=after_cond-before_cond
        assert torch.isfinite(before_cond).all() and torch.isfinite(after_cond).all()
        b=((before_cond[0]+1)/2).numpy()*16;a=((after_cond[0]+1)/2).numpy()*16
        rgb=input_rgb(c,t);d=q.astype(np.float32)/255
        higher=(q.astype(np.float32)/255>r) if r is not None else np.zeros_like(union)
        ref_rgb=overlay(rgb,[(take,[0,255,80])])
        if not ref['valid']:
            x,y,w,h=loc['attempted_native_xywh'];cv2.rectangle(ref_rgb,(x,y),(x+w,y+h),(255,0,0),3)
        native={
            'Input_RGB':rgb,'VDA':color(d,0,1),'Neck_reference':ref_rgb,
            'Raw_candidates':overlay(rgb,[(raw&higher,[255,0,180]),(raw&~higher,[0,220,255])]),
            'Selected_candidates':overlay(rgb,[(chosen&higher,[255,0,180]),(chosen&~higher,[0,220,255])])}
        shown={k:cv2.resize(v,DISPLAY,interpolation=cv2.INTER_NEAREST if k=='VDA' else cv2.INTER_LINEAR) for k,v in native.items()}
        shown.update(Original_CoC=color(b,0,16),Proposed_CoC=color(a,0,16),CoC_delta=difference_color(a-b,16))
        for name,array in shown.items():Image.fromarray(array).save(page/'originals'/name/f'{t:05d}.png')
        link(page/'native/RGB'/f'{t:05d}.jpg',image_path(c,t))
        geo=read(ROOT/'data/processed/davis2016-v2/sequences'/c['scene']/'geometry.json')['frames'][t]
        source=ROOT/'data/processed/davis2016-v2/eval'/c['scene']
        link(page/'native/Raw_VDA'/f'{t:05d}.png',source/'disparity'/geo['file'])
        link(page/'native/DAVIS_labels'/f'{t:05d}.png',ROOT/'data/raw/DAVIS-2017/DAVIS/Annotations/Full-Resolution'/c['scene']/f'{t:05d}.png')
        for name,array in [('Reference_mask',take),('Raw_candidate_mask',raw),('Selected_mask',chosen)]:
            Image.fromarray(np.uint8(array)*255).save(page/'native'/name/f'{t:05d}.png')
        union_shown=cv2.resize(union.astype(np.uint8),DISPLAY,interpolation=cv2.INTER_NEAREST)>0
        row=dict(frame_id=t,reference_valid_pixels=ref['valid_pixels'],reference_disparity=r,reference_q=ref['reference_q'],
            reference_native_xywh=loc['native_xywh'],reference_display_xywh=loc['display_xywh'],reference_method=loc['method'],
            anchor_frame=loc['anchor_frame'],reference_valid=ref['valid'],reference_foreground_fraction=ref['foreground_fraction'],
            threshold=c['candidate_rule']['absolute_disparity_threshold'],coordinate_system='native1920x1080 q/255',
            focus=focus,K=16,subject_pixels=int(union.sum()),raw_candidate_area=int(raw.sum()),
            component_count=len(components),size_qualified_component_count=sum(x['passes_min_area'] for x in components),
            eligible_area=sum(x['area'] for x in components if x['passes_min_area']),selected_area=int(chosen.sum()),
            selected_high_disparity_pixels=int((chosen&higher).sum()),selected_low_disparity_pixels=int((chosen&~higher).sum()),
            modification_fraction_subject=float(chosen.sum()/union.sum()),modification_fraction_image=float(chosen.mean()),
            candidate_evaluated=ref['valid'],missing_or_skip_reason=reason,components=components,
            native_original_CoC_selected=quantiles(before.numpy()[chosen]),native_proposed_CoC_selected=quantiles(after.numpy()[chosen]),
            condition_changed_elements=int(torch.count_nonzero(delta)),condition_max_abs_change=float(delta.abs().max()),
            condition_mean_abs_change=float(delta.abs().mean()),
            resized_condition_changed_pixels_outside_nearest_union=int(((a!=b)&~union_shown).sum()),
            native_background_changed_pixels=0,original_author_CoC_bitwise_equal=True)
        if not ref['valid']:
            for key in ['raw_candidate_area','component_count','size_qualified_component_count','eligible_area']:
                row[key]=None
        rows.append(row)
        if selected_previous is not None:
            u=int((chosen|selected_previous).sum());ru=int((raw|raw_previous).sum())
            changes.append(dict(transition=f'{t-1}->{t}',selected_symmetric_difference_pixels=int((chosen^selected_previous).sum()),
                selected_union_pixels=u,selected_change_fraction_of_union=None if not u else float((chosen^selected_previous).sum()/u),
                raw_change_fraction_of_union=None if not ru else float((raw^raw_previous).sum()/ru),
                limitation='Image-coordinate region change, not motion compensated and not a flicker metric.'))
        selected_previous=chosen;raw_previous=raw
        tile=Image.fromarray(shown['Selected_candidates']).resize((320,180),Image.Resampling.BILINEAR)
        x0,y0=(t%5)*320,(t//5)*210;sheet.paste(tile,(x0,y0+30))
        sheet_draw.text((x0+3,y0+3),f'f{t:02d} '+('SKIP' if reason else f'{chosen.sum()/union.sum():.1%} subject'),fill='black')
        combined=np.concatenate([shown['Input_RGB'],shown['Selected_candidates']],axis=1)
        cv2.putText(combined,f'frame {t:02d} | RGB / proposed selection | '+('SKIP' if reason else 'candidate only'),(15,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        Image.fromarray(combined).save(video_frames/f'{t:05d}.png')
        print('MATERIAL',t,'raw',int(raw.sum()),'selected',int(chosen.sum()),reason,flush=True)
    sheet.save(page/'selection-contact.png')
    csv_fields=[k for k in rows[0] if k not in ['components','native_original_CoC_selected','native_proposed_CoC_selected']]
    with record(c,'-frames.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,csv_fields,lineterminator='\n');writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(row[k]) if isinstance(row[k],list) else row[k] for k in csv_fields})
    frame_record=dict(created_UTC=now(),config=c,rows=rows,region_changes=changes,
        status='CPU选区生成完成，观看材料待打包',user_selection_confirmed=False,inference_started=False,
        limits=['Reference localization is RGB-anchor-assisted; this is not a fully automatic method.',
            'Large discrepancy marks a candidate, not a true depth error. Neck disparity is a diagnostic reference, not geometric truth.',
            'Native masks never change background. Author resize/antialias may spread condition differences across the projected mask boundary; recorded separately.',
            'No temporal smoothing/hysteresis; invalid frame21 and fixed area skips are deliberately exposed.',
            'No ATB, depth, optical-flow or other neural inference; no output quality claim.'])
    save(record(c,'-selection.json'),frame_record)
    package(c)


def package(c):
    import imageio_ffmpeg
    out=ROOT/c['output'];page=out/'B36';video_frames=out/'video-frames'
    assert not c['user_selection_confirmed'] and not c['inference_started']
    assert not (page/'input-selection.mp4').exists(), 'Packaging already complete; preserve existing materials'
    result=read(record(c,'-selection.json'));rows=result['rows']
    assert len(rows)==25 and all((video_frames/f'{t:05d}.png').is_file() for t in range(25))
    # Preserve an absent measurement as null, not an observed zero candidate count.
    for row in rows:
        row['candidate_evaluated']=row['reference_valid']
        if not row['reference_valid']:
            for key in ['raw_candidate_area','component_count','size_qualified_component_count','eligible_area']:
                row[key]=None
    fields=[k for k in rows[0] if k not in ['components','native_original_CoC_selected','native_proposed_CoC_selected']]
    with record(c,'-frames.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(row[k]) if isinstance(row[k],list) else row[k] for k in fields})
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-framerate','20','-i',str(video_frames/'%05d.png'),
        '-frames:v','25','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-threads','2',str(page/'input-selection.mp4')],check=True)
    # Fixed scalar legends, independent of per-frame ranges.
    legend=Image.new('RGB',(900,200),'white');draw=ImageDraw.Draw(legend)
    scalar=np.tile(np.linspace(0,1,640,dtype=np.float32),(24,1))
    for top,title,array in [(0,'VDA q/255: 0                                         1',color(scalar,0,1)),
                            (64,'CoC x K: 0                                           16',color(scalar*16,0,16)),
                            (128,'Delta: -16 (blue)          0 (white)            +16 (red)',difference_color(scalar*32-16,16))]:
        draw.text((8,top+2),title,fill='black');legend.paste(Image.fromarray(array),(8,top+22))
    legend.save(page/'fixed-color-legends.png')
    names=['Input_RGB','VDA','Neck_reference','Raw_candidates','Selected_candidates','Original_CoC','Proposed_CoC','CoC_delta']
    make_page(c,rows,names)
    result.update(status='选区材料已准备，等待用户确认',
        packaging=dict(ffmpeg=ffmpeg,ffmpeg_version=imageio_ffmpeg.get_ffmpeg_version(),fps=20,frames=25,codec='libx264',crf=18,pixel_format='yuv420p',threads=2),
        original_input_distribution_preceded_rule_choice=read(record(c,'-input-distributions.json'))['created_UTC']<c['candidate_rule']['saved_UTC'])
    record(c,'-selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('PACKAGED',page,flush=True)


def make_page(c,rows,names):
    sys.path.insert(0,str(ROOT/'src'))
    from review.any2bokeh_pages import viewer_fragment,render_review_form,payload
    out=ROOT/c['output'];page=out/'B36'
    labels=['原RGB（显示网格）','原VDA q/255：固定0–1色标','同帧颈部参考：绿；红框为无效定位尝试',
        '原始双侧候选：粉=较高视差，青=较低视差','拟采用选区（跳过帧为空）',
        '原CoC×16：固定0–16色标','拟修正CoC×16：固定0–16色标','差值：拟修正−原CoC×16，固定±16']
    item=dict(sample_id='B36',dataset='DAVIS',scene_id=c['scene'],source_group='DAVIS:mallard-water',split='exploration',
        package='B36-rule-b-selection',review_url='../review.html',frame_jumps=[0,6,12,18,21,24],initial_frame=0,
        panels=[dict(label=label,prefix=f'originals/{name}/') for label,name in zip(labels,names)],
        tracked_rois=[dict(id='neck',name='RGB定位颈部参考，待用户确认',xywh_by_frame={str(r['frame_id']):r['reference_display_xywh'] if r['reference_valid'] else None for r in rows})])
    fragment=viewer_fragment(item)
    columns=''.join(f'<th>{x}</th>' for x in ['帧','参考像素','r=q/255','原候选/保留面积','修改/主体','跳过原因','原网格入口'])
    table=[]
    for r in rows:
        t=r['frame_id'];ref='缺失' if r['reference_disparity'] is None else f"{r['reference_disparity']:.6f}"
        links=' · '.join(f'<a href="native/{name}/{t:05d}.{ext}">{label}</a>' for name,ext,label in [('RGB','jpg','原RGB'),('Raw_VDA','png','原q'),('DAVIS_labels','png','原类别标注'),('Reference_mask','png','参考mask'),('Raw_candidate_mask','png','原候选'),('Selected_mask','png','保留选区')])
        area='未测' if r['raw_candidate_area'] is None else str(r['raw_candidate_area'])
        table.append(f'<tr><td>{t}</td><td>{r["reference_valid_pixels"]}</td><td>{ref}</td><td>{area}/{r["selected_area"]}</td><td>{r["modification_fraction_subject"]:.2%}</td><td>{r["missing_or_skip_reason"] or "无"}</td><td>{links}</td></tr>')
    text='''<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>B36方案b：输入选区确认</title>
<style>body{font-family:system-ui;margin:24px;background:#f6f9ff}table{border-collapse:collapse}td,th{border:1px solid #b4c7df;padding:5px}video{width:min(100%,1200px)}.atb-panel-grid{grid-template-columns:repeat(2,minmax(0,1fr))}</style>
<h1>B36方案b：输入选区材料，等待用户确认</h1>
<p>本页只有原输入、规则候选及拟修正CoC，没有新ATB输出。请先确认颈部参考和选区；确认前不运行配对推理。差异大只是候选，不代表真实深度错误，也不能保证选到完整鸭头。</p>
<p>双侧 |d−同帧颈部中位数| &gt;0.10；原DAVIS ID≠0主体内8连通，至少64原网格像素；所有合格区域总面积超过主体25%则整帧跳过。无头部搜索限制、形态学、平滑或迟滞。颈部框复用5个RGB关键帧，其余独立CPU配准；无效帧保持原条件。</p>
<p>粉色=高于参考，青色=低于参考；绿色=颈部参考有效像素。候选为数值筛选，不是人工错误标注。原RGB为1920×1080 JPEG；查看器1024×576，RGB和覆盖图INTER_LINEAR，VDA固定色标后最近邻显示。查看器100%指1024显示网格；下表原网格入口保留1920×1080像素与原类别索引。</p>
<p>CoC先在原网格按 |q−255focus|/255 构造，再按作者顺序双线性/antialias缩放并Normalize(.5,.5)；图示为反归一化后的CoC×K（K16），固定0–16，非实测输出模糊半径。拟修正视差采用同帧颈部参考，不把CoC强制设0。差值色标固定−16至+16：蓝=减少，白=0，红=增加；不逐帧归一化。</p>
<p>原网格选区不含背景；resize/antialias可能把条件变化扩展至投影边缘周围，统计另列。区域变化未运动补偿，不等于闪烁。<a href="../review.html">填写选区反馈（独立保存key）</a> · <a href="../frames.csv">逐帧CSV</a></p>
<video controls preload="metadata" src="input-selection.mp4"></video><p>输入＋拟采用选区展示视频：25帧，20fps，原速1.25秒；libx264 CRF18/yuv420p，仅为观看派生材料，精确定位使用下方PNG。</p>'''
    text+='<img src="fixed-color-legends.png" alt="所有帧固定色标，未按帧归一化" style="max-width:100%">'+fragment+'<h2>25帧选区概览</h2><a href="selection-contact.png"><img src="selection-contact.png" style="max-width:100%"></a><h2>逐帧记录及原网格入口</h2><table><tr>'+columns+'</tr>'+''.join(table)+'</table>'
    save(out/'viewer-item.json',item)
    (page/'index.html').write_text(text)
    form=render_review_form([item],package='B1',store_key=c['review_storage_key'],title='B36方案b：颈部参考与候选选区确认',
        notice='本阶段仅确认输入定位及选区。没有新模型输出，不是视觉质量验收；选择确认需由用户明确给出，不自动启动推理。')
    start=form.index('<script id="atb-review-config" type="application/json">')+len('<script id="atb-review-config" type="application/json">');end=form.index('</script>',start)
    fc=json.loads(form[start:end]);fc['package']=item['package'];fc['fields']=['frame_id_or_transition_id','ROI_xywh','actual_view_mode','neck_reference_feedback','candidate_selection_feedback','selection_confirmation','notes','actual_playback_rate']
    fc['labels']=['帧号或transition','ROI x,y,w,h（1024显示网格）','实际观看方式','颈部参考位置反馈','候选选区反馈（漏选/多选等）','选区是否确认（由用户填写）','补充说明','实际视频倍速'];fc['visual']=[]
    form=form[:start]+payload(fc)+form[end:];(out/'review.html').write_text(form)
    save(record(c,'-user-review-blank.json'),dict(sample_id='B36',scene_id=c['scene'],status='待用户确认',
        neck_reference_feedback='',candidate_selection_feedback='',selection_confirmation='',notes='',
        storage_key=c['review_storage_key'],user_selection_confirmed=False))
    link(out/'frames.csv',record(c,'-frames.csv'))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>B36方案b：选区材料已准备，等待用户确认</h1><a href="B36/">打开25帧同步查看器</a> · <a href="review.html">选区反馈表</a> · <a href="frames.csv">逐帧CSV</a><p>没有启动ATB推理。</p>')


def verify(c):
    import imageio.v2 as imageio
    out=ROOT/c['output'];page=out/'B36';result=read(record(c,'-selection.json'))
    assert result['status']=='选区材料已准备，等待用户确认' and not result['inference_started'] and not result['user_selection_confirmed']
    original_native=ROOT/'data/processed/davis2016-v2/eval'/c['scene']
    PNGs=0
    for row in result['rows']:
        t=row['frame_id'];q,union,focus=original_geometry(c,t)
        take=np.asarray(Image.open(page/'native/Reference_mask'/f'{t:05d}.png'))!=0
        raw=np.asarray(Image.open(page/'native/Raw_candidate_mask'/f'{t:05d}.png'))!=0
        selected=np.asarray(Image.open(page/'native/Selected_mask'/f'{t:05d}.png'))!=0
        assert take.shape==raw.shape==selected.shape==(1080,1920)
        assert not (take&~union).any() and not (raw&~union).any() and not (selected&~raw).any()
        assert take.sum()==row['reference_valid_pixels'] and selected.sum()==row['selected_area']
        if row['reference_valid']:
            assert np.median(q[take])/255==row['reference_disparity']
            expected=union&(np.abs(q.astype(np.float32)/255-row['reference_disparity'])>c['candidate_rule']['absolute_disparity_threshold'])
            assert np.array_equal(raw,expected)
        else:
            assert row['candidate_evaluated'] is False and row['raw_candidate_area'] is None
        if row['missing_or_skip_reason']:
            assert not selected.any() and row['condition_changed_elements']==0
            assert (page/'originals/Original_CoC'/f'{t:05d}.png').read_bytes()==(page/'originals/Proposed_CoC'/f'{t:05d}.png').read_bytes()
        assert row['original_author_CoC_bitwise_equal'] and row['native_background_changed_pixels']==0
        for folder in (page/'originals').iterdir():
            with Image.open(folder/f'{t:05d}.png') as im:
                im.verify();assert im.mode=='RGB' and im.size==DISPLAY;PNGs+=1
        with Image.open(page/'originals/Input_RGB'/f'{t:05d}.png') as im:
            assert np.array_equal(np.asarray(im),cv2.resize(input_rgb(c,t),DISPLAY,interpolation=cv2.INTER_LINEAR))
        assert (page/'native/RGB'/f'{t:05d}.jpg').resolve()==image_path(c,t).resolve()
    video=imageio.get_reader(page/'input-selection.mp4',format='FFMPEG');meta=video.get_meta_data()
    count=0
    for im in video:
        assert im.shape==(576,2048,3) and im.dtype==np.uint8;count+=1
    video.close();assert count==25 and meta['fps']==20
    assert not (out/'predictions').exists()
    checks=dict(checked_UTC=now(),frames=25,display_panel_PNGs=PNGs,native_masks=75,
        reference_medians_verified=True,candidate_rule_checked_on_native_grid=True,native_background_unmodified=True,
        skipped_conditions_and_images_unchanged=True,all_original_CoC_bitwise_matches_author_loader=True,
        video=dict(frames=count,fps=meta['fps'],size=[2048,576]),
        inference_started=False,neural_inference_runs=0,user_selection_confirmed=False,
        rule_revisions=1,threshold_sweeps=0,visual_quality_judgments=0)
    save(out/'program-checks.json',checks);print('PROGRAM_CHECKS',json.dumps(checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['locate','distributions','materials','package','verify'])
    args=parser.parse_args();cv2.setNumThreads(1);c=read(CFG)
    globals()[args.stage](c)
