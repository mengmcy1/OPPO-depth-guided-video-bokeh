"""B19 input evidence first, then input-only second-batch sampling and materials.

No model imports or new neural metrics. Human visual fields remain pending.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from analyze_any2bokeh_failures import load, save, csv_save, image, link_originals, video_encode, contact_sheet, STYLE
from review.any2bokeh_pages import viewer_fragment, render_review_form, VISUAL, ASSETS

PENDING='待用户复核'


def now():return datetime.now(timezone.utc).isoformat()


def record(c,suffix):return c['record_prefix']+suffix


def write_new(path,data):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.read_bytes()==data, f'Preserve existing content: {path}'
    else:path.write_bytes(data)


def page(path,text):write_new(path,text.encode('utf-8'))


def png(path,array):
    path=ROOT/path
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(array).save(path)


def link(path,source):
    import os
    path=ROOT/path;source=ROOT/source;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.resolve()==source.resolve()
    else:path.symlink_to(os.path.relpath(source,path.parent))


def install_shared(c):
    """Only changed owned viewer assets, with prior versions preserved."""
    out=ROOT/c['output']
    for source in ASSETS.iterdir():
        dest=out/'review-tools'/source.name
        if dest.exists() and dest.read_bytes()==source.read_bytes():continue
        if dest.exists():
            backup=out/'page-history/20261014/review-tools'/source.name;backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():backup.write_bytes(dest.read_bytes())
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())


def blank(sample,package,record_id):
    pending={k:PENDING for k in VISUAL}
    return dict(**{k:sample[k] for k in ['dataset','scene_id','source_group','split','sample_id']},
        package=package,record_id=record_id,frame_id_or_transition_id='',ROI_xywh='',actual_view_mode='',actual_playback_rate='',
        user_observation='',severity=PENDING,confidence=PENDING,normal_motion_or_reasonable_defocus=PENDING,
        similar_in_input_or_GT=PENDING,normal_control='',notes='',**pending)


def quantiles(a):
    a=np.asarray(a,dtype=np.float64)
    if not a.size:return None
    assert np.isfinite(a).all()
    return dict(n=int(a.size),min=float(a.min()),p10=float(np.quantile(a,.1)),p25=float(np.quantile(a,.25)),
        p50=float(np.median(a)),p75=float(np.quantile(a,.75)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),
        max=float(a.max()),mean=float(a.mean()))


def outline(labels):
    result=np.zeros(labels.shape,np.uint8)
    for label in np.unique(labels):
        if label:result[cv2.morphologyEx((labels==label).astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0]=label
    return result


def overlay(rgb,edges):
    colors={1:(255,170,0),2:(255,0,150),3:(0,255,220)}
    out=rgb.copy()
    for label in np.unique(edges):
        if label:out[edges==label]=colors.get(int(label),(255,255,255))
    return out


def color(a,high):
    values=np.uint8(np.round(np.clip(a/high,0,1)*255))
    return cv2.cvtColor(cv2.applyColorMap(values,cv2.COLORMAP_VIRIDIS),cv2.COLOR_BGR2RGB)


def colorbar(path,high,title):
    a=np.broadcast_to(np.linspace(0,high,1024),(24,1024));im=Image.new('RGB',(1024,65),'white')
    im.paste(Image.fromarray(color(a,high)),(0,18));draw=ImageDraw.Draw(im);draw.text((5,3),title,fill='black')
    for v in [0,.25,.5,.75,1]:draw.text((min(980,int(v*1023)),45),f'{v*high:g}',fill='black')
    dest=ROOT/path
    if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest)


def diagnostic_contact_sheet(path,panels,labels,roi):
    """Small diagnostic crops: integer NEAREST zoom, same coordinates and aspect."""
    if path.exists():return
    x,y,w,h=roi;tile_w,tile_h=384,216;cols=3
    zoom=max(1,min(tile_w//w,tile_h//h))
    rows=(len(labels)+cols-1)//cols
    canvas=Image.new('RGB',(cols*tile_w,3*rows*(tile_h+24)),(248,250,253));draw=ImageDraw.Draw(canvas)
    for j,t in enumerate([0,1,2]):
        for i,(paths,label) in enumerate(zip(panels,labels)):
            im=Image.fromarray(image(paths[t])).crop((x,y,x+w,y+h))
            im=im.resize((w*zoom,h*zoom),Image.Resampling.NEAREST)
            left=(i%cols)*tile_w;top=(j*rows+i//cols)*(tile_h+24)
            canvas.paste(im,(left+(tile_w-im.width)//2,top+24+(tile_h-im.height)//2))
            draw.text((left+4,top+5),f'{label} | f{t:02d} | NN {zoom}x',fill=(15,40,80))
    canvas.save(path)


def native_paths(scene):
    cfg=load('configs/davis-eval-v2.json');base=ROOT/cfg['derived']/'eval'/scene
    rgb=sorted((base/'rgb').glob('*.jpg'));q=sorted((base/'disparity').glob('*.png'))
    annotations=[ROOT/cfg['source']/'Annotations'/cfg['resolution_source']/scene/f'{t:05d}.png' for t in range(25)]
    assert len(rgb)==len(q)==len(annotations)==25
    assert [p.stem for p in rgb]==[f'{t:05d}' for t in range(25)]
    assert [p.stem.split('_zf_')[0] for p in q]==[f'{t:05d}' for t in range(25)]
    return rgb,q,annotations


def b19(c):
    target=record(c,'-B19-input.json')
    if (ROOT/target).exists():print('REUSE_B19_COMPLETE',target,flush=True);return
    install_shared(c);original=load(c['source_config']);sample=next(r for r in load(original['record_prefix']+'-B-samples.json')['rows'] if r['sample_id']=='B19')
    scene=c['B19']['scene_id'];assert sample['scene_id']==scene and sample['split']=='exploration'
    folder=Path(c['output'])/c['B19']['directory'];rgb_paths,q_paths,ann_paths=native_paths(scene)
    g=load(f'data/processed/davis2016-v2/sequences/{scene}/geometry.json');assert g['source_frames']==80
    rows=[];checks=[];panels=['RGB','Original_index_contours','Display_index_contours','VDA_native_nearest','VDA_display_linear','Nominal_defocus','Frozen_ATB']
    panel_labels=['输入RGB（LINEAR显示）','原标注轮廓（原网格，NEAREST显示）','缩放标注轮廓（索引NEAREST）','原始VDA q/255（NEAREST显示）','显示网格VDA d（LINEAR）','名义离焦 K·|d-focus|','原版ATB']
    for t in range(25):
        cached=folder/'frame-stats'/f'{t:05d}.json'
        if (ROOT/cached).exists():prior=load(cached);rows.extend(prior['rows']);checks.append(prior['check']);continue
        src_rgb=np.asarray(Image.open(rgb_paths[t]).convert('RGB'));labels=np.asarray(Image.open(ann_paths[t]));q=np.asarray(Image.open(q_paths[t]))
        h,w=q.shape;assert labels.shape==(h,w) and src_rgb.shape==(h,w,3) and (w,h)==(1920,1080)
        assert q.dtype==labels.dtype==np.uint8
        prepared=np.asarray(Image.open(ROOT/f'data/processed/davis2016-v2/eval/{scene}/mask/{t:05d}.png'))
        assert np.array_equal(prepared,(labels!=0).astype(np.uint8)*255)
        focus=float(q_paths[t].stem.split('_zf_')[1]);K=16
        assert focus==float(f'{q[labels!=0].mean()/255:.5f}')==g['frames'][t]['focus']
        display_rgb=cv2.resize(src_rgb,(1024,576),interpolation=cv2.INTER_LINEAR)
        assert np.array_equal(display_rgb,image(ROOT/c['output']/f'display-input/B19/{t:05d}.png'))
        indices=cv2.resize(labels,(1024,576),interpolation=cv2.INTER_NEAREST)
        d=q.astype(np.float64)/255;display_d=cv2.resize(d,(1024,576),interpolation=cv2.INTER_LINEAR)
        nearest_d=cv2.resize(d,(1024,576),interpolation=cv2.INTER_NEAREST)
        raw_edges=outline(labels);display_edges=outline(indices);projected_edges=cv2.resize(raw_edges,(1024,576),interpolation=cv2.INTER_NEAREST)
        raw_outline=overlay(src_rgb,raw_edges);raw_q_color=color(d,1);raw_nominal_color=color(K*np.abs(d-focus),16)
        products={'Original_index_contours':cv2.resize(raw_outline,(1024,576),interpolation=cv2.INTER_NEAREST),
            'Display_index_contours':overlay(display_rgb,display_edges),'VDA_native_nearest':cv2.resize(raw_q_color,(1024,576),interpolation=cv2.INTER_NEAREST),
            'VDA_display_linear':color(display_d,1),'Nominal_defocus':color(K*np.abs(display_d-focus),16)}
        for key,a in products.items():png(folder/'input/originals'/key/f'{t:05d}.png',a)
        link(folder/'input/originals/RGB'/f'{t:05d}.png',Path(c['output'])/f'display-input/B19/{t:05d}.png')
        link(folder/'input/originals/Frozen_ATB'/f'{t:05d}.png',Path(c['output'])/f'B1/B19/originals/Frozen_ATB/{t:05d}.png')
        for key,a in [('Original_contours',raw_outline),('VDA_color',raw_q_color),('Nominal_defocus',raw_nominal_color)]:png(folder/'raw'/key/f'{t:05d}.png',a)
        for key,p in [('RGB',rgb_paths[t]),('Annotation_indices',ann_paths[t]),('VDA_q',q_paths[t])]:link(folder/'raw'/key/p.name,p.relative_to(ROOT))
        # Keep class indices rather than converting class IDs into display grayscale.
        index_image=Image.fromarray(indices,mode='P');index_image.putpalette(Image.open(ann_paths[t]).getpalette())
        dest=ROOT/folder/'display_annotation_indices'/f'{t:05d}.png';dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():index_image.save(dest)
        memberships=np.stack([cv2.resize((labels==label).astype(np.float32),(1024,576),interpolation=cv2.INTER_LINEAR) for label in np.unique(labels)])
        mixed=np.any((memberships>0)&(memberships<1),axis=0)
        frame_rows=[]
        for r in c['B19']['diagnostic_rois']:
            x,y,rw,rh=r['xywh'];x0=int(math.floor(x*w/1024));y0=int(math.floor(y*h/576));x1=int(math.ceil((x+rw)*w/1024));y1=int(math.ceil((y+rh)*h/576))
            raw_region=d[y0:y1,x0:x1];raw_labels=labels[y0:y1,x0:x1];dd=display_d[y:y+rh,x:x+rw];nn=nearest_d[y:y+rh,x:x+rw];idx=indices[y:y+rh,x:x+rw]
            raw_q=q[y0:y1,x0:x1];bins,counts=np.unique(raw_q,return_counts=True)
            per_label={str(label):dict(pixels=int((raw_labels==label).sum()),disparity=quantiles(raw_region[raw_labels==label]),nominal_defocus=quantiles(K*np.abs(raw_region[raw_labels==label]-focus))) for label in np.unique(labels)}
            row=dict(frame=t,ROI_id=r['id'],ROI_name=r['name'],ROI_status='诊断ROI；用户未确认',display_xywh=r['xywh'],native_xywh=[x0,y0,x1-x0,y1-y0],
                annotation_intersects=bool((raw_labels!=0).any()),native_valid_pixels=int(raw_region.size),display_valid_pixels=int(dd.size),
                native_annotation_ID_counts=dict(Counter(int(v) for v in raw_labels.ravel())),display_annotation_ID_counts=dict(Counter(int(v) for v in idx.ravel())),
                native_disparity=quantiles(raw_region),native_q_histogram={str(int(b)):int(n) for b,n in zip(bins,counts)},display_nearest_disparity=quantiles(nn),display_linear_disparity=quantiles(dd),
                focus=focus,K=K,native_nominal_defocus=quantiles(K*np.abs(raw_region-focus)),display_nominal_defocus=quantiles(K*np.abs(dd-focus)),native_per_annotation_ID=per_label,
                resized_annotation_membership_mixed_pixels=int(mixed[y:y+rh,x:x+rw].sum()),
                original_projected_vs_display_contour_disagreement_pixels=int((projected_edges[y:y+rh,x:x+rw]!=display_edges[y:y+rh,x:x+rw]).sum()),
                alignment_limit='Native ROI is floor/ceil projection of display rectangle; raw/LINEAR/NEAREST grids are separate. Fractional mask membership and contour differences measure resizing only, not physical RGB mixing or proven annotation offset. Static frame0 ROI does not track moving structures.')
            frame_rows.append(row)
        check=dict(frame=t,native_size=[w,h],display_size=[1024,576],palette_ids=sorted(int(v) for v in np.unique(labels)),q_finite=True,prepared_mask_matches_indices=True,focus_matches_frozen_recipe=True)
        save(cached,dict(rows=frame_rows,check=check));rows.extend(frame_rows);checks.append(check)
        print('B19_INPUT_FRAME_COMPLETE',t,flush=True)
    colorbar(folder/'disparity-scale.png',1,'Fixed q/255 scale: normalized predicted disparity, NOT metric depth')
    colorbar(folder/'defocus-scale.png',16,'Fixed nominal K*abs(d-focus): NOT measured output blur radius')
    table=record(c,'-B19-ROI-statistics.csv');csv_save(table,rows)
    # A served copy is derived from the small CSV; original evidence remains in records.
    write_new(folder/'ROI-statistics.csv',(ROOT/table).read_bytes())
    mapping=load(original['record_prefix']+'-B2-panel-mapping.json')['mapping']['DAVIS']
    write_new(folder/'method-mapping.json',(json.dumps(dict(mapping=mapping,scope='Original frozen B2 DAVIS identifiers; separate from panel display; not a claim of perceptual blinding'),ensure_ascii=False,indent=2)+'\n').encode())
    methods=c['B19']['comparison_order'];method_panels=[dict(label='输入RGB',prefix='originals/RGB/')]
    link(folder/'methods/video.mp4',Path(c['output'])/'B2/B19/video.mp4')
    link(folder/'input/video.mp4',Path(c['output'])/'B1/B19/video.mp4')
    for m in methods:
        panel_id=next(key for key,value in mapping.items() if value==m)
        for t in range(25):link(folder/f'methods/originals/{m}/{t:05d}.png',Path(f'outputs/davis-baselines-20260929/{m}/predictions/{scene}/{t:05d}.png'))
        method_panels.append(dict(label=panel_id,prefix=f'originals/{m}/'))
    for t in range(25):link(folder/f'methods/originals/RGB/{t:05d}.png',Path(c['output'])/f'display-input/B19/{t:05d}.png')
    item={k:sample[k] for k in ['sample_id','dataset','scene_id','source_group','split']};item.update(package='B2',diagnostic_rois=c['B19']['diagnostic_rois'])
    captions=dict(zip(panels,panel_labels))
    for sub,pp in [('input',[dict(label=captions[p],prefix=f'originals/{p}/') for p in panels]),('methods',method_panels)]:
        item['panels']=pp
        paths=[sorted((ROOT/folder/sub/p['prefix']).glob('*.png')) for p in pp]
        short=[p['label'] if i else 'Input_RGB' for i,p in enumerate(pp)] if sub=='methods' else [p['prefix'].split('/')[-2] for p in pp]
        suffix='-anonymous' if sub=='methods' else ''
        contact_sheet(ROOT/folder/sub/('adjacent-0-1-2'+suffix+'.png'),paths,short,3,[0,1,2])
        contact_sheet(ROOT/folder/sub/('keyframes'+suffix+'.png'),paths,short,3,[0,6,12,18,24])
        for roi in c['B19']['diagnostic_rois']:
            diagnostic_contact_sheet(ROOT/folder/sub/(roi['id']+'-nearest'+suffix+'-0-1-2.png'),paths,short,roi['xywh'])
        note='视频仅复用首批RGB/ATB时间参照；输入证据由下方PNG显示。' if sub=='input' else '视频复用原B2匿名面板；PNG沿用相同P编号，重点顺序预先固定，其他方法保留，未按结果挑选。映射单独保存于专项根目录method-mapping.json；匿名编号不等于感知盲评。'
        content='<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>B19专项 '+sub+'</title>'+STYLE+'<h1>B19 paragliding-launch 专项 '+sub+'</h1><p>'+note+f'</p><p><a href="../">专项索引/口径/原尺寸入口</a> · <a href="adjacent-0-1-2{suffix}.png">0/1/2接触图</a> · <a href="keyframes{suffix}.png">0/6/12/18/24概览</a></p><video controls preload="metadata" src="video.mp4"></video>'+viewer_fragment(item)
        content+='<p>诊断ROI局部图（整数倍最近邻，保持宽高比，图中标注NN倍率）：'+ ' '.join(f'<a href="{r["id"]}-nearest{suffix}-0-1-2.png">{r["id"]}</a>' for r in c['B19']['diagnostic_rois'])+'</p>'
        page(folder/sub/'index.html',content)
    form=render_review_form([sample],'B2',store_key=c['B19']['review_storage_key'],title='B19专项B2用户复核表',notice='已收到B19第0帧B1初步反馈。本专项等待用户核对准确绳段、持续性及同帧方法差异；所有视觉字段待填写。聊天代录与浏览器记录分开保存，不自动覆盖。')
    questions='<p>请在观察内容中分别记录：哪些绳段保留/模糊或结构损失；方法是否出现相似现象；是否持续；正常位置或时刻。诊断ROI不是用户已确认ROI，可手动改框。</p>'
    page(folder/'review.html',form.replace('<div id="forms">',questions+'<div id="forms">',1))
    blank_row=blank(sample,'B2','B19-B2-special-pending-20261014');save(record(c,'-B19-B2-blank.json'),[blank_row]);csv_save(record(c,'-B19-B2-blank.csv'),[blank_row])
    permissions=load('configs/davis-eval-v2.json')['baseline_rules']
    method_notes={'any2bokeh':'原版WholeVid：原尺寸RGB/VDA/focus/K；CoC/MPI先于作者resize；窗口/overlap保留。',
        'mpib':'原尺寸VDA/focus、RGB到1024；gamma2.2，原视差坐标与原K16。',
        'bokehme':'冻结Normalized、gamma2.2及逐帧span处理；不临时补偿K；外部VDA相同不等于内部有效离焦相同。',
        'deeplens':'DAVIS原生TF1 CPU DepthNet、自身foreground-matched焦点和native aperture0.5；非共享VDA几何等价基线。',
        'drbokeh':'DAVIS Full-public LDF+LaMa及给定VDA；DAVIS标注不作为alpha/layer输入。',
        'bokehdiff':'冻结Normalized-B及原逐帧span处理；随机流与原K16不变。'}
    native_links='<details><summary>全部25帧原尺寸RGB/原标注/原VDA与图层入口（1920×1080）</summary>'
    for t,(rgb,q) in enumerate(zip(rgb_paths,q_paths)):
        native_links+=f'<p>frame {t}: <a href="raw/RGB/{rgb.name}">原RGB</a> · <a href="raw/Annotation_indices/{t:05d}.png">原类别索引</a> · <a href="raw/VDA_q/{q.name}">原q</a> · <a href="raw/Original_contours/{t:05d}.png">原轮廓</a> · <a href="raw/VDA_color/{t:05d}.png">固定色标VDA</a> · <a href="raw/Nominal_defocus/{t:05d}.png">原网格名义离焦</a> · <a href="display_annotation_indices/{t:05d}.png">缩放索引</a></p>'
    native_links+='</details>'
    header='<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>B19输入核查</title>'+STYLE+'<h1>B19输入核查与既有B2对照</h1><p>第0帧伞绳疑似误虚化是用户提出的待验证候选，非已确认几何错误或ATB特有失败。原焦点/K不调整，DAVIS无paired bokeh GT。</p><p><a href="input/">①输入专项查看器</a> · <a href="methods/">②同帧方法对照</a> · <a href="review.html">③空白B2复核表</a> · <a href="ROI-statistics.csv">ROI统计表CSV</a> · <a href="../B2/B19/">既有B2原页</a></p>'
    header+='<p>原RGB/标注/q均1920×1080；显示/预测1024×576。原标注为P模式类别索引0/1/2/3；缩放索引使用NEAREST且保留类别。原轮廓与原VDA色图用NEAREST显示，RGB与诊断显示d用CV2 INTER_LINEAR；float64 q/255先resize，未第二次量化为方法输入。显示网格名义量是K·|resize(d)-focus|；原网格K·|d-focus|另有原尺寸入口及统计。两种运算顺序分开记录，不能冒充ATB实际CoC/MPI条件。原尺寸文件在下方，不锐化。</p><p>q/255是归一化预测视差，不是米制真实深度。K·|d-focus|是名义离焦控制量，不是实测输出模糊半径。标为前景不意味着全部处在同一深度或必须清晰。色标序列固定，ROI不单独拉伸；索引颜色：1橙、2粉、3青，语义最终待用户确认。</p><img src="disparity-scale.png"><img src="defocus-scale.png">'
    header+='<p>统计同时保留native、NEAREST显示、LINEAR显示的分位数及native q直方图，按小框及框内类别分开；不以整片伞绳统计替代具体细线。轮廓差异/混合成员计数仅量化resize效应；不能证明原标注偏移或物理RGB混合。诊断框固定在第0帧，后续不跟踪；准确ROI仍需用户反馈。</p><table><tr><th>诊断ROI</th><th>显示坐标x,y,w,h（非用户确认）</th></tr>'
    header+=''.join(f'<tr><td>{html.escape(r["name"])}</td><td>{r["xywh"]}</td></tr>' for r in c['B19']['diagnostic_rois'])+'</table>'
    header+='<h2>冻结输入权限及适配差异</h2><table><tr><th>方法</th><th>权限/适配</th></tr>'+''.join(f'<tr><td>{m}</td><td>{html.escape(method_notes[m])}</td></tr>' for m in methods)+'</table><p>共享VDA/focus/K不等于内部有效离焦完全相同。多方法相似现象仅支持共享输入可能参与，不能证明深度唯一原因；ATB相对受损也不证明独有机制。“L2”若使用仅指输入核查与位置对应，非因果已证实。</p>'+native_links
    page(folder/'index.html',header)
    save(target,dict(prepared_utc=now(),status='B19_MATERIAL_PREPARED_USER_VISUAL_REVIEW_PENDING',index=str(folder/'index.html'),ROI_statistics=table,
        native_size=[1920,1080],display_size=[1024,576],diagnostic_rois=c['B19']['diagnostic_rois'],ROI_status=c['B19']['roi_status'],
        color_scale=dict(disparity=[0,1],nominal_defocus=[0,16],LUT='VIRIDIS',per_frame_or_crop_normalization=False),
        rows=rows,frame_checks=checks,method_order=methods,method_notes=method_notes,baseline_rules=permissions,original_anonymous_mapping=mapping,
        raw_disparity_normalization=g['normalization'],user_visual_review_completed=False,original_user_feedback_unchanged=True,
        ROI_contact_rendering='Integer NEAREST zoom; aspect ratio preserved; NN multiplier labeled per panel. Whole-frame contact sheets are BILINEAR overview thumbnails.',
        nominal_map_order=dict(native='K*abs(q/255-focus) before any display resizing',display='K*abs(INTER_LINEAR(q/255)-focus), illustrative display diagnostic; not a claim of native ATB conditioning equivalence'),
        browser_storage_key=c['B19']['review_storage_key'],merge_rule='Keep browser exports and chat originals separate; reconcile by dataset/scene/frame or transition/ROI/source and record identity before merge',
        limitations=['No metric-depth truth or paired bokeh GT','Resampling diagnostics are not proof of annotation error, physical mixing or unique depth causation','Diagnostic rectangles are executor-selected, static and unconfirmed; exact rope localization remains user work']))
    print('B19_DELIVERY_SAVED',str(folder/'index.html'),len(rows),flush=True)


def input_proxy(c,scene):
    """Read input RGB and prepared union annotations only; no predictions."""
    destination=Path(c['output'])/c['batch2']['directory']/'input-proxies'/f'{scene}.json'
    if (ROOT/destination).exists():return load(destination)
    rgb,q,annotations=native_paths(scene);detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines=[];boundaries=[];areas=[];changes=[];shape=[];previous_rgb=previous_mask=None
    raw_sizes=[]
    for t in c['batch2']['proxy_frames']:
        im=image(rgb[t]);raw_sizes.append(list(Image.open(rgb[t]).size))
        raw=np.asarray(Image.open(annotations[t]));mask=cv2.resize((raw!=0).astype(np.uint8),(1024,576),interpolation=cv2.INTER_NEAREST)
        edge=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        boundaries.append(float(edge.sum()/max(1,mask.sum())));areas.append(float(mask.mean()))
        segments,widths,_,_=detector.detect(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY))
        if segments is None:score=0.
        else:
            seg=segments.reshape(-1,4);length=np.hypot(seg[:,2]-seg[:,0],seg[:,3]-seg[:,1]);narrow=(length>=40)&(widths.reshape(-1)<=3)
            score=float(length[narrow].sum(dtype=np.float64)/(1024*576))
        lines.append(score)
        normalized=im.astype(np.float32)/255
        if previous_rgb is not None:
            changes.append(float(np.mean(np.abs(normalized-previous_rgb))))
            y0,x0=np.where(previous_mask);y1,x1=np.where(mask)
            dx=float(x1.mean()-x0.mean());dy=float(y1.mean()-y0.mean())
            translated=cv2.warpAffine(previous_mask,np.array([[1,0,dx],[0,1,dy]],np.float32),(1024,576),flags=cv2.INTER_NEAREST,borderValue=0).astype(bool)
            current=mask.astype(bool);shape.append(float(1-(translated&current).sum()/max(1,(translated|current).sum())))
        previous_rgb=normalized;previous_mask=mask
    row=dict(scene_id=scene,source_group='DAVIS:'+scene,K=16,dataset='DAVIS',split='exploration',
        line_like_proxy=float(np.mean(lines)),boundary_proxy=float(np.mean(boundaries)),shape_change_proxy=float(np.mean(shape)),
        input_change_proxy=float(np.mean(changes)),foreground_fraction_proxy=float(np.mean(areas)),
        sampled_frames=c['batch2']['proxy_frames'],raw_RGB_sizes=raw_sizes,semantic_labels='待用户确认',prediction_access=False)
    save(destination,row);print('INPUT_ONLY_PROXY',scene,flush=True);return row


def select_batch(c):
    assert (ROOT/record(c,'-B19-input.json')).exists(),'B19 delivery must be saved first'
    target=record(c,'-batch2-samples.json')
    if (ROOT/target).exists():print('REUSE_BATCH2_LIST',target,flush=True);return
    original=load(c['source_config']);split=load(original['record_prefix']+'-source-split.json')['rows'];first=load(original['record_prefix']+'-B-samples.json')['rows']
    excluded={r['scene_id'] for r in first if r['dataset']=='DAVIS'}
    held={r['scene_id'] for r in split if r['dataset']=='DAVIS' and r['split']=='review'}
    remaining=sorted(r['scene_id'] for r in split if r['dataset']=='DAVIS' and r['split']=='exploration' and r['scene_id'] not in excluded)
    assert len(remaining)==27 and len(held)==15 and not set(remaining)&held
    proxies=[input_proxy(c,s) for s in remaining];selected=[];used=set()
    definitions={'line_like_proxy':'line_proxy','shape_change_proxy':'shape_change_proxy','boundary_proxy':'boundary_proxy'}
    for channel,number in c['batch2']['targeted'].items():
        candidates=sorted([p for p in proxies if p['scene_id'] not in used],key=lambda p:(-p[channel],p['scene_id']))
        for p in candidates[:number]:
            used.add(p['scene_id']);selected.append(dict(p,sampling_channel='input-targeted:'+channel,
                sampling_reason='按预先固定的输入代理排序选取；不宣称已确认细结构/遮挡语义或输出异常',proxy_definition=c['batch2'][definitions[channel]],batch=c['batch2']['name']))
    thresholds={k:float(np.quantile([p[k] for p in proxies],.6)) for k in ['line_like_proxy','boundary_proxy','shape_change_proxy']}
    ordinary=[p for p in proxies if p['scene_id'] not in used and all(p[k]<=v for k,v in thresholds.items())]
    median=float(np.median([p['input_change_proxy'] for p in ordinary])) if ordinary else None
    strata={'lower_input_change':[p for p in ordinary if p['input_change_proxy']<=median],
            'higher_input_change':[p for p in ordinary if p['input_change_proxy']>median]} if ordinary else {'lower_input_change':[],'higher_input_change':[]}
    rng=random.Random(c['batch2']['seed']);pools={}
    for stratum,pool in strata.items():
        pool=sorted(pool,key=lambda p:p['scene_id']);pools[stratum]=[p['scene_id'] for p in pool]
        chosen=rng.sample(pool,min(c['batch2']['random']['per_input_change_stratum'],len(pool)))
        for p in chosen:selected.append(dict(p,sampling_channel='fixed-seed-stratified-random:'+stratum,
            sampling_reason='三个输入结构/形状代理均不高于候选池60分位，再按RGB变化代理分层随机；不是无条件全数据集随机，非人工正常结论',
            proxy_definition=c['batch2']['ordinary_eligibility']+'; '+c['batch2']['strata'],batch=c['batch2']['name']))
    start=max(int(r['sample_id'][1:]) for r in first)+1
    for i,p in enumerate(selected):p['sample_id']=f'B{start+i:02d}'
    assert len({p['scene_id'] for p in selected})==len(selected) and not {p['scene_id'] for p in selected}&(excluded|held)
    result=dict(saved_utc=now(),name=c['batch2']['name'],seed=c['batch2']['seed'],input_only=True,predictions_opened_before_list=False,
        rules=c['batch2'],remaining_pool=remaining,first_batch_excluded=sorted(excluded),heldout_excluded=sorted(held),
        thresholds=thresholds,ordinary_pool=[p['scene_id'] for p in ordinary],input_change_stratum_median=median,random_strata=pools,
        targeted_count=sum(p['sampling_channel'].startswith('input-targeted') for p in selected),random_count=sum(p['sampling_channel'].startswith('fixed-seed') for p in selected),
        source_groups=len({p['source_group'] for p in selected}),source_overlap_with_heldout=0,rows=selected,
        semantic_coverage='All rope/wire/branch/rail/occlusion/motion semantic labels await user confirmation. Proxies can also reflect architecture, pose, contrast or camera movement. No additional data generated.',
        missing_random_slots=max(0,4-sum(p['sampling_channel'].startswith('fixed-seed') for p in selected)),
        disclosure=c['batch2']['disclosure'])
    # Freeze this manifest before accessing any selected ATB outputs.
    save(target,result);csv_save(record(c,'-batch2-samples.csv'),selected);csv_save(record(c,'-batch2-input-proxies.csv'),proxies)
    templates=[dict(blank(p,'B1',p['sample_id']+'-B1-batch2-pending-20261014'),batch=c['batch2']['name'],sampling_channel=p['sampling_channel']) for p in selected]
    save(record(c,'-batch2-B1-blank.json'),templates);csv_save(record(c,'-batch2-B1-blank.csv'),templates)
    print('BATCH2_LIST_SAVED_BEFORE_OUTPUT_ACCESS',len(selected),result['targeted_count'],result['random_count'],flush=True)


def batch_materials(c):
    manifest=load(record(c,'-batch2-samples.json'));target=record(c,'-batch2-materials.json')
    if (ROOT/target).exists():print('REUSE_BATCH2_MATERIALS',target,flush=True);return
    install_shared(c);original=load(c['source_config']);folder=Path(c['output'])/c['batch2']['directory'];entries=[]
    for sample in manifest['rows']:
        sid=sample['scene_id'];code=sample['sample_id'];dest=folder/code;complete=dest/'complete.json'
        if (ROOT/complete).exists():entries.append(load(complete));continue
        rgb_paths,q,annotation=native_paths(sid)
        rgb=dest/'originals/Input_RGB';pred=dest/'originals/Frozen_ATB'
        for t,p in enumerate(rgb_paths):png(rgb/f'{t:05d}.png',image(p))
        predicted=sorted((ROOT/f'outputs/davis-baselines-20260929/any2bokeh/predictions/{sid}').glob('*.png'))
        assert len(predicted)==25 and [p.stem for p in predicted]==[f'{t:05d}' for t in range(25)]
        for p in predicted:
            with Image.open(p) as im:assert im.size==(1024,576) and im.mode=='RGB'
        link_originals(ROOT/pred,predicted)
        pp=[sorted((ROOT/rgb).glob('*.png')),sorted((ROOT/pred).glob('*.png'))];labels=['Input RGB','Frozen ATB']
        video_encode(original,ROOT/dest/'video.mp4',pp,labels,2)
        contact_sheet(ROOT/dest/'contact.png',pp,labels,2,[0,6,12,18,24])
        viewer={k:sample[k] for k in ['sample_id','dataset','scene_id','source_group','split']};viewer.update(package='B1',labels=labels)
        title=f'{code} DAVIS {sid} K16 · 第二批定向补充B1'
        body='<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>'+html.escape(title)+'</title>'+STYLE+'<h1>'+html.escape(title)+'</h1><p>本批受到首批反馈启发；抽样仅使用输入RGB/标注/代理。语义标签及视觉结论待用户确认，不展示输出指标、误差图或预设失败标签。</p><p>来源组 '+sample['source_group']+'；探索组。25帧，20fps，K16，RGB由原JPEG以CV2 INTER_LINEAR显示到1024×576，原ATB预测保留。</p><video controls preload="metadata" src="video.mp4"></video>'+viewer_fragment(viewer)+'<p><a href="contact.png">原5帧概览</a> · <a href="../">本批索引</a></p><img src="contact.png">'
        page(dest/'index.html',body)
        entry=dict(sample_id=code,scene_id=sid,source_group=sample['source_group'],dataset='DAVIS',split='exploration',package='B1',
            index=str(dest/'index.html'),video=str(dest/'video.mp4'),frames=25,K=16,display_size=[1024,576],video_size=[2048,600],source_predictions=str(predicted[0].parent.relative_to(ROOT)),
            original_focus_values=[float(p.stem.split('_zf_')[1]) for p in q],sampling_channel=sample['sampling_channel'],visual_review=PENDING)
        save(complete,entry);entries.append(entry);print('BATCH2_MATERIAL_COMPLETE',code,sid,flush=True)
    form=render_review_form(manifest['rows'],'B1',store_key=c['batch2']['review_storage_key'],title=c['batch2']['name']+'用户复核表',
        notice='独立第二批表单，保留首批key和用户内容。本批受首批观察启发；仅在用户反馈后安排对应B2。浏览器导出与聊天代录暂不自动合并，后续按scene/frame/ROI/来源组及批次核对。',metadata_extra=('batch','sampling_channel'))
    page(folder/'review.html',form)
    body='<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>第二批定向补充B1</title>'+STYLE+'<h1>第二批定向补充B1</h1><p>首批24段清单/反馈保留。本批从剩余27段DAVIS探索视频中按输入条件补充，不是全新独立发现，不使用15段来源隔离复核。</p><p>输入定向'+str(manifest['targeted_count'])+'段＋固定种子分层随机'+str(manifest['random_count'])+'段；随机池按输入代理分层，不是全数据集无条件随机。结构/遮挡/运动的语义覆盖仍待用户确认，输出观感不用于换样。</p><p><a href="review.html">本批独立复核表（旧记录不清空）</a></p>'
    body+=''.join(f'<article><h2><a href="{e["sample_id"]}/">{e["sample_id"]} {e["scene_id"]}</a></h2><p>来源组 {e["source_group"]}；探索。视觉判断：待用户复核。</p></article>' for e in entries)
    page(folder/'index.html',body)
    save(target,dict(prepared_utc=now(),sample_list_saved_utc=manifest['saved_utc'],index=str(folder/'index.html'),items=entries,
        browser_storage_key=c['batch2']['review_storage_key'],first_batch_keys_unchanged=True,chat_and_browser_not_merged=True,
        B2_generated=False,user_visual_review_completed=False,scope=manifest['disclosure']))
    print('BATCH2_ALL_MATERIALS_COMPLETE',len(entries),flush=True)


def verify_materials(c):
    b=load(record(c,'-B19-input.json'));s=load(record(c,'-batch2-samples.json'));m=load(record(c,'-batch2-materials.json'))
    assert len(b['rows'])==150 and len(b['frame_checks'])==25 and b['color_scale']['per_frame_or_crop_normalization'] is False
    if 'HTTP_check' in b:assert b['HTTP_check']['root_directory']==c['output']
    assert all(r['native_disparity']['n']==r['native_valid_pixels']==sum(r['native_q_histogram'].values()) for r in b['rows'])
    first=load(load(c['source_config'])['record_prefix']+'-B-samples.json')['rows']
    original_ids={r['sample_id'] for r in first};original_DAVIS={r['scene_id'] for r in first if r['dataset']=='DAVIS'}
    assert not original_ids&{r['sample_id'] for r in s['rows']}
    assert not original_DAVIS&{r['scene_id'] for r in s['rows']}
    assert not set(s['heldout_excluded'])&{r['scene_id'] for r in s['rows']}
    assert len(s['rows'])==len({r['source_group'] for r in s['rows']})==len(m['items'])
    for path in [record(c,'-B19-B2-blank.json'),record(c,'-batch2-B1-blank.json')]:
        assert all(all(r[k]==PENDING for k in VISUAL) and r['user_observation']=='' for r in load(path))
    videos=[];headers=0
    for e in m['items']:
        folder=(ROOT/e['index']).parent
        for panel in ['Input_RGB','Frozen_ATB']:
            paths=sorted((folder/'originals'/panel).glob('*.png'))
            assert [p.stem for p in paths]==[f'{t:05d}' for t in range(25)]
            for p in paths:
                with Image.open(p) as im:assert im.size==(1024,576) and im.mode=='RGB';im.verify()
                headers+=1
        cap=cv2.VideoCapture(str(ROOT/e['video']));frames=0;fps=cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(600,2048,3) and np.isfinite(im).all();frames+=1
        cap.release();assert frames==25 and fps==20
        videos.append(dict(sample_id=e['sample_id'],frames=frames,fps=fps,size=[2048,600],finite=True))
    result=dict(checked_utc=now(),status='PROGRAM_CHECKS_COMPLETE_BROWSER_CHECK_PENDING',B19_ROI_rows=150,
        B19_color_scale_fixed=True,B19_index_palette_and_focus_checks=True,batch2_samples=len(s['rows']),targeted=s['targeted_count'],random=s['random_count'],
        source_groups=s['source_groups'],heldout_overlap=0,first_batch_overlap=0,PNG_integrity_checks=headers,decoded_videos=videos,
        list_saved_before_materials=s['saved_utc']==m['sample_list_saved_utc'],old_chat_records_not_merged=True,user_visual_fields_pending=True)
    target=record(c,'-verification.json')
    if not (ROOT/target).exists():save(target,result)
    print('FOLLOWUP_PROGRAM_CHECKS_PASSED',len(videos),headers,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['b19','select','materials','verify'])
    parser.add_argument('--config',default='configs/any2bokeh-followup-review-20261014.json');args=parser.parse_args()
    c=load(args.config);cv2.setNumThreads(1)
    if args.command=='b19':b19(c)
    elif args.command=='select':select_batch(c)
    elif args.command=='materials':batch_materials(c)
    else:verify_materials(c)


if __name__=='__main__':main()
