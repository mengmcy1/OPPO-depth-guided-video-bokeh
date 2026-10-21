"""CPU-only diagnosis of B36 revision1 stopping on sources21-24.

No model imports or inference command. Existing thresholds and history remain
read-only; dual-anchor attempts and scalar sensitivity never repair revision1.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import prepare_b36_rule_b as b

ROOT = b.ROOT
CONFIG = ROOT/'configs/any2bokeh-B36-rule-b-tail-20261021.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_save(path, rows):
    assert not path.exists(), path
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def rect(shape, box):
    mask = np.zeros(shape, bool)
    if box is not None:
        x, y, w, h = box
        assert x >= 0 and y >= 0 and x+w <= shape[1] and y+h <= shape[0]
        mask[y:y+h, x:x+w] = True
    return mask


def distribution(values):
    if not values.size:
        return None
    return dict(p10=float(np.quantile(values, .1)), p50=float(np.median(values)),
                p90=float(np.quantile(values, .9)), minimum=float(values.min()),
                maximum=float(values.max()), pixels=int(values.size))


def components(c, q, union, r):
    if r is None:
        return None
    raw, chosen, old, reason = b.select(c, q, union, r)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
    assert len(old) == n-1
    eligible = np.isin(labels, [x['component'] for x in old if x['passes_min_area']])
    higher = q.astype(np.float32)/255 > r
    high, low = eligible & higher, eligible & ~higher
    assert int(eligible.sum()) == int(high.sum()+low.sum())
    return dict(raw=raw, chosen=chosen, eligible=eligible, high=high, low=low,
                higher=higher, labels=labels, components=old, reason=reason,
                raw_high=int((raw & higher).sum()), raw_low=int((raw & ~higher).sum()),
                eligible_high=int(high.sum()), eligible_low=int(low.sum()),
                fraction=float(eligible.sum()/union.sum()))


def rgb_localize(c, old, out):
    """Save all existing-anchor RGB attempts BEFORE extracting their disparities."""
    assert not (out/'dual-anchor-RGB-localization.json').exists()
    refs = {a: cv2.cvtColor(b.input_rgb(c, a), cv2.COLOR_RGB2GRAY)
            for a in [18, 24]}
    rows = []
    p = c['reference']
    for t in range(18, 25):
        gray = cv2.cvtColor(b.input_rgb(c, t), cv2.COLOR_RGB2GRAY)
        for anchor in [18, 24]:
            box = old['anchors_native_xywh'][str(anchor)]
            x, y, w, h = box
            if t == anchor:
                proposal, checks, failures = box, dict(NCC=None, peak_margin=None,
                    roundtrip_error_native=None), []
            else:
                proposal, score, margin = b.match(gray, refs[anchor][y:y+h, x:x+w], box,
                                                p['search_radius_native_xy'])
                xx, yy, ww, hh = proposal
                back, reverse, _ = b.match(refs[anchor], gray[yy:yy+hh, xx:xx+ww], proposal,
                                          p['search_radius_native_xy'])
                error = float(np.hypot(back[0]-x, back[1]-y))
                failures = []
                if score < p['min_NCC']: failures.append('NCC below0.8')
                if margin < p['min_peak_margin_outside_3_native_pixels']: failures.append('ambiguous peak')
                if error > p['max_roundtrip_error_native_pixels']: failures.append('roundtrip above3')
                checks = dict(NCC=score, peak_margin=margin, reverse_NCC=reverse,
                              roundtrip_error_native=error)
            rows.append(dict(frame_id=t, anchor_frame=anchor, native_xywh=proposal,
                             RGB_checks=checks, RGB_matching_passed=not failures,
                             failure_reason='; '.join(failures), anatomical_confirmation='待用户确认'))
    b.save(out/'dual-anchor-RGB-localization.json', dict(created_UTC=b.now(), rows=rows,
        source='Only existing input RGB and old anchors. All attempts saved; no q/output selection.',
        preliminary_inspection='Both fixed anchors were also inspected in a read-only console probe before this structured replay; no parameters/boxes changed or candidates discarded.'))
    return rows


def analyze(cfg):
    c = b.read(ROOT/cfg['selection_config']); out = ROOT/cfg['output']; out.mkdir(exist_ok=True)
    assert not (out/'diagnosis.json').exists()
    oldloc = b.read(b.record(c, '-localization.json'))
    oldrows = b.read(b.record(c, '-selection.json'))['rows']
    ic = b.read(ROOT/cfg['inference_config'])
    roirows = b.read(ROOT/(ic['record_prefix']+'-regions.json'))['frames']
    protected = [ROOT/cfg['selection_config'], b.record(c, '-selection.json'), b.record(c, '-localization.json'),
                 ROOT/(ic['record_prefix']+'.json'), ROOT/(ic['record_prefix']+'-user-review.json')]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    b.save(out/'diagnostic-plan.json', dict(created_UTC=b.now(), config=cfg, protected_records=hashes))
    attempts = rgb_localize(c, oldloc, out)
    rows, ccrows, sensitivity, proposals = [], [], [], []
    native = out/'B36/native'; native.mkdir(parents=True)
    for t in cfg['focus_frames']:
        q, union, focus = b.original_geometry(c, t)
        d = q.astype(np.float32)/255
        old = oldrows[t]; loc = oldloc['rows'][t]
        refmask = rect(q.shape, loc['attempted_native_xywh']) & union
        # Keep frame21 officially invalid; attempted-box stats are separately named.
        r = old['reference_disparity']
        current = components(c, q, union, r)
        head = rect(q.shape, b.native_box(roirows[t]['head_context_xywh']))
        body = rect(q.shape, b.native_box(roirows[t]['body_display_xywh'])) & union
        boundary = union & (cv2.distanceTransform(union.astype(np.uint8), cv2.DIST_L2, 5) <= 5)
        per_frame = []
        if current is not None:
            assert int(current['raw'].sum()) == old['raw_candidate_area']
            assert int(current['eligible'].sum()) == old['eligible_area']
            assert int(current['chosen'].sum()) == old['selected_area']
            for co in current['components']:
                if not co['passes_min_area']: continue
                m = current['labels'] == co['component']
                row = dict(frame_id=t, component_id=co['component'], native_xywh=co['native_xywh'],
                    pixels=co['area'], high_pixels=int((m & current['higher']).sum()),
                    low_pixels=int((m & ~current['higher']).sum()),
                    head_context_overlap=int((m & head).sum()), body_core_overlap=int((m & body).sum()),
                    boundary_5px_overlap=int((m & boundary).sum()),
                    location='Input-defined rectangle/boundary proxies overlap; anatomical location pending user.')
                per_frame.append(row); ccrows.append(row)
            assert sum(x['pixels'] for x in per_frame) == old['eligible_area']
        frame_attempts = []
        for attempt in [a for a in attempts if a['frame_id'] == t]:
            m = rect(q.shape, attempt['native_xywh']) & union
            value = float(np.median(q[m]))/255 if m.any() else None
            _, _, w, h = attempt['native_xywh']
            valid = (attempt['RGB_matching_passed'] and int(m.sum()) >= c['reference']['min_valid_native_pixels']
                     and m.sum()/(w*h) >= c['reference']['min_reference_foreground_fraction'])
            attempt.update(valid_pixels=int(m.sum()), foreground_fraction=float(m.sum()/(w*h)),
                reference_d_distribution=distribution(d[m]), reference_q_median=None if value is None else value*255,
                reference_disparity=value, program_reference_valid=bool(valid), adopted=False)
            frame_attempts.append(attempt)
        probes = [('historical_frame20_scalar_NOT_current_reference', oldrows[20]['reference_disparity'], None)]
        probes += [(f'same_frame_anchor{a["anchor_frame"]}_attempt', a['reference_disparity'], a['program_reference_valid'])
                   for a in frame_attempts]
        if r is not None: probes += [('revision1_current', r, True)]
        for name, value, validity in probes:
            z = components(c, q, union, value)
            sensitivity.append(dict(frame_id=t, reference_source=name, source_reference_valid=validity,
                reference_q=value*255, reference_d=value,
                original_q_and_union_held_fixed=True, eligible_high_pixels=z['eligible_high'],
                eligible_low_pixels=z['eligible_low'], eligible_total_pixels=int(z['eligible'].sum()),
                fraction_subject=z['fraction'], cap_would_skip=z['fraction'] > .25,
                adopted=False, interpretation='Input scalar sensitivity only; cap pass is not correctness or adoption.'))
        candidate = next(a for a in frame_attempts if a['anchor_frame'] == 24)
        proposed = components(c, q, union, candidate['reference_disparity'])
        pending = proposed['high'] & head if t >= 21 and candidate['program_reference_valid'] else np.zeros_like(union)
        assert not pending[~union].any() and pending.sum()/union.sum() <= .25
        proposals.append(dict(frame_id=t, intervention_proposed=t >= 21,
            reference_source='same-frame original q at existing anchor24 RGB-matched box',
            reference_native_xywh=candidate['native_xywh'], reference_q=candidate['reference_q_median'],
            program_reference_valid=candidate['program_reference_valid'], user_reference_confirmed=False,
            head_context_display_xywh=roirows[t]['head_context_xywh'],
            pending_target_pixels=int(pending.sum()), pending_fraction_subject=float(pending.sum()/union.sum()),
            target_mask=f'B36/native/Pending_late_head/{t:05d}.png',
            user_target_confirmed=False, label='人工辅助末段诊断候选；不是原规则结果',
            common_evaluation_target='Same pending INPUT mask for A/R1/R2, not each condition treatment mask. Purity pending user.',
            no_intervention_if_reference_invalid=True))
        for name, mask in [('Current_high', None if current is None else current['raw'] & current['higher']),
                           ('Current_low', None if current is None else current['raw'] & ~current['higher']),
                           ('Current_eligible', None if current is None else current['eligible']),
                           ('Pending_late_head', pending)]:
            path = native/name/f'{t:05d}.png'; path.parent.mkdir(exist_ok=True)
            Image.fromarray(np.uint8(np.zeros_like(union) if mask is None else mask)*255).save(path)
        rows.append(dict(frame_id=t, current_anchor=loc['anchor_frame'], current_native_xywh=loc['attempted_native_xywh'],
            current_RGB_checks=loc['checks'], current_reference_valid=old['reference_valid'],
            current_reference_q=old['reference_q'], current_reference_d=r,
            current_reference_d_distribution=distribution(d[refmask]) if r is not None else None,
            invalid_attempt_box_d_distribution=distribution(d[refmask]) if r is None else None,
            focus=focus, subject_pixels=int(union.sum()), subject_d_distribution=distribution(d[union]),
            raw_high_pixels=None if current is None else current['raw_high'],
            raw_low_pixels=None if current is None else current['raw_low'],
            eligible_high_pixels=None if current is None else current['eligible_high'],
            eligible_low_pixels=None if current is None else current['eligible_low'],
            eligible_pixels=old['eligible_area'], eligible_fraction_subject=None if current is None else current['fraction'],
            retained_component_count=None if current is None else len(per_frame),
            selected_pixels=old['selected_area'], skip_reason=old['missing_or_skip_reason'],
            low_side_body_core_overlap=None if current is None else int((current['low'] & body).sum()),
            low_side_head_context_overlap=None if current is None else int((current['low'] & head).sum()),
            low_side_boundary_overlap=None if current is None else int((current['low'] & boundary).sum()),
            head_context_display_xywh=roirows[t]['head_context_xywh'], body_display_xywh=roirows[t]['body_display_xywh'],
            components=per_frame, dual_anchor_attempts=frame_attempts))
    result = dict(created_UTC=b.now(), status='诊断材料已准备，等待用户确认参考位置与末段局部对照',
        scene=cfg['scene'], sample_id='B36', source_group='DAVIS:mallard-water', split='exploration',
        frames=cfg['focus_frames'], config=cfg, rows=rows, components=ccrows, sensitivity=sensitivity,
        pending_control_proposals=proposals, protected_records=hashes,
        history_preserved=True, new_inference_started=False, neural_inference_runs=0,
        limitations=['No new predictions or visual quality verdicts.',
            'Foreground coverage/NCC do not prove anatomical correspondence; both anchors require user review.',
            'Frame21 official candidate fields remain null, not zero. Alternative attempts do not repair history.',
            'Sensitivity preserves original per-frame q/union, not physical geometry; earlier-frame scalar is not truth.',
            'Mixed head_context/body rectangles and boundary bands are overlapping spatial proxies, not an anatomical partition.',
            'No continuous late-correction output exists; blur regression cause is not uniquely established.'])
    b.save(out/'diagnosis.json', result)
    b.save(ROOT/(cfg['record_prefix']+'.json'), result)
    csv_save(ROOT/(cfg['record_prefix']+'-frames.csv'), [{k:v for k,v in row.items() if k not in ['components','dual_anchor_attempts']} for row in rows])
    csv_save(ROOT/(cfg['record_prefix']+'-components.csv'), ccrows)
    csv_save(ROOT/(cfg['record_prefix']+'-sensitivity.csv'), sensitivity)
    b.save(ROOT/(cfg['record_prefix']+'-pending-control.json'), dict(status='待用户确认；未生成新输出',
        plan=cfg['pending_control'], frames=proposals, intervention_source_frames=[21,22,23,24],
        all_0_20_CoC_must_match_R1_bitwise=True, author_commit=c['author_commit'], K=16,
        compare_all_25_output_frames=True, no_inference_before_confirmation=True))
    for row in rows:
        print('DIAG',row['frame_id'], 'r',row['current_reference_q'], 'eligible H/L',row['eligible_high_pixels'],row['eligible_low_pixels'],
              'bodyL',row['low_side_body_core_overlap'], 'headL',row['low_side_head_context_overlap'],
              'fraction',row['eligible_fraction_subject'],flush=True)


def materials(cfg):
    sys.path.insert(0, str(ROOT/'src'))
    from review.any2bokeh_pages import viewer_fragment, render_review_form, payload
    out = ROOT/cfg['output']; result = b.read(out/'diagnosis.json')
    c = b.read(ROOT/cfg['selection_config']); oldpage = ROOT/c['output']/'B36'; page = out/'B36'
    assert not (page/'index.html').exists()
    loc = b.read(b.record(c, '-localization.json'))['rows']
    names = ['Input_RGB','Reference_boxes','VDA','Current_high','Current_low','Current_eligible','Pending_late_head']
    labels = ['原RGB（无叠加）','原参考绿/红；anchor18黄；anchor24橙（均待位置确认）',
              '原VDA q/255，固定0–1色标','原规则高侧：粉，未做面积过滤',
              '原规则低侧：青，未做面积过滤','原规则面积过滤后：高粉/低青；ID标每个≥64组件',
              '待确认末段局部诊断候选：高侧∩旧头部上下文框；未运行']
    for name in names: (page/'originals'/name).mkdir(parents=True)
    sheet = Image.new('RGB',(7*320,3*260),'white'); draw = ImageDraw.Draw(sheet)
    rows = {r['frame_id']:r for r in result['rows']}
    pending = {r['frame_id']:r for r in result['pending_control_proposals']}
    for t in range(25):
        # Input and VDA displays are byte-for-byte links to the existing package.
        for name in ['Input_RGB','VDA']:
            b.link(page/'originals'/name/f'{t:05d}.png',oldpage/'originals'/name/f'{t:05d}.png')
        if t not in rows:
            for name, oldname in [('Reference_boxes','Neck_reference'),('Current_high','Raw_candidates'),
                                  ('Current_low','Raw_candidates'),('Current_eligible','Selected_candidates'),
                                  ('Pending_late_head','Input_RGB')]:
                im = Image.open(oldpage/'originals'/oldname/f'{t:05d}.png').convert('RGB')
                ImageDraw.Draw(im).text((15,15),'Outside18-24 diagnosis: historical panel / no new late proposal',fill='white',stroke_width=2,stroke_fill='black')
                im.save(page/'originals'/name/f'{t:05d}.png')
            continue
        row = rows[t]; rgb = b.input_rgb(c,t); q,u,f = b.original_geometry(c,t)
        current = components(c,q,u,row['current_reference_d'])
        references = rgb.copy()
        for a in row['dual_anchor_attempts']:
            x,y,w,h=a['native_xywh']; tint=(255,230,0) if a['anchor_frame']==18 else (255,120,0)
            cv2.rectangle(references,(x,y),(x+w-1,y+h-1),tint,2)
        x,y,w,h=row['current_native_xywh']
        cv2.rectangle(references,(x-2,y-2),(x+w+1,y+h+1),(0,255,70) if row['current_reference_valid'] else (255,0,0),2)
        shown = {'Reference_boxes':references}
        for name,side,tint in [('Current_high','high',[255,0,180]),('Current_low','low',[0,220,255])]:
            mask = np.asarray(Image.open(page/'native'/name/f'{t:05d}.png'))!=0
            shown[name] = b.overlay(rgb,[(mask,tint)])
        shown['Current_eligible'] = rgb.copy() if current is None else b.overlay(rgb,[(current['high'],[255,0,180]),(current['low'],[0,220,255])])
        if current is not None:
            for co in row['components']:
                x,y,w,h = co['native_xywh']
                cv2.rectangle(shown['Current_eligible'],(x,y),(x+w-1,y+h-1),(255,255,0),1)
                cv2.putText(shown['Current_eligible'],str(co['component_id']),(x,max(12,y)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,0),1)
        mask = np.asarray(Image.open(page/'native/Pending_late_head'/f'{t:05d}.png'))!=0
        shown['Pending_late_head'] = b.overlay(rgb,[(mask,[255,0,180])])
        if t>=21:
            x,y,w,h=pending[t]['reference_native_xywh']
            cv2.rectangle(shown['Pending_late_head'],(x,y),(x+w-1,y+h-1),(255,120,0),2)
        for name, a in shown.items():
            im=Image.fromarray(cv2.resize(a,b.DISPLAY,interpolation=cv2.INTER_LINEAR))
            if t==21 and name.startswith('Current_'):
                ImageDraw.Draw(im).text((14,14),'NOT EVALUATED: revision1 reference invalid; not zero candidates',fill='white',stroke_width=2,stroke_fill='black')
            im.save(page/'originals'/name/f'{t:05d}.png')
        # Native RGB + raw q and original class-index annotation remain linked/read-only.
        for name,ext in [('RGB','jpg'),('Raw_VDA','png'),('DAVIS_labels','png')]:
            b.link(page/'native'/name/f'{t:05d}.{ext}',oldpage/'native'/name/f'{t:05d}.{ext}')
        # Native full overlay for precise annotation inspection; no sharpening.
        Image.fromarray(references).save(page/'native'/f'reference-boxes-{t:05d}.png')
        x,y,w,h = row['current_native_xywh']; crop=(x-140,y-205,x+180,y+35)
        for k,name in enumerate(['Reference_boxes','VDA','Current_eligible']):
            source = (b.color(q.astype(np.float32)/255,0,1) if name=='VDA' else shown[name])
            im=Image.fromarray(source).crop(crop).resize((320,240),Image.Resampling.NEAREST)
            top=k*260; sheet.paste(im,((t-18)*320,top+20))
            draw.text(((t-18)*320+3,top+3),f'f{t} {name} anchor{row["current_anchor"]}',fill='black')
    sheet.save(page/'late-input-contact.png')
    # Reuse unchanged input-only video, never new inference/output content.
    b.link(page/'input-selection.mp4',oldpage/'input-selection.mp4')
    b.link(page/'fixed-color-legends.png',oldpage/'fixed-color-legends.png')
    item=dict(sample_id='B36',dataset='DAVIS',scene_id=c['scene'],source_group='DAVIS:mallard-water',split='exploration',
        package='B36-tail-input',review_url='../review.html',frame_jumps=list(range(18,25)),initial_frame=20,
        panels=[dict(label=label,prefix=f'originals/{name}/') for label,name in zip(labels,names)],
        tracked_rois=[dict(id='neck',name='原参考：21缺失不补',xywh_by_frame={str(t):loc[t]['display_xywh'] for t in range(25)}),
                      dict(id='anchor24',name='anchor24同帧回配框（待确认）',xywh_by_frame={str(t):b.display_box(pending[t]['reference_native_xywh']) if t in pending else None for t in range(25)}),
                      dict(id='head_context',name='旧头部上下文（混合框）',xywh_by_frame={str(t):rows[t]['head_context_display_xywh'] if t in rows else None for t in range(25)})])
    b.save(out/'viewer-item.json',item)
    css='<style>body{font-family:system-ui;margin:24px;background:#f6f9ff}td,th{border:1px solid #abc;padding:5px}table{border-collapse:collapse}video,img{max-width:100%}</style>'
    text='<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>B36末段输入诊断</title>'+css
    text+='''<h1>B36：18–24帧停止修正的输入诊断，等待确认</h1>
<p>仅CPU原输入检查，没有新ATB推理。revision1、A/R1与用户反馈保留。请先看参考框是否仍为相近颈部位置（18和24锚点也需检查），再看青色低侧扩张在身体／边界还是头部；最后确认粉色末段诊断候选。程序不判断感知质量或真实深度。</p>
<p>原RGB/VDA/DAVIS索引网格1920×1080；查看器1024×576。RGB/叠加图INTER_LINEAR，VDA固定0–1 TURBO后最近邻；接触图裁剪最近邻，不锐化。q/255不是米制深度。粉=高侧，青=低侧；高低侧均按固定绝对差&gt;0.10，先原双侧8连通再过滤面积≥64。最后一面板是独立待确认的人工辅助局部候选，不是revision1，也未生成新条件输出。</p>
<p>旧框外圈绿=程序有效，红=失败；黄=anchor18传播，橙=anchor24传播。匹配通过／主体占比高不等于解剖位置正确。21帧原候选未测，空图不是候选为0。22–24面积合格候选仍展示，即使总量&gt;25%而整帧跳过。</p>
<p>当前只调查18–24；其他帧面板注明历史展示，不作本轮新增候选证据。视频复用原输入＋revision1选区25帧/20fps CRF18/yuv420p，1×为1.25秒，慢放不插帧；下面PNG为本次精确输入诊断。24→0是循环，不是连续transition。</p>
<p><a href="../review.html">填写参考位置及末段候选反馈（独立key）</a> · <a href="../frames.csv">原因CSV</a> · <a href="../components.csv">逐组件CSV</a> · <a href="../sensitivity.csv">固定原q的参考值敏感性</a> · <a href="../diagnosis.json">完整记录</a></p>
<video controls preload="metadata" src="input-selection.mp4"></video><img src="fixed-color-legends.png" alt="固定色标">'''
    text+=viewer_fragment(item)
    text+='<h2>18–24原因表</h2><table><tr>'+''.join('<th>'+x+'</th>' for x in ['帧/锚','r / focus','合格高侧/低侧像素','总量/主体','低侧与身体核心重叠','跳过原因','原网格'])+'</tr>'
    for row in rows.values():
        t=row['frame_id'];fr=row['eligible_fraction_subject']
        links=' · '.join(f'<a href="native/{name}/{t:05d}.{ext}">{label}</a>' for name,ext,label in [('RGB','jpg','RGB'),('Raw_VDA','png','q'),('DAVIS_labels','png','原标注'),('Pending_late_head','png','待确认区域')])
        text+='<tr>'+''.join('<td>'+str(x)+'</td>' for x in [f'{t}/{row["current_anchor"]}',f'{row["current_reference_d"]} / {row["focus"]}',f'{row["eligible_high_pixels"]} / {row["eligible_low_pixels"]}','未测' if fr is None else f'{fr:.3%}',row['low_side_body_core_overlap'],html.escape(row['skip_reason']) or '无',links])+'</tr>'
    text+='</table><h2>原参考与原规则候选接触图</h2><a href="late-input-contact.png"><img src="late-input-contact.png"></a>'
    text+='<h2>逐组件位置（原网格，类别为输入代理）</h2><table><tr>'+''.join('<th>'+x+'</th>' for x in ['帧/组件','bbox','面积','高/低','头部上下文/身体核心/5px轮廓重叠'])+'</tr>'
    for row in result['components']:
        text+='<tr>'+''.join('<td>'+str(x)+'</td>' for x in [f'{row["frame_id"]}/{row["component_id"]}',row['native_xywh'],row['pixels'],f'{row["high_pixels"]}/{row["low_pixels"]}',f'{row["head_context_overlap"]}/{row["body_core_overlap"]}/{row["boundary_5px_overlap"]}'])+'</tr>'
    text+='</table><p>这些矩形/轮廓统计可重叠，不能相加成解剖分区；头部上下文含颈部与背景。组件ID仅限本帧，不是跟踪。</p>'
    text+='''<h2>唯一待确认的后续对照</h2><p>人工辅助末段诊断R2：0–20沿用R1 CoC逐位不变；21采用原anchor24在同帧RGB的回配参考，22–24沿用当前同帧参考。只在末段粉色候选（高侧合格像素∩旧头部上下文∩主体）用同帧参考视差构造CoC；不抬25%上限，不强制CoC=0、不平滑。请确认橙框是否适合作参考、粉色是否覆盖希望干预的头部且无不合适位置。若不能确认可靠参考，就停在输入诊断。</p>
<p>确认后才考虑R2推理，A/R1复用需核对版本/输入/RNG；比较完整25帧，因为末段条件可能影响较早输出。提前固定粉色输入目标mask作A/R1/R2共同统计支持，混合头部上下文另列；颈部/非目标身体/近远水面也统一，不能用R1空区对R2非空区直接比较。纯头部位置及参考解剖有效性仍待用户确认。参考标量敏感性只是输入诊断，不能按是否通过25%自动选择参考。</p>'''
    (page/'index.html').write_text(text)
    form=render_review_form([item],package='B1',store_key=cfg['review_storage_key'],title='B36末段输入参考与局部对照确认',
        notice='只确认输入位置/候选，不是新输出视觉验收。未确认前不启动推理。旧反馈不覆盖。')
    marker='<script id="atb-review-config" type="application/json">';start=form.index(marker)+len(marker);end=form.index('</script>',start)
    fc=json.loads(form[start:end]);fc['package']=item['package'];fc['fields']=['frame_id_or_transition_id','ROI_xywh','actual_view_mode','neck_reference_feedback','candidate_selection_feedback','selection_confirmation','notes','actual_playback_rate']
    fc['labels']=['帧号或transition','ROI x,y,w,h（显示网格）','实际观看方式','原框/橙色颈部参考是否合适','青色扩张位置/粉色局部候选反馈','末段诊断参考和区域是否确认（用户填写）','补充说明','实际视频倍速'];fc['visual']=[]
    (out/'review.html').write_text(form[:start]+payload(fc)+form[end:])
    b.save(ROOT/(cfg['record_prefix']+'-user-review-blank.json'),dict(status='待用户确认',scene_id=cfg['scene'],
        neck_reference_feedback='',candidate_selection_feedback='',selection_confirmation='',notes='',
        review_storage_key=cfg['review_storage_key'],new_inference_authorized=False))
    for name in ['frames','components','sensitivity']:
        b.link(out/(name+'.csv'),ROOT/(cfg['record_prefix']+'-'+name+'.csv'))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>B36末段CPU输入诊断</h1><a href="B36/">18–24同步输入与候选</a> · <a href="review.html">待确认反馈</a><p>没有启动新推理。</p>')


def verify(cfg):
    out=ROOT/cfg['output'];data=b.read(out/'diagnosis.json');c=b.read(ROOT/cfg['selection_config'])
    assert all(digest(ROOT/p)==h for p,h in data['protected_records'].items())
    panel_count=0
    for t in range(25):
        for folder in (out/'B36/originals').iterdir():
            with Image.open(folder/f'{t:05d}.png') as im:
                assert im.size==b.DISPLAY and im.mode=='RGB';im.verify();panel_count+=1
    for row in data['rows']:
        t=row['frame_id'];q,u,f=b.original_geometry(c,t);z=components(c,q,u,row['current_reference_d'])
        if z is None:
            assert t==21 and row['eligible_pixels'] is None and row['eligible_high_pixels'] is None
        else:
            assert row['eligible_high_pixels']+row['eligible_low_pixels']==row['eligible_pixels']
            assert sum(co['pixels'] for co in row['components'])==row['eligible_pixels']
            assert (z['fraction']>.25)==(t>=22)
        p=data['pending_control_proposals'][t-18]
        mask=np.asarray(Image.open(out/p['target_mask']))!=0
        assert mask.shape==q.shape and not mask[~u].any() and mask.sum()==p['pending_target_pixels']
        assert mask.sum()/u.sum()<=.25 and bool(mask.any())==(t>=21)
    assert not data['new_inference_started'] and data['neural_inference_runs']==0
    b.save(out/'program-checks.json',dict(checked_UTC=b.now(),diagnosis_frames=7,
        display_panel_PNGs=panel_count,original_records_unchanged=True,
        original_reference_and_skip_decisions_reproduced=True,component_high_low_sums_verified=True,
        fixed_q_scalar_sensitivity=True,frame21_official_candidates_NA=True,
        pending_masks_inside_subject_and_under_cap=True,proposal_not_approved=True,
        new_model_outputs=0,inference_started=False,neural_inference_runs=0))
    print('PROGRAM_CHECKS_PASS',flush=True)


def summarize(cfg):
    """Complete the input-only explanation and show all reference scores in-page."""
    out=ROOT/cfg['output']; path=out/'diagnosis.json'; data=b.read(path)
    assert 'conclusions' not in data, 'Preserve completed diagnosis; do not append twice'
    rows={r['frame_id']:r for r in data['rows']}
    last=rows[24]; first=rows[20]
    probe=next(r for r in data['sensitivity'] if r['frame_id']==24 and r['reference_source'].startswith('historical_frame20'))
    low_outside=last['eligible_low_pixels']-last['low_side_head_context_overlap']
    data['conclusions']=dict(
        observed=[
            'Source21 was skipped by fixed NCC0.8 gate (observed0.69569), not an area test; current candidate fields remain unmeasured.',
            'Source22-24 pass original reference checks but eligible total28.319/28.778/32.295% exceeds unchanged25% cap.',
            dict(change_20_to_24=dict(high_pixels=last['eligible_high_pixels']-first['eligible_high_pixels'],
                low_pixels=last['eligible_low_pixels']-first['eligible_low_pixels'],
                total_pixels=last['eligible_pixels']-first['eligible_pixels']),
                interpretation='Most eligible-area growth is on the low-disparity side, not high-side target growth.'),
            dict(frame24_low_pixels=last['eligible_low_pixels'],outside_mixed_head_context=low_outside,
                 fraction_outside_mixed_head_context=low_outside/last['eligible_low_pixels'],
                 boundary5px_overlap=last['low_side_boundary_overlap'],
                 input_overlay_check='Input-only overlay places major low-side components along the subject-label upper body/tail and lower outline; not an output quality judgement or true anatomy/segmentation-error verdict.'),
            dict(frame24_fixed_original_q=True,reference_q154_eligible=probe['eligible_total_pixels'],
                 reference_q160_eligible=last['eligible_pixels'],
                 low_q154=probe['eligible_low_pixels'],low_q160=last['eligible_low_pixels'],
                 interpretation='Reference-scalar sensitivity at one fixed input; q154 is historical, not an approved/current reference.'),
            'Same-frame anchor24 matches are about7-9 native pixels right and9-10 up relative to anchor18 attempts, with reference median4-5 q levels higher. Both manual anchors and matching anatomy remain pending user.',
            'Anchor24->source21 passes fixed RGB checks, but the original bilateral candidate would still be26.480% of subject: fixing localization alone would not bypass the original area gate.'
        ],
        interpretation='Reference patch selection and evolving input q/union both affect the candidate tail. Anchor switch contributes a patch/value discontinuity; it does not prove the reference is anatomically wrong or the depth physically wrong.',
        unresolved=['Which neck patch is anatomically comparable/appropriate across18-24, including manual anchors18/24.',
            'Whether continued LOCAL CoC correction sustains the user-preferred late appearance without unwanted effects; no R2 predictions exist.',
            'Unique cause of late blur, true physical geometry and separate temporal-module contributions remain unproven.'],
        one_next_control='Only after user confirms the displayed same-frame orange references and pink21-24 target: one manually assisted R2, originalR1 CoC0-20 bitwise preserved, late target replaced by same-frame reference then original CoC construction. Compare complete25frames to reusableA/R1 after provenance check; all fixed supports common across conditions. No cap increase or alternative inference search.',
        stop='输入诊断材料已准备，等待用户确认；未启动新推理')
    data['reproduction']=dict(environment='/root/miniconda3/envs/any2bokeh/bin/python; CPU only; cv2 threads1',
        stages=['analyze','materials','summarize','verify'],
        command='python scripts/diagnose_b36_rule_b_tail.py <stage>',
        browser='python scripts/check_b36_rule_b_materials.py --config configs/any2bokeh-B36-rule-b-tail-20261021.json; existing8765 root, private headless Chrome GPU disabled',
        source_commit='05ef3e8 before this task; author f7a16959 preserved',
        no_model_loaded=True,no_RNG_or_condition_tensor_modified=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (ROOT/(cfg['record_prefix']+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    references=[]
    for row in data['rows']:
        for a in row['dual_anchor_attempts']:
            references.append(dict(frame_id=row['frame_id'],anchor_frame=a['anchor_frame'],
                native_xywh=a['native_xywh'],NCC=a['RGB_checks']['NCC'],
                peak_margin=a['RGB_checks']['peak_margin'],roundtrip_error_native=a['RGB_checks']['roundtrip_error_native'],
                program_reference_valid=a['program_reference_valid'],failure_reason=a['failure_reason'],
                foreground_fraction=a['foreground_fraction'],valid_pixels=a['valid_pixels'],
                reference_q=a['reference_q_median'],reference_d_distribution=a['reference_d_distribution'],
                anatomy_confirmation='待用户确认',adopted=False))
    csv_save(ROOT/(cfg['record_prefix']+'-references.csv'),references)
    b.link(out/'references.csv',ROOT/(cfg['record_prefix']+'-references.csv'))
    # One fixed input-defined crop across all seven frames, including full body.
    c=b.read(ROOT/cfg['selection_config']);bounds=[]
    for t in cfg['focus_frames']:
        q,u,f=b.original_geometry(c,t);yy,xx=np.where(u)
        bounds.append((xx.min()/1.875,yy.min()/1.875,(xx.max()+1)/1.875,(yy.max()+1)/1.875))
    crop=(int(min(x[0] for x in bounds))-8,int(min(x[1] for x in bounds))-8,
          int(max(x[2] for x in bounds))+9,int(max(x[3] for x in bounds))+9)
    w,h=crop[2]-crop[0],crop[3]-crop[1];height=round(h*360/w)
    sheet=Image.new('RGB',(7*360,3*(height+20)),'white');draw=ImageDraw.Draw(sheet)
    for i,t in enumerate(cfg['focus_frames']):
        for k,name in enumerate(['Reference_boxes','Current_low','Current_eligible']):
            with Image.open(out/'B36/originals'/name/f'{t:05d}.png') as im:
                tile=im.crop(crop).resize((360,height),Image.Resampling.NEAREST)
            sheet.paste(tile,(i*360,k*(height+20)+20))
            draw.text((i*360+2,k*(height+20)+3),f'f{t} {name}',fill='black')
    sheet.save(out/'B36/full-subject-contact.png')
    text='<h2>完整主体范围：低侧扩张的位置</h2><p>所有帧同一输入定义裁剪，显示网格xyxy='+str(crop)+'；最近邻放大。青色与主体标注上缘/尾部/下缘重叠，不据此认定标注或真实几何错误。</p><a href="full-subject-contact.png"><img src="full-subject-contact.png"></a>'
    text+='<h2>旧参考框、匹配得分与分布</h2><table><tr>'+''.join('<th>'+x+'</th>' for x in ['帧','当前锚','当前NCC','当前框原网格','参考d p10/p50/p90','focus','无效原因'])+'</tr>'
    for r in data['rows']:
        distribution=r['current_reference_d_distribution']
        text+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['frame_id'],r['current_anchor'],r['current_RGB_checks']['NCC'],r['current_native_xywh'],
            None if distribution is None else [distribution[k] for k in ['p10','p50','p90']],r['focus'],r['skip_reason']])+'</tr>'
    text+='</table><p>None NCC=已有人工锚点，不代表匹配得分1或解剖位置正确。21框只是失败尝试，不能把其视差用作有效参考。</p>'
    text+='<h2>同帧双锚点诊断（全部尝试，不择优）</h2><a href="../references.csv">参考CSV</a><table><tr>'+''.join('<th>'+x+'</th>' for x in ['帧/锚','框原网格','NCC / 往返误差','程序有效/原因','q中位数','解剖位置'])+'</tr>'
    for r in references:
        text+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [f'{r["frame_id"]}/{r["anchor_frame"]}',r['native_xywh'],f'{r["NCC"]}/{r["roundtrip_error_native"]}',f'{r["program_reference_valid"]} {r["failure_reason"]}',r['reference_q'],r['anatomy_confirmation']])+'</tr>'
    text+='</table><h2>同帧固定原q：参考标量敏感性</h2><p>历史20帧q154只是敏感性探针，不能当当前正确值；失败回配也不能被采纳。是否通过面积上限不是选择参考的依据。未改变阈值、连接或上限；无新推理。</p><table><tr>'+''.join('<th>'+x+'</th>' for x in ['帧','来源','参考q','合格高/低像素','总量占主体','程序参考有效'])+'</tr>'
    for r in data['sensitivity']:
        if r['reference_source']=='revision1_current':continue
        text+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [r['frame_id'],r['reference_source'],r['reference_q'],f'{r["eligible_high_pixels"]}/{r["eligible_low_pixels"]}',f'{r["fraction_subject"]:.3%}',r['source_reference_valid']])+'</tr>'
    text+='</table><p>已支持：低侧面积扩张主导超限，参考标量敏感；仍缺用户对颈部位置、粉色局部候选的确认及末段持续修正配对输出。没有证明唯一深度错误或唯一模糊成因。</p>'
    page=out/'B36/index.html';page.write_text(page.read_text()+text)
    print('INPUT_SUMMARY_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['analyze','materials','summarize','verify'])
    args=parser.parse_args();cv2.setNumThreads(1);globals()[args.stage](b.read(CONFIG))
