"""CPU-only local inspection of existing V2 inputs and predictions.

ROI locations are saved from RGB before this command. No inference, new
intervention, condition retuning or perceptual verdict is implemented here.
"""
import csv
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from check_atb_mpi_invariance import author_imports, digest
from prepare_b36_rule_b import ROOT, native_box, read, save, now
from run_subject_coc_v2 import condition, geometry

RECORD = ROOT / '03_实验记录/records/any2bokeh-V2-local-input-output-20261022'
ROI_FILE = Path(str(RECORD) + '-ROIs.json')


def region(box, shape):
    x, y, w, h = box
    assert w > 0 and h > 0 and x >= 0 and y >= 0
    assert x + w <= shape[1] and y + h <= shape[0]
    m = np.zeros(shape, bool)
    m[y:y+h, x:x+w] = True
    return m


def stats(values, mask):
    x = values[mask]
    if not x.size:
        return dict.fromkeys(['mean', 'p10', 'p50', 'p90', 'maximum'])
    return dict(mean=float(x.mean()), p10=float(np.quantile(x,.1)),
                p50=float(np.median(x)), p90=float(np.quantile(x,.9)), maximum=float(x.max()))


def colored(values, upper):
    # One fixed scale per figure family; no per-frame or crop normalization.
    a = np.uint8(np.rint(np.clip(values / upper, 0, 1) * 255))
    return cv2.cvtColor(cv2.applyColorMap(a, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def png(path, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists()
    Image.fromarray(a).save(path)


def contact(path, title, rows, panels, native=False):
    columns = ['RGB', 'Original CoC x K', 'V2 CoC x K'] if native else ['RGB', 'Original CoC x K', 'V2 CoC x K', 'A', 'V2']
    width, row_height = 240, 190
    im = Image.new('RGB', (len(columns)*width, 60 + len(rows)*row_height), 'white')
    draw = ImageDraw.Draw(im)
    draw.text((8, 7), title, fill='black')
    for j, name in enumerate(columns):
        draw.text((j*width+8, 33), name, fill='black')
    for i, r in enumerate(rows):
        box = native_box(r['display_xywh']) if native else r['display_xywh']
        x, y, w, h = box
        factor = max(1, min((width-16)//w, (row_height-50)//h))
        for j, a in enumerate(panels):
            crop = Image.fromarray(a).crop((x, y, x+w, y+h))
            crop = crop.resize((w*factor, h*factor), Image.Resampling.NEAREST)
            im.paste(crop, (j*width+8, 60+i*row_height+22))
        draw.text((8, 60+i*row_height+3), r['roi_id'], fill='black')
        draw.text((8, 60+(i+1)*row_height-20), f'xywh={box}; nearest {factor}x; pending ROI confirmation', fill='black')
    assert not path.exists()
    im.save(path)


def case_summaries(rows):
    def selected(scene, names):
        return [r for r in rows if r['scene_id']==scene and r['roi_id'] in names
                and r['support']=='target_annotation_intersection']
    return dict(
        B20=dict(evidence=selected('stroller',['adult_shoe_image_left','adult_shoe_image_right']),
            observation='成人左右鞋及不同代表帧覆盖差异大。24帧图像左鞋交集全部触发限幅，实际条件明确变化；右鞋大部分未触发。不能把成人鞋合并解释为未覆盖或修正无效。',
            next_check='先由用户确认左右鞋框及原RGB对应细节。左鞋24帧确认后，对照仍保留的名义离焦与已有输出响应，讨论条件利用/边界纹理是否需要后续检查；右鞋先核查原离焦与覆盖。当前不运行编码检查、不降低tau。'),
        B34=dict(evidence=selected('cows',['bell_core']),
            observation='0/12/24帧铃铛核心主体交集均未触发限幅，实际CoC逐值不变。外围混合框可有少量变化，不能代表铃铛核心已被修改。',
            next_check='先确认铃铛核心及颈部比较框。位置成立后，比较原RGB与输出、原离焦和颈部差异；先讨论原名义离焦是否符合期望，不能称已修但无效，也不据此直接降低全局tau或确认深度错误。'),
        train=dict(evidence=selected('train',['rear_carriage_core_candidate']),
            observation='末节车厢核心候选在0/6/12帧主体交集全部触发，实际CoC及输出响应明显大于本次鞋/铃铛候选。',
            next_check='确认用户所指末节及帧号，再判断主体保护是否不合理压平纵深。仅作为改变景深的案例，不列为已确认深度错误。'),
        evidence_boundary='位置均待用户确认；只覆盖所列代表帧。全主体其他位置也被V2修改，局部输出响应不是隔离因果实验，不将直接响应大小解释为质量或真实深度正确性。')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only command: explicitly hide CUDA'
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    c = read(ROI_FILE)
    out, source = ROOT/c['output'], ROOT/c['source_V2']
    assert not (out/'analysis.json').exists(), 'Preserve completed inspection'
    rows = [r for r in c['rows'] if r['active']]
    assert len(rows) == 27 and not c['user_roi_confirmed']
    split = read(ROOT/'03_实验记录/records/any2bokeh-failure-analysis-20261009-source-split.json')['rows']
    allowed = {r['scene_id'] for r in split if r['dataset']=='DAVIS' and r['split']=='exploration'}
    assert {r['scene_id'] for r in rows} <= allowed
    author_cfg, _, _, _, Dataset, _ = author_imports()  # definitions only; no model instances/weights
    measurements, verification, items = [], [], []
    for scene in dict.fromkeys(r['scene_id'] for r in rows):
        folder = source/'scenes'/scene
        prepared = read(folder/'prepared.json')
        bundle = torch.load(folder/'inputs.pt', map_location='cpu', weights_only=True, mmap=True)
        ds = Dataset(str(ROOT/'outputs/davis-baselines-20260929/any2bokeh/audit'/scene/'input.csv'), sample_size=(576, 1024))
        floats = {v: np.load(folder/v/'float.npy', mmap_mode='r') for v in ['A', 'V2']}
        assert all(a.dtype == np.float32 and a.shape == (25,576,1024,3) for a in floats.values())
        for t in sorted({r['frame_id'] for r in rows if r['scene_id']==scene}):
            rois = [r for r in rows if r['scene_id']==scene and r['frame_id']==t]
            q, label, focus = geometry(scene, t)
            assert q.shape == (1080,1920)
            d = q.astype(np.float64)/255
            union = label != 0
            limited = union & ((d<focus-1/16) | (d>focus+1/16))
            changed = d.copy()
            changed[union] = np.clip(d[union], focus-1/16, focus+1/16)
            original, before = condition(ds, q, focus)
            candidate, after = condition(ds, changed*255, focus)
            slots = [[g,j] for g, group in enumerate(prepared['groups']) for j, f in enumerate(group) if f==t]
            for g,j in slots:
                assert digest(original) == digest(bundle['base']['coc'][g,j])
                assert digest(candidate) == digest(bundle['candidate_coc'][g,j])
            assert all(torch.equal(x[0], x[1]) and torch.equal(x[0], x[2]) for x in [original,candidate])
            native_before, native_after = 16*np.abs(d-focus), 16*np.abs(changed-focus)
            native_rgb = cv2.cvtColor(cv2.imread(str(ROOT/'data/processed/davis2016-v2/eval'/scene/'rgb'/f'{t:05d}.jpg')), cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(native_rgb, (1024,576), interpolation=cv2.INTER_LINEAR)
            images = {v: np.asarray(Image.open(folder/v/'png'/f'{t:05d}.png').convert('RGB')) for v in ['A','V2']}
            for v in images:
                assert np.isfinite(floats[v][t]).all()
                assert np.array_equal(images[v], np.uint8(np.rint(np.clip(floats[v][t],0,1)*255)))
            display_label = cv2.resize(label, (1024,576), interpolation=cv2.INTER_NEAREST)
            norm_delta = np.abs(candidate[0].numpy()-original[0].numpy())
            io_delta = after-before
            out_delta = np.abs(images['V2'].astype(np.int16)-images['A'].astype(np.int16))
            float_delta = np.abs(floats['V2'][t].astype(np.float64)-floats['A'][t].astype(np.float64))
            for r in rois:
                nr, dr = region(native_box(r['display_xywh']), q.shape), region(r['display_xywh'], (576,1024))
                target = label!=0 if r['target_annotation_id'] is None else label==r['target_annotation_id']
                target_d = display_label!=0 if r['target_annotation_id'] is None else display_label==r['target_annotation_id']
                for support in ['rectangle','target_annotation_intersection']:
                    nm = nr if support=='rectangle' else nr & target
                    dm = dr if support=='rectangle' else dr & target_d
                    ids = {str(i):int((label[nm]==i).sum()) for i in np.unique(label[nm])}
                    count, count_d = int(nm.sum()), int(dm.sum())
                    fc = int((limited & nm).sum())
                    selected_delta = out_delta[dm]
                    selected_float = float_delta[dm]
                    row = dict(sample_id=r['sample_id'],scene_id=scene,frame_id=t,roi_id=r['roi_id'],role=r['role'],
                        localization_status=r['localization_status'],support=support,display_xywh=r['display_xywh'],native_xywh=native_box(r['display_xywh']),
                        K=16,tau=1,focus=focus,native_pixels=count,display_pixels=count_d,native_label_counts=ids,
                        native_subject_fraction=None if not count else float((nm & union).sum()/count),
                        native_limited_pixels=fc,native_limited_fraction=None if not count else fc/count,
                        **{'native_before_'+k:v for k,v in stats(native_before,nm).items()},
                        **{'native_after_'+k:v for k,v in stats(native_after,nm).items()},
                        **{'actual_before_'+k:v for k,v in stats(before,dm).items()},
                        **{'actual_after_'+k:v for k,v in stats(after,dm).items()},
                        actual_K_abs_delta_mean=None if not count_d else float(np.abs(io_delta[dm]).mean()),
                        actual_K_abs_delta_max=None if not count_d else float(np.abs(io_delta[dm]).max()),
                        actual_K_signed_delta_mean=None if not count_d else float(io_delta[dm].mean()),
                        actual_normalized_abs_delta_mean=None if not count_d else float(norm_delta[dm].mean()),
                        actual_normalized_abs_delta_max=None if not count_d else float(norm_delta[dm].max()),
                        actual_changed_pixels=int(np.count_nonzero(norm_delta[dm])),
                        actual_changed_fraction=None if not count_d else float(np.count_nonzero(norm_delta[dm])/count_d),
                        output_RGB8_abs_delta_mean=None if not count_d else float(selected_delta.mean()),
                        output_RGB8_abs_delta_max=None if not count_d else int(selected_delta.max()),
                        output_RGB8_changed_bytes=int(np.count_nonzero(selected_delta)),
                        output_float_abs_delta_mean=None if not count_d else float(selected_float.mean()),
                        output_float_abs_delta_max=None if not count_d else float(selected_float.max()),
                        output_float_changed_elements=int(np.count_nonzero(selected_float)))
                    measurements.append(row)
            frame_path = out/rois[0]['sample_id']/f'{t:05d}'
            frame_path.mkdir(parents=True)
            paths = {}
            panels = [rgb,colored(before,4),colored(after,4),images['A'],images['V2']]
            names = ['RGB','Original_CoC','V2_CoC','A','V2']
            for name,a in zip(names,panels):
                p=frame_path/(name+'.png');png(p,a);paths[name]=str(p.relative_to(out))
            for name,a in [('Native_RGB',native_rgb),('Native_Original_CoC',colored(native_before,4)),('Native_V2_CoC',colored(native_after,4)),
                           ('Original_CoC_0_16',colored(before,16)),('V2_CoC_0_16',colored(after,16)),('CoC_abs_delta_0_1',colored(np.abs(io_delta),1))]:
                p=frame_path/(name+'.png');png(p,a);paths[name]=str(p.relative_to(out))
            overlay = Image.fromarray(rgb)
            draw = ImageDraw.Draw(overlay)
            for i,r in enumerate(rois):
                x,y,w,h=r['display_xywh'];color='cyan' if 'comparison' in r['role'] else 'yellow'
                draw.rectangle([x,y,x+w-1,y+h-1],outline=color,width=1)
                draw.text((x,y+h+1),str(i+1),fill=color)
            overlay.save(frame_path/'ROI_locations.png');paths['ROI_locations']=str((frame_path/'ROI_locations.png').relative_to(out))
            contact(frame_path/'five-panel.png', f'{rois[0]["sample_id"]} frame {t}; actual 1024x576 grid; CoC x K fixed 0..4 (saturated above 4)',rois,panels)
            contact(frame_path/'native.png', f'{rois[0]["sample_id"]} frame {t}; native 1920x1080 grid; CoC x K fixed 0..4',rois,[native_rgb,colored(native_before,4),colored(native_after,4)],native=True)
            for r in rois:
                m=next(x for x in measurements if all(x[k]==r[k] for k in ['scene_id','frame_id','roi_id']) and x['support']=='target_annotation_intersection')
                items.append(dict(**r,native_xywh=native_box(r['display_xywh']),paths=paths,statistics=m,
                                  original_rgb=str((ROOT/'data/processed/davis2016-v2/eval'/scene/'rgb'/f'{t:05d}.jpg').relative_to(ROOT))))
            verification.append(dict(scene_id=scene,frame_id=t,actual_conditions_match_saved_tensors_bitwise=True,
                original_normalized_sha256=digest(original),V2_normalized_sha256=digest(candidate),real_slots=slots,
                preNormalize_from_exact_original_helper=True,RGB8_matches_saved_float_quantization=True,source_native_size=[1920,1080],display_size=[1024,576]))
        del bundle,ds,floats
    result=dict(created_UTC=now(),source_commit=c['project_commit'],ROI_record=str(ROI_FILE.relative_to(ROOT)),
                author_commit=author_cfg['author_commit'],frozen_V2_config='configs/any2bokeh-subject-coc-clamp-v2-20261021.json',
                execution=dict(command="CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/any2bokeh/bin/python scripts/inspect_v2_local_io.py",
                               python=sys.version,torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,
                               torch_threads=4,opencv_threads=1,CUDA_VISIBLE_DEVICES='',cuda_initialized=False,neural_inference_calls=0),
                scope='CPU reconstruction of already-run frozen conditions and local measurements of existing paired predictions; zero inference.',
                ROI_user_confirmed=False,adult_shoe_user_confirmed=True,rows=measurements,checks=verification,
                case_summaries=case_summaries(measurements),
                units=dict(native='K*abs(q/255-focus), nominal control, not measured blur radius',
                           actual='K times abs-before-resize/bilinear-antialias/CenterCrop before Normalize; reconstructed normalized tensors bitwise verified',
                           normalized='actual saved encoder input channel0 delta; three channels identical; no K multiplication inside model input',
                           output='RGB8 channel levels 0..255; float outputs 0..1; response not quality'),
                limitations=c['limits']+['Only representative frames, not a whole-clip temporal diagnosis.','An unchanged local condition can still have output response through nonlocal/spatiotemporal context; not an isolated causal intervention.'])
    save(Path(str(RECORD)+'.json'),result)
    with Path(str(RECORD)+'.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,list(measurements[0]),lineterminator='\n');writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in measurements)
    save(out/'analysis.json',result)
    save(out/'viewer-items.json',items)
    shutil.copyfile(Path(str(RECORD)+'.csv'),out/'statistics.csv')
    shutil.copyfile(ROI_FILE,out/'ROI-candidates.json')
    page(out,items)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(active_ROIs=len(rows),statistics_rows=len(measurements),verified_source_frames=len(verification),cuda_initialized=False),ensure_ascii=False),flush=True)


def page(out,items):
    data=json.dumps(items,ensure_ascii=False).replace('</','<\\/')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V2局部输入—输出核查</title>
<style>body{font:16px/1.65 sans-serif;margin:24px;background:#f2f6fb;color:#18324d}a{color:#175b9c}button,select{font:inherit;margin:4px;padding:5px}section{background:white;padding:15px;margin:16px 0}#location{max-width:100%;width:1024px} .panels{display:flex;gap:8px;overflow:auto}.panels figure{margin:0;min-width:210px;text-align:center}canvas{border:1px solid #789;image-rendering:pixelated;cursor:crosshair}table{border-collapse:collapse;font-size:14px}th,td{border:1px solid #abc;padding:5px}#summary{overflow:auto}textarea{width:95%;height:70px} .notice{border-left:5px solid #da9d29;padding:10px;background:#fff6df}</style>
<h1>成人鞋子／铃铛：现有 V2 局部核查</h1>
<p class="notice">先确认框的位置。黄色＝问题位置候选；青色＝邻近比较区，不是修正参考。所有框仍待用户确认；成人鞋已由用户确认，左右指图像位置。没有新增模型推理，没有改变 τ=1 或 mask。</p>
<p>B20、B34 仅取 0／12／24 代表帧；train 仅取 0／6／12。框按各帧 RGB 单独定位，不能据此宣称完整时序结论。train 是主体保护改变景深的候选，不是已确认深度错误。</p>
<p>左／右是图像位置，不保证跨帧对应同一解剖侧。图中五列先看位置再比输出；统计仍以保存的候选框为准。鞋和铃铛都不是完整像素级部件分割，细小框及主体交集仍可能包含别的结构。</p>
<nav><a href="#B20" onclick="choose('B20')">① B20 成人鞋</a> ｜ <a href="#B34" onclick="choose('B34')">② B34 铃铛</a> ｜ <a href="#V2-stat-01" onclick="choose('V2-stat-01')">附：train 末节候选</a> ｜ <a href="statistics.csv">完整统计 CSV</a> ｜ <a href="ROI-candidates.json">位置与历史对照</a></nav>
<section><label>案例 <select id="sample"></select></label><label>代表帧 <select id="frame"></select></label><label>局部 <select id="roi"></select></label><p id="description"></p>
<p>原 RGB 定位图（点击可按输入坐标新选共同查看框；统计始终使用已保存框，手动查看框不会改写统计）：</p><img id="location"><p id="legend"></p>
<button id="reset">恢复统计框</button><span id="box"></span><p><a id="full">显示网格 RGB（1024×576）</a> ｜ <a id="fullNative">原网格 RGB（1920×1080）</a> ｜ <a id="full25">既有完整25帧 A／V2 查看页</a> ｜ <a id="contact">五列接触图</a> ｜ <a id="nativeContact">原网格三列图</a></p></section>
<section><h2>同位置五列：实际 1024×576 网格</h2><p>RGB 按冻结展示口径线性缩小。CoC 为实际 Resize／CenterCrop 后、Normalize 前乘 K 的名义量；不是实测输出模糊半径。所有局部等倍率最近邻放大，不锐化。色标固定 0–4，超过4饱和；另有完整0–16色标入口。</p><div id="actual" class="panels"></div><p><a id="scaleA">原 CoC 全图0–16</a> ｜ <a id="scaleV">V2 CoC 全图0–16</a> ｜ <a id="delta">CoC绝对差全图0–1</a></p><div id="summary"></div></section>
<section><h2>原 1920×1080 网格</h2><p>原 RGB／原名义离焦／V2名义离焦，同色标0–4。边界按 left/top floor、right/bottom ceil 从显示框映回，像素数与显示网格分开。矩形可能含背景；表中同时保留矩形和目标标注交集，交集也不是鞋/铃铛专用真值 mask。</p><div id="native" class="panels"></div></section>
<section><h2>位置确认（独立保存，不覆盖已有观感记录）</h2><p>请先确认成人左右鞋、铃铛核心及邻近比较区是否准确。若不准，请写代表帧与坐标。是否有改善仍由你判断，程序不填写视觉通过。</p><label>位置确认 <select id="confirmed"><option>待用户确认</option><option>位置准确</option><option>需要调整</option><option>无法确定</option></select></label><p id="storage"></p><textarea id="note" placeholder="位置反馈／用户观察，自行填写"></textarea><button id="save">保存当前局部</button><button id="export">导出本页记录</button><p id="saved"></p></section>
<p>下一步只在位置确认后讨论：未覆盖、缩放后条件变化小，或条件变化明确但输出响应小。未改局部仍可能受其他位置和时序上下文影响；这些不是隔离的局部因果干预。当前不调τ、不改mask、不训练、不运行编码或其他模型。播放24→0是回到开头，不是原视频连续transition。</p>
<script>const items=__DATA__, KEY='atb-v2-local-io-20261022-roi-review-v1';
const $=id=>document.getElementById(id),uniq=a=>[...new Set(a)],opt=(id,vals)=>{$(id).innerHTML='';vals.forEach(v=>$(id).add(new Option(v,v)))};let current,viewBox,dragStart,renderId=0;const images=new Map();
opt('sample',uniq(items.map(x=>x.sample_id)));$('storage').textContent='浏览器存储 key：'+KEY;
function choose(s){$('sample').value=s;frames()}function frames(){opt('frame',uniq(items.filter(x=>x.sample_id==$('sample').value).map(x=>x.frame_id)));rois()}
function rois(){opt('roi',items.filter(x=>x.sample_id==$('sample').value&&x.frame_id==+$('frame').value).map(x=>x.roi_id));show()}
async function load(path){if(!images.has(path))images.set(path,new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>reject(Error(path));i.src=path}));return images.get(path)}
function newPanels(id,names){$(id).innerHTML='';for(const n of names){const f=document.createElement('figure'),c=document.createElement('canvas'),cap=document.createElement('figcaption');cap.textContent=n;f.append(cap,c);$(id).append(f)}}
newPanels('actual',['RGB','原 CoC×K','V2 CoC×K','A','V2']);newPanels('native',['原 RGB','原名义离焦','V2名义离焦']);
async function draw(){const token=++renderId,r=current,b=viewBox,nb=[Math.floor(b[0]*1.875),Math.floor(b[1]*1.875),Math.ceil((b[0]+b[2])*1.875)-Math.floor(b[0]*1.875),Math.ceil((b[1]+b[3])*1.875)-Math.floor(b[1]*1.875)];$('box').textContent='查看框 xywh='+b+'；统计框='+r.display_xywh;
 await Promise.all([['actual',['RGB','Original_CoC','V2_CoC','A','V2'],b],['native',['Native_RGB','Native_Original_CoC','Native_V2_CoC'],nb]].map(async([id,keys,box])=>{const z=Math.max(1,Math.floor(Math.min(300/box[2],240/box[3])));await Promise.all(keys.map(async(k,j)=>{const im=await load(r.paths[k]);if(token!==renderId)return;const c=$(id).querySelectorAll('canvas')[j];c.width=box[2]*z;c.height=box[3]*z;const ctx=c.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(im,...box,0,0,c.width,c.height);c.dataset.source=r.paths[k];c.dataset.box=JSON.stringify(box)}))}));
}
function fmt(x){return x==null?'不适用':typeof x=='number'?x.toFixed(6):x}function table(r){let m=r.statistics;const columns=[['原网格像素数',m.native_pixels],['目标标注交集：原名义离焦p10 / p50 / p90',[m.native_before_p10,m.native_before_p50,m.native_before_p90].map(fmt).join(' / ')],['触发限幅像素 / 比例',m.native_limited_pixels+' / '+(100*m.native_limited_fraction).toFixed(2)+'%'],['实际CoC×K：原p50 → V2 p50',fmt(m.actual_before_p50)+' → '+fmt(m.actual_after_p50)],['实际CoC×K |差|均值 / 最大',fmt(m.actual_K_abs_delta_mean)+' / '+fmt(m.actual_K_abs_delta_max)],['Normalize后真实输入 |差|均值 / 最大',fmt(m.actual_normalized_abs_delta_mean)+' / '+fmt(m.actual_normalized_abs_delta_max)],['实际条件改变的显示像素 / 分母',m.actual_changed_pixels+' / '+m.display_pixels],['V2−A RGB8 |差|均值 / 最大',fmt(m.output_RGB8_abs_delta_mean)+' / '+fmt(m.output_RGB8_abs_delta_max)],['V2−A float |差|均值 / 最大',fmt(m.output_float_abs_delta_mean)+' / '+fmt(m.output_float_abs_delta_max)]];const t=document.createElement('table');for(const [a,b] of columns){const tr=t.insertRow();tr.insertCell().textContent=a;tr.insertCell().textContent=b}const div=$('summary');div.replaceChildren(t);const p=document.createElement('p');p.textContent='以上为框与目标标注的交集；全矩形含背景统计单列在 CSV。输出差异是响应，不是改善分数。';div.append(p)}
function id(){return current.sample_id+'/'+current.frame_id+'/'+current.roi_id}function stored(){try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return {}}}
function show(){current=items.find(x=>x.sample_id==$('sample').value&&x.frame_id==+$('frame').value&&x.roi_id==$('roi').value);viewBox=current.display_xywh.slice();$('description').textContent=current.sample_id+' / '+current.scene_id+' / 源帧 '+current.frame_id+'；'+current.role+'；精确位置待用户确认';$('location').src=current.paths.ROI_locations;$('legend').textContent=items.filter(x=>x.sample_id==current.sample_id&&x.frame_id==current.frame_id).map((x,i)=>(i+1)+': '+x.roi_id).join('；');for(const [k,n] of [['full','RGB'],['fullNative','Native_RGB'],['scaleA','Original_CoC_0_16'],['scaleV','V2_CoC_0_16'],['delta','CoC_abs_delta_0_1']])$(k).href=current.paths[n];$('full25').href='../subject-coc-clamp-v2-20261021/'+current.sample_id+'/';$('contact').href=current.paths.RGB.replace('RGB.png','five-panel.png');$('nativeContact').href=current.paths.RGB.replace('RGB.png','native.png');table(current);const x=stored()[id()];$('confirmed').value=x?.location_confirmation||'待用户确认';$('note').value=x?.note||'';draw().catch(e=>$('saved').textContent='图片加载失败 '+e.message)}
$('sample').onchange=frames;$('frame').onchange=rois;$('roi').onchange=show;$('reset').onclick=()=>{viewBox=current.display_xywh.slice();draw()};
function point(e){const r=$('location').getBoundingClientRect();return [Math.max(0,Math.min(1023,Math.floor((e.clientX-r.left)*1024/r.width))),Math.max(0,Math.min(575,Math.floor((e.clientY-r.top)*576/r.height)))]}
$('location').onpointerdown=e=>{dragStart=point(e);$('location').setPointerCapture(e.pointerId);e.preventDefault()};$('location').onpointerup=e=>{if(!dragStart)return;const p=point(e),x=Math.min(p[0],dragStart[0]),y=Math.min(p[1],dragStart[1]);viewBox=[x,y,Math.max(1,Math.abs(p[0]-dragStart[0])+1),Math.max(1,Math.abs(p[1]-dragStart[1])+1)];dragStart=null;draw()};
$('save').onclick=()=>{const d=stored();d[id()]={sample_id:current.sample_id,scene_id:current.scene_id,frame_id:current.frame_id,roi_id:current.roi_id,statistical_box:current.display_xywh,view_box:viewBox,location_confirmation:$('confirmed').value,note:$('note').value,visual_verdict:'待用户复核',saved_UTC:new Date().toISOString()};localStorage.setItem(KEY,JSON.stringify(d));$('saved').textContent='当前局部已保存；尚未导出。'};
$('export').onclick=()=>{const d={storage_key:KEY,records:stored()},a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{type:'application/json'}));a.download='V2-local-io-user-ROI-review-'+new Date().toISOString().replaceAll(':','-')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
frames();if(location.hash&&items.some(x=>x.sample_id==location.hash.slice(1)))choose(location.hash.slice(1));
</script></html>'''
    (out/'index.html').write_text(template.replace('__DATA__',data))


if __name__ == '__main__':
    main()
