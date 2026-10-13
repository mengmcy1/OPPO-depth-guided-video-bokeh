"""Offline PNG review controls; no inference, metric access or visual decisions."""
import html
import json
from pathlib import Path

ASSETS = Path(__file__).parent / 'assets'
BEGIN = '<!-- ATB_SYNC_VIEWER_BEGIN -->'
END = '<!-- ATB_SYNC_VIEWER_END -->'
INDEX_BEGIN = '<!-- ATB_REVIEW_TOOL_NOTE_BEGIN -->'
INDEX_END = '<!-- ATB_REVIEW_TOOL_NOTE_END -->'
VISUAL = ['视觉是否异常', '是否闪烁', '是否有halo/颜色污染/细节损坏', '视觉自然程度', '人工严重程度', '是否属于感知失败', '是否视觉复核通过']
FIELDS = ['frame_id_or_transition_id', 'ROI_xywh', 'actual_view_mode', 'user_observation', 'severity', 'confidence',
          'normal_motion_or_reasonable_defocus', 'similar_in_input_or_GT', 'normal_control', 'notes'] + VISUAL + ['actual_playback_rate']
LABELS = ['帧号或transition', 'ROI x,y,w,h（原图像素）', '实际观看方式：原速/慢放/逐帧/相邻帧/接触图', '用户观察到的现象',
          '严重程度', '置信程度', '是否正常运动或合理虚化', '输入/GT是否已有类似现象', '正常对照', '补充说明'] + VISUAL + ['实际视频倍速（1/0.5/0.25等；PNG逐帧无倍速，可留空）']


def payload(value):
    return json.dumps(value, ensure_ascii=False).replace('<', '\\u003c')


def viewer_fragment(item):
    panels = [{'label': label, 'prefix': 'originals/' + label.replace(' ', '_') + '/'} for label in item['labels']]
    config = {k: item[k] for k in ['sample_id', 'dataset', 'scene_id', 'source_group', 'split', 'package']}
    config.update(frames=25, width=1024, height=576, fps=20, panels=panels, review_url='../review.html')
    return BEGIN + '''
<link rel="stylesheet" href="../../review-tools/viewer.css">
<section id="atb-viewer" class="atb-viewer" aria-label="精确PNG逐帧查看器">
  <h2>播放与精确逐帧定位</h2>
  <div class="atb-toolbar" id="video-rates">
    <span>视频倍速：</span><button data-rate="1" aria-pressed="true">1×</button>
    <button data-rate="0.5" aria-pressed="false">0.5×</button><button data-rate="0.25" aria-pressed="false">0.25×</button>
    <output id="rate-label">当前1×</output><button id="video-note">记录当前视频观察（帧待定位）</button>
  </div>
  <p>慢放仅改变播放器倍速，不插帧、不生成新预测。请保留1×作为正常播放观感参照。MP4播放时间不用于精确帧定位。</p>
  <p class="atb-warning" id="loop-warning">循环播放的“24→0”是回到开头，不属于原视频连续transition，不能记成模型闪烁。</p>
  <div class="atb-toolbar">
    <button id="previous-frame">上一帧</button><label for="frame-slider">帧号0–24：</label>
    <input id="frame-slider" type="range" min="0" max="24" value="0" step="1"><button id="next-frame">下一帧</button>
    <output id="frame-label">帧0/24</output>
  </div>
  <div class="atb-toolbar">
    <button data-mode="frame" aria-pressed="true">同步逐帧</button><button data-mode="adjacent" aria-pressed="false">相邻两帧对照</button>
    <label for="transition-slider">连续transition起点0–23：</label><input id="transition-slider" type="range" min="0" max="23" value="0" step="1">
    <output id="transition-label">0→1</output>
  </div>
  <p>所有面板使用原始PNG并同步同一帧；方向键←/→逐帧或逐transition移动，端点不循环。输入文字时方向键仍用于编辑。</p>
  <div class="atb-toolbar">
    <button data-size="fit" aria-pressed="true">适应面板</button><button data-size="native" aria-pressed="false">原尺寸100%</button>
    <button data-size="double" aria-pressed="false">原尺寸200%</button><label><input id="roi-enabled" type="checkbox">手动框选ROI</label>
    <button id="clear-roi">清除ROI</button><output id="roi-label">ROI未选择</output>
  </div>
  <p>开启ROI后可在任意PNG面板拖框；所有方法及相邻两帧共用原图坐标。ROI默认空白，仅由用户选取。100%保留源像素，200%使用统一最近邻展示。</p>
  <div class="atb-toolbar"><button id="png-note">把当前帧/transition及ROI带入复核表</button><span id="frame-status" role="status">准备PNG</span></div>
  <div id="synchronous-panels" class="atb-panel-grid"></div>
  <details open><summary>全部25帧缩略图入口（输入RGB预览；每帧含各面板原尺寸链接）</summary><div id="all-frames" class="atb-thumbnails"></div></details>
</section>
<script id="atb-viewer-config" type="application/json">''' + payload(config) + '''</script>
<script src="../../review-tools/viewer.js"></script>
''' + END


def enhance_scene_html(text, item):
    fragment = viewer_fragment(item)
    if BEGIN in text:
        start = text.index(BEGIN)
        end = text.index(END, start) + len(END)
        return text[:start] + fragment + text[end:]
    assert '</video>' in text, 'Existing video element required'
    return text.replace('</video>', '</video>' + fragment, 1)


def enhance_index_html(text, package):
    text = text.replace('打开空白人工复核表</a>', '打开人工复核表（保留已有记录）</a>')
    note = INDEX_BEGIN + '<p class="note">观看页已增加1×/0.5×/0.25×倍速、原PNG同步逐帧、25帧缩略图、相邻两帧和手动ROI。页面中的定位按钮可新增复核条目，保留旧记录。循环24→0不是连续transition。</p>' + INDEX_END
    if INDEX_BEGIN in text:
        start = text.index(INDEX_BEGIN); end = text.index(INDEX_END, start) + len(INDEX_END)
        return text[:start] + note + text[end:]
    return text.replace('</h1>', '</h1>' + note, 1)


def install_assets(output):
    destination = Path(output) / 'review-tools'
    destination.mkdir(parents=True, exist_ok=True)
    for source in ASSETS.iterdir():
        target = destination / source.name
        if target.exists():
            assert target.read_bytes() == source.read_bytes(), 'Use explicit page updater to preserve prior asset versions'
        else:
            target.write_bytes(source.read_bytes())


def render_review_form(samples, package='B1'):
    assert package in ['B1', 'B2']
    metadata = [{k: s[k] for k in ['sample_id', 'dataset', 'scene_id', 'source_group', 'split']} for s in samples]
    config = dict(samples=metadata, fields=FIELDS, labels=LABELS, visual=VISUAL, package=package,
                  store_key=f'atb-failure-{package}-user-20261009')
    title = html.escape(package + '用户复核表')
    gate = '<p class="atb-warning">B2仅在用户完成B1记录后使用；页面功能更新不代表已完成B1或视觉复核。</p>' if package == 'B2' else ''
    return f'''<!doctype html><meta charset="utf-8"><title>{title}</title>
<link rel="stylesheet" href="../review-tools/viewer.css">
<main class="atb-review"><h1>{title}</h1>{gate}
<p>所有视觉结论由用户填写。保留当前浏览器已有记录；可为同一scene新增多个帧/ROI条目。导出会使用新的时间戳文件名，不覆盖旧导出文件。</p>
<p>循环的24→0不是连续transition。尚未定位时可保留帧号空白并自行描述观察，不必对应到五帧概览。</p>
<div id="context-banner" class="atb-note" hidden><p id="context-description"></p><button id="import-context">新增一条定位记录（不覆盖已有条目）</button></div>
<div class="atb-toolbar"><button id="add-record">新增空白记录</button><button id="export-records">导出用户JSON</button><span id="status" role="status"></span></div>
<div id="forms"></div></main>
<script id="atb-review-config" type="application/json">{payload(config)}</script>
<script src="../review-tools/review-form.js"></script>'''
