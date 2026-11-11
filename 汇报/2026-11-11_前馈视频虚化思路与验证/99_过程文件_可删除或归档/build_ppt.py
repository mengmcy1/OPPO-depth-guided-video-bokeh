import copy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.oxml.ns import qn
from lxml import etree
import numpy as np
from PIL import Image
from matplotlib import cm

BASE = '/tmp/claude-0/-root-autodl-tmp/ac32074c-db8a-4e0f-87bd-8cdb40db961a/scratchpad/pptref/p1105.pptx'
T = '/tmp/claude-0/-root-autodl-tmp/ac32074c-db8a-4e0f-87bd-8cdb40db961a/scratchpad/build/tiles/'
OUT = '/tmp/claude-0/-root-autodl-tmp/ac32074c-db8a-4e0f-87bd-8cdb40db961a/scratchpad/build/out.pptx'

BLUE = RGBColor(0x0D, 0x56, 0x9A); DARK = RGBColor(0x10, 0x1F, 0x37); LIGHT = RGBColor(0xEE, 0xF5, 0xFB)
CYAN = RGBColor(0x22, 0xC7, 0xE2); RED = RGBColor(0xC0, 0x39, 0x2B); GRAY = RGBColor(0x7F, 0x7F, 0x7F)
LBLUE = RGBColor(0x8D, 0xB8, 0xE0); WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation(BASE)
# keep cover only
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst)[1:]:
    prs.part.drop_rel(sldId.rId); sldIdLst.remove(sldId)
INNER = [l for l in prs.slide_layouts if l.name == '内页'][0]


def set_font(run, size, bold=False, color=DARK):
    run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ('a:latin', 'a:ea'):
        el = rPr.find(qn(tag))
        if el is not None: rPr.remove(el)
    lat = etree.SubElement(rPr, qn('a:latin')); lat.set('typeface', 'Arial')
    ea = etree.SubElement(rPr, qn('a:ea')); ea.set('typeface', 'Microsoft YaHei')


def text(slide, x, y, w, h, lines, size=16, bold=False, color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, autofit=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(36000)
    tf.vertical_anchor = anchor
    if isinstance(lines, str): lines = [lines]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        t, s, b, c = ln if isinstance(ln, tuple) else (ln, size, bold, color)
        r = p.add_run(); r.text = t; set_font(r, s, b, c)
    return tb


def title(slide, t):
    text(slide, 0.73, 0.31, 11.0, 0.48, t, size=28.5, bold=True, color=BLUE)


def conclusion(slide, t):
    text(slide, 0.85, 6.52, 11.65, 0.4, t, size=18, color=DARK)


def box(slide, x, y, w, h, lines, fill=LIGHT, line=BLUE, size=16, bold_first=True, color=DARK, shape=MSO_SHAPE.ROUNDED_RECTANGLE, align=PP_ALIGN.CENTER):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE: s.adjustments[0] = 0.12
    if fill is None: s.fill.background()
    else: s.fill.solid(); s.fill.fore_color.rgb = fill
    if line is None: s.line.fill.background()
    else: s.line.color.rgb = line; s.line.width = Pt(1)
    s.shadow.inherit = False
    tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Emu(54000)
    if isinstance(lines, str): lines = [lines]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        if isinstance(ln, tuple): t, sz, b, c = ln
        else: t, sz, b, c = ln, (size if i == 0 else size - 2), (bold_first and i == 0), color
        r = p.add_run(); r.text = t; set_font(r, sz, b, c)
    return s


def arrow(slide, x1, y1, x2, y2, color=BLUE, width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(width)
    ln = c.line._get_or_add_ln(); te = etree.SubElement(ln, qn('a:tailEnd')); te.set('type', 'triangle')
    return c


def line(slide, x1, y1, x2, y2, color=CYAN, width=1.25, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(width)
    if dash:
        ln = c.line._get_or_add_ln(); d = etree.SubElement(ln, qn('a:prstDash')); d.set('val', 'dash')
    return c


def pic(slide, name, x, y, w=None, h=None):
    path = T + name + '.png'
    kw = {}
    if w: kw['width'] = Inches(w)
    if h: kw['height'] = Inches(h)
    p = slide.shapes.add_picture(path, Inches(x), Inches(y), **kw)
    p.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF); p.line.width = Pt(0.5)
    return p


def new_slide(t):
    s = prs.slides.add_slide(INNER)
    for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
    title(s, t); return s


def style_chart(ch, colors, maxv=None, fmt='0.0', label_size=12):
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(13); ch.legend.font.name = 'Microsoft YaHei'
    plot = ch.plots[0]; plot.gap_width = 60; plot.overlap = -10
    plot.has_data_labels = True; dl = plot.data_labels
    dl.font.size = Pt(label_size); dl.font.bold = True; dl.font.color.rgb = DARK; dl.number_format = fmt; dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for s, c in zip(plot.series, colors):
        s.format.fill.solid(); s.format.fill.fore_color.rgb = c
    va = ch.value_axis; va.visible = False; va.has_major_gridlines = False
    if maxv is not None: va.maximum_scale = maxv
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(12); ca.tick_labels.font.name = 'Microsoft YaHei'
    ca.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF); ca.has_major_gridlines = False


def table(slide, x, y, w, rows, col_w, row_h=0.45, size=14, header_fill=BLUE, highlight_last=False, mark_colors=True):
    nr, nc = len(rows), len(rows[0])
    gs = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tb = gs.table
    for j, cw in enumerate(col_w): tb.columns[j].width = Inches(cw)
    for i in range(nr):
        tb.rows[i].height = Inches(row_h)
        for j in range(nc):
            cell = tb.cell(i, j); val = rows[i][j]
            cell.fill.solid()
            if i == 0: cell.fill.fore_color.rgb = header_fill
            elif highlight_last and i == nr - 1: cell.fill.fore_color.rgb = RGBColor(0xD6, 0xE6, 0xF5)
            else: cell.fill.fore_color.rgb = LIGHT if i % 2 == 1 else WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Emu(43200); cell.margin_top = cell.margin_bottom = Emu(18000)
            tf = cell.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if j > 0 or i == 0 else PP_ALIGN.LEFT
            r = p.add_run(); r.text = str(val)
            col = WHITE if i == 0 else DARK
            bold = (i == 0) or (j == 0) or (highlight_last and i == nr - 1)
            if mark_colors and i > 0 and j > 0:
                if val.startswith('✓'): col = BLUE; bold = True
                elif val.startswith('✗'): col = RED
                elif val.startswith('△'): col = GRAY
            set_font(r, size, bold, col)
    return gs


# ---------------- slide 1 cover ----------------
cov = prs.slides[0]
for sh in cov.shapes:
    if sh.name == 'cover-title':
        tf = sh.text_frame; ps = tf.paragraphs
        lines = ['前馈视频高斯光学虚化', '思路与可行性验证']
        for p, t in zip(ps, lines):
            for r in p.runs[1:]: r._r.getparent().remove(r._r)
            p.runs[0].text = t
        for p in ps[len(lines):]: p._p.getparent().remove(p._p)
    if sh.name == 'cover-date':
        sh.text_frame.paragraphs[0].runs[0].text = '2026.11.11'

# ---------------- slide 2: aperture = sum of views ----------------
s = new_slide('虚化可以看成光圈上多个视角的叠加')
# aperture diagram
cx, cy, R = 1.95, 3.55, 1.15
ap = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - R), Inches(cy - R), Inches(2 * R), Inches(2 * R))
ap.fill.solid(); ap.fill.fore_color.rgb = LIGHT; ap.line.color.rgb = BLUE; ap.line.width = Pt(1.5)
pts = []
for rr, n in [(0, 1), (0.42, 6), (0.84, 12)]:
    for k in range(n):
        a = 2 * np.pi * k / max(n, 1) + (0.26 if n == 12 else 0)
        pts.append((cx + rr * np.cos(a), cy + rr * np.sin(a)))
hl = [(cx - 0.98, cy + 0.26), (cx + 0.98, cy - 0.26), (cx + 0.33, cy - 0.92)]  # left, right, top (matches chosen sub-views)
for (px, py) in pts:
    d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(px - 0.045), Inches(py - 0.045), Inches(0.09), Inches(0.09))
    d.fill.solid(); d.fill.fore_color.rgb = LBLUE; d.line.fill.background()
for (px, py) in hl:
    d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(px - 0.08), Inches(py - 0.08), Inches(0.16), Inches(0.16))
    d.fill.solid(); d.fill.fore_color.rgb = CYAN; d.line.fill.background()
text(s, 0.55, 4.85, 2.8, 0.7, ['光圈平面', ('每个点是一个子相机', 14, False, DARK)], size=16, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
# sub-views
sv_x, sv_w = 3.75, 1.5
text(s, 3.1, 1.05, 2.8, 0.35, '子视角（64 个中的 3 个）', size=14, color=DARK, align=PP_ALIGN.CENTER)
order = [2, 0, 1]  # top, left, right
ys = [1.45, 3.05, 4.65]
for yy, k in zip(ys, order):
    pic(s, f'p2_sub{k}', sv_x, yy, w=sv_w)
for (px, py), yy in zip([hl[2], hl[0], hl[1]], ys):
    line(s, px + 0.08, py, sv_x, yy + sv_w / 2)
arrow(s, 5.45, 3.8, 6.05, 3.8)
text(s, 5.3, 3.3, 0.9, 0.4, '平均', size=16, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
for i, (k, lab) in enumerate([('u0', '64 个子视角平均'), ('cycles', '物理渲染真值')]):
    x0 = 6.25 + i * 3.15
    text(s, x0, 1.05, 3.0, 0.35, lab, size=14, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    pic(s, f'p2_u0_{k}_r0', x0 + 0.35, 1.45, w=2.3)
    pic(s, f'p2_u0_{k}_r1', x0 + 0.35, 3.85, w=2.3)
text(s, 6.25, 6.17, 6.3, 0.3, '下排为上排的局部放大；对焦 5.0 m、f/1.4', size=12, color=GRAY, align=PP_ALIGN.CENTER)
conclusion(s, '3DGS 渲染任意视角的成本很低，可以直接按光圈积分得到虚化，焦点和光圈都是用户参数。')

# ---------------- slide 3: partial occlusion ----------------
s = new_slide('前景虚化后会透出被挡住的背景')
# ray diagram
lens_x, top, bot = 1.25, 1.75, 5.35
lens = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(lens_x - 0.18), Inches(top), Inches(0.36), Inches(bot - top))
lens.fill.solid(); lens.fill.fore_color.rgb = LIGHT; lens.line.color.rgb = BLUE; lens.line.width = Pt(1.5)
fg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3.35), Inches(1.9), Inches(0.32), Inches(1.75))
fg.fill.solid(); fg.fill.fore_color.rgb = DARK; fg.line.fill.background()
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(5.75), Inches(1.6), Inches(0.14), Inches(3.9))
bg.fill.solid(); bg.fill.fore_color.rgb = LBLUE; bg.line.fill.background()
P = (5.75, 3.2)
# rays: from lens points to P; those crossing the foreground bar are blocked
for ly in [1.95, 2.6, 3.25, 3.9, 4.55, 5.15]:
    xb = 3.35
    yb = ly + (P[1] - ly) * (xb - lens_x) / (P[0] - lens_x)
    blocked = 1.9 <= yb <= 3.65
    if blocked:
        line(s, lens_x, ly, xb, yb, color=RED, width=1.25, dash=True)
    else:
        line(s, lens_x, ly, P[0], P[1], color=CYAN, width=1.5)
dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(P[0] - 0.08), Inches(P[1] - 0.08), Inches(0.16), Inches(0.16))
dot.fill.solid(); dot.fill.fore_color.rgb = CYAN; dot.line.fill.background()
text(s, 0.55, 5.45, 1.4, 0.35, '镜头光圈', size=14, color=DARK, align=PP_ALIGN.CENTER)
text(s, 2.75, 1.35, 1.6, 0.35, '前景（虚化）', size=14, color=DARK, align=PP_ALIGN.CENTER)
text(s, 5.1, 5.55, 1.5, 0.35, '背景（对焦）', size=14, color=DARK, align=PP_ALIGN.CENTER)
text(s, 3.3, 4.3, 2.9, 0.75, ['光圈下半部分的光线', '绕过前景，看到背后的点'], size=14, color=BLUE, align=PP_ALIGN.CENTER)
# MPIB figure
im = Image.open(T + 'p3_mpib_fig1.png'); ar = im.size[1] / im.size[0]
pw = 5.75
pic(s, 'p3_mpib_fig1', 6.85, 1.25, w=pw)
text(s, 6.85, 1.3 + pw * ar, pw, 0.35, 'MPIB（ECCV 2022）Fig. 1：前景边缘的部分遮挡效果', size=12, color=GRAY, align=PP_ALIGN.CENTER)
conclusion(s, '要做对前景边缘的虚化，必须知道前景背后的内容；单张图只能补全，视频里这些内容常在其他帧出现过。')

# ---------------- slide 4: gaps of existing feed-forward GS ----------------
s = new_slide('现有前馈高斯都难以保持当前帧的像素')
cols = [0.75, 4.95, 9.15]; cw = 3.45
heads = ['按深度直接生成高斯', '单图前馈高斯（SHARP）', '视频前馈高斯（MoVieS）']
for x0, h in zip(cols, heads):
    box(s, x0, 1.1, cw, 0.5, h, fill=LIGHT, line=BLUE, size=16)
# col1 transmittance
pic(s, 'p4_trans_VG_SHARP', cols[0] + 0.3, 1.8, w=2.85)
text(s, cols[0], 4.7, cw, 0.35, '可见层透光图（越亮漏得越多）', size=14, color=DARK, align=PP_ALIGN.CENTER)
text(s, cols[0], 5.15, cw, 0.9, ['估计深度下边缘出现裂缝', '焦内比学习式高斯低约 7 dB'], size=16, bold=True, color=RED, align=PP_ALIGN.CENTER)
# col2 zero aperture error
pic(s, 'p4_zero_err5', cols[1] + 0.3, 1.8, w=2.85)
text(s, cols[1], 4.7, cw, 0.35, '光圈为 0 时与输入图的误差 ×5', size=14, color=DARK, align=PP_ALIGN.CENTER)
text(s, cols[1], 5.15, cw, 0.9, ['重渲染输入帧只有 38–39 dB', '被遮挡的后层由网络补全'], size=16, bold=True, color=RED, align=PP_ALIGN.CENTER)
# col3 movies
pic(s, 'p4_movies_input', cols[2] + 0.05, 2.35, w=1.65)
pic(s, 'p4_movies_m1', cols[2] + 1.8, 2.35, w=1.65)
text(s, cols[2] + 0.05, 1.95, 1.65, 0.35, '输入帧', size=14, color=DARK, align=PP_ALIGN.CENTER)
text(s, cols[2] + 1.8, 1.95, 1.65, 0.35, '多帧拼出的当前帧', size=14, color=DARK, align=PP_ALIGN.CENTER)
text(s, cols[2], 4.7, cw, 0.35, '运动的叶片被多帧平均，出现重影', size=14, color=DARK, align=PP_ALIGN.CENTER)
text(s, cols[2], 5.15, cw, 0.9, ['焦内比 2.5D 方法低 2–12 dB'], size=16, bold=True, color=RED, align=PP_ALIGN.CENTER)
conclusion(s, '可见层既需要学习式的高斯构造，也需要锚定当前帧的输入像素。')

# ---------------- slide 5: system ----------------
s = new_slide('系统构想：可见层锚定当前帧，隐藏层取自视频')
SUB = RGBColor(0x59, 0x59, 0x59)
inp = box(s, 0.75, 2.55, 2.2, 1.9, [('输入', 18, True, WHITE), ('手持视频', 14, False, WHITE), ('点选对焦主体', 14, False, WHITE), ('光圈大小', 14, False, WHITE)], fill=BLUE, line=BLUE)
mx, mw, mh = 3.6, 3.55, 1.35
geo = box(s, mx, 1.15, mw, mh, [('几何前端', 17, True, DARK), ('多帧一致的深度与相机位姿', 14, False, DARK), ('借鉴：DA3 多视角深度 + 视频多视角校准', 11, False, SUB)])
vis = box(s, mx, 2.85, mw, mh, [('① 可见层', 17, True, DARK), ('当前帧像素锚定的前馈高斯', 14, False, DARK), ('借鉴：SHARP 两层前馈高斯', 11, False, SUB)])
hid = box(s, mx, 4.55, mw, mh, [('② 隐藏层', 17, True, DARK), ('视频缓存，只在光圈会看到的边缘取用', 14, False, DARK), ('借鉴：StreamSplat 视频记忆；MPIB 部分遮挡', 11, False, SUB)])
ren = box(s, 7.85, 2.65, 2.75, 1.75, [('③ 物理光圈渲染', 17, True, DARK), ('子相机渲染 → 按深度合成 → 平均', 14, False, DARK), ('借鉴：光场重对焦', 11, False, SUB)])
out = box(s, 11.05, 2.65, 1.55, 1.75, [('输出', 18, True, WHITE), ('可随时改焦点和光圈的虚化视频', 13, False, WHITE)], fill=BLUE, line=BLUE)
arrow(s, 2.95, 3.0, 3.6, 1.82); arrow(s, 2.95, 3.5, 3.6, 3.52); arrow(s, 2.95, 4.0, 3.6, 5.22)
arrow(s, mx + mw / 2, 2.5, mx + mw / 2, 2.85)
arrow(s, mx + mw, 3.52, 7.85, 3.35); arrow(s, mx + mw, 5.22, 7.85, 3.85)
arrow(s, 10.6, 3.52, 11.05, 3.52)
conclusion(s, '网络只负责场景内容，虚化由物理相机决定，换焦点、换光圈只需重新渲染。')

# ---------------- slide 6: comparison table ----------------
s = new_slide('现有工作各自覆盖了一部分')
rows = [
    ['方法', '前馈', '视频输入', '当前帧保真', '被遮挡内容', '物理光圈', '时序一致'],
    ['BokehMe 等 2.5D 方法', '✓', '单图 / 逐帧', '✓', '✗', '△', '✗'],
    ['MPIB、Dr.Bokeh', '✓', '单图', '✓', '△ 网络补全', '✓', '✗'],
    ['Any-to-Bokeh（视频扩散）', '✓', '✓', '△', '✗', '✗ 生成式', '△'],
    ['Vid2Bokeh', '✓', '单图', '△', '✗', '✗ 生成式', '✗'],
    ['SHARP 等单图前馈高斯', '✓', '单图', '△', '△ 网络补全', '✗ 新视角', '✗'],
    ['视频高斯（MoVieS、StreamSplat）', '✓', '✓', '✗', '△', '✗', '✓'],
    ['景深高斯（DOF-GS、DoF-Gaussian、CoCoGaussian）', '逐场景优化', '静态多视角', '✓', '△', '△ 近似', '—'],
    ['本方案目标', '✓', '✓', '✓', '✓ 视频取回', '✓', '✓'],
]
table(s, 0.75, 1.1, 11.8, rows, [3.7, 1.3, 1.4, 1.35, 1.4, 1.35, 1.3], row_h=0.56, size=13, highlight_last=True)
conclusion(s, '就目前检索到的文献，还没有看到同时满足这六项的工作。')

# ---------------- slide 7: controlled validation ----------------
s = new_slide('受控场景：被遮挡层与光圈渲染两项前提成立')
text(s, 0.75, 1.0, 5.6, 0.35, 'Blender 手持视频，带物理景深真值', size=16, bold=True, color=BLUE)
for i, (k, lab) in enumerate([('p9e_f001-N1.4-F1.4_gt', '对焦 1.4 m'), ('p9e_f030-N1.4-F5_gt', '对焦 5.0 m')]):
    pic(s, k, 0.85 + i * 2.85, 1.4, w=2.4)
    text(s, 0.85 + i * 2.85, 3.83, 2.4, 0.3, lab, size=13, color=DARK, align=PP_ALIGN.CENTER)
text(s, 0.75, 4.2, 5.6, 0.35, '前景边缘局部（对焦 5.0 m）', size=16, bold=True, color=BLUE)
for i, (k, lab) in enumerate([('p7_patch_gt', '物理真值'), ('p7_patch_o5', '只有可见层'), ('p7_patch_o6rec', '加入被遮挡层')]):
    pic(s, k, 0.85 + i * 1.9, 4.6, w=1.3)
    text(s, 0.85 + i * 1.9 - 0.15, 5.93, 1.6, 0.28, lab, size=12, color=DARK, align=PP_ALIGN.CENTER)
# chart hidden layer
text(s, 6.75, 1.0, 5.9, 0.35, '可恢复区域 PSNR（dB，越高越好，f/1.4）', size=16, bold=True, color=BLUE)
cd = CategoryChartData(); cd.categories = ['对焦 1.4 m', '对焦 5.0 m']
cd.add_series('只有可见层', (29.26, 29.01)); cd.add_series('加入被遮挡层', (31.12, 35.23))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.75), Inches(1.4), Inches(5.9), Inches(3.3), cd)
style_chart(gf.chart, [LBLUE, BLUE], maxv=38, fmt='0.0', label_size=13)
gf.chart.value_axis.minimum_scale = 20
# renderer accuracy cards
text(s, 6.75, 4.85, 5.9, 0.35, '64 个子视角积分 vs 物理真值（全图 PSNR）', size=16, bold=True, color=BLUE)
for i, (v, lab) in enumerate([('52.6 dB', '对焦 1.4 m'), ('53.9 dB', '对焦 5.0 m')]):
    box(s, 6.95 + i * 2.85, 5.3, 2.5, 0.95, [(v, 22, True, BLUE), (lab, 13, False, DARK)], fill=LIGHT, line=BLUE)
conclusion(s, '补上真实被遮挡内容后，可恢复区域提升 1.9–6.2 dB；子视角积分与物理真值高度一致。')

# ---------------- slide 8: charts learned GS + anchoring ----------------
s = new_slide('学习式高斯 + 像素锚定：与 BokehMe 的对比')
meths = ['BokehMe\n(DA3 深度)', 'BokehMe\n(SHARP 深度)', '按深度\n直接生成高斯', 'SHARP\n两层', 'SHARP +\n像素锚定']
psnr = {'对焦 1.4 m': (33.98, 36.21, 33.89, 35.63, 36.77), '对焦 5.0 m': (39.00, 39.27, 36.31, 38.01, 40.57)}
lp = {'对焦 1.4 m': (0.0283, 0.0197, 0.0371, 0.0163, 0.0133), '对焦 5.0 m': (0.0078, 0.0085, 0.0197, 0.0074, 0.0054)}
for i, (data, lab, fmt, mx_, mn_) in enumerate([(psnr, '全图 PSNR（dB，越高越好）', '0.0', 42, 30), (lp, '全图 LPIPS（越低越好）', '0.000', 0.042, 0)]):
    x0 = 0.75 + i * 6.05
    text(s, x0, 1.0, 5.8, 0.35, lab, size=16, bold=True, color=BLUE)
    cd = CategoryChartData(); cd.categories = meths
    for k, v in data.items(): cd.add_series(k, v)
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x0), Inches(1.35), Inches(5.85), Inches(3.75), cd)
    style_chart(gf.chart, [LBLUE, BLUE], maxv=mx_, fmt=fmt, label_size=10)
    gf.chart.value_axis.minimum_scale = mn_
rows = [['连续 8 帧的时序变化误差（越低越好）', 'BokehMe（SHARP 深度）', 'SHARP 两层', 'SHARP + 像素锚定'],
        ['对焦 1.4 m', '0.0114', '0.0126', '0.0107'],
        ['对焦 5.0 m', '0.0102', '0.0140', '0.0088']]
table(s, 0.75, 5.2, 11.8, rows, [4.0, 2.9, 2.3, 2.6], row_h=0.38, size=13, mark_colors=False)
conclusion(s, '加入像素锚定后，两档焦点的 PSNR 和 LPIPS 都优于同深度的 BokehMe，连续帧的变化误差也更低。')

# ---------------- slide 9: error maps ----------------
s = new_slide('误差分布：锚定消除了背景纹理上的误差')
heads = ['物理真值', 'BokehMe 误差', 'SHARP 两层误差', '加像素锚定后误差']
xs = [1.55, 4.1, 6.65, 9.2]; iw = 2.35
for x0, h in zip(xs, heads):
    text(s, x0, 1.0, iw, 0.35, h, size=15, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
for r, (case, lab) in enumerate([('f001-N1.4-F1.4', '对焦\n1.4 m'), ('f030-N1.4-F5', '对焦\n5.0 m')]):
    y0 = 1.4 + r * 2.5
    text(s, 0.65, y0 + 0.8, 0.85, 0.8, lab.split('\n'), size=15, bold=True, color=DARK, align=PP_ALIGN.CENTER)
    for x0, k in zip(xs, ['gt', 'o3_err', 'sharp_err', 'anchor_err']):
        pic(s, f'p9e_{case}_{k}', x0, y0, w=iw)
# colorbar
grad = (cm.inferno(np.linspace(1, 0, 256))[:, :3] * 255).astype(np.uint8)
Image.fromarray(np.repeat(grad[:, None, :], 20, axis=1)).save(T + 'cbar.png')
s.shapes.add_picture(T + 'cbar.png', Inches(11.75), Inches(1.4), Inches(0.22), Inches(4.85))
text(s, 12.0, 1.3, 0.9, 0.3, '≥0.04', size=11, color=DARK)
text(s, 12.0, 6.0, 0.9, 0.3, '0', size=11, color=DARK)
conclusion(s, '锚定消除了 SHARP 在背景纹理上的误差，整体误差低于 BokehMe（f/1.4，第 1 / 30 帧）。')

# ---------------- slide 10: next ----------------
s = new_slide('当前问题与下一步')
box(s, 0.75, 1.15, 5.75, 1.75, [('视频隐藏层：收益空间存在，规则检索拿不到', 17, True, RED), ('几何一致时，真实隐藏层在远焦全图 +1.5 dB', 14, False, DARK), ('规则检索只拿到很小一部分，需要学习式聚合', 14, False, DARK)], fill=LIGHT, line=RED)
box(s, 6.8, 1.15, 5.75, 1.75, [('风险', 17, True, RED), ('真实视频的相机位姿：估计位姿的尺度不一致', 14, False, DARK), ('目前只在一个受控场景上验证', 14, False, DARK)], fill=LIGHT, line=RED)
steps = [('1  检索诊断', '并行准备合成训练数据'), ('2  是否进入训练', '11 月底决定'), ('3  网络训练', '12 月'), ('4  真实视频评测', '1 月')]
text(s, 0.75, 3.35, 2.6, 0.38, '当前', size=14, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
for i, (a, b) in enumerate(steps):
    x0 = 0.75 + i * 2.98
    cur = i == 0
    box(s, x0, 3.75, 2.6, 1.6, [(a, 19, True, WHITE if cur else DARK), (b, 14, False, WHITE if cur else DARK)], fill=BLUE if cur else LIGHT, line=BLUE)
    if i < 3: arrow(s, x0 + 2.62, 4.55, x0 + 2.96, 4.55)
conclusion(s, '目标 ICCV 2027（3 月上旬截稿），11 月底根据隐藏层的验证结果决定是否进入训练。')

# remove inherited theme effects (shadow / reflection) on all added shapes
for sl in list(prs.slides)[1:]:
    for sh in sl.shapes:
        spPr = sh._element.find(qn('p:spPr'))
        if spPr is None: continue
        for el in spPr.findall(qn('a:effectLst')): spPr.remove(el)
        eff = etree.Element(qn('a:effectLst'))
        anchor = None
        for tag in ('a:scene3d', 'a:sp3d', 'a:extLst'):
            anchor = spPr.find(qn(tag))
            if anchor is not None: break
        if anchor is not None: anchor.addprevious(eff)
        else: spPr.append(eff)
prs.save(OUT)
print('saved', OUT, len(prs.slides))
