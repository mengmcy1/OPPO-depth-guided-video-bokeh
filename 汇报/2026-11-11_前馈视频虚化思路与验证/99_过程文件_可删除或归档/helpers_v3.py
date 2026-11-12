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

BASE = '/tmp/claude-0/-root-autodl-tmp/f8fd591a-d469-47bc-af6f-33dd7c2d4b3d/scratchpad/pptref/p1105.pptx'
T = '/tmp/claude-0/-root-autodl-tmp/f8fd591a-d469-47bc-af6f-33dd7c2d4b3d/scratchpad/build/tiles/'
OUT = '/tmp/claude-0/-root-autodl-tmp/f8fd591a-d469-47bc-af6f-33dd7c2d4b3d/scratchpad/build/out_v3.pptx'

BLUE = RGBColor(0x0D, 0x56, 0x9A); DARK = RGBColor(0x10, 0x1F, 0x37); LIGHT = RGBColor(0xEE, 0xF5, 0xFB)
CYAN = RGBColor(0x22, 0xC7, 0xE2); RED = RGBColor(0xC0, 0x39, 0x2B); GRAY = RGBColor(0x7F, 0x7F, 0x7F)
LBLUE = RGBColor(0x8D, 0xB8, 0xE0); WHITE = RGBColor(0xFF, 0xFF, 0xFF)

USER = '/root/autodl-tmp/OPPO-depth-guided-video-bokeh/汇报/2026-11-11_前馈视频虚化思路与验证/99_过程文件_可删除或归档/历史版本/前馈视频虚化思路与验证_1111_7页用户修改版.pptx'
prs = Presentation(USER)
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


