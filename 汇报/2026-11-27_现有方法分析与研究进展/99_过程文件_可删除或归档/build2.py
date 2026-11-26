import copy, io
exec(open('/tmp/claude-0/-root-autodl-tmp/cd48efee-2dd0-4643-97fb-2190d6ffb097/scratchpad/p1127/helpers.py', encoding='utf-8').read())
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
R = '/root/autodl-tmp/OPPO-depth-guided-video-bokeh/汇报/'
USERF = R + '2026-11-27_现有方法分析与研究进展/99_过程文件_可删除或归档/历史版本/现有方法分析与研究进展_1127_用户修改版_1126.pptx'
OUT = '/tmp/claude-0/-root-autodl-tmp/cd48efee-2dd0-4643-97fb-2190d6ffb097/scratchpad/p1127/out2.pptx'
prs = Presentation(USERF)
INNER = [l for l in prs.slide_layouts if l.name == '内页'][0]


def blank():
    s = prs.slides.add_slide(INNER)
    for ph in list(s.placeholders): ph._element.getparent().remove(ph._element)
    return s


def clone(src_path, idx):
    src = Presentation(src_path).slides[idx]
    s = blank()
    for sh in src.shapes:
        if sh.shape_type == 13:
            p = s.shapes.add_picture(io.BytesIO(sh.image.blob), sh.left, sh.top, sh.width, sh.height)
            p.crop_left, p.crop_right, p.crop_top, p.crop_bottom = sh.crop_left, sh.crop_right, sh.crop_top, sh.crop_bottom
            ln = sh._element.spPr.find(qn('a:ln'))
            if ln is not None: p._element.spPr.append(copy.deepcopy(ln))
        elif getattr(sh, 'has_chart', False) and sh.has_chart:
            ch = sh.chart
            cd = CategoryChartData(); cd.categories = list(ch.plots[0].categories)
            for se in ch.plots[0].series: cd.add_series(se.name, list(se.values))
            gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, sh.left, sh.top, sh.width, sh.height, cd)
            new_part = gf.chart_part
            src_el = copy.deepcopy(sh.chart_part._element)
            ext = src_el.find(qn('c:externalData'))
            new_ext = new_part._element.find(qn('c:externalData'))
            if ext is not None and new_ext is not None: ext.set(qn('r:id'), new_ext.get(qn('r:id')))
            elif ext is not None: src_el.remove(ext)
            new_part._element = src_el
        else:
            s.shapes._spTree.append(copy.deepcopy(sh._element))
    return s


def move(slide, pos):
    lst = prs.slides._sldIdLst
    el = [e for e in lst if prs.slides.get(int(e.get('id'))) is None or True][0]
    ids = list(lst)
    target = [e for e in ids if prs.part.related_part(e.rId) is slide.part][0]
    lst.remove(target); lst.insert(pos, target)


def set_title(slide, new):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.top < Inches(0.9) and sh.text_frame.text.strip():
            r = sh.text_frame.paragraphs[0].runs
            r[0].text = new
            for x in r[1:]: x.text = ''
            return


# keep references to user slides before re-ordering
orig = list(prs.slides)

# DOF method page (from 1105 p2)
s_dof = clone(R + '2026-11-05_3DGS景深复现/3DGS景深方法复现_1105.pptx', 1)
# ATB method page
s_atb = blank()
title(s_atb, 'Any-to-Bokeh 方法')
p = s_atb.shapes.add_picture('/root/autodl-tmp/any-to-bokeh/assets/method.png', Inches(0.75), Inches(1.1), width=Inches(11.8))
text(s_atb, 0.75, 1.1 + 11.8 * 434 / 1840 + 0.02, 11.8, 0.3, '图源：Any-to-Bokeh（ICLR 2026）', size=11, color=GRAY, align=PP_ALIGN.RIGHT)


def rich(slide, x, y, w, h, rows, size=18, space=10):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, segs in enumerate(rows):
        pp = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); pp.space_after = Pt(space)
        for t, b, c in segs:
            r = pp.add_run(); r.text = t; set_font(r, size, b, c)
    return tb


rich(s_atb, 0.85, 4.35, 11.6, 1.9, [
    [('输入　', True, BLUE), ('视频，加上由逐帧深度和焦点算出的离焦图', False, DARK)],
    [('主干　', True, BLUE), ('Stable Video Diffusion，一步生成；按深度分层的注意力加时序模块', False, DARK)],
    [('输出　', True, BLUE), ('直接生成虚化视频，VAE 解码器微调以保留细节', False, DARK)],
], size=18, space=10)
conclusion(s_atb, '模糊多少完全由逐帧深度决定，扩散模型负责把虚化画面画出来。')

# controlled-scene page (from 1111 p5)
s_ctl = clone(R + '2026-11-11_前馈视频虚化思路与验证/前馈视频虚化思路与验证_1111.pptx', 4)
set_title(s_ctl, '受控场景验证')

# redesign module page: user slide index 8 in original list
s_mod = orig[8]
sp = s_mod.shapes._spTree
for sh in list(s_mod.shapes):
    if sh.has_text_frame and sh.top < Inches(0.9): continue
    sp.remove(sh._element)
labels = ['单目视频', '几何前端', 'OVG-Net', '模块一', '模块二', '模块三', '隐藏层', 'K64 渲染']
bw, step = 1.25, 1.51
xs = [0.75 + i * step for i in range(8)]
fy, fh = 1.35, 0.7
for a in xs[:-1]: arrow(s_mod, a + bw, fy + fh / 2, a + step, fy + fh / 2, width=1.5)
for i, (x, t) in enumerate(zip(xs, labels)):
    fs = 13 if len(t) > 5 else 14
    if 3 <= i <= 5: box(s_mod, x, fy, bw, fh, [(t, fs, True, WHITE)], fill=BLUE, line=BLUE)
    elif i == 6: box(s_mod, x, fy, bw, fh, [(t, fs, True, BLUE)], fill=LIGHT, line=BLUE)
    else: box(s_mod, x, fy, bw, fh, [(t, fs, True, DARK)])
text(s_mod, xs[3], 0.98, xs[5] + bw - xs[3], 0.35, '可见层精修', size=13, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
cards = [('模块一　细粒度几何', '在 OVG-Net 后接门控深度修正分支，加入冻结的 Depth Anything 特征', '每个像素前、后两个深度和权重', '细杆深度更准，缝隙里能看到后面的墙', '进行中'),
         ('模块二　表面连通', '1 万参数的撕裂头', '相邻像素是否断开；同一表面内深度连续插值', '焦内主体不再从接缝漏光，近焦提高约 0.7 dB', '已训练'),
         ('模块三　边界分层', '边界颜色分解头', '边缘像素的前景色和背景色', '虚化后前景边缘的色边和鬼影', '待做'),
         ('隐藏层', '观测记忆 + 查询', '被挡住的背景高斯：颜色、深度、可信度', '远焦时露出真实背景', '已有')]
cw, cg = 2.85, 0.133
cy, chh = 2.5, 3.6
src_x = [xs[3], xs[4], xs[5], xs[6]]
for i, (h, net, out, gain, st) in enumerate(cards):
    x = 0.75 + i * (cw + cg)
    card = box(s_mod, x, cy, cw, chh, [(h, 16, True, BLUE)], fill=LIGHT if i < 3 else WHITE, line=BLUE, align=PP_ALIGN.LEFT)
    card.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    card.text_frame.margin_left = card.text_frame.margin_right = Emu(90000); card.text_frame.margin_top = Emu(90000)
    tf = card.text_frame
    for lab, val in [('网络', net), ('输出', out), ('改善', gain), ('状态', st)]:
        pp = tf.add_paragraph(); pp.space_before = Pt(8)
        r = pp.add_run(); r.text = lab + '　'; set_font(r, 14, True, BLUE)
        r = pp.add_run(); r.text = val; set_font(r, 14, False, DARK)
conclusion(s_mod, '可见层先把表面的位置、连接和边缘颜色做对，隐藏层再补被挡住的背景。')

# order: cover, DOF method, DOF results, DOF problems, ATB method, ATB results, ATB p1, ATB p2, idea, controlled, modules, bottleneck, data, training
two = orig[1]
order_slides = [orig[0], s_dof, orig[2], orig[3], s_atb, orig[4], orig[5], orig[6], orig[7], s_ctl, orig[8], orig[9], orig[10], orig[11]]
lst = prs.slides._sldIdLst
idmap = {prs.part.related_part(e.rId): e for e in list(lst)}
for e in list(lst): lst.remove(e)
for sl in order_slides: lst.append(idmap[sl.part])
prs.part.drop_rel(idmap[two.part].rId)
prs.part.rename_slide_parts([e.rId for e in lst])

for sl in [s_dof, s_atb, s_ctl, s_mod]:
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
# page numbers are template fields; nothing to fix
prs.save(OUT); print('saved', OUT, len(prs.slides))
