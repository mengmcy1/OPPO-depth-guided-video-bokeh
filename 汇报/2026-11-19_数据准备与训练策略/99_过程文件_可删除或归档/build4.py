exec(open('/tmp/claude-0/-root-autodl-tmp/cd48efee-2dd0-4643-97fb-2190d6ffb097/scratchpad/build/helpers.py', encoding='utf-8').read())
OUT = '/tmp/claude-0/-root-autodl-tmp/cd48efee-2dd0-4643-97fb-2190d6ffb097/scratchpad/build/out4.pptx'

ids = prs.slides._sldIdLst
for sid in list(ids)[1:]:
    prs.part.drop_rel(sid.rId); ids.remove(sid)
for sh in prs.slides[0].shapes:
    if sh.has_text_frame:
        for para in sh.text_frame.paragraphs:
            for r in para.runs:
                if r.text == '思路与可行性验证': r.text = '数据准备与训练策略'
                if r.text == '2026.11.11': r.text = '2026.11.19'


def rich(slide, x, y, w, h, rows, size=20, space=14):
    """rows: list of [(text, bold, color), ...]; one paragraph per row."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, segs in enumerate(rows):
        sp = space
        if isinstance(segs, tuple): segs, sp = segs
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(sp)
        for t, b, c in segs:
            r = p.add_run(); r.text = t; set_font(r, size, b, c)
    return tb


# 2
s = new_slide('可见层改为自己训练')
rich(s, 0.85, 1.3, 11.6, 4.8, [
    [('第一版　', True, BLUE), ('可见层直接用 SHARP，只训练隐藏层', False, DARK)],
    [('第二版　', True, BLUE), ('可见层改为自己训练（OVG-Net），隐藏层查询保留', False, DARK)],
    [('', False, DARK)],
    [('原因一　', True, BLUE), ('SHARP 会把被遮挡的空间补满，挡住隐藏层找回的真实背景', False, DARK)],
    [('原因二　', True, BLUE), ('几何准不准是收益最大的因素，值得自己学', False, DARK)],
], size=22, space=16)
conclusion(s, '可见层只表达看得见的部分，背景只由隐藏层从视频观测中给出。')

# 3
s = new_slide('OVG-Net 网络结构')
xs = [0.75, 3.85, 6.95, 10.05]; w = 2.5
for a, b in zip(xs[:-1], xs[1:]): arrow(s, a + w, 2.6, b, 2.6, width=2)
items = [('输入', '当前帧 · 相邻帧', '三种深度先验'),
         ('编码', '冻结 DINOv2', '高分辨率 CNN'),
         ('深度', '逐像素选择先验', '局部多视角修正'),
         ('输出', '可见表面深度 · 覆盖率', '背景深度锚点 · 前景色')]
for i, (x, (h, l1, l2)) in enumerate(zip(xs, items)):
    cur = (i == 2)
    box(s, x, 1.6, w, 2.0, [(h, 20, True, WHITE if cur else BLUE), (l1, 15, False, WHITE if cur else DARK), (l2, 15, False, WHITE if cur else DARK)],
        fill=BLUE if cur else LIGHT, line=BLUE)
rich(s, 0.85, 4.3, 11.6, 1.6, [
    [('三种先验各有短板（SHARP、DepthSplat、三角化），由网络逐像素选择，再在边缘处用相邻帧修正。', False, DARK)],
    [('背景深度锚点只交给隐藏层去查询，可见层自己不放背景。', False, DARK)],
], size=17, space=10)
conclusion(s, '可训练参数约 65 万；焦点和光圈不作为输入，换焦点时可见层不用重算。')

# 4
s = new_slide('五档数据')
cols = [('G', '公开合成几何', 'TartanAir V2 · 2 万帧'), ('C', '卡片场景', '自建 · 5,000 段'),
        ('B', '自有 Blender', '自建 · 300 个场景'), ('R', '真实视频', 'DL3DV · 500 段'),
        ('A', '抠图合成', 'AM-2k · 1,800 前景')]
cw, gap = 2.3, 0.075
for i, (k, name, sub) in enumerate(cols):
    x = 0.75 + i * (cw + gap)
    pic(s, k, x, 1.3, w=cw)
    text(s, x, 3.15, cw, 0.42, f'{k}　{name}', size=17, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    text(s, x, 3.6, cw, 0.4, sub, size=14, color=DARK, align=PP_ALIGN.CENTER)
rich(s, 0.85, 4.55, 11.6, 1.5, [
    [('合成数据（G、C、B、A）给精确标签：深度、前后两层、覆盖率、被遮挡的面、景深真值。', False, DARK)],
    [('真实视频（R）只做自监督：零光圈还原原图、静止区域多帧一致。', False, DARK)],
], size=17, space=10)
conclusion(s, '每个样本只用它真实具有的标签计算损失。')

# 6
s = new_slide('训练阶段与目前结果')
rich(s, 0.85, 1.2, 11.6, 5.0, [
    [('阶段 1　可见层几何', True, BLUE), ('　已在小数据上完成，将用扩充数据从头重训', False, GRAY)],
    ([('学深度、前后两层与覆盖率；远焦比 SHARP 高 6.4 dB，近焦还低 1.2 dB', False, DARK)], 22),
    [('阶段 2　边界拆分', True, BLUE), ('　旧数据上试跑中，正式训练接在重训之后', False, GRAY)],
    ([('几何冻结，学前景色与足迹；边界像素的背景交给隐藏层', False, DARK)], 22),
    [('阶段 3　联合光学', True, BLUE)],
    ([('加入 K=64 景深损失，可见层与隐藏层一起微调', False, DARK)], 22),
    [('阶段 4　时序与动态', True, BLUE)],
    [('运动前景、连续推焦', False, DARK)],
], size=19, space=4)
conclusion(s, '重训直接沿用阶段 1 最后定下的配置，不再重复探索。')

# 7
s = new_slide('下周计划')
rich(s, 0.85, 1.3, 11.6, 4.8, [
    [('1　', True, BLUE), ('接入全部素材来源，按素材划分训练 / 开发 / 测试，定版场景生成器', False, DARK)],
    [('2　', True, BLUE), ('批量生成扩充数据：Blender 300 个场景、卡片 5,000 段、真实视频 500 段等', False, DARK)],
    [('3　', True, BLUE), ('数据就绪后从头重训阶段 1，在新的 15 个开发场景上评价', False, DARK)],
    [('4　', True, BLUE), ('旧数据上的阶段 2 试跑做完评价，先看边界拆分和前景漏光的初步情况', False, DARK)],
], size=20, space=18)

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
prs.save(OUT); print('saved', OUT, len(prs.slides))
