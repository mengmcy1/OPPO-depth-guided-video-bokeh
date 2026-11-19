exec(open('/tmp/claude-0/-root-autodl-tmp/cd48efee-2dd0-4643-97fb-2190d6ffb097/scratchpad/build/helpers.py', encoding='utf-8').read())

SEP = RGBColor(0xBF, 0xBF, 0xBF)

# keep the cover only
ids = prs.slides._sldIdLst
for sid in list(ids)[1:]:
    prs.part.drop_rel(sid.rId); ids.remove(sid)
cover = prs.slides[0]
for sh in cover.shapes:
    if not sh.has_text_frame: continue
    for para in sh.text_frame.paragraphs:
        for r in para.runs:
            if r.text == '思路与可行性验证': r.text = '数据准备与训练策略'
            if r.text == '2026.11.11': r.text = '2026.11.19'


def vsep(s, x, y1, y2):
    line(s, x, y1, x, y2, color=SEP, width=1)


# ---------------- 2 ----------------
s = new_slide('可见层改为自己训练')
arrow(s, 5.95, 2.5, 7.35, 2.5, width=2)
box(s, 0.75, 1.5, 5.0, 2.0, [('第一版', 20, True, BLUE), ('可见层：SHARP，直接使用', 17, False, DARK),
                             ('隐藏层：观测记忆 + 查询', 17, False, DARK)])
box(s, 7.55, 1.5, 5.0, 2.0, [('第二版', 20, True, WHITE), ('可见层：OVG-Net，自己训练', 17, False, WHITE),
                             ('隐藏层：保留，并扩展到边界', 17, False, WHITE)], fill=BLUE, line=BLUE)
for x, t in [(0.75, 'SHARP 会占据被遮挡的空间'), (6.75, '几何是收益最大的因素')]:
    text(s, x, 4.3, 5.85, 0.6, t, size=20, bold=True, color=BLUE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
vsep(s, 6.665, 4.35, 4.85)
conclusion(s, '可见层只表达看得见的部分，背景只由隐藏层从视频观测中给出。')

# ---------------- 3 ----------------
s = new_slide('OVG-Net 网络结构')
# arrows first
arrow(s, 2.65, 1.45, 3.0, 1.35); arrow(s, 2.65, 1.45, 3.0, 2.2)
arrow(s, 2.65, 3.05, 3.0, 3.05)
for y in (1.35, 2.2, 3.05): arrow(s, 5.3, y, 5.65, y)
arrow(s, 7.45, 1.75, 7.8, 1.75); arrow(s, 7.45, 3.05, 7.8, 3.05)
arrow(s, 10.1, 1.75, 10.45, 1.75); arrow(s, 10.1, 3.05, 10.45, 3.05)
line(s, 2.65, 4.1, 8.95, 4.1, color=BLUE, width=1.5); arrow(s, 8.95, 4.1, 8.95, 3.5)
# column 1: inputs
box(s, 0.75, 1.05, 1.9, 0.8, [('当前帧', 16, True, DARK)])
box(s, 0.75, 2.65, 1.9, 0.8, [('深度先验 ×3', 16, True, DARK), ('各带可信度', 13, False, DARK)])
box(s, 0.75, 3.7, 1.9, 0.8, [('相邻帧', 16, True, DARK), ('带位姿', 13, False, DARK)])
# column 2: encoders
box(s, 3.0, 1.0, 2.3, 0.7, [('冻结 DINOv2', 15, True, DARK), ('语义与全局几何', 13, False, DARK)])
box(s, 3.0, 1.85, 2.3, 0.7, [('高分辨率 CNN', 15, True, DARK), ('边缘细节', 13, False, DARK)])
box(s, 3.0, 2.7, 2.3, 0.7, [('先验编码', 15, True, DARK), ('三种先验分别编码', 13, False, DARK)])
# column 3: decoder
box(s, 5.65, 1.0, 1.8, 2.4, [('逐级解码', 16, True, DARK), ('逐像素特征', 13, False, DARK)])
# column 4: trained depth steps
box(s, 7.8, 1.3, 2.3, 0.9, [('先验选择 + 残差', 15, True, WHITE), ('逐像素选用最可靠的先验', 12, False, WHITE)], fill=BLUE, line=BLUE)
box(s, 7.8, 2.6, 2.3, 0.9, [('局部多视角修正', 15, True, WHITE), ('只在低可信与深度断层处', 12, False, WHITE)], fill=BLUE, line=BLUE)
# column 5: per-pixel outputs
box(s, 10.45, 1.0, 2.1, 3.4, [('每个像素输出', 16, True, BLUE), ('主表面深度', 15, False, DARK), ('背景深度锚点', 15, False, DARK),
                              ('前景覆盖率', 15, False, DARK), ('纯前景色', 15, False, DARK), ('可信度', 15, False, DARK)])
# system strip
text(s, 0.75, 4.75, 4.0, 0.4, '在系统中的位置', size=15, bold=True, color=BLUE)
xs = [0.75, 3.15, 5.55, 7.95, 10.35]
for x0, x1 in zip(xs[:-1], xs[1:]): arrow(s, x0 + 2.0, 5.55, x1, 5.55)
labels = ['几何前端（冻结）', 'OVG-Net 可见层', '按新几何重建记忆', '隐藏层查询', 'K=64 渲染']
for i, (x, t) in enumerate(zip(xs, labels)):
    if i == 1: box(s, x, 5.25, 2.0, 0.6, [(t, 14, True, WHITE)], fill=BLUE, line=BLUE)
    else: box(s, x, 5.25, 2.0, 0.6, [(t, 14, True, DARK)])
conclusion(s, '可训练参数约 65 万；焦点和光圈不作为输入，可见层每帧只算一次。')

# ---------------- 4 ----------------
s = new_slide('五档数据')
cols = [('G', '公开合成几何', 'TartanAir V2', '11 环境 · 10,032 帧', '深度、位姿'),
        ('C', '卡片场景', '自建，解析计算', '200 段 · 1,605 帧', '隐藏面、光线命中\n景深参考'),
        ('B', '自有 Blender', '网格求交 + Cycles', '10 场景 · 130 帧', '前后两层、覆盖率\n景深真值'),
        ('R', '真实视频', 'DL3DV', '60 段 · 291 帧', '零光圈重建\n多视角一致'),
        ('A', '抠图合成', 'AM-2k + BG-20k', '前景 1,800 · 背景 2,048', '前景覆盖率\n纯前景色')]
cw, gap = 2.3, 0.075
for i, (k, name, src, scale, role) in enumerate(cols):
    x = 0.75 + i * (cw + gap)
    pic(s, k, x, 1.05, w=cw)
    text(s, x, 2.9, cw, 0.42, f'{k}　{name}', size=17, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    text(s, x, 3.35, cw, 0.36, src, size=14, color=DARK, align=PP_ALIGN.CENTER)
    text(s, x, 3.72, cw, 0.36, scale, size=12, bold=True, color=DARK, align=PP_ALIGN.CENTER)
    text(s, x, 4.15, cw, 0.7, role.split('\n'), size=14, color=DARK, align=PP_ALIGN.CENTER)
for i in range(1, 5): vsep(s, 0.75 + i * (cw + gap) - gap / 2, 2.95, 4.75)
box(s, 0.75, 5.3, 11.8, 0.6, [('独立测试集已封存：18 个 Blender 场景 + 1 个卡片序列，不参与训练与任何选择', 15, False, DARK)], fill=WHITE, line=CYAN)
conclusion(s, '合成数据负责精确监督，真实数据负责适应真实画面；每个样本只用它真实具有的标签。')

# ---------------- 5 ----------------
s = new_slide('损失与训练阶段')
groups = [('几何', '视差换算的深度 · 前后两层 · 覆盖率'),
          ('遮挡与光学', '误占遮挡空间惩罚 · 零光圈重建 · 景深渲染'),
          ('外观', '纯前景色 · 合成一致')]
gw = 11.8 / 3
for i, (g, items) in enumerate(groups):
    x = 0.75 + i * gw
    text(s, x, 1.0, gw, 0.42, g, size=18, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    text(s, x, 1.42, gw, 0.4, items, size=14, color=DARK, align=PP_ALIGN.CENTER)
for i in (1, 2): vsep(s, 0.75 + i * gw, 1.05, 1.8)
stages = [('阶段 1　可见层几何', '已完成', ['1a：G60 / C25 / B15', '1b：C40 / B35 / G15 / R10', '学深度、前后两层与覆盖率']),
          ('阶段 2　边界拆分', '进行中', ['2a：A70 / C30', '2b：C40 / B40 / A20', '几何冻结，训练前景色与足迹']),
          ('阶段 3　联合光学', '', ['B 光学档 50', 'C30 / R10 / G10', '加入 K=64 景深损失']),
          ('阶段 4　时序与动态', '', ['运动前景、连续推焦', '', '可见层时序适配'])]
bw, bg = 2.85, 0.13
for i in range(3): arrow(s, 0.75 + (i + 1) * bw + i * bg, 3.9, 0.75 + (i + 1) * (bw + bg), 3.9)
for i, (h, st, lines) in enumerate(stages):
    x = 0.75 + i * (bw + bg)
    cur = (i == 1)
    fg = WHITE if cur else DARK
    hd = [(h, 17, True, WHITE if cur else BLUE)]
    hd.append((st if st else ' ', 14, False, WHITE if cur else GRAY))
    hd += [(' ', 8, False, fg)] + [(l if l else ' ', 13, False, fg) for l in lines]
    box(s, x, 2.3, bw, 3.2, hd, fill=BLUE if cur else LIGHT, line=BLUE)
conclusion(s, '每个阶段只加入当前要学的量，后面阶段保留前面的数据，防止遗忘。')

# ---------------- 6 ----------------
s = new_slide('下周计划')
todo = [('完成阶段 2 训练与评价', '2b 共 3 万步，重点看边缘 ±2 px 与零光圈重建'),
        ('两组固定对照', '开关第二遍查询；边界不拆分 / 真值背景 / 检索背景'),
        ('查明加入隐藏层后焦内变差的原因', '逐光线检查前景内部漏光：面片缝隙、排序或颜色'),
        ('准备阶段 3', '扩充 Blender 景深真值场景；测 K=64 景深损失的显存与耗时')]
for i, (t, d) in enumerate(todo):
    y = 1.15 + i * 1.25
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.85), Inches(y + 0.2), Inches(0.6), Inches(0.6))
    c.fill.solid(); c.fill.fore_color.rgb = BLUE; c.line.fill.background(); c.shadow.inherit = False
    tf = c.text_frame; tf.margin_left = tf.margin_right = 0; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = str(i + 1); set_font(r, 18, True, WHITE)
    box(s, 1.7, y, 10.85, 1.0, [(t, 18, True, DARK), (d, 15, False, DARK)], align=PP_ALIGN.LEFT)
conclusion(s, '阶段 2 结束后，根据边界拆分是否有收益决定阶段 3 的重点。')

# remove inherited theme effects on new slides
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
