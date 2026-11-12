p = '/tmp/claude-0/-root-autodl-tmp/f8fd591a-d469-47bc-af6f-33dd7c2d4b3d/scratchpad/build/build_ppt.py'
s = open(p, encoding='utf-8').read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)


# slide 2: drop gray caption
rep("text(s, 6.25, 6.17, 6.3, 0.3, '下排为上排的局部放大；对焦 5.0 m、f/1.4', size=12, color=GRAY, align=PP_ALIGN.CENTER)\n", "")
# slide 3: plain label instead of small gray figure caption
rep("text(s, 6.85, 1.3 + pw * ar, pw, 0.35, 'MPIB（ECCV 2022）Fig. 1：前景边缘的部分遮挡效果', size=12, color=GRAY, align=PP_ALIGN.CENTER)",
    "text(s, 6.85, 1.35 + pw * ar, pw, 0.35, 'MPIB（ECCV 2022）', size=14, color=DARK, align=PP_ALIGN.CENTER)")
# slide 5: new module boxes
rep("geo = box(s, mx, 1.15, mw, mh, [('几何前端', 17, True, DARK), ('多帧一致的深度与相机位姿', 14, False, DARK), ('借鉴：DA3 多视角深度 + 视频多视角校准', 11, False, SUB)])",
    "geo = box(s, mx, 1.15, mw, mh, [('几何前端', 17, True, DARK), ('现成深度 / 位姿 + 多视角校准', 14, False, DARK), ('DA3 深度；稠密光流三角化校准', 11, False, SUB)])")
rep("vis = box(s, mx, 2.85, mw, mh, [('① 可见层', 17, True, DARK), ('当前帧像素锚定的前馈高斯', 14, False, DARK), ('借鉴：SHARP 两层前馈高斯', 11, False, SUB)])",
    "vis = box(s, mx, 2.85, mw, mh, [('① 可见层', 17, True, DARK), ('当前帧像素锚定的高斯', 14, False, DARK), ('第一版直接使用 SHARP 几何', 11, False, SUB)])")
rep("hid = box(s, mx, 4.55, mw, mh, [('② 隐藏层', 17, True, DARK), ('视频缓存，只在光圈会看到的边缘取用', 14, False, DARK), ('借鉴：StreamSplat 视频记忆；MPIB 部分遮挡', 11, False, SUB)])",
    "hid = box(s, mx, 4.55, mw, mh, [('② 隐藏层', 17, True, DARK), ('观测记忆 + 光圈条件 Query', 14, False, DARK), ('几何 / 外观联合修正', 11, False, SUB)])")
# slide 9 conclusion without parenthetical
rep("conclusion(s, '锚定消除了 SHARP 在背景纹理上的误差，整体误差低于 BokehMe（f/1.4，第 1 / 30 帧）。')",
    "conclusion(s, '锚定消除了 SHARP 在背景纹理上的误差，整体误差低于 BokehMe。')")

# old slide 10 -> new slides 10-13
i0 = s.index('# ---------------- slide 10: next ----------------')
i1 = s.index('# remove inherited theme effects')
new = r'''# ---------------- slide 10: rule retrieval gains ----------------
GRAYB = RGBColor(0xA6, 0xA6, 0xA6); MBLUE = RGBColor(0x4A, 0x86, 0xC8)
s = new_slide('视频隐藏层：简单规则已拿回约一半收益')
series = [('BokehMe（校准深度）', (29.04, 28.67), (37.40, 40.21)),
          ('只有可见层', (29.96, 29.44), (37.82, 40.99)),
          ('+ 原规则检索', (30.13, 29.48), (37.91, 40.93)),
          ('+ 来源边缘剔除', (30.51, 31.00), (37.99, 41.85)),
          ('+ 真实隐藏层（上限）', (30.93, 32.63), (38.02, 42.46))]
cols5 = [GRAYB, LBLUE, MBLUE, BLUE, DARK]
for i, (lab, idx, mn, mx_) in enumerate([('可恢复区域 PSNR（dB）', 1, 26, 34), ('全图 PSNR（dB）', 2, 35, 44)]):
    x0 = 0.75 + i * 6.05
    text(s, x0, 1.0, 5.8, 0.35, lab, size=16, bold=True, color=BLUE)
    cd = CategoryChartData(); cd.categories = ['对焦 1.4 m', '对焦 5.0 m']
    for sr in series: cd.add_series(sr[0], sr[idx])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x0), Inches(1.4), Inches(5.85), Inches(4.85), cd)
    style_chart(gf.chart, cols5, maxv=mx_, fmt='0.0', label_size=10)
    gf.chart.value_axis.minimum_scale = mn
    gf.chart.legend.font.size = Pt(11)
conclusion(s, '远焦时完整流程比校准后的 BokehMe 全图高 1.6 dB；规则已拿回约 60% 的差距。')

# ---------------- slide 11: what remains ----------------
s = new_slide('剩下的误差：取错了表面层')
text(s, 0.75, 1.0, 6.3, 0.35, '可恢复区域的误差来自哪里', size=16, bold=True, color=BLUE)
parts = [('深度偏近', (0.394, 0.572), BLUE), ('深度偏远', (0.210, 0.180), LBLUE), ('运动物体', (0.063, 0.088), GRAYB),
         ('颜色误差', (0.053, 0.024), CYAN), ('其他', (0.016, 0.002), RGBColor(0xD9, 0xD9, 0xD9)),
         ('深度与颜色都正确', (0.264, 0.134), RGBColor(0xDC, 0xE6, 0xF0))]
cd = CategoryChartData(); cd.categories = ['对焦 5.0 m', '对焦 1.4 m']
for nm, v, _ in parts: cd.add_series(nm, (v[1], v[0]))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED_100, Inches(0.75), Inches(1.4), Inches(6.3), Inches(4.85), cd)
ch = gf.chart
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
ch.legend.font.size = Pt(12); ch.legend.font.name = 'Microsoft YaHei'
plot = ch.plots[0]; plot.gap_width = 45; plot.overlap = 100
for sr, (nm, v, c) in zip(plot.series, parts):
    sr.format.fill.solid(); sr.format.fill.fore_color.rgb = c
    if nm not in ('其他', '颜色误差'):
        sr.data_labels.show_value = True; sr.data_labels.number_format = '0%'; sr.data_labels.number_format_is_linked = False
        sr.data_labels.font.size = Pt(12); sr.data_labels.font.bold = True
        sr.data_labels.font.color.rgb = WHITE if c in (BLUE,) else DARK
ch.value_axis.visible = False; ch.value_axis.has_major_gridlines = False
ch.category_axis.tick_labels.font.size = Pt(13); ch.category_axis.tick_labels.font.name = 'Microsoft YaHei'
for i, (v, lab) in enumerate([('99.8%', '有候选的比例'), ('23%', '其中深度正确')]):
    box(s, 7.45 + i * 2.6, 1.4, 2.4, 1.0, [(v, 22, True, BLUE if i == 0 else RED), (lab, 13, False, DARK)], fill=LIGHT, line=BLUE if i == 0 else RED)
text(s, 7.45, 2.65, 5.0, 0.35, '单独把一项换成真值，能收回多少差距', size=16, bold=True, color=BLUE)
rows = [['替换内容', '1.4 m', '5.0 m'],
        ['只换深度', '−140%', '−121%'],
        ['只换颜色', '+88%', '+15%'],
        ['只补漏检', '+17%', '0%'],
        ['对照：来源边缘剔除规则', '+59%', '+58%']]
table(s, 7.45, 3.05, 5.0, rows, [2.6, 1.2, 1.2], row_h=0.5, size=14, mark_colors=False)
conclusion(s, '深度错误和来源边缘混色耦合在一起，单独修一项无效，需要联合修正。')

# ---------------- slide 12: first hidden-layer network ----------------
s = new_slide('第一版隐藏层网络')
steps = [('观测记忆', '多表面假设 · 来源记录', LIGHT, DARK),
         ('候选', '光圈去遮挡区内，每个位置 8 个', LIGHT, DARK),
         ('几何解码', '选择 · 深度修正 · 可信度', BLUE, WHITE),
         ('外观解码', '按新位置回来源帧取色 · 混色识别', BLUE, WHITE),
         ('隐藏高斯', '每个位置 2 个', LIGHT, DARK),
         ('K=64 物理渲染', '与可见层一起按深度合成', LIGHT, DARK)]
bw, gap, y0 = 1.7, 0.32, 1.75
text(s, 0.75 + 2 * (bw + gap), 1.25, 2 * bw + gap, 0.4, '训练的部分', size=15, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
for i, (a, b, f, c) in enumerate(steps):
    x0 = 0.75 + i * (bw + gap)
    box(s, x0, y0, bw, 1.55, [(a, 16, True, c), (b, 12, False, c)], fill=f, line=BLUE)
    if i < 5:
        if i == 2:
            arrow(s, x0 + bw, y0 + 0.5, x0 + bw + gap, y0 + 0.5)
            arrow(s, x0 + bw + gap, y0 + 1.05, x0 + bw, y0 + 1.05)
        else:
            arrow(s, x0 + bw, y0 + 0.78, x0 + bw + gap, y0 + 0.78)
cards = [('标签按子光圈光线定义', '候选与任一子光圈光线击中的表面一致，即为正确'),
         ('颜色只取自真实来源帧', '网络只做选择、修正与融合，不凭空生成'),
         ('可信度不乘进不透明度', '只用于选择、拒绝和触发补全')]
for i, (a, b) in enumerate(cards):
    box(s, 0.75 + i * 4.0, 3.85, 3.75, 1.55, [(a, 17, True, BLUE), (b, 14, False, DARK)], fill=LIGHT, line=BLUE)
conclusion(s, '第一版只恢复静态隐藏背景；可见层固定为 SHARP + 颜色锚定，不参与训练。')

# ---------------- slide 13: training plan ----------------
s = new_slide('训练计划与判断标准')
text(s, 0.75, 0.95, 3.55, 0.35, '下一步', size=14, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
stg = [('阶段 1', '静态场景 + 运动相机', '256 / 5 帧 · 直接监督'),
       ('阶段 2', '512 / 13 帧', '+ K=64 渲染损失 · 位姿扰动'),
       ('阶段 3', '加入运动前景', '时序损失（小权重）')]
for i, (a, b, c) in enumerate(stg):
    x0 = 0.75 + i * 4.15
    cur = i == 0
    col = WHITE if cur else DARK
    box(s, x0, 1.3, 3.55, 1.4, [(a, 18, True, col), (b, 14, False, col), (c, 14, False, col)], fill=BLUE if cur else LIGHT, line=BLUE)
    if i < 2: arrow(s, x0 + 3.57, 2.0, x0 + 4.13, 2.0)
crit = [('成功线', '留出场景上收回 75–80% 的差距', '或远焦全图比规则方法高 0.3 dB', BLUE),
        ('转向点', '1–2 周内仍达不到规则方法', '转为"规则 + 光圈分析 + 评测集"', RED),
        ('风险', '真实视频位姿 · 视频补全基线', '目前只有一个受控场景', RED)]
for i, (a, b, c, lc) in enumerate(crit):
    box(s, 0.75 + i * 4.15, 3.0, 3.55, 1.5, [(a, 17, True, lc), (b, 14, False, DARK), (c, 14, False, DARK)], fill=LIGHT, line=lc)
tl = [('11 月下旬', '数据管线 · 阶段 1'), ('12 月', '阶段 2 · 阶段 3'), ('1 月', '真实视频 · 对比实验'), ('2 月', '论文写作')]
for i, (a, b) in enumerate(tl):
    x0 = 0.75 + i * 3.0
    box(s, x0, 4.85, 2.7, 1.1, [(a, 16, True, BLUE), (b, 14, False, DARK)], fill=WHITE, line=BLUE)
    if i < 3: arrow(s, x0 + 2.72, 5.4, x0 + 2.98, 5.4)
conclusion(s, '目标 ICCV 2027（3 月上旬截稿）；训练开始后 1–2 周判断第一版是否成立。')

'''
s = s[:i0] + new + s[i1:]
open(p, 'w', encoding='utf-8').write(s)
print('patched')
