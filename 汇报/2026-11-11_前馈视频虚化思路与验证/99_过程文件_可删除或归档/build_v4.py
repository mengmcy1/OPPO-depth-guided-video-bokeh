exec(open('/tmp/claude-0/-root-autodl-tmp/f8fd591a-d469-47bc-af6f-33dd7c2d4b3d/scratchpad/build/helpers_v3.py', encoding='utf-8').read())
OUT = '/tmp/claude-0/-root-autodl-tmp/f8fd591a-d469-47bc-af6f-33dd7c2d4b3d/scratchpad/build/out_v4.pptx'

sldIdLst = prs.slides._sldIdLst
assert len(sldIdLst) == 7
last = list(sldIdLst)[6]
prs.part.drop_rel(last.rId); sldIdLst.remove(last)
n_keep = len(prs.slides)
SUBC = RGBColor(0x59, 0x59, 0x59)
GRAYB = RGBColor(0xA6, 0xA6, 0xA6)


def dot(s, x, y, r, c):
    d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - r), Inches(y - r), Inches(2 * r), Inches(2 * r))
    d.fill.solid(); d.fill.fore_color.rgb = c; d.line.fill.background(); return d


def polyline(s, pts, color=BLUE, width=2.0, dash=False):
    E = 914400
    b = s.shapes.build_freeform(int(pts[0][0] * E), int(pts[0][1] * E), scale=1.0)
    b.add_line_segments([(int(x * E), int(y * E)) for x, y in pts[1:]], close=False)
    sh = b.convert_to_shape(); sh.fill.background()
    sh.line.color.rgb = color; sh.line.width = Pt(width)
    if dash:
        ln = sh.line._get_or_add_ln(); d = etree.SubElement(ln, qn('a:prstDash')); d.set('val', 'dash')
    return sh


def lens(s, x, top, bot):
    l = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - 0.16), Inches(top), Inches(0.32), Inches(bot - top))
    l.fill.solid(); l.fill.fore_color.rgb = LIGHT; l.line.color.rgb = BLUE; l.line.width = Pt(1.5)


# ================= 7. hidden-layer query overview =================
s = new_slide('隐藏层查询：整体链路')
stages = [('观测记忆', '只存真实观测\n同一位置多个表面', LIGHT, DARK),
          ('光圈条件查询', '只在光圈会看到的位置\n候选按光圈可见度排序', BLUE, WHITE),
          ('联合修正', '几何与颜色一起改\n修正位置 · 重新取色', BLUE, WHITE),
          ('隐藏高斯', '每个位置 2 个', LIGHT, DARK),
          ('K=64 物理渲染', '与可见层按深度合成', LIGHT, DARK)]
xs = [0.75, 3.15, 5.85, 8.55, 10.45]; ws = [2.1, 2.4, 2.4, 1.6, 2.1]
y0, hh = 1.9, 2.0
for (a, b, f, c), x0, w in zip(stages, xs, ws):
    lines_ = [(a, 17, True, c)] + [(t, 13, False, c) for t in b.split('\n')]
    box(s, x0, y0, w, hh, lines_, fill=f, line=BLUE)
for i in range(4):
    arrow(s, xs[i] + ws[i], y0 + hh / 2, xs[i + 1], y0 + hh / 2)
text(s, 3.15, 1.35, 5.1, 0.4, '训练的部分', size=15, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
# feedback: predictions -> short cache -> (verified) -> memory
box(s, 5.0, 4.55, 3.2, 0.8, [('短期缓存：存预测', 15, True, DARK)], fill=WHITE, line=GRAYB)
line(s, 9.35, y0 + hh, 9.35, 4.95, color=GRAYB, width=1.25, dash=True)
arrow(s, 9.35, 4.95, 8.2, 4.95, color=GRAYB, width=1.25)
line(s, 5.0, 4.95, 1.8, 4.95, color=GRAYB, width=1.25, dash=True)
arrow(s, 1.8, 4.95, 1.8, y0 + hh, color=GRAYB, width=1.25)
text(s, 1.95, 4.95, 3.0, 0.4, '被新帧看到并验证后才写入', size=13, color=SUBC)
conclusion(s, '从真实观测中查找、修正被遮挡的表面，不凭空生成。')

# ================= 8. memory + query paths =================
s = new_slide('观测记忆与查询路径')
text(s, 0.75, 1.0, 5.6, 0.4, '观测记忆', size=16, bold=True, color=BLUE)
cam = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(1.75), Inches(0.5), Inches(0.4))
cam.fill.solid(); cam.fill.fore_color.rgb = DARK; cam.line.fill.background()
line(s, 1.45, 1.95, 6.1, 1.95, color=GRAYB, width=1.5, dash=True)
for x, a in [(2.9, '第 6 帧'), (4.1, '第 11 帧'), (5.3, '第 40 帧')]:
    dot(s, x, 1.95, 0.11, CYAN)
    text(s, x - 0.75, 2.15, 1.5, 0.35, a, size=13, color=SUBC, align=PP_ALIGN.CENTER)
text(s, 0.75, 2.55, 5.6, 0.4, '同一位置保留多个表面假设，每个都记下来源', size=14, color=DARK)
box(s, 0.75, 3.2, 2.65, 1.6, [('长期记忆', 17, True, WHITE), ('存事实', 15, False, WHITE), ('真实观测', 13, False, WHITE)], fill=BLUE, line=BLUE)
box(s, 3.7, 3.2, 2.65, 1.6, [('短期缓存', 17, True, DARK), ('存预测', 15, False, DARK), ('只在当前窗口', 13, False, SUBC)])
arrow(s, 3.7, 4.0, 3.4, 4.0)
text(s, 0.75, 4.95, 5.6, 0.4, '预测被新帧看到并验证一致后，才写入记忆', size=14, color=BLUE)
text(s, 6.95, 1.0, 5.6, 0.4, '查询路径', size=16, bold=True, color=BLUE)
box(s, 6.95, 1.45, 2.7, 0.65, [('光圈去遮挡区的位置', 14, True, WHITE)], fill=BLUE, line=BLUE)
box(s, 6.95, 2.5, 2.7, 0.7, [('有可信候选？', 15, True, DARK)])
box(s, 10.1, 2.5, 2.45, 0.7, [('选择 · 修正 · 融合', 14, True, BLUE)], fill=WHITE, line=BLUE)
box(s, 6.95, 3.6, 2.7, 0.7, [('有真实观测？', 15, True, DARK)])
box(s, 10.1, 3.6, 2.45, 0.7, [('扩大搜索范围', 14, True, BLUE)], fill=WHITE, line=BLUE)
box(s, 6.95, 4.7, 2.7, 0.7, [('补全，标为生成内容', 14, True, RED)], fill=WHITE, line=RED)
arrow(s, 8.3, 2.1, 8.3, 2.5)
arrow(s, 9.65, 2.85, 10.1, 2.85); text(s, 9.6, 2.42, 0.55, 0.35, '是', size=13, color=BLUE, align=PP_ALIGN.CENTER)
arrow(s, 8.3, 3.2, 8.3, 3.6); text(s, 8.35, 3.22, 0.5, 0.35, '否', size=13, color=RED)
arrow(s, 9.65, 3.95, 10.1, 3.95); text(s, 9.6, 3.52, 0.55, 0.35, '是', size=13, color=BLUE, align=PP_ALIGN.CENTER)
arrow(s, 8.3, 4.3, 8.3, 4.7); text(s, 8.35, 4.32, 0.5, 0.35, '否', size=13, color=RED)
conclusion(s, '记忆只存真实观测；所有帧都没看到的内容，只能补全，不冒充“恢复”。')

# ================= 9. ranking + joint correction =================
s = new_slide('候选怎么排序、怎么修正')
text(s, 0.75, 1.0, 5.4, 0.4, '排序：光圈可见度', size=16, bold=True, color=BLUE)
lx, top, bot = 1.15, 1.6, 4.9
lens(s, lx, top, bot)
oc = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3.1), Inches(1.7), Inches(0.26), Inches(1.6))
oc.fill.solid(); oc.fill.fore_color.rgb = DARK; oc.line.fill.background()
P = (5.65, 2.85)
vis = 0; tot = 0
for ly in [1.85, 2.4, 2.95, 3.5, 4.05, 4.6]:
    yb = ly + (P[1] - ly) * (3.1 - lx) / (P[0] - lx)
    tot += 1
    if 1.7 <= yb <= 3.3:
        line(s, lx, ly, 3.1, yb, color=RED, width=1.25, dash=True)
    else:
        vis += 1; line(s, lx, ly, P[0], P[1], color=CYAN, width=1.5)
dot(s, P[0], P[1], 0.1, CYAN)
text(s, 5.2, 3.0, 1.0, 0.35, '候选', size=14, color=DARK, align=PP_ALIGN.CENTER)
text(s, 0.5, 4.95, 1.3, 0.35, '光圈', size=14, color=DARK, align=PP_ALIGN.CENTER)
text(s, 2.55, 1.25, 1.4, 0.35, '前景', size=14, color=DARK, align=PP_ALIGN.CENTER)
box(s, 0.75, 5.35, 5.4, 0.65, [('可见度 = 能看到候选的光线比例（此例 %d / %d）' % (vis, tot), 14, False, DARK)], fill=WHITE, line=CYAN)
text(s, 6.75, 1.0, 5.8, 0.4, '修正：几何和颜色一起改', size=16, bold=True, color=BLUE)
cyc = [(8.45, 1.5, 2.6, 0.8, '候选：深度与颜色成对'),
       (10.4, 2.85, 2.15, 0.95, '选择 · 修正位置'),
       (8.45, 4.35, 2.6, 0.8, '按新位置回来源帧取色'),
       (6.75, 2.85, 2.15, 0.95, '检查颜色是否一致')]
for x, y, w, h, t in cyc:
    box(s, x, y, w, h, [(t, 14, True, DARK)])
arrow(s, 11.05, 1.9, 11.45, 2.85)
arrow(s, 11.45, 3.8, 11.05, 4.75)
arrow(s, 8.45, 4.75, 7.85, 3.8)
arrow(s, 7.85, 2.85, 8.45, 1.9)
text(s, 8.85, 3.1, 1.8, 0.5, '迭代', size=18, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
box(s, 6.75, 5.35, 5.8, 0.65, [('只改深度、颜色仍来自错误来源，结果反而更差', 14, False, RED)], fill=WHITE, line=RED)
conclusion(s, '光圈决定先看哪个候选，颜色始终从真实来源帧读取并随位置一起更新。')

# ================= 10. flicker sources =================
s = new_slide('闪烁从哪里来')
groups = [('几何', [('对焦距离抖动', '整幅画面的模糊在“呼吸”'), ('静态区域深度', '背景虚化大小逐帧跳动')], BLUE),
          ('隐藏层', [('表面来回切换', '前景边缘后面的内容在闪'), ('补全区域', '每帧补出来的都不一样')], BLUE),
          ('渲染', [('子光圈采样', '随机采样带来噪点抖动')], BLUE),
          ('运动物体', [('边缘覆盖率', '逐帧估计，第一版的已知局限')], RED)]
for i, (g, items, lc) in enumerate(groups):
    x0 = 0.75 + i * 3.0
    box(s, x0, 1.25, 2.8, 0.6, [(g, 18, True, WHITE)], fill=BLUE if lc == BLUE else RED, line=BLUE if lc == BLUE else RED)
    for j, (a, b) in enumerate(items):
        box(s, x0, 2.1 + j * 1.65, 2.8, 1.45, [(a, 16, True, DARK), (b, 14, False, SUBC)], line=lc)
conclusion(s, '大部分闪烁，来自同一个表面在不同帧被重新估计。')

# ================= 11. two-pass flow =================
s = new_slide('两遍流程：同一个表面只有一份')
box(s, 0.75, 1.25, 2.2, 1.0, [('第 1 遍', 18, True, WHITE), ('每个片段一次', 13, False, WHITE)], fill=BLUE, line=BLUE)
box(s, 3.3, 1.25, 2.7, 1.0, [('输入帧的真实观测', 15, True, DARK)])
box(s, 6.4, 1.25, 3.3, 1.0, [('静态记忆', 16, True, DARK), ('世界坐标 · 片段编号 · 多表面', 12, False, SUBC)])
box(s, 10.1, 1.25, 2.45, 1.0, [('冻结', 16, True, WHITE)], fill=BLUE, line=BLUE)
arrow(s, 6.0, 1.75, 6.4, 1.75); arrow(s, 9.7, 1.75, 10.1, 1.75)
box(s, 0.75, 2.95, 2.2, 1.2, [('第 2 遍', 18, True, DARK), ('逐帧，含非输入帧', 13, False, DARK)])
box(s, 3.3, 2.95, 3.0, 1.2, [('可见层', 16, True, DARK), ('当前帧颜色 · 深度取自记忆', 13, False, SUBC)])
box(s, 6.7, 2.95, 3.0, 1.2, [('隐藏层', 16, True, DARK), ('已确认的直接读取', 13, False, SUBC)])
box(s, 10.1, 2.95, 2.45, 1.2, [('K=64 渲染', 16, True, DARK), ('当前焦点、光圈', 13, False, SUBC)])
arrow(s, 6.3, 3.55, 6.7, 3.55); arrow(s, 9.7, 3.55, 10.1, 3.55)
arrow(s, 6.75, 2.25, 5.9, 2.95); arrow(s, 8.2, 2.25, 8.2, 2.95)
for i, t in enumerate(['同一面墙，每帧属性一致', '不需要在线维护记忆', '换焦点只需重新渲染']):
    box(s, 0.75 + i * 4.0, 4.75, 3.8, 0.9, [(t, 16, True, BLUE)], fill=WHITE, line=CYAN)
conclusion(s, '同一个静态表面，在整段视频里只用一份几何和外观。')

# ================= 12. specific measures =================
s = new_slide('具体对策')
# panel A: focus curve
text(s, 0.75, 1.05, 3.9, 0.4, '焦点曲线', size=16, bold=True, color=BLUE)
ax0, ay0, aw, ah = 1.05, 4.6, 3.45, 2.85
line(s, ax0, ay0, ax0 + aw, ay0, color=SUBC, width=1.25); line(s, ax0, ay0, ax0, ay0 - ah, color=SUBC, width=1.25)
text(s, ax0 + aw - 0.9, ay0 + 0.02, 1.0, 0.3, '时间', size=12, color=SUBC)
text(s, ax0 - 0.35, ay0 - ah - 0.35, 1.6, 0.3, '1 / 对焦距离', size=12, color=SUBC)
keys = [(0.0, 0.25), (0.35, 0.3), (0.7, 0.75), (1.0, 0.8)]
import math
pts = []
for k in range(41):
    t = k / 40
    # smooth step-like interpolation through keyframes
    for i in range(len(keys) - 1):
        if keys[i][0] <= t <= keys[i + 1][0]:
            u = (t - keys[i][0]) / (keys[i + 1][0] - keys[i][0]); u = u * u * (3 - 2 * u)
            v = keys[i][1] + (keys[i + 1][1] - keys[i][1]) * u; break
    pts.append((ax0 + 0.15 + t * (aw - 0.3), ay0 - 0.15 - v * (ah - 0.4)))
polyline(s, pts, color=BLUE, width=2.25)
for t, v in keys:
    dot(s, ax0 + 0.15 + t * (aw - 0.3), ay0 - 0.15 - v * (ah - 0.4), 0.09, CYAN)
box(s, 0.75, 4.95, 3.9, 0.95, [('关键帧点选主体', 15, True, DARK), ('在逆深度上平滑插值', 14, False, SUBC)])
# panel B: hysteresis
text(s, 4.95, 1.05, 3.9, 0.4, '隐藏表面：双阈值', size=16, bold=True, color=BLUE)
bx0, by0, bw, bh = 5.25, 4.6, 3.45, 2.85
line(s, bx0, by0, bx0 + bw, by0, color=SUBC, width=1.25); line(s, bx0, by0, bx0, by0 - bh, color=SUBC, width=1.25)
text(s, bx0 + bw - 0.9, by0 + 0.02, 1.0, 0.3, '时间', size=12, color=SUBC)
text(s, bx0 - 0.35, by0 - bh - 0.35, 1.2, 0.3, '可信度', size=12, color=SUBC)
y06 = by0 - 0.15 - 0.6 * (bh - 0.4); y04 = by0 - 0.15 - 0.4 * (bh - 0.4)
line(s, bx0, y06, bx0 + bw, y06, color=BLUE, width=1.0, dash=True)
line(s, bx0, y04, bx0 + bw, y04, color=RED, width=1.0, dash=True)
text(s, bx0 + bw - 0.75, y06 - 0.32, 0.9, 0.3, '0.6', size=12, color=BLUE)
text(s, bx0 + bw - 0.75, y04 + 0.02, 0.9, 0.3, '0.4', size=12, color=RED)
qs = [0.3, 0.45, 0.65, 0.58, 0.52, 0.47, 0.55, 0.49, 0.44, 0.5, 0.36, 0.3]
qpts = [(bx0 + 0.15 + i / (len(qs) - 1) * (bw - 0.3), by0 - 0.15 - q * (bh - 0.4)) for i, q in enumerate(qs)]
polyline(s, qpts, color=DARK, width=2.0)
box(s, 4.95, 4.95, 3.9, 0.95, [('高于 0.6 才启用', 15, True, DARK), ('低于 0.4 且连续 2 帧才停用', 14, False, SUBC)])
# panel C
text(s, 9.15, 1.05, 3.4, 0.4, '其他', size=16, bold=True, color=BLUE)
for i, (a, b) in enumerate([('补全区域', '只补一次，所有帧复用'), ('子光圈采样', '固定 64 个位置'), ('时序损失', '小权重，只作辅助')]):
    box(s, 9.15, 1.55 + i * 1.47, 3.4, 1.27, [(a, 16, True, DARK), (b, 14, False, SUBC)])
conclusion(s, '稳定性来自表示本身，时序损失只作辅助。')

# ================= 13. training stages =================
s = new_slide('训练阶段')
stg = [('阶段 1（下一步）', ['静态场景 + 运动相机', '256 / 5 帧', '轻量候选融合']),
       ('阶段 2', ['512 / 13 帧', '+ 物理景深渲染损失', '+ 位姿扰动']),
       ('阶段 3', ['加入运动前景', '微调查询', '+ 时序损失'])]
for i, (a, items) in enumerate(stg):
    x0 = 0.75 + i * 4.15
    cur = i == 0
    col = WHITE if cur else DARK
    box(s, x0, 1.8, 3.55, 3.0, [(a, 20, True, col)] + [(t, 16, False, col) for t in items], fill=BLUE if cur else LIGHT, line=BLUE)
    if i < 2: arrow(s, x0 + 3.57, 3.3, x0 + 4.13, 3.3)
conclusion(s, '第一版只训练隐藏层，可见层与几何前端都固定。')

for sl in list(prs.slides)[n_keep:]:
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
