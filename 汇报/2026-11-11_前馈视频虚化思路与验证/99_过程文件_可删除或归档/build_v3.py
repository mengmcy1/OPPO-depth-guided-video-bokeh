exec(open('/tmp/claude-0/-root-autodl-tmp/f8fd591a-d469-47bc-af6f-33dd7c2d4b3d/scratchpad/build/helpers_v3.py', encoding='utf-8').read())

# drop the user's last slide (old training page); keep slides 1-6 untouched
sldIdLst = prs.slides._sldIdLst
assert len(sldIdLst) == 7
last = list(sldIdLst)[6]
prs.part.drop_rel(last.rId); sldIdLst.remove(last)
n_keep = len(prs.slides)
SUBC = RGBColor(0x59, 0x59, 0x59)

# ---------------- slide 7: query ----------------
s = new_slide('隐藏层查询：从视频记忆中选出正确的表面')
cx = [0.75, 4.85, 8.95]; cw = 3.6
heads = ['① 观测记忆', '② 查询的两条路径', '③ 几何 / 外观联合修正']
for x0, h in zip(cx, heads):
    box(s, x0, 1.15, cw, 0.55, [(h, 17, True, WHITE)], fill=BLUE, line=BLUE)
arrow(s, cx[0] + cw, 1.425, cx[1], 1.425); arrow(s, cx[1] + cw, 1.425, cx[2], 1.425)
ys = [1.95, 3.1, 4.25]; bh = 0.9
colA = [('只存经过验证的真实观测', '来源帧颜色 · 校准深度 · 来源记录'),
        ('同一位置保留 2–4 个表面假设', '不强行平均成一个表面'),
        ('网络预测只进短期缓存', '不写回记忆：记忆存事实')]
colB = [('有可信候选', '选择 · 修正 · 融合', BLUE),
        ('有观测但候选不可信', '扩大搜索范围', BLUE),
        ('所有帧都没看到', '交给补全，标记为生成内容', RED)]
colC = ['几何：选择候选、修正位置', '按新位置回到来源帧重新取色', '检查多个来源的颜色是否一致']
for y, (a, b) in zip(ys, colA):
    box(s, cx[0], y, cw, bh, [(a, 15, True, DARK), (b, 13, False, SUBC)])
for y, (a, b, lc) in zip(ys, colB):
    box(s, cx[1], y, cw, bh, [(a, 15, True, RED if lc == RED else DARK), (b, 13, False, SUBC)], line=lc)
for i, (y, a) in enumerate(zip(ys, colC)):
    box(s, cx[2], y, cw, bh, [(a, 15, True, DARK)])
    if i < 2: arrow(s, cx[2] + cw / 2, y + bh, cx[2] + cw / 2, ys[i + 1])
fx = cx[2] + cw + 0.2
line(s, cx[2] + cw, ys[2] + bh / 2, fx, ys[2] + bh / 2, color=BLUE, width=1.5)
line(s, fx, ys[2] + bh / 2, fx, ys[0] + bh / 2, color=BLUE, width=1.5)
arrow(s, fx, ys[0] + bh / 2, cx[2] + cw, ys[0] + bh / 2)
box(s, 0.75, 5.4, 11.8, 0.7, [('候选按“光圈可见度”排序：K 条子光圈光线中，有多少条能看到这个候选', 15, False, DARK)], fill=WHITE, line=CYAN)
conclusion(s, '查询只在光圈会看到的位置发起，颜色始终来自真实观测，几何和颜色一起修正。')

# ---------------- slide 8: temporal consistency ----------------
s = new_slide('跨帧闪烁：同一个表面在整段视频中只有一份')
text(s, 0.75, 1.0, 4.3, 0.4, '两遍流程', size=16, bold=True, color=BLUE)
box(s, 0.75, 1.5, 4.3, 1.45, [('第 1 遍：每个视频片段一次', 16, True, WHITE), ('输入帧的真实观测 → 静态记忆', 13, False, WHITE), ('建立后冻结', 13, False, WHITE)], fill=BLUE, line=BLUE)
arrow(s, 2.9, 2.95, 2.9, 3.35)
box(s, 0.75, 3.35, 4.3, 2.05, [('第 2 遍：逐帧渲染（含非输入帧）', 16, True, DARK), ('可见层：当前帧颜色', 13, False, DARK), ('静态区域深度：取自同一份记忆', 13, False, DARK), ('隐藏表面：已确认的直接读取', 13, False, DARK)])
rows = [['闪烁来源', '处理方式'],
        ['对焦距离抖动', '关键帧点选主体，在逆深度上平滑插值'],
        ['静态区域深度逐帧不同', '深度取自同一份记忆'],
        ['隐藏表面在帧间切换', '表面片段编号 + 启用 / 停用双阈值'],
        ['补全区域逐帧不同', '只补一次，所有帧复用'],
        ['渲染采样', '固定 64 个子光圈位置'],
        ['运动物体边缘', '第一版只用当前帧，作为已知局限']]
table(s, 5.45, 1.0, 7.1, rows, [2.65, 4.45], row_h=0.62, size=14, mark_colors=False)
conclusion(s, '稳定性主要来自持久的表面表示，时序损失只作辅助。')

# ---------------- slide 9: training stages ----------------
s = new_slide('训练阶段')
rows = [['', '阶段 1（下一步）', '阶段 2', '阶段 3'],
        ['数据', '静态场景 + 带平移的运动相机', '同左', '加入运动前景与连续片段；静态样本保留 30–50%'],
        ['分辨率 / 上下文', '256 / 5 帧', '512 / 13 帧', '512 / 13 帧 + 连续帧'],
        ['训练对象', '轻量候选融合', '不够时升级为完整查询解码', '微调查询；必要时加动态候选拒绝'],
        ['监督', '内容 · 位置 · 颜色 · 可信度', '+ K=64 物理景深渲染损失', '+ 小权重时序损失'],
        ['几何输入', '校准估计深度为主，真值几何只作热身', '+ 校准前的位姿扰动', '同左'],
        ['进入下一步', '留出场景上超过规则方法', '收回 75–80% 差距；1–2 周达不到则转向', '—']]
table(s, 0.75, 1.05, 11.8, rows, [2.0, 3.27, 3.27, 3.26], row_h=0.64, size=13, mark_colors=False)
box(s, 0.75, 5.7, 11.8, 0.6, [('阶段 1 有效后：最小边缘解混（不训练），处理可见层边缘的前景 / 背景混合色', 15, False, DARK)], fill=WHITE, line=CYAN)
conclusion(s, '第一版只训练隐藏层，SHARP 可见层与几何前端都固定。')

# remove inherited theme effects on newly added slides only
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
