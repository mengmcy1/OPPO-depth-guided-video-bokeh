exec(open('/tmp/claude-0/-root-autodl-tmp/f8fd591a-d469-47bc-af6f-33dd7c2d4b3d/scratchpad/build/helpers_v3.py', encoding='utf-8').read())

# drop the user's last slide (old training page); keep slides 1-6 untouched
sldIdLst = prs.slides._sldIdLst
assert len(sldIdLst) == 7
last = list(sldIdLst)[6]
prs.part.drop_rel(last.rId); sldIdLst.remove(last)
n_keep = len(prs.slides)
SUBC = RGBColor(0x59, 0x59, 0x59)

# ---------------- slide 7: query (concise) ----------------
s = new_slide('隐藏层查询：从视频记忆中选出正确的表面')
cx = [0.75, 4.85, 8.95]; cw = 3.6
heads = ['① 观测记忆', '② 两条查询路径', '③ 几何 / 外观联合修正']
for x0, h in zip(cx, heads):
    box(s, x0, 1.3, cw, 0.6, [(h, 18, True, WHITE)], fill=BLUE, line=BLUE)
arrow(s, cx[0] + cw, 1.6, cx[1], 1.6); arrow(s, cx[1] + cw, 1.6, cx[2], 1.6)
ys = [2.3, 3.55, 4.8]; bh = 0.85
for y, t in zip(ys, ['只存真实观测', '每处保留多个表面', '预测不写回记忆']):
    box(s, cx[0], y, cw, bh, [(t, 17, False, DARK)])
for y, (t, lc) in zip(ys, [('有候选：选择与融合', BLUE), ('候选不可信：扩大搜索', BLUE), ('都没看到：补全并标注', RED)]):
    box(s, cx[1], y, cw, bh, [(t, 17, False, RED if lc == RED else DARK)], line=lc)
for i, (y, t) in enumerate(zip(ys, ['修正位置', '回到来源帧取色', '检查颜色是否一致'])):
    box(s, cx[2], y, cw, bh, [(t, 17, False, DARK)])
    if i < 2: arrow(s, cx[2] + cw / 2, y + bh, cx[2] + cw / 2, ys[i + 1])
fx = cx[2] + cw + 0.2
line(s, cx[2] + cw, ys[2] + bh / 2, fx, ys[2] + bh / 2, color=BLUE, width=1.5)
line(s, fx, ys[2] + bh / 2, fx, ys[0] + bh / 2, color=BLUE, width=1.5)
arrow(s, fx, ys[0] + bh / 2, cx[2] + cw, ys[0] + bh / 2)
conclusion(s, '只在光圈会看到的位置查询，颜色始终来自真实观测。')

# ---------------- slide 8: temporal (concise) ----------------
s = new_slide('跨帧闪烁：同一个表面在整段视频中只有一份')
box(s, 0.75, 1.5, 4.4, 1.5, [('第 1 遍', 18, True, WHITE), ('建立静态记忆，然后冻结', 16, False, WHITE)], fill=BLUE, line=BLUE)
arrow(s, 2.95, 3.0, 2.95, 3.65)
box(s, 0.75, 3.65, 4.4, 1.5, [('第 2 遍', 18, True, DARK), ('逐帧渲染，读同一份记忆', 16, False, DARK)])
cards = [('对焦抖动', '逆深度平滑插值'), ('静态区域深度', '取自记忆'),
         ('隐藏表面切换', '片段编号 + 双阈值'), ('渲染采样', '固定 64 个子光圈')]
for i, (a, b) in enumerate(cards):
    x0 = 5.75 + (i % 2) * 3.45; y0 = 1.5 + (i // 2) * 1.95
    box(s, x0, y0, 3.2, 1.65, [(a, 17, True, BLUE), (b, 16, False, DARK)])
conclusion(s, '稳定性来自持久的表面表示，时序损失只作辅助。')

# ---------------- slide 9: training stages (concise) ----------------
s = new_slide('训练阶段')
stg = [('阶段 1（下一步）', ['静态场景 + 运动相机', '256 / 5 帧', '轻量候选融合']),
       ('阶段 2', ['512 / 13 帧', '+ 物理景深渲染损失', '+ 位姿扰动']),
       ('阶段 3', ['加入运动前景', '微调查询', '+ 时序损失'])]
for i, (a, items) in enumerate(stg):
    x0 = 0.75 + i * 4.15
    cur = i == 0
    col = WHITE if cur else DARK
    box(s, x0, 2.0, 3.55, 0.75, [(a, 19, True, col)], fill=BLUE if cur else LIGHT, line=BLUE)
    box(s, x0, 2.9, 3.55, 1.9, [(t, 17, False, DARK) for t in items], fill=WHITE, line=BLUE)
    if i < 2: arrow(s, x0 + 3.57, 2.375, x0 + 4.13, 2.375)
conclusion(s, '第一版只训练隐藏层，可见层与几何前端都固定。')

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
