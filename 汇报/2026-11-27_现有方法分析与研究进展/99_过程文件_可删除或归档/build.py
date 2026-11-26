exec(open('/tmp/claude-0/-root-autodl-tmp/cd48efee-2dd0-4643-97fb-2190d6ffb097/scratchpad/p1127/helpers.py', encoding='utf-8').read())

ids = prs.slides._sldIdLst
for sid in list(ids)[1:]:
    prs.part.drop_rel(sid.rId); ids.remove(sid)
for sh in prs.slides[0].shapes:
    if sh.has_text_frame:
        for para in sh.text_frame.paragraphs:
            for r in para.runs:
                if r.text == '思路与可行性验证': r.text = '现有方法分析与研究进展'
                if r.text == '2026.11.11': r.text = '2026.11.27'


def rich(slide, x, y, w, h, rows, size=20, space=14, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, segs in enumerate(rows):
        sp = space
        if isinstance(segs, tuple): segs, sp = segs
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(sp); p.alignment = align
        for t, b, c in segs:
            r = p.add_run(); r.text = t; set_font(r, size, b, c)
    return tb


def label(s, x, y, w, t, size=15, color=DARK, bold=False):
    text(s, x, y, w, 0.36, t, size=size, bold=bold, color=color, align=PP_ALIGN.CENTER)


def ttable(s, x, y, w, rows, col_w, row_h=0.5, size=15, hl_row=None):
    gs = table(s, x, y, w, rows, col_w, row_h=row_h, size=size, mark_colors=False)
    tb = gs.table
    for i in range(1, len(rows)):
        for j in range(len(rows[0])):
            cell = tb.cell(i, j)
            cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT if i % 2 == 1 else WHITE
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.bold = (j == 0) or (hl_row == i)
                    r.font.color.rgb = BLUE if hl_row == i else DARK
    return gs


# 2 two routes
s = new_slide('现有方法的两条路线')
rows = [('DOF-GS\nDoF-Gaussian', ['多视角离焦照片', '逐场景优化三维高斯', '渲染时加入镜头离焦']),
        ('Any-to-Bokeh', ['单目视频', '逐帧深度 → 离焦图', '视频扩散模型生成虚化'])]
for r, (name, steps) in enumerate(rows):
    y = 1.5 + r * 2.1
    text(s, 0.75, y + 0.05, 2.6, 0.9, name.split('\n'), size=19, bold=True, color=BLUE, anchor=MSO_ANCHOR.MIDDLE)
    xs = [3.45, 6.55, 9.65]
    for a, b in zip(xs[:-1], xs[1:]): arrow(s, a + 2.75, y + 0.5, b, y + 0.5, width=2)
    for x, t in zip(xs, steps): box(s, x, y, 2.75, 1.0, [(t, 17, True, DARK)])
conclusion(s, '一类是三维表示但要逐场景优化，一类是前馈但只有逐帧深度。')

# 3 DoF-GS results
s = new_slide('DOF-GS 与 DoF-Gaussian 复现结果')
text(s, 0.75, 1.05, 5.6, 0.4, '真实镜头景深（DoF-NeRF）', size=17, bold=True, color=BLUE)
ttable(s, 0.75, 1.55, 5.6, [['方法', 'PSNR ↑', 'LPIPS ↓'], ['DOF-GS', '28.43', '0.096'], ['DoF-Gaussian', '29.54', '0.090'], ['清晰渲染 + BokehMe', '29.09', '0.092']],
       [2.6, 1.5, 1.5], row_h=0.55)
text(s, 6.95, 1.05, 5.6, 0.4, '分区域 PSNR ↑', size=17, bold=True, color=BLUE)
ttable(s, 6.95, 1.55, 5.6, [['方法', '前景边缘', '焦内', '焦外'], ['DOF-GS', '25.04', '31.53', '23.49'], ['DoF-Gaussian', '26.07', '33.08', '24.86'], ['清晰渲染 + BokehMe', '25.89', '32.21', '24.15']],
       [2.6, 1.0, 1.0, 1.0], row_h=0.55)
conclusion(s, '误差集中在前景边缘；三维方法和 2.5D 后处理在全图上只差约 1 dB。')

# 4 DoF-GS problems
s = new_slide('DOF-GS 与 DoF-Gaussian 存在的问题')
tw = 3.0; th = 2.25; xs = [0.75, 3.95, 7.15]
for x, k, n in zip(xs, ['kG', 'kA', 'kB'], ['真值', 'DOF-GS', 'DoF-Gaussian']):
    pic(s, k, x, 1.45, w=tw); label(s, x, 1.05, tw, n, bold=True, color=BLUE)
pic(s, 'kFull', 10.35, 1.45, w=2.2); label(s, 10.35, 1.05, 2.2, '原图位置', bold=True, color=BLUE)
rich(s, 0.85, 4.05, 11.6, 2.3, [
    [('1　', True, BLUE), ('每个场景单独优化：COLMAP 约 32 分钟、训练 22–31 分钟，只能处理静态场景', False, DARK)],
    [('2　', True, BLUE), ('离焦参数要逐场景学，学偏后焦内也会糊：焦内 DOF-GS 22.7 dB，DoF-Gaussian 30.2 dB', False, DARK)],
    [('3　', True, BLUE), ('误差集中在遮挡边缘，前景边缘比焦内低 6–7 dB', False, DARK)],
], size=18, space=12)
conclusion(s, '逐场景优化很难直接用于手机拍摄的动态视频。')

# 5 ATB results
s = new_slide('Any-to-Bokeh 复现结果')
ttable(s, 0.75, 1.3, 11.8, [['方法', '合成 PSNR ↑', '合成 LPIPS ↓', '真实视频边缘保留 ↑', '每帧时间 (s) ↓', '峰值显存 (GiB) ↓'],
                            ['Any-to-Bokeh', '39.71', '0.015', '0.950', '0.80', '42.1'],
                            ['BokehMe', '33.33', '0.025', '0.954', '0.01', '0.8'],
                            ['MPIB', '31.62', '0.028', '0.930', '0.24', '2.2'],
                            ['Dr.Bokeh', '31.59', '0.029', '0.877', '0.34', '2.6']],
       [2.3, 1.8, 1.8, 2.3, 1.7, 1.9], row_h=0.6, hl_row=1)
conclusion(s, 'Any-to-Bokeh 画质最好，但耗时和显存也最高。')

# 6 ATB problem 1
s = new_slide('Any-to-Bokeh 问题一：细结构')
dw = 2.6; dh = 1.95
label(s, 0.75, 1.0, 5.3, '鸭头：只改送进模型的离焦图', bold=True, color=BLUE)
for c, (fr, a, b) in enumerate([('第 0 帧', 'du0', 'dc0'), ('第 24 帧', 'du3', 'dc3')]):
    x = 0.75 + c * (dw + 0.1)
    label(s, x, 1.4, dw, fr, size=14)
    pic(s, a, x, 1.8, w=dw); pic(s, b, x, 1.8 + dh + 0.1, w=dw)
text(s, 0.75 + 2 * (dw + 0.1), 1.8 + dh / 2 - 0.2, 0.9, 0.4, '原版', size=14, color=DARK)
text(s, 0.75 + 2 * (dw + 0.1), 1.8 + dh * 1.5 - 0.1, 0.9, 0.4, '修正后', size=14, color=DARK)
uw = 2.75; ux = 6.95
label(s, ux, 1.0, 5.6, '伞绳：多种方法都会变糊', bold=True, color=BLUE)
for i, (k, n) in enumerate([('u_in', '原视频'), ('u_atb', 'Any-to-Bokeh')]):
    x = ux + i * (uw + 0.1); label(s, x, 1.4, uw, n, size=14); pic(s, k, x, 1.8, w=uw)
label(s, ux, 3.65, uw, 'MPIB', size=14); pic(s, 'u_mpib', ux, 4.05, w=uw)
conclusion(s, '逐帧深度的局部误差会直接变成虚化错误；只改离焦图，鸭头就全程变清晰。')

# 7 ATB problem 2
s = new_slide('Any-to-Bokeh 问题二：2.5D 表示与效率')
rich(s, 0.85, 1.4, 11.6, 4.5, [
    [('1　', True, BLUE), ('只有逐帧深度，前景后面被挡住的背景只能由扩散模型“猜”', False, DARK)],
    [('2　', True, BLUE), ('每换一次焦点或光圈都要重跑扩散：每帧约 0.8 s，峰值显存约 42 GiB', False, DARK)],
    [('3　', True, BLUE), ('耗时主要在编解码：VAE 相关阶段占 80%，扩散主干只占 19%', False, DARK)],
    [('4　', True, BLUE), ('训练数据是前后两平面拼贴，检验不了真实三维几何', False, DARK)],
], size=20, space=20)
conclusion(s, '2.5D 方法看不到被挡住的真实背景，换焦点的代价也高。')

# 8 our idea
s = new_slide('我们的路线思路')
label(s, 0.75, 1.0, 5.6, '虚化 = 光圈上多个视角的平均', size=17, bold=True, color=BLUE)
pic(s, 'ap_avg', 0.75, 1.5, w=2.75); pic(s, 'ap_gt', 3.6, 1.5, w=2.75)
label(s, 0.75, 4.3, 2.75, '64 个子视角平均', size=14); label(s, 3.6, 4.3, 2.75, '物理渲染真值', size=14)
label(s, 6.95, 1.0, 5.6, '前景虚化后会透出被挡住的背景', size=17, bold=True, color=BLUE)
pic(s, 'mpib', 6.95, 1.5, w=5.6)
label(s, 6.95, 5.35, 5.6, '图源：MPIB', size=12, color=GRAY)
conclusion(s, '虚化交给物理光圈渲染，被挡住的背景从视频其他帧找回。')

# 9 system + modules
s = new_slide('系统结构与模块设计')
xs = [0.75, 3.15, 5.55, 7.95, 10.35]; w = 2.2
labels = ['单目视频', '几何前端（冻结）', '可见层 OVG-Net', '隐藏层', 'K=64 光圈渲染']
for a, b in zip(xs[:-1], xs[1:]): arrow(s, a + w, 1.6, b, 1.6, width=2)
for i, (x, t) in enumerate(zip(xs, labels)):
    if i in (2, 3): box(s, x, 1.2, w, 0.8, [(t, 15, True, WHITE)], fill=BLUE, line=BLUE)
    else: box(s, x, 1.2, w, 0.8, [(t, 15, True, DARK)])
ttable(s, 0.75, 2.5, 11.8, [['模块', '解决的问题', '做法', '状态'],
                            ['模块一　细粒度几何', '表面在哪里', '细杆、缝隙用前后两个深度分量', '进行中'],
                            ['模块二　表面连通', '表面怎么连', '学习撕裂场：同一表面连上，遮挡处断开', '已训练'],
                            ['模块三　边界分层', '挡住谁', '边缘像素拆成前景色和背景色', '待做'],
                            ['隐藏层', '被挡住的是什么', '从其他帧的真实观测中检索', '已有']],
       [2.6, 2.0, 5.4, 1.8], row_h=0.58)
conclusion(s, '核心难点：同一表面要连续不漏光，真实遮挡边界又要正确断开。')

# 10 bottleneck
s = new_slide('当前的主要瓶颈')
tw = 2.3; gap = 0.075
names = [('r_gt', '真值'), ('r_ours', '我们'), ('r_sharp', 'SHARP'), ('r_drb', 'Dr.Bokeh（真值深度）'), ('r_gto', '我们 + 真值几何')]
for i, (k, n) in enumerate(names):
    x = 0.75 + i * (tw + gap); label(s, x, 1.05, tw, n, size=14, bold=True, color=BLUE); pic(s, k, x, 1.45, w=tw)
rich(s, 0.85, 4.15, 11.6, 1.8, [
    [('栏杆场景，对焦远处背景：栏杆应该虚掉，我们的结果里仍然清晰。', False, DARK)],
    [('开发集远焦可恢复区：', True, BLUE), ('我们 24.0 dB，SHARP 21.0 dB，换成真值几何 35.2 dB', False, DARK)],
], size=18, space=12)
conclusion(s, '最大差距在细结构和遮挡边缘的几何。')

# 11 data
s = new_slide('数据准备')
cols = [('G', '公开合成几何', 'TartanAir V2', '74 环境 · 2 万帧'), ('C', '卡片场景', '自建', '5,000 段'),
        ('B', '自有 Blender', '自建', '295 场景'), ('R', '真实视频', 'DL3DV', '500 段'),
        ('A', '抠图合成', 'AM-2k 等', '1,000 前景')]
cw, gap = 2.3, 0.075
for i, (k, name, src, sub) in enumerate(cols):
    x = 0.75 + i * (cw + gap)
    pic(s, 't_' + k, x, 1.25, w=cw)
    text(s, x, 3.1, cw, 0.42, f'{k}　{name}', size=17, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    text(s, x, 3.53, cw, 0.36, src, size=14, color=DARK, align=PP_ALIGN.CENTER)
    text(s, x, 3.88, cw, 0.4, sub, size=15, bold=True, color=DARK, align=PP_ALIGN.CENTER)
rich(s, 0.85, 4.6, 11.6, 1.6, [
    [('另外：', True, BLUE), ('237 个 Blender 场景带景深真值；开发集 15 个场景；封存测试集 30 个场景', False, DARK)],
    [('合成数据给精确标签，真实视频只做自监督。', False, DARK)],
], size=17, space=10)
conclusion(s, '五档数据已全部生成。')

# 12 training
s = new_slide('训练策略与进度')
rich(s, 0.85, 1.2, 11.6, 5.0, [
    [('阶段 1　可见层几何', True, BLUE), ('　已用扩充数据重训', False, GRAY)],
    [('阶段 2　边界拆分', True, BLUE), ('　用真值背景拆分也没有提升，保留不拆分', False, GRAY)],
    [('阶段 3　光学联合训练', True, BLUE), ('　三步都没有超过起点，保留起点', False, GRAY)],
    [('模块二　表面撕裂场', True, BLUE), ('　已训练，近焦焦内提高约 0.7 dB', False, GRAY)],
    [('模块一　细粒度几何', True, BLUE), ('　进行中', False, GRAY)],
], size=20, space=18)
conclusion(s, '下一步：模块一 → 接入模块二 → 模块三 → 动态视频。')

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
