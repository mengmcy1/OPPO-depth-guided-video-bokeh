import re
SRC = '/tmp/claude-0/-root-autodl-tmp/f8fd591a-d469-47bc-af6f-33dd7c2d4b3d/scratchpad/build/build_docx_13p.py'
OUTPY = '/tmp/claude-0/-root-autodl-tmp/f8fd591a-d469-47bc-af6f-33dd7c2d4b3d/scratchpad/build/build_docx_9pB.py'
s = open(SRC, encoding='utf-8').read().replace("scratchpad/build/out_v4.docx'", "scratchpad/build/out_v5.docx'")
starts = [m.start() for m in re.finditer(r"^H\('第 \d+ 页", s, flags=re.M)]
end = s.index('doc.save(OUT)'); head = s[:starts[0]]
blk = {}
for i, st in enumerate(starts):
    en = starts[i + 1] if i + 1 < len(starts) else end
    blk[int(re.match(r"H\('第 (\d+) 页", s[st:]).group(1))] = s[st:en]

def demote(b, label):
    # turn a page block into a sub-part: drop its H line, prefix S() titles
    lines = b.split('\n')
    out = ["S('%s')" % label]
    for l in lines[1:]:
        m = re.match(r"S\('(.*)'\)$", l)
        out.append("P('【%s】%s')" % (label, m.group(1)) if m else l)
    return '\n'.join(out)

b7 = blk[7].replace("P('第 6 页给出了隐藏层网络的模块，这一页把核心思路串成一条完整链路，后面两页再分别展开。')",
                    "P('第 6 页给出了隐藏层网络的模块，这一页把核心思路串成一条完整链路。各环节的细节写在本节后半部分，供追问时参考。')")
b7 = b7.replace("（第 8 页）", "（见本节备查一）").replace("（第 8、9 页）", "（见本节备查一、二）").replace("（第 9 页）", "（见本节备查二）")
b8 = demote(blk[8], '备查一：观测记忆与查询路径').replace("（第 9 页）", "（见备查二）")
b9 = demote(blk[9], '备查二：候选排序与联合修正').replace("第 13 页讲稿中的成功线", "第 9 页讲稿中的成功线")
sec7 = b7 + b8 + '\n' + b9 + '\n'
b11 = re.sub(r"^H\('第 11 页　两遍流程：同一个表面只有一份'\)", "H('第 8 页　跨帧一致的两遍式渲染流程')", blk[11], count=1)
b11 = b11.replace("P('同一个静态表面，在整段视频里只用一份几何和外观。')", "P('静态表面在整段视频中共享同一份几何与外观，从表示层面保证跨帧一致。')")
b10 = demote(blk[10], '备查一：闪烁从哪里来').replace("（第 11 页）", "（即本页的两遍流程）").replace("（第 12 页）", "（见备查二）")
b12 = demote(blk[12], '备查二：具体对策').replace("左边两张小图是示意，不是实验数据。", "")
sec8 = b11 + b10 + '\n' + b12 + '\n'
sec9 = re.sub(r"^H\('第 13 页　训练阶段'\)", "H('第 9 页　训练阶段')", blk[13], count=1)
body = ''.join(blk[k] for k in range(1, 7)) + sec7 + sec8 + sec9
body = body.replace("（第 6–9 页）", "（第 6、7 页）").replace("它达到第 13 页讲稿中的成功线", "它达到第 9 页讲稿中的成功线").replace("见第 9 页）", "见第 7 页讲稿备查二）")
head = head.replace("（13 页）", "（9 页）")
open(OUTPY, 'w', encoding='utf-8').write(head + body + s[end:])
print('refs', sorted(set(re.findall(r"第 \d+ 页", ''.join(l for l in body.split('\n') if not l.startswith("H('"))))))
