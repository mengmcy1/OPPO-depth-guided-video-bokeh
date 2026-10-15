"""Fill only the measured MagicBokeh DAVIS row and corresponding narration.

Existing slides, other method values, and efficiency tables remain byte-identical
at the PPTX package-member level except the VEPI slide and its notes.
"""
from copy import deepcopy
import json
import posixpath
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from docx import Document
from lxml import etree as ET

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'汇报/2026-10-16_基线更新与ATB分析'
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}


def replace_paragraph(paragraph,text):
    properties=deepcopy(paragraph.runs[0]._r.rPr) if paragraph.runs else None
    paragraph.clear();run=paragraph.add_run(text)
    if properties is not None:run._r.insert(0,properties)


def main():
    record=ROOT/'03_实验记录/records/magicbokeh-davis-50-20261015.json'
    r=json.loads(record.read_text());assert r['videos']==50 and r['frames']==1250
    assert r['status']=='MAGICBOKEH_DAVIS_50_QUANTITATIVE_COMPLETE_USER_VISUAL_REVIEW_NOT_PERFORMED'
    value=r['VEPI_public_mp4'];png=r['VEPI_png'];delta=r['codec_delta']
    details=(f'MagicBokeh补测：50段×25帧，1024×576，VEPI-public-mp4={value:.8f}；PNG诊断={png:.8f}；MP4−PNG={delta:+.8f}。'
        'DAVIS原类别ID!=0合并目标（沿用旧口径含ID255），最近邻缩放后采用B最大8连通域、预测PNG8深度中位数最近像素选点。'
        '焦点取自身DAdepth选点深度，不取VDA数值；标注只在生成器外辅助一个点，不输入GT mask/alpha。多实例可能选中不同组件，不保证所有实例同焦。'
        'K16，原控制公式abs(d-focus)*K/32、0.1阈值、FP16六图块及PIL/AdaIN均不变。预检8段后复用200帧，只补42段；seed=42+25*固定scene索引+帧号。'
        'VEPI沿旧Sobel/ROI、H.264 quality8/20fps/yuv420p与先帧均值后视频等权均值；不代表整体虚化质量。只完成量化/程序核验，未新增人工视觉结论或复核组失败发现。')
    backup=ROOT/'outputs/magicbokeh-davis-20261015/report-backup';backup.mkdir(parents=True,exist_ok=True)
    path=REPORT/'基线更新与ATB分析_1016.pptx';doc=REPORT/'基线更新与ATB分析_1016_讲稿.docx'
    for source in [path,doc]:
        dest=backup/source.name
        if not dest.exists():shutil.copy2(source,dest)
    with ZipFile(path) as f:contents={name:f.read(name) for name in f.namelist()}
    changed={};old_rows=[]
    for name,data in contents.items():
        if not name.startswith('ppt/slides/slide') or not name.endswith('.xml'):continue
        root=ET.fromstring(data)
        for table in root.findall('.//a:tbl',NS):
            rows=table.findall('a:tr',NS)
            texts=[[''.join(tc.itertext()) for tc in row.findall('a:tc',NS)] for row in rows]
            if not texts or not any('VEPI' in v for v in texts[0]):continue
            for row in rows:
                cells=row.findall('a:tc',NS)
                label=''.join(t.text or '' for t in cells[0].findall('.//a:t',NS))
                old_rows.append([label,''.join(t.text or '' for t in cells[1].findall('.//a:t',NS))])
                if label=='MagicBokeh':
                    current=cells[1].findall('.//a:t',NS)
                    assert len(current)==1 and (current[0].text or '') in ['',f'{value:.3f}']
                    current[0].text=f'{value:.3f}'
                    changed[name]=ET.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
                    relationships='ppt/slides/_rels/'+Path(name).name+'.rels'
                    rels=ET.fromstring(contents[relationships])
                    note_rel=next(rel for rel in rels if rel.get('Type','').endswith('/notesSlide'))
                    target=note_rel.get('Target')
                    notes=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(posixpath.dirname(name),target))
                    if notes in contents:
                        n=ET.fromstring(contents[notes]);candidate=n.findall('.//a:t',NS)
                        target=next(t for t in candidate if '来源' in (t.text or ''))
                        if 'MagicBokeh补测：' not in target.text:target.text+='\n'+details
                        changed[notes]=ET.tostring(n,xml_declaration=True,encoding='UTF-8',standalone=True)
    assert len(old_rows)==8 and len(changed)==2
    with ZipFile(path,'w',ZIP_DEFLATED) as f:
        for name,data in contents.items():f.writestr(name,changed.get(name,data))
    d=Document(doc);replacements={
        '汇报日期为2026年10月16日，实验和案例依据截至10月14日。':
            '汇报日期为2026年10月16日，ATB实验和案例依据截至10月14日；MagicBokeh DAVIS补测单列2026年10月15日运行记录。本次承接9月30日汇报，重点讲新增MagicBokeh参照、统一硬件的成本分析、已经验证的ATB实现优化，以及真实视频局部问题的定位。',
        'DAVIS评价包含50段真实视频，每段25帧，共六种方法。':
            f'DAVIS评价包含50段真实视频，每段25帧，现补齐七种方法。MagicBokeh的MP4主VEPI为{value:.5f}，PNG诊断为{png:.5f}，编码差值为{delta:+.5f}；预检8段200帧复用后只补42段。原六方法不重跑：BokehMe为0.954，ATB为0.950。VEPI关注主体边缘保留，不能单独排名整体虚化或时序质量。',
        'DAVIS没有配对虚化真值，因此我们将边缘指标与原视频、局部放大及连续帧观察结合。MagicBokeh已在表中留出一行':
            'DAVIS没有配对虚化真值，因此我们将边缘指标与原视频、局部放大及连续帧观察结合。'+details+' DeepLens仍采用自身估深和原生默认光圈，原六方法输入适配及旧数值均保留。'}
    hits=[]
    for paragraph in d.paragraphs:
        for start,text in replacements.items():
            if paragraph.text.startswith(start):replace_paragraph(paragraph,text);hits.append(start);break
    assert len(hits)==3;d.save(doc)
    protocol=ROOT/'03_实验记录/Benchmark与Baseline统一评价协议.md';text=protocol.read_text()
    header='**冻结状态：DAVIS_2016_V2_SIX_BASELINES_COMPLETE_AND_FROZEN（2026-09-30）。** 每方法50×25帧，六方法7500帧及300段统一H.264视频完整；预检8段均复用。主列采用MP4 VEPI，PNG和codec差值另列诊断。BokehCrafter继续deferred，不计入本轮六方法。'
    replacement='**当前状态：原六方法冻结结果保留，MagicBokeh DAVIS定量补测完成（2026-10-15）。** 七方法各50×25帧，共8750帧、350段统一H.264视频；MagicBokeh复用8段200帧预检，只新增42段。主列采用MP4 VEPI，PNG与编码差值单列诊断；MagicBokeh仅完成量化及程序检查，新增视觉结论待用户复核。BokehCrafter继续deferred。'
    assert header in text;text=text.replace(header,replacement,1)
    table_anchor='| Ours | later | later | later | later | later |'
    assert table_anchor in text
    text=text.replace(table_anchor,f'| MagicBokeh（DAVIS区域辅助选焦） | {value:.5f} | 待用户复核 | 待用户复核 | 待用户复核 | 待用户复核 |\n'+table_anchor,1)
    note=(f'\nMagicBokeh新增量化：VEPI-public-mp4 **{value:.5f}**，VEPI-png **{png:.5f}**，MP4−PNG **{delta:+.5f}**。'
        '输入为DAVIS非零标注union辅助的自身深度单点选焦，非共享VDA数值条件；多实例/组件选择限制及批准规则见上文。'
        '预检200帧及全量1250帧的选点、坐标、原生条件、种子、RGB/finite核验通过，不按VEPI或观感回调。'
        '原六行数值与定性列保持原样；新增行不自动填视觉质量结论，也不把新全量输出用于来源隔离组失败发现。'
        '[新增逐帧/逐视频及七方法同口径记录](records/magicbokeh-davis-50-20261015.json)、[最终程序核验](records/magicbokeh-davis-final-check-20261015.json)。\n')
    text=text.replace(table_anchor,table_anchor+'\n'+note,1)
    text=text.replace('| Table 2 | DAVIS，六方法结果已完成；MagicBokeh尚未评价 |', '| Table 2 | DAVIS，原六方法冻结；MagicBokeh已补50×25量化，新增视觉复核仍待用户 |')
    protocol.write_text(text)
    reproduction=ROOT/'03_实验记录/MPIB-public-proxy-200-video-复现汇报.md';text=reproduction.read_text()
    anchor='| Any-to-Bokeh | 0.95011 | 0.95400 | -0.00389 | 0.944 | +0.00611 |'
    assert anchor in text
    text=text.replace(anchor,anchor+f'\n| MagicBokeh（DAVIS union区域辅助选焦） | {value:.5f} | {png:.5f} | {delta:+.5f} | — | — |\n'+note,1)
    text=text.replace('MagicBokeh尚未进入DAVIS，当前不阻塞ATB机制分析。','MagicBokeh已补DAVIS50×25量化，原六方法结果与ATB机制分析材料保持独立；新增视觉判断待用户复核。')
    reproduction.write_text(text)
    with ZipFile(path) as f:
        actual={name for name in contents if f.read(name)!=contents[name]};assert actual==set(changed)
    result=dict(PPT_changed_members=sorted(actual),other_PPT_members_unchanged=True,old_table_rows=old_rows,
        filled_value=f'{value:.3f}',narration_paragraphs_changed=3,other_report_variants_unchanged=True,
        backup=str(backup.relative_to(ROOT)),formal_protocol_and_reproduction_tables_updated=True)
    (ROOT/'outputs/magicbokeh-davis-20261015/report-update.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('UPDATED_MAGICBOKEH_DAVIS_REPORT_ONLY',json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
