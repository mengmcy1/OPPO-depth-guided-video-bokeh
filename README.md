# OPPO Depth-Guided Video Bokeh

低帧率深度引导的端到端视频虚化统一框架研究。

本课题面向云侧视频虚化渲染，以高帧率 RGB 视频与低帧率粗糙深度为输入，探索深度补全、精细主体与边界建模、可控虚化渲染，并关注时序一致性和处理效率。

## 当前状态

仓库处于项目准备阶段，目前只有课题材料、参考论文和基础目录框架，尚未选择 baseline、dataset、对比方法或评价指标，也没有开始算法实现和正式实验。

当前进度见：[`03_实验记录/当前实验进度与方法论.md`](03_实验记录/当前实验进度与方法论.md)。

## 目录

```text
00_项目资料/          课题材料与需求原文
01_文献/              参考论文
02_方案设计/          问题、方法与实验设计
03_实验记录/          当前进度、历史记录与小型结果
src/                  方法代码
scripts/              训练、推理、评价与数据处理入口
configs/              实验配置
data/                 本地数据集，不提交 Git
models/               本地模型与权重，不提交 Git
outputs/              本地实验输出，不提交 Git
logs/                 本地运行日志，不提交 Git
```

项目协作和文档分工见 [`AGENTS.md`](AGENTS.md)。

## 已收录资料

课题材料：

- `00_项目资料/OPPO影像课题提报模板_给学生.pdf`

参考论文：

- `01_文献/03257-AAAI26.HuangY-CV.pdf`：BokehFlow；
- `01_文献/03980-AAAI26.WangQ-CV.pdf`：BokehCrafter；
- `01_文献/ANY-TO-BOKEH.pdf`：Any-to-Bokeh；
- `01_文献/MagicBokeh-CVPR2026.pdf`：MagicBokeh；
- `01_文献/Siggraph2025-Bokeh Diffusion- Defocus Blur Control in Text-to-Image Diffusion Models.pdf`：Bokeh Diffusion。

参考论文的权利归原作者及相应权利人所有。
