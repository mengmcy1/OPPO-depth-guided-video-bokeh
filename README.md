# OPPO Depth-Guided Video Bokeh

面向云侧离线精修的视频虚化研究与公开方法复现。

当前主线以 **Any-to-Bokeh** 为核心视频 baseline。Synthetic 七方法（含已冻结目标引导点选焦 B 的 MagicBokeh）、DAVIS 六方法及模型级成本评价已完成；首轮 Any-to-Bokeh 模块分析确认 VAE 编码/解码约占模型段时间的81%，RGB特征缓存为6.47GiB。下一步验证 VAE/缓存及重叠计算的优化假设，结合可重复失败场景确定 Ours。

课题原文中的低帧率粗深度输入、深度时空补全、多头联合预测等仍作为历史提案和候选扩展保留，但不再作为当前必须实现的核心创新或 Ours 前置条件。当前有效研究方向、执行顺序和冻结结果以中央进度文档为准。

## 项目进度

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
