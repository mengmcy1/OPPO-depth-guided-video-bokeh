# Benchmark、指标与 Baseline 统一评价协议（第一版）

## 1. 目的

本文件用于固定当前阶段的 Benchmark、评价指标、Baseline 与实验输出格式，避免后续每复现一个方法就临时调整评价协议。

当前阶段优先完成：

1. 固定主 Benchmark；
2. 固定核心指标；
3. 固定主对比表结构；
4. 固定效率测量协议；
5. 固定真实视频定性/边缘评价方式；
6. 明确哪些方法适合作为核心 baseline，哪些只适合作为补充参考；
7. 从 Any-to-Bokeh 开始按统一模板积累结果。

---

# 2. Benchmark

## 2.1 主 Benchmark：Synthetic Paired Video Benchmark

用途：

> 作为主要定量比较数据集，评价最终 Bokeh 质量与时序表现。

优先参考 Any-to-Bokeh 的 synthetic test protocol：

- paired RGB / Geometry / Bokeh GT；
- 视频级数据；
- 支持不同 focus plane；
- 支持不同 blur strength；
- 可计算全参考指标。

当前目标规模可参考：

- 200 videos；
- 25 frames / video；
- 1024×576。

如果无法获得论文完全相同的测试集，则按官方数据生成流程重建固定测试集，并明确标记为：

**our reproduced synthetic benchmark**

同时固定：

- 数据来源；
- 随机种子；
- 视频长度；
- 分辨率；
- focus 配置；
- blur strength；
- 预处理方式。

一旦 Benchmark 确定，不随算法改变。

---

## 2.2 Real-world Benchmark：DAVIS

用途：

> 验证真实视频泛化、主体边缘保持、发丝/细结构以及视觉时序稳定性。

由于 DAVIS 没有对应 Bokeh GT，因此不强行计算：

- PSNR；
- SSIM；
- LPIPS（相对于 Bokeh GT）。

第一阶段主要使用：

- VEPI；
- Qualitative Video；
- 固定关键帧；
- 后续必要时增加 blinded user study。

优先从 Any-to-Bokeh 官方使用的固定 DAVIS demo 序列开始。

---

## 2.3 可选 Benchmark：VBS

VBS（Video Bokeh Scenes）暂作为第二阶段补充。

价值：

- paired video Bokeh；
- Static Focus；
- Dynamic Focus；
- 适合研究真实焦点变化下的 temporal behavior。

暂不把 VBS 设为第一轮复现的阻塞项。

---

# 3. 核心指标

## 3.1 单帧 / 重建质量

| 指标 | 趋势 | 主要含义 |
|---|---:|---|
| PSNR | ↑ | 像素级重建误差 |
| SSIM | ↑ | 结构相似度 |
| LPIPS | ↓ | 感知特征相似度 |

三者建议同时保留。

---

## 3.2 视频整体与时序质量

| 指标 | 趋势 | 主要含义 |
|---|---:|---|
| FVD | ↓ | 视频整体分布质量，兼顾视觉与动态 |
| VFID-I | ↓ | 基于 I3D 视频特征的分布距离 |
| FD | ↓ | 预测视频与 GT 运动 / 光流关系差异 |
| RM | ↓ | 预测帧间变化与 GT 帧间变化的一致性 |

核心原则：

[
Temporal Consistency 
eq Temporal Constancy
]

评价应该关注：

[
Predicted Change approx Real Change
]

而不是要求所有相邻帧尽可能相同。

---

## 3.3 真实边缘

### VEPI ↑

用于评价：

- Focused Subject Edge；
- Hair；
- Thin Structure；
- 主体边缘是否被背景虚化侵蚀。

真实 DAVIS 第一阶段优先采用：

[
VEPI + Qualitative Video
]

---

# 4. 主定量表

## Table A：Synthetic Paired Video Benchmark

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | VFID-I ↓ | FD ↓ | RM ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepLens |  |  |  |  |  |  |  |
| BokehDiff |  |  |  |  |  |  |  |
| BokehMe |  |  |  |  |  |  |  |
| Dr.Bokeh |  |  |  |  |  |  |  |
| MPIB |  |  |  |  |  |  |  |
| Any-to-Bokeh |  |  |  |  |  |  |  |
| BokehCrafter* |  |  |  |  |  |  |  |
| Ours |  |  |  |  |  |  |  |

第一轮尽量沿用 Any-to-Bokeh 正式论文的 baseline 集合，保证评价协议一致。

---

# 5. 效率表

## Table B：Efficiency

| Method | Resolution | Frames | Core Time ↓ | E2E Time ↓ | FPS ↑ | Peak VRAM ↓ | Params |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepLens | 1024×576 | 25 |  |  |  |  |  |
| Dr.Bokeh | 1024×576 | 25 |  |  |  |  |  |
| BokehMe | 1024×576 | 25 |  |  |  |  |  |
| MPIB | 1024×576 | 25 |  |  |  |  |  |
| BokehDiff | 1024×576 | 25 |  |  |  |  |  |
| Any-to-Bokeh | 1024×576 | 25 |  |  |  |  |  |
| BokehCrafter* | 1024×576 | 25 |  |  |  |  |  |
| Ours | 1024×576 | 25 |  |  |  |  |  |

统一要求：

- 同一 GPU；
- 同一 CUDA / PyTorch 环境；
- 相同分辨率；
- 相同视频长度；
- batch size = 1；
- warm-up；
- 至少重复 5 次；
- 报告平均值；
- 记录 Peak VRAM。

区分：

### Core Time

只统计核心 Bokeh 网络推理。

### E2E Time

包含：

```text
RGB
↓
Depth Estimation
↓
Mask / Target Extraction
↓
Bokeh Network
↓
Final Video
```

防止不同方法将大量计算放在 preprocessing 中而导致不公平比较。

---

# 6. Real-world DAVIS 表

## Table C：Real-world DAVIS

| Method | VEPI ↑ | Qualitative Video | Hair / Thin Edge | Temporal Artifact |
|---|---:|---|---|---|
| DeepLens |  |  |  |  |
| Dr.Bokeh |  |  |  |  |
| BokehMe |  |  |  |  |
| MPIB |  |  |  |  |
| BokehDiff |  |  |  |  |
| Any-to-Bokeh |  |  |  |  |
| BokehCrafter* |  |  |  |  |
| Ours |  |  |  |  |

其中 Hair / Thin Edge 与 Temporal Artifact 第一阶段只做定性观察，不自行构造 1～5 分“伪定量指标”。

---

# 7. Any-to-Bokeh 第一轮实验

## Experiment A：Official Reproduction

目的：

> 确认官方代码、checkpoint、预处理、推理和 metrics pipeline 全部跑通。

输出：

- predicted video；
- GT video（若有）；
- comparison video；
- metrics；
- runtime；
- peak VRAM；
- config。

指标：

- PSNR；
- SSIM；
- LPIPS；
- FVD；
- VFID-I；
- FD；
- RM。

---

## Experiment B：Depth / Disparity Noise Robustness

目的：

> 对应 Q10，观察 Geometry Error 如何传递到最终 Bokeh。

设置：

- Clean；
- Mild corruption；
- Medium corruption；
- Severe corruption。

候选 corruption：

- Gaussian / smooth perturbation；
- Elastic perturbation；
- Dilation / Erosion；
- Boundary corruption。

记录：

- PSNR；
- LPIPS；
- FVD；
- FD；
- RM；
- 关键视频。

建议绘制：

[
NoiseLevel ightarrow PSNR
]

[
NoiseLevel ightarrow FVD
]

[
NoiseLevel ightarrow FD/RM
]

---

## Experiment C：Dynamic Focus

目的：

> 验证真实 Focus 随时间变化时，模型能否产生合理而连续的 Bokeh 变化。

控制：

[
FocusDistance(t)
]

主要观察：

- FD；
- RM；
- 视频定性结果；
- Focus transition 是否出现 flicker。

---

## Experiment D：Dynamic Blur Strength

控制：

[
K(t)
]

观察：

- blur strength 变化是否连续；
- foreground 是否稳定；
- Bokeh transition 是否抖动。

主要记录：

- FD；
- RM；
- 视频；
- 必要时 LPIPS / PSNR。

---

# 8. 输出目录建议

每个算法 / 每个实验统一保存：

```text
output/
├── videos/
│   ├── predicted.mp4
│   ├── gt.mp4
│   └── comparison.mp4
│
├── frames/
│   └── key_frames/
│
├── metrics.json
│   ├── PSNR
│   ├── SSIM
│   ├── LPIPS
│   ├── FVD
│   ├── VFID-I
│   ├── FD
│   └── RM
│
├── efficiency.json
│   ├── core_time
│   ├── e2e_time
│   ├── fps
│   └── peak_vram
│
└── config.yaml
    ├── checkpoint
    ├── resolution
    ├── frames
    ├── focus
    ├── K
    └── preprocessing
```

---

# 9. 导师给出的五篇论文：建议复现优先级

当前五篇：

1. BokehFlow
2. BokehCrafter
3. Any-to-Bokeh
4. MagicBokeh
5. Bokeh Diffusion

建议不是按发表顺序全部复现，而是按：

> **与 OPPO 课题的任务匹配度 + 官方代码完整度 + 对后续模型设计的直接帮助**

排序。

---

## Priority 1：Any-to-Bokeh —— 必跑

### 原因

它与我们当前问题最接近：

```text
Video
+
Explicit Geometry
+
Temporal Modeling
+
Bokeh
```

可以直接帮助理解：

- Geometry 条件如何进入网络；
- Spatial Module；
- Temporal Module；
- Geometry Noise Augmentation；
- Temporal Geometry Denoising；
- Decoder / Skip Feature；
- 边缘细节恢复。

它也是当前最适合建立统一复现实验协议的模型。

### 定位

[
oxed{	ext{核心 baseline + 源码教材}}
]

---

## Priority 2：BokehCrafter —— 理论上值得跑，但取决于官方代码

### 原因

它是真正的视频 Bokeh 方法，并且：

- 直接研究 temporal consistency；
- 包含 Dynamic Focus；
- 使用 Video Diffusion；
- VBS Benchmark 很有价值。

从任务匹配角度，它比 MagicBokeh / Bokeh Diffusion 更适合作为 Video Bokeh baseline。

### 当前限制

截至当前公开检索，只明确找到正式论文页面，没有确认到与 Any-to-Bokeh 同等完整的官方公开代码仓库。

因此：

> 如果后续拿到作者源码 / checkpoint，优先级可以直接升到第二。

若没有完整代码，则暂时不为了“凑五篇”自行重写模型。

### 定位

[
oxed{	ext{高价值视频 baseline，但受源码可得性限制}}
]

---

## Priority 3：MagicBokeh —— 建议跑，但不放核心 baseline 主表

### 原因

MagicBokeh 很值得复现用于学习：

- Focus-Aware Mask Attention；
- Geometry / Defocus Conditioning；
- Degradation-Aware Depth；
- Task Conflict；
- Alternating Training；
- one-step diffusion。

但它解决的是：

> **单图、高倍变焦输入下的恢复 + Bokeh Rendering**

而我们的核心任务是：

> **视频 + 时序 + Low-FPS Depth Guidance**

它没有：

- Temporal Module；
- Video Consistency；
- Low-FPS Depth Propagation。

因此把它与 Any-to-Bokeh / 我们的方法直接放在同一张 Video Benchmark 主表，会存在明显的 input/task mismatch。

### 推荐用途

- 跑官方 demo；
- 测单帧视觉效果；
- 拆 FAMA 代码；
- 拆 DA Depth；
- 研究 Mask 如何进入 Attention；
- 作为模块设计参考。

### 不建议

暂时不要把它放入：

```text
PSNR / SSIM / LPIPS / FVD / FD / RM
```

那张核心 Video Baseline 表中。

尤其：

- FVD；
- FD；
- RM；

对纯单图方法本身就不是同等任务条件。

### 定位

[
oxed{	ext{模块参考方法 / 补充 baseline，而不是核心 Video baseline}}
]

---

## Priority 4：Bokeh Diffusion —— 推理值得跑，完整复现优先级较低

### 价值

重点学习：

- Real + Synthetic Hybrid Training；
- independent Bokeh conditioning；
- Grounded Self-Attention；
- generative Bokeh prior；
- Physical Renderer + Neural Correction 的动机。

### 与我们的差异

它主要是：

```text
Text / Scene
+
Bokeh Parameter
↓
Image Generation / Editing
```

不是视频 Dense Prediction。

### 当前公开代码情况

目前官方仓库明确已经发布：

- model checkpoint；
- inference code。

但 README 仍将 training code 列在 TODO。

所以目前更适合：

> 跑 inference、观察 Bokeh Appearance、研究生成式 refinement。

不适合作为第一批完整训练复现对象。

### 定位

[
oxed{	ext{Neural Bokeh / Renderer Refiner 的参考}}
]

---

## Priority 5：BokehFlow —— 暂不优先跑

### 原因

它本身很有研究价值：

- Geometry Prior；
- Flow Matching；
- Depth-free Bokeh；
- Edge Evaluation。

但与我们最关键的：

- Low-FPS Depth；
- Video Temporal Modeling；
- Alpha；
- Dense Geometry；

距离较远。

同时当前没有确认到和 Any-to-Bokeh / MagicBokeh 一样成熟的官方公开源码。

网络上存在同名的非官方 “BokehFlow” 项目，需要避免与论文官方实现混淆。

### 定位

[
oxed{	ext{先读论文、保留思想，不作为当前复现优先级}}
]

---

# 10. 当前推荐顺序

如果只考虑导师给出的这五篇，我建议：

```text
第一阶段
Any-to-Bokeh
        ↓
必须完整跑通、建立 benchmark

第二阶段
BokehCrafter
        ↓
如果官方代码 / checkpoint 可获得，则优先复现

第三阶段
MagicBokeh
        ↓
跑官方 inference + 拆关键模块
不进入核心 Video baseline 主表

第四阶段
Bokeh Diffusion
        ↓
跑 inference，重点研究 Neural Bokeh Appearance / Refinement

第五阶段
BokehFlow
        ↓
暂不强求复现，除非官方源码后续公开
```

如果 BokehCrafter 暂无可用官方源码，则实际执行顺序变成：

```text
Any-to-Bokeh
→ MagicBokeh
→ Bokeh Diffusion
→ 等待 / 补充 BokehCrafter
→ BokehFlow 暂缓
```

---

# 11. 关于 MagicBokeh 是否适合作为我们的 baseline

当前结论：

[
oxed{	ext{不适合作为第一版核心 Video Baseline}}
]

原因不是 MagicBokeh 不好，而是**任务定义不完全一致**。

MagicBokeh：

- 单图；
- 高倍变焦 / degradation；
- Real-ISR + Bokeh；
- 无 temporal modeling。

我们的目标：

- Video；
- High-FPS RGB；
- Low-FPS Coarse Depth；
- temporal geometry；
- Depth / Alpha / Bokeh；
- Video consistency。

因此如果把 MagicBokeh 和我们直接放在一张主表里比较：

- FVD；
- FD；
- RM；
- Temporal Consistency；

会缺乏公平性。

更合适的做法是：

### 核心 Baseline

优先选：

- Any-to-Bokeh；
- MPIB；
- BokehMe；
- Dr.Bokeh；
- DeepLens / BokehDiff；
- 后续如有源码的 BokehCrafter。

### 补充参考

- MagicBokeh；
- Bokeh Diffusion；
- BokehFlow。

其中 MagicBokeh 主要承担：

> **模块级参考，而不是任务级 baseline。**

尤其适合我们研究：

- Mask → Attention；
- Geometry Robustness；
- Alternating Multi-task Training。


---

# 12. 当前核心 Baseline：为什么选这 7 个

当前计划将以下 7 个方法作为主要对比对象，再加上 Ours：

| Method | 代表路线 | 为什么要选 | 重点比较什么 |
|---|---|---|---|
| **DeepLens** | 单帧学习式 Bokeh / Neural Rendering | 代表较早期的学习式 Bokeh 路线，可作为从传统单帧学习方法到后续几何/生成式方法的参考基线 | 单帧重建质量、边缘保持、逐帧应用到视频后的 temporal artifact |
| **Dr.Bokeh** | Physical RGBD / Occlusion-aware Rendering | 代表强物理模型与显式遮挡处理 | 物理可控性、Depth Boundary、partial occlusion、color bleeding，以及与 Ours 的 neural correction 是否带来额外收益 |
| **BokehMe** | Physical + Neural Hybrid | 与本项目“Physical Renderer + Neural Residual”思想最接近 | 混合物理/神经路线的视觉收益、边缘 artifact、真实感与效率 |
| **MPIB** | MPI / Explicit Layered Geometry | 代表显式多平面/分层几何和遮挡建模 | Layered Geometry 对边缘和遮挡的处理，以及与我们 Depth + Alpha 分层思路的差异 |
| **BokehDiff** | One-step Diffusion / Generative Bokeh | 代表现代生成式 Bokeh 与强图像先验 | LPIPS/FVD 等感知质量、复杂边缘/高光表现，以及生成式路线与物理可控路线的质量-效率权衡 |
| **Any-to-Bokeh** | Video + Explicit Geometry + Temporal Modeling | 与我们的 Video Bokeh 任务最接近，也是当前最重要 baseline | 视频时序一致性、Geometry 条件利用、动态 Focus、Depth/Disparity Error 鲁棒性 |
| **BokehCrafter*** | Video Diffusion | 代表以 Video Diffusion / Temporal Prior 直接生成 Bokeh Video 的路线 | FVD、FD、RM、Dynamic Focus、长视频 temporal consistency，以及与显式 Geometry Video 方法的差异 |
| **Ours** | Low-FPS Depth-guided Video + Depth/Alpha + Hybrid Rendering | 本项目 | Low-FPS coarse Depth 条件下的最终 Bokeh 质量、时序、边缘、效率及鲁棒性 |

`* BokehCrafter` 只有在能够获得可运行的官方实现 / checkpoint，并能在与其他方法相同的 Benchmark 和评价协议上产生结果时，才进入正式定量主表。

---

# 13. 为什么不是“越多 Baseline 越好”

当前 Baseline 的选择原则不是简单追求数量，而是覆盖主要技术范式，并确保每个方法都有明确比较价值：

```text
DeepLens
→ 单帧学习式 Neural Bokeh

Dr.Bokeh
→ 强 Physical / RGBD Renderer

BokehMe
→ Physical + Neural Hybrid

MPIB
→ Explicit Layered Geometry

BokehDiff
→ One-step Generative / Diffusion

Any-to-Bokeh
→ Geometry-guided Video Bokeh

BokehCrafter
→ Video Diffusion / Temporal Prior

Ours
→ Low-FPS Depth-guided Video + Dense Geometry/Alpha + Hybrid Rendering
```

这样主实验可以回答的不只是“谁的 PSNR 更高”，而是：

1. **纯物理模型 vs 神经模型**：真实感、边缘和效率如何权衡？
2. **显式 Geometry vs Generative Prior**：哪类方法在可控性、边界质量和感知质量上更有优势？
3. **逐帧单图方法 vs 原生 Video 方法**：Temporal Modeling 到底带来多少收益？
4. **High-FPS / Per-frame Geometry vs Low-FPS Coarse Depth**：我们的核心输入设定是否能在更弱几何条件下保持竞争力？
5. **纯生成 Bokeh vs Physical + Neural Hybrid**：强生成先验是否值得其额外计算成本？
6. **Layered / Occlusion-aware Rendering vs Alpha-assisted Hybrid Rendering**：边缘和遮挡问题由哪种表示处理得更好？

---

# 14. 每类 Baseline 最重点看的指标

不同方法最值得关注的指标并不完全一样，但正式主表仍使用统一指标。

## 14.1 DeepLens / Dr.Bokeh / BokehMe / MPIB

这些方法如果逐帧用于视频，需要重点观察：

- PSNR ↑；
- SSIM ↑；
- LPIPS ↓；
- VEPI ↑；
- FD ↓；
- RM ↓；
- 实际视频中的 flicker。

这里最关键的问题是：

> 单帧效果好，是否意味着连续视频也稳定？

---

## 14.2 BokehDiff

重点：

- LPIPS ↓；
- FVD ↓；
- 边缘 / Highlight / Hair 的视觉质量；
- Core Time / FPS；
- Peak VRAM。

这里重点回答：

> 强生成先验是否提升感知真实感，以及这种提升付出了多少效率和可控性成本。

---

## 14.3 Any-to-Bokeh / BokehCrafter

重点：

- FVD ↓；
- VFID-I ↓；
- FD ↓；
- RM ↓；
- Dynamic Focus；
- Temporal Artifact；
- 长视频稳定性；
- Runtime / VRAM。

它们是与我们最关键的视频级对比。

---

## 14.4 Ours

除统一主表外，需要重点增加：

- Low-FPS Depth 不同采样率；
- Coarse Depth Noise；
- Mask / Confidence；
- Depth / Alpha Boundary；
- E2E Runtime；
- Peak VRAM。

也就是说，我们最终不只要回答：

> Ours 的 Bokeh 是否更好？

还要回答：

> 在只有低帧率、带噪粗 Depth 的更困难输入条件下，是否仍然能获得稳定、可控且高质量的视频 Bokeh？

