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



---

# 15. Baseline 的输入类型：单图逐帧 vs 原生视频

当前核心 baseline 并不都是“输入视频并做跨帧时序建模”。

需要明确区分：

| Method | 原生输入形式 | 原生视频模型 | 视频实验中的使用方式 |
|---|---|---:|---|
| **DeepLens** | 单张 RGB + Depth / Estimated Depth + Focus/Aperture Control | ❌ | 对视频逐帧独立处理 |
| **Dr.Bokeh** | 单张 RGB + Depth + Lens / Focus Parameters | ❌ | 对视频逐帧独立渲染 |
| **BokehMe** | 单张 All-in-Focus RGB + Disparity + Focus Disparity + Blur Strength (K) | ❌ | 对视频逐帧独立处理 |
| **MPIB** | 单张 All-in-Focus RGB + Disparity + Focus Disparity + Blur Strength (K) | ❌ | 对视频逐帧独立处理 |
| **BokehDiff** | 单张 All-in-Focus RGB + Geometry / Defocus Control | ❌ | 对视频逐帧独立处理 |
| **Any-to-Bokeh** | RGB Video + Per-frame Disparity / Depth + Focus Target / Focus Plane + Blur Strength | ✅ | 原生视频输入，并进行 temporal modeling |
| **BokehCrafter** | All-in-Focus Video + Reference Bokeh Image + Text Instruction | ✅ | 原生视频输入，并进行 temporal modeling |
| **Ours** | High-FPS RGB Video + Low-FPS Coarse Depth + Validity Mask / Confidence + Focus / Camera Parameters | ✅ | 原生视频输入，并显式利用低帧率几何先验 |

因此目前可以把对比方法简单理解为：

[
5 个单图/逐帧方法
+
2 个原生 Video 方法
+
Ours
]

这组设计能够直接比较：

[
Frame	ext{-}wise Processing
quad vs quad
Temporal Video Modeling
]

---

# 16. 各 Baseline 输入说明

## 16.1 DeepLens

典型输入可以理解为：

```text
Single RGB Image
+
Depth / Estimated Depth
+
Focus / Aperture Control
        ↓
DeepLens
        ↓
Bokeh Image
```

它本质上是单图学习式浅景深方法。

用于视频 Benchmark 时，需要逐帧运行：

```text
RGB_t + Depth_t
↓
DeepLens
↓
Bokeh_t
```

它本身没有跨帧 temporal modeling。

---

## 16.2 Dr.Bokeh

典型输入：

```text
RGB
+
Depth
+
Focus / Lens Parameters
        ↓
Occlusion-aware Physical Renderer
        ↓
Bokeh
```

核心特点不是学习视频时序，而是利用显式 Geometry 处理：

- partial occlusion；
- depth discontinuity；
- foreground/background color bleeding。

用于视频时同样是逐帧渲染。

---

## 16.3 BokehMe

典型输入：

```text
All-in-Focus RGB
+
Disparity Map
+
Focus Disparity
+
Blur Strength K
        ↓
BokehMe
        ↓
Bokeh Image
```

其中：

- Disparity 决定场景相对深度；
- Focus Disparity 决定焦平面；
- (K) 控制虚化强度。

视频实验中逐帧输入对应 RGB / Disparity。

---

## 16.4 MPIB

输入形式与 BokehMe 类似：

```text
All-in-Focus RGB
+
Disparity
+
Focus Disparity
+
Blur Strength K
        ↓
MPI Representation / Layered Rendering
        ↓
Bokeh Image
```

区别主要在内部表示：

> MPIB 将 Geometry 显式转换为 Multi-Plane Image（多平面图像）/ layered representation，以增强 partial occlusion 和前后景关系处理。

因此它非常适合和我们未来的：

```text
Depth
+
Alpha / Layer Information
```

做结构层面对比。

---

## 16.5 BokehDiff

典型输入可概括为：

```text
All-in-Focus RGB
+
Geometry / Depth-dependent Defocus Control
        ↓
One-step Diffusion
        ↓
Bokeh Image
```

它仍然属于单图方法。

其核心价值在于：

> 使用生成式先验，在显式 Bokeh / CoC 约束下恢复更真实的 lens blur appearance。

视频比较时只能逐帧运行，除非额外引入 temporal module。

---

## 16.6 Any-to-Bokeh

这是与我们最接近的输入结构之一：

```text
RGB Video
+
Per-frame Disparity / Depth
+
Focus Target / Focus Plane
+
Blur Strength
        ↓
Spatial Geometry Modeling
+
Temporal Module
        ↓
Bokeh Video
```

关键点：

> 它基本假设每一帧已经具有可用的 Geometry。

因此它和我们最大的输入差异在于：

```text
Any-to-Bokeh:
High-FPS RGB
+
Per-frame / High-FPS Geometry

Ours:
High-FPS RGB
+
Low-FPS Coarse Depth
```

这也是我们 Q3：

[
Low	ext{-}FPS Depth
ightarrow
High	ext{-}FPS Geometry
]

的核心来源。

---

## 16.7 BokehCrafter

其输入和传统 Depth-guided 方法明显不同：

```text
All-in-Focus Video
+
Reference Bokeh Image
+
Text Instruction
        ↓
Video Diffusion
        ↓
Bokeh Video
```

其中：

- All-in-Focus Video 提供场景内容；
- Reference Bokeh Image 提供 Bokeh appearance / blur style；
- Text Instruction 提供控制语义。

因此它虽然是原生 Video Bokeh 方法，但并不依赖显式 Depth / Disparity 输入。

这意味着：

> BokehCrafter 可以作为“Video Generative Prior”路线的对照，但与我们并不是完全相同的输入条件。

---

## 16.8 Ours

当前设想输入：

```text
High-FPS RGB Video
+
Low-FPS Coarse Depth
+
Depth Validity Mask
+
Depth Confidence
+
Focus / Camera Parameters
        ↓
Temporal Geometry Propagation
+
Video Transformer
        ↓
High-FPS Depth
+
Alpha
+
Physical Renderer
+
Neural Residual
        ↓
Final Video Bokeh
```

我们的核心输入特征是：

> Geometry 存在，但它是低帧率、粗糙且可能带错误的真实 Depth Prior。

因此本项目不是简单的：

```text
RGB Video → Bokeh Video
```

也不是：

```text
RGB + Perfect Per-frame Depth → Bokeh
```

而是：

[
High	ext{-}FPS RGB
+
Low	ext{-}FPS Noisy Geometry
ightarrow
High	ext{-}FPS Stable Bokeh
]

---

# 17. 为什么单图方法仍然值得放进 Video Benchmark

单图方法逐帧应用时：

```text
Frame t-1 → Bokeh Model → Output t-1
Frame t   → Bokeh Model → Output t
Frame t+1 → Bokeh Model → Output t+1
```

不同帧之间没有直接的信息交互。

即使每一帧单独看起来质量很好，连续播放时仍可能出现：

- Depth Error 在不同帧间跳动；
- Subject Boundary flicker；
- Hair / Thin Structure 抖动；
- Blur Radius 不连续；
- Focus Boundary 跳变。

而原生 Video 方法：

```text
Frame t-1
Frame t
Frame t+1
   ↓
Temporal Module / Temporal Attention
   ↓
Video Output
```

能够显式利用跨帧信息。

因此主表同时保留单图逐帧方法和原生 Video 方法，可以回答：

> **Temporal Modeling 本身到底带来了多少收益？**

---

# 18. 公平比较需要区分两种 Geometry Protocol

我们的真实输入条件为：

[
RGB_{highfps}
+
Depth_{lowfps}^{coarse}
]

而部分 baseline 原本假设每一帧都已有：

[
RGB_t + Depth_t
]

或：

[
RGB_t + Disparity_t
]

因此正式比较时不能：

- 给 baseline 每帧 GT / 高质量 Depth；
- 同时只给 Ours Low-FPS Coarse Depth。

否则 Geometry 输入条件不公平。

建议未来正式实验分成两套协议。

## Protocol A：Ideal / Per-frame Geometry

所有需要 Geometry 的方法都提供相同质量、相同帧率的 per-frame geometry：

```text
RGB Video
+
Per-frame Depth / Disparity
        ↓
各方法
```

主要回答：

> **当 Geometry 已经给定时，各种 Renderer / Video Model 本身谁更好？**

重点比较：

- PSNR；
- SSIM；
- LPIPS；
- VEPI；
- FVD；
- FD；
- RM；
- Runtime。

---

## Protocol B：OPPO Realistic Low-FPS Coarse Depth

所有方法从真实任务条件出发：

[
High	ext{-}FPS RGB
+
Low	ext{-}FPS Coarse Depth
]

对于原本要求每帧 Depth / Disparity 的 baseline，统一使用相同的前处理：

```text
Low-FPS Coarse Depth
        ↓
统一的 Depth Interpolation / Propagation
        ↓
Per-frame Geometry
        ↓
DeepLens / Dr.Bokeh / BokehMe / MPIB / ...
```

而 Ours 可以直接利用：

```text
Low-FPS Depth
+
RGB Temporal Feature
+
Mask / Confidence
```

完成 learned temporal geometry propagation。

这套协议主要回答：

> **在真正的 OPPO 输入条件下，我们的一体化 Video Geometry + Bokeh 方法是否比“先补 Depth，再逐帧渲染”的传统 pipeline 更稳定、更准确、更高效？**

这是本项目最终最关键的比较之一。


---

# 19. Benchmark 与指标的对应关系

这一节固定不同 Benchmark 应该计算哪些指标，并区分：

1. **论文原生指标（paper-native metrics）**：原论文在该 Benchmark 上实际报告的指标；
2. **统一扩展指标（our unified metrics）**：为了让不同算法在统一协议下横向比较，我们额外计算的指标。

不能把“某个指标理论上可以算”与“原论文就是这样评的”混为一谈。

---

## 19.1 多个 paired Benchmark 共用的核心指标

对于存在 paired Bokeh GT 的 Benchmark，统一保留：

| 指标 | 趋势 | 含义 |
|---|---:|---|
| PSNR | ↑ | 像素级重建准确性 |
| SSIM | ↑ | 结构保持 |
| LPIPS | ↓ | 感知特征相似度 |

因此：

\[
\boxed{PSNR,\ SSIM,\ LPIPS}
\]

作为 Any-to-Bokeh Synthetic、VBS 和 OPPO-style Synthetic 的共同画质指标。

---

## 19.2 所有方法共用的效率指标

效率不属于某一个特定 Benchmark，可统一记录：

| 指标 | 趋势 | 含义 |
|---|---:|---|
| Core Time | ↓ | 核心模型推理时间 |
| E2E Time | ↓ | 包含 Depth / Mask 等前处理的完整耗时 |
| FPS | ↑ | 实际处理帧率 |
| Peak VRAM | ↓ | 峰值显存 |
| Params | ↓ | 模型参数量 |

统一测试条件：

- 同一 GPU；
- 相同分辨率；
- 相同帧数；
- batch size = 1；
- warm-up 后测量；
- 多次重复取均值。

---

# 20. Any-to-Bokeh Synthetic Benchmark：指标

## 20.1 论文原生指标

Any-to-Bokeh 的 synthetic test set 原生报告：

\[
\boxed{
PSNR,\ SSIM,\ LPIPS,\ FVD,\ VFID\text{-}I,\ FD,\ RM
}
\]

其中：

- **PSNR ↑**：像素级 fidelity；
- **SSIM ↑**：结构相似度；
- **LPIPS ↓**：感知相似度；
- **FVD ↓**：视频整体分布质量；
- **VFID-I ↓**：基于 I3D feature 的视频分布距离；
- **FD ↓**：Flow Difference，比较预测与 GT 的运动/光流关系；
- **RM ↓**：Relation Metric，比较预测帧间变化与 GT 帧间变化。

RM 的核心不是要求相邻帧不变化，而是：

\[
Predicted\ Temporal\ Change
\approx
GT\ Temporal\ Change
\]

## 20.2 我们的建议

这一套指标建议全部沿用。

推荐主表：

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | VFID-I ↓ | FD ↓ | RM ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepLens | | | | | | | |
| Dr.Bokeh | | | | | | | |
| BokehMe | | | | | | | |
| MPIB | | | | | | | |
| BokehDiff | | | | | | | |
| Any-to-Bokeh | | | | | | | |
| BokehCrafter* | | | | | | | |
| Ours | | | | | | | |

---

# 21. VBS Benchmark：指标

VBS 是 BokehCrafter 的视频 Bokeh Benchmark，包含：

- paired Bokeh video；
- Static Focus；
- Dynamic Focus；
- 不同 blur strength。

## 21.1 BokehCrafter 论文原生指标

BokehCrafter 在 VBS 上使用：

\[
\boxed{
PSNR,\ SSIM,\ LPIPS
}
\]

评价 Bokeh rendering quality。

同时使用专门的 temporal consistency formulation：

\[
L_{temp}
=
\sqrt{
\frac{1}{N-1}
\sum_{i=1}^{N-1}
\left|
(F_{i+1}^{GT}-F_i^{GT})
-
(F_{i+1}^{pred}-F_i^{pred})
\right|^2
}
\]

它比较：

\[
GT\ frame\text{-}to\text{-}frame\ change
\]

和：

\[
Predicted\ frame\text{-}to\text{-}frame\ change
\]

是否一致。

因此仍体现：

\[
Temporal\ Consistency
\neq
Temporal\ Constancy
\]

## 21.2 我们在 VBS 上建议使用

### 必须保留原生指标

\[
\boxed{
PSNR,\ SSIM,\ LPIPS,\ Temporal\ Consistency
}
\]

这样可以和 BokehCrafter 原论文对齐。

### 推荐额外统一计算

如果评价代码能够统一，可额外计算：

\[
FVD,\ VFID\text{-}I,\ FD,\ RM
\]

这些应明确标记为：

> our unified evaluation extension

而不是声称 BokehCrafter 原论文全部使用了这些指标。

## 21.3 Static / Dynamic Focus 分组

VBS 最值得单独报告：

| Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Temporal Consistency ↓ | FD ↓ | RM ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Static Focus | | | | | | |
| Dynamic Focus | | | | | | |

目的：

> 检查模型是否会把真实的动态对焦变化错误地压成“稳定但错误”的输出。

---

# 22. DAVIS Real-world Benchmark：指标

DAVIS 是真实视频，没有严格配对的真实 Bokeh GT。

因此不能把 against-Bokeh-GT 的：

- PSNR；
- SSIM；
- LPIPS；

作为主指标。

如果 FD / RM 的定义依赖 GT Bokeh temporal change，也不能直接照搬 synthetic benchmark。

## 22.1 原生指标：VEPI

Any-to-Bokeh 在 DAVIS 上使用：

\[
\boxed{VEPI\uparrow}
\]

VEPI（Video Edge Preservation Index）主要评价：

- focused subject edge；
- Hair；
- Thin Structure；
- 前景边缘是否被背景 blur 吞噬。

它利用 DAVIS 的 foreground segmentation mask 构造 ROI，并比较原始 focused foreground 与 Bokeh 输出的 Laplacian edge response。

## 22.2 DAVIS 推荐评价方式

第一阶段：

\[
\boxed{
VEPI
+
Qualitative\ Video
+
Fixed\ Key\ Frames
}
\]

后续需要更正式时，再增加：

- blinded user study；
- MOS / preference study。

第一阶段不要自行构造没有文献依据的“1～5 分 temporal stability”。

## 22.3 固定定性观察项

每段视频建议固定观察：

- Hair / Thin Structure；
- Foreground Halo；
- Background Leakage；
- Color Bleeding；
- Focus Error；
- Temporal Flicker；
- Occlusion / Disocclusion；
- Camera Motion；
- Fast Motion。

---

# 23. OPPO-style Low-FPS Coarse Depth Benchmark：指标

这是为了本项目核心输入条件构造的专项 Benchmark：

\[
High\text{-}FPS\ RGB
+
Low\text{-}FPS\ Coarse\ Depth
\]

如果它建立在有完整 GT 的 synthetic data 上，则可以同时拥有：

- Bokeh GT；
- Full-FPS Depth GT；
- 可选 Alpha GT。

因此可以同时评价最终 Bokeh 与中间任务。

## 23.1 Final Bokeh

尽量与 Any-to-Bokeh Synthetic 对齐：

\[
\boxed{
PSNR,\ SSIM,\ LPIPS,\ FVD,\ VFID\text{-}I,\ FD,\ RM
}
\]

这样能够直接观察：

> Depth FPS / Depth Quality 下降以后，最终 Bokeh 如何退化。

## 23.2 Depth 指标

如果有 Full-FPS Depth GT：

| 指标 | 趋势 | 含义 |
|---|---:|---|
| AbsRel | ↓ | 相对深度误差 |
| RMSE | ↓ | 均方根深度误差 |
| δ1 | ↑ | 阈值内深度像素比例 |
| Depth Edge Error | ↓ | 深度突变边缘恢复 |

这可以回答：

> Temporal Geometry Propagation 是否真正恢复了高帧率 Geometry。

## 23.3 Alpha 指标

如果存在 Alpha GT：

| 指标 | 趋势 | 含义 |
|---|---:|---|
| SAD | ↓ | Alpha absolute difference |
| MSE | ↓ | Alpha 均方误差 |
| Gradient Error | ↓ | 发丝/软边缘恢复 |
| Connectivity Error | ↓ | 前景拓扑/连通性 |

Depth / Alpha 指标单独建表，不与最终 Bokeh 指标混成一张表。

## 23.4 Low-FPS Depth 专项变量

建议 Depth FPS：

\[
30,\ 15,\ 10,\ 5,\ 2\ fps
\]

例如：

| Depth FPS | PSNR ↑ | LPIPS ↓ | FVD ↓ | FD ↓ | RM ↓ | Depth AbsRel ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | | | | | | |
| 15 | | | | | | |
| 10 | | | | | | |
| 5 | | | | | | |
| 2 | | | | | | |

Depth Quality 建议分：

- Clean；
- Mild；
- Medium；
- Severe。

候选退化：

- smooth / Gaussian perturbation；
- local holes；
- boundary corruption；
- dilation / erosion；
- temporal missing frames；
- confidence degradation。

可进一步做二维鲁棒性表：

| Depth FPS \ Quality | Clean | Mild | Medium | Severe |
|---|---:|---:|---:|---:|
| 30 fps | | | | |
| 15 fps | | | | |
| 10 fps | | | | |
| 5 fps | | | | |

---

# 24. 四个 Benchmark 的指标总表

| Benchmark | PSNR | SSIM | LPIPS | FVD | VFID-I | FD | RM | Temporal Consistency | VEPI | Depth | Alpha |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Any-to-Bokeh Synthetic** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | 可选 | — | — |
| **VBS** | ✅ | ✅ | ✅ | 推荐扩展 | 推荐扩展 | 推荐扩展 | 推荐扩展 | ✅ 原生 | 可选 | — | — |
| **DAVIS** | — | — | — | — | — | — | — | 定性为主 | ✅ | — | — |
| **OPPO-style Synthetic** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 可选 | 推荐 | ✅ | ✅（若有 GT） |

---

# 25. 最终统一指标层级

## 第一层：所有 paired Bokeh Benchmark 共用

\[
\boxed{
PSNR,\ SSIM,\ LPIPS
}
\]

回答：

> 最终 Bokeh 的单帧重建质量怎么样？

## 第二层：有 GT 的视频 Benchmark 尽量统一

建议：

\[
\boxed{
FVD,\ FD,\ RM
}
\]

并在需要与 Any-to-Bokeh 完全对齐时保留：

\[
VFID\text{-}I
\]

回答：

> 视频整体质量和 temporal behavior 怎么样？

VBS 同时必须保留 BokehCrafter 原论文的 Temporal Consistency，方便与原论文结果对齐。

## 第三层：专项指标

### DAVIS

\[
\boxed{VEPI}
\]

### OPPO-style

\[
\boxed{
Depth\ Metrics
+
Alpha\ Metrics
+
Depth\ FPS/Noise\ Robustness
}
\]

---

# 26. 当前 Benchmark 体系

建议现阶段固定为：

~~~text
1. Any-to-Bokeh Synthetic
   → 主定量
   → PSNR / SSIM / LPIPS
   → FVD / VFID-I / FD / RM

2. VBS
   → 第二视频 Benchmark
   → PSNR / SSIM / LPIPS
   → BokehCrafter Temporal Consistency
   → Static / Dynamic Focus 分组
   → 可选统一 FVD / FD / RM

3. DAVIS
   → Real-world Generalization
   → VEPI
   → Qualitative Video
   → User Study（后续）

4. OPPO-style Low-FPS Depth
   → Project-specific Benchmark
   → Final Bokeh Metrics
   → Depth Metrics
   → Alpha Metrics
   → FPS / Noise Robustness
~~~

四层分别回答：

1. 标准 paired 场景下做得准不准？
2. 动态对焦与复杂视频条件下稳不稳？
3. 真实场景泛化行不行？
4. Low-FPS + Coarse / Noisy Depth 下，我们的核心创新有没有价值？


---

# 27. VBS 主表与 Dynamic Focus 专项评测的最终组织方式

当前决定：

> **VBS 本身作为一个完整 Benchmark，不再把 Dynamic Focus 单独称为新的 Benchmark。**

VBS 内部包含：

- S-Focus：Static Focus；
- D-Focus：Dynamic Focus。

因此：

~~~text
VBS Benchmark
│
├── Static Focus samples
└── Dynamic Focus samples
~~~

Dynamic Focus 的定位是：

> **VBS Dynamic-Focus Subset Evaluation / VBS 动态对焦专项评测**

而不是一个独立的新 Benchmark。

---

# 28. 最终 Benchmark 表格模板

## Table 1：Any-to-Bokeh Synthetic

统一使用六个核心指标：

\[
PSNR,\ SSIM,\ LPIPS,\ FVD,\ FD,\ RM
\]

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | FD ↓ | RM ↓ |
|---|---:|---:|---:|---:|---:|---:|
| DeepLens |  |  |  |  |  |  |
| Dr.Bokeh |  |  |  |  |  |  |
| BokehMe |  |  |  |  |  |  |
| MPIB |  |  |  |  |  |  |
| BokehDiff |  |  |  |  |  |  |
| Any-to-Bokeh |  |  |  |  |  |  |
| BokehCrafter* |  |  |  |  |  |  |
| **Ours** |  |  |  |  |  |  |

说明：

- PSNR / SSIM / LPIPS：最终 Bokeh 单帧质量；
- FVD：整段视频的总体分布质量；
- FD：预测与 GT 的运动/光流关系差异；
- RM：预测帧间变化与 GT 帧间变化的一致性；
- BokehCrafter* 只有在能够在同一 Benchmark 上运行时进入正式定量表。

---

## Table 2：VBS

前六个指标与 Any-to-Bokeh Synthetic 保持一致，并额外增加 VBS / BokehCrafter 原生使用的 Temporal Consistency：

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | FD ↓ | RM ↓ | Temporal Consistency ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepLens |  |  |  |  |  |  |  |
| Dr.Bokeh |  |  |  |  |  |  |  |
| BokehMe |  |  |  |  |  |  |  |
| MPIB |  |  |  |  |  |  |  |
| BokehDiff |  |  |  |  |  |  |  |
| Any-to-Bokeh |  |  |  |  |  |  |  |
| BokehCrafter |  |  |  |  |  |  |  |
| **Ours** |  |  |  |  |  |  |  |

说明：

- PSNR / SSIM / LPIPS：paired Bokeh reconstruction；
- FVD / FD / RM：为了与 Any-to-Bokeh Synthetic 统一而额外计算；
- Temporal Consistency：保留 BokehCrafter / VBS 的原生时序评价方式。

因此 VBS 主表可以同时满足：

1. 与 Any-to-Bokeh Synthetic 横向使用相同的六个指标；
2. 与 BokehCrafter 原论文保持 Temporal Consistency 的可比性。

---

## Table 3：VBS Focus-type Analysis

Dynamic Focus 不作为新 Benchmark，而是 VBS 内部的专项子集分析。

优先比较真正具有 Video Temporal Modeling 的方法：

| Method | Focus Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD ↓ | RM ↓ | Temporal Consistency ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Any-to-Bokeh | Static |  |  |  |  |  |  |
| Any-to-Bokeh | Dynamic |  |  |  |  |  |  |
| BokehCrafter | Static |  |  |  |  |  |  |
| BokehCrafter | Dynamic |  |  |  |  |  |  |
| **Ours** | Static |  |  |  |  |  |  |
| **Ours** | Dynamic |  |  |  |  |  |  |

重点回答：

> 当 Focus Plane 随时间真实变化时，模型能否既保留正确的 Focus Transition，又避免错误 Flicker？

这里检验的是：

\[
Temporal\ Consistency
\neq
Temporal\ Constancy
\]

---

## Table 4：DAVIS Real-world

DAVIS 没有 paired Bokeh GT，因此不强行计算 against-GT 的 PSNR / SSIM / LPIPS。

| Method | VEPI ↑ | Hair / Thin Structure | Halo / Leakage | Temporal Flicker | Qualitative Video |
|---|---:|---|---|---|---|
| DeepLens |  |  |  |  |  |
| Dr.Bokeh |  |  |  |  |  |
| BokehMe |  |  |  |  |  |
| MPIB |  |  |  |  |  |
| BokehDiff |  |  |  |  |  |
| Any-to-Bokeh |  |  |  |  |  |
| BokehCrafter* |  |  |  |  |  |
| **Ours** |  |  |  |  |  |

其中：

- VEPI 是主要量化指标；
- Hair / Thin Structure、Halo / Leakage、Temporal Flicker 第一阶段作为固定 qualitative checklist；
- 后续如需量化真实视频主观质量，再增加 blinded user study / MOS。

---

## Table 5：Efficiency

所有算法尽可能在统一硬件、分辨率和帧数下重新测量：

| Method | Resolution | Frames | Core Time ↓ | E2E Time ↓ | FPS ↑ | Peak VRAM ↓ | Params ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepLens | 1024×576 | 25 |  |  |  |  |  |
| Dr.Bokeh | 1024×576 | 25 |  |  |  |  |  |
| BokehMe | 1024×576 | 25 |  |  |  |  |  |
| MPIB | 1024×576 | 25 |  |  |  |  |  |
| BokehDiff | 1024×576 | 25 |  |  |  |  |  |
| Any-to-Bokeh | 1024×576 | 25 |  |  |  |  |  |
| BokehCrafter* | 1024×576 | 25 |  |  |  |  |  |
| **Ours** | 1024×576 | 25 |  |  |  |  |  |

Core Time：

> 只统计核心网络 / Renderer 的计算时间。

E2E Time：

> 包含 Depth Estimation、Mask Extraction、Geometry Preprocessing、核心模型与 Renderer 的完整 pipeline 时间。

---

# 29. 当前 Benchmark Suite 的最终层级

~~~text
1. Unified Paired Video Bokeh Benchmark Suite
   │
   ├── Set A: Any-to-Bokeh Synthetic
   │      └── PSNR / SSIM / LPIPS / FVD / FD / RM
   │
   └── Set B: VBS
          ├── PSNR / SSIM / LPIPS / FVD / FD / RM
          ├── + Temporal Consistency
          └── Dynamic Focus Subset 专项分析

2. DAVIS
   └── Real-world Generalization
       VEPI + Qualitative Video

3. OPPO-style Low-FPS Coarse Depth Benchmark
   └── Project-specific
       Final Bokeh + Depth + Alpha + FPS/Noise Robustness
~~~

“Unified”指：

- 统一输出格式；
- 统一核心评价代码；
- 统一六个主指标；
- 尽可能统一输入预处理；

但：

> 不把 Any-to-Bokeh Synthetic 和 VBS 的样本混成一个数据集，也不把两个数据集的指标简单平均成一个总分。



---

# 30. Track A / Track B 双轨评价协议

为了把“Bokeh 模型本身的能力”和“Low-FPS Coarse Depth 适应能力”分开评价，当前建议采用两条实验 Track。

---

## Track A：Unified Per-frame Geometry

### 输入条件

对于需要 Geometry 的方法，尽可能提供统一质量、统一帧率的 per-frame Depth / Disparity，并统一：

- Focus Plane；
- Blur Strength / \(K\)；
- 分辨率；
- 视频长度；
- 评价代码。

示意：

~~~text
RGB Video
+
Per-frame Geometry
+
Focus / Blur Control
        ↓
各 Baseline / Ours
        ↓
Bokeh Video
~~~

### 主要回答

> **当 Geometry 条件已经充分提供时，各方法的 Bokeh Rendering / Video Modeling 本身谁更强？**

这里尽量削弱 Geometry 输入质量差异的影响，更关注：

- Renderer；
- Neural Prior；
- Occlusion Handling；
- Temporal Modeling；
- Boundary Quality；
- Video Generation Quality。

### 主要指标

- PSNR；
- SSIM；
- LPIPS；
- FVD；
- FD；
- RM；
- VBS 上额外 Temporal Consistency；
- DAVIS 不属于该 Track 的主要 quantitative 场景。

---

## Track B：Low-FPS Coarse Geometry

### 输入条件

所有方法从更接近本项目真实条件的输入出发：

\[
High\text{-}FPS\ RGB
+
Low\text{-}FPS\ Coarse\ Depth
\]

对于原本需要每帧 Geometry 的 baseline，统一增加相同的 Depth interpolation / propagation preprocessing：

~~~text
Low-FPS Coarse Depth
        ↓
统一 Depth Interpolation / Propagation
        ↓
Per-frame Geometry
        ↓
Baseline
~~~

而 Ours 可以直接利用：

~~~text
High-FPS RGB
+
Low-FPS Coarse Depth
+
Mask / Confidence
        ↓
Learned Temporal Geometry Propagation
        ↓
Depth / Alpha / Bokeh
~~~

### 主要回答

> **在 Low-FPS、粗糙、可能存在误差的 Geometry 条件下，谁更稳定、更准确？**

这条 Track 才直接对应本项目的核心问题：

\[
Low\text{-}FPS\ Coarse\ Depth
\rightarrow
High\text{-}FPS\ Stable\ Video\ Bokeh
\]

### 主要指标

最终 Bokeh：

- PSNR；
- SSIM；
- LPIPS；
- FVD；
- FD；
- RM。

若有 Geometry GT：

- AbsRel；
- RMSE；
- \(\delta_1\)；
- Depth Edge Error。

若有 Alpha GT：

- SAD；
- MSE；
- Gradient Error；
- Connectivity Error。

同时记录：

- Depth FPS；
- Depth Noise Level；
- Core Time；
- E2E Time；
- FPS；
- Peak VRAM。

---

# 31. 哪些 Benchmark 做双 Track

不需要把所有 Benchmark 机械重复两遍。

建议如下：

| Benchmark | Track A：Per-frame Geometry | Track B：Low-FPS Coarse Depth | 说明 |
|---|:---:|:---:|---|
| **Any-to-Bokeh Synthetic** | ✅ | ✅ | 有 GT / Geometry，最适合做双 Track |
| **VBS** | ✅ | ✅ | 有 GT，且可结合 Static / Dynamic Focus |
| **DAVIS** | ❌ 不作为严格 Track A | ✅ 以真实条件泛化为主 | 无真实 Bokeh GT / Full Geometry GT |
| **OPPO-style Benchmark** | 可作为参考上限 | ✅ 核心 | 本身就是为 Track B 设计 |

---

# 32. Any-to-Bokeh Synthetic 的双 Track

## Track A

~~~text
RGB Video
+
Full / Per-frame Geometry
+
Focus / K
        ↓
各方法
~~~

回答：

> 如果大家都有完整 Geometry，谁的 Bokeh 本身最好？

---

## Track B

从原本 Full-FPS Geometry 人为降采样：

~~~text
30 fps Geometry
↓
15 / 10 / 5 / 2 fps
↓
加入 Noise / Hole / Boundary Error
↓
各方法
~~~

回答：

> Geometry 帧率和质量下降后，各方法性能如何退化？

建议同时画：

\[
DepthFPS \rightarrow PSNR
\]

\[
DepthFPS \rightarrow LPIPS
\]

\[
DepthFPS \rightarrow FVD
\]

\[
DepthFPS \rightarrow FD/RM
\]

---

# 33. VBS 的双 Track

## Track A

使用 VBS 原始完整 Geometry / Focus 条件。

主表：

\[
PSNR,\ SSIM,\ LPIPS,\ FVD,\ FD,\ RM,\ Temporal\ Consistency
\]

---

## Track B

对 VBS Geometry 做 Low-FPS / Coarse 模拟：

\[
Full\ Geometry
\rightarrow
Low\text{-}FPS\ Geometry
\]

再分别测试：

- Static Focus；
- Dynamic Focus。

这样可以额外回答：

> 当 Geometry 本身已经变稀疏时，Dynamic Focus 是否更容易造成时序错误？

---

# 34. DAVIS 不做机械双 Track

DAVIS 的主要职责是：

> **Real-world Generalization**

它没有严格 paired Bokeh GT，也没有天然完整 Geometry GT。

因此不建议强行构造一个“Track A DAVIS”和“Track B DAVIS”并比较全参考指标。

更合适的是：

~~~text
Real RGB Video
+
现实可获得的 Geometry / Estimated Geometry
        ↓
各方法
        ↓
VEPI
+
Qualitative Video
+
Fixed Key Frames
~~~

重点观察：

- Hair；
- Thin Structure；
- Halo；
- Leakage；
- Flicker；
- Occlusion；
- Fast Motion。

---

# 35. OPPO-style Benchmark 的定位

OPPO-style Benchmark 本身就是 Track B 的专项 Benchmark。

核心条件：

\[
High\text{-}FPS\ RGB
+
Low\text{-}FPS\ Coarse\ Depth
\]

因此它不需要再完整复制一个 Track A。

可以保留 Full-FPS / Clean Geometry 结果作为：

> Upper-bound / Reference Setting

但主要实验仍放在：

- Depth FPS；
- Depth Noise；
- Missing Geometry；
- Confidence；
- Boundary Error；

这些真实问题上。

---

# 36. 双 Track 的最终作用

两个 Track 分别回答不同科学问题：

## Track A

\[
\boxed{
\text{Bokeh Rendering / Video Modeling 本身谁更强？}
}
\]

尽量控制 Geometry 输入差异。

## Track B

\[
\boxed{
\text{Low-FPS Coarse Geometry 条件下谁更强？}
}
\]

直接验证本项目的核心贡献。

因此两个 Track 都做是有价值的，但不是简单把所有 Benchmark 复制两遍，而是：

~~~text
Any-to-Bokeh Synthetic
├── Track A
└── Track B

VBS
├── Track A
└── Track B
    └── Dynamic Focus 专项

DAVIS
└── Real-world Generalization

OPPO-style
└── Track B Core Benchmark
~~~

这样可以把：

> Geometry 条件差异

与：

> Bokeh 模型 / Temporal Modeling 能力差异

尽可能分开分析。
