# Benchmark、Baseline 与统一评价协议

> 当前版本：第一版统一实验协议  
> 适用项目：OPPO Depth-Guided Video Bokeh  
> 目的：在开始大规模复现和自研模型前，先固定 Benchmark、Baseline、评价指标、输入公平性和结果表格，后续所有算法尽量按同一协议填写结果。

---

# 1. 当前实验目标

本项目最终关注的是：

\[
\text{High-FPS RGB Video}
+
\text{Low-FPS Coarse Depth}
\rightarrow
\text{High-FPS Stable Video Bokeh}
\]

并计划同时涉及：

- High-FPS Dense Depth；
- Alpha Matte；
- Physical Bokeh Rendering；
- Neural Residual / Refinement；
- Temporal Modeling。

因此实验需要同时回答四类问题：

1. 最终 Bokeh 画质是否好？
2. 视频时序是否稳定且保留真实变化？
3. 低帧率、粗糙、带误差的 Depth 是否会导致明显性能下降？
4. 模型的效率是否适合后续工程化？

---

# 2. Benchmark 总体结构

当前建议采用三层 Benchmark 体系。

~~~text
Benchmark Suite
│
├── 1. Unified Paired Video Bokeh Benchmark Suite
│      ├── Set A: Any-to-Bokeh Synthetic
│      └── Set B: VBS
│
├── 2. DAVIS
│      └── Real-world Generalization
│
└── 3. OPPO-style Low-FPS Coarse Depth Benchmark
       └── Project-specific Benchmark
~~~

这里的 Unified 指：

- 尽量统一输入预处理；
- 统一输出格式；
- 统一核心评价代码；
- 统一核心指标；

但不会把 Any-to-Bokeh Synthetic 与 VBS 的样本混在一起，也不会把两个数据集的结果直接平均成一个总分。

---

# 3. Benchmark 1：Any-to-Bokeh Synthetic

## 3.1 定位

这是当前最重要的 paired video Bokeh 主定量 Benchmark。

参考 Any-to-Bokeh 的测试协议：

- synthetic paired video；
- 有 Bokeh GT；
- 有 Geometry / disparity；
- 有 focus plane；
- 有 blur strength；
- 视频级测试；
- 可进行完整的全参考定量评价。

参考规模：

- 200 videos；
- 25 frames / video；
- 1024 × 576。

如果无法拿到论文完全相同的测试集，则按官方数据生成逻辑重建固定测试集，并明确标记为：

> our reproduced synthetic benchmark

同时固定：

- 数据来源；
- 随机种子；
- 视频长度；
- 分辨率；
- focus 设置；
- blur strength；
- geometry 生成方式。

---

## 3.2 主指标

统一使用六个核心指标：

\[
\boxed{
PSNR,\ SSIM,\ LPIPS,\ FVD,\ FD,\ RM
}
\]

| Metric | 趋势 | 主要含义 |
|---|---:|---|
| PSNR | ↑ | 像素级重建准确性 |
| SSIM | ↑ | 结构相似度 |
| LPIPS | ↓ | 感知相似度 |
| FVD | ↓ | 视频整体分布质量 |
| FD | ↓ | 预测与 GT 的运动 / 光流关系差异 |
| RM | ↓ | 预测帧间变化与 GT 帧间变化的一致性 |

Any-to-Bokeh 原论文还使用 VFID-I。为了完整复现原论文时可以继续计算，但当前主表优先保留以上六个指标，减少重复的视频分布指标。

---

# 4. Benchmark 2：VBS

## 4.1 定位

VBS（Video Bokeh Scenes）作为第二个 paired video Bokeh Benchmark。

它的重要价值是：

- 视频级 paired Bokeh；
- 多种 focus / blur setting；
- Static Focus；
- Dynamic Focus；
- 更适合专门分析动态对焦时的 temporal behavior。

---

## 4.2 主指标

VBS 主表前六个指标与 Any-to-Bokeh Synthetic 保持一致：

\[
PSNR,\ SSIM,\ LPIPS,\ FVD,\ FD,\ RM
\]

同时增加 BokehCrafter / VBS 原生使用的：

\[
\boxed{Temporal\ Consistency}
\]

因此 VBS 主表一共七个指标：

\[
\boxed{
PSNR,\ SSIM,\ LPIPS,\ FVD,\ FD,\ RM,\ Temporal\ Consistency
}
\]

需要注明：

- PSNR / SSIM / LPIPS / Temporal Consistency：用于和 BokehCrafter 原协议对齐；
- FVD / FD / RM：属于我们的统一扩展评价，用于和 Any-to-Bokeh Synthetic 统一。

---

## 4.3 Dynamic Focus 的定位

Dynamic Focus 不是一个新的 Benchmark。

它是 VBS 内部的专项测试子集：

~~~text
VBS
├── Static Focus
└── Dynamic Focus
~~~

因此单独做：

> VBS Dynamic-Focus Subset Evaluation

用于分析：

\[
Temporal\ Consistency
\neq
Temporal\ Constancy
\]

也就是模型既要抑制错误 Flicker，也必须保留真实的焦点变化。

---

# 5. Benchmark 3：DAVIS

## 5.1 定位

DAVIS 负责 Real-world Generalization。

它是真实自然视频，但没有严格 paired 的真实 Bokeh GT。

因此不能像 synthetic benchmark 一样直接使用 against-GT 的：

- PSNR；
- SSIM；
- LPIPS。

---

## 5.2 主要指标

第一阶段使用：

\[
\boxed{
VEPI
+
Qualitative\ Video
+
Fixed\ Key\ Frames
}
\]

其中 VEPI 主要评价：

- focused subject edge；
- Hair；
- Thin Structure；
- 主体边缘是否被背景 Blur 吞噬。

后续如需要更正式的真实视频评价，再增加：

- blinded user study；
- MOS / preference study。

---

## 5.3 固定定性观察项

DAVIS 每段视频固定观察：

- Hair / Thin Structure；
- Foreground Halo；
- Background Leakage；
- Color Bleeding；
- Focus Error；
- Temporal Flicker；
- Occlusion / Disocclusion；
- Camera Motion；
- Fast Motion。

不自行构造没有文献依据的“1～5 分稳定性指标”。

---

# 6. Benchmark 4：OPPO-style Low-FPS Coarse Depth

## 6.1 定位

这是后续最能体现本项目核心创新的专项 Benchmark。

输入模拟：

\[
High\text{-}FPS\ RGB
+
Low\text{-}FPS\ Coarse\ Depth
\]

如果基于有完整 GT 的 synthetic dataset 构造，可以同时拥有：

- Full-FPS Bokeh GT；
- Full-FPS Depth GT；
- 可选 Alpha GT。

---

## 6.2 Final Bokeh 指标

尽量与 Any-to-Bokeh Synthetic 保持一致：

\[
\boxed{
PSNR,\ SSIM,\ LPIPS,\ FVD,\ FD,\ RM
}
\]

---

## 6.3 Depth 指标

如果有 Full-FPS Depth GT：

| Metric | 趋势 | 主要含义 |
|---|---:|---|
| AbsRel | ↓ | 相对深度误差 |
| RMSE | ↓ | 均方根深度误差 |
| \(\delta_1\) | ↑ | 阈值内的深度像素比例 |
| Depth Edge Error | ↓ | 深度突变边缘恢复 |

---

## 6.4 Alpha 指标

如果有 Alpha GT：

| Metric | 趋势 | 主要含义 |
|---|---:|---|
| SAD | ↓ | Alpha absolute difference |
| MSE | ↓ | Alpha 均方误差 |
| Gradient Error | ↓ | 发丝 / 软边缘恢复 |
| Connectivity Error | ↓ | 前景连通与拓扑结构 |

Depth 和 Alpha 指标单独建表，不与最终 Bokeh 主表混在一起。

---

## 6.5 Low-FPS / Noise 专项变量

Depth FPS 建议至少测试：

\[
30,\ 15,\ 10,\ 5,\ 2\ fps
\]

Depth Quality 可分：

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

---

# 7. 当前核心 Baseline

当前核心比较方法：

| Method | 代表路线 | 原生输入类型 | 原生视频模型？ |
|---|---|---|:---:|
| DeepLens | 单帧学习式 Neural Bokeh | RGB + Depth + Focus/Aperture | ❌ |
| Dr.Bokeh | Physical RGBD / Occlusion-aware Rendering | RGB + Depth + Lens/Focus Parameters | ❌ |
| BokehMe | Physical + Neural Hybrid | RGB + Disparity + Focus Disparity + \(K\) | ❌ |
| MPIB | MPI / Explicit Layered Geometry | RGB + Disparity + Focus Disparity + \(K\) | ❌ |
| BokehDiff | One-step Diffusion / Generative Bokeh | RGB + Geometry / Defocus Control | ❌ |
| Any-to-Bokeh | Video + Geometry + Temporal Modeling | RGB Video + Per-frame Geometry + Focus + \(K\) | ✅ |
| BokehCrafter* | Video Diffusion | AiF Video + Reference Bokeh Image + Text | ✅ |
| Ours | Low-FPS Depth-guided Video + Hybrid Rendering | High-FPS RGB + Low-FPS Coarse Depth + Mask/Confidence + Camera/Focus | ✅ |

\* BokehCrafter 只有在能获得可运行实现 / checkpoint，并能在统一 Benchmark 上实际产生结果时才进入正式定量主表。

---

# 8. 为什么选这些 Baseline

## DeepLens

代表较早期学习式单帧 Bokeh。

重点比较：

- 单帧重建质量；
- 边缘保持；
- 逐帧应用后是否出现 temporal artifact。

## Dr.Bokeh

代表强物理 RGBD Renderer 和显式遮挡处理。

重点比较：

- Depth Boundary；
- Partial Occlusion；
- Color Bleeding；
- 物理可控性；
- Ours 的 Neural Correction 是否带来额外收益。

## BokehMe

代表：

\[
Physical + Neural
\]

与我们：

\[
Physical\ Renderer + Neural\ Residual
\]

的整体思想最接近。

重点比较：

- Hybrid Rendering 是否改善物理 Renderer artifact；
- 边缘质量；
- 真实感；
- 效率。

## MPIB

代表 Explicit Layered Geometry / MPI。

重点比较：

- Layered Geometry；
- Occlusion；
- 前后景关系；
- 与我们 Depth + Alpha 分层思路的差异。

## BokehDiff

代表现代 One-step Diffusion / Generative Bokeh。

重点比较：

- LPIPS；
- FVD；
- Hair / Highlight / Complex Boundary；
- Generative Prior 带来的视觉收益；
- Runtime / VRAM 成本。

## Any-to-Bokeh

当前最重要的直接 baseline。

它代表：

\[
Video + Explicit\ Geometry + Temporal\ Modeling
\]

重点比较：

- Temporal Consistency；
- Geometry Condition；
- Dynamic Focus；
- Geometry Noise Robustness；
- 与我们 Low-FPS Geometry 输入条件的差异。

## BokehCrafter

代表 Video Diffusion / Temporal Prior。

重点比较：

- FVD；
- FD；
- RM；
- Temporal Consistency；
- Dynamic Focus；
- Video Diffusion 与显式 Geometry Video 方法的差异。

---

# 9. 单图方法 vs 原生视频方法

目前核心 baseline 可以分成：

\[
5\ 个单图/逐帧方法
+
2\ 个原生 Video 方法
+
Ours
\]

单图方法用于视频时：

~~~text
Frame t-1 → Model → Output t-1
Frame t   → Model → Output t
Frame t+1 → Model → Output t+1
~~~

没有显式跨帧信息交互。

原生视频方法：

~~~text
Frame t-1
Frame t
Frame t+1
   ↓
Temporal Module / Temporal Attention
   ↓
Video Output
~~~

因此将两类方法同时放进 Video Benchmark，可以直接回答：

> Temporal Modeling 本身到底带来了多少收益？

---

# 10. 统一输入协议：Track A / Track B

为了把 Geometry 输入质量差异与 Bokeh / Video 模型能力差异尽可能分开，建议采用双 Track。

---

## Track A：Unified Per-frame Geometry

### 输入

对于需要 Geometry 的方法，尽可能提供统一质量、统一帧率的：

- Per-frame Depth / Disparity；
- Focus Plane；
- Blur Strength / \(K\)。

~~~text
RGB Video
+
Per-frame Geometry
+
Focus / Blur Control
        ↓
各方法
        ↓
Bokeh Video
~~~

### 回答的问题

> 当 Geometry 条件已经充分提供时，谁的 Bokeh Rendering / Video Modeling 本身更强？

### 重点比较

- Renderer；
- Neural Prior；
- Occlusion Handling；
- Boundary Quality；
- Temporal Modeling；
- Video Quality。

---

## Track B：Low-FPS Coarse Geometry

### 输入

\[
High\text{-}FPS\ RGB
+
Low\text{-}FPS\ Coarse\ Depth
\]

对于原本必须使用每帧 Geometry 的 baseline：

~~~text
Low-FPS Coarse Depth
        ↓
统一 Depth Interpolation / Propagation
        ↓
Per-frame Geometry
        ↓
Baseline
~~~

Ours：

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

### 回答的问题

> 在 Low-FPS、粗糙、可能存在误差的 Geometry 条件下，谁更稳定、更准确？

这条 Track 直接对应本项目的核心创新。

---

# 11. 哪些 Benchmark 做双 Track

不需要把所有 Benchmark 机械做两遍。

| Benchmark | Track A | Track B | 说明 |
|---|:---:|:---:|---|
| Any-to-Bokeh Synthetic | ✅ | ✅ | 最适合做双 Track |
| VBS | ✅ | ✅ | 可结合 Static / Dynamic Focus |
| DAVIS | ❌ 不做严格 Track A | ✅ 真实条件泛化为主 | 无 paired Bokeh GT / Full Geometry GT |
| OPPO-style | 可作为 Upper-bound | ✅ 核心 | 本身就是 Track B 专项 |

---

# 12. 最终结果表模板

## Table 1：Any-to-Bokeh Synthetic

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | FD ↓ | RM ↓ |
|---|---:|---:|---:|---:|---:|---:|
| DeepLens |  |  |  |  |  |  |
| Dr.Bokeh |  |  |  |  |  |  |
| BokehMe |  |  |  |  |  |  |
| MPIB |  |  |  |  |  |  |
| BokehDiff |  |  |  |  |  |  |
| Any-to-Bokeh |  |  |  |  |  |  |
| BokehCrafter* |  |  |  |  |  |  |
| Ours |  |  |  |  |  |  |

建议分别报告 Track A 与 Track B。

---

## Table 2：VBS

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | FD ↓ | RM ↓ | Temporal Consistency ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepLens |  |  |  |  |  |  |  |
| Dr.Bokeh |  |  |  |  |  |  |  |
| BokehMe |  |  |  |  |  |  |  |
| MPIB |  |  |  |  |  |  |  |
| BokehDiff |  |  |  |  |  |  |  |
| Any-to-Bokeh |  |  |  |  |  |  |  |
| BokehCrafter |  |  |  |  |  |  |  |
| Ours |  |  |  |  |  |  |  |

建议分别报告 Track A 与 Track B。

---

## Table 3：VBS Focus-type Analysis

Dynamic Focus 是 VBS 子集专项，不是独立 Benchmark。

优先比较真正具备 Video Temporal Modeling 的方法：

| Method | Focus Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD ↓ | RM ↓ | Temporal Consistency ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Any-to-Bokeh | Static |  |  |  |  |  |  |
| Any-to-Bokeh | Dynamic |  |  |  |  |  |  |
| BokehCrafter | Static |  |  |  |  |  |  |
| BokehCrafter | Dynamic |  |  |  |  |  |  |
| Ours | Static |  |  |  |  |  |  |
| Ours | Dynamic |  |  |  |  |  |  |

---

## Table 4：DAVIS Real-world

| Method | VEPI ↑ | Hair / Thin Structure | Halo / Leakage | Temporal Flicker | Qualitative Video |
|---|---:|---|---|---|---|
| DeepLens |  |  |  |  |  |
| Dr.Bokeh |  |  |  |  |  |
| BokehMe |  |  |  |  |  |
| MPIB |  |  |  |  |  |
| BokehDiff |  |  |  |  |  |
| Any-to-Bokeh |  |  |  |  |  |
| BokehCrafter* |  |  |  |  |  |
| Ours |  |  |  |  |  |

---

## Table 5：Efficiency

| Method | Resolution | Frames | Core Time ↓ | E2E Time ↓ | FPS ↑ | Peak VRAM ↓ | Params ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepLens | 1024×576 | 25 |  |  |  |  |  |
| Dr.Bokeh | 1024×576 | 25 |  |  |  |  |  |
| BokehMe | 1024×576 | 25 |  |  |  |  |  |
| MPIB | 1024×576 | 25 |  |  |  |  |  |
| BokehDiff | 1024×576 | 25 |  |  |  |  |  |
| Any-to-Bokeh | 1024×576 | 25 |  |  |  |  |  |
| BokehCrafter* | 1024×576 | 25 |  |  |  |  |  |
| Ours | 1024×576 | 25 |  |  |  |  |  |

---

## Table 6：OPPO-style Low-FPS Depth Robustness

| Depth FPS | Depth Quality | PSNR ↑ | LPIPS ↓ | FVD ↓ | FD ↓ | RM ↓ | Depth AbsRel ↓ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30 | Clean |  |  |  |  |  |  |
| 15 | Clean |  |  |  |  |  |  |
| 10 | Clean |  |  |  |  |  |  |
| 5 | Clean |  |  |  |  |  |  |
| 2 | Clean |  |  |  |  |  |  |

后续可进一步增加 Mild / Medium / Severe。

---

# 13. 效率测试统一规则

所有方法尽量在同一条件下重新测试：

- 同一 GPU；
- 同一 CUDA / PyTorch 环境；
- 同一分辨率；
- 同一视频长度；
- batch size = 1；
- warm-up；
- 至少重复 5 次；
- 报平均值；
- 记录 Peak VRAM。

区分：

## Core Time

只统计核心模型 / Renderer。

## E2E Time

包含：

~~~text
RGB
↓
Depth / Geometry Estimation
↓
Mask / Target Extraction
↓
Geometry Preprocessing
↓
Bokeh Model / Renderer
↓
Output
~~~

---

# 14. Any-to-Bokeh 第一轮建议实验

当前已经开始复现 Any-to-Bokeh，建议第一轮至少完成：

## A. Official Reproduction

目标：

> 官方 checkpoint + 官方输入 + 官方 pipeline 完整跑通。

保存：

- predicted video；
- GT；
- comparison video；
- metrics；
- runtime；
- peak VRAM；
- config。

## B. Geometry Noise Robustness

设置：

- Clean；
- Mild；
- Medium；
- Severe。

候选 corruption：

- blur / smooth perturbation；
- elastic perturbation；
- dilation / erosion；
- boundary corruption；
- missing geometry。

记录：

- PSNR；
- LPIPS；
- FVD；
- FD；
- RM。

## C. Dynamic Focus

控制：

\[
FocusDistance(t)
\]

观察：

- Focus transition；
- FD；
- RM；
- Temporal Flicker。

## D. Dynamic Blur Strength

控制：

\[
K(t)
\]

观察：

- blur change 是否连续；
- foreground 是否稳定；
- temporal artifact。

---

# 15. 输出目录建议

每个算法 / 实验统一保存：

~~~text
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
│
├── efficiency.json
│
└── config.yaml
~~~

至少记录：

## metrics.json

- PSNR；
- SSIM；
- LPIPS；
- FVD；
- FD；
- RM；
- Temporal Consistency（VBS 时）；
- VEPI（DAVIS 时）。

## efficiency.json

- core_time；
- e2e_time；
- fps；
- peak_vram；
- params。

## config.yaml

- checkpoint；
- resolution；
- frames；
- focus；
- \(K\)；
- geometry source；
- preprocessing。

---

# 16. 当前已确定与待确认

## 已确定

- 核心 baseline：DeepLens、Dr.Bokeh、BokehMe、MPIB、BokehDiff、Any-to-Bokeh、BokehCrafter*、Ours；
- Any-to-Bokeh Synthetic + VBS 组成 Unified Paired Video Bokeh Benchmark Suite；
- DAVIS 负责真实泛化；
- OPPO-style 负责 Low-FPS Coarse Depth 专项；
- Any-to-Bokeh Synthetic 与 VBS 尽量统一 PSNR / SSIM / LPIPS / FVD / FD / RM；
- VBS 额外保留 Temporal Consistency；
- Dynamic Focus 属于 VBS 专项子集；
- Track A / Track B 两条评价线同时保留；
- DAVIS 不机械做双 Track；
- 效率必须区分 Core Time 与 E2E Time。

## 仍需继续确认

### 1. Baseline 输入适配细节

需要继续核对各官方实现：

- DeepLens；
- Dr.Bokeh；
- BokehMe；
- MPIB；
- BokehDiff；
- Any-to-Bokeh；
- BokehCrafter。

尤其要确认：

> 在 Any-to-Bokeh 原实验中，各 baseline 实际使用的 Geometry / Focus / Alpha 输入是如何适配的。

在这部分核实前，不把最终统一输入 adapter 写死。

### 2. BokehCrafter 可复现性

若无法获得稳定官方实现 / checkpoint，则不进入正式 quantitative 主表。

### 3. OPPO-style Benchmark 的真实 Depth 退化模型

最终需要根据真实 OPPO Stereo Depth 的：

- FPS；
- hole；
- confidence；
- boundary error；
- scale / noise；

重新确定 synthetic degradation，而不是只使用任意人工噪声。

---

# 17. 当前执行顺序

~~~text
1. Any-to-Bokeh
   ├── 官方 Demo / Test 跑通
   ├── 记录统一指标
   ├── 记录效率
   └── 阅读 baseline evaluation implementation

2. 核对各 baseline 的真实输入接口
   ├── BokehMe
   ├── MPIB
   ├── Dr.Bokeh
   ├── DeepLens
   └── BokehDiff

3. 最终确定 Track A 的统一输入 adapter

4. 再确定 Track B：
   Low-FPS Coarse Depth → baseline 的公平适配方式

5. 开始批量复现其他 baseline
~~~

当前阶段最重要的不是继续增加方法或指标，而是：

> **先把 Any-to-Bokeh 的官方评价链路和 baseline 输入适配彻底搞清楚，再冻结最终统一输入协议。**
