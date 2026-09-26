# Benchmark、Baseline 与统一评价协议

> 当前版本：VBS 移除后的统一实验协议  
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

实验需要回答四类问题：

1. 最终 Bokeh 画质是否好？
2. 视频时序是否稳定且保留真实变化？
3. 低帧率、粗糙、带误差的 Depth 是否会导致明显性能下降？
4. 模型的效率是否适合后续工程化？

---

# 2. 当前 Benchmark 总体结构

VBS 暂时从正式实验方案中移除，原因是当前尚未找到可靠、可核实的公开下载入口，不让不可获取的数据集阻塞实验主线。

当前保留三层：

~~~text
Benchmark Suite
│
├── 1. Paired Synthetic Video Bokeh Benchmark
│      └── Any-to-Bokeh Synthetic
│          ├── Official Test Set（若后续可获得）
│          └── Reproduced Any-to-Bokeh-style Set（Plan B）
│
├── 2. DAVIS
│      └── Real-world Generalization
│
└── 3. OPPO-style Low-FPS Coarse Depth Benchmark
       └── Project-specific Benchmark
~~~

说明：

- **Any-to-Bokeh Synthetic**：负责有 GT 的主定量评价；
- **DAVIS**：负责真实视频泛化；
- **OPPO-style**：负责 Low-FPS / Coarse / Noisy Depth 条件下的专项评价；
- **VBS**：仅保留为未来可选补充项，若后续官方公开且容易获取，再考虑加入，不进入当前执行主线。

---

# 3. Benchmark 1：Any-to-Bokeh Synthetic

## 3.1 定位

这是当前最重要的 paired video Bokeh 主定量 Benchmark。

参考 Any-to-Bokeh 论文测试协议：

- synthetic paired video；
- 有 Bokeh GT；
- 有 Geometry / disparity；
- 有 focus plane；
- 有 blur strength；
- 视频级测试；
- 可进行完整的全参考定量评价。

论文参考规模：

- 200 videos；
- 25 frames / video；
- 1024 × 576。

当前官方完整测试包尚未确认可获得，因此设置两种情况：

### 情况 A：官方 Test Set 后续可获得

直接按照作者测试集和评价脚本进行正式复现。

### 情况 B：官方 Test Set 仍无法获得

按照官方论文和公开的 \`dataset_synthsis.py\` 生成逻辑构建固定的：

> **Reproduced Any-to-Bokeh-style Synthetic Benchmark**

必须明确标注为自建重现版本，不能声称与作者原 200-video test set 完全一致。

需要固定：

- 前景 / 背景数据来源；
- 随机种子；
- 视频数量；
- 视频长度；
- 分辨率；
- object 数量；
- geometry / disparity 生成方式；
- focus 设置；
- blur strength；
- renderer；
- train / test 隔离规则。

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

Any-to-Bokeh 原论文还使用 VFID-I。完整复现原论文时可以继续计算，但当前主表优先保留以上六个指标。

---

# 4. Benchmark 2：DAVIS

## 4.1 定位

DAVIS 负责 Real-world Generalization。

它是真实自然视频，但没有严格 paired 的真实 Bokeh GT，因此不能像 synthetic benchmark 一样直接使用 against-GT 的：

- PSNR；
- SSIM；
- LPIPS。

Any-to-Bokeh 官方仓库已经提供 DAVIS demo 和 VEPI 评价代码，因此 DAVIS 是当前最容易直接落地的真实视频 Benchmark。

---

## 4.2 主要指标

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

## 4.3 固定定性观察项

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

# 5. Benchmark 3：OPPO-style Low-FPS Coarse Depth

## 5.1 定位

这是后续最能体现本项目核心创新的专项 Benchmark。

输入模拟：

\[
High\text{-}FPS\ RGB
+
Low\text{-}FPS\ Coarse\ Depth
\]

如果基于有完整 GT 的 synthetic paired dataset 构造，可以同时拥有：

- Full-FPS Bokeh GT；
- Full-FPS Depth GT；
- 可选 Alpha GT。

Any-to-Bokeh-style Synthetic 数据可以作为构造该 Benchmark 的重要基础。

---

## 5.2 Final Bokeh 指标

尽量与 Any-to-Bokeh Synthetic 保持一致：

\[
\boxed{
PSNR,\ SSIM,\ LPIPS,\ FVD,\ FD,\ RM
}
\]

---

## 5.3 Depth 指标

如果有 Full-FPS Depth GT：

| Metric | 趋势 | 主要含义 |
|---|---:|---|
| AbsRel | ↓ | 相对深度误差 |
| RMSE | ↓ | 均方根深度误差 |
| \(\delta_1\) | ↑ | 阈值内的深度像素比例 |
| Depth Edge Error | ↓ | 深度突变边缘恢复 |

---

## 5.4 Alpha 指标

如果有 Alpha GT：

| Metric | 趋势 | 主要含义 |
|---|---:|---|
| SAD | ↓ | Alpha absolute difference |
| MSE | ↓ | Alpha 均方误差 |
| Gradient Error | ↓ | 发丝 / 软边缘恢复 |
| Connectivity Error | ↓ | 前景连通与拓扑结构 |

Depth 和 Alpha 指标单独建表，不与最终 Bokeh 主表混在一起。

---

## 5.5 Low-FPS / Noise 专项变量

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

最终退化模型应尽量依据真实 OPPO Stereo Depth 的统计特征，而不是只使用任意人工噪声。

---

# 6. 当前核心 Baseline

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

\* BokehCrafter 作为方法 baseline 可以保留；是否进入正式 quantitative 主表，取决于能否获得可运行实现 / checkpoint，并在当前 Benchmark 上公平运行。它不再依赖 VBS 才能保留。

---

# 7. 为什么选这些 Baseline

## DeepLens

代表较早期学习式单帧 Bokeh，主要观察单帧质量与逐帧处理后的视频稳定性。

## Dr.Bokeh

代表强物理 RGBD Renderer 和显式遮挡处理，重点比较 Depth Boundary、Partial Occlusion、Color Bleeding 和物理可控性。

## BokehMe

代表：

\[
Physical + Neural
\]

与本项目：

\[
Physical\ Renderer + Neural\ Residual
\]

的整体思想较接近。

## MPIB

代表 Explicit Layered Geometry / MPI，适合比较 layered geometry、occlusion 和前后景关系。

## BokehDiff

代表现代 One-step Diffusion / Generative Bokeh，重点观察感知质量、复杂边缘与生成式先验的收益和成本。

## Any-to-Bokeh

当前最重要的直接 baseline，代表：

\[
Video + Explicit\ Geometry + Temporal\ Modeling
\]

重点比较：

- Temporal Consistency；
- Geometry Condition；
- Dynamic Focus；
- Geometry Noise Robustness；
- 与 Low-FPS Geometry 输入条件的差异。

当前 Any-to-Bokeh 已在 96 GB GPU 上完整跑通。

## BokehCrafter

代表 Video Diffusion / Temporal Prior。若后续能获得稳定可运行实现，可作为额外 video baseline；否则不阻塞主实验。

---

# 8. 单图方法 vs 原生视频方法

当前核心 baseline 可以分为：

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

因此将两类方法同时放进 Video Benchmark，可以直接观察 Temporal Modeling 的作用。

---

# 9. 统一输入协议：Track A / Track B

为了把 Geometry 输入质量差异与 Bokeh / Video 模型能力差异尽可能分开，保留双 Track。

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

重点比较：

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

# 10. 哪些 Benchmark 做双 Track

| Benchmark | Track A | Track B | 说明 |
|---|:---:|:---:|---|
| Any-to-Bokeh Synthetic / Reproduced Set | ✅ | ✅ | 最适合做双 Track |
| DAVIS | ❌ 不做严格 Track A | ✅ 真实条件泛化为主 | 无 paired Bokeh GT / Full Geometry GT |
| OPPO-style | 可作为 Upper-bound | ✅ 核心 | 本身就是 Track B 专项 |

Track A / Track B 不意味着所有 Benchmark 都机械重复两遍。

---

# 11. 最终结果表模板

## Table 1：Any-to-Bokeh Synthetic / Reproduced Synthetic

建议分别报告 Track A 与 Track B。

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

若使用自建版本，表题必须明确写：

> Reproduced Any-to-Bokeh-style Synthetic Benchmark

不能与原论文官方 200-video test set 的 paper-reported numbers 直接混为同一测试集结果。

---

## Table 2：DAVIS Real-world

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

## Table 3：Efficiency

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

## Table 4：OPPO-style Low-FPS Depth Robustness

| Depth FPS | Depth Quality | PSNR ↑ | LPIPS ↓ | FVD ↓ | FD ↓ | RM ↓ | Depth AbsRel ↓ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30 | Clean |  |  |  |  |  |  |
| 15 | Clean |  |  |  |  |  |  |
| 10 | Clean |  |  |  |  |  |  |
| 5 | Clean |  |  |  |  |  |  |
| 2 | Clean |  |  |  |  |  |  |

后续可进一步增加 Mild / Medium / Severe。

---

# 12. 效率测试统一规则

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

# 13. Any-to-Bokeh 当前实验状态与下一步

## A. Official Reproduction

当前状态：

> Any-to-Bokeh 已经在 96 GB GPU 上完整跑通。

接下来保存并统一记录：

- predicted video；
- comparison video；
- runtime；
- peak VRAM；
- config；
- 后续 Benchmark 对应 metrics。

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

# 14. 输出目录建议

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
├── efficiency.json
└── config.yaml
~~~

## metrics.json

根据 Benchmark 保存：

- PSNR；
- SSIM；
- LPIPS；
- FVD；
- FD；
- RM；
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

# 15. 当前已确定与待确认

## 已确定

- VBS 暂时移出正式实验方案；
- 核心 Benchmark：Any-to-Bokeh Synthetic / Reproduced Synthetic、DAVIS、OPPO-style；
- 核心 baseline：DeepLens、Dr.Bokeh、BokehMe、MPIB、BokehDiff、Any-to-Bokeh、BokehCrafter*、Ours；
- Any-to-Bokeh 已在 96 GB GPU 上完整跑通；
- Any-to-Bokeh Synthetic 保留 Official / Reproduced 两种路径；
- DAVIS 负责真实泛化；
- OPPO-style 负责 Low-FPS Coarse Depth 专项；
- paired synthetic benchmark 统一 PSNR / SSIM / LPIPS / FVD / FD / RM；
- Track A / Track B 两条评价线同时保留；
- DAVIS 不机械做双 Track；
- 效率必须区分 Core Time 与 E2E Time。

## 仍需继续确认

### 1. Any-to-Bokeh Synthetic 的最终来源

优先级：

1. 继续确认官方 200-video test set 是否能获取；
2. 若不可获取，基于公开 \`dataset_synthsis.py\` 和论文协议构建可复现测试集。

需要核对公开脚本中尚缺失或依赖作者本地路径的：

- ModuleRenderRT；
- 前景 JSON；
- Background 集；
- AM-2k / VideoMatte240K 具体组织；
- test split；
- renderer 依赖。

### 2. Baseline 输入适配细节

需要继续核对：

- DeepLens；
- Dr.Bokeh；
- BokehMe；
- MPIB；
- BokehDiff；
- Any-to-Bokeh；
- BokehCrafter。

尤其需要确认：

> 各方法如何在 Track A 下公平使用 Geometry / Focus / Alpha；在 Track B 下如何公平接入 Low-FPS Coarse Depth。

### 3. BokehCrafter 可复现性

若无法获得稳定官方实现 / checkpoint，则不进入正式 quantitative 主表。

### 4. OPPO-style Benchmark 的真实 Depth 退化模型

最终需要根据真实 OPPO Stereo Depth 的：

- FPS；
- hole；
- confidence；
- boundary error；
- scale / noise；

重新确定 synthetic degradation。

---

# 16. 当前执行顺序

~~~text
1. Any-to-Bokeh
   ├── 96 GB 上完整运行：已完成
   ├── 记录效率
   ├── 跑 DAVIS / VEPI
   └── 核对 synthetic generation pipeline

2. Any-to-Bokeh Synthetic
   ├── 继续确认官方 test set
   └── 并行准备 Reproduced Any-to-Bokeh-style Set

3. 核对各 baseline 的真实输入接口
   ├── BokehMe
   ├── MPIB
   ├── Dr.Bokeh
   ├── DeepLens
   └── BokehDiff

4. 最终确定 Track A 的统一输入 adapter

5. 确定 Track B：
   Low-FPS Coarse Depth → baseline 的公平适配方式

6. 批量复现其他 baseline
~~~

当前阶段最重要的是：

> **先把 Any-to-Bokeh 的 synthetic generation / evaluation 链路和 baseline 输入适配搞清楚，再冻结最终统一输入协议。**
