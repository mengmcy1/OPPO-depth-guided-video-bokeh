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


---

# 17. Any-to-Bokeh-style Synthetic Benchmark 的构建细节

这一节记录当前公开的 \`dataset_synthsis.py\` 所体现的数据生成逻辑，用于后续构建：

> **Reproduced Any-to-Bokeh-style Synthetic Benchmark**

这里的目标不是声称 100% 复刻论文官方 200-video test set，而是尽量复现其核心生成思想，并构建一套可固定、可复现、可供所有 baseline 公平比较的 paired video bokeh benchmark。

---

## 17.1 最终希望生成的数据

每个 synthetic scene 最终应至少包含：

~~~text
scene_xxxx/
├── aif/
│   └── 全清晰 RGB 视频帧
│
├── disparity/
│   └── 每帧视差
│
├── mask/
│   └── 每个前景对象的二值遮罩
│
├── alpha/
│   └── 每个前景对象的软透明度图（建议额外保存）
│
├── bokeh_gt/
│   └── 对应的 Bokeh 真值视频
│
└── metadata.json
    └── focus / K / depth / source / seed 等元信息
~~~

其中：

- **全清晰图（All-in-Focus，AiF）**：前景和背景都保持清晰；
- **视差（Disparity）**：表示场景相对远近的一种几何量，通常近处视差更大；
- **Alpha Matte**：0～1 的软透明度图，用于描述发丝、软边界等；
- **Mask**：0/1 二值遮罩；
- **Bokeh GT**：由渲染器生成的虚化真值。

最终核心 paired data 为：

\[
RGB + Geometry + Focus + K + Bokeh\ GT
\]

这套数据既可以用于 Track A，也可以作为 Track B 的基础。

---

## 17.2 前景数据来源

当前公开脚本中可以确认至少使用了两类前景数据。

### 1. VideoMatte240K

VideoMatte240K 是视频抠图数据。

它提供：

- 视频前景；
- 对应透明度 / Alpha；
- 连续时间序列。

公开脚本中能看到类似：

~~~text
VideoMatte240K_JPEG_HD/train/fgr/...
VideoMatte240K_JPEG_HD/train/pha/...
~~~

其中：

- \`fgr\`：foreground，前景 RGB；
- \`pha\`：pha / alpha，透明度图。

这类数据特别适合：

- 人物；
- 发丝；
- 衣服边缘；
- 连续运动前景。

---

### 2. AM-2k

脚本中也明确出现：

~~~text
AM-2k/train/fg/...
AM-2k/train/mask/...
~~~

AM-2k 属于图像抠图数据。

和 VideoMatte240K 的区别可以简单理解为：

~~~text
VideoMatte240K
→ 本身就是视频前景

AM-2k
→ 单张前景图片
~~~

对于单张图像素材，可以通过重复 25 帧并人为施加位置变化，转成 synthetic video foreground。

---

## 17.3 前景统一成 25 帧序列

作者并不是直接在生成脚本里扫描整个数据集，而是通过：

\`merge_f25_test.json\`

统一组织前景。

其逻辑可以理解为：

~~~text
Object 0
├── 25 张 foreground frames
└── 25 张对应 alpha / mask

Object 1
├── 25 张 foreground frames
└── 25 张对应 alpha / mask
~~~

JSON 中核心字段是：

~~~text
clip_paths
clip_mask_paths
~~~

即：

- 前景帧路径；
- 对应遮罩 / Alpha 路径。

因此后续我们自己构建时，也建议先把所有前景素材统一转换成：

> **每个对象固定 25 帧的标准序列。**

---

## 17.4 背景数据

当前脚本引用了作者服务器中的：

~~~text
BG_test
Backgrounds
BG_imgs
~~~

这些实际背景素材没有随仓库完整公开。

因此后续自建 Benchmark 时需要替换为公开、可重复获取的背景图像来源。

建议背景覆盖：

- 室内；
- 街道；
- 建筑；
- 公园；
- 自然景观；
- 办公环境；
- 复杂纹理背景。

背景数据必须固定版本与 split，避免测试集随时间变化。

---

## 17.5 当前脚本的基本生成参数

公开脚本当前默认值包括：

~~~text
synthsis_images_num = 300
obj_num = 4
resolution = 1024 × 576
frame_num = 25
~~~

也就是：

- 每个视频 25 帧；
- 分辨率 1024×576；
- 每个场景最多组合 4 个前景对象；
- 当前脚本默认生成 300 个 scene。

需要注意：

> 论文描述的 synthetic test set 为 200 videos，而当前公开脚本默认值为 300。

因此可以进一步确认：

> **公开 \`dataset_synthsis.py\` 是数据生成代码版本，不等同于论文官方 200-video test set 的完整一键复现包。**

---

## 17.6 前景如何放进画面

对于 \`obj_num = 4\`，脚本先把画布划分成若干区域，为每个对象分配一个初始位置。

之后再加入随机位置扰动，避免每个 scene 都呈现规则网格布局。

可以简单理解为：

~~~text
1024 × 576 Canvas
│
├── Object A
├── Object B
├── Object C
└── Object D
~~~

但每个对象会有独立的随机位置。

---

## 17.7 随机缩放与旋转

每个前景对象会做数据增强。

### 缩放

当前代码大致使用：

\[
0.5 \sim 0.8
\]

的随机比例。

### 旋转

随机范围大致为：

\[
-10^\circ \sim +10^\circ
\]

目的：

- 增加场景变化；
- 防止相同素材每次出现形式完全一致。

---

## 17.8 视频运动是怎么来的

脚本会对每帧的前景中心位置增加随机扰动，例如：

~~~text
x / y 方向约 -40 ～ +40 pixel
~~~

因此前景会随时间产生位置变化。

对于 VideoMatte240K：

- 原前景人物本身已经有动作；
- 再叠加整体位置移动。

对于 AM-2k：

- 原图本身静止；
- 主要依赖 synthetic position change 形成视频运动。

因此 synthetic video 中同时可能包含：

- 人体内部动作；
- 整体对象移动；
- 多对象相对运动。

---

## 17.9 前景的深度 / 前后关系

脚本会人为给每个前景对象和背景赋予不同的几何位置。

当前参数类似：

~~~text
foreground dof = (1, 5)
background dof = (100, 200)
~~~

可以简单理解为：

> 前景对象分布在较近的不同深度层；背景放在远处。

多个 foreground 的深度顺序还会随机打乱，因此可以生成不同的前后遮挡关系。

---

## 17.10 视差如何生成

代码中场景几何使用类似：

\[
d(x,y)=\frac{1-ax-by}{c}
\]

的形式。

当前生成设置中：

\[
a=0,\quad b=0
\]

因此近似为：

\[
\boxed{d=\frac{1}{c}}
\]

也就是：

- \(c\) 小 → 更近；
- disparity 大；
- \(c\) 大 → 更远；
- disparity 小。

例如：

\[
c=1 \Rightarrow d=1
\]

\[
c=5 \Rightarrow d=0.2
\]

背景由于 \`c\` 更大，因此 disparity 很小。

每一帧都会生成一张对应的 disparity map。

---

## 17.11 Alpha 与 Mask 的作用

### Alpha

Alpha 是 0～1 的软透明度。

例如发丝边缘：

\[
\alpha = 0.4
\]

可以表示该像素同时包含前景与背景。

合成公式：

\[
I=\alpha F+(1-\alpha)B
\]

因此 Alpha 对：

- Hair；
- Thin Structure；
- Soft Boundary；

尤其重要。

### Mask

脚本还会通过类似：

\[
\alpha > 0.5
\]

得到二值 Mask。

Mask 主要用于：

- 标出物体区域；
- 计算对象几何；
- 确定 focus target；
- 后续评价。

---

## 17.12 AiF 视频如何生成

**AiF（All-in-Focus，全清晰图）**：

> 所有深度层都保持清晰。

脚本根据对象的前后关系，把：

~~~text
Background
↓
Far Foreground
↓
...
↓
Near Foreground
~~~

利用 Alpha 逐层合成。

最终每帧保存为：

~~~text
aif/0.jpg
aif/1.jpg
...
aif/24.jpg
~~~

这可以作为 Bokeh 模型的 RGB 输入。

---

## 17.13 Bokeh GT 如何生成

作者使用：

\`ModuleRenderRT\`

完成虚化真值渲染。

可以把它理解成：

> **光线追踪式 Bokeh 渲染模块（Ray-Tracing Bokeh Renderer）**

输入主要包括：

~~~text
RGB Layers
+
Alpha Layers
+
Geometry
+
Focus
+
Blur Strength K
~~~

输出：

~~~text
Bokeh GT Video
~~~

这一步把前面构造的 synthetic scene 转成 paired Bokeh ground truth。

---

## 17.14 K：虚化强度

参数 \(K\) 用于控制虚化强度。

简单理解：

\[
K\uparrow
\Rightarrow
Blur\uparrow
\]

当前 test generation 分支会从：

\[
\{4,\ 8,\ 16,\ 32\}
\]

中选择不同的 Bokeh strength。

因此同一类场景可以覆盖：

- 轻微虚化；
- 中等虚化；
- 强虚化。

---

## 17.15 Focus：焦点如何设置

脚本会计算每个 foreground object 对应的焦平面值。

可以理解为：

> 当前希望保持清晰的对象，对应一个 focus disparity。

同一个 AiF scene 可以针对不同 object 分别生成 Bokeh：

~~~text
同一个 AiF Video
│
├── Focus Object 0 → Bokeh GT 0
├── Focus Object 1 → Bokeh GT 1
├── Focus Object 2 → Bokeh GT 2
└── Focus Object 3 → Bokeh GT 3
~~~

如果目标对象随时间发生深度变化，focus 也可以随对象对应 geometry 变化。

这对于：

- moving subject；
- dynamic focus；
- temporal evaluation；

都很有价值。

---

## 17.16 整个生成流程

完整逻辑可以概括为：

~~~text
VideoMatte240K / AM-2k
        ↓
Foreground + Alpha
        │
        ├──────────────┐
        │              │
公开 Background Images │
        │              │
        └──────┬───────┘
               ↓
随机选择多个前景
               ↓
随机：
位置
缩放
旋转
运动
深度
前后关系
               ↓
合成 25 帧 AiF Video
               ↓
生成每帧 Disparity
               ↓
保存 Alpha / Mask
               ↓
设置 Focus Object
+
Blur Strength K
               ↓
Bokeh Renderer
               ↓
Bokeh GT Video
~~~

最终形成：

\[
\boxed{
RGB + Disparity + Alpha/Mask + Focus + K + Bokeh\ GT
}
\]

---

## 17.17 当前公开脚本的三个主要缺口

### 1. \`ModuleRenderRT\` 缺失

公开脚本引用：

~~~python
from z_bokeh.data_process.bokeh_kernel import ModuleRenderRT
~~~

但当前公开仓库中没有找到对应完整实现。

因此：

> **作者原 Bokeh Renderer 目前不是直接可运行状态。**

这是自建 benchmark 当前最大的技术缺口。

---

### 2. 作者背景数据缺失

如：

~~~text
BG_test
Backgrounds
BG_imgs
~~~

均为作者内部服务器路径。

需要替换成我们自己选定的公开背景数据集。

---

### 3. \`merge_f25_test.json\` 未提供

公开代码给出了 JSON 格式示例，但没有提供作者实际使用的完整 test JSON。

因此我们需要自己：

~~~text
VideoMatte240K / AM-2k
↓
统一成 25-frame clips
↓
生成自己的 merge_f25_test.json
~~~

---

## 17.18 我们自己的版本建议额外保存什么

为了同时服务后续 Track A / Track B，建议比原脚本多保存：

~~~text
scene_xxxx/
├── rgb/
├── depth_gt/
├── disparity_gt/
├── alpha/
├── mask/
├── bokeh_gt/
└── metadata.json
~~~

其中 \`metadata.json\` 至少记录：

- random seed；
- foreground source；
- background source；
- object depth；
- object order；
- focus object；
- per-frame focus；
- \(K\)；
- resolution；
- number of frames；
- renderer version。

这样后续可以直接从 Full-FPS Geometry 构造：

~~~text
Track A
→ Full per-frame Geometry

Track B
→ Downsample / Corrupt Geometry
   30 → 15 → 10 → 5 → 2 fps
   + Noise / Hole / Boundary Error
~~~

而无需重新生成场景。

---

## 17.19 当前构建原则

当前建议遵循：

> **复现 Any-to-Bokeh 的生成逻辑，而不是强求复刻作者完全相同的 200 个测试样本。**

优先保证：

- 公开数据源；
- 固定 split；
- 固定随机种子；
- 固定生成脚本；
- 固定 renderer；
- 固定 metadata；
- 所有 baseline 使用同一测试数据。

这样才能把它真正作为一个：

> **可复现、可审计、可公平比较的 paired video bokeh benchmark。**
