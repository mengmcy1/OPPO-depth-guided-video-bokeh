# 五篇 Bokeh 文献横向汇总：对当前 Q1～Q11 的支持与空白

## 0. 汇总范围

本文件只汇总当前仓库中已经完成逐篇阅读的 5 篇文献，不引入新的论文：

1. **BokehFlow: Depth-Free Controllable Bokeh Rendering via Flow Matching**  
   仓库：`03257-AAAI26.HuangY-CV.pdf`

2. **BokehCrafter: Taming Video Diffusion Models for Controllable Bokeh Rendering**  
   仓库：`03980-AAAI26.WangQ-CV.pdf`

3. **Any-to-Bokeh: Arbitrary-Subject Video Refocusing with Video Diffusion Model**  
   仓库：`ANY-TO-BOKEH.pdf`

4. **MagicBokeh: Towards Photorealistic and Efficient Bokeh Rendering via Diffusion Framework**  
   仓库：`MagicBokeh-CVPR2026.pdf`

5. **Bokeh Diffusion: Defocus Blur Control in Text-to-Image Diffusion Models**  
   仓库：`Siggraph2025-Bokeh Diffusion- Defocus Blur Control in Text-to-Image Diffusion Models.pdf`

本汇总目的不是重新复述每篇论文，而是回答：

- 这 5 篇对我们现有 Q1～Q11 分别提供了什么证据；
- 哪些结论已经开始收敛；
- 哪些问题仍然没有被真正解决；
- 我们当前技术路线中哪些部分得到了文献支持，哪些仍然属于待验证假设。

---

# 1. 五篇论文各自的定位

## 1.1 BokehFlow

### 核心路线

```text
All-in-Focus RGB
+
Bokeh Control
        ↓
Flow Matching / Generative Model
        ↓
Bokeh
```

### 与我们最相关的点

- 即使最终不显式输入 Depth，**Geometry / Depth 相关预训练**仍然有帮助；
- 可以利用 Depth + Renderer 构造 Synthetic Bokeh GT；
- Bokeh 质量评价不能只看全图，**Edge / Boundary Region** 应单独评价；
- 生成式先验能处理部分复杂边缘问题，因此支持 Neural Rendering / Neural Residual 的价值。

### 它主要不能回答

- Low-FPS Depth → High-FPS Geometry；
- Depth / Alpha 多任务关系；
- Token Merging；
- 视频时序一致性。

---

## 1.2 BokehCrafter

### 核心路线

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

### 与我们最相关的点

- Video Backbone / Temporal Attention 本身就能提供时序先验；
- **Temporal Consistency 不等于 Temporal Constancy**；
- 正确的视频可以有真实的动态变化，尤其 Dynamic Focus；
- Video 级 Synthetic Bokeh 数据构造可行；
- Depth / Disparity Error 会明显污染最终 Bokeh；
- Blur 与 (1/z-1/z_f) 之类的 focus-relative geometry 密切相关。

### 它主要不能回答

- Low-FPS Depth 如何真正传播；
- Depth Edge 如何进入 Alpha；
- Token Merging；
- 多任务共享。

---

## 1.3 Any-to-Bokeh

### 核心路线

```text
RGB Video
+
Per-frame Geometry
+
Focus
        ↓
Focal-relative Geometry / MPI
        ↓
Geometry-guided Spatial Module
        ↓
Temporal Module
        ↓
One-step Video Generator
        ↓
Bokeh Video
```

### 与我们最相关的点

- 使用 **Focus-relative Geometry** 而不是只看绝对 Depth；
- Geometry Mask 可以直接参与 Attention，而不必作为普通图像通道；
- 主动对 Disparity 做 Corruption / Noise Augmentation；
- Temporal Module 不只负责“平滑”，还可以承担 **Temporal Geometry Denoising**；
- 细节恢复不能只依赖深层 Backbone，需要 Decoder / Skip Feature / Gradient 类约束；
- 边缘质量必须单独评价。

### 它主要不能回答

- 它假设每帧已有 Geometry，因此没有解决我们的 Low-FPS Depth → High-FPS Geometry；
- 没有 Alpha Head；
- 没有真正的 Depth / Alpha / Bokeh 三任务共享。

---

## 1.4 MagicBokeh

### 核心路线

```text
Low-quality RGB
        ↓
Geometry Estimation
        ↓
Focus-relative Defocus Map
        ↓
Focus-aware Attention
        ↓
One-step Diffusion
        ↓
High-quality Bokeh
```

同时联合考虑 Real-ISR 与 Bokeh Rendering。

### 与我们最相关的点

- Object Mask、Focus Mask、Depth、Alpha、Depth Validity Mask 必须区分；
- Mask 可以作为 **Attention Bias / Feature Gate**；
- 通过 Degradation-Aware Depth / Teacher-Student Distillation 提升 Geometry 鲁棒性；
- Multi-task Sharing 会真实地产生 Task Conflict；
- Alternating Training 是 PCGrad 之外的另一种冲突缓解思路；
- 不同任务标签不一定需要在同一个 sample 里同时存在；
- Depth / Disparity → Focus-relative Geometry → Defocus Map 的接口非常自然。

### 它主要不能回答

- 视频时序；
- Low-FPS Depth Propagation；
- Token Merging 后 Dense Recovery。

---

## 1.5 Bokeh Diffusion

### 核心路线

```text
Scene / Text
+
Independent Bokeh Parameter
        ↓
Diffusion Model
        ↓
Controlled Bokeh Image
```

### 与我们最相关的点

- 物理 Bokeh 控制与 Scene Content 应解耦；
- Synthetic Paired Bokeh 与 Real In-the-Wild Bokeh 可以承担不同训练职责；
- 经典 Depth-based Renderer 在 Depth Discontinuity、Hair、Thin Structure、Transparency 处会出现明显困难；
- 支持 **Physical Renderer + Neural Residual**；
- Neural Bokeh Module 更适合作为 Rendering Correction，而不是自由修改 Scene Content；
- 再次支持 Focus-relative disparity / CoC 作为更自然的 Bokeh 几何量。

### 它主要不能回答

- 视频时序；
- Low-FPS Depth Propagation；
- Token Merging；
- Depth → Alpha；
- 多任务共享。

---

# 2. Q1～Q11 × 五篇文献证据矩阵

符号说明：

- ★★★：直接且重要的支持；
- ★★：较强启发 / 部分直接证据；
- ★：间接启发；
- —：基本不能回答。

| 问题 | BokehFlow | BokehCrafter | Any-to-Bokeh | MagicBokeh | Bokeh Diffusion |
|---|---:|---:|---:|---:|---:|
| **Q1 Spatial Token Merging 后如何恢复 Dense Depth / Alpha？** | — | — | ★ | ★ | — |
| **Q2 Depth Edge 如何提供给 Alpha Head？** | — | — | — | ★ | — |
| **Q3 Low-FPS Depth 如何传播到 High-FPS RGB？** | — | — | ★ | — | — |
| **Q4 Alpha 中“前景”与 Focus 如何区分？** | ★ | ★★ | ★★ | ★★ | ★ |
| **Q5 Depth / Alpha / Bokeh 标签如何组织？** | ★★ | ★★ | ★ | ★★★ | ★★★ |
| **Q6 如何区分错误闪烁与真实变化？** | — | ★★★ | ★★★ | — | — |
| **Q7 Alpha 是否足够支撑最终边缘合成？** | ★ | ★ | ★ | ★★ | ★★★ |
| **Q8 Depth Head 输出什么表示？** | ★ | ★★ | ★★★ | ★★★ | ★★★ |
| **Q9 Backbone 是否适合 Dense / Fine-detail Prediction？** | ★★ | ★★ | ★★★ | ★★ | ★ |
| **Q10 Coarse Depth 错误时如何避免错误传播？** | ★ | ★★ | ★★★ | ★★★ | ★ |
| **Q11 三任务共享主干是否真的有收益？** | — | — | ★ | ★★★ | — |

最明显的结果是：

> **Q5、Q6、Q8、Q10 已经获得了较多直接或间接文献证据。**

而：

> **Q1、Q2、Q3 仍然明显缺乏直接证据。**

Q11 目前只有 MagicBokeh 对“共享任务存在冲突”给出了较直接的现实案例，但还没有真正回答我们的 Depth + Alpha + Bokeh 三任务共享是否有收益。

---

# 3. Q1：Spatial Token Merging 后如何恢复 Dense Depth / Alpha？

## 现有 5 篇能告诉我们的

没有任何一篇真正研究：

[
Nightarrow N'
]

之后再恢复 Dense Depth / Alpha。

所以目前不能声称该问题已有解法。

但 Any-to-Bokeh 与 MagicBokeh 共同提供一个重要旁证：

> **不同空间区域不应该被同等对待。**

它们都显式强调：

- Focus Boundary；
- Depth Discontinuity；
- Subject Edge；
- 高敏感区域；

需要更细的空间处理。

因此未来 Spatial Token Merging 至少值得研究：

```text
平坦背景 / 高冗余区域
→ 更积极 Merge

Depth Edge / Alpha Edge / Focus-sensitive Region
→ 少 Merge / 不 Merge
```

同时 Any-to-Bokeh 的 Decoder / Skip Feature 设计也说明：

> 即使主干做了压缩，最终 Dense Prediction 仍很可能需要高分辨率旁路或 Multi-scale Feature 来恢复细节。

## 当前状态

**仍是明显空白项。**

不能因为上述启发就提前决定具体 Token Merging 方案。

---

# 4. Q2：Depth Edge 如何提供给 Alpha Head？

五篇论文都没有真正做：

```text
Depth Head
→ Depth Edge
→ Alpha Head
```

因此这个问题仍未被直接解决。

MagicBokeh 只提供了一个结构启发：

```text
Geometry-derived Mask
→ Attention Bias / Guidance
→ Feature
```

所以后面 Q2 不应只考虑：

```text
Depth Edge
concat
Alpha Feature
```

也可以考虑：

- Boundary-aware Attention；
- Geometry-guided Attention；
- Boundary Gate；
- Auxiliary Edge Feature。

但目前没有证据说明哪一种最适合 Alpha。

## 当前状态

**基本空白，需要专项查 Video Matting / Depth-guided Matting / Cross-task Dense Prediction。**

---

# 5. Q3：Low-FPS Depth 如何传播到 High-FPS RGB？

这是当前 5 篇文献中最明显的空白之一。

Any-to-Bokeh 虽然有 Temporal Module，但其基本前提是：

> 每一帧已经有 Geometry / Disparity。

我们的任务却是：

```text
RGB：
1 2 3 4 5 6 7 8

Depth：
1     4     7
```

因此真正的问题是：

[
Low	ext{-}FPS Real Coarse Depth
ightarrow
High	ext{-}FPS Geometry
]

现有 5 篇没有回答：

- 是否先做 Flow Warp；
- 是否在 Feature Level Propagate；
- 是否做 Learned Interpolation；
- RGB 是否参与 correspondence；
- Confidence 如何参与传播；
- Occlusion / Disocclusion 怎么处理。

Any-to-Bokeh 只能支持：

> Temporal Module 可以在已有几何基础上做鲁棒性增强和 temporal denoising。

但不能把它当成 Q3 的直接解法。

## 当前状态

**核心空白项之一。**

---

# 6. Q4：Alpha 的图层含义与 Focus 选择

这一项已经出现比较一致的文献趋势。

BokehCrafter、Any-to-Bokeh、MagicBokeh 都从不同角度表明：

[
oxed{
Object / Layer

eq
Focus Selection
}
]

同一个人物 / 图层可以一直存在，但：

[
FocusDistance(t)
]

可以变化，从而导致该人物在某些时刻清晰、某些时刻离焦。

因此：

> **Alpha 不应该简单定义为“最终保持清晰的区域”。**

更合理的理解仍是：

[
Alpha
ightarrow
Layer / Coverage
]

而：

[
Depth + Focus + Camera Parameters
ightarrow
Sharpness / Blur
]

另外还必须区分：

[
Object Mask

eq
Alpha

eq
Focus Mask

eq
Depth Validity Mask
]

## 当前状态

**概念层面已经较明显收敛，但 Alpha 的最终任务定义仍需结合真实数据 / 下游 Renderer 决定。**

---

# 7. Q5：Depth / Alpha / Bokeh 标签和训练数据怎样组织？

这是 5 篇文献中收获最大的方向之一。

## 7.1 Synthetic Bokeh GT 是可行且常见的

BokehFlow、BokehCrafter、MagicBokeh、Bokeh Diffusion 都不同程度采用：

[
RGB + Geometry
ightarrow
Renderer
ightarrow
Synthetic Bokeh
]

因此：

> **Bokeh GT 不一定全部依赖真实多光圈采集。**

这个方向已有较多文献支持。

---

## 7.2 Synthetic 不等于 Real

Bokeh Diffusion 特别提醒：

Synthetic Pair 的优势是：

- 同场景严格配对；
- 几何与 Bokeh 的对应关系明确；
- 控制变量可以精确改变。

但 Real Data 才提供：

- 真实镜头分布；
- 真实 Highlight；
- 真实 Hair / Transparency；
- 真实 Lens Imperfection；
- 真实背景统计。

因此非常值得保留：

[
oxed{
Synthetic for Supervision
+
Real for Realism
}
]

---

## 7.3 三类标签不一定必须同时出现在一个样本里

MagicBokeh 的 Alternating Training 给了一个重要启发：

[
Multi	ext{-}task

eq
Every Sample Has All Labels
]

如果以后数据是：

```text
Dataset A:
RGB + Depth

Dataset B:
RGB + Alpha

Dataset C:
RGB + Bokeh
```

仍然可以研究：

```text
Depth Batch
→ Depth Objective

Alpha Batch
→ Alpha Objective

Bokeh Batch
→ Bokeh Objective
```

再共享部分网络。

## 当前状态

**已经形成较清晰的候选路线，但最终训练策略必须等真实 OPPO 数据和标签情况确定。**

---

# 8. Q6：如何区分错误闪烁和真实变化？

BokehCrafter 与 Any-to-Bokeh 对这一项帮助很大。

## 8.1 Temporal Consistency ≠ Temporal Constancy

BokehCrafter 给出的核心启发是：

[
oxed{
Temporal Consistency

eq
相邻帧越像越好
}
]

更合理的是：

[
oxed{
Predicted Change
approx
Real Change
}
]

例如：

- 人向镜头靠近；
- Depth 真正发生变化；
- Dynamic Focus；
- 遮挡出现 / 消失；

这些都属于真实变化，不应该被 Temporal Loss 误判成 flicker。

---

## 8.2 时序稳定不应只靠 Loss

BokehCrafter 与 Any-to-Bokeh 都强调 Video Backbone / Temporal Block 本身的重要性。

因此我们目前更合理的理解是：

```text
Temporal Feature Modeling
→ 第一层时序稳定

Temporal Loss
→ 额外约束
```

而不是把希望全部压到 Optical-Flow Warp Loss 上。

---

## 8.3 Temporal Module 还可以纠正 Geometry Noise

Any-to-Bokeh 说明：

Temporal Block 可以利用前后帧信息抵抗单帧 Disparity Error。

所以我们的 Temporal Module 未来可能同时承担：

1. High-FPS Geometry Reconstruction；
2. Temporal Geometry Denoising；
3. Depth / Alpha / Bokeh Temporal Stability。

## 当前状态

**原则已经明显收敛，但 Occlusion / Disocclusion / Scene Flow / Visibility-aware Loss 等具体问题仍未解决。**

---

# 9. Q7：Alpha 是否足够支撑最终边缘合成？

五篇文献没有直接证明：

[
RGB + Depth + Alpha
]

就一定足以产生高质量 Bokeh。

相反，多篇生成式 Bokeh 工作都绕过显式 Alpha，并直接学习复杂的 Bokeh Boundary。

Bokeh Diffusion 进一步明确指出传统 Depth-based Renderer 在：

- Depth Discontinuity；
- Hair；
- Thin Structure；
- Transparency；
- Foreground / Background Boundary；

处容易产生：

- Color Bleeding；
- Halo；
- Edge Artifact。

因此我们目前可以较有依据地保留：

[
oxed{
Physical Renderer
+
Neural Residual
}
]

其职责可以进一步明确：

```text
Depth
→ Geometry

Alpha
→ Layer / Coverage

Physical Renderer
→ 主体可解释 Defocus

Neural Residual
→ Hair / Transparency / Highlight /
  Halo / Color Bleeding / Lens Imperfection
```

但仍然没有回答：

- Foreground Color F 是否需要单独估计；
- Background Layer 是否需要显式恢复；
- 是否需要 Decontamination；
- Alpha 是否足够覆盖 Semi-transparent Boundary。

## 当前状态

**混合物理 + 神经路线得到较强支持，但 Renderer / Matting Interface 仍未解决。**

---

# 10. Q8：Depth Head 应输出什么 Geometry？

这是五篇中收敛最明显的问题之一。

连续多篇工作都使用类似：

[
r
=
K
left|
rac1z-rac1{z_f}
ight|
]

或：

[
r=K|d-d_f|
]

这说明真正自然连接到 Bokeh 的往往是：

[
oxed{
Geometry Relative to Focus
}
]

而不只是“绝对 Depth 数字”。

因此当前可以明确区分：

## 问题 A：Depth Head 的内部输出

可以是：

- Depth；
- Inverse Depth；
- Disparity；
- 其他 Geometry。

## 问题 B：Renderer 的输入

可以进一步转换成：

- Focus-relative Depth；
- Disparity Difference；
- Defocus Map；
- CoC Map。

所以：

[
Depth Head Output

eq
Renderer Input
]

是完全合理的。

同时仍必须单独回答：

> 几何输出是否具有绝对尺度 / 单位 / 标定？

## 当前状态

**表示接口已经开始收敛，但最终 Depth Head 输出形式仍需结合 OPPO 数据格式、标定和 GT 决定。**

---

# 11. Q9：Backbone 是否适合 Dense / Fine-detail Prediction？

5 篇文献给了几个比较一致的启发。

## 11.1 Pretraining Task 很重要

BokehFlow 表明：

> Geometry / Depth 相关预训练对 Bokeh 有帮助。

所以 Backbone 不能只按分类 / 视频理解性能选。

---

## 11.2 Temporal Modeling 本身很重要

BokehCrafter / Any-to-Bokeh 说明：

> Temporal Attention / Video Prior 可以直接改善视频稳定性。

因此 Backbone 调研必须记录：

- Temporal Attention；
- Factorized Spatial / Temporal Attention；
- 3D Attention；
- Recurrent Propagation；
- Sliding Window；
- 其他 temporal mechanism。

---

## 11.3 高级语义 Backbone 不代表自动保留 Hair / Boundary

Any-to-Bokeh 专门通过 Decoder Refinement、Skip Feature、Gradient / Texture Loss 修复细节。

因此未来 Q9 必须同时研究：

- Spatial Downsampling；
- Multi-scale Feature；
- High-resolution Feature；
- Skip Connection；
- Decoder；
- Boundary-aware Loss。

## 当前状态

**已经明确了“选 Backbone 时要看什么”，但还没有确定具体 Backbone。**

---

# 12. Q10：Coarse Depth 错误时怎么办？

这是目前文献支持最丰富的问题之一。

## 12.1 第一原则已经很明确

[
oxed{
Coarse Depth

eq
Ground Truth
}
]

BokehCrafter、Any-to-Bokeh、MagicBokeh 都说明 Geometry Error 会污染最终 Bokeh。

因此 Coarse Depth 更合理的定位是：

> Noisy Geometric Prior。

---

## 12.2 目前已经出现四类候选机制

### A. Reliability Encoding

我们的：

```text
Depth
+
Mask
+
Confidence
```

用于告诉网络：

- 哪些 Depth 有效；
- 哪些更可靠。

现有 Bokeh 文献没有直接给出 D/M/C 最终融合公式，但 Any-to-Bokeh 与 MagicBokeh 都说明：

> Mask 可以直接参与 Attention / Gating，而不必只是普通输入通道。

---

### B. Improve Geometry Quality

MagicBokeh：

```text
HQ Teacher
↓
Reliable Geometry / Feature
      ↑
Distillation
      ↑
LQ Student
```

说明可以主动提升 Geometry Estimator / Refiner 对恶劣输入的鲁棒性。

---

### C. Geometry Noise Augmentation

Any-to-Bokeh：

```text
Geometry
↓
Elastic / Noise / Morphological Corruption
↓
Noisy Geometry
↓
训练网络学会容忍 / 修正
```

说明可以主动让网络见到“错误 Depth”。

---

### D. Temporal Correction

Any-to-Bokeh：

```text
邻帧 Geometry
↓
Temporal Block
↓
纠正单帧异常
```

## 当前状态

**问题定位和候选策略已经明显丰富，但我们的 D/M/C 融合、Confidence Calibration 和 Coarse Depth Loss 仍未确定。**

---

# 13. Q11：三任务共享主干是否真的有收益？

5 篇里只有 MagicBokeh 给出了较直接的 Multi-task Conflict 案例。

它的：

[
Super	ext{-}Resolution
]

与：

[
Bokeh
]

共享网络时存在：

[
Sharpen
quad vs quad
Blur
]

的明显目标冲突。

因此：

[
oxed{
Shared Backbone

eq
Automatically Better
}
]

已经有较现实的支持。

同时它还说明：

> PCGrad 不是唯一选择。

还可以有：

- Alternating Training；
- Staged Training；
- Partial Sharing；
- Task-specific Decoder；
- Selective Freezing。

但是 MagicBokeh 并没有研究我们的：

[
Depth + Alpha + Bokeh
]

三任务。

因此现阶段不能据此判断：

- 应该完全共享；
- 应该完全独立；
- 应该共享 Encoder、分开 Decoder；
- 是否需要 Cross-task Connection。

## 当前状态

**“共享可能有冲突”已有支持，但“我们的三任务怎样共享最优”仍然是开放问题。**

---

# 14. 五篇论文形成的几个较稳定共识

## 共识 1：Geometry 对 Bokeh 很重要

BokehFlow 即使最终 Depth-free，Geometry Pretraining 仍有帮助。

Any-to-Bokeh、MagicBokeh 等则直接显式使用 Geometry。

因此：

[
oxed{
Geometry Prior Is Important
}
]

这一点已经很稳。

这支持我们继续利用 OPPO 提供的 Coarse Depth，而不是完全丢弃它。

---

## 共识 2：Geometry 很重要，但绝对不能盲信

多篇工作都说明错误 Depth / Disparity 会直接污染 Bokeh。

因此：

[
oxed{
Use Geometry
quad but quad
Do Not Trust Geometry Blindly
}
]

这也正是我们：

[
Depth + Mask + Confidence
]

设计的基本合理性。

---

## 共识 3：Bokeh 更自然依赖 Focus-relative Geometry

多篇论文共同出现：

[
Depth / Disparity
ightarrow
Relative to Focus
ightarrow
Defocus / CoC
]

所以 Renderer 接口应重点考虑：

- Focus Distance；
- Defocus Map；
- CoC；
- Disparity Difference。

---

## 共识 4：时序稳定首先是 Representation / Backbone 问题之一

视频论文都没有把全部希望放在后处理 Temporal Loss 上。

所以：

[
Video Transformer / Temporal Attention
]

本身应该承担真正的时序建模职责。

---

## 共识 5：Boundary Quality 必须单独处理和评价

多篇论文都专门：

- 设计 Edge / Boundary Metric；
- 处理 Focus Boundary；
- 修复 Hair / Thin Structure；
- 加 Gradient / Texture 约束。

因此我们的最终实验不能只报全图 PSNR / SSIM。

至少应该有：

```text
Full-image Quality
+
Boundary Quality
+
Temporal Quality
```

三个层面。

---

## 共识 6：Synthetic Bokeh Data 是现实可行路线

多篇论文都采用：

[
RGB+Geometry
ightarrow
Renderer
ightarrow
Synthetic Bokeh
]

所以 Synthetic Bokeh GT 本身并不是不合理。

真正的问题是：

> 如何补足 Synthetic-to-Real Gap。

Bokeh Diffusion 给出的：

[
Synthetic for Supervision
+
Real for Realism
]

非常值得保留。

---

## 共识 7：Physical Renderer + Neural Correction 有较强合理性

传统物理 / Depth Renderer 具有：

- 可解释；
- 参数可控；
- 与 Focus / Aperture 关系明确；

但复杂边缘与真实镜头效应难以完全建模。

生成式 Bokeh 工作则说明 Neural Prior 能处理：

- Hair；
- Transparency；
- Highlight；
- Color Bleeding；
- Complex Boundary。

因此我们的：

[
oxed{
Physical Renderer
+
Neural Residual
}
]

目前得到较好的文献支撑。

---

## 共识 8：Alpha / Object / Focus / Validity 是不同语义

必须持续区分：

[
Object Mask

eq
Alpha

eq
Focus Mask

eq
Depth Validity Mask
]

Alpha 更合理的职责仍是：

> Layer / Coverage。

Focus 则由 Geometry + Focus Parameter 决定。

---

# 15. 当前技术路线中得到文献支持的部分

目前可以认为以下设计有比较明确的文献动机：

### 1. 保留显式 Coarse Depth

因为 Geometry 对 Bokeh 很重要。

### 2. 给 Coarse Depth 配 Mask / Confidence

因为 Geometry Error 会直接污染 Bokeh，且 Mask 可以作为 Attention / Gate 类控制信息。

但 D/M/C 的具体编码方式仍未确定。

### 3. 使用 Video Temporal Modeling

因为 Video Prior / Temporal Attention 对抑制 Flicker 和 Geometry Noise 都有帮助。

### 4. Depth → Focus-relative Geometry → Renderer

比直接 Depth → Blur 更符合已有方法和物理模型。

### 5. Physical Renderer + Neural Residual

物理 Renderer 提供可控主结构，Neural Residual 修复杂成像误差。

### 6. Synthetic + Real 的混合训练数据思路

Synthetic 负责精确监督，Real 负责真实镜头分布。

### 7. 边缘 / 时序必须单独评价

不能只看整图 PSNR / SSIM。

---

# 16. 当前仍然不能下结论的部分

以下设计还不能因为这 5 篇论文就直接确定。

## 16.1 D / M / C 到底怎样 Embedding

候选包括：

- 直接多通道；
- 分支编码；
- Mask 作为 Gate；
- Confidence 作为 Weight；
- Attention Bias。

但现有 5 篇不能替我们决定。

---

## 16.2 Temporal Propagation 到底如何完成

现有 Bokeh 文献没有真正解决：

[
Low	ext{-}FPS Depth
ightarrow
High	ext{-}FPS Geometry
]

因此不能提前决定：

- Optical Flow；
- Warping；
- Learned Propagation；
- Feature Interpolation；
- Cross-Attention；

到底哪一种最好。

---

## 16.3 Spatial Token Merging 是否应该第一版就使用

现有文献只能支持：

> Merge 应该保护 Geometry / Boundary-sensitive Region。

但没有证明：

> 本项目一定需要 Spatial Token Merging。

因此仍应保留：

```text
With Spatial Token Merging
vs
Without Spatial Token Merging
```

作为候选比较，而不是预设实施顺序。

---

## 16.4 Depth Edge 到 Alpha 的具体连接

目前没有直接文献证据。

---

## 16.5 Alpha 是否需要额外 Foreground Color / Background Layer

目前仍未解决。

---

## 16.6 Depth / Alpha / Bokeh 的最优共享方式

目前只能确认：

> 共享可能有收益，也可能产生冲突。

但共享到什么程度仍然没有答案。

---

# 17. 当前最需要补文献的空白项

只根据这五篇的证据分布，目前后续最值得定向补充的是：

## 第一优先级：Q3

[
Low	ext{-}FPS Depth
ightarrow
High	ext{-}FPS Geometry
]

未来应定向查：

- Video Depth Propagation；
- Depth Completion；
- Temporal Depth Completion；
- Optical-flow-guided Geometry Propagation；
- Sparse / Low-rate Geometry Fusion；
- Temporal Feature Warping。

---

## 第二优先级：Q1

[
Spatial Token Merging
ightarrow
Dense Depth / Alpha
]

未来应定向查：

- Token Merging / Pruning for Dense Prediction；
- Dense ViT Token Recovery；
- Boundary-aware Token Merging；
- Multi-scale Decoder；
- Skip Feature Recovery。

---

## 第三优先级：Q2

[
Depth Edge
ightarrow
Alpha
]

未来应定向查：

- Depth-guided Matting；
- Geometry-guided Matting；
- Boundary-aware Matting；
- Cross-task Feature Guidance；
- Multi-task Dense Prediction。

---

## 第四优先级：Q11

[
Depth + Alpha + Bokeh
]

共享主干的收益 / 冲突。

未来应定向查：

- Multi-task Dense Prediction；
- Shared Encoder + Task-specific Decoder；
- Cross-task Feature Interaction；
- Gradient Conflict；
- PCGrad / GradNorm / Alternating Training。

---

# 18. 五篇读完后的当前整体判断

目前五篇 Bokeh 文献并没有直接给出一套可以照搬的完整网络。

但它们已经帮助我们把很多最开始比较模糊的问题拆得更清楚：

```text
输入：
High-FPS RGB
+
Low-FPS Coarse Depth
+
Mask / Confidence

        ↓

[仍然最缺文献]
Low-FPS Depth → High-FPS Geometry
D/M/C 如何融合

        ↓

Video Temporal Representation
→ 有较强文献支持

        ↓

[仍然最缺文献]
Spatial Token Merging
Dense Feature Recovery

        ↓

Depth / Alpha / Bokeh Heads
→ Multi-task Sharing 仍需验证

        ↓

Depth / Disparity
+
Focus
→ Defocus / CoC
→ 文献支持逐渐明确

        ↓

Physical Renderer
+
Neural Residual
→ 有较强文献支持

        ↓

Final Video Bokeh

评价：
Full-image
+
Boundary
+
Temporal
```

因此现阶段最重要的变化不是“已经确定最终网络”，而是：

> **哪些部分已经有文献支撑，哪些部分仍然只是我们的方案假设，现在已经可以清楚地区分。**

---

# 19. 当前结论

读完现有 5 篇 Bokeh 文献后，可以暂时把 Q1～Q11 分成三组。

## A. 已有较强文献线索，可以开始逐步收敛

- **Q4**：Alpha / Layer 与 Focus 应区分；
- **Q5**：Synthetic + Real / 多数据集训练是现实路线；
- **Q6**：Temporal Consistency 不等于不变化；
- **Q8**：Geometry → Focus-relative → Defocus / CoC；
- **Q10**：Coarse Depth 是 noisy prior，需 Reliability + Robustness；
- **Q9 的部分问题**：Temporal Backbone、Decoder、Skip Feature、Geometry Pretraining 都重要。

## B. 有启发，但还不能确定

- **Q7**：Physical Renderer + Neural Residual 合理，但 Alpha / F / B 接口未解决；
- **Q11**：共享确实可能冲突，但我们的三任务共享方式未知。

## C. 仍然明显缺乏直接文献

- **Q1**：Token Merging → Dense Recovery；
- **Q2**：Depth Edge → Alpha；
- **Q3**：Low-FPS Depth → High-FPS Geometry。

因此，在不继续扩展 Bokeh 文献的前提下，当前这五篇已经基本完成了它们最应该承担的作用：

> **帮助我们确认 Bokeh / Geometry / Temporal / Data / Renderer 层面的共性问题，同时暴露出下一轮真正需要跨方向专项检索的空白。**
