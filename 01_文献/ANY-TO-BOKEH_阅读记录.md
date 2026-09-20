# ANY-TO-BOKEH 阅读记录

## 论文信息

- **题目**：Any-to-Bokeh: Arbitrary-Subject Video Refocusing with Video Diffusion Model
- **会议**：ICLR 2026
- **仓库原文**：`01_文献/ANY-TO-BOKEH.pdf`
- **记录目的**：判断该工作对当前项目 Q1～Q11 的帮助，重点关注显式几何先验、焦平面相关表示、时序建模、Depth/Disparity 鲁棒性、Decoder 细节恢复与视频级 Bokeh 控制。

---

# 1. 论文主要解决什么问题

Any-to-Bokeh 面向**视频 Bokeh / Video Refocusing**。

其核心输入大致包括：

```text
RGB Video
+
Depth / Disparity-related Geometry
+
Focus Plane
+
Blur Strength
```

然后利用视频扩散模型生成最终 Bokeh Video。

与前两篇不同，它重新显式使用几何先验，因此和我们的总体思路更接近。

但它仍然与我们有明显区别：

- 它假设每一帧都有可用的 Depth / Disparity；
- 它直接生成最终 Bokeh RGB；
- 没有显式 Depth Head、Alpha Head、Physical Renderer；
- 我们则是低帧率 Coarse Depth → 高帧率几何 → Depth / Alpha / Bokeh 多任务 → Physical Renderer + Neural Residual。

---

# 2. One-Step Video Diffusion

论文基于 Stable Video Diffusion 一类视频生成 Backbone，但不是传统多步去噪推理，而是采用 one-step 方式减少计算量和多步采样带来的时序不稳定。

整体可以理解为：

```text
RGB Video
+
Geometry / Focus Control
        ↓
One-Step Video Diffusion
        ↓
Bokeh Video
```

因此它虽然仍然属于生成式 Bokeh，但工程形态已经比 BokehCrafter 的多步 diffusion 更接近一次前向的思路。

---

# 3. Focal-Relative Geometry

论文没有简单把原始 Depth / Disparity 直接作为唯一控制变量，而是重点构造“相对于焦平面”的几何信息。

核心思想是：

[
Geometry
ightarrow
	ext{Difference to Focus Plane}
ightarrow
Bokeh Control
]

也就是说，Bokeh 更关心：

> 当前点距离焦平面有多远。

而不只是：

> 当前点离相机有多少米。

这种思路对 Q8 很有价值。

因此需要区分：

1. Depth Head 内部预测什么几何表示；
2. Renderer / Bokeh Module 实际最自然需要什么几何表示。

两者不一定完全相同。

例如网络可以预测：

[
Depth
]

但最终 Renderer 使用的可能是：

[
Depth	ext{-}relative	ext{-}to	ext{-}focus
]

或者 disparity / inverse-depth relative to focus。

---

# 4. Focal-Plane-Adapted MPI

论文使用 MPI（Multi-Plane Image / 多平面表示）来表达相对于焦平面的几何层级。

其核心思想不是简单等间隔划分深度层，而是：

> 焦平面附近划得更细，远离焦平面的区域可以逐渐变粗。

原因是：

- 焦平面附近轻微的几何误差就可能造成清晰 / 模糊归属错误；
- Depth Discontinuity 和人物边缘对 Bokeh 特别敏感；
- 已经明显离焦的远处区域，对小范围几何误差相对不那么敏感。

这一点虽然不能直接解决我们的 Token Merging，但给 Q1 一个重要启发：

[
oxed{
Spatial Computation Should Be Geometry/Boundary Aware
}
]

也就是说未来 Spatial Token Merging 不一定应该只根据 Token 相似度决定是否合并，还可以研究是否引入：

- Depth Boundary；
- Focus Proximity；
- Confidence；
- Foreground Boundary；

来保护关键位置。

---

# 5. MPI Attention 与 Mask 的使用方式

论文中的 MPI 信息不是简单地作为一个普通输入通道全部拼接进网络，而会直接参与 Attention / Spatial Guidance。

这对我们新增的 D / M / C 融合问题非常有启发。

它说明：

> Mask 不一定必须被当成“普通图像 Feature”。

Mask 也可以承担：

- Attention Mask；
- Spatial Guidance；
- Feature Gate；
- Region Selection；

等作用。

因此对于我们的：

[
[D,M,C]
]

可以把未来候选思路拆成：

```text
D
→ Geometry Feature

M
→ Validity Mask / Gate / Attention Guidance

C
→ Reliability Weight / Confidence Encoding
```

而不是预先假设一定：

```text
[D,M,C]
→ 三通道直接一起做 Embedding
```

需要强调：

> Any-to-Bokeh 的 MPI Mask 表示的是焦平面相关的几何区域，并不是我们的 Depth Validity Mask。

所以这里只能作为“Mask 如何参与网络”的结构启发，不能直接照搬。

---

# 6. Projection / Adapter 对 Q2 的启发

论文中的视觉条件特征也需要先经过轻量 Adapter / Projection，再与主网络中的 Query Feature 对齐。

这直接说明：

[
	ext{空间尺寸相同}

eq
	ext{Feature Dimension 相同}
]

例如两个模态即使都对应同一个 (H	imes W) 图像：

- RGB Encoder 输出维度可能是 (C_1)；
- Geometry Encoder 输出维度可能是 (C_2)。

因此仍然需要：

[
C_1ightarrow C
]

[
C_2ightarrow C
]

映射到统一表示空间。

这一点可以认为支持了我们当前对 Projection Layer 的理解。

---

# 7. Disparity Noise Training

这是论文对我们 Q10 最有价值的部分之一。

作者考虑到真实 Depth / Disparity 并不可靠，因此在训练阶段主动对几何输入加入扰动。

包括类似：

- Elastic Transform；
- Perlin Noise；
- Morphological Dilation；
- Morphological Erosion；

等 corruption。

并且重点污染：

- Focus Region；
- Disparity Boundary；
- 物体边缘附近。

其目的不是让网络机械复制输入 Disparity，而是让网络学会：

> 输入几何存在错误时，仍然利用 RGB + Temporal Context 输出稳定 Bokeh。

这和我们的核心原则高度一致：

[
oxed{
Coarse Depth

eq
Ground Truth
}
]

---

# 8. 对 Q10 的直接启发

我们现在可以把 Coarse Depth 鲁棒性分成三个层面考虑。

## 层面 A：输入可靠性编码

```text
Depth
+
Mask
+
Confidence
```

告诉网络：

> 哪些 Depth 是有效的、哪些更可信。

## 层面 B：Geometry Noise Augmentation

```text
Clean / Better Depth
        ↓
Artificial Corruption
        ↓
Noisy Depth
        ↓
训练网络纠错
```

主动让网络在训练时见到错误几何。

## 层面 C：Temporal Geometry Denoising

利用邻帧时序信息：

```text
t-1：正常
t：Depth 异常
t+1：正常
        ↓
Temporal Model
        ↓
不要完全跟随 t 的错误值
```

这三种机制理论上并不冲突。

Any-to-Bokeh 至少为后两种提供了较直接的文献启发。

---

# 9. Temporal Block 的作用

论文的时序模块不仅用于：

> 让视频看起来连续。

还用于：

> 对单帧几何噪声进行时间域上的鲁棒处理。

这给我们的 Temporal Module 增加了第二层含义：

### 原来的理解

[
Low	ext{-}FPS Depth
ightarrow
High	ext{-}FPS Geometry
]

### 新的理解

还可以承担：

[
Temporal Geometry Denoising
]

也就是说未来的 Video Transformer / Temporal Attention 可能同时承担：

1. 补充缺失时刻的 Geometry；
2. 利用邻帧抑制单帧 Depth Error；
3. 保持最终 Depth / Alpha / Bokeh 时序稳定。

---

# 10. 对 Q3 的帮助边界

Any-to-Bokeh 对 Q3 有启发，但**并没有真正解决我们的 Low-FPS Depth Propagation**。

它通常假设视频每帧已经具有对应的 Depth / Disparity。

而我们的输入是：

```text
RGB：
1 2 3 4 5 6 7 8

Depth：
1     4     7
```

因此它没有回答：

> 缺失帧的几何信息究竟怎样从真实 Depth 帧传播过去。

它能告诉我们的只是：

> 当每帧已经有某种 Geometry Representation 后，Video Temporal Module 可以继续利用多帧信息提高稳定性和鲁棒性。

所以 Q3 仍然需要专门查：

- Sparse / Low-FPS Depth Propagation；
- Optical-Flow-guided Depth Propagation；
- Temporal Feature Warping；
- Video Depth Completion。

---

# 11. Decoder Fine Detail Recovery

论文还专门考虑 latent / VAE 压缩导致的高频细节损失。

尤其：

- Hair；
- Object Edge；
- Fine Texture；
- Focus Boundary。

因此在后期训练中会专门优化 Decoder，并通过高分辨率 / Skip Feature 等方式帮助恢复细节，同时加入基于 Gradient / Texture 的约束。

这一点对 Q9 很重要。

它说明：

> 强 Video Backbone 并不能自动保证 Dense Prediction / Boundary Detail。

后续我们选 Backbone 时，必须同时研究：

- 是否输出 Multi-scale Feature；
- 是否保留 Shallow High-resolution Feature；
- Decoder 如何接主干；
- 是否有 Skip Connection；
- Boundary / Gradient Loss 是否必要。

尤其 Alpha Head 不应该默认只依赖最后一层低分辨率 Video Token 再简单上采样。

---

# 12. Boundary Metric

论文还特别关注 focused subject edge 的保真度，并提供专门的边缘相关评价指标 / 评价思路。

这与前面 BokehFlow 的 Edge PSNR / SSIM 形成一致趋势：

> Bokeh 不能只看全图指标。

我们的未来评价体系应该明确分为：

```text
1. Full-image Quality
2. Boundary / Edge Quality
3. Temporal Quality
```

这一方向已经得到多篇 Bokeh 文献共同支持。

---

# 13. Long Video Inference

视频模型通常只能处理固定长度 Clip。

因此论文还采用 Overlap / Weighted Fusion 一类策略连接相邻 Clip，以减少：

```text
Clip 1 | Clip 2
         ↑
       边界跳变
```

这不是我们当前最核心的问题，但如果以后模型采用滑窗 Video Transformer，也会遇到类似问题。

因此未来 Long Video Inference 需要单独考虑：

- Overlap；
- Sliding Window；
- Boundary Fusion；
- Hidden State / Memory；
- Clip-to-Clip Consistency。

---

# 14. Alpha 与 Focus 的关系

Any-to-Bokeh 并没有 Alpha Head。

其推理流程中可以利用目标对象 Mask / Focus Target 来指定需要对焦的主体，同时几何信息决定当前焦平面。

这一点继续支持我们的 Q4：

[
oxed{
	ext{Object / Alpha Layer}

eq
	ext{Focus Selection}
}
]

也就是说：

- Alpha 可以表示人物 / 图层覆盖；
- Focus Target 可以随用户控制改变；
- 清晰程度由 Focus + Geometry 决定。

不能预先把“Alpha=清晰区域”。

---

# 15. 对 Q7 的帮助

Any-to-Bokeh 仍然采用：

[
RGB+Geometry
ightarrow
Neural Bokeh
]

而不是显式使用：

[
alpha F+(1-alpha)B
]

因此它再次说明：

> Neural Generator 可以直接学习复杂的 Bokeh Boundary 和真实视觉效果。

这继续支持我们的：

[
B_{final}
=
B_{physical}
+
Delta B_{neural}
]

但仍然不能回答：

> 仅靠 Depth + Alpha + RGB 的 Physical Renderer 是否已经足够处理 Hair / Transparency / Foreground Color Decontamination。

所以 Q7 仍保持开放。

---

# 16. 对 Q11 的帮助

这篇不是 Depth / Alpha / Bokeh 三任务 Multi-task Model。

因此不能直接回答：

> 三个任务共享 Backbone 是否比独立网络更好。

但其结构说明：

```text
Shared Video Backbone
+
Geometry-specific Module
+
Temporal Module
+
Decoder Refinement
```

是一种常见的“共享主干 + 专门子模块”分工模式。

这可以作为未来 Q11 Partial Sharing 的结构启发，但不能作为多任务共享有效性的直接证据。

---

# 17. Q1～Q11 对应关系

| 问题 | Any-to-Bokeh 帮助程度 | 当前判断 |
|---|---|---|
| Q1 Token Merging 后 Dense 恢复 | △ | 无 Token Merging，但支持 Geometry / Boundary-aware spatial computation 与 Skip Feature 恢复细节 |
| Q2 Depth Edge → Alpha | ❌ | 无 Alpha Head |
| Q3 Low-FPS Depth Propagation | △ | Temporal Module 可利用多帧 Geometry，但假设每帧已有 Depth |
| Q4 Alpha / Focus 定义 | ✅ 间接有价值 | 支持对象 / 图层与 Focus Target 分离 |
| Q5 多标签数据 | △ | 不是 Depth+Alpha+Bokeh 三任务联合标签方案 |
| Q6 Temporal Consistency | ✅ 很有价值 | Temporal Block + Longer Sequence + Noisy Geometry Training |
| Q7 Alpha 是否足够 Renderer | △ | Neural Generator 绕过显式 compositing，支持 Neural Residual 思路 |
| Q8 Depth 表示 | ✅ 很有价值 | 强调 focal-relative disparity / geometry |
| Q9 Backbone / Decoder | ✅ 很有价值 | Video Prior + Temporal Module + Decoder / Skip Feature 细节恢复 |
| Q10 Coarse Depth 错误 | ★★★ 非常有价值 | Geometry corruption + Temporal Refinement |
| Q11 Multi-task Sharing | ❌ / △ | 无三任务联合，仅能提供 shared backbone + specialized module 的结构启发 |

---

# 18. 当前最值得保留的四条启发

## 启发 1：主动做 Geometry Noise Augmentation

未来不能只依赖 Mask / Confidence 告诉网络 Depth 哪里错。

还可以主动训练：

[
Clean Geometry
ightarrow
Corrupted Geometry
]

让模型学会纠错，而不是复制输入。

---

## 启发 2：Bokeh Geometry 应围绕 Focus 表达

网络内部 Depth Representation 与 Renderer 的 Bokeh Geometry Representation 可以分开。

最终 Blur 更直接依赖：

[
Geometry Relative to Focus
]

而不是单纯绝对 Depth。

---

## 启发 3：高频细节需要 Decoder 专门恢复

不能假设 Video Transformer 的最终低分辨率 Feature 自动包含 Hair / Fine Boundary。

后续需要重点调查：

- Multi-scale Decoder；
- Skip Connection；
- Gradient / Boundary Loss；
- High-resolution Branch。

---

## 启发 4：Mask 不一定只是输入通道

Mask 可以承担：

- Attention Guidance；
- Validity Gate；
- Spatial Constraint。

因此我们的：

[
D,M,C
]

未来可以考虑分工式编码，而不是预先固定为三通道直接 Embedding。

---

# 19. 与我们的结构对比

## Any-to-Bokeh

```text
RGB Video
+
Per-frame Depth / Disparity
+
Focus
        ↓
Focal-relative Geometry
        ↓
Geometry-guided Spatial Module
        ↓
Temporal Block
        ↓
Video Generator
        ↓
Bokeh
```

## 我们当前路线

```text
High-FPS RGB
+
Low-FPS Coarse Depth
+
Mask
+
Confidence
        ↓
Embedding / Alignment
        ↓
Temporal Depth Propagation
        ↓
Token Merging
        ↓
Video Transformer
        ↓
Depth / Alpha / Bokeh Heads
        ↓
Physical Renderer
+
Neural Residual
```

共同点：

[
oxed{
Explicit Geometry
+
Video Temporal Prior
}
]

关键区别：

> Any-to-Bokeh 假设每帧已经有 Geometry；我们要解决的是 Low-FPS Coarse Depth 如何真正传播并重建 High-FPS Geometry。

因此，我们目前最特殊、最没有被前三篇解决的核心问题仍然是：

[
oxed{
Low	ext{-}FPS Depth
ightarrow
High	ext{-}FPS Geometry
}
]

以及：

[
oxed{
Depth / Alpha / Bokeh Multi-task Sharing
}
]

这两个方向后续仍需专项文献调研。
