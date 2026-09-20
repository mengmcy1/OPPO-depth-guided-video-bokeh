# Bokeh Diffusion 阅读记录

## 论文信息

- **题目**：Bokeh Diffusion: Defocus Blur Control in Text-to-Image Diffusion Models
- **会议**：ACM SIGGRAPH Asia 2025
- **作者**：Armando Fortes, Tianyi Wei, Shangchen Zhou, Xingang Pan
- **仓库原文**：01_文献/Siggraph2025-Bokeh Diffusion- Defocus Blur Control in Text-to-Image Diffusion Models.pdf
- **记录目的**：判断该工作对当前项目 Q1～Q11 的帮助，重点关注物理 Defocus 参数、Synthetic + Real 混合数据、复杂边缘 Bokeh、控制信息注入和生成式 Bokeh 对 Physical Renderer 的启发。

---

# 1. 论文主要解决什么问题

Bokeh Diffusion 不是传统的后处理 Bokeh Renderer，而是在 Text-to-Image Diffusion Model 内部直接加入可控的 Defocus / Bokeh 参数。

传统文本生成模型通常只能通过类似“f/1.8”“shallow depth of field”的 Prompt 间接控制景深。

问题是：

> 修改 Prompt 中的光圈 / Bokeh 描述，往往不只改变 Blur，还会改变人物、物体位置和场景内容。

因此论文希望实现：

[
	ext{Same Scene}
+
K_{	ext{bokeh}}
ightarrow
	ext{Different Bokeh Level}
]

即改变 Bokeh 强度时，场景本身尽量保持不变。

---

# 2. 与我们项目的根本区别

Bokeh Diffusion 的主要任务是：

~~~text
Text Prompt
+
Physical Bokeh Parameter K
        ↓
Diffusion Model
        ↓
Generated Image with Controlled Bokeh
~~~

以及对真实图像进行 inversion 后的 Bokeh Editing。

我们的任务则是：

~~~text
High-FPS RGB Video
+
Low-FPS Coarse Depth
+
Mask / Confidence
        ↓
Geometry Propagation
        ↓
Depth / Alpha / Bokeh
        ↓
Physical Renderer
+
Neural Residual
~~~

因此这篇不能作为我们的整体网络模板。

它最有价值的是：

- Bokeh 的物理控制量怎么表达；
- Synthetic Bokeh 数据怎样和真实镜头数据结合；
- 为什么经典 Depth-based Renderer 在复杂边缘仍然容易出错；
- 生成式先验怎样补充 Physical Rendering。

---

# 3. Thin-Lens / CoC 与 Bokeh 参数

论文显式从 Thin Lens Model 出发描述 Circle of Confusion（CoC）。

在实际 Blur Rendering 中，可写成：

[
r
=
KcdotDelta disp
]

其中：

[
Delta disp
=
left|
rac{1}{S_1}
-
rac{1}{S_2}
ight|
]

即 Blur Radius 本质上与当前点相对于焦平面的 disparity / inverse-depth difference 密切相关。

---

# 4. 对 Q8 的启发

这进一步支持前面多篇论文形成的共同结论：

[
Depth / Disparity
ightarrow
Focus	ext{-}Relative Geometry
ightarrow
Defocus / CoC
]

比简单：

[
Depth
ightarrow
Bokeh
]

更符合物理 Bokeh 的定义。

因此需要明确区分：

## Depth Head Representation

网络内部可以预测：

- Depth；
- Inverse Depth；
- Disparity；
- 其他 Geometry。

## Renderer Representation

Renderer 真正使用的可以是：

- Disparity Difference；
- CoC；
- Defocus Map；
- Focus-relative Geometry。

二者不需要完全相同。

---

# 5. Decoupled Bokeh Conditioning

论文没有把物理 Bokeh 参数 K 塞进 Text Prompt，而是把它作为独立控制条件。

先经过：

[
c_b=MLP(K)
]

把一个标量 Bokeh 参数转成 Feature。

然后增加独立的 Bokeh Cross-Attention：

[
h_{new}
=
Attention(Q,K_{text},V_{text})
+
lambda Attention(Q,K_b,V_b)
]

其核心思想是：

> 语义内容控制与摄影 / Bokeh 控制最好保持一定程度的解耦。

这样改变 K 时，不需要同时改变文本语义。

---

# 6. 对我们条件融合方式的启发

这不能直接回答 RGB / Depth Token 如何融合，但提供了一个通用原则：

> 不同语义的控制信息，不一定应该全部拼进一个输入 Feature。

例如我们未来有：

- RGB Feature；
- Geometry Feature；
- Focus Distance；
- Aperture；
- Blur Strength；
- Depth Confidence。

其中：

~~~text
RGB / Depth
→ 主视觉 / 几何 Feature

Focus / Aperture / Blur Parameter
→ 独立 Control Embedding
~~~

可能比所有变量全部 concat 更加清晰。

因此以后 Physical Renderer 参数或 Bokeh Head 条件可以考虑独立编码，而不必和 Depth Token 混为同一种信息。

---

# 7. Hybrid Training：真实 Bokeh + Synthetic Bokeh

这是这篇对我们 Q5 最有价值的部分。

论文指出：

> 真正“同一个场景、不同光圈”的大规模真实配对数据非常难采集。

原因包括：

- 同一机位需要多次拍摄；
- 动态人物 / 动物无法保持完全一致；
- 多光圈真实数据集通常规模较小；
- 真实摄影数据还存在拍摄偏好，例如人像常用大光圈、风景常用小光圈。

因此作者没有只用 Synthetic Data，也没有只用 Real Data。

而是构造：

[
oxed{
Real In	ext{-}the	ext{-}Wild
+
Synthetic Paired Bokeh
}
]

的 Hybrid Training Pipeline。

---

# 8. Real In-the-Wild 数据

作者从 Flickr / Megalith-10M 一类真实摄影数据中筛选约 15K 图像，并利用 EXIF 信息获得：

- F-number；
- Focal Length；
- 等真实镜头参数。

同时还为图像构造：

- Depth Estimate；
- Foreground Mask；
- Caption。

这些真实照片提供真正来自物理镜头的 Bokeh Distribution。

也就是说真实数据主要负责教模型：

[
	ext{真实镜头 Bokeh 长什么样}
]

---

# 9. Synthetic Blur Augmentation

Real In-the-Wild Data 有一个严重问题：

> 同一场景没有不同 Blur Level 的严格配对。

因此模型可能把场景内容变化误认为 Bokeh 参数变化。

所以作者找到接近 All-in-Focus 的图像，再利用 Depth + Bokeh Renderer 合成不同 Blur Level。

得到：

~~~text
Same Scene
Sharp / Low Bokeh
↕
Same Scene
Strong Bokeh
~~~

这种配对数据。

Synthetic Pair 的主要作用不是提供最真实的镜头外观，而是明确告诉网络：

> 改变 K 应该只改变 Blur，而不应该改变 Scene Content。

---

# 10. 对 Q5 的重要启发

这篇比前面“单纯 Synthetic Bokeh GT”进一步走了一步。

它告诉我们：

> **Synthetic 与 Real 的职责可以分开。**

可以理解为：

### Synthetic Data

负责：

- 精确监督；
- 同场景配对；
- Geometry → Bokeh 对应关系；
- 控制变量变化。

### Real Data

负责：

- 真实镜头分布；
- 真实 Highlight；
- 真实 Lens Imperfection；
- 真实 Hair / Transparency / Background Appearance。

这一点非常适合我们的项目。

未来如果无法获得：

[
RGB+Depth+Alpha+Bokeh
]

完整真实四元组，也不一定要求一个数据集解决所有问题。

可能采用：

~~~text
Synthetic Data
→ 强监督 Geometry / Alpha / Bokeh correspondence

Real Bokeh Data
→ 监督最终视觉真实性 / Neural Residual
~~~

这应该列入 Q5 的重要候选方向。

---

# 11. Grounded Self-Attention

论文还有一个核心模块：

**Grounded Self-Attention**。

目的是：

> 当 Bokeh 参数变化时，尽量保持 Scene Layout 不变。

它使用一个 Pivot Image（例如清晰版本）作为场景结构锚点。

在生成另一个 Bokeh Level 时，让 Target Image 的 Attention 受 Pivot Feature 约束。

简单理解：

~~~text
Pivot / Sharp Image
→ 告诉模型：
  人在哪
  桌子在哪
  物体形状是什么

Target Bokeh Feature
→ 告诉模型：
  现在应该是什么 Blur Level
~~~

二者通过特定 Self-Attention 组合，使改变 Bokeh 时尽量只改变模糊效果，而不改变物体形状和位置。

---

# 12. 这个机制对我们有什么启发

我们的任务不是生成不同随机场景，因此没有完全相同的问题。

但是其思想非常有意义：

> **需要把“内容结构”和“Bokeh Appearance”解耦。**

对应我们：

~~~text
Original RGB
→ Scene / Appearance Anchor

Depth + Alpha
→ Geometry / Layer

Camera Parameters
→ Defocus Control

Bokeh Head
→ Residual Appearance
~~~

因此 Bokeh Head 最好不要拥有无限制修改整个 Scene Content 的自由。

它更合理的职责应该偏向：

> 修正 Bokeh Appearance，而不是重新生成主体结构。

这与我们当前“Neural Residual 而不是完整 Neural Bokeh Image”的理解是一致的。

---

# 13. 经典 Physical Renderer 的问题

论文明确指出，传统 Depth-based Scatter / Bokeh Renderer 在 Depth Discontinuity 处容易产生 Color Bleeding。

典型区域包括：

- Hair；
- Thin Structure；
- Foreground / Background Edge；
- Translucent Material。

因为一个像素的颜色被根据 Blur Radius 向邻域散射时，如果前景 / 背景的 Depth 不连续，很容易发生错误颜色混合。

这一点对我们的 Q7 非常重要。

---

# 14. 对 Q7：Alpha 是否足够？

Bokeh Diffusion 并没有直接回答：

> Alpha + Depth 是否足够。

但是它非常明确地支持：

> **仅靠经典 Geometry-based Blur Rendering，在复杂边缘很难达到完全真实的效果。**

论文认为生成模型的先验在：

- Thin Hair；
- Translucent Material；
- Complex Boundary；

等区域可以产生更自然结果。

因此我们的：

[
Physical Renderer
+
Neural Residual
]

现在有更直接的文献依据。

可以把职责理解成：

~~~text
Physical Renderer
→ 大部分可解释、可控的 Defocus

Neural Residual
→ Depth Discontinuity / Hair / Transparency
  / Highlight / Color Bleeding 等复杂误差
~~~

---

# 15. 但 Q7 仍然没有完全解决

Bokeh Diffusion 直接采用生成模型，没有研究我们的显式：

[
I=alpha F+(1-alpha)B
]

所以它没有告诉我们：

- Foreground Color F 是否要单独估计；
- Background Color 如何恢复；
- Alpha 是否足以解决 Hair；
- 是否需要 Decontamination。

因此 Q7 依然开放。

它只能证明：

> 复杂边界确实是经典 Renderer 的真实弱点，并且 Neural Prior 可以帮助修正。

---

# 16. 对 Alpha / Object Mask 的启发边界

论文训练数据处理中使用了 Foreground Mask，但它不是我们任务中的 Alpha Matte Head。

这里仍然要区分：

[
Foreground/Object Mask

eq
Alpha Matte
]

二值主体 Mask 无法表达：

- Hair fractional coverage；
- Motion blur；
- Semi-transparent material；
- Soft boundary。

因此这篇不能用来回答 Q4 中 Alpha 的完整定义。

---

# 17. 对 Q10 的帮助有限

论文在构造 Synthetic Bokeh 时依赖估计 Depth，因此 Depth Quality 会影响合成结果。

但是：

> Bokeh Diffusion 并没有专门研究 noisy Depth 如何进入网络、Confidence 怎么编码或如何利用时序纠正 Depth。

而且最终推理模型并不要求用户输入 Depth Map。

所以不能把它当作 Q10 的直接解决方案。

Q10 目前仍然主要参考：

- Any-to-Bokeh 的 Geometry Corruption；
- MagicBokeh 的 Degradation-Aware Depth；
- 后续 Depth Completion / Refinement 文献。

---

# 18. 对 Q9 的启发

Bokeh Diffusion 在：

- Stable Diffusion / U-Net；
- FLUX / Transformer；

两类不同生成架构上都验证了其 Bokeh Conditioning 思路。

这说明：

> Bokeh 的 Geometry / Physical Control 不一定依赖某一种特定 Backbone。

但它仍然是图像生成任务，并不能证明这些 Backbone 适合我们的：

- Dense Depth；
- Alpha Matting；
- Video Temporal Prediction。

所以 Q9 中它更适合支持：

> Control / Adapter 可以相对 Backbone 独立设计。

而不能直接支持我们选 FLUX / Diffusion Transformer 做主干。

---

# 19. Q1～Q11 对应关系

| 问题 | Bokeh Diffusion 帮助程度 | 当前判断 |
|---|---|---|
| Q1 Token Merging Dense Recovery | ❌ | 不涉及 Token Merging / Dense Decoder |
| Q2 Depth Edge → Alpha | ❌ | 无 Alpha Head |
| Q3 Low-FPS Depth Propagation | ❌ | 单图 T2I / Editing 方法 |
| Q4 Alpha / Focus 定义 | △ | 使用 Foreground Mask，但不是 Alpha Matte，不能直接回答 |
| Q5 三标签 / 数据来源 | ★★★ 非常有价值 | Real ITW + Synthetic Paired Bokeh 的 Hybrid Training 非常值得借鉴 |
| Q6 Temporal Consistency | ❌ | 非视频方法 |
| Q7 Alpha 是否足够 Renderer | ★★ 很有价值的间接证据 | 明确指出经典 Depth Renderer 在 Depth Boundary / Thin / Transparent 区域困难，支持 Neural Residual |
| Q8 Depth Representation | ✅ 有价值 | Thin Lens / CoC / disparity difference 再次支持 Focus-relative Geometry |
| Q9 Backbone | △ | Conditioning 可跨 U-Net / Transformer，但不是 Dense Video Backbone 证据 |
| Q10 Coarse Depth Error | △ / 较弱 | 训练数据依赖 Depth Estimate，但没有 noisy-depth robustness 方法 |
| Q11 Multi-task Sharing | ❌ | 不涉及我们的多任务共享 |

---

# 20. 当前最值得保留的四条启发

## 启发 1：Q5 应考虑 Hybrid Real + Synthetic Training

不必在：

~~~text
全真实
vs
全合成
~~~

之间二选一。

更合理的是让两类数据承担不同职责：

~~~text
Synthetic
→ 精确配对 / Geometry Supervision

Real
→ Lens Realism / Domain Distribution
~~~

---

## 启发 2：Bokeh Control Parameter 应与 Scene Content 解耦

例如：

~~~text
Scene Feature
+
Geometry Feature
+
Independent Camera/Bokeh Control Embedding
~~~

而不是把所有变量机械 concat。

---

## 启发 3：Physical Renderer 在 Depth Discontinuity 上确实存在天然困难

Hair、Thin Structure、Transparency、Occlusion Boundary 等问题不能轻视。

因此：

[
Physical Renderer + Neural Residual
]

是一条合理的混合路线。

---

## 启发 4：Neural Residual 不应该随意改变场景结构

Grounded Self-Attention 的核心思想是：

> Bokeh 变化不等于 Scene Content 变化。

对于我们而言，Bokeh Head / Residual Head 更适合作为：

[
	ext{Rendering Correction}
]

而不是：

[
	ext{Free Image Generator}
]

---

# 21. 当前结论

Bokeh Diffusion 对我们的主要价值集中在：

[
oxed{
Q5, Q7, Q8
}
]

其中：

- **Q5**：非常重要。提出 Real In-the-Wild + Synthetic Pair 的 Hybrid Training；
- **Q7**：明确支持经典 Depth Renderer 在复杂边缘存在真实局限，强化 Neural Residual 的必要性；
- **Q8**：再次支持 Focus-relative disparity / CoC 作为更自然的 Bokeh 几何表示。

它基本不能解决：

- Q1 Spatial Token Merging；
- Q2 Depth → Alpha；
- Q3 Low-FPS Depth Propagation；
- Q6 Temporal Consistency；
- Q11 Multi-task Sharing。

因此读完五篇 Bokeh 文献后，后续文献检索必须开始向：

- Video Depth Propagation；
- Depth Completion / Refinement；
- Video Matting；
- Dense Token Merging / Pruning；
- Multi-task Dense Prediction；

这些更专门的方向扩展。
