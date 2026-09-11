# OPPO Depth-Guided Video Bokeh
## 待调研实现问题清单

> 本文用于持续记录当前方案中已经发现、但暂时不急于给出具体实现方案的问题。
>
> **处理原则：**
>
> 1. 先把问题本身定义清楚；
> 2. 明确它为什么重要、会影响哪些模块；
> 3. 区分“计划书已经明确”与“仍待确定”；
> 4. 对于具体实现细节，暂不提前拍板；
> 5. 后续带着问题查阅相关文献，了解已有方法如何处理；
> 6. 再结合本项目的数据条件、任务目标和效率要求决定最终方案。

---

# Q1. Spatial Token Merging 后如何恢复逐位置的 Depth 与 Alpha？

## 问题描述

当前流程中存在：

```text
Spatial Token Merging
        ↓
Video Attention
        ↓
Depth / Alpha / Bokeh Heads
```

Spatial Token Merging 会把多个空间位置对应的 Token 压缩成更少的 Token：

\[
N\rightarrow N',\qquad N'<N
\]

但 Depth 与 Alpha 都属于 dense prediction，最终需要恢复到逐位置甚至逐像素输出：

\[
D,\alpha\in\mathbb{R}^{H\times W}
\]

因此需要回答：多个图像位置被合并成一个 Token 后，如何重新建立与原始位置的对应，并保留足够的局部细节？

如果多个 patch 被合并成一个 Token，**仅靠把同一个 merged feature 复制回原位置，无法保证人物轮廓、发丝、半透明区域等细节已经恢复**。复制可以恢复位置对应，但是否还需要结合高分辨率特征、解码结构或其他信息，需要进一步调研。

## 为什么重要

这会直接影响：

- Depth discontinuity 的定位；
- Alpha 边缘质量；
- 发丝与半透明区域；
- 前景 / 背景融合；
- 最终 Bokeh 边缘的 halo、foreground erosion 等问题。

## 当前已知

计划书提出类似 ToMe 的空间 Token Merging，用于压缩冗余 Token、降低 Video Attention 开销，但没有明确：

- merged token 如何恢复到原始位置；
- 是否需要 dense decoder；
- 是否需要保留高分辨率 feature；
- 边界区域是否允许 merge；
- 如何避免高频细节损失。

## 当前阶段原则

**不提前确定 Spatial Token Merging 的实施顺序，也不预先决定它是否必须进入第一版。**

后续调研时同时考虑：

- 纳入 Spatial Token Merging 的版本；
- 不进行空间 Token 压缩、保持完整空间 Token 的版本。

不合并版本可以作为候选 baseline / 对照，用于判断空间压缩带来的效率收益是否值得其恢复复杂度与潜在精度损失。最终是否在第一版中启用 Spatial Token Merging，待文献调研和实验设计后决定。

## 后续文献调研方向

- Token Merging / Token Pruning 在 dense prediction 中的应用；
- Vision Transformer 降 Token 后如何恢复空间分辨率；
- Depth Estimation / Semantic Segmentation / Matting 中的 Token Reduction；
- spatial mapping / assignment；
- multi-scale feature / decoder / skip feature；
- edge-aware / boundary-aware token merging。

---

# Q2. Depth Head 如何向 Alpha Head 提供边缘提示？

## 问题描述

三个 Head 并不是完全独立并行。更准确地说，Alpha Head 同时接收：

1. 共享主干产生的 \(F_{\text{shared}}\)；
2. Depth Head 产生的深度边缘辅助提示 \(E_D\)。

关系可表示为：

\[
F_{\text{shared}}\rightarrow Head_{\text{depth}}\rightarrow D^{pred}\rightarrow E_D
\]

同时：

\[
F_{\text{shared}},E_D\rightarrow AlphaHead\rightarrow\alpha
\]

```text
Shared Feature ───────────────────────→ Alpha Head → Alpha
      │                                     ↑
      └→ Depth Head → D_pred → Depth Edge ─┘

Shared Feature ───────────────────────→ Bokeh Head
```

因此更准确的结构表述是：**共享主干、三个任务分支，其中 Depth 分支向 Alpha 分支额外提供边缘提示；Alpha Head 本身仍直接使用共享特征。**

## 为什么重要

Depth discontinuity 往往与前景 / 背景边界有关，可帮助 Alpha Head 处理人物轮廓、发丝、半透明边缘等。但：

\[
Depth\ Edge\neq Alpha\ Edge
\]

人物身体内部、手臂与躯干之间、衣物褶皱等也可能存在深度变化，却不一定是 Alpha 的前景 / 背景边界。

## 当前已知

计划书明确提出使用 Head_depth 的深度不连续图作为 Alpha Head 的边缘提示，但没有明确边缘如何提取、在哪一层注入、具体如何融合、是否通过额外 loss 建立联系，以及如何避免人物内部深度变化误导 Alpha。

## 当前阶段原则

明确：**Alpha Head 同时依赖共享特征与 Depth Edge 辅助提示。Depth Edge 是辅助提示，不要求与 Alpha Edge 处处一致。**具体怎样注入、怎样融合仍保持开放。

## 后续文献调研方向

- Depth-guided matting；
- Geometry-guided alpha prediction；
- Depth-assisted segmentation；
- Boundary-aware multi-task learning；
- cross-task feature fusion / attention。

---

# Q3. Temporal Depth Propagation 中，参与融合的 Depth 特征从哪些帧、哪些位置取？

## 问题描述

草案曾建议使用 `concat + MLP` 或 `weighted addition` 完成 RGB / Depth feature fusion，但这里需要先明确：**参与融合的 Depth feature 是怎样生成、从哪些帧和哪些空间位置获得的？**

“怎么融合”和“融合谁”是两个不同问题。

例如，第 2 帧没有真实 Depth，如果直接使用：

\[
RGB_2(x,y)+D'_2(x,y)
\]

进行同位置融合，那么仅从这个融合操作本身，不能判断是否已经完成运动感知的跨帧传播。关键还要看 \(D'_2\) 是怎么生成的。

如果 \(D'_2\) 只是简单时间插值或同位置插值，其中可能没有充分处理物体运动；如果插值阶段本身已经使用 Optical Flow、warping 或 learned correspondence，那么跨帧运动对应可能已经部分或全部发生在 Depth 插值 / 对齐阶段。

因此不能预先假设一定还需要额外设置一个独立 correspondence 模块。

## 为什么重要

这决定了“低帧率 Depth 引导高帧率 RGB”的真实数据流：运动对应是在输入侧 Depth 插值阶段完成，还是在 token / feature propagation 阶段完成，或由两者共同承担。

## 当前已知

计划书提出时间戳插值、Optical Flow 引导、可学习传播 / merging，并希望将低帧率几何先验传播到缺失 Depth 的 RGB 帧，但没有明确：

- 光流作用于原始 Depth、Depth feature 还是 Token；
- 插值 Depth 是否已经运动感知；
- 候选 Depth feature 来自哪些帧；
- 每个 RGB Token 最终使用哪些空间位置的 Depth 信息；
- 遮挡、出画、新出现区域如何处理。

## 当前阶段原则

**仅凭“当前 RGB Token 与同位置插值 Depth Token 做融合”，不能判断是否已经完成运动感知的跨帧传播；必须结合插值 Depth 的生成过程一起判断。**

后续调研不预先要求一定存在独立 correspondence 模块，而是研究：

```text
低帧率真实 Depth
        ↓
插值 / warp / learned propagation
        ↓
得到当前时刻可用的 Depth 表示
        ↓
RGB-Depth Fusion
```

这些步骤在已有方法中通常如何划分或联合实现。

## 后续文献调研方向

- optical-flow-guided depth interpolation；
- video feature warping；
- motion-aware temporal fusion；
- temporal correspondence；
- cross-frame token matching；
- sparse / low-frame-rate depth propagation；
- video depth completion；
- confidence-aware temporal propagation。

---

# Q4. Alpha Head 中“前景”究竟指什么？

## 问题描述

当前草案将 Alpha Head 描述为输出“前景 Alpha Matte”，但“前景”尚未严格定义。至少可能包括：

- 所有人物 / 某类语义主体；
- 用户指定的某一个主体；
- 某种用于分层渲染的前景图层。

这里需要特别区分：

\[
\boxed{\text{Alpha 的图层 / 覆盖含义}}
\]

与：

\[
\boxed{\text{Renderer 的对焦与清晰度选择}}
\]

Alpha 可以描述某个对象或图层在像素中的覆盖比例，并不必然等同于“最终保持清晰的区域”。

例如画面里两个人一前一后，即使两个人都具有有效 Alpha / 都属于人物前景，最终谁更清晰、谁被虚化，仍可能由 Depth、focus distance、camera parameters 共同决定，而不一定要求 Alpha 只包含前面的人。

因此，“所有人物 Alpha”和“指定主体 Alpha”仍是待选任务定义，不能仅根据最终清晰对象预先确定 Alpha 的标签范围。

## 为什么重要

会直接影响 Alpha GT 如何构造、Alpha Head 需要什么输入、是否需要用户选择信息、多人物场景怎样处理、Renderer 如何使用 Alpha 与 Depth，以及用户改变焦点时是否需要重新预测 Alpha。

## 当前已知

计划书强调时序一致的前景 / 人物 Alpha，用于人物边缘、发丝、半透明区域 soft compositing，并提出 Depth discontinuity 为 Alpha 提供边缘提示，但没有明确：

- Alpha 是否针对所有人物；
- 是否针对指定主体；
- Alpha 是否随 focus point / focus distance 改变；
- 多主体场景下 Alpha GT 应如何定义；
- Alpha 与 Renderer 中对焦选择的职责边界。

## 当前阶段原则

暂时不默认 `all-person mask`，也不默认 `selected-subject matte`。先明确：**Alpha 的图层 / 覆盖含义与 Renderer 的对焦选择是否解耦，以及多人物场景中的 Alpha 标签应如何定义。**

## 后续文献调研方向

- Portrait Video Bokeh 中 Alpha / foreground matte 的定义；
- Computational Bokeh 中主体选择与对焦控制；
- Multi-person portrait matting；
- Subject-aware / target-aware matting；
- focus-conditioned rendering。

---

# Q5. 三种训练标签能否在同一段视频中同时获得？

## 问题描述

联合训练理想情况下希望同一段视频同时具备：

- 高质量 Depth；
- 精细 Alpha；
- 目标 Bokeh / 虚化视频。

即：

\[
\{RGB,\ D^{GT},\ \alpha^{GT},\ Bokeh^{GT}\}
\]

现实中不同数据集往往只提供其中一部分标签，例如一套数据有 Depth，另一套有人像 Alpha，第三套有 Bokeh。这些不能直接拼接成“同场景三标签”。

如果 \(Bokeh^{GT}\) 完全由当前基础 Renderer 自己生成，网络可能只是在逼近该 Renderer，也不能据此证明 Bokeh Head 学到了超出该 Renderer 的真实镜头残差或真实成像特性。

## 为什么重要

会直接决定 Stage 2 三任务联合训练是否可行、三个 Head 是否能在同一 batch 同时监督、多任务 loss 与 PCGrad 是否适用，以及是否需要 synthetic data、partial-label training 或多数据源交替训练。

## 当前阶段原则

**根据最终能够获得的数据类型和标签组合，再决定 Stage 1 / Stage 2 / Stage 3 的具体训练方式。**不先假设一定存在完整三标签数据。

至少需要区分：

```text
A：同场景具备 Depth + Alpha + Bokeh
B：三种标签分散在不同数据集
C：只有 Depth + Alpha，Bokeh GT 需要合成
D：有真实 Bokeh，但没有完整 Depth / Alpha
```

## 后续文献调研方向

- partially labeled multi-task learning；
- multi-dataset multi-task training；
- heterogeneous supervision；
- synthetic data for depth / matting / bokeh；
- synthetic GT 与 real GT 的结合；
- portrait bokeh / computational photography 数据集标注方式；
- sim-to-real / synthetic-to-real。

## 当前结论

最终训练策略应以实际能够获得的数据和标签组合为依据，再决定是否进行完整三标签联合训练，以及如何组织不同任务的数据与监督。

---

# Q6. 时序一致性损失如何区分“错误闪烁”和“真实变化”？

## 问题描述

草案目前用 Optical Flow Warp 表达一种直观的时序一致性约束：

\[
L_{\text{temp-depth}}=\left\|Warp(D_t,F_{t\rightarrow t+1})-D_{t+1}\right\|_1
\]

但**位置对应正确，并不意味着对应位置的深度数值应该保持不变。**

例如人物向镜头靠近，即使光流准确找到同一个人，其真实深度也应变化。如果简单要求 `Warp(D_t) ≈ D_{t+1}`，可能把正常运动和真实深度变化误判为时序不一致。

此外还存在遮挡、解遮挡、物体进出画面等情况，新显露区域甚至不存在有效上一帧对应点。

## 为什么重要

项目真正要抑制的是 Depth flicker、Alpha 边缘抖动、Bokeh 强度 / 边界不合理跳变，而不是压制真实前后运动、真实深度变化、物体形变、遮挡 / 解遮挡或合理焦点变化。

## 当前已知

计划书明确提出使用 Optical Flow Warp、对 Depth / Alpha / Bokeh 增加跨帧一致性约束、对 Bokeh 边缘加入 jitter penalty、对 Renderer 参数增加 temporal low-pass，但没有进一步说明如何区分真实变化与错误 flicker、如何处理遮挡 / 解遮挡以及 visibility。

## 当前阶段原则

草案中的 warp + L1 公式只保留为**概念示意**，不能直接视为最终训练损失。最终 temporal loss 必须允许真实运动、深度变化以及遮挡 / 解遮挡。

## 后续文献调研方向

- temporal consistency for video depth estimation；
- motion-aware temporal loss；
- optical-flow-guided consistency；
- occlusion-aware / visibility-aware temporal loss；
- scene flow / 3D motion consistency；
- temporal consistency for video matting；
- flicker suppression without over-smoothing。

---

# Q7. 预测 Alpha 是否足以支撑最终边缘合成？

## 问题描述

草案中的 Alpha Compositing 形式为：

\[
I_{\text{out}}=\alpha F+(1-\alpha)B_{\text{blur}}
\]

其中除了 \(\alpha\)，还需要前景颜色 \(F\) 和虚化后的背景颜色 \(B_{\text{blur}}\)。当前网络数据流只明确预测了 Alpha，没有说明这两者如何得到。

尤其在发丝、半透明区域，原始 RGB 像素本身已经是前景与背景混合后的结果：

\[
I=\alpha F+(1-\alpha)B
\]

因此不能简单认为原始 RGB 就等于纯前景颜色 \(F\)。

## 为什么重要

如果后续 Renderer 需要显式的前景 / 背景分层，仅预测 Alpha 可能不足以支撑边界区域的高质量重建，会影响 color bleeding、halo 等问题。

## 当前已知

当前草案明确了 Alpha Head、Depth 驱动物理模糊、Alpha 用于 soft compositing，以及 Renderer 会生成虚化结果，但没有明确：

- Renderer 是否显式分离 foreground / background color；
- \(F\) 是否直接来自原始 RGB；
- \(B_{\text{blur}}\) 如何构造；
- 是否需要额外的 foreground color / background reconstruction。

## 当前阶段原则

先作为 Renderer / Matting 交界处的开放问题记录，**不立即增加新的输出 Head**。后续先调研已有 Video Matting、Portrait Bokeh、Layered Rendering 方法如何处理 Alpha 已知但前景 / 背景颜色未显式分离的问题。

## 后续文献调研方向

- foreground color estimation；
- foreground-background decomposition；
- compositional matting；
- layered rendering；
- alpha-aware background blur；
- hair / semi-transparent boundary reconstruction；
- decontamination / color bleeding suppression。

---

# Q8. Depth Head 输出的“深度”具体是什么表示？

## 问题描述

这里应分开两个问题。

### 问题 A：网络输出采用什么几何表示？

可能包括：

- depth；
- inverse depth；
- disparity；
- normalized depth；
- 其他几何表示。

### 问题 B：该表示的尺度和单位是否已知？

采用 inverse depth 或 disparity，并不意味着只能表达相对深度。如果相机内参、baseline、标定关系或单位已知，也可能转换成真实物理距离。

相反，某些网络即使输出名义上的“depth”，也可能存在 scale ambiguity、scale + shift ambiguity 或未知绝对单位。

因此“表示形式”和“是否具有绝对尺度”不能混为一谈。

relative depth 更准确地说是：**表达场景的相对远近或几何关系，但绝对尺度、单位或偏移可能不确定。**

## 为什么重要

会同时影响：

1. Depth Head 的输出形式；
2. Depth supervision / loss；
3. 与 Physical Renderer 的连接。

如果 Renderer 使用真实镜头参数和物理 CoC 模型，需要明确网络输出能否通过已知标定关系转换成 Renderer 所需的物理量。

## 当前已知

目前项目计划强调 OPPO 双目系统提供粗深度先验、Depth Head 预测高帧率稠密几何信息、最终使用几何信息驱动物理虚化，但仍不能确定：

- 真机输入给出 depth、disparity 还是其他形式；
- 是否具有稳定 metric scale；
- 训练 GT 使用何种表示；
- 不同数据源是否具有统一尺度；
- Renderer 是否严格依赖米制距离。

## 当前阶段原则

优先依赖**最终获得的数据、标定信息和真机 Depth 形式**再决定。数据到位后分别确认：

1. 输出形式：depth / inverse depth / disparity / 其他；
2. 数值尺度和单位是否已知；
3. 是否存在 scale / shift ambiguity；
4. 是否有相机标定参数可用于转换；
5. GT Depth 的来源和精度；
6. Renderer 对物理尺度的实际需求。

## 后续文献调研方向

- metric depth estimation；
- relative / scale-invariant depth estimation；
- stereo depth / disparity representation；
- inverse depth prediction；
- monocular / video depth 的 scale ambiguity；
- depth-to-defocus / depth-to-CoC mapping；
- computational photography 中 depth calibration。

## 当前结论

**Depth Head 的“输出表示”和“是否具有绝对尺度”是两个独立问题。**最终根据真机数据、标定参数、GT 表示以及 Renderer 需求分别确定，不能简单把 inverse depth / disparity 等同于 relative depth。

---

# Q9. 候选预训练主干是否适合精细的 Depth / Alpha Dense Prediction？

## 问题描述

草案列出了 Video Swin、VideoMAE、MViT、TimeSformer 等候选 Backbone。它们具有较强的视频时空建模能力，但“能够理解视频内容”和“能够恢复逐像素精细结构”并不是同一个要求。

即使完全不做 Spatial Token Merging，Backbone 自身的 Patch Size、Spatial Downsampling、Hierarchical Stage、Feature Resolution 也可能已经对细节造成压缩。

## 为什么重要

会直接影响：

- Depth 边界精度；
- Alpha Matte 细节恢复；
- Decoder 能获得哪些空间尺度的 feature；
- 是否需要 multi-scale / skip features；
- 最终 Bokeh 边缘质量；
- 计算量与空间分辨率之间的权衡。

## 与 Q1 的关系

Q1 讨论额外 Token Compression 造成的信息损失；Q9 讨论即使不做 Token Merging，Backbone 自身的 patch embedding 和层级下采样是否已经损失关键空间细节。

## 当前阶段原则

不提前锁定 Video Swin、VideoMAE 等具体 Backbone，而是先看与本项目最接近的任务中研究者实际使用了哪些 backbone，以及为什么这么选。

调研每篇相关论文时优先记录：

```text
论文
→ Backbone
→ 预训练任务 / 预训练数据
→ 原始预训练目标
→ 空间下采样倍率
→ 是否输出多尺度特征
→ Decoder / Head 如何连接主干
→ 是否利用不同分辨率特征恢复细节
→ 作者选择该 Backbone 的理由
→ 边界质量方面的实验依据
→ 时序稳定性方面的实验依据
→ 是否适合本项目
```

尤其关注三点：

1. **预训练任务是什么**：视频识别、Masked Video Modeling、图像 / 视频重建、深度、分割等；
2. **怎样连接 Decoder / Head**：是否使用多个 stage、FPN / U-Net、多尺度融合、浅层高分辨率特征；
3. **效果证据是什么**：是否有边界质量、细小结构、Depth boundary、temporal consistency、flicker / jitter 等实验。

## 后续文献调研方向

- Video Depth Estimation；
- Video Matting；
- Video Semantic / Instance Segmentation；
- Video Object Segmentation；
- Portrait Video Bokeh；
- Dense Prediction Transformer；
- hierarchical / multi-scale video backbone。

---

# Q10. 粗深度存在错误时，训练约束是否会迫使网络保留这些错误？

## 问题描述

草案同时提到高质量 Depth GT 监督、Coarse Depth 几何先验约束和 Edge-aware Loss。

若真实深度为：

\[
D^{GT}=2.0\ \text{m}
\]

而粗深度误估为：

\[
D^{coarse}=3.0\ \text{m}
\]

如果同时直接要求预测接近 GT 和粗深度，就会形成冲突。

因此需要明确：**把 Coarse Depth 作为网络输入，并不等价于把它作为“正确答案”监督网络输出。**

## 为什么重要

如果错误 Coarse Depth 被过强写入 loss，可能导致网络无法纠正双目匹配错误、高质量 GT 与粗深度监督产生冲突、错误边界被保留，并使 Confidence / Validity Mask 失去意义。

## 当前已知

当前草案已有 Validity Mask、Confidence、Coarse Depth 输入、高质量 Depth GT，但没有明确：

- “粗深度几何先验约束”具体约束什么；
- 是否直接约束数值；
- Mask / Confidence 如何参与；
- 低置信区域是否弱化或取消约束；
- 粗深度是否只作为输入特征而不额外形成监督 loss。

## 当前阶段原则

先记录问题，不提前确定具体 loss。保留基本原则：

\[
\boxed{\text{Coarse Depth 是带噪先验，不是天然正确的 GT}}
\]

后续先明确它在训练中承担“输入先验”“软约束”“可靠区域锚点”还是其他角色。

## 后续文献调研方向

- noisy depth supervision；
- confidence-aware depth completion；
- sparse / noisy depth guided estimation；
- sensor / stereo depth refinement；
- uncertainty-aware / robust loss；
- depth completion 中 sparse depth 作为 input 与 supervision 的区别。

---

# Q11. 三个任务共享主干是否真的带来整体收益？

## 问题描述

当前方案希望 Depth、Alpha、Bokeh 三个任务共享视频主干 / shared representation，再分别进入对应任务分支。

需要验证的不是“共享后每个子任务都必须超过三个独立网络”，而是：

> **共享表征是否能在任务质量、最终虚化效果和计算开销之间带来整体收益。**

一种共享模型即使某个单项任务只达到接近的质量，只要明显减少参数量 / 计算量、提升最终 Bokeh 效果，或改善跨任务协同 / 时序一致性，也可能是有价值的设计。

## 为什么重要

不同任务对特征的要求可能不同：Depth 强调几何连续性与稳定性；Alpha 需要保留发丝、半透明区域等高频细节；Bokeh / Rendering 更关注最终视觉效果与虚化边界。因此共享既可能产生正迁移，也可能产生 negative transfer。

共享结构的价值应综合评估：

\[
\text{Task Quality}+\text{Final Bokeh Quality}+\text{Efficiency}
\]

## 与 PCGrad 的关系

PCGrad 只能尝试缓解联合训练中的梯度冲突，**不能证明共享主干本身一定是最优结构**。

即使使用 PCGrad，也仍要回答：

- 三个任务共享到哪一层；
- 哪些模块任务专属；
- 部分共享是否比完全共享更合适；
- 共享后是否在整体质量—效率权衡上获得收益。

## 当前阶段原则

不提前假设 Fully Shared Backbone 一定优于 Task-specific Networks，也不提前固定共享范围。后续通过文献调研与候选消融共同判断：**哪些层适合共享、哪些层需要任务专属，以及共享是否在任务质量、最终虚化效果和计算开销之间带来值得保留的收益。**

## 后续文献调研方向

- Multi-task Learning；
- hard / soft parameter sharing；
- partially shared encoder；
- shared backbone + task-specific decoder；
- cross-task feature interaction；
- negative transfer / gradient conflict；
- PCGrad / GradNorm / uncertainty weighting；
- Depth + Segmentation / Matting / Rendering 联合任务。

调研论文时重点记录：

```text
论文
→ 哪些任务联合训练
→ 共享哪些层
→ 哪些层任务专属
→ 是否有 cross-task connection
→ 是否使用 PCGrad / GradNorm 等方法
→ 各子任务质量变化
→ 最终任务 / 渲染效果变化
→ 参数量 / FLOPs / 延迟变化
→ 作者如何证明质量—效率权衡有收益
```

候选比较可以包括：

```text
A：三个任务完全独立
B：共享 Backbone，Head 独立
C：部分共享 + Task-specific Decoder
D：共享 Backbone + Cross-task Connection
E：在候选共享结构上加入 PCGrad / 其他冲突缓解
```

这些仅作为**候选比较方向**，后续根据文献和实验资源选择必要对照，不预先要求全部执行。

## 当前结论

**共享主干的目标不是强制每个子任务都超过独立网络，而是验证它是否在任务质量、最终虚化效果和计算开销之间带来整体收益。PCGrad 只能作为冲突缓解工具，不能替代对共享结构本身的验证。**
