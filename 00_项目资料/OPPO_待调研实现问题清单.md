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

当前流程是：

```text
Spatial Token Merging
        ↓
Video Attention
        ↓
Depth / Alpha / Bokeh Heads
```

但 Spatial Token Merging 会将多个空间位置压缩为更少的 Token：

\[
N\rightarrow N',\qquad N'<N
\]

而 Depth 与 Alpha 都属于 dense prediction，最终需要恢复到逐位置甚至逐像素输出：

\[
D,\alpha\in\mathbb{R}^{H\times W}
\]

因此中间存在一个关键缺口：多个图像位置被合并成一个 Token 后，如何重新恢复到原来的空间位置，并保留足够的局部细节？例如四个 patch 合成一个 Token 后，不能简单把同一个 merged feature 再复制回四个位置，因为人物轮廓、发丝、半透明区域等高频细节可能已经丢失。

## 为什么重要

这个问题会直接影响 Depth discontinuity、Alpha 边缘、发丝与半透明区域、前景/背景融合，以及最终 Bokeh 边缘的 halo、foreground erosion 等问题。

## 当前已知

计划书提出类似 ToMe 的空间 Token Merging，用于压缩冗余 Token、降低 Video Attention 开销，但目前没有明确：merged token 如何恢复到原始位置、是否需要 dense decoder、是否保留高分辨率 feature、边界区域是否允许 merge，以及如何避免高频细节损失。

## 当前阶段原则

第一版 baseline 暂时不把 Spatial Token Compression 作为必须模块，优先保证：

```text
Temporal Depth Propagation / Cross-modal Fusion
        ↓
Video Attention
        ↓
Dense Depth / Alpha / Bokeh
```

完整跑通。Spatial Token Merging 暂作为后续效率优化问题。

## 后续文献调研方向

- Token Merging / Token Pruning 在 dense prediction 中的应用；
- Vision Transformer 降 Token 后如何恢复空间分辨率；
- Depth Estimation / Semantic Segmentation / Matting 中的 Token Reduction；
- merged token 的 spatial mapping / assignment；
- multi-scale feature / decoder / skip feature；
- edge-aware / boundary-aware token merging。

---

# Q2. Depth Head 如何向 Alpha Head 提供边缘提示？

## 问题描述

当前方案并不是三个 Head 完全独立并行。更准确的结构是：

\[
F_{\text{shared}}
\rightarrow
Head_{\text{depth}}
\rightarrow
D^{pred}
\rightarrow
Depth\ Edge
\rightarrow
AlphaHead
\]

因此应表述为：**共享主干、三个任务分支，其中 Depth 分支向 Alpha 分支提供边缘提示。** 这不意味着需要重复运行 backbone，仍然可以一次前向并端到端训练，只是任务 Head 之间存在有向依赖。

## 为什么重要

Depth discontinuity 往往与前景/背景边界相关，可帮助 Alpha Head 处理人物轮廓、发丝、半透明边缘等。但：

\[
Depth\ Edge\neq Alpha\ Edge
\]

人物身体内部、手臂与躯干之间、衣物褶皱等都可能存在深度变化，却不一定是 Alpha 的前景/背景边界。

## 当前已知

计划书明确提出使用 Head_depth 的深度不连续图作为 Alpha Head 的边缘提示，但没有明确边缘如何提取、在哪一层注入、具体如何融合、是否通过额外 loss 建立联系，以及如何避免人物内部深度变化误导 Alpha。

## 当前阶段原则

先统一结构表述，并明确：**Depth Edge 是辅助提示，不要求与 Alpha Edge 处处一致。** 具体融合方式暂不确定。

## 后续文献调研方向

- Depth-guided matting；
- Geometry-guided alpha prediction；
- Depth-assisted segmentation；
- Boundary-aware multi-task learning；
- Depth 与 Segmentation / Matting 的跨任务特征传递；
- cross-task feature fusion / attention。

---

# Q3. Temporal Depth Propagation 中，参与融合的 Depth 特征从哪些帧、哪些位置取？

## 问题描述

草案曾建议第一版用 `concat + MLP` 或 `weighted addition` 完成 RGB / Depth feature fusion，但这里缺少更前置的问题：**参与融合的 Depth feature 到底从哪里来？**

“怎么融合”和“融合谁”是两个不同问题。

例如：

```text
第 1 帧：有真实 Depth
第 2 帧：无真实 Depth
第 3 帧：无真实 Depth
第 4 帧：有真实 Depth
```

如果人物发生移动，第 2 帧只使用：

\[
RGB_2(x,y)+D'_2(x,y)
\]

进行同位置融合，只能说明当前 RGB 与当前插值 Depth 做了跨模态融合，不能说明已经实现：

\[
D_{t_1}(x_1,y_1)
\rightarrow
RGB_{t_2}(x_2,y_2)
\]

这种真正的跨帧、跨位置几何传播。

## 为什么重要

这直接决定“低帧率 Depth 引导高帧率 RGB”到底是在做 temporal geometry propagation，还是只是在“先把 Depth 插值成高帧率，再做普通 RGB-D fusion”。

## 当前已知

计划书提出时间戳插值、Optical Flow 引导、可学习传播/merging，并希望将低帧率几何先验传播到缺失 Depth 的 RGB 帧。但没有明确候选 Depth feature 来自哪些帧、每个 RGB Token 去哪些空间位置寻找对应、是否只用最近前后真实 Depth、是否同时使用插值 Depth、如何处理遮挡/出画/新出现区域等。

## 当前阶段原则

先把流程拆成：

```text
候选 Depth Token 从哪里来？
        ↓
跨帧 / 跨位置 correspondence
        ↓
RGB 与 Depth 如何融合？
```

`concat / MLP / gating / weighted addition` 只回答最后一步“如何融合”，不能替代前面的 correspondence 问题。具体传播方法暂不确定。

## 后续文献调研方向

- optical-flow-guided feature propagation；
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

- 所有人物；
- 用户指定的某一个主体；
- 当前对焦对象 / 焦平面对应的主体。

例如画面里两个人一前一后，如果只让前面的人保持清晰、后面的人虚化，那么“所有人物都是前景”和“只把选中的人作为前景”对应不同标签和网络任务。

## 为什么重要

这会直接影响 Alpha GT 如何构造、Alpha Head 需要什么输入、是否需要用户选择信息、多人物场景怎样处理、Renderer 如何理解前景/背景，以及用户改变焦点时是否需要重新预测 Alpha。

## 当前已知

计划书强调时序一致的前景/人物 Alpha，用于人物边缘、发丝、半透明区域 soft compositing，并提出 Depth discontinuity 为 Alpha 提供边缘提示，但没有明确 Alpha 是否针对所有人物、指定主体、对焦主体，或是否与 focus point / focus distance 有关。

## 当前阶段原则

暂时不默认：

\[
\alpha=\text{all-person mask}
\]

也不默认：

\[
\alpha=\text{selected-subject matte}
\]

先把 Alpha 的任务定义确认清楚，再讨论具体 Head 结构和训练方式。

## 后续文献调研方向

- Portrait Video Bokeh 中 Alpha / foreground matte 的定义；
- Computational Bokeh 中主体选择与对焦控制；
- Multi-person portrait matting；
- Subject-aware / target-aware matting；
- focus-conditioned rendering；
- Bokeh / depth-of-field 方法中 Alpha 的实际作用。

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

但现实中不同数据集往往只提供其中一部分。例如一套数据有 Depth，另一套有人像 Alpha，第三套有 Bokeh，这些不能直接拼接成“同场景三标签”。

同时，如果：

\[
Bokeh^{GT}
\]

完全由我们自己的基础 Renderer 生成，那么网络可能只是学习逼近这个 Renderer，不能证明学到了超出该 Renderer 的真实镜头残差或真实成像特性。

## 为什么重要

该问题会直接决定 Stage 2 三任务联合训练是否可行、三个 Head 是否能在同一 batch 同时监督、多任务 loss 和 PCGrad 是否适用，以及是否需要 synthetic data、partial-label training 或多数据源交替训练。

## 当前阶段原则

**根据最终能够获得的数据类型和标签组合，再决定 Stage 1 / Stage 2 / Stage 3 的具体训练方式。** 不先假设一定存在完整三标签数据。

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
- neural rendering 中 synthetic GT 与 real GT 的结合；
- portrait bokeh / computational photography 数据集标注方式；
- 是否存在 RGB / Depth / Alpha / Defocus GT 同场景数据；
- sim-to-real / synthetic-to-real。

## 当前结论

最终训练策略应以实际能够获得的数据和标签组合为依据，再决定是否进行完整三标签联合训练，以及如何组织不同任务的数据与监督。

---

# Q6. 时序一致性损失如何区分“错误闪烁”和“真实变化”？

## 问题描述

草案目前用 Optical Flow Warp 表达一种直观的时序一致性约束：

\[
L_{\text{temp-depth}}
=
\left\|
Warp(D_t,F_{t\rightarrow t+1})-D_{t+1}
\right\|_1
\]

但**位置对应正确，并不意味着对应位置的深度数值应该保持不变。**

例如人物向镜头靠近：

```text
t 时刻：      Depth = 3.0 m
t+1 时刻：    Depth = 2.8 m
```

即使准确找到同一个人，其真实深度也应该发生变化。如果简单要求 `Warp(D_t) ≈ D_{t+1}`，可能把正常运动和真实深度变化误判为时序不一致。

同时视频中还有遮挡、解遮挡、物体进出画面等情况，新显露区域甚至不存在有效上一帧对应点。

## 为什么重要

项目真正要抑制的是 Depth flicker、Alpha 边缘抖动、Bokeh 强度/边界不合理跳变，而不是压制人物真实前后运动、真实深度变化、形变、遮挡/解遮挡或合理焦点变化。

## 当前已知

计划书提出 Optical Flow Warp、Depth/Alpha/Bokeh 跨帧一致性、Bokeh edge jitter penalty 和 Renderer 参数 temporal low-pass，但没有进一步说明如何区分真实变化与错误 flicker、如何处理遮挡/可见性，以及 temporal loss 应比较绝对数值、残差、局部结构还是其他表示。

## 当前阶段原则

当前 `warp + L1` 公式只保留为**概念示意**，不能直接视为最终训练损失。最终实现必须允许真实运动、深度变化、遮挡与解遮挡。

## 后续文献调研方向

- temporal consistency for video depth estimation；
- motion-aware temporal loss；
- optical-flow-guided consistency；
- occlusion-aware / visibility-aware temporal loss；
- scene flow / 3D motion consistency；
- temporal consistency for video matting；
- flicker suppression without over-smoothing；
- depth change modeling under camera / object motion。

---

# Q7. 预测 Alpha 是否足以支撑最终边缘合成？

## 问题描述

草案使用经典 Alpha Compositing：

\[
I_{\text{out}}
=
\alpha F+(1-\alpha)B_{\text{blur}}
\]

其中：

- \(\alpha\)：前景覆盖比例；
- \(F\)：前景颜色；
- \(B_{\text{blur}}\)：虚化后的背景颜色。

但当前网络数据流只明确预测了 Alpha，并没有说明前景颜色 \(F\) 从哪里来、虚化背景 \(B_{\text{blur}}\) 如何构造。

尤其在发丝、半透明区域，原始 RGB 像素本身已经是：

\[
I=\alpha F+(1-\alpha)B
\]

所以：

\[
I\neq F
\]

不能简单把原始 RGB 当成纯前景颜色。

## 为什么重要

如果 Physical Renderer 需要显式前景/背景分层，那么仅预测 Alpha 可能不足以支撑高质量边界重建，尤其是发丝、半透明边缘和 color bleeding 抑制。

## 当前已知

当前草案明确 Alpha 用于 soft compositing、Depth 用于控制模糊、Renderer 会生成虚化结果，但没有明确 Renderer 是否显式分离 foreground/background color，是否需要 foreground color estimation / background reconstruction，以及边界混色区域如何处理。

## 当前阶段原则

先作为 Renderer / Matting 交界处的开放问题记录，**不立即增加新的输出 Head**，也不提前规定必须预测 foreground/background color。

## 后续文献调研方向

- image / video matting 中 foreground color estimation；
- foreground-background decomposition；
- compositional matting；
- portrait bokeh / defocus rendering 边界颜色处理；
- layered rendering；
- alpha-aware background blur；
- hair / semi-transparent boundary reconstruction；
- decontamination / color decontamination；
- halo / color bleeding suppression。

---

# Q8. Depth Head 输出的“深度”具体是什么表示？

## 问题描述

草案将 Depth Head 输出描述为高精度稠密深度，并进一步用于：

\[
Depth\rightarrow CoC / Blur\ Radius
\]

但还缺少一个关键约定：Depth Head 输出的是**真实物理距离**还是**相对深度 / 排序关系**？

### 情况 A：真实物理距离

例如：

\[
D(x,y)=2.0\ \text{m}
\]

可以直接与对焦距离、焦距、光圈、薄透镜 / CoC 模型建立明确物理关系。

### 情况 B：相对深度

只保证：

\[
D_A<D_B
\]

表示 A 比 B 更近，但数值不一定对应真实米制距离，不能直接把数值当作米输入物理虚化公式。

## 为什么重要

这会同时影响：

1. Depth Head 输出形式：metric depth、inverse depth、disparity、normalized depth、relative depth 等；
2. Depth Supervision：不同数据集可能提供真值米制深度、视差、相对深度或 scale-ambiguous depth；
3. Renderer 连接：如果只输出相对深度，还需要说明如何映射到 Renderer 所需尺度或 blur control quantity。

## 当前已知

项目强调 OPPO 双目提供粗深度先验、Depth Head 预测高帧率稠密深度、最终深度驱动物理虚化，但还不能据此确定双目输入是否保留稳定 metric scale、训练数据是否提供米制 GT，以及 Renderer 是否严格依赖真实距离。

## 当前阶段原则

**等最终获得的数据和真机深度形式后再决定。** 先确认：

1. OPPO 原始 Depth 的单位与尺度；
2. 是否具有稳定 metric scale；
3. GT Depth 的来源与精度；
4. 各训练数据的 depth representation；
5. Renderer 对绝对尺度的实际需求。

## 后续文献调研方向

- metric depth estimation；
- relative / scale-invariant depth estimation；
- stereo depth / disparity；
- inverse depth prediction；
- monocular / video depth scale ambiguity；
- depth-to-defocus / depth-to-CoC mapping；
- computational photography 中 depth calibration；
- relative depth 用于 bokeh rendering 时的尺度映射。

---

# Q9. 候选预训练主干是否适合精细的 Depth / Alpha Dense Prediction？

## 问题描述

草案列出了 Video Swin、VideoMAE、MViT、TimeSformer 等候选 Video Backbone。这些模型通常具备较强视频时空建模能力，但：

> **“能够理解视频内容”与“能够恢复逐像素精细结构”不是同一个要求。**

例如视频分类只需要识别“一个人在走路”，可以容忍部分发丝细节丢失；但 Alpha Matting 需要精确保留人物轮廓、发丝、半透明区域、亚像素边界等。

即使第一版完全关闭 Spatial Token Merging，Backbone 自身的 Patch Size、Spatial Downsampling、Hierarchical Stage 和 Feature Resolution 也可能已经压缩掉关键细节。

## 为什么重要

Backbone 会直接影响 Depth 边界精度、Alpha Matte 细节恢复、Decoder 能利用哪些空间尺度 feature、是否需要额外 multi-scale / skip features，以及最终 Bokeh 边缘质量。

因此不能只看它是不是 Video Transformer、有没有预训练权重、时序建模能力强不强，还要看不同 stage 的空间分辨率及 dense decoder 能利用哪些中间特征。

## 与 Q1 的关系

Q1 讨论**额外 Token Compression**带来的信息损失；Q9 讨论**即使完全不做 Token Merging，Backbone 自身的 patch embedding 和层级下采样是否已经损失细节**。

## 当前阶段原则

这一项适合优先通过文献调研来判断，不提前锁定具体 backbone。重点看与本项目相近的 Video Depth / Video Matting / Video Segmentation / Bokeh 工作实际用了什么主干，以及为什么这么选。

## 文献记录重点

对每篇相关论文建议记录：

```text
论文
→ Backbone
→ 预训练任务 / 预训练数据
→ 原始预训练目标是什么
→ 空间下采样倍率
→ 是否输出多尺度特征
→ Decoder / Head 如何连接主干
→ 是否利用不同分辨率特征恢复细节
→ 作者选择该 Backbone 的理由
→ 边界质量方面的实验依据
→ 时序稳定性方面的实验依据
→ 是否适合本项目
```

### 额外关注 1：预训练任务

不能只看“有没有预训练权重”，还要看主干原本学的是视频分类 / 动作识别、Masked Video Modeling、图像/视频重建、深度、分割 / Dense Prediction，还是其他自监督任务。不同预训练目标会影响空间细节、时序信息和几何先验。

### 额外关注 2：怎样连接 Decoder / Head

需要记录是否只使用最后一层 feature、是否同时用多个 stage、是否有 FPN / U-Net 式多尺度融合、是否保留浅层高分辨率 feature、Decoder 怎样逐步恢复空间分辨率、Depth / Alpha 是否共享 decoder 等。

### 额外关注 3：效果证据

不能因为论文用了某个 Backbone，就认为它适合本项目。应检查是否有与本项目真正相关的实验，例如人物边界、细小结构、发丝、Depth boundary accuracy、temporal consistency、flicker / jitter、多帧稳定性、高分辨率 dense prediction 等。如果只报告分类准确率或整体平均指标，证据是不够的。

## 后续文献调研方向

- Video Depth Estimation；
- Video Matting；
- Video Semantic / Instance Segmentation；
- Video Object Segmentation；
- Portrait Video Bokeh；
- Dense Prediction Transformer；
- hierarchical Video Transformer；
- multi-scale video backbone。

---

# Q10. 粗深度存在错误时，训练约束是否会迫使网络保留这些错误？

## 问题描述

草案同时提到高质量 Depth GT 监督、Coarse Depth 几何先验约束和 Edge-aware Loss。这里存在潜在冲突。

假设：

\[
D^{GT}=2.0\ \text{m}
\]

而粗深度误估为：

\[
D^{coarse}=3.0\ \text{m}
\]

如果同时使用：

\[
L_{GT}=|D^{pred}-D^{GT}|
\]

和：

\[
L_{coarse}=|D^{pred}-D^{coarse}|
\]

那么两个 loss 会分别要求预测接近 2 m 和 3 m，形成直接冲突。

因此需要明确：**把 Coarse Depth 作为网络输入，并不等于把它作为“正确答案”监督网络输出。**

## 为什么重要

如果错误 Coarse Depth 被过强写入 loss，可能导致网络无法纠正传感器 / 双目错误、高质量 GT 与粗深度监督产生梯度冲突、人物轮廓和遮挡区错误被保留、Confidence / Validity Mask 失去意义，甚至让 Depth Head 退化成“复制粗深度”。

## 当前已知

当前方案已经有 Validity Mask、Confidence、Coarse Depth 输入、高质量 Depth GT 和粗深度几何先验，但没有明确“粗深度几何先验约束”具体是什么、是否直接约束数值、Mask/Confidence 如何参与、低置信度区域是否取消约束，以及粗深度是否只作为输入而不形成额外监督。

## 当前阶段原则

当前先记录，不提前确定具体 loss。保留基本原则：

\[
\boxed{\text{Coarse Depth 是先验，不是天然正确的 GT}}
\]

后续再决定 Coarse Depth 在训练中承担“输入先验”“软约束”“可靠区域锚点”还是其他角色。

## 后续文献调研方向

- noisy depth supervision；
- confidence-aware depth completion；
- sparse / noisy depth guided depth estimation；
- sensor depth refinement；
- stereo depth refinement；
- uncertainty-aware loss；
- robust loss for noisy depth labels；
- depth completion 中 sparse depth 作为 input 与 supervision 的区别；
- RGB-D refinement 中如何避免复制错误先验。

---

# Q11. 三个任务共享主干是否真的优于分别处理？

## 问题描述

当前总体方案希望 Depth、Alpha、Bokeh 三个任务共享同一个视频主干 / shared representation，再分别进入任务分支。这隐含一个重要假设：

> **三个任务能够通过共享表征互相促进，并且联合训练优于完全独立的任务网络。**

但这仍需要实验验证。

不同任务对特征的要求可能不完全一致。例如：

- Depth 希望同一物体表面的几何预测平滑、稳定；
- Alpha Matting 需要保留人物轮廓、发丝、半透明区域等高频细节；
- Bokeh / Rendering 更关注最终视觉效果、虚化边界和镜头表现。

因此共享特征可能产生正迁移，也可能产生 negative transfer。

## 为什么重要

如果共享有效，可以减少重复特征提取、降低参数量和计算量，并让任务间互相利用信息；如果任务冲突明显，也可能导致某一任务性能下降、共享特征难以同时满足几何平滑与边界精细要求，并需要更多 task-specific layers。

## 与 PCGrad 的关系

草案提出使用 PCGrad 类方法缓解多任务梯度冲突，但需要明确：

> **PCGrad 只能尝试缓解梯度冲突，不能证明共享主干本身一定是最优结构。**

即使使用 PCGrad，仍需回答三个任务应该共享到哪一层、哪些模块任务专属、部分共享是否更优，以及联合训练是否真的提升最终结果。

## 当前阶段原则

不提前假设 Fully Shared Backbone 一定优于 Task-specific Networks，也不提前固定共享范围。后续应结合文献和消融实验判断：哪些层适合共享、哪些层任务专属，以及联合训练是否真实带来收益。

## 后续文献调研方向

- Multi-task Learning；
- hard parameter sharing；
- soft parameter sharing；
- partially shared encoder；
- shared backbone + task-specific decoder；
- cross-task feature interaction；
- negative transfer；
- gradient conflict；
- PCGrad / GradNorm / uncertainty weighting；
- Depth + Segmentation / Matting / Rendering 联合任务。

对每篇相关论文建议记录：

```text
论文
→ 哪些任务联合训练
→ 共享哪些层
→ 哪些层任务专属
→ 是否有 cross-task connection
→ 是否使用 PCGrad / GradNorm 等冲突缓解方法
→ 与单任务模型相比是否提升
→ 哪些任务提升、哪些任务下降
→ 参数量 / FLOPs 是否降低
→ 最终作者如何证明共享有效
```

## 后续验证思路

后续可考虑比较：

```text
A：三个任务完全独立
B：共享 Backbone，Head 独立
C：部分共享 + Task-specific Decoder
D：共享 Backbone + Cross-task Connection
E：D + PCGrad / 其他梯度冲突缓解
```

以验证共享是否真正有效。

## 当前结论

> **“三个任务共享主干会互相促进”目前仍然是需要验证的研究假设。PCGrad 可以作为缓解梯度冲突的工具，但不能替代对共享结构本身的消融验证。后续应重点调研已有多任务方法共享哪些层、保留哪些任务专属模块，以及它们通过什么实验说明联合训练确实优于单任务处理。**

---

# 后续记录规则

之后发现新的实现问题，继续按以下格式追加：

```text
Qx. 问题标题

1. 问题描述
2. 为什么重要
3. 当前已知 / 计划书是否明确
4. 当前阶段原则
5. 后续文献调研方向
```

对于尚未调研过的实现问题：

> **优先记录，不急于提出具体方案。**

后续统一进入文献调研阶段，再逐项补充：

```text
相关论文
→ 别人的解决方式
→ 优缺点
→ 是否适合本项目
→ 最终采用方案
```
