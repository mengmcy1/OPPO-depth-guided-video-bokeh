# OPPO Depth-Guided Video Bokeh
## 技术路线补充理解与实施草案

> 本文用于补充《OPPO影像课题提报模板_给学生.pdf》中“低帧率深度引导的端到端视频虚化统一框架”的技术理解、模块数据流、训练方式与当前待确认的实现细节。
>
> **说明**
>
> - **计划书已明确**：表示原始项目申请书中已经明确提出的设计。
> - **当前理解 / 实现建议**：表示根据计划书和当前讨论形成的落地思路，尚未通过实验确定，不应视为最终方案。
> - 本文优先梳理“为什么这么做”和“模块之间怎么连接”，暂不提前锁定具体 backbone、dataset、loss 权重和工程参数。
>
> **当前状态更新（2026-09-30）**
>
> 本文以下内容继续作为“原始低帧率深度方案的技术理解与候选设计”保留，不删除也不改写历史来源。经当前阶段讨论，低帧率粗深度已不再作为必须坚持的核心创新方向；现阶段主线调整为：**公开 baseline 复现与扩充（短期加入 MagicBokeh） → Any-to-Bokeh 时间/显存瓶颈分析 → 特定失败场景分析 → 基于证据确定 Ours 的针对性改进**。因此本文中的低帧率 Depth propagation、三头结构、Token Merging 等不再代表已经冻结的 Ours 架构。当前有效执行顺序以 `03_实验记录/当前实验进度与方法论.md` 为准。

---

# 1. 总体任务定义

## 1.1 输入

### 计划书已明确

系统输入为：

- 高帧率 RGB 视频；
- OPPO 手机双目系统同步记录的低帧率、低精度、边界粗糙深度；
- 粗深度对应的有效性信息与置信度信息。

训练时可将一段视频表示为：

\[
RGB \in \mathbb{R}^{B\times T\times 3\times H\times W}
\]

其中：

- \(B\)：Batch Size，一次训练包含的视频 clip 数量；
- \(T\)：每个 clip 包含的视频帧数；
- \(H,W\)：图像空间分辨率。

对应粗深度可表示为：

\[
D_{\text{coarse}}\in\mathbb{R}^{B\times T\times 1\times H\times W}
\]

真实手机 Depth 原始帧率低于 RGB，上式中的 \(T\) 表示经过输入侧时间对齐后的形式。

## 1.2 输出

统一网络需要产生三类中间输出：

\[
D_{\text{dense}}
\]

高帧率、高精度稠密深度；

\[
\alpha
\]

时序一致的前景 Alpha Matte；

\[
\Delta B
\]

用于补偿物理虚化近似误差的神经虚化残差。

最终通过物理虚化渲染层得到：

\[
B_{\text{final}}
\]

即最终视频 Bokeh 结果。

---

# 2. 双模态 Token 化与低帧率深度先验注入

## 2.1 RGB Token 化

### 计划书已明确

RGB 帧通过 Patch Embedding 编码为 RGB Tokens，并保留高帧率 RGB 全帧信息。

对于单帧：

\[
I_t\in\mathbb{R}^{3\times H\times W}
\]

将图像划分为 \(P\times P\) patch。单个 patch：

\[
p_i\in\mathbb{R}^{3\times P\times P}
\]

展平后：

\[
p_i\in\mathbb{R}^{3P^2}
\]

经过 Patch Embedding：

\[
r_i=W_rp_i+b_r
\]

得到 RGB Token：

\[
r_i\in\mathbb{R}^{C}
\]

其中 \(C\) 为统一 token feature dimension，例如 512，但具体取值尚未确定。

### 当前待确定

- Patch Size \(P\)
- Token Dimension \(C\)
- 是否直接沿用预训练 Video Transformer 的 Patch Embedding

## 2.2 低帧率 Depth 的时间对齐

假设：

```text
RGB:    I1  I2  I3  I4  I5  I6  I7
Depth:  D1          D4          D7
```

输入网络前，可先将低帧率 Depth 在时间维形式上对齐至 RGB 帧率：

```text
RGB:    I1  I2  I3  I4  I5  I6  I7
Depth:  D1  D2' D3' D4  D5' D6' D7
```

这里 \(D_2',D_3',D_5',D_6'\) 只是输入侧的初步对齐结果，并不是最终高质量深度。

真正的“深度高帧率化”仍由后续：

\[
Token\ Merging
\rightarrow
Video\ Attention
\rightarrow
Depth\ Head
\]

完成。

### 当前待确定

Depth 时间对齐方式可考虑：

- 时间线性插值；
- 最近邻；
- 光流辅助 warp；
- 其他 learned interpolation。

计划书没有规定具体实现。

### 关键澄清：初步时间对齐 ≠ 精细运动对应

这一阶段首先解决的是：

> **每个 RGB 时间点是否存在一个可供后续网络使用的 Depth 候选。**

例如：

```text
RGB:    I1   I2   I3   I4
Depth:  D1             D4
```

经过简单时间插值后得到：

```text
RGB:    I1   I2   I3   I4
Depth:  D1   D2'  D3'  D4
```

它完成的是 **time slot alignment（时间槽对齐）**。

但当物体发生运动时，某个物体在不同帧中的空间位置会改变，因此：

\[
(x,t_1)\not\leftrightarrow(x,t_2)
\]

并不一定对应同一真实物体。

所以简单插值只能回答：

> “中间帧先有一个 Depth 候选。”

后续 Token Merging / Temporal Propagation 仍需要结合 RGB 特征、时间关系、光流和置信度，进一步解决：

> “这个几何信息应该从哪一帧、哪个空间位置传播到当前 RGB Token。”

因此两个阶段不要混为一谈：

\[
\boxed{\text{初步对齐：时间轴补齐}}
\]

\[
\boxed{\text{Token-level propagation：特征级运动对应与几何传播}}
\]

---

# 3. Validity Mask 与 Confidence

## 3.1 Validity Mask

### 计划书已明确

有效性 Mask 用于标记：

- 粗深度可信区域；
- 遮挡区域；
- 深度空洞；
- 双目匹配失败位置。

可形式化为：

\[
M(x,y)=
\begin{cases}
1,&\text{存在有效粗深度}\\
0,&\text{无效/空洞/遮挡}
\end{cases}
\]

其作用是让网络知道：

> 这个位置是否存在可直接利用的几何先验。

### 当前理解

Mask 更适合作为：

- 双目深度算法已有输出；
- 或根据无效值、遮挡判断等规则预处理得到。

第一版方案中不建议额外训练一个 Mask Prediction Network。

## 3.2 Confidence

### 计划书已明确

Confidence 用于表达：

> 当前深度值即使存在，到底有多可信。

例如：

```text
平坦墙面：      confidence = 0.95
人物轮廓：      confidence = 0.45
发丝区域：      confidence = 0.20
```

计划书提出依据真机统计得到的深度误差分布进行加权。

### 当前可能实现

Confidence 可来自：

- 双目算法自带 matching confidence；
- 左右一致性检查；
- matching cost；
- 重投影误差；
- 真机 Depth 与高质量 GT Depth 的误差统计映射。

因此可将 Depth 输入理解为：

\[
X_D=[D,M,C]
\]

再进一步编码为 Depth Tokens。

### 当前原则

\[
\text{Mask：回答“有没有”}
\]

\[
\text{Confidence：回答“有的话可信多少”}
\]

## 3.3 D / M / C 如何进入 Depth Token

### 第一版推荐 Baseline

最简单直接的实现是将：

\[
D,\ M,\ C
\]

在 channel 维拼接：

\[
X_D=[D,M,C]
\]

因此单帧 Depth 输入可以表示为：

\[
X_D\in\mathbb{R}^{3\times H\times W}
\]

再与 RGB 类似按 \(P\times P\) patch 切分，并通过独立的 Depth Patch Embedding：

\[
d_i=W_Dx_i^D+b_D
\]

得到：

\[
d_i\in\mathbb{R}^{C}
\]

可以简单理解为：

```text
Depth ───────┐
Mask ────────┼─→ Channel Concat → Depth Patch Embedding → Depth Token
Confidence ──┘
```

这种方式结构最简单，适合作为第一版 baseline。

### 后续可进一步研究

Mask 与 Confidence 也可以不直接作为普通图像通道，而用于显式控制 Depth Feature：

\[
f_D'=M\cdot C\cdot f_D
\]

此时：

- Mask 决定某个几何信息能否参与；
- Confidence 决定它以多大权重参与。

还可以分别编码：

\[
D\rightarrow E_D
\]

\[
(M,C)\rightarrow E_{\text{aux}}
\]

再：

\[
DepthToken=Fuse(E_D,E_{\text{aux}})
\]

这些属于后续可比较的设计方案，目前计划书没有锁定。

### 插值 Depth 的 Confidence

还应特别区分：

```text
D1     D2'     D3'     D4
真实   插值     插值     真实
```

\(D_2',D_3'\) 没有直接传感器观测，因此其 Confidence 原则上不应默认与真实 Depth 帧完全相同。

后续可以把：

- 距离最近真实 Depth 帧的时间间隔；
- optical-flow warp 的一致性；
- 插值前后深度差异；

作为 interpolated Depth confidence 的附加依据。

这一点属于当前实现建议，需要后续实验验证。

---

# 4. RGB / Depth 跨模态投影与位置编码

## 4.1 跨模态投影

RGB 与 Depth 的原始特征语义不同，因此分别进行可学习投影：

\[
\hat r_i=W_rr_i+b_r
\]

\[
\hat d_i=W_dd_i+b_d
\]

使二者最终具有相同特征维度：

\[
\hat r_i,\hat d_i\in\mathbb{R}^{C}
\]

这些 projection layer 一般不需要单独标签，其参数通过最终 Depth / Alpha / Bokeh / Temporal Loss 端到端反向传播学习。

其目的不是保持原特征完全不变，而是学习：

> 将 RGB 和 Depth 转换为适合共同融合的表征。

### 为什么 RGB 和 Depth 画面大小相同，仍需要独立 Embedding / Projection？

这里需要区分：

\[
\boxed{\text{Spatial Size}}
\]

和：

\[
\boxed{\text{Feature Dimension}}
\]

RGB 与 Depth 可以具有相同的：

\[
H\times W
\]

空间分辨率，但它们的原始 feature 语义并不相同。

例如只使用单通道 Depth 时，同样一个 \(P\times P\) patch：

RGB 原始展开维度为：

\[
3P^2
\]

而 Depth 为：

\[
P^2
\]

如果采用前面的：

\[
[D,M,C]
\]

三通道方案，那么二者原始 patch 数值维度甚至可以同为：

\[
3P^2
\]

但仍然建议使用不同的 Embedding 参数：

\[
W_{RGB}\neq W_D
\]

因为：

- RGB 表示颜色、纹理和语义；
- Depth / Mask / Confidence 表示几何、有效性和可信程度。

所以“统一维度”的真正含义不是让两张图大小一样，而是让两种模态最终都映射到同一个 token feature space：

\[
r_i,d_i\in\mathbb{R}^{C}
\]

### 一个实现上的简化

如果 RGB Patch Embedding 与 Depth Patch Embedding 已经都直接输出 \(C\) 维 token，那么：

> Patch Embedding 本身就可以同时承担“token 化 + feature dimension 投影”的作用。

此时未必还需要额外再堆一个专门的“小网络”把二者从不同维度投影到同维度。

是否增加额外 modality projection，可以作为后续消融实验决定。

## 4.2 共享位置编码

每个 token 天然具有：

\[
(x,y,t)
\]

其中：

- \(x,y\)：patch 的空间索引；
- \(t\)：视频帧索引。

位置编码可表示为：

\[
PE(x,y,t)\in\mathbb{R}^{C}
\]

一种直观实现是：

\[
PE(x,y,t)=E_x(x)+E_y(y)+E_t(t)
\]

然后：

\[
r'_{x,y,t}=r_{x,y,t}+PE(x,y,t)
\]

\[
d'_{x,y,t}=d_{x,y,t}+PE(x,y,t)
\]

位置编码通常不是把 512 维 token 变成 515 维，而是将 \((x,y,t)\) 编码为同维向量，再与 token 相加。

RGB 和 Depth 对应位置共享同一位置编码，使模型更容易建立：

\[
RGB_{x,y,t}\leftrightarrow Depth_{x,y,t}
\]

的跨模态对应关系。

---

# 5. Token Merging

计划书将 Token Merging 作为核心模块，并赋予两个不同作用：

1. 时间维：低帧率几何先验向高帧率 RGB 序列传播；
2. 空间维：压缩冗余 token，降低 Video Attention 计算量。

## 5.0 与前面“简单 Depth 上采样”的区别

这里不应把 Token Merging 简单理解成：

> “前面已经把 Depth 插值到 RGB 帧率，现在再对齐一次。”

前面的时间上采样主要解决：

\[
D_1,D_4
\rightarrow
D_1,D_2',D_3',D_4
\]

即 **时间采样密度问题**。

而 Token Merging 中的 temporal propagation 更关注：

\[
\boxed{\text{几何信息应该传播到哪个 RGB Token}}
\]

例如人物从第 1 帧的：

\[
x=30
\]

运动到第 2 帧：

\[
x=36
\]

简单时间插值并不会自动保证：

\[
Depth_{t=1,x=30}
\rightarrow
RGB_{t=2,x=36}
\]

因此还需要结合：

- RGB feature similarity；
- 时间戳；
- Optical Flow；
- Mask；
- Confidence；

进行 feature-level 几何传播。

所以这里更准确的理解是：

\[
\boxed{\text{Temporal Depth Propagation / Fusion}}
\]

而不是仅仅“再次做 RGB–Depth 尺寸对齐”。

## 5.1 时间维：几何先验传播

例如：

```text
RGB Tokens:    R1  R2  R3  R4
Depth Tokens:  D1          D4
```

目标不是直接在 Token Merging 中输出：

```text
D1 D2 D3 D4
```

而是使 \(R_2,R_3\) 在 feature level 提前获得来自 \(D_1,D_4\) 的几何信息。

可以抽象为：

\[
\tilde d_{t,i}
=
\sum_jw_{t,i,j}d_j
\]

再：

\[
f_{t,i}=Fuse(r_{t,i},\tilde d_{t,i})
\]

其中 \(w_{t,i,j}\) 表示某个 Depth Token 对当前 RGB Token 的贡献大小。

## 5.2 融合权重怎么学习

### 当前理解 / 实现示例

可以设计一个轻量融合模块：

\[
w_{ij}
=
g(r_i,d_j,c_j,m_j,\Delta t,p_i,p_j,\text{flow})
\]

输入可以包含：

- RGB Token；
- Depth Token；
- Confidence；
- Validity Mask；
- 时间距离；
- 空间位置；
- Optical Flow。

输出：

\[
w_{ij}\in[0,1]
\]

表示 Depth Token \(j\) 应向 RGB Token \(i\) 注入多少几何信息。

例如：

\[
\tilde d_i=\sum_jw_{ij}d_j
\]

然后：

\[
f_i=r_i+\tilde d_i
\]

或：

\[
f_i=MLP([r_i,\tilde d_i])
\]

### 当前待确定

具体融合方式可以是：

- concat + MLP；
- gated addition；
- cross-attention；
- learned token matching；
- ToMe 风格 merge。

计划书目前没有锁定具体结构。

## 5.3 时间传播中的光流

当人物运动时：

\[
\text{same position}\neq\text{same object}
\]

因此不能仅按同一 \((x,y)\) 位置传播 Depth。

Optical Flow 用于描述：

\[
F_{t\rightarrow t+1}(x,y)=(u,v)
\]

即该位置从第 \(t\) 帧移动到下一帧的位移。

可利用光流将已有 Depth Token 的几何信息传播到对应运动位置，再结合时间距离进行加权。

计划书明确提出：

> 时间戳插值 + 光流引导加权。

## 5.4 空间维：冗余 Token 压缩

平坦背景区域可能存在大量相似 Token：

\[
t_1\approx t_2\approx t_3\approx t_4
\]

参考 Token Merging（ToMe）思想：

\[
t_1,t_2,t_3,t_4\rightarrow\bar t
\]

将相似 token 聚合，保留代表性“质心”表征。

使 token 数量：

\[
N\rightarrow N',\qquad N'<N
\]

从而降低后续 Video Attention 的计算和显存需求。

需要区分：

\[
\boxed{\text{时间传播：补信息}}
\]

\[
\boxed{\text{空间压缩：省计算}}
\]

## 5.5 当前推荐的模块拆分方式

虽然计划书把两部分统称为 Token Merging，但第一版实现时不建议强行让一个“小网络”同时承担全部功能。

更容易实现和验证的结构是拆成两个连续子模块：

```text
RGB Tokens             Depth Tokens
     \                    /
      \                  /
       Temporal Depth Propagation
       / Cross-modal Fusion
                ↓
       Geometry-aware Tokens
                ↓
          Spatial ToMe Merge
                ↓
        Fewer Fused Tokens
                ↓
          Video Transformer
```

### 子模块 A：Temporal Depth Propagation / Cross-modal Fusion

输入可包含：

\[
RGB\ Tokens,\ Depth\ Tokens,\ M,\ C,\Delta t,\ Flow
\]

输出：

\[
F_{\text{geo-aware}}
\]

即已经带有几何先验的高帧率视觉 tokens。

第一版可从最简单的：

- concat + MLP；
- gated addition；

开始。

再逐步比较：

- cross-attention；
- flow-guided token matching；
- learned propagation。

### 子模块 B：Spatial Token Merging

对子模块 A 的输出计算 token similarity，将平坦背景等高冗余区域合并。

其目标单纯是：

\[
N\rightarrow N'
\]

降低后续 Video Attention 的 sequence length。

### 为什么推荐拆开？

因为两个子任务的目标不同：

\[
\text{Propagation}
\rightarrow
\text{提高几何信息利用率}
\]

而：

\[
\text{Merging}
\rightarrow
\text{降低计算冗余}
\]

拆开之后更容易：

- 单独训练和调试；
- 做 ablation；
- 判断性能提升究竟来自哪一个模块；
- 避免一个“万能融合网络”同时承担过多目标。

---

# 6. Video Attention 时空建模

Token Merging 后得到：

\[
F_{\text{merged}}
\]

再输入 Video Attention。

计划书采用：

\[
\text{帧内窗口 Spatial Attention}
+
\text{帧间 Temporal Attention}
\]

并结合 sliding window 处理长视频。

## 6.1 帧内窗口 Spatial Attention

每一帧内部划分局部窗口，每个窗口内部进行 Self-Attention。

主要学习：

- 局部纹理；
- 人体结构；
- 前景边界；
- 深度变化；
- 局部语义关系。

窗口化 Attention 主要是为了降低高分辨率视频的计算成本。

## 6.2 帧间 Temporal Attention

Spatial Attention 后，再建模不同帧之间的关系。

需要特别注意：

\[
s_{1,5},s_{2,5},s_{3,5}
\]

只表示第 1、2、3 帧中相同空间索引的 token，并不保证来自同一个真实物体。

因此后续实际设计需要研究：

- temporal attention 是否仅连接同位置 token；
- 是否扩大 temporal search region；
- 是否结合 optical flow；
- 是否先做 motion alignment。

计划书尚未规定这些细节。

## 6.3 Video Attention 输出

Video Attention 输出不是 Depth、Alpha 或 Bokeh，而是：

\[
F_{\text{shared}}
\]

即多帧时序连贯共享表征。

可抽象为：

\[
F_{\text{shared}}
\in
\mathbb{R}^{B\times T\times N\times C}
\]

具体 Tensor 形态取决于最终 backbone。

随后：

\[
F_{\text{shared}}
\rightarrow
\begin{cases}
Head_{\text{depth}}\\
Head_{\alpha}\\
Head_{\text{bokeh}}
\end{cases}
\]

---

# 7. Backbone 选择与训练方式

## 7.1 计划书已明确

计划书要求：

- Video Attention 时空建模；
- 三任务共享 encoder / representation。

但没有规定：

- 必须从零训练；
- 必须使用某一个 Video Transformer；
- 必须全量 fine-tune。

## 7.2 当前建议

第一版优先采用已有预训练 Video Backbone，例如候选：

- Video Swin Transformer；
- VideoMAE；
- MViT；
- TimeSformer。

RGB / Depth 的 token 化、投影、融合与 Token Merging 仍需要针对本项目进行适配。

### Stage A：冻结大部分 Backbone

先训练：

- RGB / Depth Projection；
- Token Merging；
- 三个 Head；
- Backbone 最后少量 blocks。

目标是先验证整个 pipeline 是否成立，减少训练成本与不稳定因素。

### Stage B：部分解冻

若发现预训练 Video Backbone 难以适应：

\[
RGB+Depth
\]

融合后的特征分布，则逐步解冻后 1/3 或后 1/2 的 Video Attention blocks。

### Stage C：Full Fine-tuning

如果数据量足够、loss 稳定且算力允许，再尝试全量 fine-tune。

### 当前原则

是否冻结多少层不提前写死，应由验证实验决定。

第一版不建议直接从零训练完整 Video Transformer。

---

# 8. 三头解耦输出

## 8.1 Depth Head

输出：

\[
D^{pred}
\]

即与 RGB 帧率一致的高帧率稠密深度。

训练目标包含：

- GT Depth 数值约束；
- 粗糙 Depth 几何先验约束；
- Depth Edge / Edge-aware Loss。

最终 Bokeh 对 foreground/background depth discontinuity 非常敏感。

深度边界偏移可能直接造成：

- halo；
- color bleeding；
- foreground erosion。

## 8.2 Alpha Head

输出：

\[
\alpha(x,y)\in[0,1]
\]

用于描述像素属于前景的覆盖比例。

重点处理：

- 发丝；
- 毛发；
- 半透明区域；
- 运动模糊边缘。

计划书采用：

\[
L_1+L_{\text{Laplacian}}
\]

进行监督。

## 8.3 Depth Edge 对 Alpha Head 的提示

计划书提出：

> 将 Head_depth 的深度不连续图作为 Alpha Head 的边缘提示。

可以先从：

\[
D^{pred}
\]

提取：

\[
E_D=|\nabla D^{pred}|
\]

得到 Depth Discontinuity Map。

再用于：

\[
AlphaHead(F_{\text{shared}},E_D)
\rightarrow\alpha
\]

其含义是：

> Depth Head 告诉 Alpha Head 哪些区域很可能存在几何前后景边界。

### 当前待确定

Depth Edge 的注入方式可以考虑：

- concat；
- gating；
- cross-attention；
- auxiliary edge feature；
- edge-alignment loss。

### 与 SLIDE 的联系

这一思想与 SLIDE 中利用 depth discontinuity 建模前后景边界、再利用 alpha matte 补充发丝等细结构的动机相关。

区别在于：

> SLIDE 更偏向在 soft layering 阶段融合 depth 与 alpha；

而本项目希望进一步在多任务网络中，让 Depth Edge 直接辅助 Alpha Prediction。

## 8.4 Bokeh Head

Bokeh Head 暂时理解为：

\[
Head_{\text{bokeh}}\rightarrow\Delta B
\]

即 Neural Bokeh Residual。

不是直接从零生成：

\[
B_{\text{final}}
\]

而是采用：

\[
B_{\text{final}}
=
B_{\text{physical}}
+
\Delta B
\]

的 Base + Delta 思路。

其目标是补偿显式物理模型难以精确模拟的部分，例如：

- 高光弥散；
- 口径蚀；
- 复杂镜头残余效应。

---

# 9. 物理虚化渲染层

## 9.1 基本定义

物理 Renderer 不是额外的大型神经网络，而更接近：

\[
B_{\text{physical}}
=
R(I,D,\alpha,\theta)
\]

其中：

- \(I\)：RGB；
- \(D\)：Depth；
- \(\alpha\)：Alpha；
- \(\theta\)：镜头参数；
- \(R\)：显式物理虚化渲染函数。

## 9.2 Depth → CoC / Blur Kernel

Depth 决定：

> 每个像素应该虚化多少。

根据薄透镜模型：

\[
D(x,y)
\rightarrow
r(x,y)
\]

其中 \(r(x,y)\) 可以理解为该像素对应的 CoC（Circle of Confusion）/ blur radius。

不同深度对应不同模糊程度。

## 9.3 Camera Parameters

镜头参数包括：

- 光圈；
- 焦距；
- 对焦距离；
- 光圈叶片形状。

这些参数共同决定：

- 模糊强度；
- blur kernel shape；
- 焦外光斑形状。

因此物理 Bokeh 与简单 Gaussian Blur 不同。

## 9.4 Depth Layering

直接在完整二维图像上使用大 blur kernel 容易导致：

- 前后景颜色渗漏；
- 人物边缘 halo；
- 发丝污染。

因此可利用 Depth 将场景分层：

\[
D\rightarrow L_1,L_2,\ldots,L_K
\]

分别处理不同深度层，再进行合成。

## 9.5 Alpha Soft Matting

Alpha 用于控制：

> 清晰前景与虚化背景在边缘处如何平滑融合。

基本合成关系：

\[
I_{\text{out}}
=
\alpha F+(1-\alpha)B_{\text{blur}}
\]

对发丝等区域：

\[
0<\alpha<1
\]

可以避免二值 mask 带来的剪纸感与边缘穿帮。

## 9.6 Neural Residual

最终：

\[
B_{\text{final}}
=
B_{\text{physical}}
+
\Delta B
\]

其中：

- \(B_{\text{physical}}\)：物理上基本正确；
- \(\Delta B\)：神经网络补充真实镜头残余效应。

该结构与常见的 Base + Delta / residual learning 思想一致。

## 9.7 可微渲染

### 当前建议

Renderer 最好尽可能设计为 differentiable。

这样：

\[
L_{\text{render}}
\]

可以通过 Renderer 回传到：

\[
Depth\ Head
\]

和：

\[
Alpha\ Head
\]

使网络最终优化目标不只是“Depth 数值准确”，而是“Depth / Alpha 能产生更好的最终 Bokeh”。

---

# 10. 时序一致性保障

计划书从三个层次约束 temporal consistency。

## 10.1 Feature Level

Video Attention 在共享 feature 层建立：

\[
F_t\leftrightarrow F_{t+1}
\]

的时序关系。

使三个 Head 都从统一的时空 representation 出发。

## 10.2 Output Level

利用 Optical Flow Warp：

\[
Warp(D_t,F_{t\rightarrow t+1})
\]

与：

\[
D_{t+1}
\]

比较：

\[
L_{\text{temp-depth}}
=
\|
Warp(D_t)-D_{t+1}
\|_1
\]

Alpha 和 Bokeh 同理。

同时增加虚化边缘抖动惩罚。

## 10.3 Renderer Level

即使 Depth 只有轻微帧间噪声，也可能造成 blur radius 不断变化。

因此可对：

- CoC；
- Blur Kernel Radius；
- 其他 renderer parameter；

增加 temporal low-pass constraint。

例如：

\[
\bar r_t
=
\beta\bar r_{t-1}
+
(1-\beta)r_t
\]

减少 Bokeh breathing / 虚化边缘呼吸效应。

## 10.4 当前主干数据流的统一理解

截至目前，可以将完整前向过程概括为：

```text
High-FPS RGB
        +
Low-FPS Depth + Mask + Confidence
        ↓
Depth 时间轴初步对齐
        ↓
RGB / Depth Patch Embedding
        ↓
统一 Token Feature Space
+ Shared (x, y, t) Position Encoding
        ↓
Temporal Depth Propagation / Cross-modal Fusion
        ↓
Spatial Token Merging
        ↓
Video Transformer
Spatial Attention + Temporal Attention
        ↓
Shared Video Representation
      /         |          \
 Depth Head   Alpha Head   Bokeh Head
     ↓           ↓            ↓
     D           α            ΔB
      \          |           /
       RGB + Camera Parameters
                ↓
      Differentiable Physical Renderer
                ↓
          Physical Bokeh
                +
               ΔB
                ↓
           Final Bokeh
```

其中需要特别记住：

\[
\boxed{\text{Depth 不是直接等于 Blur Radius}}
\]

而是：

\[
Depth
\rightarrow
CoC / Blur\ Kernel
\rightarrow
\text{虚化大小}
\]

Alpha 主要负责：

\[
\text{前景/背景边界的软归属与合成}
\]

Bokeh Head 在当前详细方案中更适合理解为：

\[
\Delta B
\]

即对物理渲染结果的 neural residual correction。

---

# 11. 整体训练策略

## 11.1 Stage 1：Depth / Alpha 基础能力预训练

### 计划书已明确

使用公开数据分别预训练：

- Depth Head；
- Alpha Head；

并共享 Encoder。

### 当前理解

第一阶段不一定需要将整个：

\[
TokenMerging
+
VideoAttention
+
3Heads
\]

全部从零共同训练。

原因：

- Depth 与 Matting 公共数据集未必统一；
- 很多 Matting 数据不是完整视频；
- 公共数据未必具备真实低帧率手机双目 Depth；
- 从零训练 Video Transformer 成本较高。

### 当前建议

优先：

```text
Pretrained Video/Image Backbone
          ↓
冻结大部分 Backbone
          ↓
训练对应 Head + Projection + 部分高层模块
```

随后根据验证结果逐步解冻。

## 11.2 Stage 2：三任务联合训练

使用同一场景同时具有：

\[
RGB
+
Depth^{GT}
+
Alpha^{GT}
+
Bokeh^{GT}
\]

的数据。

完整启用：

\[
Tokenization
\rightarrow
TokenMerging
\rightarrow
VideoAttention
\rightarrow
3Heads
\]

总 Loss 可抽象为：

\[
L
=
\lambda_dL_{\text{depth}}
+
\lambda_\alpha L_{\alpha}
+
\lambda_bL_{\text{bokeh}}
+
\lambda_tL_{\text{temporal}}
\]

计划书还提出：

> 使用 PCGrad 类梯度手术缓解多任务梯度冲突。

## 11.3 Stage 3：全链路端到端微调

完整链路：

```text
High-FPS RGB + Low-FPS Coarse Depth
                ↓
       Dual-modal Tokenization
                ↓
          Token Merging
                ↓
          Video Attention
                ↓
        Shared Representation
          /       |       \
      Depth     Alpha      ΔB
          \       |       /
           Physical Renderer
                ↓
           Final Bokeh
```

最终：

\[
L_{\text{render}}
\]

能够沿：

\[
Renderer
\rightarrow
Depth/Alpha
\rightarrow
VideoAttention
\rightarrow
TokenMerging
\]

回传。

此阶段模型的评价重点从单个子任务进一步转向最终视频虚化质量。

---

# 12. 当前尚未确定的问题

以下内容在计划书中没有明确，应作为后续方案设计与 baseline 实验的重点：

1. RGB / Depth 的真实帧率比例；
2. 输入 clip 长度 \(T\)；
3. Patch Size；
4. Token Feature Dimension；
5. Depth 时间上采样方式；
6. Validity Mask 的具体生成规则；
7. Confidence 的具体计算方式；
8. RGB / Depth Projection 结构；
9. RGB / Depth Token 的具体融合算子；
10. Token Merging 时间传播算法；
11. Optical Flow 在 Token Merging 中的具体使用方式；
12. Token Spatial Compression Ratio；
13. ToMe 是否直接复用或需要修改；
14. Video Attention Backbone 选择；
15. Temporal Attention 的运动对齐方案；
16. Backbone Stage 1 冻结范围；
17. Stage 2 Backbone 解冻范围；
18. 是否进行 Full Fine-tuning；
19. Depth Edge 的提取方式；
20. Depth Edge 向 Alpha Head 的注入方式；
21. Depth Loss 与 Edge-aware Loss 设计；
22. Alpha Loss 具体组成；
23. Bokeh Residual 的表示方式；
24. Physical Renderer 的具体数学模型；
25. Renderer 是否完全可微；
26. CoC / Depth Layering 的实现方式；
27. Temporal Loss 设计；
28. Optical Flow 来源；
29. 多任务 Loss 权重；
30. PCGrad 是否真正需要；
31. 三标签训练数据如何构造；
32. 真机 Coarse Depth 退化模型；
33. 最终 Dataset / Split / Metrics；
34. 效果与效率 baseline。

---

# 13. 当前最小实施路线建议

在正式创新前，优先解决：

```text
Step 1
确定一个可复现 Video Backbone
        ↓
Step 2
完成 RGB / Depth Token 输入接口
        ↓
Step 3
实现最简单的跨模态 Fusion Baseline
        ↓
Step 4
跑通 Depth Head
        ↓
Step 5
增加 Alpha Head
        ↓
Step 6
实现基础 Differentiable Bokeh Renderer
        ↓
Step 7
加入 Bokeh Residual
        ↓
Step 8
最后再研究 Token Merging 和 Temporal Consistency 创新
```

原则：

> 先建立一个能训练、能验证、结构简单的完整 baseline，再逐个验证计划书中的核心创新模块。

目前最值得重点研究的两个方向：

1. **低帧率 Depth Token 如何有效传播到高帧率 RGB 序列；**
2. **Depth / Alpha / Bokeh 三任务如何在共享视频表征中互相促进，而不是发生负迁移。**

这两个问题比一开始重新设计完整 Video Transformer 更接近本课题真正的研究重点。
