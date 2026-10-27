# Elastic3D：Controllable Stereo Video Conversion with Guided Latent Decoding

> 论文技术笔记。以论文正文及 Supplementary Material 为主要依据；涉及原始 Stable Video Diffusion（SVD）配置时单独注明。  
> 论文：Nando Metzger、Prune Truong、Goutam Bhat、Konrad Schindler、Federico Tombari。  
> 项目主页：https://elastic3d.github.io/  
> 对应正文：Sections 1–7；补充材料：Sections A–H。

## 0. 整体数据流（先看这一部分）

左眼视频 $V_L$ 首先经过**冻结的 VAE Encoder**，编码为左眼潜变量 $z_L$；同时，左眼参考图像经过 **CLIP 的 ViT 图像编码器**，得到图像语义条件 Embedding。另一边，左眼原始视频还会进入一个由 VAE Encoder 初始化的 **Extractor $G$**，提取多个分辨率的引导特征，供最后的 Decoder 使用。

将 $z_L$ 与**同形状的全零潜变量**沿通道维拼接，组成 U-Net 的输入；用户设定的视差强度 $\delta$ 经过数值缩放和广播，形成视差 Token $\tau(\delta)$，再与 CLIP 图像条件 Token 拼接，作为 U-Net Transformer 的条件输入。

随后，数据先经过 **U-Net 的 CNN 下采样和特征提取**，在各级模块中通过**空间 Self-Attention、条件 Cross-Attention 和时间 Attention/时间卷积**进行空间与帧间建模，最后通过 U-Net 上采样，在**一次前向传播**后输出右眼潜变量 $\hat z_R$。这里不执行通常的多次迭代去噪，也不显式估计深度和进行右眼 Warping。

最后，$\hat z_R$ 进入 **Guided VAE Decoder $D'$** 逐级上采样；在每个对应尺度，Decoder 当前的右眼特征作为 **Query**，Extractor 提供的左眼同一水平行的特征作为 **Key/Value**，执行 **Epipolar Cross-Attention（极线交叉注意力）**，把匹配到的原图细节通过残差方式注入 Decoder，最终输出右眼 RGB 视频 $\hat V_R$。

```text
输入：左眼 RGB 视频 V_L                  用户设定视差 δ
         │                                   │
         ├─────────────┐                     └─ 缩放、广播 → τ(δ) ─┐
         │             │                                            │
         ▼             ▼                                            │
  冻结 VAE Encoder   CLIP 图像 ViT                                   │
         │             │                                            │
         ▼             ▼                                            │
   左眼 latent z_L   CLIP Embedding ─────────────── 拼接条件 Token ─┤
         │                                                          │
         ├──── 与同形状 zero latent 沿通道拼接                        │
         ▼                                                          │
   8 通道 latent 输入                                                 │
         │                                                          │
         ▼                                                          │
   Video U-Net：CNN 下采样 → Spatial/Temporal Transformer            │
                （条件 Cross-Attention 读取 CLIP 和 τ(δ)）◄──────────┘
                → CNN 上采样及输出卷积
         │
         ▼
    右眼 latent ẑ_R ────────────────────┐
                                        ▼
V_L ──→ Extractor G ─→ 多尺度左眼特征 ─→ Guided Decoder D'
                                        │
                   Epipolar Cross-Attention（同一水平行搜索）
                   + 残差注入 + 逐级上采样
                                        │
                                        ▼
                                 右眼 RGB 视频 V̂_R
```

整体公式（论文 Eq. 1）：

$$
\boxed{\hat V_R=D'\left(f_\theta(0,z_L,\tau(\delta)),V_L\right),\qquad z_L=E(V_L)}
$$

- $E$：冻结的 VAE Encoder；$f_\theta$：经单步任务训练的 SVD Video U-Net；$\tau(\delta)$：视差条件；$D'$：引导式 Decoder。
- **空间输入**：$[0,z_L]$ 沿通道拼接；**条件输入**：CLIP Embedding 和 $\tau(\delta)$ 沿条件 Token 维拼接。这是两种不同的拼接方式。
- **原始 SVD 配置与 Elastic3D 不同的地方**：原始 SVD 从随机噪声开始多步生成；Elastic3D 将生成状态设为全零，并训练为固定时间步的一步预测。上述 CLIP 视觉编码器、8 通道 U-Net 等背景细节可对照 [SVD 原始配置](https://github.com/Stability-AI/generative-models/blob/main/configs/inference/svd.yaml)；论文未给出 Elastic3D 全部模块的可执行源码。

---

## 1. 任务定义、动机与目标

### 1.1 输入和输出

$$
V_L\in\mathbb R^{N\times H\times W\times3},\qquad
\hat V_R\in\mathbb R^{N\times H\times W\times3}
$$

- 输入：经双目立体校正对应假设的左眼 RGB 视频，包含 $N$ 帧。
- 输出：同一场景、从水平偏移的右相机观察到的右眼 RGB 视频。
- 测试时用户额外指定整体视差强度标量 $\delta$（单位按图像像素理解）。
- 最终左右眼视频组合为 $(V_L,\hat V_R)$；与显式三维重建方法不同，不生成可自由渲染的三维网格或 3DGS 场景。

### 1.2 现有方法问题

传统方法：单目深度估计 $\rightarrow$ 几何 Warping $\rightarrow$ 遮挡区补全/扩散细化。问题包括单目深度误差、Warping 空洞、多阶段开销，以及潜空间扩散的 VAE 细节损失。直接生成方法（如 Eye2Eye）可以跳过显式深度和 Warping，但可能缺少目标视差强度控制，且像素空间多步扩散成本较高。（正文 §1–2）

### 1.3 五项质量要求（正文 §3）

| 要求 | 技术含义 |
|---|---|
| Geometric Correctness | 遵守极线约束、合理的相对深度顺序 |
| 3D-Effect Control | 连续、直观地控制左右眼的立体强度 |
| Stereoscopic Fidelity / Detail Preservation | 共同可见区域的纹理对应一致，减轻双眼竞争 |
| Plausible Disocclusion Handling | 右眼新显露区域内容合理，尽可能利用视频其他帧的信息 |
| Temporal Stability | 相邻帧运动与纹理尽量连续，减少闪烁和抖动 |

---

## 2. 网络结构与单步右眼潜变量生成（正文 §4.1）

### 2.1 VAE 编码与张量形状

左眼视频逐帧编码：

$$z_L=E(V_L)$$

按照原始 SVD 的 8 倍空间压缩、4 个 latent 通道设置，在 $N=16$、$H=W=512$ 的常见实验示例中：

| 张量 | 形状（忽略 batch 维） | 含义 |
|---|---|---|
| 左眼 RGB | $(16,3,512,512)$ | 原始输入 |
| 左眼 latent $z_L$ | $(16,4,64,64)$ | VAE 压缩表示 |
| 全零 latent $0$ | $(16,4,64,64)$ | 单步生成状态 |
| 拼接 $[0,z_L]$ | $(16,8,64,64)$ | Video U-Net 的通道输入 |
| 预测右眼 latent $\hat z_R$ | $(16,4,64,64)$ | U-Net 的输出 |
| 最终右眼 RGB | $(16,3,512,512)$ | Guided Decoder 的输出 |

上述通道、压缩比和网络配置来自**原始 SVD 实现及其作为 Elastic3D 主干的对应关系**；Elastic3D 论文核心公式没有逐项写出所有张量通道数。4 通道的 $64\times64$ latent 与 $512\times512\times3$ RGB 图像的元素数量比为 $1:48$。（正文 §4.3）

### 2.2 并行图像条件：CLIP / ViT

- CLIP（Contrastive Language–Image Pretraining）**视觉**编码器用于提取参考图像的全局视觉 Embedding；不是生成视频的主网络，也不是 VAE 之后串联运行。
- 原始 SVD 配置使用 `FrozenOpenCLIPImagePredictionEmbedder`，其视觉编码器默认为 `ViT-H-14`、交叉注意力条件维度为 1024；原始配置 `n_cond_frames: 1`。不能仅根据该配置断言 Elastic3D 对每一帧单独执行一次 CLIP。
- VAE latent 通过通道拼接进入 U-Net 的 CNN；CLIP Embedding 作为外部 `context` 进入 U-Net 内的 Cross-Attention。
- 原始 SVD 还使用 FPS、Motion Bucket、Conditioning Augmentation 等额外数值条件，经向量嵌入路径注入 U-Net。**Elastic3D 论文没有逐项说明这些额外条件在其训练与推理中的具体保留/固定设置**，不将其写作已验证的 Elastic3D 设计。

原始 SVD 配置中 Video U-Net 的主要参数（用于理解骨干结构）：

```yaml
in_channels: 8
out_channels: 4
model_channels: 320
channel_mult: [1, 2, 4, 4]
num_res_blocks: 2
context_dim: 1024
video_kernel_size: [3, 1, 1]
use_spatial_context: true
```

主干可以概括为：CNN 卷积及多尺度下采样 $\rightarrow$ 空间/时间特征建模 $\rightarrow$ U-Net 跳连与上采样 $\rightarrow$ 4 通道 latent 输出。`SpatialVideoTransformer` 内部既有空间自注意力，也有读取 CLIP 等外部条件的交叉注意力，之后配合时间注意力处理视频帧间关系；因此它是 CNN + Transformer 的混合架构，而非纯 ViT。

### 2.3 一步扩散的数学形式

常规扩散的加噪公式：

$$
z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad
\epsilon\sim\mathcal N(0,I)
$$

在最大时间步 $T$，有 $\bar\alpha_T\approx0$。单步方案把噪声输入替换为其期望值 $\epsilon=0$，使初始生成状态近似为 $z_T=0$，并固定时间步为 $T$。

按照论文附录 §C 的 v-prediction 推导，在该极限下：

$$
 v_T\approx-z_R,\qquad
 \boxed{\hat z_R=-\hat v_\theta(0,T,c)}
$$

这里 $c$ 包含左眼与视差条件。正文把输出记为：

$$
\boxed{\hat z_R=f_\theta(0,z_L,\tau(\delta))}
$$

说明：不是把原始多步 SVD 直接改成一步调用就能生成右眼；作者重新训练/微调预训练 U-Net，使其适应全零状态、固定时间步、直接预测目标右眼 latent 的任务。

### 2.4 推理过程中的 Warping-Free

推理时不调用外部单目深度估计器，不先得到右眼 Warping 图像，不需要对 Warping 空洞进行显式补全。U-Net 在受条件约束下直接生成完整右眼 latent。**训练时仍可能使用双目视差估计与 Forward Warping 产生增强样本**；“Warping-Free”专指生成流水线没有显式 Warping 阶段。

---

## 3. 立体强度控制：$\delta$ 与条件 Token（正文 §4.2；附录 A.2）

### 3.1 控制量定义

从视频**第一帧**的左到右视差图 $D^0_{L\to R}$ 计算：

$$
\boxed{\delta=P_{50}(D^0_{L\to R})}
$$

$P_{50}$ 是中位数（第 50 百分位）。

- 训练时有真实左、右眼成对视频，可使用立体匹配得到视差图；论文 §6.1 使用 **FoundationStereo** 估计用于计算 $\delta$ 的视差图。论文某处称 GT disparity / GT depth，实践中需区分传感器真值与算法估计的 pseudo-GT。
- 推理时只有左眼视频时，**由用户指定 $\delta$**；不要求存在真实右眼图像或由网络预测 $\delta$。
- $\delta$ 不是通过随机初始化、随 U-Net 权重共同更新的待学习参数；待学习的是生成网络对该条件的响应。
- $\delta$ 是全局标量，不是完整的逐像素视差图，也不是物理相机基线 $B$。对于理想平行双目：$d=fB/Z$，基线变化会影响视差，但实际 $\delta$ 不能唯一决定完整几何。

### 3.2 Token 构造与注入

论文附录 A.2 给出的做法：

1. 将原始视差标量按约 $10^{-2}$ 的比例缩放，改善数值稳定性；
2. 将标量沿特征维广播为 $\tau(\delta)$；
3. 与输入帧的标准 CLIP Embedding **沿条件 Token 维拼接**；
4. 将条件送入 U-Net 的空间注意力层。

这种标量广播方式本身不需要新增可学习的 Token 投影参数。与把视差图堆成第 9 个空间通道完全不同。

### 3.3 控制精度边界

$\delta$ 是**学习得到的条件控制**，不是硬几何约束：输入 $\delta=8\,\mathrm{px}$ 不保证输出实际视差中位数严格为 8 px。即使中位数匹配，各个物体的视差分布也可能有误。作者附录 H 明确承认全局中位数条件在背景占比很大、镜头推进等场景中存在歧义。

### 3.4 其他视差统计量（附录 F.2，Table 14）

作者比较 Median、Average 和 Max；三者均可作为条件。在 iPhone 数据集上：

| 条件 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Average | 23.4 | 0.916 | 0.180 |
| Max | 22.9 | 0.902 | 0.184 |
| Median（论文默认） | 22.5 | 0.890 | 0.193 |

最终使用 Median，理由主要是稳健、对异常值不敏感、含义直观；不是因为它在每一个数据集上都最高分。

---

## 4. Guided Latent Decoder（正文 §4.3；附录 B）

### 4.1 解码瓶颈与输入

标准潜变量解码：$\hat V_R=D(\hat z_R)$；改进后：

$$\boxed{\hat V_R=D'(\hat z_R,V_L)}$$

其中：

- $D'$：由预训练 Decoder $D$ 的权重初始化；
- $G$：由预训练 VAE Encoder $E$ 初始化，从**左眼原始 RGB**提取多尺度引导特征 $\{g_i\}$；
- Decoder 在对应的上采样尺度产生特征图 $h_i$；
- 不直接按相同 $(x,y)$ 坐标复制，因为左右眼有水平视差。

### 4.2 多尺度特征

论文定义对应层的两组特征：

$$
h_i\in\mathbb R^{H_i\times W_i\times C},\qquad
g_i\in\mathbb R^{H_i\times W_i\times C}
$$

- $h_i$：右眼 Decoder 当前尺度的特征；
- $g_i$：由左眼 Extractor 提取的同尺度引导特征；
- 各尺度通过注意力注入细节，避免只有 $4\times64\times64$ latent 时丢失高频信息。

**论文给出了多尺度机制，但没有在正文逐层列出 $G$、$D'$ 的全部通道数和具体模块定义；不能把示意尺寸当作完整源码配置。**

### 4.3 Epipolar Cross-Attention 的 Q/K/V

对于立体校正后的左右眼图像，匹配点应位于**同一水平行**。右眼位置 $p=(u,v)$ 的 Query 从 Decoder 特征得到：

$$Q_p=h_i(p)W_Q\in\mathbb R^d$$

左眼特征只取第 $v$ 行 $g_i(v)$，计算 Key 和 Value：

$$
K_{\mathrm{row}}=g_i(v)W_K\in\mathbb R^{W_i\times d},\qquad
V_{\mathrm{row}}=g_i(v)W_V\in\mathbb R^{W_i\times d}
$$

沿这一行计算缩放点积注意力：

$$
\alpha_p=\operatorname{Softmax}\left(\frac{Q_pK_{\mathrm{row}}^{\mathsf T}}{\sqrt d}\right)V_{\mathrm{row}}
$$

再经过输出投影、残差写回：

$$
A_{\mathrm{epipolar}}(h_i(p),g_i)=\alpha_pW_{\mathrm{out}}
$$

$$
\boxed{h_i'(p)=h_i(p)+A_{\mathrm{epipolar}}(h_i(p),g_i)}
$$

以上对应补充材料公式 (5)–(9)。注意力允许从左眼**整条水平极线**读取候选纹理，并非先预测单一视差后精确 Warp，也不能保证每次都找到正确对应。

### 4.4 复杂度与内存

对于特征图 $H\times W$：

| 搜索方式 | 每个右眼 Query 的候选位置 | 注意力复杂度 |
|---|---:|---|
| 全局 Cross-Attention | $HW$ | $O(H^2W^2)$ |
| Epipolar Cross-Attention | $W$ | $O(HW^2)$ |

当 $H=W=512$，只考虑**单帧、单头、FP16 注意力矩阵**：

- 全局搜索：约 **128 GiB**；
- 逐行极线搜索：约 **256 MiB**。

这不是网络的总显存：真实开销还包括多头、batch、QKV、中间激活、Extractor、U-Net 等。

### 4.5 初始化

- $D'$ 用预训练 $D$ 初始化；$G$ 用预训练 $E$ 初始化。
- 新增 Epipolar Attention 的输出投影矩阵 $W_{\mathrm{out}}$ **零初始化**。
- 初始有 $\alpha_pW_{\mathrm{out}}=0$，所以 $h_i'=h_i$，不立即破坏原 Decoder 的预训练映射；训练后引导分支逐步产生贡献。

### 4.6 Decoder 的独立训练

使用**真实右眼** $V_R$，而不是第一阶段 U-Net 预测出的 $\hat z_R$：

$$
z_R=E(V_R),\qquad \tilde V_R=D'(z_R,V_L)
$$

$$
\boxed{\mathcal L_D=\mathcal L_1(\tilde V_R,V_R)+\mathcal L_{\mathrm{LPIPS}}(\tilde V_R,V_R)}
$$

- L1 和 LPIPS 等权重；
- 训练 **50,000 iterations**；学习率 **$1\times10^{-4}$**；精度 **BF16**（作者报告有助于避免新增注意力数值溢出）；
- Decoder 作为独立模块训练，可替换其他基于 latent 的生成管线的普通 Decoder；
- 论文没有逐参数说明 $G$ 的冻结策略及所有内部子模块的梯度开关，不能从文字推断具体 Parameter Group；
- 不进行 U-Net 与 Guided Decoder 的联合训练，原因是显存/训练成本较高。（正文 §4.3、附录 A.3/H）

---

## 5. 完整训练策略

训练整体上分为 **A：单步 U-Net 的训练** 和 **B：Guided Decoder 的单独训练**。其中 U-Net 训练又有两个时间阶段。

### 5.1 训练数据与标签

基本训练样本为同一段视频的左右眼对：

$$\{(V_L,V_R)\}$$

从左右视图估计第一帧视差图，再取：$\delta=P_{50}(D^0_{L\to R})$。训练时知道右眼 GT；部署时只有左眼视频及指定 $\delta$。

### 5.2 Stage A：单步 U-Net

**第一阶段：基础单步模型**

- 预训练权重：Stable Video Diffusion；冻结 VAE；微调 U-Net 完成单步潜变量合成；
- **200k iterations**；
- 补充材料称 *base unconditional model*，按上下文这里指**没有启用额外的视差条件机制**，不应解释为完全没有左眼图像条件。

**第二阶段：加入 $\delta$ 的条件微调**

- **170k iterations**；
- 将 $\tau(\delta)$ 和 CLIP 条件拼接；
- 每个 batch 以 **1/3 : 1/3 : 1/3** 概率从真实 Stereo4D、真实 Ego4D、Synthetic Stereo4D 中采样。

**真实数据损失**：将预测 latent 与 $z_R=E(V_R)$ 比较，同时用冻结的标准 VAE Decoder 解码预测 latent，在 RGB 空间计算 L1、SSIM、LPIPS。论文描述为 latent L2 与图像空间三个目标等权重组合：

$$
\mathcal L_{\mathrm{real}}=
\mathcal L_{\mathrm{latent,L2}}+
\mathcal L_{\mathrm{L1}}+
\mathcal L_{\mathrm{SSIM}}+
\mathcal L_{\mathrm{LPIPS}}
$$

这里的 SSIM 项表示以 SSIM 构造的训练目标；具体数值形式、归一化与实现细节没有在正文完整给出。冻结标准 Decoder 的**参数**不更新，但像素损失的梯度可穿过该 Decoder 返回 U-Net。

**超参数**：U-Net 学习率 **$2\times10^{-6}$**；采用 DeepSpeed Stage 2、FP16 混合精度，优化器状态可 CPU Offload。（附录 A.2）

### 5.3 Synthetic Baseline Augmentation（合成基线增强）

目的：真实相机基线分布较窄，训练模型响应更广范围的目标视差。

作者从 Stereo4D **预计算 2,500 个随机合成样本**。给定真实样本的视差/深度几何，抽取缩放因子：

$$
s\in\{0.05,0.1,0.2,0.4,0.6,0.8,1.25,1.5,2.0,3.0\}
$$

将视差按比例缩放并进行前向映射：

$$D'=sD,\qquad \delta'=s\delta$$

在理想双目几何中 $d=fB/Z$，故这种操作相当于模拟 $B'=sB$。用新的视差投影左眼像素，得到合成右眼训练目标 $\tilde V_R$。

**合成数据的监督**：

$$
\mathcal L_{\mathrm{synthetic}}\;=\;\text{Masked L1 on valid warped pixels}
$$

- 仅在 Forward Warping 有效的目标像素上监督；
- 新显露背景在左眼中不可见，因此不能凭空拥有真实 RGB 监督；还可能有目标位置空洞、投影覆盖冲突、边界无效等；
- 论文未给出精确 Mask 生成/归一化代码；
- 和真实双目数据不同：真实右相机拍到了新显露区域的实际 RGB，因此完整右眼图像本身可以作为 RGB 监督。

另有 **1% Zero-Disparity Augmentation**：令 $\delta=0$、目标右眼帧等于左眼帧。

### 5.4 数据增强、帧率及动态形状（附录 A.1，Table 8）

- Batch size：**1**；
- 随机时间子采样：目标帧率 **5–30 FPS**；
- 随机空间缩放：**[0.3, 1.0]**；
- 使用不同空间分辨率与帧数的 Bucketing：

| 训练分辨率 $[H,W]$ | 帧数 $N$ | 长宽比例 |
|---|---:|---|
| [512, 512] | 4 | 1:1 |
| [768, 320] / [320, 768] | 4 | 2.4:1 |
| [1024, 256] / [256, 1024] | 4 | 4:1 |
| [1280, 256] / [256, 1280] | 3 | 5:1 |
| [256, 256] | 16 | 1:1 |
| [192, 192] | 25 | 1:1 |
| [128, 256] / [256, 128] | 25 | 2:1 |

**区分：正文 §6.1 评价示例采用 16 帧、512×512，而附录 Table 8 是训练阶段的分桶增强配置；二者不是同一个固定输入配置。**

### 5.5 Stage B：Guided Decoder

- 同样使用 Stereo4D、Ego4D 的左右眼真实视频；
- 输入：真实右眼 latent $z_R=E(V_R)$，以及左眼原始视频 $V_L$；
- 目标：重建真实 $V_R$；
- 损失：等权重 L1 + LPIPS；
- 训练 **50k iterations**、学习率 **$1\times10^{-4}$**、BF16；
- 单独优化，不与 U-Net 联训。

### 5.6 训练和推理的差异表

| 条目 | 训练阶段 | 推理阶段 |
|---|---|---|
| 右眼真值 $V_R$ | 有，用于监督 | 无 |
| 视差图 $D$ | 可从左右眼估计；用于构建 $\delta$ 和增强 | 不要求提供 |
| $\delta$ | 来自样本视差图统计，增强时同步缩放 | 用户指定 |
| Forward Warping | 用于创建部分 Synthetic Stereo4D 样本 | 不运行 |
| U-Net | 微调单步直接预测任务 | 运行一次前向传播 |
| Guided Decoder | 用 GT latent 独立训练 | 用预测 latent 实际解码 |

---

## 6. 数据集：训练、测试、分布差异

### 6.1 训练数据（正文 §6.1）

| 数据集 | 用途 | 特点 |
|---|---|---|
| **Stereo4D** | 训练；同分布测试 | 来自网络立体视频的真实场景、动态物体；文中描述相机基线接近 **63 mm**；采用 M2SVid 提供的立体校正划分 |
| **Ego4D** | 训练；补充材料测试 | 第一视角视频、较大视差场景；采用相应立体校正处理版本 |
| **Synthetic Stereo4D** | 第二阶段的视差条件增强 | 从 Stereo4D 通过视差缩放和 Forward Warping 预生成的 2,500 个随机样本，不是由完整 3D 场景重渲染的新数据集 |

### 6.2 测试数据（正文 §6.1–6.3；附录 E.2）

| 数据集 | 域类型 | 已知相机信息/考查内容 |
|---|---|---|
| Stereo4D test | In-domain | 与训练场景/典型基线相近 |
| Apple Vision Pro (AVP) Spatial Video | 内容域外 | 相机基线约 **63.8 mm**，与训练常见基线相近 |
| iPhone Spatial Video | 内容与相机配置域外 | 基线约 **19.2 mm**，重点检验 $\delta$ 条件对不同立体强度的适应 |
| Ego4D test | 补充材料测试 | 第一视角、较大视差条件下的泛化 |

### 6.3 基准评测的条件信息和公平性

- 所有方法以单目左眼作为生成时的**视觉输入**。
- 为减小不同相机的目标立体强度差异，作者允许使用由测试集 GT 几何统计得到的额外**全局**信息：Elastic3D 接受 GT 左右视差的中位数 $\delta$；Warp-based 基线使用基于 GT 几何得到的尺度/偏移，将单目深度对齐。
- Warping 基线使用 DepthCrafter 估计相对深度后再 Warp；这与直接把 GT stereo depth 当作深度图送入生成器不同，但**不是完全没有 GT 几何辅助的部署测试**。
- 论文比较旨在研究“给定目标立体强度/参考几何标定后”的生成质量；不能据此认为仅凭单目视频就能自动恢复真实右相机的精确基线。

---

## 7. 评价指标与设计动机（正文 §5；附录 D）

论文把评价拆为**整体质量、双目纹理一致性、相对几何准确性、时间稳定性**四类。前两项新增 Matchability Error、P-PSNR，用于缓解普通像素指标把视差对齐误差与纹理损失混在一起的问题。

### 7.1 整体质量

| 指标 | 方向 | 计算对象 | 解释 |
|---|---|---|---|
| **PSNR** | ↑ | 预测右眼 vs GT 右眼 | 基于 MSE 的像素重建质量，受横向视差错位影响大 |
| **SSIM** | ↑ | 预测右眼 vs GT 右眼 | 亮度、对比度、结构相似性 |
| **LPIPS** | ↓ | 预测右眼 vs GT 右眼 | 学习式特征空间感知差异 |

### 7.2 Matchability Error（论文 Eq. 4、附录 Eq. 13–14）

使用 **DeDoDe v2** 匹配器，分别为 GT 双目和预测双目建立符合极线约束的匹配集合：$M_{\mathrm{gt}}$、$M_{\mathrm{pred}}$。

$$
\boxed{E_{\mathrm{Match}}=1-\frac{|M_{\mathrm{gt}}\cap M_{\mathrm{pred}}|}{|M_{\mathrm{gt}}\cup M_{\mathrm{pred}}|}}
$$

也可分解为：

$$
E_{\mathrm{Match}}=\frac{N_{\mathrm{FP}}+N_{\mathrm{FN}}}{N_{\mathrm{TP}}+N_{\mathrm{FP}}+N_{\mathrm{FN}}}
$$

- TP：GT 和预测都有的正确对应；
- FN / Omission：GT 有、预测缺失，常对应过度模糊或细节缺失；
- FP / Hallucination：预测产生额外的错误对应，可能对应生成幻觉；
- **越小越好**；主表以类似 `27.8` 的形式报告，按百分比刻度理解，而不是 27.8 这个直接概率值。
- 这是用于双眼竞争风险的**可匹配性代理指标**，不等于直接的人眼舒适度评分。

### 7.3 Patch-wise PSNR（P-PSNR）

- 在左眼中提取 **$16\times16$** 图像块；
- 沿对应水平极线在生成右眼中寻找最匹配的块；
- 用最佳匹配得到的局部重建质量构建 PSNR；
- **越大越好**，对一定程度的水平视差/位置偏移更稳健，仍对模糊和细节损失敏感。

附录 D 使用**人为水平平移**以及**高斯模糊**验证指标特性：P-PSNR 对小范围平移比标准 PSNR 稳定，而模糊仍会降低其分数；Matchability Error 对纹理模糊比 SSIM 更敏感。

### 7.4 Disp. err（几何误差）

- 使用 **FoundationStereo** 分别从生成与参考双目估计密集视差；
- 对 $D_{\mathrm{pred}}$ 相对于参考视差执行**最小二乘尺度、偏移对齐**；
- 报告对齐后的 **MAE（平均绝对误差）**；越小越好。

用简化形式写为：

$$
(a^*,b^*)=\arg\min_{a,b}\sum_p\left(aD_{\mathrm{pred}}(p)+b-D_{\mathrm{GT}}(p)\right)^2
$$

$$
E_{\mathrm{disp}}=\operatorname{MAE}\left(a^*D_{\mathrm{pred}}+b^*,D_{\mathrm{GT}}\right)
$$

**关键评测边界**：尺度/偏移对齐削弱了目标视差强度误差，更重视相对深度结构；这个指标**不能单独证明输入 $\delta=8$ 时输出实际中位视差恰好为 8 px**。

### 7.5 Temp. err（时间误差）

- 用 **RAFT** 分别估计 GT 右眼视频、预测右眼视频的相邻帧光流；
- 计算两组光流的 **EPE（Endpoint Error，终点误差）**；越小越好。

$$
E_{\mathrm{temp}}\approx\frac{1}{|\Omega|}\sum_{p\in\Omega}\|F_{\mathrm{pred}}(p)-F_{\mathrm{GT}}(p)\|_2
$$

衡量运动与 GT 的一致程度，但不覆盖所有主观闪烁、颜色稳定性问题。

### 7.6 附加评价：头显用户研究与 iSQoE

- 头显偏好测试：**9 名参与者、26 个场景**，每个竞争方法共 **120 次成对判断**。与 M2SVid 对比时，Elastic3D 被选择 **98/120（81.7%）**；与 Eye2Eye 对比时，被选择 **60/120（50.0%）**，平局占 37.5%。
- iSQoE：一种无参考沉浸式立体质量分数（↓）。论文报告多种现代生成方法得分集中在约 0.50，甚至部分结果可能比 GT 分数更低，认为其在高质量范围出现**饱和**，区分能力有限。（附录 E.1、E.3）

---

## 8. 实验结果及消融（正文 §6；补充材料 E–F）

### 8.1 视差条件消融（Table 1）

| 数据集 | 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Disp. err ↓ |
|---|---|---:|---:|---:|---:|
| Stereo4D | w/o disparity conditioning | 25.1 | 0.880 | 0.192 | 1.27 |
| Stereo4D | Elastic3D | **26.1** | **0.913** | **0.176** | **1.24** |
| iPhone | w/o disparity conditioning | 18.7 | 0.703 | 0.289 | **0.64** |
| iPhone | Elastic3D | **22.5** | **0.890** | **0.193** | 0.77 |

iPhone 在不同基线场景中 PSNR **+3.8 dB**；但经过对齐的 Disp. err 没有同步提升，因此不能把图像对齐收益等同为各项几何结构指标的全面改善。

### 8.2 直接生成 vs Warping 基线（Table 2；AVP）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Matchability Error ↓ | Disp. err ↓ |
|---|---:|---:|---:|---:|---:|
| Warp-based | 24.5 | 0.827 | 0.198 | **26.8** | 2.33 |
| Elastic3D | **25.9** | **0.894** | **0.196** | 30.9 | **1.74** |

Elastic3D 在此测试中整体质量及对齐后几何误差更好，但 Warp-based 的 Matchability Error 略好，不能概括成所有指标均胜出。

### 8.3 Guided Decoder 的独立重建消融（Table 3；Stereo4D GT latent）

| 解码方式 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 标准 Stable Diffusion Decoder | 30.2 | 0.963 | 0.106 |
| Guided Decoder | **34.3** | **0.981** | **0.068** |

这是**用真实右眼 latent 直接重建**的实验，排除了 U-Net 预测误差；PSNR 提升 **4.1 dB**。

### 8.4 完整生成中的 Guided Decoder 消融（Table 4；Stereo4D）

| 模型 | PSNR ↑ | LPIPS ↓ | Matchability Error ↓ | P-PSNR ↑ |
|---|---:|---:|---:|---:|
| Elastic3D w/o $D'$ | 25.2 | 0.212 | 41.9 | 26.1 |
| Elastic3D + $D'$ | **26.1** | **0.176** | **27.8** | **27.4** |
| M2SVid | 24.6 | 0.206 | 39.6 | 26.3 |
| M2SVid + $D'$（无需重训） | 25.2 | 0.175 | **24.8** | 27.5 |

说明 Guided Decoder 可以在 M2SVid 中作为解码器替换模块使用，但其收益仍需结合具体输入分布与几何任务判断。

### 8.5 与 SOTA 比较：主要重建指标（Tables 5–7）

**Stereo4D Test（Table 6）**

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Matchability Error ↓ | Disp. err ↓ |
|---|---:|---:|---:|---:|---:|
| SVG | 22.9 | 0.874 | 0.258 | 40.0 | 1.69 |
| StereoCrafter | 23.6 | 0.874 | 0.258 | 43.9 | 1.75 |
| M2SVid | 24.6 | 0.819 | 0.206 | 39.6 | 1.56 |
| Eye2Eye | 21.1 | 0.780 | 0.313 | 35.8 | 2.45 |
| ReStereo | 21.2 | 0.788 | 0.307 | 41.3 | 1.95 |
| **Elastic3D** | **26.1** | **0.913** | **0.176** | **27.8** | **1.24** |

**Apple Vision Pro Spatial Video（Table 5）**

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Matchability Error ↓ | Disp. err ↓ |
|---|---:|---:|---:|---:|---:|
| SVG | 19.3 | 0.690 | 0.410 | 56.3 | 3.71 |
| StereoCrafter | 22.5 | 0.826 | 0.323 | 51.8 | 2.30 |
| M2SVid | 24.4 | 0.821 | 0.221 | 41.5 | 2.30 |
| Eye2Eye | 20.6 | 0.733 | 0.392 | 39.2 | 3.82 |
| **Elastic3D** | **25.9** | **0.894** | **0.196** | **30.9** | **1.74** |

**iPhone Spatial Video（Table 7）**

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Matchability Error ↓ | Disp. err ↓ |
|---|---:|---:|---:|---:|---:|
| SVG | 16.3 | 0.600 | 0.387 | 49.0 | 1.45 |
| StereoCrafter | 21.9 | 0.877 | 0.257 | 45.1 | 0.63 |
| M2SVid | **22.9** | 0.865 | 0.205 | 38.4 | **0.60** |
| Eye2Eye | 20.2 | 0.818 | 0.281 | 32.0 | 1.11 |
| **Elastic3D** | 22.5 | **0.890** | **0.193** | **26.5** | 0.77 |

iPhone 是内容和基线均存在分布变化的测试：Elastic3D 的 SSIM/LPIPS/Matchability/P-PSNR 最好，但 PSNR 与 Disp. err 并非第一。

**Ego4D（附录 Table 10）**：Elastic3D 的 PSNR **19.8**、SSIM **0.760**、LPIPS **0.276**、Matchability **27.0**、P-PSNR **25.9**；附录同时报告视差 MAE **3.30**、光流 EPE **5.67**。

### 8.6 速度与流水线分解（附录 Table 12）

测试设备 **NVIDIA H100**，每段 **16 帧、512×512**，单位秒：

| 方法 | 模型推理 | 单目深度 | Warping | 合计 |
|---|---:|---:|---:|---:|
| SVG | 45.0 | 3.3 | 7.2 | 55.5 |
| StereoCrafter | 2.6 | 3.3 | 0.9 | 6.8 |
| M2SVid | **0.8** | 3.3 | 0.9 | 5.0 |
| **Elastic3D** | 1.7 | — | — | **1.7** |

- Elastic3D 模型本体推理 **1.7 s** 比 M2SVid 的 **0.8 s** 慢；增加的 Extractor 和极线注意力带来开销。
- 但 Elastic3D 无需外部 Monodepth / Warp 阶段，因此在论文统计的完整比较口径下为 **1.7 s vs 5.0 s**。
- 这里的合计包含表中指定的阶段，**并不一定包含视频文件读取、写出、编解码或完整产品部署开销**；切勿直接与其他工作不同统计口径的数字比较。
- 由于 Eye2Eye 的模型代码不可公开运行，论文没有给出同硬件的实测速度；附录关于 64 倍空间位置、50 步去噪的推导属于理论量级对比，不是实测加速比。

---

## 9. 论文明确的限制（附录 H）

1. **长视频和高分辨率显存开销**：在单张 H100 80GB 上，论文报告 512×512 下可处理约 **45 帧**；需要进一步研究分块、跨块时序衔接、空间 Tiling。
2. **极大视差/超出基线分布的泛化**：当视差对应的基线远大于典型训练值（约 63 mm）时，直接生成可能退化，显式几何 Warp 有时更有优势。
3. **单标量条件的歧义**：中位视差无法描述完整逐像素分布；背景主导或镜头 Zoom 可能让 $\delta$ 不能充分反映前景变化。
4. **几何不严格刚性**：可能生成局部起伏的墙面、不一致的前后关系等 Geometry Hallucination；隐式几何不能保证所有像素满足物理场景。
5. **生成与解码分开训练**：避免高额联训显存成本，但 Decoder 训练输入的 GT latent 与实际推理预测 latent 存在潜在分布差异。
6. **极线注意力实现开销**：当前采用逐行 Python 循环，存在解释器调用与 GPU 利用率问题；作者指出可改用自定义 CUDA Kernel 并行执行。
7. **左眼不可见纹理**：引导注意力只能利用左眼存在的高分辨率信息；右眼独有的新显露区域仍需依赖 latent 中的生成信息与解码先验，不能简单复制原图。

---

## 10. 论文参数与技术结论速查

| 条目 | 论文/实现信息 |
|---|---|
| 主干 | Stable Video Diffusion（预训练 Video U-Net） |
| U-Net 预测 | 一步、固定最大噪声时间步、zero latent 初始状态 |
| 左眼编码 | 冻结 VAE Encoder；按 SVD 示例 4 通道、8× 空间压缩 |
| 原始 SVD 额外视觉条件 | CLIP ViT 图像编码器；通过 Cross-Attention 读取 |
| 控制值 | $\delta=P_{50}(D^0_{L\to R})$，像素视差中位数 |
| 视差 Token | 数值按约 $10^{-2}$ 缩放并广播，与 CLIP Embedding 拼接 |
| U-Net 学习率 | $2\times10^{-6}$ |
| U-Net 训练 | 基础 200k + 条件微调 170k iterations |
| Synthetic Stereo4D | 2,500 个样本；$s\in\{0.05,0.1,0.2,0.4,0.6,0.8,1.25,1.5,2,3\}$ |
| 条件微调采样 | Stereo4D : Ego4D : Synthetic Stereo4D = 1:1:1 |
| 零视差增强 | 1% batches，$\delta=0$，左右目标一致 |
| U-Net 真实样本损失 | latent L2 + RGB L1 + SSIM + LPIPS |
| 合成样本损失 | Masked RGB L1 |
| Guided Decoder 输入 | 预测/真实右眼 latent + 左眼原始 RGB 引导特征 |
| Decoder 训练 | GT right latent + left guidance，50k iterations，BF16，LR $10^{-4}$ |
| Decoder 损失 | L1 + LPIPS |
| 极线注意力 | 右眼 Query，左眼同一水平行的 Key / Value |
| 极线注意力复杂度 | $O(HW^2)$，全局 Cross-Attention 为 $O(H^2W^2)$ |
| 原始评测视频 | 常用 16 帧、512×512 |
| 速度测试 | H100，Elastic3D 1.7 s / 16 帧（论文 Table 12 口径） |

## 11. 源文定位与核验说明

| 具体内容 | 论文位置 |
|---|---|
| 五项质量目标 | §3 |
| 端到端整体数据流 | Figure 2、§4 |
| 一步扩散公式 | §4.1，Eq. 1；补充材料 §C，Eqs. 10–12 |
| $\delta$ 定义与控制 | §4.2，Eqs. 2–3；Figure 3；补充材料 §A.2、§F.2 |
| Guided Decoder 与 Figure 4 | §4.3；补充材料 §B，Eqs. 5–9 |
| 训练分桶和超参数 | 补充材料 §A.1–A.3，Table 8 |
| 评测指标 | §5.1；补充材料 §D，Figures 6–10 |
| 数据集、基线及主要结果 | §6，Tables 1–7；Figure 5 |
| 人类偏好与延迟 | 补充材料 §E，Tables 9、12 |
| Ego4D 补充结果 | 补充材料 Table 10 |
| 中位数与平均值消融 | 补充材料 §F.2，Table 14 |
| 方法局限 | 补充材料 §H |

**资料**：

- Elastic3D 论文与项目主页：https://elastic3d.github.io/
- 原始 SVD 官方网络配置（仅用于核对骨干结构）：https://github.com/Stability-AI/generative-models/blob/main/configs/inference/svd.yaml
- 原始 SVD CLIP 图像编码器源码：https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/encoders/modules.py

**核验范围说明**：论文给出了主要模块、数学公式、训练策略、数据集和实验协议，但并未在正文或补充材料中公布全部 Elastic3D 可执行源码和每一层的详细超参数。文中具体原始 SVD 通道配置、CLIP ViT 型号等属于**SVD 官方配置可核对的信息**，与 **Elastic3D 论文直接明确的信息**已分别标记。
