# InfiniDepth → InfiniSplat v1 → InfiniSplat v2：技术路线、实现区别与景深虚化研究启发

> 文献精读与技术对比笔记 · 核对日期：2026-11-03  
> 核心关注：**深度如何使用、Gaussian 如何生成与控制、前景深度断层、错误点、遮挡/去遮挡，以及对视频可控景深虚化的可借鉴性**。  
> **事实分层**：本文明确区分【论文】、【公开源码】与【研究推断/待验证】。表中数字来自对应论文，**并非本地复现实验**。

## 0. 一页结论（最值得记住的六句话）

1. **InfiniDepth** 首先是一种**连续二维坐标查询的深度估计模型**，本身并不是 InfiniSplat v1 必选的预处理器。它也包含自适应表面采样和单图高斯新视角合成的应用。
2. **InfiniSplat v1（RGB-only）使用 DepthPro；RGB+稀疏深度/LiDAR 版本才使用 InfiniDepth-Metric**。两种版本均以冻结深度先验构建几何，再学习高斯属性。
3. v1 的核心机制为 **Geometry-Guided Sampling（几何引导采样）+ Implicit Gaussian Decoding（隐式高斯解码）**：从“固定像素中心生成高斯”变成“依三维表面积在非整数像素位置采样，再查询图像特征预测高斯”。
4. **v2 不是 v1 原样升级成多视角**。它以多视角 Depth Anything 3（DA3）为基础，将**冻结的几何分支**和**可训练的高斯属性分支**解耦；不把 v1 的局部三角形面积采样作为主干机制。
5. v2 对部分 **Flying Points（悬浮点）**通过跨视角监督学出低 Opacity，再经过合并与剪枝抑制其影响；**它没有保证把错误深度修到正确位置，也没有专门生成完全不可见的背景**。
6. 对可控景深/视频虚化，最值得迁移的不是把整套 NVS 模型直接当作 Bokeh 模型，而是**几何采样、深度可靠性门控、背景分层/可见性处理、跨视角/跨帧监督**等思想。**三篇都没有直接解决给定焦点与光圈的光学 CoC 虚化问题**。

---

## 1. 文献关系与总览

| 维度 | InfiniDepth | InfiniSplat v1 | InfiniSplat v2 |
|---|---|---|---|
| 论文 | *InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields* | *InfiniSplat: Implicit Gaussian Decoding for Large-Baseline Monocular View Synthesis* | *InfiniSplat v2: Analyzing and Improving Feed-Forward 3DGS from Any Views* |
| 发表/版本 | CVPR 2026 | SIGGRAPH Asia 2026 / TOG | arXiv 2026-10 |
| 核心任务 | 细粒度单图深度估计，含 NVS 应用 | 单图前馈 3DGS / 大基线 NVS | 任意多输入视角前馈 3DGS / NVS |
| 推理输入 | RGB；Metric/传感器增强版本可接收稀疏深度 | 1 张 RGB；LiDAR 版额外输入稀疏深度 | 多张 RGB；相机姿态与内参可给可不给 |
| 深度来源 | **自己的隐式深度网络** | RGB: **DepthPro**；LiDAR: **InfiniDepth-Metric** | **DA3 多视角几何分支** |
| 三维点/高斯中心来源 | 连续深度查询 → 自适应表面采样 | 深度反投影 → 局部三角形面积加权 → 非网格查询点 | 多视角深度反投影 → 点合并；**中心固定** |
| 高斯属性 | NVS 应用中的 Gaussian Head | DINOv3 + CNN → 门控融合 → 14 维偏移 | DA3 属性分支 + DPT Gaussian Head → 11 维属性 |
| 是否学习中心偏移 | NVS GS Head 可预测位置偏移 | **是**：二维位置 + 逆深度等偏移 | **否**：主模型与 Refiner 都保持中心不变 |
| 主要处理问题 | 高分辨率深度、可见表面密度不均匀 | 像素对齐高斯在大视角下表面断裂 | 不可靠多视角点云导致的渲染伪影，几何/属性耦合 |
| 动态视频约束 | 无显式时序一致性约束 | 单图、静态场景为主 | 同一静态场景多视角，非动态 4DGS |
| 是否直接渲染光学虚化 | **否** | **否** | **否** |

**为什么不能把三篇简单写成串行模块？**

- InfiniDepth 是**可以独立使用**的深度表示和模型，v1 在 LiDAR 分支复用了其 Metric 变体。
- v1 的 RGB-only 路线可以**完全不调用 InfiniDepth**；v1 的高斯采样策略与 InfiniDepth 的“按真实三维覆盖面积分配查询”有思想联系，但实现不是完全相同的公式。
- v2 换成 DA3 的多视角几何与属性解耦架构，核心问题不再是“单图像素中心采样”。

**相关论文与项目：**

- InfiniDepth：[论文](https://arxiv.org/html/2601.03252) · [CVPR 2026 正式页面](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_InfiniDepth_Arbitrary-Resolution_and_Fine-Grained_Depth_Estimation_with_Neural_Implicit_Fields_CVPR_2026_paper.html) · [GitHub](https://github.com/zju3dv/InfiniDepth)
- InfiniSplat v1：[论文](https://arxiv.org/html/2608.02437) · [项目页](https://zju3dv.github.io/InfiniSplat/) · [GitHub v1](https://github.com/zju3dv/InfiniSplat/tree/main/InfiniSplat_v1)
- InfiniSplat v2：[论文](https://arxiv.org/html/2610.20341) · [项目页](https://zju3dv.github.io/InfiniSplat2/) · [GitHub v2](https://github.com/zju3dv/InfiniSplat/tree/main/InfiniSplat_v2)

---

## 2. InfiniDepth：连续深度查询与表面采样（略读版）

### 2.1 主线数据流

```text
RGB I
  └─ ViT 编码多层 Token
      └─ Reassemble → 多尺度特征金字塔 {F_k}
          └─ 对任意连续坐标 q=(x,y) 进行双线性查询
              └─ 多尺度门控融合 + MLP
                  └─ d_I(q) 连续深度查询值
                      ├─ 任意目标分辨率的 Depth Map
                      └─ 面积自适应查询 → 3D Point Cloud → GS Head → NVS（应用）
```

**核心表示**（论文 Eq. 2）：

$$
\boxed{d_I(x,y)=N_\theta\big(I,(x,y)\big)}
$$

与固定 $H\times W$ 深度网格相比，网络接受任意连续二维查询坐标，在相应位置获取局部特征并预测深度。这里的“任意分辨率”指**连续可查询性**，不代表真实几何信息无限增加，也不自动恢复不可见区域。

**多尺度融合**（论文 Eq. 3–4）：

$$
\mathbf h_{k+1}=\mathrm{FFN}_{k}\!\left(\mathbf f^{k+1}_{q}+\mathbf g_k\odot \mathrm{Linear}(\mathbf h_k)\right),\qquad d_I(q)=\mathrm{MLP}(\mathbf h_L).
$$

浅层高分辨率特征提供局部细节，深层特征提供上下文，融合后在任意坐标输出深度。

### 2.2 Infinite Depth Query：按三维面积分配亚像素查询

对图像上位置 $(x,y)$，计算其在三维表面上的面积覆盖近似权重（论文 Eq. 5）：

$$
\boxed{w(x,y)=\frac{d_I(x,y)^2}{|\mathbf n(x,y)\cdot\mathbf v(x,y)|+\varepsilon}}.
$$

- $d_I^2$：深度越远，同样像素通常覆盖更大的三维表面积；
- $|\mathbf n\cdot\mathbf v|$：表面越倾斜，像素对应的实际表面积可能越大；
- 在离散像素上根据 $w$ 分配查询预算，再为抽中的像素生成**随机亚像素查询坐标**。

表面法线可通过连续隐式场的 Jacobian（雅可比）求导：

$$
\mathbf n=\frac{\partial_x\mathbf X\times\partial_y\mathbf X}{\|\partial_x\mathbf X\times\partial_y\mathbf X\|}.
$$

**与 v1 的关系**：InfiniDepth 用的是基于深度平方与表面方向的局部面积近似，v1 则用**反投影局部三角形的实际面积**作为采样权重，属于相似思路的不同实现。

### 2.3 GS Head 与限制

在 InfiniDepth 的 NVS 应用中，按照连续深度场查询三维点，叠加颜色、射线和 ViT 特征，由 Gaussian Head 预测位置、颜色等属性偏移形成高斯。

- **能力**：可以改善可见表面的细节与采样均匀度；
- **非能力**：未观测背面/被遮挡背景的真实纹理不能凭空被连续查询恢复；
- **视频**：缺少显式时序约束，不能直接视为稳定的视频几何重建模型。

**论文位置**：§3.1–3.3；Appendix A.2、A.3；§4.5。

---

## 3. InfiniSplat v1：Geometry-Guided Sampling + Implicit Gaussian Decoding

### 3.1 两个版本与输入

```text
RGB-only:
    单张 RGB ── DepthPro ──────────────────┐
                                         ├→ Geometry-Guided Sampling → Base Gaussians
RGB + LiDAR:
    单张 RGB + 稀疏深度 ─ InfiniDepth-Metric ┘

共同图像分支:
    RGB ─→ DINOv3 ViT + CNN ─→ 坐标双线性查询 ─→ 门控融合 ─→ 14D Gaussian Delta
                                                         ↓
    Base Gaussians ────────────────────→ Gaussian Composer ─→ 3D Gaussians → 3DGS Render
```

【论文】LiDAR 版本在训练时使用 **1,500 个稀疏深度样本**作为 Depth Prompt；不是指最终只有 1,500 个高斯。两种版本的几何先验冻结。

### 3.2 Geometry-Guided Sampling：从固定像素中心到表面上的支持点

**步骤 A：从预测深度反投影**

$$
\mathbf X(p)=\Pi^{-1}(p,D(p);\hat K).
$$

**步骤 B：根据相邻三维点构成局部三角形**

$$
\boxed{A_t=\tfrac12\|[\mathbf X(p_b)-\mathbf X(p_a)]\times[\mathbf X(p_c)-\mathbf X(p_a)]\|_2}.
$$

**步骤 C：剔除深度跨越太大的候选三角形**

当相邻顶点处于不同深度层（例如前景人物 2 m、背景墙 5 m），如果强行连接，会产生一块不存在的“斜坡表面”，其巨大的面积可能误导采样。论文明确：对相对深度变化大的三角形先剔除，再进行面积采样。

**步骤 D：面积权重、非整数坐标采样**

$$
\boxed{w_t=A_t/\sum_{t'} A_{t'}}.
$$

对有效三角区域按面积分配采样点，形成 $\mathcal S=\{q_i\}_{i=1}^{N}$，$q_i$ 可以是 $(125.3,87.6)$ 这种非整数像素位置；接着双线性读取 RGB 与 Depth。

**步骤 E：Base Gaussian**

$$
\bar g_i=(\bar\mu_i,\bar s_i,\bar r_i,\bar c_i,\bar\alpha_i),\quad
\bar\mu_i=\Pi^{-1}(q_i,D(q_i);\hat K),\quad\bar c_i=I(q_i).
$$

关键：这个局部三角形只用于**推断采样预算与排除无效连接**，不是把场景替换成最终 Mesh 渲染；最终表示仍然是可微渲染的 3D Gaussians。

### 3.3 Implicit Gaussian Decoding：在非整数坐标上预测属性

先提取两个互补分支：

- **DINOv3 ViT-L/16**：分辨率较粗，提供物体结构/语义上下文，支撑跨域泛化；
- **BasicEncoder CNN**：空间分辨率较高，提供局部纹理、颜色和边界；
- 这里的“语义 vs 细节”是**功能倾向**，不是人为把不同物体区域硬编码到某个分支。

采样特征：

$$
 f_i^{DINO}=\mathcal B(F^{DINO},q_i),\quad f_i^{CNN}=\mathcal B(F^{CNN},q_i).
$$

通道门控融合：

$$
\tilde f_i^{CNN}=P(f_i^{CNN}),\quad
\boldsymbol\gamma_i=\sigma(W[f_i^{DINO},\tilde f_i^{CNN}]),
$$

$$
\boxed{h_i=\boldsymbol\gamma_i\odot f_i^{DINO}+(1-\boldsymbol\gamma_i)\odot\tilde f_i^{CNN}}.
$$

> $\boldsymbol\gamma$ 是**特征融合门控权重**，不是高斯的不透明度 $\alpha$。具体哪个位置偏向 DINO，需要对训练后的 `gate_weights` 做可视化；论文并未逐像素报告这一结果。

最后共享 MLP：

$$
\boxed{\Delta g_i=\mathrm{MLP}(h_i)\in\mathbb R^{14}}.
$$

| 14 维通道 | 数量 | 实际用途 |
|---|---:|---|
| 平面位置偏移 | 2 | 调整采样位置 $(x,y)$ |
| 深度偏移 | 1 | 在实现中调整**逆深度参数**，使用 Softplus 保证合法性 |
| Scale 偏移 | 3 | 三轴尺度调整 |
| Rotation 偏移 | 4 | 四元数残差，最终归一化 |
| Color 偏移 | 3 | 颜色 Sigmoid 空间调整 |
| Opacity 偏移 | 1 | 不透明度 Logit/Sigmoid 空间调整 |
| **合计** | **14** | |

位置（示意）可理解为 $(q_i+\Delta q_i,d_i+\Delta d_i)\xrightarrow{\Pi^{-1}}\mu_i$；**精确的代码实现是在 NDC 空间和逆深度空间构造中心，不是直接把 14 个无约束数值加到 XYZRGB 上**。

其余属性采用“基础值 + 有界残差”的策略：

$$
 g_i^{(a)}=\psi_a\left(\psi_a^{-1}(\bar g_i^{(a)})+\lambda_a\Delta g_i^{(a)}\right),
$$

使预测既能修正基础高斯，又保持颜色、Opacity、Scale 等参数在合理数值空间中。

**注意**：v1 的高斯位置**可以被学习修正**，与 v2 的“中心固定”是根本不同的设计。

### 3.4 训练与监督

- **单图 Context 作为网络输入**，用同场景 **3 个 Target Views** 进行新视角渲染监督；
- 训练数据：**Hypersim**；论文默认每张源图采样 **150 万支持点**；
- 训练损失：RGB L1、冻结 VGG 的 Feature/Gram 感知损失、Gaussian Scale/Local Smoothness 正则；
- DINO/CNN、门控和高斯 MLP 学习；DepthPro/InfiniDepth 先验冻结；
- 推理不需要目标视角 GT，也不需要针对每个场景执行几十万步 GS 优化。

论文损失的结构为：

$$
\mathcal L=\lambda_{rgb}\mathcal L_{rgb}+\lambda_{perc}\mathcal L_{perc}+\lambda_{reg}\big(\lambda_{scale}\mathcal L_{scale}+\lambda_{smooth}\mathcal L_{smooth}\big).
$$

这里不是直接对每个高斯提供 GT 属性，而是通过可微 3DGS 渲染到 Target Views，让 RGB 误差反向约束高斯。

### 3.5 v1 的论文实验结果与解释

**Table 1，RGB-only，四个真实测试集的均值**（**在作者的基线选择、可见区域和评测设置下**）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| SHARP | 18.475 | 0.758 | 0.299 |
| InfiniSplat-RGB | **20.394** | **0.806** | **0.277** |
| 相对 SHARP | +1.919 dB | +0.048 | -0.022 |

**Table 2，ScanNet++ 消融 PSNR**：

| 配置 | PSNR ↑ | 相对完整模型变化 |
|---|---:|---:|
| 完整模型 | **22.240** | — |
| 无 DINO | 12.501 | -9.739 |
| 无 CNN | 20.368 | -1.872 |
| 不学习高斯参数更新 | 19.861 | -2.379 |
| 无正则化 | 20.717 | -1.523 |
| 无 Geometry-Guided Sampling | 21.576 | -0.664 |
| 无 Implicit Decoder | 20.811 | -1.429 |

**如何解读**：

1. DINO 分支在这个跨域设置下非常关键，不是只依赖局部 CNN 就足够；
2. 仅根据深度和 RGB 初始化 Base Gaussian 的质量不足，仍需学习更新；
3. Geometry-Guided Sampling 提供独立收益，但**不能把整篇论文的全部性能提升都归因于它**；
4. 大视角下的结构连续性改善，属于几何支持点布局和属性学习共同作用。

**实验边界**：数据要求源—目标视锥重叠 >60%，部分评测有可见区域 Mask；因此结果**并不证明对完全不可见的背景有真实重建能力**。

### 3.6 官方源码导航

- [RGB-only Encoder：`encoder_infinisplat.py`](https://github.com/zju3dv/InfiniSplat/blob/main/InfiniSplat_v1/src/model/encoder/encoder_infinisplat.py)：调用 DepthPro、采样、特征提取、14D 解码、输出 Gaussians。
- [LiDAR Encoder：`encoder_infinidepth_query.py`](https://github.com/zju3dv/InfiniSplat/blob/main/InfiniSplat_v1/src/model/encoder/encoder_infinidepth_query.py)：使用 InfiniDepth + `prompt_disparity`/`prompt_mask`。
- [Surface Sampling：`sampling_utils.py`](https://github.com/zju3dv/InfiniSplat/blob/main/InfiniSplat_v1/src/model/encoder/depth/infinidepth/sampling_utils.py)：局部表面采样。
- [Implicit Head：`implicit_gs_head.py`](https://github.com/zju3dv/InfiniSplat/blob/main/InfiniSplat_v1/src/model/encoder/gaussian/implicit_gs_head.py)：`grid_sample`、门控融合、`out_dim=14`。
- [Gaussian Decoder：`gaussian_decoder.py`](https://github.com/zju3dv/InfiniSplat/blob/main/InfiniSplat_v1/src/model/encoder/gaussian/gaussian_decoder.py)：基础值初始化和偏移合成。

**源码核查补充**：公开的 v1 `EncoderInfiniSplat.forward()` 直接接收 `context["intrinsics"]` 并用于反投影；所以虽然论文将深度与相机先验抽象为几何预测函数，但不能据此断言该源码入口会从任意 RGB 自动预测所有相机内参。

---

## 4. InfiniSplat v2：DA3 几何 + 高斯属性分支 + 可选学习式细化

### 4.1 v2 Stage I：完整数据流

```text
N 张 RGB（位姿 / Intrinsics 可选）
   │
   ├─ Geometry Branch: DA3（冻结）
   │      ├─ 多视角 Depth
   │      └─ Camera Pose / Intrinsics
   │           ↓
   │       Depth + Camera Unprojection → Gaussian Centers μ（固定）
   │
   └─ Attribute Branch: DA3 特征（部分微调）
          └─ DPT-style Gaussian Decoder
               └─ Gaussian attributes ξ = {opacity α, scale s, rotation q, color c}
   │
   └─ Combine {μ, ξ} → Voxel Merge → Opacity Prune → Stage I Gaussians
                                                └─ gsplat → Novel Views
```

### 4.2 几何属性解耦：为什么不再自由改位置？

定义一个高斯：

$$
\mathcal G=\{(\mu_m,\xi_m)\}_{m=1}^M,\quad \xi_m=(\alpha_m,\mathbf s_m,\mathbf q_m,\mathbf c_m).
$$

- **Geometry Branch**：从 DA3 初始化，冻结 Depth 与 Camera Heads，给出高斯中心 $\mu_m$；
- **Attribute Branch**：独立 DA3 初始化，仅微调**最后 10 个 Transformer Block**及 Gaussian Head；
- 主模型不预测中心 Offset；通过更新 Color、Scale、Rotation、Opacity 改善图像质量。

【论文实验结论】在该训练方案下，联合微调几何（w/o G）或者为中心额外预测 3D Offset（w/ O）均使 NVS 结果变差，作者认为 RGB Loss 会利用位置漂移来补偿属性误差，导致几何退化。**这不是“所有深度校正都无效”的通用结论**。

### 4.3 相机尺度对齐：训练不能混用不一致的坐标系

先基于预测点云的平均径向距离归一化：

$$
\boxed{s_{norm}=\tfrac1{|\mathcal V_{pts}|}\sum_j\|x_j\|_2},\qquad\bar D_i=\hat D_i/s_{norm}.
$$

训练时以同一参考相机为坐标原点，用 Umeyama 从 GT 输入相机中心拟合**尺度**到预测相机轨迹，**只取尺度，不将整个旋转/平移相似变换强行作用于 GT 相机**；然后：

$$
\mu_{i,u}^{cand}=\bar D_i(u)\,\tilde R_i^* (K_i^*)^{-1}u+\tilde t_i^*.
$$

这样训练时的反投影与 GT Target Camera 处于同一几何尺度与坐标框架，减少相机误差污染高斯属性学习。

**不要误解成**：推理必须有 GT Camera。v2 支持 Pose-conditioned 与 Pose-free，后者使用 DA3 预测的相机。

### 4.4 大角度目标视角监督

作者定义目标相机相对输入相机的最小光轴夹角：

$$
\boxed{a(v,\mathcal C)=\min_{i\in\mathcal C}\arccos(f_v^\top f_i)}.
$$

训练时 70% 从大角度候选池、30% 从小角度池采样目标视角。意义在于：**近视角可能容忍错误的高斯形状，而侧视图更容易暴露尺度/旋转不合理的问题**。

但论文使用 GT 深度构造 Target Visibility Mask，目标像素必须能够由至少一个输入视角以深度一致方式支持：

$$
\boxed{|z_{proj}-d_{src}|\leq\max(0.03,0.01d_{src})}.
$$

因此它重点学习**已有可见表面的大视角渲染**，并非无条件填补所有新出现的隐藏背景。

### 4.5 Voxel Merging + Opacity Pruning

DA3 不同输入视角可能反投影出位置接近的冗余点，也可能出现相互不一致的悬浮点。

**第一步：Voxel Merging** 将近邻点与属性合并，减少冗余。

**第二步：Opacity Pruning** 对学习出的低不透明度点进行剪枝：

$$
\boxed{\alpha_m<0.01\quad\Rightarrow\quad\text{删除该高斯}}.
$$

【论文】附录记载：归一化空间体素边长 **0.004**，训练与推理均在合并后剪掉 Opacity <0.01 的高斯。

【公开推理代码核对】`InfiniSplat_v2/src/infinisplat/gaussians.py` 的当前发布实现中，**实际动态体素边长**由高斯尺度中位数计算：

```python
median = exp(mean(log(scales), axis=-1)).median()
voxel = median * 1.4
# 按体素索引分组，属性求均值
# merged['opacities'] >= .01 保留
```

这和论文写出的**固定 0.004 体素边长**表述并不完全一致。**如果要严格复现实验，必须记录使用的是论文协议还是当前发布推理代码，不可假定两者完全等价。**

【机制解释】假设错误点在两个输入视角都导致重影，降低该点 Opacity 可能减少跨视角 RGB Loss，故属性网络有机会学到抑制这类错误点；但 **Opacity 不是一个有监督校准过的“深度正确概率”**。正确的半透明结构也可能具有低 Opacity；一致性的判断还受尺度、遮挡、视角覆盖影响。

### 4.6 Stage II：Learned Rendering-Feedback Refinement（论文设计）

Stage II 冻结 Stage I Reconstructor，给每个高斯注入**当前渲染误差对属性的梯度**：

$$
 g^t=\operatorname{RenderGrad}(\mathcal G^t; I_{input},\Pi_{input}),\quad
 h^t=E(\mathcal G^t,\operatorname{sg}(g^t),f).
$$

KNN Recurrent Transformer 结合当前属性、特征、渲染梯度、邻居和历史状态：

$$
(\Delta\xi_m^t,b_m^{t+1})=U(\xi_m^t,h_m^t,b_m^t),
$$

$$
\boxed{\xi_m^{t+1}=\mathrm{Apply}(\xi_m^t,\Delta\xi_m^t),\qquad\mu_m^{t+1}=\mu_m^0}.
$$

- 4 层 KNN Transformer，每点 8 邻居；循环隐藏状态 16 维；内部 64 维；
- 每轮重新渲染输入视角、重新求反馈梯度，复用 DPT 特征和固定位置的 KNN 图；
- 训练时展开 1–3 轮，论文推理 3 轮；
- 与常规 Adam 区别：**梯度是学习式更新网络的输入，而不是直接执行固定公式的梯度下降**；
- Stage II 不提供 GT Target 作为推理输入，但训练时仍有输入/目标视角监督；
- **高斯中心和数量均固定**，只更新属性。

论文 ETH3D 的同等 **3 次更新预算**对照（附录 Table 8）：

| 方法 | PSNR ↑ | LPIPS ↓ |
|---|---:|---:|
| Stage I | 16.9643 | 0.4339 |
| + Adam 3 步 | 16.9890 | 0.4312 |
| + Learned Refiner 3 步 | **17.6074** | **0.4176** |

**不能把这组实验理解为**学习式 Refiner 比已经充分迭代收敛的逐场景优化一定更好；它比较的是相同的 3 步预算。

**开源限制**：当前核查的 v2 公共 `model.py`、`pipeline.py` 主要实现**Stage I 前馈重建**。论文描述了 Stage II，但公开推理 API 没有看到完整三轮渲染反馈细化调用。因此不要默认 `pip install .` 后运行的推理包含论文 `Ours*`（Refined）结果。

### 4.7 v2 关键消融与结果

**Table 2，ETH3D / DTU，Stage I 消融；LPIPS 越小越好**：

| 方法 | ETH3D LPIPS | DTU LPIPS |
|---|---:|---:|
| 完整 Stage I | **0.4339** | **0.2070** |
| w/o A：不微调属性 Backbone | 0.4388 | 0.2612 |
| w/o G：不冻结几何 | 0.5204 | 0.2695 |
| w/ O：允许中心偏移 | 0.4998 | 0.2826 |
| w/o C：训练不使用对齐 GT Camera | 0.5017 | 0.3237 |
| w/o V：不用大角度采样 | 0.4533 | 0.2295 |

**更可靠的结论**：冻结中心与几何/属性解耦是该方法的有效设计；相机尺度与坐标匹配、目标视角角度也显著影响高斯学习。

**主表六数据集平均**（与 v1 的实验协议不同，不可直接横向比较）：

| 推理设置 | 模型 | LPIPS ↓ | DISTS ↓ |
|---|---|---:|---:|
| Pose-free | Ours（Stage I） | **0.3026** | **0.1341** |
| Pose-free | Ours*（Stage II） | 0.2970 | 0.1379 |
| Pose-conditioned | Ours（Stage I） | 0.2472 | 0.1163 |
| Pose-conditioned | Ours*（Stage II） | 0.2392 | **0.1159** |

Stage II 对 Pose-free LPIPS 有所改善，但 **Pose-free DISTS 由 0.1341 上升至 0.1379**：并非所有指标都单调改善。

### 4.8 v2 官方源码导航及发布范围

- [主模型 `model.py`](https://github.com/zju3dv/InfiniSplat/blob/main/InfiniSplat_v2/src/infinisplat/model.py)：几何与属性两个 DA3 分支、深度/相机、输出高斯；
- [高斯属性/合并 `gaussians.py`](https://github.com/zju3dv/InfiniSplat/blob/main/InfiniSplat_v2/src/infinisplat/gaussians.py)：Gaussian Head 的 11 维输出、体素合并和 Opacity 阈值；
- [输入与推理 `pipeline.py`](https://github.com/zju3dv/InfiniSplat/blob/main/InfiniSplat_v2/src/infinisplat/pipeline.py)：输入视角、位姿模式、推理结果导出；
- [README 与运行方法](https://github.com/zju3dv/InfiniSplat/blob/main/InfiniSplat_v2/README.md) · [公开权重](https://huggingface.co/Edisoneh/InfiniSplat-v2)。

**源代码层面的另一处差别**：v2 Gaussian Head 是 **11 通道输出**——Rotation 4 + Scale 3 + Opacity 1 + SH0 Color 3——不包含 v1 的二维坐标/深度 3 通道偏移。颜色使用来自输入 RGB 的基础 SH0 值叠加残差。`merge()` 对近邻属性求平均并做四元数归一化。

---

## 5. 专题：前景深度断层、错误点、遮挡与真正的背景补全

### 5.1 先区分三个不同的错误

设图中前景人像深度 2 m、背景墙深度 5 m。

| 问题 | 现象 | 是否需要恢复被遮挡的真实背景？ |
|---|---|---|
| **A. 深度断层错误** | 人像边缘深度被平滑成 3 m，产生“斜坡表面” | 不一定；主要是可见边界几何错 |
| **B. 悬浮/离群点** | 不应存在的点位于 2–5 m 的空中，侧面看有重影 | 不一定；可先抑制不合理点 |
| **C. 去遮挡缺失** | 视角/物体移动后露出原照片看不到的背景 | **需要**：必须补充真实观测、生成先验或分层补全 |

**重要：避免把“剔除错误点”表述成“补全了遮挡背景”。**

### 5.2 三篇方法对 A/B/C 的真实处理边界

| 方法 | A：断层伪表面 | B：离群悬浮点 | C：原本不可见的背景 |
|---|---|---|---|
| InfiniDepth | 连续查询提高可见边缘深度的表达精度；面积权重可能对高斜率敏感 | 非专门的离群点剪枝模型 | **不直接补全** |
| InfiniSplat v1 | **剔除跨较大相对深度差的局部三角形**，避免错误三角面过度采样 | 几何初始化 + 属性更新可能部分缓解，但没有 v2 的专门不透明度一致性实验 | **不直接补全** |
| InfiniSplat v2 | 不以三角形筛选为主；几何深度错位仍可能保留 | **多视角合并 + Opacity 学习/剪枝**降低一部分离群点贡献 | 若其他输入视角看到背景则有机会重建；**所有输入都看不到则仍缺失** |

### 5.3 v2 的 Opacity 可靠性实验：到底证明了什么？

【论文 §4.3 / Appendix B.4】在 **Hypersim 验证集**，使用 **GT Depth 独立构造几何正确性标签**，不利用 Opacity 来“自证正确”。

- 一个点如果**至少两视角违反 GT 已知自由空间**，且**没有任何视角与真实表面一致**，标为 Geometrically Inconsistent；
- 一个点如果至少两视角表面一致且无自由空间冲突，标为 Surface-consistent；其余标为 Unknown；
- 误差容忍：$\epsilon(d)=\max(0.03\mathrm{m},0.01d)$；**深度边缘样本被排除**，防止 GT 边缘本身不可靠。

在阈值 $\alpha<0.01$ 下：

| 验证结果 | 论文统计 |
|---|---:|
| 被剪掉的“几何不一致点”比例 | **51.8%** |
| 被剪掉的全部高斯比例 | **7.5%** |
| 被保留的“表面一致点”比例 | **93.3%** |

**解释**：网络通过图像重建损失，有机会给部分跨视角冲突点分配更低的 Opacity，剪枝确实具有选择性。

**不能这样推导**：

- 不能说“解决了 51.8% 的前景边缘错误”，因为该实验**排除深度边缘**；
- 不能说“Opacity 就是准确的深度置信度”，它只是渲染贡献参数；
- 不能说“残余 48.2% 都必须再删除”，可能包括复杂的真实几何/标签不确定性。

### 5.4 研究层面的启发：几何修正和影响抑制是两个目标

- **几何修正**：预测 $\Delta D$ 或局部深度残差，使几何点回到正确位置；
- **贡献抑制**：几何可能仍旧错误，但预测一个权重/Opacity 使其在渲染中的贡献降低；
- **内容补全**：重建或生成不曾看见的背景纹理与几何，并处理前景—背景 Alpha 合成。

它们具有不同的监督、评测指标与失败模式，不能只用一个 PSNR 指标替代全部判断。

---

## 6. 放回可控景深虚化任务：SHARP、MPIB、DOF-GS、DoF-Gaussian 的关系

### 6.1 先纠正一个容易混淆的术语

**SHARP** 确实是**单图前馈 3DGS**，目标是单图生成可渲染的新视角。  
**MPIB** 的全称是 *MPIB: An MPI-Based Bokeh Rendering Framework for Realistic Partial Occlusion Effects*（ECCV 2022），核心是 **Multiplane Image，MPI 多平面图像表示**，**不属于 3D Gaussian Splatting**；它直接针对景深虚化与 Partial Occlusion（部分遮挡）建模。

这是它们在研究任务中的关键区别：

| 方法 | 场景表示 | 主要任务 | 是否直接控制景深/焦点 | 重要启发 |
|---|---|---|---|---|
| SHARP | 单图 3DGS | 临近视角新视角合成 | 不以光学虚化为目标 | 低成本单图可渲染表示 |
| MPIB | 多平面 MPI | 虚化合成、半透明部分遮挡 | **是，属于 Bokeh 模型** | 解决前景散焦与背景显露的分层问题 |
| InfiniDepth | 连续深度隐式场；可搭 GS Head | 高精细深度与 NVS 应用 | 否 | 更细深度查询、面积导向采样 |
| InfiniSplat v1 | 单图表面对齐 3DGS | 大基线 NVS | 否 | 高斯布局及深度断层伪表面抑制 |
| InfiniSplat v2 | 多视角 DA3 + 3DGS | 高质量多视角 NVS | 否 | 固定几何、学习属性与错误点贡献 |
| DOF-GS / DoF-Gaussian | 场景特定 3DGS + 离焦模型 | Defocus/Refocus/AiF | **是** | 如何将焦距/光圈/CoC 注入 GS 或像素空间 |

**3DGS 不等于景深虚化**：标准 3DGS 的高斯投影椭圆是一种渲染空间覆盖，不等于物理散焦的 Circle of Confusion（CoC，弥散圆）。如果要得到可控对焦，需要额外引入相机光圈/焦平面/遮挡处理等光学模型。

### 6.2 景深虚化应优先关心哪些环节？

对输入视频、深度与指定主体焦点，典型简化链路是：

```text
RGB 视频 + 深度 D + 对焦主体/距离 + 光圈控制
      ↓
深度边缘可靠性 / 前背景分层 / 去遮挡内容
      ↓
每帧每层计算 CoC（依焦距、焦平面、光圈）
      ↓
光学离焦合成（含部分遮挡与背景显露）
      ↓
时序一致性约束
      ↓
输出 Video Bokeh
```

三篇论文主要对**深度和三维表示的组织**有贡献，并没有给出这条完整的 Bokeh 路线。

### 6.3 我们可以从这条路线借鉴什么（均为【研究推断】，不是作者已验证的 Bokeh 结果）

**方向 A：边缘感知的深度/几何采样**

- 借鉴 v1 的局部深度差阈值，避免跨越前景/背景界面的错误连接；
- 若要构建 GS-based Bokeh，可尝试按深度与表面方向分配高斯采样预算；
- 但应注意：**大面积采样权重不一定适合景深边缘**，否则可能给深度不可靠区域过多的高斯。

**方向 B：几何保持 vs 几何纠错双分支**

- 参考 v2：冻结一个可信几何主干，通过权重/Opacity 学习减少不可靠几何的渲染贡献；
- 另设一个可选 Residual Depth Head，在有几何监督的情况下进行可控修正；
- 不要直接将 RGB 重建损失无约束地作用于所有深度点，否则可能产生不合理的“为拟合图像而移动几何”。

**方向 C：分层遮挡与部分遮挡（MPIB 的研究切入点）**

- 对背景缺失问题，单纯删掉前景错误高斯不会产生正确的背景；
- 需要真实多帧观测、分层重建、局部补全/生成先验，或针对半透明散焦边缘的 Alpha 合成；
- 因此如果研究目标是可控 Bokeh，MPIB 的 Partial Occlusion 思想可能比 InfiniSplat 的某些 NVS 细节更加直接相关。

**方向 D：视频时序约束**

- InfiniSplat v2 的“任意多视角”默认来自**同一静态场景**，不等于不同时间运动物体的动态 4D 重建；
- 若要应用于视频，应显式加入可见性/光流/跨帧对应/动态遮挡约束；
- 特别避免仅对每帧独立剪枝，导致边缘 Opacity 时序跳变、散焦边缘闪烁。

---

## 7. 可执行的最小验证实验建议（不是创新点结论）

| 实验 | 最小干预 | 观察指标 | 试图回答的问题 |
|---|---|---|---|
| E1：深度断层 | 对深度边缘进行局部平滑、偏移、错误连接三类扰动 | 边缘漂浮像素率、深度误差、渲染裂缝、可见性 | v1 式断层筛选是否有帮助？ |
| E2：高斯贡献 | 对错深度点分别采用原样、Opacity 门控、位置残差、门控+残差 | PSNR/LPIPS + 几何误差 + 有效高斯占比 | 只抑制与真纠正分别解决什么？ |
| E3：遮挡缺失 | 相机/人物移动显露背景，区分“另有帧观测”和“完全无观测” | 遮挡边缘 RGB、背景恢复、闪烁 | Opacity 剪枝是否只是隐藏问题，还是实际补全？ |
| E4：视频稳定 | 独立逐帧 vs 跨帧一致 Opacity/Geometry | 时序 LPIPS、边缘闪烁、遮挡后的重现稳定性 | 静态 NVS 思想移植视频的代价？ |
| E5：物理焦点控制 | 固定几何，改变对焦距离和光圈 CoC | 目标对焦区域清晰度、散焦连续性、遮挡关系 | GS 表示能否支持真正的连续可控 Refocus？ |

**实验设计原则**：

1. 不能把论文中的跨数据集 NVS PSNR 直接与 OPPO Video Bokeh 的 PSNR/FD/FVD 横向比较；任务、GT 与可见区域定义均不同。
2. 需要记录 3DGS 生成延迟、单视角渲染延迟、VRAM 峰值、输入深度/估深成本；不要把“渲染 0.009 秒”误当成“完整单图生成 0.009 秒”。
3. 所有失败案例按 A/B/C 三类归因：深度断层、悬浮点、背景缺失；必要时另统计动态物体时间不一致。
4. 先在一个小静态场景验证几何机制，再决定是否值得投入到视频模型训练中。

---

## 8. 开源与复现边界（2026-11-03 核查）

| 项目 | 官方仓库公开 | 推理代码 | 权重 / 数据说明 | 训练/论文全部模块公开程度 |
|---|---|---|---|---|
| InfiniDepth | 是 | 有 | 官方仓库列出 RGB 与传感器增强模型、训练评测 | 仓库宣布训练评测代码公开；需按变体验证依赖与权重 |
| InfiniSplat v1 | 是 | 有 | RGB-only 与 LiDAR 项目入口 | 核心 Encoder/采样/Decoder 源码可读；按具体实验配置复现 |
| InfiniSplat v2 | 是 | **Stage I 推理** | README 给出 Hugging Face 权重、安装与示例 | **论文 Stage II Learned Refiner 的完整公开推理入口未确认**；不默认等于完整论文 Ours* |

**v2 论文与代码的特殊注意事项**：

1. 论文体素边长 0.004（归一化空间），当前公开 `gaussians.py` 是中位高斯尺度 ×1.4；应记录代码版本与实测 voxel size。
2. v2 公开推理具有 `da3-pose`、`pose-free`、`provided` 等不同调用模式，默认未知位姿输入会先执行 DA3 位姿预预测；与论文抽象的 Pose-free/Pose-conditioned 分类核对时要说明具体模式。
3. 论文 `Ours` 是 Stage I；`Ours*` 是 Stage II 三次学习式细化。没有 Stage II 不能声称得到 `Ours*`。
4. 仓库公开不等于所有评测划分/训练脚本/权重完全可复现；做 SOTA 对比要逐项核对。

---

## 9. 最终对照：三篇最核心的实现差异

| 问题 | InfiniDepth | InfiniSplat v1 | InfiniSplat v2 |
|---|---|---|---|
| **深度怎么算？** | 自身 ViT + 隐式连续查询 | 冻结 DepthPro 或 InfiniDepth-Metric | 冻结多视角 DA3 Geometry Branch |
| **高斯放哪？** | NVS 用连续深度点/自适应查询 | 三角形面积指导的非网格支持点 | DA3 多视角反投影点，体素合并 |
| **是否改位置？** | NVS Head 有位置 Offset | **14D 更新中包含位置/深度** | **不改中心，只学属性** |
| **属性如何来？** | 点特征 + Gaussian Head | DINO + CNN 门控 + 共享 MLP | DA3 Attribute Backbone + DPT Head |
| **深度边缘如何处理？** | 连续细粒度深度，未专设遮挡层 | **深度断层三角面筛除** | **通过属性/Opacity 影响渲染** |
| **错误悬浮点如何处理？** | 非主要目标 | 非主要目标 | Voxel Merge、低 Opacity 抑制/剪枝 |
| **缺失背景能否补？** | 不保证 | 不保证 | 多输入可提供补充观测；全不可见区域不保证 |
| **优化与训练关键？** | 连续 GT 坐标深度监督 | 单源图 + 多目标视角的 NVS 监督 | 冻结几何、相机尺度对齐、大角度监督、可选 Learned Refiner |
| **直接控制 CoC/光圈/焦距？** | 否 | 否 | 否 |

**一句话总结这条线：**

> **InfiniDepth 解决“可见表面的深度怎样更精细地查询”；InfiniSplat v1 解决“高斯怎样沿预测表面更合理地分布”；InfiniSplat v2 解决“面对不完美的多视角几何，如何通过属性学习、相机对齐和渲染反馈得到更高质量的可渲染场景”。**
>
> 对可控虚化研究，应该把它们当作**几何与表示学习的技术储备**，不要直接将其当作已解决前景遮挡、背景补全或物理散焦渲染的 Bokeh 模型。

---

## 10. 主要原始来源（可追溯）

1. Hao Yu et al., **InfiniDepth**, CVPR 2026：[arXiv HTML](https://arxiv.org/html/2601.03252)（§3、Appendix A）｜[官方源码](https://github.com/zju3dv/InfiniDepth)。
2. Jiawei Wang et al., **InfiniSplat**, SIGGRAPH Asia 2026 / TOG：[arXiv HTML](https://arxiv.org/html/2608.02437)（§3–5、Table 1–3）｜[官方源码](https://github.com/zju3dv/InfiniSplat/tree/main/InfiniSplat_v1)。
3. Yongzhen Hu et al., **InfiniSplat v2**, arXiv 2026-10：[arXiv HTML](https://arxiv.org/html/2610.20341)（§3–4、Appendix A/B/C）｜[官方源码](https://github.com/zju3dv/InfiniSplat/tree/main/InfiniSplat_v2)。
4. Lars Mescheder et al., **SHARP: Sharp Monocular View Synthesis in Less Than a Second**：[Apple 论文介绍](https://machinelearning.apple.com/research/sharp-monocular-view)｜[官方仓库](https://github.com/apple/ml-sharp)。
5. Juewen Peng et al., **MPIB: An MPI-Based Bokeh Rendering Framework for Realistic Partial Occlusion Effects**, ECCV 2022：[项目与论文](https://juewenpeng.github.io/MPIB/)。

> 后续如需纳入研究仓库，建议路径：`docs/literature/InfiniDepth_InfiniSplat_v1_v2_技术路线对比与虚化启发.md`。此文件目前仅在对话中生成，**未自动提交到 GitHub**。
