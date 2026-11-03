# DoF-Gaussian（CVPR 2025）论文技术细节完整笔记

**论文**：*DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting*  
**作者**：Liao Shen, Tianqi Liu, Huiqiang Sun, Jiaqi Li, Zhiguo Cao, Wei Li, Chen Change Loy  
**会议**：CVPR 2025，pp. 26462–26471  
**正式论文**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_DoF-Gaussian_Controllable_Depth-of-Field_for_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)  
**arXiv**：https://arxiv.org/abs/2503.00746  
**项目主页**：https://dof-gaussian.github.io/  
**官方代码**：https://github.com/leoShen917/DoF-Gaussian

> **记录约定**：正文按正式论文的 §1–6、公式 (1)–(13) 整理；`【论文】`表示论文描述，`【源码】`表示对公开 GitHub 主分支的核对，`【推导】`表示帮助理解的数学说明。两者不一致处单独标注。本文是**每个静态场景分别优化的 3DGS 方法**，不是单张 RGB 端到端直接预测 GS 的通用网络；也不是动态 4DGS。代码与数据公开状态核查日期：2026-11-03。

## 0. 完整数据流（最先看）

输入同一静态场景的**多视角散焦 RGB**。先利用 **COLMAP / SfM（Structure from Motion，运动恢复结构）** 得到相机位姿和稀疏点云，初始化一组可优化的 **3D Gaussians**。另外，将稀疏点投影回各训练图像形成稀疏深度图，用这些具有当前场景几何约束的深度微调一个单目深度网络，生成该场景各视角的**稠密深度先验**，随后对齐其尺度。

在 GS 主训练中，逐次采样训练视角 $m$，所有视角**共享同一组三维高斯**，但各自具有可学习的光圈参数 $\mathcal A_m$ 和对焦位置 $\mathcal F_m$。当前 GS 先通过通常的 3DGS 光栅化得到**未施加镜头散焦的 RGB 图 $C_m$ 与深度图 $D_m$**；然后根据每个像素深度 $d$ 及 $(\mathcal A_m,\mathcal F_m)$ 计算 CoC（Circle of Confusion，弥散圆）尺度，通过基于邻域权重的可微 **Scatter（颜色散射）**，将源像素颜色贡献分配到附近像素并归一化，得到散焦图像 $C_m^*$。用 $C_m^*$ 与原始散焦照片 $I_m$ 计算 RGB 损失。与此同时，GS 的渲染深度 $D_m$ 与稠密先验 $D_{\mathrm{pred},m}$ 比较，并增加几何法线等约束。

在前期训练中，模型学习场景和镜头的散焦成像关系；到一定训练迭代之后，通过当前深度与对焦位置的差异生成**清晰区域相关的权重图**，对较清晰、较可靠的观测给予更大重建权重。这样不需要单独的全清晰 RGB GT，也不需要 ILN 之类的清晰区域预测网络。30,000 次迭代后，可对同一场景指定**新的相机视角、对焦位置及光圈参数**，重新渲染景深效果；令光圈参数为 0 时，不再额外施加散焦，得到全清晰候选渲染。

```text
多视角散焦 RGB: I_1, I_2, ... I_M（静态场景）
      │
      ▼
COLMAP / SfM → 相机内外参 + 稀疏三维点云
      │
      ├─────────▶ 初始化共享 3D Gaussians S
      │                        │
      │                        └──────────────┐
      ▼                                       │
投影稀疏点 → 各视角稀疏深度 D_sparse           │
      │                                       │
微调单目深度网络（per-scene）                   │
      │                                       │
预测稠密深度 + COLMAP 尺度对齐                 │
      ▼                                       ▼
深度先验 D_pred,m                   从 S 在视角 m 光栅化
      │                             ├─ RGB C_m（无额外镜头散焦）
      │                             └─ Depth D_m
      │                                       │
      └── L_depth(D_m,D_pred,m)                 ├─ 学习 (A_m,F_m)
                                            │
                                            ▼
                         r(d)=A_m × |1/F_m − 1/d|
                                            │
                           逐像素 CoC + 邻域散射/归一化
                                            │
                                            ▼
                                   散焦 RGB C*_m
                                            │
                              与真实散焦 I_m 比较 L_rec
                                            │
                           10k 后使用清晰区域权重 L_focus
                                            │
                         + 深度、法线等损失，共同优化 GS
                                            │
                              30k 迭代 → 场景 GS*
                                            │
                              新视角 + 新 A + 新 F
                                            ▼
                                  全清晰/散焦 RGB
```

**三种深度不可混淆：**

| 名称 | 来源 | 用途 |
|---|---|---|
| $D_{\mathrm{sparse}}$ | COLMAP 三维稀疏点投影 | **微调深度网络**的稀疏监督 |
| $D_{\mathrm{pred}}$ | 经过当前场景微调的深度网络输出 | 稠密先验，**监督 GS 几何** |
| $D$ | 当前共享 GS 的光栅化输出 | **真正用于散焦计算的深度**，并接受深度先验监督 |

**关键理解**：它不是把 $D_{\mathrm{pred}}$ 直接替换为渲染深度；主要理由是要让光学散焦使用的深度、RGB 内容、新视角几何最终归属于同一组 GS。

---

## 1. Introduction：问题与贡献（§1）

### 1.1 问题定义

标准 3DGS 建模针孔相机，较适合全清晰多视角输入；现实照片经常存在浅景深，离焦高斯若被当作普通图像纹理拟合，可能破坏底层清晰场景表示。作者希望实现：

- 用多视角**散焦** RGB 恢复尽可能清晰的 3DGS 场景；
- 在新的虚拟相机位姿下做 Novel View Synthesis（新视角合成）；
- 重新指定光圈、对焦距离，产生可控制的景深，并支持不同 CoC 形状。

### 1.2 三个难点及方案

| 难点 | 具体影响 | 对应设计 |
|---|---|---|
| 针孔成像无法模拟真实散焦 | GS 可能通过模糊自身颜色或几何去拟合输入 | Lens-based imaging model（镜头成像 + 可微散焦） |
| 散焦输入降低 SfM/GS 几何质量 | 错误深度导致 CoC 不准确 | Per-scene adjustment of depth priors（逐场景深度先验） |
| 理想圆盘核与真实镜头不完全一致 | 全局拟合散焦可能牺牲清晰纹理 | Defocus-to-focus adaptation（后期清晰区域重点监督） |

另外构建用于学习**光圈/对焦参数真值**和 Refocusing（重对焦）效果评价的合成数据集。

---

## 2. Related Work（§2）

- **Depth-of-Field Rendering（景深渲染）**：通常从全清晰 RGB（有时加深度）生成可控散焦，但不必恢复统一的三维场景。
- **Image Deblurring（图像去模糊）**：将模糊输入映射为清晰输出，通常没有可解释的镜头参数接口；Motion Blur 与 Defocus Blur 的成因不同。
- **Novel View Synthesis（新视角合成）**：NeRF/3DGS 在场景表达和新视角渲染方面表现良好，但针对散焦输入和事后光圈控制需要专门机制。

本文将**三维场景重建 + 可微散焦 + 光学参数可控**整合在一起。

---

## 3. Preliminary：3DGS 基础（§3，Eqs. 1–4）

每个三维高斯有中心 $\boldsymbol\mu$、协方差 $\Sigma$、不透明度、SH 颜色等参数：

$$G(\mathbf X)=\exp[-\tfrac12(\mathbf X-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf X-\boldsymbol\mu)]\tag{1}$$

$$\boxed{\Sigma=RSS^\top R^\top}\tag{2}$$

- $R$：高斯朝向；$S$：三个轴向尺度。
- 投影到图像平面后二维高斯协方差为：

$$\boxed{\Sigma'=JW\Sigma W^\top J^\top}\tag{3}$$

- $W$ 是相机空间变换的线性部分，$J$ 是透视投影的局部雅可比。

按深度顺序做 Alpha Compositing：

$$C(\mathbf x)=\sum_i c_i\alpha_i\prod_{j<i}(1-\alpha_j)\tag{4}$$

**区别于后文**：这里 $C$ 是原始 GS 渲染出的 RGB，**还没有施加镜头虚化**。后面才用 $C,D$ 生成散焦 $C^*$。

---

## 4. Method 总览（Figure 2）

整个 Method 分三部分：

1. **§4.1 Lens-based Depth-of-Field Model**：定义镜头/CoC，用二维 RGB + Depth 的颜色散射生成散焦。
2. **§4.2 Per-Scene Adjustment of Depth Priors**：以 COLMAP 稀疏深度适配单目深度网络，监督 GS 几何。
3. **§4.3 Defocus-to-Focus Adaptation**：训练后期强调清晰区域，提高底层 GS 纹理质量。

---

## 5. §4.1 镜头成像模型与可微散焦（Eqs. 5–7；Figure 3、Algorithm 1）

### 5.1 CoC 如何由深度与镜头参数控制？

$$\boxed{r(d)=\mathcal A\left|\frac1{\mathcal F}-\frac1d\right|}\tag{5}$$

| 符号 | 含义 |
|---|---|
| $d$ | 当前源像素处的 **GS 渲染深度** |
| $\mathcal F$ | 对焦距离 / 对焦位置参数（Focus Distance） |
| $\mathcal A$ | 光圈相关参数（Aperture Parameter） |
| $r(d)$ | 控制源像素散射覆盖范围的 CoC 尺度 |

当 $d=\mathcal F$ 时 $r(d)=0$；光圈参数增大时非焦平面位置的虚化更强。这里的光学参数**不应未经标定就解释为真实镜头的毫米或 f-number**；公开实现尤其使用归一化的逆深度（见 §10）。

**后期调整对焦位置不是学习一个独立分割 Mask**：推理时直接给新的 $\mathcal F$，重新计算每个像素的 CoC；给新的 $\mathcal A$，控制整体离焦强度。模型对**同深度的区域**通常都会保留相似对焦状态；不能仅凭这个公式任意令同深度的一个物体清晰、另一个模糊。

### 5.2 为什么先渲染 RGB 和 Depth？

给定 GS 场景 $\mathcal S$、当前相机位姿 $P_m$：

$$C_m=\operatorname{RasterizeRGB}(\mathcal S,P_m)$$
$$D_m=\operatorname{RasterizeDepth}(\mathcal S,P_m)$$

然后使用各像素的 $D_m$ 进行散焦。**模糊发生在 RGB/Depth 光栅化完成之后，而不是直接修改每个投影高斯的协方差。**

### 5.3 Algorithm 1：颜色散射 Scatter

思路是：源像素 $i$ 根据其深度 $d_i$ 计算 CoC 尺度 $r(d_i)$，再向距离小于相应 CoC 的目标像素 $j$ 分配颜色贡献。作者用平滑可微的 $\operatorname{Func}$ 代替硬圆盘判定：

$$\boxed{\operatorname{Func}(d,l)=\frac12+\frac12\tanh\left(\alpha[r(d)-l]\right)}\tag{6}$$

- $l=\|i-j\|$：两个像素中心的距离；
- $\alpha$：过渡边界的陡峭程度；**论文实验 $\alpha=4$**；
- $l\ll r(d)$ 时权重接近 1；$l\gg r(d)$ 时趋近 0。

对每一个目标像素，累积颜色及权重并归一化（以下为省略 Gamma 变换的**解释性简写**）：

$$\lambda_{ij}=\operatorname{Func}(d_i,\|i-j\|),\quad C_j^*\approx\frac{\sum_i\lambda_{ij}C_i}{\sum_i\lambda_{ij}}$$

**Algorithm 1 的完整实现概念**：先执行 $C^\gamma$，将散射结果累积到 $C^*$、权重累积到 $\Phi$，输出 $\big(C^*/\Phi\big)^{1/\gamma}$。论文允许改变 CoC 支持形状，例如近似六边形光斑。

### 5.4 对比真实散焦照片的重建损失

$$\boxed{\mathcal L_{\mathrm{rec}}=(1-\lambda)\mathcal L_1(I,C^*)+\lambda\mathcal L_{\mathrm{D\text{-}SSIM}}(I,C^*)}\tag{7}$$

论文使用 $\lambda=0.2$。$I$ 是**真实散焦训练照片**；$C^*$ 是模型模拟散焦后的图像，**不是**把散焦结果叠加到 $I$ 上。这里没有全清晰 RGB GT。

**关闭镜头散焦**：推理时设置 $\mathcal A=0$，得到无额外 CoC 散焦的 RGB（其真实清晰程度取决于训练出的 GS）。

---

## 6. §4.2 逐场景深度先验（Eqs. 8–9；Figure 4）

### 6.1 为什么不直接使用普通单目深度？

深度网络对模糊输入可能出现系统误差，并且预测尺度与 COLMAP 的场景尺度未必一致。作者用当前场景稀疏几何进行**适配而不是直接把原始网络输出当成 GT**。

### 6.2 COLMAP 稀疏深度的形成

世界坐标系点 $\mathbf P_k$ 通过第 $m$ 个相机外参：

$$\mathbf p_{k,m}^{cam}=R_m\mathbf P_k+t_m,\qquad z_{k,m}=(\mathbf p_{k,m}^{cam})_z$$

投影到像素 $(u_k,v_k)$ 后，仅在有效对应位置设置稀疏深度 $D_{\mathrm{sparse}}(u_k,v_k)=z_{k,m}$，其余像素不参加稀疏监督。注意 SfM 可能具有整体尺度歧义。

### 6.3 场景微调和 SiLog

$$D_{\mathrm{pred},m}=F_\theta(I_m)$$

论文公式 (8)：

$$\boxed{\mathcal L_{\mathrm{silog}}=\frac1{2M}\sum_{i=1}^{M}\left[\log(e^s D_{\mathrm{pred},i})-\log D_{\mathrm{sparse},i}\right]^2}\tag{8}$$

- $M$：有效稀疏监督像素数；$e^s$：尺度对齐项；
- 使用对数深度可降低全局尺度不一致的干扰；
- **【推导】**当 $s$ 可自由优化时，它等价于消除对数深度残差的均值，从而强调相对深度结构。

### 6.4 稠密先验监督 GS 的深度

$$\boxed{\mathcal L_{\mathrm{depth}}=\|D-D_{\mathrm{pred}}\|_2}\tag{9}$$

- $D$ 来自 GS，直接参与 CoC 计算。
- $D_{\mathrm{pred}}$ 是先验，目标是约束 GS 几何，不是用于推理的固定外部深度输入。
- 论文参考 Rade-GS 的深度光栅化技术。
- 【源码】公开 `train.py` 深度损失具体用 `l1_loss`，后期为 `l1_loss_focus`，与正文 Eq. (9) 的范数形式不同。

### 6.5 为什么不直接拿 $D_{\mathrm{pred}}$ 替换 $D$？

如果这样做，每个训练视角可能用一张互相不严格一致的单目预测深度来控制散焦，而 RGB 仍来自共享 GS；在新视角上又需要重新估深。本文希望散焦深度与 RGB、三维几何来自同一组可优化 GS。**不过这不是“GS 深度必然比先验精确”的证明；需要消融才能比较。**

### 6.6 官方训练顺序

【源码】`train.py` 先执行 `depth_priors.train(args)`，保存/读取微调网络预测的深度图并做 `align_scales`，再进入 GS 的主迭代。它**不是**每次更新 GS 都同时重新训练一次深度网络。参考：[train.py](https://github.com/leoShen917/DoF-Gaussian/blob/main/train.py)、[depth_priors.py](https://github.com/leoShen917/DoF-Gaussian/blob/main/bokeh_renderer/depth_priors.py)。

---

## 7. §4.3 Defocus-to-Focus Adaptation（Eqs. 10–13）

### 7.1 动机与两个阶段

真实相机的点扩散函数（PSF）不一定符合本文的理想散焦近似。若始终强制拟合整张照片的所有散焦位置，GS 的本底纹理可能反而变模糊。因此先学习镜头成像，再重点监督可靠的清晰区域。

- **0–10k 迭代：**通过普通散焦 RGB 重建和深度监督学习 GS/镜头参数。
- **10k–30k 迭代：**在 RGB、深度损失中加入对焦相关的像素权重，更多关注输入图中相对清晰的部分；**仍然使用原始散焦 RGB 作为 GT**。

### 7.2 权重与对焦偏差

令：

$$x=\left|\frac1{\mathcal F}-\frac1d\right|$$

论文公式 (10)：

$$\boxed{\Psi(x)=\begin{cases}1,&n<t\\[3pt](1+e^{-a(x-b)})^{-1},&n\ge t\end{cases}}\tag{10}$$

论文设置 $t=10000,a=15,b=0.3$。严格按公式，$x$ 越大（离焦越强）$\Psi$ 越大。论文还用这个函数调整每像素光圈：

$$\boxed{\mathcal A'=\mathcal A\Psi(x)}\tag{11}$$

这对于压低对焦附近像素的光圈系数有合理的数学方向。但**与正文“更强调清晰区域”的损失权重表述直接照抄会出现语义方向冲突**，应结合公开代码理解。

### 7.3 【源码关键差异】焦区损失使用 $1-\Psi$，不是直接 $\Psi$

`utils/loss_utils.py`：

```python
# 核心逻辑的节选（官方实现）
x = (x-x.min())/(x.max()-x.min())
weight = 1/(1+torch.exp(-15*(x-0.3)))
focus_weight = 1-bokeh_weight(torch.abs(disps-disp_focus.detach()))
```

因此，**实际用于加权 L1 损失的是 $W_{\mathrm{focus}}=1-\Psi$**（并在计算前对视差等量归一化）：越靠近焦平面，权重越大。

### 7.4 【源码关键差异】光圈自适应在主训练调用路径未启用

`bokeh_renderer/utils.py` 有 `render_bokeh()` 与 `render_bokeh_ex()`：

- `render_bokeh_ex()` 在 10k 后执行 `defocus = defocus * bokeh_weight(...)`；
- 主训练 `train.py` 实际调用 `render_bokeh()`；其中相似的乘权代码被**注释**。

所以论文 Eq. (11) 描述的是一种设计，但按公开主训练入口执行，**不能默认认为该像素级光圈再加权已经启用**。后续做严格复现时要记录调用路径。

### 7.5 几何法线约束

$$\boxed{\mathcal L_{\mathrm{normal}}=\sum_i w_i(1-\mathbf n_i^\top\hat{\mathbf n}_i)}\tag{12}$$

- $\mathbf n_i$：Gaussians / splat 给出的几何法线。
- $\hat{\mathbf n}_i$：渲染深度图经有限差分计算的法线。
- 当两者方向一致时，余弦接近 1，损失较小；作用主要是改善表面几何一致性。

### 7.6 论文总损失与代码实际损失

论文公式 (13)：

$$\boxed{\mathcal L=\Psi\odot(\mathcal L_{\mathrm{rec}}+w_d\mathcal L_{\mathrm{depth}})+w_n\mathcal L_{\mathrm{normal}}}\tag{13}$$

实验设置 $w_d=0.01,w_n=0.05$，$\odot$ 是逐元素加权。**注意 Eq. (10) 的符号方向与 Eq. (13) 的 Focus 语义问题。**

公开 `train.py` 中实际组合为：

```python
rgb_loss = (1 - lambda_dssim) * Ll1_render + lambda_dssim * (1 - ssim(...))
loss = rgb_loss + lambda_depth * depth_loss \
       + lambda_depth_normal * depth_normal_loss \
       + lambda_distortion * distortion_loss
```

- 10k 后 **L1 RGB** 与 **L1 depth** 调用 `l1_loss_focus`，SSIM 不同样做逐像素 Focus 加权。
- `lambda_depth = opt.lambda_depth * (1-iteration/opt.iterations*0.9)`，深度损失权重逐渐下降。
- 主训练包含 Depth Distortion 与 Normal Consistency 项（依据开关/权重配置）。
- **特别重要的梯度细节**：`render_bokeh()` 中 `disps = disp_map[0].detach()`，因此散焦 RGB Loss **不会直接沿 CoC/视差分支对 GS 渲染深度求导**；这与论文概念上“镜头可微”的笼统说法应分开。RGB 仍可沿渲染颜色路径更新 GS，深度也通过深度损失与几何损失更新。

**总结**：这里没有一个单独将模糊 RGB 转成清晰 RGB 的 U-Net，也没有整张 AiF GT；改善清晰度依赖多视角中已有的清晰部分以及损失加权。

---

## 8. Section 5：实验设置、数据与评价

### 8.1 训练设置（论文）

| 项目 | 设定 |
|---|---|
| 场景优化 | **每场景 30,000 次迭代** |
| Focus Adaptation | **第 10,000 次附近开始** |
| 平滑散射参数 | $\alpha=4$ |
| Sigmoid 参数 | $a=15,b=0.3,t=10,000$ |
| 深度/法线权重 | $w_d=0.01,w_n=0.05$（论文设置） |
| 基础框架 | Mip-Splatting / 3DGS + Rade-GS 几何相关模块 |
| 光学参数 | 每个训练视角可学习一组 $(\mathcal A_m,\mathcal F_m)$ |

### 8.2 使用哪些数据？

**真实散焦重建评测**：Deblur-NeRF 的 10 个 Real Defocus Blur 场景（Cake、Caps、Cisco、Coral、Cupcake、Cups、Daisy、Sausage、Seal、Tools）。这些场景有多视角输入及相应评价数据。

**其他场景**：Real Forward-facing（LLFF 等）与 Tanks and Temples（T&T）用于清晰输入/泛化实验。

**作者新合成的可控 DoF 数据集**（§5.1、Figure 6）：在原有多视角图像基础上，用深度估计和单图 DoF 渲染方法生成不同对焦位置、光圈下的散焦输入，同时保存已知的镜头参数作为 GT。新合成数据用于比较学习的镜头参数和重对焦质量，并非真实镜头精确标定数据。

> **开源情况（2026-11-03 核查）**：官方 GitHub 的 Preparation 链接提供了**原始 Deblur-NeRF 数据**，但**未找到作者新合成评测集的完整处理版下载入口**；GitHub [Issue #6](https://github.com/leoShen917/DoF-Gaussian/issues/6) 仍在请求公开处理后的 Real Forward-facing / T&T 等数据。可以依据补充材料构建相似数据，但不能声称其协议与作者原始 benchmark 完全一致。

### 8.3 评价指标

- **PSNR**（Peak Signal-to-Noise Ratio，峰值信噪比）：越大越好；衡量像素重建误差。
- **SSIM**（Structural Similarity，结构相似性）：越大越好；衡量局部亮度、对比度及结构相似性。
- **LPIPS**（Learned Perceptual Image Patch Similarity，感知距离）：越小越好。
- **$\delta_{\mathcal A}$ 与 $\delta_{\mathcal F}$**：合成数据上学习的光圈、对焦位置与已知设置之间的平均绝对误差，越低越好。
- 论文没有为本文提供动态视频时序稳定性指标；其核心输入是**静态多视角场景**。

### 8.4 Table 1：Deblur-NeRF 真实散焦的全清晰新视角重建

**10 场景均值**（DoF-Gaussian 的三项均值可由其 Table 1 的 10 行直接平均，四舍五入）：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Deblur-NeRF | 23.47 | 0.720 | 0.121 |
| DoF-NeRF | 22.70 | 0.682 | 0.185 |
| DP-NeRF | 23.67 | 0.730 | 0.108 |
| BAGS | 23.95 | 0.754 | 0.094 |
| Deblurring 3DGS | 23.71 | 0.747 | 0.107 |
| **DoF-Gaussian** | **23.97** | **0.755（约 0.756）** | **0.093** |

DoF-Gaussian 的平均 PSNR 比 BAGS 仅高约 **0.02 dB**：不能描述为所有场景大幅领先；其重要优势还包括可控镜头散焦。论文 Table 1 未与 DOF-GS 按统一协议直接比较。

### 8.5 Table 2：作者合成可控 DoF 数据集

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | $\delta_{\mathcal A}$ ↓ | $\delta_{\mathcal F}$ ↓ |
|---|---:|---:|---:|---:|---:|
| DoF-NeRF | 25.59 | 0.788 | 0.207 | 0.196 | 0.256 |
| **DoF-Gaussian** | **28.70** | **0.864** | **0.095** | **0.126** | **0.079** |

在作者的合成协议内，提升新视角散焦图像质量并减小镜头参数估计误差；**不能外推到真实标定镜头的绝对距离精度**。

### 8.6 Table 3：全清晰输入设置

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Mip-Splatting | 27.05 | 0.893 | **0.115** |
| DoF-Gaussian | **27.81** | **0.902** | 0.117 |

表明模型在全清晰输入下仍可以竞争，但 LPIPS 不优于 Mip-Splatting。

### 8.7 Table 4：组件消融（重点）

| 变体 | Lens | Depth | Adaptation | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|:---:|:---:|:---:|---:|---:|---:|
| #1 | × | × | × | 21.31 | 0.636 | 0.239 |
| #2 | ✓ | × | × | 23.05 | 0.728 | 0.109 |
| #3 | ✓ | × | ✓ | 23.59 | 0.742 | 0.104 |
| #4 | ✓ | ✓ | × | 23.42 | 0.738 | 0.098 |
| **#5 Full** | ✓ | ✓ | ✓ | **23.97** | **0.756** | **0.093** |

**可得出的直接差异**：

- 加入 Lens 模型：#1→#2，PSNR **+1.74 dB**。
- 加入 Focus Adaptation（有 Lens，无 Depth）：#2→#3，**+0.54 dB**。
- 加入 Depth Prior（有 Lens，无 Adaptation）：#2→#4，**+0.37 dB**。
- Full 相对基础模型：#1→#5，**+2.66 dB**。

不同模块增益不能任意线性叠加。**镜头模型贡献最明显，深度约束和清晰区域加权均有额外帮助。**

### 8.8 Table 5：不同深度监督对比

| 方案 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 未进行逐场景微调的深度 | 23.53 | 0.739 | 0.114 |
| COLMAP 稀疏深度直接监督 | 23.44 | 0.737 | 0.113 |
| **逐场景微调的稠密深度** | **23.97** | **0.756** | **0.093** |

这证明在其测试配置下，**利用 SfM 稀疏深度微调后生成的稠密先验**更有效。但这些是 RGB 重建指标，不等于所有像素的深度 GT 误差被定量证明更低。

### 8.9 Figure 7：应用

能在训练后设置不同对焦位置、光圈强度、相机位姿和局部放大，还能变更 CoC 形状以产生不同形状的散焦光斑。静态场景中沿轨迹渲染视频不意味着支持随时间运动物体的动态 4D 重建。

### 8.10 补充材料中的效率（指定硬件/协议）

论文报告在 RTX A6000 上 GS 训练约 **18 min**、渲染约 **364 FPS**，逐场景深度网络微调另约 **3 min**。这些是特定硬件、分辨率和计时范围下的数据，**不是全流程含 COLMAP、数据预处理、编译、磁盘 I/O 的统一端到端耗时，也不适合与不同论文的 FPS 直接比较**。

---

## 9. Section 6：结论与局限

**主要结论**：通过显式镜头散焦、场景深度先验和后期清晰区域监督，实现从浅景深多视角输入进行场景清晰重建，并支持事后改变对焦/光圈等摄影控制。

**明确边界**：

1. **全部视角都模糊的位置难以恢复真实细节**；焦区加权并不会凭空补出从未观测到的高频纹理。
2. 深度先验是伪标签，不是真实 GT；复杂场景下它可能误导几何，所以公开代码逐渐降低其权重。
3. 镜头模型是理想近似，不能等同真实镜头全部物理 PSF、像差或复杂遮挡成像。
4. 当前重点为静态多视角场景，不提供动态前后景运动一致性保证。
5. 纸面上声明的部分可微/加权流程和公开主训练代码存在差异，必须以检查代码路径为准。

---

## 10. 论文与源码对应关系（复现时必看）

| 论文模块 | 主要源码 | 核查重点 |
|---|---|---|
| 训练入口 | [`train.py`](https://github.com/leoShen917/DoF-Gaussian/blob/main/train.py) | 先训练深度先验、再进行 GS 迭代；30k；10k 后 loss_focus |
| GS 渲染输出 | [`gaussian_renderer/`](https://github.com/leoShen917/DoF-Gaussian/tree/main/gaussian_renderer) | `rendered_image`, `rendered_depth`, normal, distortion 等 |
| 光学散焦 / Scatter | [`bokeh_renderer/utils.py`](https://github.com/leoShen917/DoF-Gaussian/blob/main/bokeh_renderer/utils.py) | `render_bokeh()`、`render_bokeh_ex()`、归一化视差、`detach` |
| Scatter CUDA | [`bokeh_renderer/`](https://github.com/leoShen917/DoF-Gaussian/tree/main/bokeh_renderer) | `ModuleRenderScatter` / 可微邻域散射 |
| 深度网络适配 | [`bokeh_renderer/depth_priors.py`](https://github.com/leoShen917/DoF-Gaussian/blob/main/bokeh_renderer/depth_priors.py) | COLMAP 有效深度、场景级微调、输出缓存 |
| Focus Loss | [`utils/loss_utils.py`](https://github.com/leoShen917/DoF-Gaussian/blob/main/utils/loss_utils.py) | `bokeh_weight()`、`l1_loss_focus()`；真实权重为 $1-\Psi$ |
| 学习的镜头变量 | [`scene/gaussian_model.py`](https://github.com/leoShen917/DoF-Gaussian/blob/main/scene/gaussian_model.py) | `get_K_bokeh`, `get_disp_focus` |
| 评测入口 | [`metric.py`](https://github.com/leoShen917/DoF-Gaussian/blob/main/metric.py) | PSNR/SSIM/LPIPS，Deblur-NeRF 评测选项 |
| 渲染入口 | [`render.py`](https://github.com/leoShen917/DoF-Gaussian/blob/main/render.py) | 指定镜头参数和输出路径 |

### 10.1 源码里的光学参数是怎么传的？

公开训练代码中，调用形式为：

```python
render_bokeh(rendered_image, rendered_depth,
             K_bokeh=10*K_bokeh[random_view],
             gamma=2,
             disp_focus=disp_focus[random_view],
             defocus_scale=1, iteration=iteration)
```

并先根据渲染深度：

```python
disp_map = 1. / torch.max(1e-10 * torch.ones_like(rendered_depth), rendered_depth)
disps = disp_map[0].detach()
disps = torch.clamp((disps-disps.min()) /
                    (torch.quantile(disps, 0.95)-disps.min()),
                    min=1e-5, max=1.0)
signed_disp = disps - disp_focus
defocus = K_bokeh * signed_disp / defocus_scale
```

由此可知：

- `K_bokeh` 直接控制 CoC / 散焦尺度；
- `disp_focus` 在**归一化逆深度（视差）空间**设定对焦位置，**并非可以不做转换就传入“米”**；
- 使用了 95% 分位数与 min-max 归一化；修改预处理会影响镜头参数的可比性；
- `disps.detach()` 截断散焦支路到 GS 深度的梯度，**源码实际梯度行为与“理论上完全可微”要分开分析**。

### 10.2 可复现性与数据公开的限制

官方 GitHub 仓库公开，`README.md` 给出 clone、依赖、COLMAP、Deblur-NeRF 数据、训练和评测命令。但补充数据/定制光学应用的发布范围不等于论文所有数据和功能都已打包：README 的 `Applications` 仍写着 `coming soon`；作者合成 benchmark 未发现完整公开下载。代码仓库能 clone **不等于已验证当前机器上无错运行**。

---

## 11. 技术参数速查与常见误区

| 容易误解的说法 | 准确说法 |
|---|---|
| “模型逐张图像预测各自独立的 3DGS” | **所有训练视角共享一个 GS 场景**，逐场景优化 |
| “估深网络 $D_{pred}$ 直接控制虚化” | $D_{pred}$ 主要**监督 GS**；散焦使用 $D=\operatorname{RenderDepth}(GS)$ |
| “直接把模糊结果加在原始照片上” | 对 GS 渲染的清晰 RGB **进行像素散射和归一化**，生成合成散焦 RGB，再与原图监督 |
| “修改光圈就能任意选中某个物体” | 原生通过**对焦距离/视差平面**调焦，不是任意 2D 实例 Mask 控制 |
| “没有全清晰 GT 就没有清晰监督” | 清晰区域的**高权重散焦重建误差**对 GS 提供间接清晰纹理监督 |
| “论文完全可微，所以深度梯度必能穿过 CoC” | **公开主渲染实现对 `disps` 做了 `detach()`**，特定路径并不回传到 Depth |
| “A=0 后一定得到真实清晰图” | 只保证**不再额外加镜头散焦**，不保证底层 GS 的丢失细节可以恢复 |
| “合成 benchmark 已下载” | 原始 Deblur-NeRF 可得；**新光学参数合成 benchmark 尚无已确认完整公开入口** |

---

## 12. 原文公式、图表与权威资料索引

**论文公式**：Eqs. (1)–(4) 标准 3DGS；(5) CoC；(6) 可微散射权重；(7) RGB 重建；(8) 稀疏深度 SiLog；(9) GS 深度先验；(10) Focus 权重；(11) 后期光圈加权；(12) 法线一致性；(13) 总损失。

**论文图表**：Figure 1 功能；Figure 2 全流程；Figure 3 镜头模型；Algorithm 1 颜色散射；Figure 4 几何先验可视化；Figure 5 定性对比；Figure 6 合成数据；Figure 7 交互；Figure 8 消融；Table 1–5（真实数据、合成参数、全清晰输入、模块消融、深度先验消融）。

**一手资料**：

- 正式论文：https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_DoF-Gaussian_Controllable_Depth-of-Field_for_3D_Gaussian_Splatting_CVPR_2025_paper.pdf
- arXiv（含补充）：https://arxiv.org/pdf/2503.00746
- 项目主页：https://dof-gaussian.github.io/
- 官方实现：https://github.com/leoShen917/DoF-Gaussian
- 数据 README：https://github.com/leoShen917/DoF-Gaussian#preparation
- 数据集公开问题：https://github.com/leoShen917/DoF-Gaussian/issues/6

**一句话记忆**：**DoF-Gaussian = 多视角 GS 场景 + 逐场景稠密深度先验 + 光栅化后按逐像素 CoC 做颜色散射 + 训练后期清晰区域加权 + 光学参数可控。**
