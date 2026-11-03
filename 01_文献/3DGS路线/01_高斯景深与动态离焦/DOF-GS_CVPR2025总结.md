# DOF-GS（CVPR 2025）论文技术细节完整笔记

**论文**：*DOF-GS: Adjustable Depth-of-Field 3D Gaussian Splatting for Post-Capture Refocusing, Defocus Rendering and Blur Removal*  
**作者**：Yujie Wang, Praneeth Chakravarthula, Baoquan Chen  
**会议**：CVPR 2025，pp. 21297–21306  
**正式论文**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DOF-GS_Adjustable_Depth-of-Field_3D_Gaussian_Splatting_for_Post-Capture_Refocusing_Defocus_CVPR_2025_paper.pdf)  
**项目**：https://dof-gs.github.io/  
**官方代码**：https://github.com/rongduo/DOFGS_implementation

> 记录约定：以 **CVPR 2025 正文**为主要依据；涉及公开代码的实现细节会明确标为**官方源码/特定配置**；一般 3DGS 数学补充标为**基础原理**。不将推导、源码选项、图像例子当作论文中所有实验的固定配置。

## 0. 完整数据流（先看这一部分）

输入**同一个静态场景、不同视角、具有不同程度散焦模糊的多张 RGB 图像**。首先通过 **COLMAP / SfM（运动恢复结构）** 从多视角图像估计相机位姿和稀疏三维点云，用点云初始化一组 **3D Gaussians**，同时为每个训练视角初始化可学习的**对焦距离** $f_m$ 和**光圈控制参数** $Q_m$。

训练时，每次采样一个视角 $m$，使用当前 3DGS 和该视角的相机参数进行两路渲染：**第一路是散焦渲染**，先将三维高斯投影为二维高斯，根据各高斯中心的相机坐标深度 $z_k$ 及 $f_m,Q_m$ 计算 **CoC（弥散圆）半径**，将 CoC 转换为二维高斯模糊核，通过**二维协方差相加**把散焦作用到每一个高斯上，再进行 Rasterization 和 Alpha Compositing，得到散焦 RGB 图像 $\tilde I_m$，并同时生成 **Depth Map** 和 **CoC Map**。散焦图像直接与原始训练照片 $I_m$ 比较，形成重建监督。

**第二路是全清晰（AiF）渲染**，对相同的高斯场景设置 $Q^*=0$，取消额外的光学散焦核，输出全清晰候选图像 $\tilde I_m^*$。将第一路的散焦 RGB、Depth Map、CoC Map、视角编号编码和像素位置编码送入 **ILN（In-Focus Localization Network，清晰区域定位网络）**，得到清晰区域 Mask $\mathcal M_m^*$。用 Mask 按像素融合全清晰图与散焦图，再与真实原始照片比较，计算细节增强损失；同时利用 CoC 与 Mask 的相关性损失和 Mask 正则项提供约束。随后通过反向传播**共同优化 3DGS 参数、逐视角相机光学参数和 ILN 权重**。所有训练视角共享同一组三维高斯，迭代优化后获得具有尽可能清晰细节的三维场景。

推理/渲染时，**不需要再次输入训练照片，也不需要 ILN 重新预测 Mask**；使用已经优化的 3DGS，指定相机位置、对焦距离 $f$ 和光圈参数 $Q$，运行可微 DOF Rasterizer，即可生成**新的视角、全清晰图像或不同程度的景深散焦图像**。

```text
多视角散焦 RGB：I_1, I_2, ..., I_M（同一个静态场景）
     │
     ▼
COLMAP / SfM → 相机位姿 {R_m,t_m} + 稀疏三维点云
     │
     ├── 初始化 3D Gaussians：S = {u_k,q_k,s_k,o_k,h_k}
     └── 每个训练视角初始化可学习参数 (f_m,Q_m)

训练：每次采样一个视角 m，共享当前 GS 场景 S
     │
     ├────────────────── 路线 A：散焦渲染 ──────────────────┐
     │    3D Gaussian → 相机坐标/深度 z_k → 投影为 2D Gaussian
     │    → R_k = Q_m/2 × |1/z_k - 1/f_m| → CoC Blur Kernel
     │    → Σ''_k = Σ'_k + a_k I → Alpha 合成
     │    → 散焦 RGB Ĩ_m + Depth Map D̃_m + CoC Map M_coc
     │       │                                                │
     │       └── 与原图 I_m 计算 L_rec                       │
     │                                                        │
     └────────────────── 路线 B：全清晰渲染 ────────────────┐   │
          同一个 GS，设 Q*=0 → AiF RGB Ĩ*_m               │   │
                                                           │   │
                    ILN ← (Ĩ_m,D̃_m,M_coc,PE(m),PE(x)) ◄─────┘
                           │                               │
                           ▼                               │
                       清晰 Mask M*_m                      │
                           │                               │
                           ▼                               │
              Ĩ_c = M*_m×Ĩ*_m+(1−M*_m)×Ĩ_m ◄───────────────┘
                           │
                           ├── L_detail：融合图 vs 原图 I_m
                           ├── L_mk：Mask 与 CoC 相关性
                           └── L_reg：Mask 正则化
                                      │
                                      ▼
                         总损失、反向传播、更新参数
                                      │
                      所有视角反复训练（每场景 40k 次）
                                      ▼
                       优化后的共享 3D Gaussians S*
                                      │
                        新相机位姿 + 新 f + 新 Q
                                      ▼
                        全清晰/散焦 RGB 新视角图像
```

**三个监督对象的区分**：

- **原始散焦照片 $I_m$ 是 RGB Ground Truth（GT）**；没有直接给出每个视角的全清晰 GT、逐像素清晰 Mask GT。
- **散焦图 $\tilde I_m$** 与原图比较：监督真实拍摄的散焦外观与场景。
- **全清晰图 $\tilde I_m^*$** 不是整张直接与散焦原图比较，而是根据预测清晰区域 Mask 与散焦图融合后再比较，避免把原图的模糊强加到所有 AiF 像素上。

---

## 1. 问题设定与方法定位（§1–2）

### 1.1 任务与输入输出

给定 $M$ 个同一静态场景的多视角图像：

$$\{I_m\}_{m=1}^{M},\qquad I_m\in\mathbb R^{H\times W\times3}$$

这些图像可能具有不同对焦位置和散焦程度。求一个场景的显式三维高斯集合 $\mathcal S$、每个训练视角的光学参数 $(f_m,Q_m)$，使可微景深渲染结果与照片接近。重建完成后根据**新的相机视角与光学参数**渲染 RGB 图像。

**不是**一个对任意单张图片执行一次前馈推理即可输出结果的网络；对每个场景要单独优化。论文中的输入是**多视角图像**，没有引入 4D 动态高斯模型。

### 1.2 三项主要贡献（Introduction）

1. **CoC-guided DOF Rendering**：给标准 3DGS Rasterizer 加入基于有限孔径相机模型的、对各高斯深度相关的散焦计算。
2. **Learnable Camera Parameters**：对每个训练视角学习一组 $(f_m,Q_m)$，用于拟合拍摄时的散焦状态，并支持训练完成后可控调焦。
3. **CoC-guided Detail Enhancement**：通过渲染 CoC、Depth、散焦 RGB，训练轻量 ILN 预测清晰区域 Mask，从无 AiF GT 的散焦照片中提供可靠的清晰细节监督。

---

## 2. 3DGS 场景表示与初始化（§3.1.1，Eqs. 1–5）

### 2.1 COLMAP/SfM 初始化和深度来源

- SfM 利用多视角特征匹配及三角测量恢复**相机位姿与稀疏点云**，再初始化三维高斯中心；在合成散焦数据实验中，作者也使用散焦输入经过 COLMAP 得到的位姿/稀疏点云，而不是直接读取合成数据的真值位姿。（§4.1）
- 第 $k$ 个高斯在世界坐标系中的中心为 $\mathbf u_k$，第 $m$ 个视角的外参记为 $(R_m,t_m)$：

$$\mathbf p_{k,m}^{\mathrm{cam}}=R_m\mathbf u_k+t_m$$

- **CoC 采用该中心在相机坐标系中的 $z$ 坐标**：

$$\boxed{z_{k,m}=(\mathbf p_{k,m}^{\mathrm{cam}})_z}$$

- 这里不是从 RGB 单独运行 Depth Anything / MiDaS 预测稠密深度；SfM 的几何是**初始化**，训练中高斯中心继续优化，因此 $z_{k,m}$ 也随之更新。
- 一个像素可能累积多个不同深度高斯的贡献，不能把 CoC 模型误解为对每个 RGB 像素固定一张单值深度图后执行统一模糊。
- **尺度注意事项**：没有外部真实尺度约束的 SfM 通常存在整体尺度歧义；优化得到的 $z$ 与 $f$ 必须处在一致的场景尺度下。论文的 $f,Q$ 数值不等于已经恢复真实物理米制标定。

### 2.2 每个三维高斯的参数

$$
\mathcal G_k(\mathbf p)=\exp\left[-\frac12(\mathbf p-\mathbf u_k)^\top\Sigma_k^{-1}(\mathbf p-\mathbf u_k)\right]\tag{1}
$$

$$\mathcal S=\{\mathbf u_k,\mathbf q_k,\mathbf s_k,o_k,\mathbf h_k\}_{k=1}^{K}$$

| 参数 | 意义 | 典型自由度 |
|---|---|---:|
| $\mathbf u_k$ | 中心位置 | 3 |
| $\mathbf s_k$ | 三个方向的尺度 | 3 |
| $\mathbf q_k$ | 旋转四元数 | 4 |
| $o_k$ | 不透明度 | 1 |
| $\mathbf h_k$ | Spherical Harmonics（SH，球谐函数）颜色系数 | 与 SH 阶数有关 |

标准 3DGS 的协方差可由旋转与尺度构造（**基础 3DGS 原理**）：

$$\Sigma_k=R(\mathbf q_k)S_kS_k^\top R(\mathbf q_k)^\top,\qquad S_k=\operatorname{diag}(s_x,s_y,s_z)$$

### 2.3 3D → 2D 投影与二维高斯

$$\boxed{\Sigma'_k=JW\Sigma_kW^\top J^\top}\tag{2}$$

- $W$：世界空间至相机空间的线性变换；$J$：透视投影在高斯中心附近的雅可比矩阵。
- 投影后的二维高斯中心为 $\mathbf u'_k$；二维协方差为 $\Sigma'_k\in\mathbb R^{2\times2}$。

$$
\mathcal G'_k(\mathbf x)=\exp\left[-\frac12(\mathbf x-\mathbf u'_k)^\top(\Sigma'_k)^{-1}(\mathbf x-\mathbf u'_k)\right]\tag{3}
$$

### 2.4 透明度合成与基本重建损失

$$\tilde I(\mathbf x)=\sum_{i=1}^{N_{\mathbf x}}T_i\alpha_i\mathbf c_i,\quad T_i=\prod_{j<i}(1-\alpha_j)\tag{4}$$

- 每个像素按高斯深度顺序累积颜色，$T_i$ 为前面高斯留下的透射率；实际 3DGS 实现还要结合独立的不透明度 $o_i$，正文 Eq. (4) 为简写。
- 标准场景优化（不含景深模型时）：

$$\min_{\mathcal S}\sum_{m=1}^{M}\mathcal L_{\mathrm{rec}}(\tilde I_m,I_m)\tag{5}$$

---

## 3. 有限孔径相机模型与 CoC（§3.1.2，Eqs. 6–7）

### 3.1 物理参数

| 符号 | 英文 | 中文 | 备注 |
|---|---|---|---|
| $F$ | Focal Length | 镜头焦距 | 与对焦距离不同 |
| $A$ | Aperture Diameter | 光圈**直径** | 与 f-number $N=F/A$ 不同 |
| $f$ | Focal Distance | 对焦距离 | 哪个深度近似最清晰 |
| $z_o$ | Object Depth | 物点深度 | 在当前相机坐标系下 |
| $Q$ | Simplified Aperture Parameter | 简化光圈参数 | $Q=FA$，并非独立测得的物理孔径 |

薄透镜关系给出 CoC 半径：

$$\boxed{R^{(\mathrm{coc})}=\frac12 FA\frac{|z_o-f|}{z_o(f-F)}}\tag{6}$$

对通常 $F\ll f$ 的拍摄情形近似 $f-F\approx f$：

$$\boxed{R^{(\mathrm{coc})}=\frac12Q\left|\frac1{z_o}-\frac1f\right|},\qquad Q=FA\tag{7}$$

关键关系：

- $z_o=f\Rightarrow R^{(\mathrm{coc})}=0$（处于对焦距离，理想情况下无额外散焦）。
- 保持 $f,z_o$ 不变时，增大 $Q$ 会增大离焦 CoC。
- 散焦强度与**逆深度差** $|1/z_o-1/f|$ 有关，不只是普通距离差 $|z_o-f|$。
- 每个训练视角有单独的 $(f_m,Q_m)$；训练完成后重新指定参数可实现焦点与散焦强度控制。

### 3.2 CoC 对深度误差的依赖

每个高斯的光学 CoC 为：

$$R_{k,m}^{(\mathrm{coc})}=\frac{Q_m}{2}\left|\frac1{z_{k,m}}-\frac1{f_m}\right|$$

- 几何深度错误可以传递到 CoC，再影响 CoC Map 和 ILN Mask。
- 但 $\mathbf u_k$、$f_m,Q_m$ 可随训练优化，不能把初始 SfM 深度当作不可修正的固定标签。
- CoC 受到深度、焦距和光圈的联合影响；仅凭散焦重建误差，并不必然唯一确定真实几何与真实物理镜头参数。
- 论文没有报告专门扰动初始 SfM 深度/位姿后验证 CoC 和 Mask 鲁棒性的实验；因此不能把“对深度初始化误差非常鲁棒”当作已验证结论。

---

## 4. Differentiable DOF Rendering：可微散焦渲染（§3.2，Eqs. 8–12）

### 4.1 为什么在每个高斯上计算模糊

对第 $k$ 个高斯先用中心深度 $z_{k,m}$ 计算 CoC，再将模糊作用在已投影的二维高斯 $\mathcal G'_k$ 上；之后执行 Alpha Compositing。

**不采用**“先把整个场景渲染为清晰 RGB，再对整张图统一 Gaussian Blur”的流程。

论文作出近似：**一个二维高斯的覆盖范围内的深度统一取其三维中心深度**。这避免了每个高斯覆盖像素重新计算不同 CoC，但跨深度边界/大尺度高斯的处理仍存在近似误差。

### 4.2 CoC 近似为各向同性二维高斯核

$$\boxed{\Sigma_k^{(\mathrm{coc})}=a_kI_2}\tag{8, 核心协方差}$$

- Isotropic（各向同性）：两个轴具有相同的方差 $a_k$。
- 将均匀 CoC 圆盘分布与高斯核进行 $L_2$ 近似拟合（论文 Eq. 9）得到：

$$\boxed{a_k=\frac{\left(R_k^{(\mathrm{coc})}\right)^2}{2\ln4}}\tag{9, 结果}$$

- 标准归一化高斯核可写作 $B_k(\mathbf x)=(2\pi a_k)^{-1}\exp[-\|\mathbf x\|^2/(2a_k)]$。
- **公式排印提示**：CVPR 正文 Eq. (8) 印刷的核函数指数不符合标准归一化高斯形式；这里保留论文明确的 $\Sigma^{(\mathrm{coc})}=aI$ 与 Eq. (9) 求得的 $a_k$，将标准高斯核公式单列为数学解释。

### 4.3 高斯卷积等价于二维协方差相加

$$\mathcal G''_k=\mathcal G'_k*B_k$$

高斯卷积封闭性允许在标准高斯假设下直接计算：

$$\boxed{\Sigma''_k=\Sigma'_k+a_kI_2}$$

其中：

- $\Sigma'_k$：投影后尚未施加光学散焦的二维形状；
- $a_kI_2$：额外的散焦核协方差；
- $\Sigma''_k$：散焦后的二维形状。

因此，修改的是**当前视图的二维投影高斯**，不是在三维场景中永久改变高斯的几何尺度。

### 4.4 Rasterization、贡献调整与可微性

- 对 $\mathcal G''_k$ 重新确定其覆盖范围；扩展后的高斯可能参与更多像素的计算。
- 按照原有高斯光栅化方式执行深度有序的 Alpha 合成。
- **官方 CUDA 实现补充**：不仅给二维协方差的两个对角元素加 $a_k$，还基于新旧协方差行列式比值修正高斯对像素的峰值贡献/Alpha，避免仅扩大形状就增加不合理总贡献；公式正文没有逐项列出所有工程数值处理。
- 为了让 $f_m,Q_m,\mathbf u_k$ 等参数接受图像损失梯度，作者在自定义 CUDA Rasterizer 中实现了对应的前向运算与导数。

### 4.5 散焦重建监督（Eq. 10）

$$\boxed{\min_{\mathcal S,\{f_m,Q_m\}}\sum_{m=1}^{M}\mathcal L_{\mathrm{rec}}(\tilde I_m,I_m)}\tag{10}$$

$\tilde I_m$ 使用当前 $f_m,Q_m$ 及 CoC 散焦渲染，$I_m$ 是**真实拍摄的散焦训练照片**。这一监督同时约束场景内容与拍摄时的光学散焦效果。

### 4.6 Depth Map 和 CoC Map（Eqs. 11–12）

散焦高斯的有效 Alpha 记为 $\alpha'_i$，透射率记为 $T'_i$：

$$\boxed{\tilde D_m(\mathbf x)=\sum_iT'_i\alpha'_i z_{i,m}}\tag{11}$$

$$\boxed{\mathcal M_m^{(\mathrm{coc})}(\mathbf x)=\sum_iT'_i\alpha'_iR_{i,m}^{(\mathrm{coc})}}\tag{12}$$

- **Depth Map**：通过相同 Alpha 权重把各高斯中心深度合成为像素值；不是单独预训练的 DepthNet 预测结果。
- **CoC Map**：通过相同权重将各高斯的 CoC 半径合成为像素散焦线索。
- 这两张图与散焦 RGB 同时由 Rasterizer 输出，主要为下一步 ILN 提供信息。
- 本式是论文的**未写显式权重归一化项**的合成形式；不要直接等同于把逐像素三维几何解成唯一 GT 深度。

---

## 5. CoC-Guided Detail Enhancement：细节增强（§3.3，Eqs. 13–18）

### 5.1 为什么必须额外监督全清晰图像

直接让散焦重建 $\tilde I_m\approx I_m$，可能让底层 GS 参数吸收输入照片的模糊，从而影响 $Q=0$ 时的 AiF（All-in-Focus，全清晰）结果。多个视角对焦区域不同，可利用**各视角原本清晰的区域**给共享 GS 场景提供可靠的纹理监督。

### 5.2 两路渲染和 ILN

每个训练迭代，对采样视角 $m$：

1. **散焦路径**：$\tilde I_m=R_{\mathrm{DOF}}(\mathcal S,P_m,f_m,Q_m)$，并输出 $\tilde D_m,\mathcal M_m^{(\mathrm{coc})}$；
2. **AiF 路径**：$\tilde I_m^*=R_{\mathrm{DOF}}(\mathcal S,P_m,f_m,Q^*=0)$；
3. ILN 使用 CoC、Depth、散焦 RGB、视角编码、像素位置编码预测清晰区域权重。

论文 Eq. (13)：

$$
\boxed{\mathcal M_m^*=\operatorname{ILN}(\tilde I_m,\tilde D_m,\mathcal M_m^{(\mathrm{coc})},\operatorname{PE}(m),\operatorname{PE}(\mathbf x))}\tag{13}
$$

- $\mathcal M_m^*\in[0,1]^{H\times W}$；**论文符号含义**：值越大，越倾向该位置是清晰区域。
- PE = Positional Encoding，位置编码；分别提供训练图像编号与像素坐标信息。
- 论文说明 ILN 使用四层 2D 卷积及 Sigmoid。实际源码 `MaskNet` 同时包含逐视角可学习 Embedding、卷积与位置编码，详见 §10。

### 5.3 没有 Mask GT：CoC 相关性监督

CoC 较大通常表示散焦较强，与“清晰 Mask”应当负相关。作者使用 Pearson Correlation（皮尔逊相关系数）：

$$
\boxed{\mathcal L_{\mathrm{mk}}=1-\frac{\operatorname{Cov}(1-\mathcal M_m^*,\mathcal M_m^{(\mathrm{coc})})}{\sqrt{\operatorname{Var}(1-\mathcal M_m^*)\operatorname{Var}(\mathcal M_m^{(\mathrm{coc})})}}}\tag{14}
$$

该损失鼓励 $1-\mathcal M_m^*$（模糊倾向）与 CoC Map 呈正相关，但不要求逐像素值相等；CoC 计算本身可能受几何误差影响，故相关性损失不能保证预测 Mask 与真实清晰区域完全一致。

### 5.4 融合图像以及细节损失

$$\boxed{\tilde I_m^c=\mathcal M_m^*\odot\tilde I_m^*+(1-\mathcal M_m^*)\odot\tilde I_m}\tag{15}$$

其中 $\odot$ 表示逐像素相乘，Mask 沿 RGB 通道广播：

- 清晰区域 $\mathcal M_m^*\approx1$：主要使用 AiF 渲染监督；
- 散焦区域 $\mathcal M_m^*\approx0$：主要保留散焦渲染，不用模糊 GT 强迫 AiF 表示变模糊。

$$\boxed{\mathcal L_{\mathrm{detail}}=\|\tilde I_m^c-I_m\|+\lambda_{\mathrm{dssim}}\mathcal L_{\mathrm{dssim}}(\tilde I_m^c,I_m)}\tag{16}$$

论文使用重建范数项和 DSSIM（结构不相似性）项；**代码中可进一步核对到具体 `l1_loss` 和 `1-ssim` 实现**，见 §10。

### 5.5 Mask 正则与总损失

$$\boxed{\mathcal L_{\mathrm{reg}}=-\mathcal M_m^*\log\mathcal M_m^*}\tag{17}$$

该项鼓励 Mask 更接近 0 或 1，论文仅写了这一项，而非完整二元交叉熵；实际计算需对像素聚合并处理数值稳定性。

$$
\boxed{\mathcal L=\mathcal L_{\mathrm{rec}}+\mathcal L_{\mathrm{detail}}+\lambda_{\mathrm{mk}}\mathcal L_{\mathrm{mk}}+\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}}\tag{18}
$$

共同优化变量为：

$$\boxed{\mathcal S,\quad\Omega_{\mathrm{ILN}},\quad\{f_m,Q_m\}_{m=1}^{M}}$$

无 AiF GT：所有监督目标依然来自同一视角的原始拍摄散焦图 $I_m$，只是采用了不同的渲染与空间权重。

### 5.6 多视角共同训练的含义

- 每轮采样一个训练视角；每张图像可具有不同清晰区域。
- 所有视角更新**同一个**高斯场景，图像 A 中清晰的前景和图像 B 中清晰的背景可共同提升最终 AiF 表示。
- 这不保证从未在任何视角清晰出现的真实纹理能够被准确恢复（论文 §5 的局限）。

---

## 6. 训练流程与优化设置（§4.1；官方代码）

### 6.1 论文报告的固定参数

| 项目 | 数值/设置 | 依据 |
|---|---|---|
| 训练方式 | 每场景单独优化 | §4.1 |
| 场景迭代数 | **40,000 iterations** | §4.1 |
| $\lambda_{\mathrm{dssim}}$ | **0.2** | §4.1 |
| $\lambda_{\mathrm{mk}}$ | **0.001** | §4.1 |
| $\lambda_{\mathrm{reg}}$ | **0.0001** | §4.1 |
| 三维场景初始化 | 多视角 SfM/COLMAP 相机位姿 + 稀疏点云 | §3、§4.1 |
| 光学参数 | 每张训练视角学习 $(f_m,Q_m)$ | §3.2 |
| 全清晰路径 | $Q^*=0$ | §3.3 |
| 训练监督 | 重建损失 + 细节损失 + CoC/Mask 相关性 + Mask 正则 | Eq. (18) |

论文没有把每个训练场景的所有优化器学习率、阶段开关、每次稠密化参数完全压在正文 §4.1 表内；需要结合实际配置文件区分。

### 6.2 官方代码：一个场景配置示例

官方 `config_files/config_real_cake.yaml` 的部分**实际字段**：

```yaml
scene_name: [real_cake]
iterations: 40000
densify_until_iter: 25000
num_extra_points: 30000
fdist_lr: 0.05
position_lr_init: 0.0005
position_lr_final: 0.000005
scaling_lr: 0.01
infocus_start_iter: 5000
maskNet_start_iter: 5000
mask_nn_lr: 5e-4
iter_defocus_start: 100
iter_stop_cam_gd: 15000
lambda_infocus: 1.0
lambda_dssim: 0.2
lambda_mask_loss: 0.002
use_mask_loss: true
use_depth_estimator: false
```

- 以上是 **real_cake 公开配置例子**；不能不加核对地认为所有训练场景设置完全相同。
- `use_depth_estimator: false` 表示默认可不额外调用深度估计器监督，但**仍需 SfM 的三维几何初始化及优化中的 GS 深度用于 CoC**。
- `infocus_start_iter`、`maskNet_start_iter` 说明代码存在延后启用细节/Mask 模块的策略，而不是从第 0 步所有损失必定同时启动。
- `densify_until_iter` 表示高斯场景仍可能进行稠密化与剪枝；“优化三维位置”不等于点数量完全固定。
- 这里的 `lambda_mask_loss: 0.002` 不等同于正文显式给出的 $\lambda_{\mathrm{mk}}=0.001$；论文抽象损失权重与公开代码具体损失拆分、缩放方式存在层次区别。

### 6.3 官方代码中的训练梯度流（重要区别）

`src/model_dofgs/training.py` 使用 `blur_img1` 和 `blur_img2` 两次融合，并配合 `detach()`：

```python
# 与论文符号相反：源码 infocus_mask 越大越偏向散焦分支
blur_img1 = (1.0 - infocus_mask) * aif_img.detach() + infocus_mask * image
loss_defocus = (1.0 - opt.lambda_dssim) * l1_loss(blur_img1, gt_image) \
             + opt.lambda_dssim * (1.0 - ssim(blur_img1, gt_image))
loss_defocus.backward(retain_graph=True)

blur_img2 = (1.0 - infocus_mask) * aif_img + infocus_mask * image.detach()
loss_aif = (1.0 - opt.lambda_dssim) * l1_loss(blur_img2, gt_image) \
         + opt.lambda_dssim * (1.0 - ssim(blur_img2, gt_image))
loss = opt.lambda_infocus * loss_aif + maskloss
loss.backward()
```

**必须区分**：

1. **论文 $\mathcal M_m^*$ 是清晰区域 Mask**，公式 (15) 里 $M\to1$ 更偏向 AiF 图。
2. **源码变量 `infocus_mask` 在混合式中值越大越偏向散焦图**，方向与论文符号相反，可通过 $M_{\mathrm{code}}\approx1-\mathcal M_{\mathrm{paper}}^*$ 对照，不能根据变量名直接理解。
3. `detach()` 分开两条渲染路径的梯度；代码的优化实现不是简单把论文 Eq. (18) 的所有张量一次相加后执行一次 `backward()`。
4. `maskloss` 受 `use_mask_loss` 开关控制；需按具体配置核对哪些正则在何时启用。

### 6.4 训练 vs. 推理

| 内容 | 场景优化训练 | 渲染/推理 |
|---|---|---|
| 输入 | 多视角散焦 RGB | 已优化的 GS + 目标相机参数 |
| COLMAP | 用于重建初始位姿、稀疏点云 | 新视角渲染通常不再次运行 |
| 三维高斯 | 可更新、可稠密化 | 固定已优化高斯 |
| $(f_m,Q_m)$ | 每个训练视角学习 | 用户直接设置新的 $f,Q$ |
| 散焦图 $\tilde I_m$ | 渲染并与 GT 比较 | 可输出作为结果 |
| AiF 图 $\tilde I_m^*$ | 用于细节监督 | 设 $Q=0$ 可直接输出 |
| ILN | 预测训练视角 Mask | **不需要** |
| 原始训练图 GT | 用于损失函数 | **不需要** |
| 反向传播 | 有 | 无 |

---

## 7. 数据集与评价协议（§4.1、§4.3）

### 7.1 使用 Deblur-NeRF 提供的真实与合成散焦数据

| 数据集 | 论文用途 | 数据特征 | 公开代码中的示例 |
|---|---|---|---|
| **Real Defocus Blur** | 场景训练、测试和高分辨率评价 | 真实拍摄的多视角散焦照片 | `defocuscake`、`defocuscaps` 等 |
| **Synthetic Defocus Blur** | 场景训练、测试 | 合成散焦多视角图像；作者使用从模糊照片经 COLMAP 估计的位姿/点云 | `defocuscozy2room`、`defocusfactory` 等 |

- 同一场景内用训练视角优化该场景 GS，用留出的测试视角评估 Novel View Synthesis（新视角合成）。
- 论文的主评价包括真实/合成数据各自的**600×400 分辨率**平均结果；高分辨率实验针对真实散焦数据使用 1200×800 与 2400×1600。
- **真实数据**没有逐像素的 GT 深度/CoC/清晰 Mask 可供作者直接监督；评测 AiF 图像使用数据集提供的参考视角/参考图像，不能等同“训练阶段所有像素拥有 AiF GT”。
- 训练样本是静态场景的多个视角，不是逐帧独立变化物体的动态视频序列。

### 7.2 作者实际比较的算法

- **NeRF、Mip-Splatting**：典型清晰场景重建/投影基线；
- **Deblur-NeRF、DP-NeRF、PDRF**：基于 NeRF 的模糊重建与去模糊；
- **BAGS、Deblur-GS**：基于 3DGS 的模糊重建方法；
- **DOF-GS**：在三维重建同时学习可调景深光学模型。

论文 Table 1 主要比较**全清晰新视角重建**而不是不同光圈下每一张光学散焦图像的 GT 误差。

### 7.3 评价指标

| 指标 | 方向 | 作用 | 注意事项 |
|---|---|---|---|
| **PSNR** | ↑ | 像素级重建质量 | 分辨率、测试图像和 RGB 范围要一致 |
| **SSIM** | ↑ | 亮度/对比度/结构相似性 | 不是显式深度或 CoC 误差 |
| **Peak GPU Memory (GB)** | ↓ | 高分辨率场景优化的峰值显存 | 论文 Table 2 为实验口径，非推理显存 |
| **人工标注焦距区间对齐** | 定性 | 比较学习到的对焦距离与人工估计的深度区间 | **不是**实测物理镜头对焦 GT |
| **视觉化景深控制** | 定性 | 固定 $Q$ 改变 $f$、固定 $f$ 改变 $Q$ | 主要展示可控渲染现象 |

基础 PSNR 公式（假设使用相同动态范围 $MAX$）：

$$\mathrm{PSNR}=10\log_{10}\frac{MAX^2}{\mathrm{MSE}(\hat I,I_{\mathrm{GT}})}$$

---

## 8. 论文实验结果（§4.2–4.4；Figures 4–8，Tables 1–3）

### 8.1 相机参数有效性（Figures 4–7）

- **Figure 4**：对比 CoC Map 在第 0 与 1,500 次迭代的变化，显示优化后 CoC 与观测模糊程度更相关。
- **Figure 5**：展示原训练视角、对应 CoC Map 与 ILN 预测的 In-Focus Mask，验证其清晰区域定位效果（以可视化为主）。
- **Figure 6**：选三个场景，人工把对焦距离划成最多四个深度区间，对照优化后的 $f_m$；结果基本位于相应区间。焦距转换为**屈光度（逆距离）并归一化**展示，**不是**真实物理误差数值。
- **Figure 7**：固定 $Q$ 改变 $f$，清晰平面位置改变；固定 $f$ 改变 $Q$，远离对焦平面的散焦程度改变。例图展示 $f\in\{24,41,140\}$，$Q\in\{101,140,196\}$；这些数值是场景中的控制参数，不应直接当成真实相机的米/毫米标定值。

### 8.2 Table 1：600×400 的 AiF 新视角结果（论文原表）

**Real Defocus Dataset：**

| 方法 | PSNR ↑ | SSIM ↑ |
|---|---:|---:|
| NeRF | 22.70 | 0.6789 |
| Mip-Splatting | 21.23 | 0.6292 |
| Deblur-NeRF | 23.98 | 0.7388 |
| DP-NeRF | 24.12 | 0.7451 |
| PDRF | **24.31** | 0.7559 |
| BAGS | 24.17 | 0.7664 |
| Deblur-GS | 24.21 | 0.7631 |
| **DOF-GS** | 24.12 | **0.7667** |

**Synthetic Defocus Dataset：**

| 方法 | PSNR ↑ | SSIM ↑ |
|---|---:|---:|
| NeRF | 25.74 | 0.7769 |
| Mip-Splatting | 23.73 | 0.7314 |
| Deblur-NeRF | 28.02 | 0.8491 |
| DP-NeRF | 28.91 | 0.8687 |
| PDRF | **29.96** | **0.8923** |
| BAGS | 28.89 | 0.8859 |
| Deblur-GS | 29.43 | 0.8907 |
| **DOF-GS** | 29.39 | 0.8914 |

**结果边界**：DOF-GS 在真实数据的 SSIM 最高；在合成数据中，PDRF 的 PSNR 和 SSIM 最高。DOF-GS 的主要价值是**在具备可调景深和拍摄后重新对焦能力的同时，仍保持有竞争力的 AiF 新视角质量**；不能把整张 Table 1 概括为“DOF-GS 全指标 SOTA”。

### 8.3 Table 2：高分辨率与峰值显存（Real Defocus Dataset）

| 方法 | 1200×800 PSNR ↑ | SSIM ↑ | Peak GPU GB ↓ | 2400×1600 PSNR ↑ | SSIM ↑ | Peak GPU GB ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Mip-Splatting | 20.51 | 0.5995 | 8.75 | 20.35 | 0.6220 | 32.13 |
| BAGS | 23.60 | 0.7345 | 21.69 | — | — | — |
| DOF-GS w/o detail enhancement | 23.36 | 0.7306 | 9.56 | 23.07 | 0.7190 | 38.89 |
| **DOF-GS full** | **23.66** | **0.7376** | 13.06 | **23.45** | **0.7221** | 39.05 |

BAGS 在 2400×1600 没有给出可用结果，论文标注是训练显存开销过高。

**区分：**该显存表来自高分辨率场景优化实验；不等于同分辨率单帧渲染的显存，论文在此没有完整公开任意新视角的毫秒级延迟表。

### 8.4 Table 3：消融实验（Real Defocus Dataset）

| 变体 | $\mathcal L_{\mathrm{detail}}$ | $\mathcal L_{\mathrm{mk}}$ | $\mathcal L_{\mathrm{reg}}$ | PSNR ↑ | SSIM ↑ |
|---|:---:|:---:|:---:|---:|---:|
| w/o detail enhancement | × | × | **✓** | 23.36 | 0.7306 |
| w/o $\mathcal L_{\mathrm{mk}}$ | ✓ | × | ✓ | 24.09 | 0.7642 |
| w/o $\mathcal L_{\mathrm{reg}}$ | ✓ | ✓ | × | 24.09 | 0.7640 |
| **Full model** | ✓ | ✓ | ✓ | **24.12** | **0.7667** |

**注意：第一行的正则列在 CVPR 正文 Table 3 原表中是“✓”，不能凭实验名称自行改为“×”；具体关闭细节模块时正则仍作用于何处，需要参考相应代码分支，不能只凭表名推断。**

- 完整方法相较 w/o detail enhancement：PSNR **+0.76 dB**，SSIM **+0.0361**；
- 移除 $L_{\mathrm{mk}}$：PSNR 从 24.12 降至 24.09，SSIM 从 0.7667 降至 0.7642；
- 移除 $L_{\mathrm{reg}}$：PSNR 从 24.12 降至 24.09，SSIM 从 0.7667 降至 0.7640。

因此主要定量提升来自**有 Mask 引导的 AiF 细节增强监督**，相关性和二值倾向正则作为辅助约束，改善幅度较小。

---

## 9. 论文结论与明确局限（§5）

1. **静态场景前提**：输入为同一场景多视角照片，每个场景单独优化；论文不是直接处理任意运动视频的可变形 4DGS 系统。
2. **需要初始多视角几何**：COLMAP/SfM 提供初始相机位姿与稀疏点云，训练中的 GS 深度会继续调整，但错误位姿/退化 SfM 会影响重建和 CoC。
3. **全视角均严重模糊的真实细节难恢复**：如果某块纹理在全部照片中都模糊，缺少可用于 AiF 恢复的清晰观察。（正文明确）
4. **相机位姿联合优化尚待完善**：作者把训练过程中进一步优化 Camera Pose 作为未来方向。（正文明确）
5. **光学模型近似**：采用薄透镜近似与各向同性高斯核近似 CoC 圆盘；每个高斯按中心深度使用统一 CoC，未显式建模真实镜头全部复杂光学效应。
6. **几何与光学参数误差耦合**：$z,f,Q$ 共同影响 CoC，错误深度可能由其他参数部分补偿；这是由模型形式带来的分析结论，论文**没有针对初始深度误差专门做可量化的鲁棒性实验**。
7. **数值展示与物理标定有区别**：由于 SfM 尺度歧义和参数简化，展示 $f,Q$ 可连续控制，不等于已证明真实镜头参数在每个场景都被精确校准。

---

## 10. 官方源码定位与复现注意事项

| 组件 | 公开源码或配置 | 核对重点 |
|---|---|---|
| 训练入口 | [`train.py`](https://github.com/rongduo/DOFGS_implementation/blob/main/train.py) | 配置加载、场景优化调用 |
| 场景核心参数 | [`gaussian_model.py`](https://github.com/rongduo/DOFGS_implementation/blob/main/src/model_dofgs/gaussian_model.py) | 高斯位置、颜色、尺度、不透明度，光学参数和优化管理 |
| ILN / MaskNet | [`neural_module.py`](https://github.com/rongduo/DOFGS_implementation/blob/main/src/model_dofgs/neural_module.py) | 四层卷积，视角 Embedding、位置编码、Mask 输出 |
| 损失与训练循环 | [`training.py`](https://github.com/rongduo/DOFGS_implementation/blob/main/src/model_dofgs/training.py) | 两路渲染、`detach()`、`blur_img1/2`、Mask loss、稠密化 |
| GPU 散焦光栅化 | [`diff-gaussian-rasterization-dof`](https://github.com/rongduo/DOFGS_implementation/tree/main/submodules/diff-gaussian-rasterization-dof) | 计算 CoC、修改二维协方差、Alpha 修正、渲染与梯度 |
| 真实场景配置 | [`config_real_cake.yaml`](https://github.com/rongduo/DOFGS_implementation/blob/main/config_files/config_real_cake.yaml) | 40k iterations、训练阶段开关、学习率 |
| 项目入口/数据说明 | [`README.md`](https://github.com/rongduo/DOFGS_implementation/blob/main/README.md) | 数据下载、场景脚本、配置参数和检查点说明 |

**官方代码仓库备注**：README 明确指出原始 checkpoint 因服务器迁移遗失，后续重新优化了部分场景，**公开的重训练权重及结果可能与论文表格略有差异**；复现实验时需记录当前 Git commit、场景、配置、分辨率与评测脚本。

---

## 11. 技术参数速查

| 项目 | 记录 |
|---|---|
| 方法类别 | 显式静态 3DGS + 有限孔径相机 + 可微散焦 Rasterizer |
| 训练输入 | 同一场景多视角散焦 RGB |
| 初始几何 | COLMAP / SfM 相机位姿与稀疏点云 |
| 可优化场景参数 | $\mathbf u,\mathbf q,\mathbf s,o,\mathbf h$ |
| 逐视角光学参数 | 对焦距离 $f_m$、简化光圈 $Q_m=FA$ |
| CoC 核心公式 | $R_k=\tfrac12Q_m|1/z_k-1/f_m|$ |
| 模糊核方差 | $a_k=R_k^2/(2\ln4)$ |
| 改写后的二维协方差 | $\Sigma''_k=\Sigma'_k+a_kI_2$ |
| 渲染输出 | 散焦 RGB + Alpha 合成深度图 + CoC Map |
| 全清晰路径 | $Q^*=0$，得到 AiF RGB |
| ILN 输入 | 散焦 RGB、Depth、CoC、训练视角和像素位置编码 |
| ILN 输出 | 清晰区域 Mask（论文定义） |
| 训练监督 | $L_{\mathrm{rec}}+L_{\mathrm{detail}}+\lambda_{mk}L_{mk}+\lambda_{reg}L_{reg}$ |
| 迭代次数 | 每场景 **40,000** |
| 论文权重 | $\lambda_{dssim}=0.2,\lambda_{mk}=0.001,\lambda_{reg}=0.0001$ |
| 数据集 | Deblur-NeRF Real Defocus + Synthetic Defocus |
| 主指标 | AiF 新视角 PSNR、SSIM；高分辨率另报告峰值 GPU 显存 |
| 渲染控制 | 新视角 + 新对焦距离 $f$ + 新光圈参数 $Q$ |
| 推理是否运行 ILN | 否；ILN 主要服务于训练细节监督 |
| 是否额外需要逐帧单目深度网络 | 不需要（源码默认关闭可选深度估计监督） |
| 最主要局限 | 多视角均模糊时真实细节缺失；初始位姿精度及几何/光学耦合 |

## 12. 原文公式、图表定位

| 位置 | 内容 |
|---|---|
| Figure 1 | 输入散焦多视角与拍摄后景深控制示例 |
| Figure 2；§3.1–3.2 | 三维高斯投影、CoC 核、二维高斯散焦、光学相机模型 |
| Eqs. (1)–(5) | 标准 3DGS 表示、投影、Alpha 合成、重建 |
| Eqs. (6)–(7) | 有限孔径相机与 CoC 公式 |
| Eqs. (8)–(10) | 高斯散焦核、模糊渲染与可微优化 |
| Eqs. (11)–(12) | Depth Map、CoC Map |
| Figure 3；Eqs. (13)–(18) | ILN、Mask、融合图像与总损失 |
| Figures 4–5 | CoC/Mask 与真实照片模糊程度的关联 |
| Figure 6 | 优化对焦距离与人工标注区间 |
| Figure 7 | 调节 $f,Q$ 的景深控制 |
| Figure 8；Table 1 | AiF 新视角定性与 600×400 定量结果 |
| Table 2 | 高分辨率 PSNR/SSIM/峰值显存 |
| Table 3 | Detail / Mask 相关性 / Mask 正则消融 |
| §5 | 论文结论及未来相机位姿优化方向 |

### 主要来源

1. [DOF-GS：CVPR 2025 正式论文（含公式、表格）](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DOF-GS_Adjustable_Depth-of-Field_3D_Gaussian_Splatting_for_Post-Capture_Refocusing_Defocus_CVPR_2025_paper.pdf)
2. [DOF-GS 作者项目主页](https://dof-gs.github.io/)
3. [DOFGS_implementation 官方 GitHub](https://github.com/rongduo/DOFGS_implementation)
4. [公开训练代码 `training.py`](https://github.com/rongduo/DOFGS_implementation/blob/main/src/model_dofgs/training.py)
5. [公开 ILN 模块 `neural_module.py`](https://github.com/rongduo/DOFGS_implementation/blob/main/src/model_dofgs/neural_module.py)
6. [真实场景配置例子 `config_real_cake.yaml`](https://github.com/rongduo/DOFGS_implementation/blob/main/config_files/config_real_cake.yaml)

**信息分层**：正文公式和论文表格优先按 CVPR 版本记录；具体 PyTorch/CUDA 训练变量和场景配置属于**公开源码核对信息**；深度误差传播及高斯协方差卷积推导属于模型数学机制说明；没有报告定量结果的方面不写作论文已验证性能。
