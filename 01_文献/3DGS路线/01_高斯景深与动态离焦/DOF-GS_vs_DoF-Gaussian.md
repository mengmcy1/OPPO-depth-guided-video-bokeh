# DOF-GS vs DoF-Gaussian：CVPR 2025 实现机制与复现对比

**对比对象**  
- **DOF-GS**：*Adjustable Depth-of-Field 3D Gaussian Splatting for Post-Capture Refocusing, Defocus Rendering and Blur Removal*（Yujie Wang 等，CVPR 2025）  
  [论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DOF-GS_Adjustable_Depth-of-Field_3D_Gaussian_Splatting_for_Post-Capture_Refocusing_Defocus_CVPR_2025_paper.pdf)｜[代码](https://github.com/rongduo/DOFGS_implementation)
- **DoF-Gaussian**：*Controllable Depth-of-Field for 3D Gaussian Splatting*（Liao Shen 等，CVPR 2025）  
  [论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_DoF-Gaussian_Controllable_Depth-of-Field_for_3D_Gaussian_Splatting_CVPR_2025_paper.pdf)｜[代码](https://github.com/leoShen917/DoF-Gaussian)

**资料基准**：两篇 CVPR 2025 论文、作者项目/官方 GitHub 主分支，2026-11-03 核查。全文区分**论文描述、公开源码实际调用路径、推断**。此文偏向“**具体是靠什么控制虚化、在哪一步添加散焦、训练时怎样恢复清晰 GS**”，以及后续公平复现所需要的协议。

## 0. 先给结论：最重要的三个区别

1. **散焦发生位置不同**：DOF-GS 是**每个三维高斯投影到二维后、Alpha 合成之前**先施加 CoC；DoF-Gaussian 是**GS 先光栅化成 RGB + Depth 图像后**，对像素颜色执行深度相关的邻域散射。
2. **清晰图像的训练方式不同**：DOF-GS 专门渲染一条 **$Q=0$ 的 AiF 分支**，再用 **ILN（清晰区域定位网络）**预测 Mask 融合监督；DoF-Gaussian **没有相同的独立 AiF 融合分支**，主要用散焦图像重建，并在 **10k 迭代后提高焦区 L1 权重**，依靠多视角中已有清晰观测恢复细节。
3. **深度约束不同**：DOF-GS 的 CoC 来自**各高斯中心的相机深度**，默认不额外使用深度网络；DoF-Gaussian 的 CoC 来自**GS 已光栅化的逐像素深度图**，另通过 **COLMAP 稀疏深度微调单目网络 → 稠密深度先验**约束 GS 几何。

共同点：两者都是**多视角散焦 RGB → COLMAP 初始化 → 一个共享的、按场景优化的静态 GS → 每视角可学习光圈和对焦参数 → 训练完后重新设定光学参数来调焦**。不是“一篇直接预测 GS，另一篇不是”；也不是任何一篇天然有完整 AiF RGB GT。

---

## 1. 总体数据流：两套实现并排看

### 1.1 DOF-GS：Per-Gaussian Blur Before Compositing

```text
同一场景的多视角散焦 RGB
        │
      COLMAP / SfM
        │
        ▼
    共享 3D Gaussians S  ← 所有训练视角联合优化同一套参数
        │
        ├─ 当前视角 m / (f_m, Q_m)
        │        │
        │        ▼
        │   高斯中心相机深度 z_k
        │        │
        │   CoC 半径 R_k = Q_m/2 × |1/z_k − 1/f_m|
        │        │
        │   修改 2D Gaussian 协方差 Σ'' = Σ' + a_k I
        │        │
        │   Alpha Compositing → 散焦 RGB Ĩ_m
        │                        + Depth Map + CoC Map
        │        │
        │        └─ Ĩ_m vs 原始散焦 RGB：L_rec
        │
        └─ 同一个 S，设置 Q*=0 → AiF 候选 Ĩ*_m
                                 │
        ILN(散焦 RGB, Depth, CoC, 视角/像素编码)
                                 │
                        预测清晰区域 Mask M*_m
                                 │
                 M*×AiF + (1−M*)×Defocus
                                 │
                  融合图 vs 原始散焦 RGB：L_detail
                                 │
                          + Mask/CoC 约束
                                 │
                               更新 GS

训练后：任意新相机视角 + 新 f + 新 Q → DOF Rasterizer → RGB
```

### 1.2 DoF-Gaussian：Per-Pixel Scatter After Rasterization

```text
同一场景的多视角散焦 RGB
       │
     COLMAP / SfM
       │
       ├─ 初始化共享 3D Gaussians S
       │
       └─ 稀疏深度 D_sparse
               │
            微调深度网络（逐场景）
               │
          稠密深度先验 D_pred ─────────┐
                                      │
采样视角 m，用 S 做普通光栅化       │
       ├─ RGB C_m（无额外镜头散焦）  │
       └─ Depth D_m ───────────────── L_depth
                 │
          学习当前视角 (A_m, F_m)
                 │
        r(d)=A_m×|1/F_m−1/d|
                 │
        每个 RGB 像素颜色向邻域散射
                 │
             散焦 RGB C*_m
                 │
           C*_m vs 真实散焦 RGB
                 │
        前 10k 普通重建；之后焦区加权
                 │
        + 稠密深度、法线等几何约束
                 │
               更新 GS

训练后：任意新相机视角 → RGB/Depth → 新 A/F → Scatter → RGB
```

**易误解点**：在 DoF-Gaussian 中，$C_m$ 是可理解为无额外镜头散焦的 RGB 候选，但**不是一张单独拿 AiF GT 来监督的结果**；它受到后续散焦重建损失的间接训练。

---

## 2. 控制虚化的公式：到底靠什么？

### 2.1 DOF-GS：每高斯 CoC + 协方差扩展

对于高斯 $k$ 在相机坐标系的中心深度 $z_k$：

$$\boxed{R_k^{\mathrm{CoC}}=\frac{Q_m}{2}\left|\frac1{z_k}-\frac1{f_m}\right|}$$

- **$f_m$：对焦距离**。当 $z_k\approx f_m$，该高斯附加光学散焦接近零；改变 $f_m$ 就是移动焦平面。
- **$Q_m$：光圈相关控制量**。在其他条件不变时，它越大，离焦 CoC 越大；改 $Q_m$ 就是在控制虚化强度。
- 所有高斯各自算 CoC，**不需要先得到完整二维稠密深度图才能算这个值**。

用各向同性高斯核近似 CoC 的圆盘核，定义：

$$a_k=\frac{(R_k^{\mathrm{CoC}})^2}{2\ln4},\qquad \Sigma_k^{\mathrm{coc}}=a_k I_2$$

对投影后的二维协方差直接执行：

$$\boxed{\Sigma_k''=\Sigma_k'+a_k I_2}$$

这由**两个高斯卷积的协方差相加性质**得到；无需先渲染一张 RGB 再对其做逐像素模糊。实现还会对扩展后的高斯不透明度作幅值修正。对应：[forward.cu](https://github.com/rongduo/DOFGS_implementation/blob/main/submodules/diff-gaussian-rasterization-dof/cuda_rasterizer/forward.cu) 中 `coc_radius`、`coc_coeff`、`cov.x / cov.z += coc_coeff` 和 `o_coef`。

**所以它是“在每个高斯身上施加光学扩展，再按深度合成”**；同一像素可能来自多个深度、CoC 不同的高斯。该机制并不表示能完美恢复所有遮挡背后的颜色（仍受已有 GS 几何/可见背景约束）。

### 2.2 DoF-Gaussian：逐像素 CoC + Scatter

已渲染得到同一视角的 RGB $C$ 与深度 $D$，对于每个源像素的深度 $d$：

$$\boxed{r(d)=\mathcal A_m\left|\frac1{\mathcal F_m}-\frac1d\right|}$$

- **$\mathcal F_m$：对焦位置/对焦距离相关参数**，决定哪一个深度附近不额外模糊；
- **$\mathcal A_m$：光圈相关参数**，控制与焦平面偏离时的散射范围。

不是对全图统一卷积：对于源像素 $i$ 和目标像素 $j$，根据它们的距离 $l=\|i-j\|$ 设置：

$$\boxed{\lambda_{ij}=\frac12+\frac12\tanh\{\alpha[r(d_i)-\|i-j\|]\}}$$

颜色和权重进行累积并归一化（省略 Gamma 前后变换的解释性形式）：

$$C_j^*\approx\frac{\sum_i\lambda_{ij}C_i}{\sum_i\lambda_{ij}}$$

模型能通过 **$\mathcal F_m$ 改对焦平面、$\mathcal A_m$ 改模糊强度**，还可以修改 Scatter 的 CoC 形状来表现非圆形光斑。

**所以它是“先把 GS 投影合成 RGB/深度，再让 RGB 像素根据深度扩散”**。这类图像空间的散射实现更直接接近一种基于深度图的后处理，也有遮挡边缘处理的潜在限制；但不能因此断言它一定会比前者差，需用具体场景测试。

### 2.3 参数符号和代码中的实际含义要分开

| 论文 / 代码 | DOF-GS | DoF-Gaussian |
|---|---|---|
| 论文的对焦参数 | $f_m$ | $\mathcal F_m$ |
| 论文的光圈参数 | $Q_m$ | $\mathcal A_m$ |
| 控制散焦的核心条件 | $z_k=f_m$ | $d=\mathcal F_m$（理论式） |
| 代码可学习字段 | 焦距相关 `F`、光圈相关 `K` | `get_disp_focus`、`get_K_bokeh` |
| 代码实际焦点坐标 | 3D 高斯中心的**相机深度** | **归一化逆深度 / disparity** 空间 |
| 代码中的 CoC | `0.5*K*abs(1/F−1/z)` | `K_bokeh*(normalized_disparity−disp_focus)` |

**不能直接将两篇代码的 $f,F,Q,A$ 赋相同数值，就认为焦平面和光圈强度相同。**

DoF-Gaussian 的 [bokeh_renderer/utils.py](https://github.com/leoShen917/DoF-Gaussian/blob/main/bokeh_renderer/utils.py) 使用 $1/D$，按场景和当前渲染视角进行 min / quantile 归一化，再减 `disp_focus`；而 DOF-GS 的实现直接使用高斯中心的 `p_view.z`。两套数值标度**并不天然一致**。

如需对齐焦平面，建议选取一个**实际可见的三维表面/目标深度**，在各自的表示尺度内计算其目标焦点参数；如果要对齐光圈，使用**观测得到的 CoC 像素半径**，而不是简单将 $Q=A$。

---

## 3. 为什么两篇都能重新对焦？

两篇训练完成后都有明确的光学控制变量，不需要再运行一个端到端网络来生成新的焦点效果。

| 操作 | DOF-GS | DoF-Gaussian |
|---|---|---|
| 调整对焦位置 | 指定新的 $f$，重新计算每个高斯的 CoC | 指定新的 $\mathcal F$ / `disp_focus`，重新计算每个像素的 CoC |
| 调整光圈/强度 | 指定新的 $Q$ | 指定新的 $\mathcal A$ / `K_bokeh` |
| 关闭光学散焦 | $Q=0$ | $\mathcal A=0$ / `K_bokeh=0` |
| 改变视角 | 先投影新视角的高斯，再散焦合成 | 先在新视角光栅化 RGB/Depth，再做 Scatter |
| 直接圈选任意形状区域为焦点 | **论文未原生实现** | **论文未原生实现** |

**“重新指定对焦区域”在论文里本质上是“指定对焦距离（焦平面）”**。若要实现交互上的点击主体对焦，可由点击坐标读取深度，再映射到两篇论文各自的对焦参数；这是可以扩展的应用，不应说是论文已经提供任意实例 Mask 对焦。

---

## 4. 深度来源：最容易搞混的部分

### 4.1 DOF-GS

SfM 恢复相机位姿与稀疏点云 → 初始化三维高斯位置 $\mathbf u_k$ → 针对视角 $m$：

$$\mathbf p_{k,m}^{cam}=R_m\mathbf u_k+t_m,\qquad z_{k,m}=(\mathbf p_{k,m}^{cam})_z$$

该 $z_{k,m}$ 直接用于每高斯 CoC。GS 中心在训练时可优化，所以不是“初始化深度一直不变”。默认 `use_depth_estimator: false`，意味着**论文核心实验不要求一个外部稠密深度先验网络**，但仍依赖 SfM/GS 的三维几何。

### 4.2 DoF-Gaussian

两个并行分支：

- **GS 本体**渲染出 $D_m$，参与真正的镜头散焦；
- **场景深度先验**：COLMAP 稀疏点 → 投影 $D_{sparse}$ → 微调单目网络 → 稠密 $D_{pred}$ → 对齐尺度 → 监督 $D_m$。

$$\mathcal L_{depth}=\|D_m-D_{pred,m}\|_2\quad\text{（论文写法）}$$

公开训练代码深度损失用 L1，10k 后用加权 L1。这样能利用先验改善几何，但**先验不是 GT**，可能与真值不符。

### 4.3 几何错误如何传播？

两篇都存在：**错误 GS 几何 → 错误深度 → 错误 CoC → 错误散焦**。

差别是 DOF-GS 的 CoC 按各高斯中心分配，DoF-Gaussian 按已合成出的像素深度分配。两者都可以在多视角损失约束下改善几何；DoF-Gaussian 另用一个经过微调的稠密深度先验提供额外约束。

注意：多视角 GS 可以修正初始位置误差，**但论文没有证明对任意严重错误的相机位姿或所有视角都模糊的输入都鲁棒**。DOF-GS 还明确将相机位姿进一步优化列为未来研究方向。

---

## 5. 清晰 GS 靠什么训练？（两篇最重要的第二个差别）

### 5.1 DOF-GS：有独立 AiF 渲染分支 + ILN，但没有 AiF GT

对于当前视角：

- 散焦渲染 $\tilde I_m$：使用所学 $f_m,Q_m$；
- 全清晰候选 $\tilde I_m^*$：设置 $Q^*=0$；
- 同时渲染 CoC Map、Depth Map；
- ILN（轻量卷积网络）输入散焦 RGB、Depth、CoC 及位置编码，输出清晰区域 Mask $M_m^*$；
- 构造：

$$\boxed{\tilde I_m^c=M_m^*\tilde I_m^*+(1-M_m^*)\tilde I_m}$$

再计算 $\mathcal L_{detail}(\tilde I_m^c,I_m)$，并有散焦重建、Mask 与 CoC 相关、Mask 正则项。

**准确说法**：它有一个专门提升 AiF 质量的监督路径，但**没有完整的 AiF RGB GT**。输入照片在焦区内近似清晰，这些像素能够作为全清晰分支的局部监督。

【源码】`src/model_dofgs/training.py` 将同一混合图构造为 `blur_img1`、`blur_img2`，对两个路径分别 `detach()` 后计算损失和反传。代码变量名 `infocus_mask` 在实际混合表达式中的权重方向，与论文 $M^*$ 的清晰区语义相反，**读代码需要按乘在哪张图上判断 Mask 的意义**。

### 5.2 DoF-Gaussian：没有独立 AiF 分支，用 Focus Adaptation 加权

早期对整张散焦渲染图 $C^*$ 计算 RGB 重建；10k 后根据 $|1/\mathcal F-1/d|$ 构造焦区权重，使原图中本来较清晰的位置对 L1 误差影响更大。理论上：

$$\Psi(x)=\frac1{1+e^{-a(x-b)}},\qquad x=\left|\frac1{\mathcal F}-\frac1d\right|$$

但**源码实际用 $1-\Psi$ 作为 `l1_loss_focus` 的权重**：高权重在焦区；并非直接把 $\Psi$ 乘在 RGB L1 上。源码后期 SSIM 不同样逐像素加权。

这里“清晰区域”并没有对应的完整无模糊 GT，只是认为其在**原始散焦 RGB 内就已经相对清晰**。多视角中不同区域可能在不同照片里成为焦区，因此能互补细节。

### 5.3 同样的失效边界

如果一个物体在所有训练视角都严重散焦，**没有可靠清晰观测，任何一篇都不能保证恢复真实高频纹理**。设置 $Q=0$ 或 $A=0$ 只是取消额外光学散焦，不等于把 GS 中已丢失的纹理“反卷积”恢复。

---

## 6. 关键训练损失与梯度：论文 vs 主训练源码

| 项目 | DOF-GS | DoF-Gaussian |
|---|---|---|
| 主要 RGB 损失 | 散焦渲染图 vs 真实散焦照片 | 散焦渲染图 vs 真实散焦照片 |
| 额外清晰监督 | 融合 AiF/散焦图 vs 真实照片的 `L_detail` | 10k 后对焦区提高 RGB L1 权重 |
| Mask 模块 | **有 ILN**，输出每视角 Mask | **无 ILN**，使用基于光学位置的权重函数 |
| Mask / Focus 约束 | CoC 与 Mask 相关；Mask 正则 | Sigmoid 权重、Focus Loss |
| 外部稠密深度网络 | 默认非必需 | **必要组成**（按论文主流程） |
| 几何额外约束 | 多视角 GS 重建为主 | 深度先验、法线一致性、源码中的 depth distortion |
| 总迭代数（论文） | **40,000** | **30,000** |
| 两阶段策略 | 无同等 10k Focus 切换 | **10,000 次后**启用 Focus 加权 |
| 分离的镜头参数 | 每张训练图像学习 $f,Q$ | 每张训练图像学习 $\mathcal F,\mathcal A$ |

**特别容易漏掉的梯度区别：**

- **DOF-GS**：修改二维 Gaussian 协方差在自定义 Rasterizer 内完成，具有相应 CUDA 梯度实现；但其 AiF/散焦融合支路使用 `detach()` 做了明确的梯度拆分，因此不能把每个损失都想成自动穿过所有分支。
- **DoF-Gaussian**：论文强调可微镜头散焦，但公开主调用 `render_bokeh()` 中 `disps = disp_map[0].detach()`，**散焦 RGB 损失不会沿着 CoC 对渲染深度的链路回传 GS**；GS 的深度仍经由深度先验、法线/几何损失等更新。相反，RGB 颜色链路仍可参与优化。
- **DoF-Gaussian** 论文 Eq. (11) 还提出 $\mathcal A'=\mathcal A\Psi$ 的后期逐像素光圈缩放；实际 `train.py` 调的是 `render_bokeh()`，该部分在该函数中被注释，而另一个 `render_bokeh_ex()` 才启用类似逻辑。

以上均为**当前公开主分支的已核实调用关系**，不是对所有历史代码版本的绝对断言。

---

## 7. 图像形成、边界行为与复杂度：可推断什么？

| 维度 | DOF-GS | DoF-Gaussian |
|---|---|---|
| 模糊基本单元 | 投影后的**二维 Gaussian** | 光栅化后的**RGB 像素** |
| 景深相关变量 | 各 Gaussian 的**中心相机深度** | **合成后像素深度图** |
| 扩散操作 | 修改二维协方差，改变 Gaussian footprint | 按 CoC 邻域累积散射权重和颜色 |
| 颜色合成顺序 | **先散焦、后 Alpha 合成** | **先 Alpha 合成、后 Scatter** |
| 多层遮挡 | 不同深度高斯在合成前保留各自 CoC | RGB/深度图已合成，部分层信息可能丢失 |
| 可改 CoC 形状 | 论文主模型为各向同性高斯近似，未将任意形状列为核心功能 | 论文展示修改圆形为六边形等光斑 |
| 实现开销取决于 | 高斯数、扩散后的 footprint、tile 覆盖等 | 图像像素数、CoC 散射半径、邻域 CUDA 实现等 |

**遮挡边界是值得专门检查的点，但不能直接拿这个表下最终好坏结论**。例如，前景轮廓旁边的背景在单层 Depth 图中可能已被遮挡，Scatter 难以凭空得到被遮挡的 RGB；但是 DOF-GS 若没有足够的背层高斯，也同样没有真实的被遮挡纹理可用。

建议评测时重点看：前景头发/物体边缘、两层深度大跳变、高亮 Bokeh 光斑、背景遮挡、焦平面跨边界以及改变对焦位置时是否出现不连续伪影。

---

## 8. 两篇都开源了吗？数据是否都能直接使用？

### 8.1 代码公开：都能获取，但“能复现”需要区分层级

| 项目 | DOF-GS | DoF-Gaussian |
|---|---|---|
| 公开 GitHub 仓库 | **是** | **是** |
| 训练代码 | **有** | **有** |
| CUDA/渲染模块 | DOF Gaussian Rasterizer | Scatter 模块 + 3DGS Rasterizer |
| 渲染/评测脚本 | README 提供 `scripts/val_real.sh`、`val_syn.sh` | README 提供 `render.py`、`metric.py` |
| 模型/配置 | YAML、场景配置、可下载重建场景入口 | 训练参数、逐场景深度模块 |
| Deblur-NeRF 数据入口 | **有**，真实和合成 | **有**，真实数据入口 |
| 额外自建镜头参数合成 benchmark | 不属于该论文核心自建 benchmark | **尚未发现完整公开处理版** |
| 关键复现提醒 | README 称**原始 checkpoint 在服务器迁移时丢失**，重新训练恢复的模型可能与论文分数略不同 | README 的 `Applications` 标为 `coming soon`，论文演示不等于所有应用接口都打包好 |

官方仓库与 README：

- https://github.com/rongduo/DOFGS_implementation
- https://github.com/leoShen917/DoF-Gaussian
- 未找到新合成 benchmark 完整下载的公开讨论：https://github.com/leoShen917/DoF-Gaussian/issues/6

**“公开可访问”不等于“已经在你的服务器上运行验证成功”**；不同 CUDA/PyTorch/依赖与 git submodules、作者修改版 COLMAP 都需要本地检查。另应核对各仓库及上游 3DGS 的许可证，再决定是否直接复用组件到其他项目。

### 8.2 两篇论文的原始报告数字不能直接等同于公平比赛

DOF-GS 论文在 Deblur-NeRF 真实散焦场景的报告均值为：**PSNR 24.12、SSIM 0.7667**（CVPR 论文 Table 1）。

DoF-Gaussian 在 Deblur-NeRF 真实散焦场景的报告均值约为：**PSNR 23.97、SSIM 0.755–0.756、LPIPS 0.093**（论文 Table 1 的 10 场景）。

虽然两个任务和数据源有交集，但**不能直接宣布 DOF-GS 击败 DoF-Gaussian**：作者对输入分辨率、训练/测试帧集合、RGB 预处理、COLMAP 数据、评价边界、颜色空间及计算指标的细节，尚未在同一实验中统一；DoF-Gaussian 自己的文章也没给出统一跑出来的 DOF-GS 对照。

---

## 9. 如何公平对比：建议的可执行实验协议

目标是**一组统一数据、两份官方实现、共同评测脚本**，而不是将不同论文的表格直接拼接。

### 9.1 Track A：全清晰新视角恢复（最容易先跑通）

1. 选两篇都有 README 入口的 **Deblur-NeRF Real Defocus Blur**，起步选择 `defocuscake`、`defocuscaps` 等相同场景；正式报告覆盖相同的 10 个真实场景。
2. 固定同一份原始 RGB、相机位姿来源、裁剪/缩放、有效训练/测试帧列表。建议同时保存输入图 SHA256，避免两个 repo 载入的图像版本不同。
3. 先各自按**官方默认配置**训练，保留方法差异（ILN vs depth priors 等）；不要为了“公平”去掉其中一篇的核心模块。
4. 从相同测试视角渲染 **AiF**：DOF-GS 设 $Q=0$；DoF-Gaussian 设 $\mathcal A=0$。
5. 用统一的 Python 评测脚本，对**相同 GT、相同分辨率与颜色空间**计算 PSNR、SSIM、LPIPS，并记录每场景和平均值。
6. 统一报告 **训练总耗时与额外深度先验微调成本**，避免把 DoF-Gaussian 的深度阶段漏掉。

输出建议：`scene, method, seed, train_frames, test_frames, resolution, psnr, ssim, lpips, train_min, extra_depth_min, peak_vram_gib, checkpoint_id`。

### 9.2 Track B：重对焦/散焦能力（不应与 AiF 混为一个表）

使用固定的**同一优化后相机位姿**，分别改变焦平面位置与 CoC 尺度。推荐先用定性/可解释实验：

- 指定同一个可见对象为焦点；在两套方法**自身标度**下映射目标深度。
- 对齐一组观测 CoC 半径（例如 0、若干像素），而不是直接对齐 $Q$ 与 $\mathcal A$ 的数字。
- 保存 RGB、Depth、CoC / 视差、焦点参数、目标物体 Mask（若另有人工标注）。
- 比较焦区位置、非焦区模糊程度、焦点渐变、前后景边缘、虚化光斑形状和伪影。
- 真实照片若没有同一视角同一镜头设置的散焦 GT，**不要假装能算严格的散焦 PSNR**。作者新合成评测集未公开，若自己合成，必须注明 **self-constructed / reproduction of protocol**。

### 9.3 Track C：效率与资源

统一 GPU、驱动、CUDA、输出分辨率、场景、视角数量与预热次数。至少拆成：

| 阶段 | DOF-GS | DoF-Gaussian |
|---|---|---|
| SfM/COLMAP | 记录 | 记录；包含修改版 COLMAP 准备 |
| 深度先验 | 若默认关闭则记 0 | 单独记录微调/推理/尺度对齐 |
| GS 训练 | 记录 40k 默认设置 | 记录 30k 默认设置 |
| AiF 推理 | Rasterizer with $Q=0$ | 标准 Rasterizer with $A=0$ |
| 景深推理 | DOF Rasterizer | 普通 RGB/Depth Rasterizer + Scatter |
| 内存 | `max_memory_allocated` / `nvidia-smi` | 同口径记录；注意临时 RGB/Depth/Scatter 缓存 |

各方法**默认迭代数不同**，因此建议同时报告：默认完成时间、相同迭代预算下质量曲线、达到同一 PSNR 门槛的时间（需明确实验设置）。

### 9.4 Track D：几何与遮挡鲁棒性（可选研究扩展）

使用经过人工审核的同一 SfM 初始化，再对部分高斯位置、相机位姿或初始深度施加可控扰动。比较：

- 最终全清晰 PSNR/SSIM/LPIPS 的下降；
- GS 深度对独立几何 GT 的误差（若有 GT），不能仅凭更平滑的深度图声称更准确；
- 对焦区域移动是否稳定；
- 前景边缘 CoC 泄漏、背景空洞、遮挡伪影。

**这属于建议的新实验，不是两篇论文都已经完成的同协议鲁棒性结论。**

---

## 10. 工程环境与复现风险速查

### DOF-GS

```bash
# 作者 README 的仓库地址
 git clone --recursive https://github.com/rongduo/DOFGS_implementation.git
# 安装按 environment.yml 和 README 的 PyTorch / CUDA 依赖执行
# 作者 README：
sh scripts/train_real.sh
sh scripts/val_real.sh
```

- CUDA 自定义算子：`submodules/diff-gaussian-rasterization-dof`。
- 主要方法：`src/model_dofgs/`。
- 真实/合成场景配置和不同训练选项：`config_files/`。
- 论文 40k；`use_depth_estimator` 默认 false。
- README 声明原始模型 checkpoint 在迁移时丢失，重训练版本结果可能略不同。

### DoF-Gaussian

```bash
 git clone --recursive https://github.com/leoShen917/DoF-Gaussian.git
# 安装需检查仓库 README；留意其修改版 COLMAP 的编译
sh colmap.sh <path-to-scene>
python train.py -s <path-to-scene> -m <output-dir> --eval -r 4 --llffhold 8
python metric.py -m <output-dir> -r 4 -t deblur-nerf
```

- 自己的散焦 Scatter 实现：`bokeh_renderer/`。
- 深度网络逐场景适配：`bokeh_renderer/depth_priors.py`。
- 归一化视差/可学习参数：`render_bokeh`、`get_K_bokeh`、`get_disp_focus`。
- 论文 30k；主训练在 10k 之后调用 Focus Loss；`Applications` 页面状态不等同所有交互功能已经提供可运行脚本。
- **两个项目采用的环境依赖不同**，推荐独立 Conda 环境，不要将不同 Rasterizer 混装到同一环境。

---

## 11. 如果只保留一张速查表

| 最关键的问题 | DOF-GS | DoF-Gaussian |
|---|---|---|
| **谁在控制对焦位置？** | 每视角 $f_m$ | 每视角 $\mathcal F_m$（源码 `disp_focus`） |
| **谁在控制虚化强度？** | 每视角 $Q_m$ | 每视角 $\mathcal A_m$（源码 `K_bokeh`） |
| **CoC 从什么深度计算？** | **每高斯中心深度** $z_k$ | **每像素 GS 渲染深度** $D(x,y)$ |
| **虚化加在哪里？** | **二维高斯协方差** | **光栅化后 RGB 的颜色散射** |
| **何时合成颜色？** | 高斯扩展**之后**做 Alpha 合成 | 先 Alpha 合成 RGB，再 Scatter |
| **有没有深度网络？** | 默认**没有必需的稠密估深** | **有，逐场景适配提供稠密先验** |
| **有没有专门 AiF 分支？** | **有**，$Q=0$ + ILN 融合监督 | **无相同分支**，焦区加权重建 |
| **有完整 AiF GT 吗？** | **没有** | **没有** |
| **推理可重新对焦吗？** | **可以** | **可以** |
| **都开源训练代码了吗？** | **是** | **是** |
| **公开数据是否足够先跑公平对比？** | 与 DoF-Gaussian 共享 Deblur-NeRF 真实场景 | 与 DOF-GS 共享 Deblur-NeRF 真实场景 |

> **一句话记住**：**DOF-GS 是“每高斯 CoC → 扩协方差 → 合成 RGB + ILN 辅助清晰监督”；DoF-Gaussian 是“GS 先输出 RGB/Depth → 每像素 CoC → Scatter + 稠密深度先验 + 后期焦区加权”。两篇都靠对焦距离和光圈参数控制虚化，差别不在“能不能调焦”，而在“何处发生虚化、用什么深度、如何监督清晰 GS”。**

---

## 12. 一手来源与源码定位

**DOF-GS**

1. [CVPR 2025 正式论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DOF-GS_Adjustable_Depth-of-Field_3D_Gaussian_Splatting_for_Post-Capture_Refocusing_Defocus_CVPR_2025_paper.pdf) —— §3.1.2、§3.2、§3.3、§4。
2. [官方仓库与 README](https://github.com/rongduo/DOFGS_implementation) —— 依赖、数据、训练评测、checkpoint 声明。
3. [CUDA Gaussian defocus](https://github.com/rongduo/DOFGS_implementation/blob/main/submodules/diff-gaussian-rasterization-dof/cuda_rasterizer/forward.cu) —— `coc_radius`、`coc_coeff`、`o_coef`。
4. [训练与双分支混合](https://github.com/rongduo/DOFGS_implementation/blob/main/src/model_dofgs/training.py) —— `blur_img1`、`blur_img2`、`detach()`。
5. [ILN / MaskNet](https://github.com/rongduo/DOFGS_implementation/blob/main/src/model_dofgs/neural_module.py) —— 网络模块。

**DoF-Gaussian**

6. [CVPR 2025 正式论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Shen_DoF-Gaussian_Controllable_Depth-of-Field_for_3D_Gaussian_Splatting_CVPR_2025_paper.pdf) —— §4.1–4.3、§5。
7. [官方仓库与 README](https://github.com/leoShen917/DoF-Gaussian) —— 依赖、COLMAP、Deblur-NeRF、评测命令。
8. [训练入口](https://github.com/leoShen917/DoF-Gaussian/blob/main/train.py) —— `depth_priors.train()`、`render_bokeh()`、`l1_loss_focus()`、几何损失。
9. [镜头渲染与视差处理](https://github.com/leoShen917/DoF-Gaussian/blob/main/bokeh_renderer/utils.py) —— `disps.detach()`、`render_bokeh` vs `render_bokeh_ex`。
10. [焦区权重](https://github.com/leoShen917/DoF-Gaussian/blob/main/utils/loss_utils.py) —— `bokeh_weight`、`focus_weight = 1-bokeh_weight`。
11. [新合成数据公开状态 Issue #6](https://github.com/leoShen917/DoF-Gaussian/issues/6) —— 公开处理版请求。

**本文结论边界**：源码已经公开、方法原理已经能逐项核对；但本文没有在本地安装环境、运行 CUDA 算子、重训练两篇论文或用统一数据复测，因此**没有实际得出两者的公平实验胜负，也不假称已验证运行时显存/速度**。
