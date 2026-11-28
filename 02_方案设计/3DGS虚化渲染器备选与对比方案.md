# 3DGS 虚化渲染器备选与对比方案

日期：2026-11-28。状态：独立候选文档，供后续阅读和比较；不改变总体方案、当前 K64 主线或正式执行顺序，不代表已批准开展实验。

## 1. 研究定位与现有方法

我们的核心工作是面向有限光圈成像的前馈 3DGS 场景构建，可视为专门针对小基线新视角合成（Novel View Synthesis, NVS）优化的三维表面表示。神经网络负责几何、颜色、遮挡和多帧隐藏表面；渲染器负责光圈及对焦距离控制。原则上两者应解耦，使同一场景可以复用不同的渲染器。

现有 K64 方法：在圆形光瞳上采样 64 个虚拟针孔位置，平移相机并调整主点，使指定焦平面上的投影一致；逐视角完成可见层、隐藏层及 F2 回退合成，按既有线性 RGB、曝光和截断约定积分。焦外点因视差扩散形成弥散圆（Circle of Confusion, CoC）。K64 是有限光圈积分的数值近似，不是完整物理场景的严格真值。像素面积采样和光圈采样是不同维度，比较时不得混用。

代码入口：`src/analytic_aperture.py`、`src/layered_aperture.py`。现有几何、颜色和可见性缺陷不能单靠更换渲染器保证修复。

## 2. 方法总览

| 备选 | 核心机制 | 预期用途 | 独立移植是否必需重训 |
|---|---|---|---|
| K64（现有） | 多针孔视角光栅化后积分 | 物理参考、质量基准 | 否 |
| DOF-GS / Cinematic Gaussians | 对投影高斯进行离焦卷积 | 速度与质量折中 | 仅移植确定性卷积不一定 |
| Closed-Form Convolution | 更准确的光圈核解析卷积 | 光斑形状与效率候选 | 全文和代码待查 |
| 3DGRT / 3DGRUT | 直接沿光线查询并合成高斯 | 可见性和渲染机制诊断 | 独立渲染不一定，参数兼容待查 |
| CoCoGaussian | 预测多组 CoC 高斯，渲染后加权 | 更灵活的离焦核表达 | 完整原方法涉及学习模块 |
| DoF-Gaussian | RGB/深度上的 CoC 空间变化散射 | 简单二维对照 | 单独移植散射不一定 |

计算成本、质量与兼容性均为设计层判断，不是本项目实测结果。

## 3. 解析高斯离焦卷积：DOF-GS 与 Cinematic Gaussians

**论文依据。** DOF-GS（CVPR 2025）将有限光圈产生的 CoC 模糊核引入二维投影高斯的渲染；Cinematic Gaussians（Pacific Graphics 2024）采用高斯近似光圈核并同时研究 HDR 辐射场、曝光和景深。

设原始二维投影协方差为 Σ，离焦核为各向同性高斯，协方差为 σ_b²I，则卷积后 Σ′ = Σ + σ_b²I。对未归一化高斯响应，若目标是保持单颗高斯积分，可用 sqrt(det Σ / det Σ′) 对峰值进行归一化。实际实现应核对原论文的不透明度、截断和 alpha 合成约定。σ_b 必须按同一光学参数和 CoC 半径/直径定义标定；不能直接将任意 CoC 数值当作标准差。

**优点（判断）：** 一次主要光栅化，避免逐光圈子视角重复投影，可能显著加速；确定性卷积便于可微训练。Cinematic Gaussians 的 HDR 处理也与本项目高光截断问题相关。

**局限：** 圆形均匀光圈并非高斯核；先卷积各颗高斯再合成与逐光圈视角重新排序、遮挡、合成后积分通常不等价。细杆、遮挡边界及背景显露处尤其值得比较。较大的高斯足迹也会增加 tile 覆盖成本，不能预先承诺加速比。

阅读优先：DOF-GS 的离焦渲染及光学参数部分、Cinematic Gaussians 的 aperture Gaussian/HDR 章节。移植目标是独立离焦渲染器，不等同于完整论文方法复现。

- DOF-GS：[论文](https://arxiv.org/abs/2405.17351) · [项目](https://dof-gs.github.io/) · [代码](https://github.com/rongduo/DOFGS_implementation)
- Cinematic Gaussians：[论文](https://arxiv.org/abs/2406.07329) · [项目](https://cinegs.mpi-inf.mpg.de/) · [代码](https://github.com/Hans1984/CineGS)

## 4. Closed-Form Convolution for Physically-Accurate Defocus in Gaussian Splatting

2026 年 ACM TOG 工作。根据公开摘要，使用误差函数（erf）推导高斯与光圈核的解析卷积，并以矩形分解近似多边形光圈，旨在改善简单高斯模糊核的形状近似。各向异性投影、矩形分解误差和合成顺序需读全文确认。

**价值（判断）：** 可能是比简单协方差相加更精确的快速渲染备选，特别适合检查高光光斑、圆形/多边形光圈轮廓。精确单颗响应不等于精确跨表面遮挡，不能据标题认定解决漏光和隐藏背景。

- [DOI: 10.1145/3821567](https://doi.org/10.1145/3821567)
- [杭州电子科技大学发表公告](https://bmci.hdu.edu.cn/2026/0830/c7486a299831/page.htm)

本轮核实发表信息及公开摘要，未取得全文、未确认官方源码；暂列阅读候选，不登记复现。

## 5. 3DGRT / 3DGRUT：高斯光线追踪

3D Gaussian Ray Tracing（3DGRT，SIGGRAPH Asia 2024）利用包围体层次结构（Bounding Volume Hierarchy, BVH）和 GPU 光线追踪查询光线经过的高斯，沿光线排序与合成。不同于现有 K64 的逐子相机投影光栅化，可以直接组织像素及光圈采样光线。光圈积分仍然需要多采样，并不意味着只计算一条光线。3DGUT（3D Gaussian Unscented Transform）是非线性投影/光栅化方向，3DGRUT 将两类渲染形式结合；不能将 3DGUT 本身等同于光线追踪。

官方 3DGRUT playground 提供 samples per pixel、aperture size 和 focus Z 控制。这说明已有景深功能，但不证明其能直接兼容本项目训练或 F2 合成。

**本项目优先考虑的质量诊断：** 固定同一场景，以同一光圈、焦平面、像素采样及背景合成对比 K64 raster 与 K64 ray tracing。若漏光下降，提示原渲染机制存在可改善空间；若仍存在，则应检查表面接缝、足迹、opacity、隐藏颜色与几何。不能将差异直接归因于排序一项，因为响应核与合成实现也可能不同。光线追踪不会自动将半透明高斯变成实体表面，也无法补全缺失几何。

迁移前核对：坐标系与相机内外参、尺度/协方差、旋转格式、opacity/alpha 约定、粒子响应与截断阈值、颜色/SH、深度排序、F2 剩余透射率、零光圈一致性、BVH 构建成本与动态场景更新、可微训练接口、RTX 5090 环境。若不能复用完全一致的高斯参数和合成协议，应明确记录差异，不称为严格单变量实验。

- [3DGRT 论文](https://arxiv.org/abs/2407.07090)
- [3DGUT 项目](https://research.nvidia.com/labs/toronto-ai/3DGUT/)
- [3DGRUT 官方代码](https://github.com/nv-tlabs/3DGRUT) · [景深控制说明](https://github.com/nv-tlabs/3DGRUT/blob/main/threedgrut_playground/README.md)
- [NVIDIA Vulkan Gaussian Splatting](https://github.com/nvpro-samples/vk_gaussian_splatting)

## 6. CoCoGaussian：多组高斯表达 CoC

CVPR 2025。由深度和光圈信息计算 CoC，并通过学习的高斯偏移等参数表达弥散圆。原方法包含多组渲染及组合，不应简化成所有偏移高斯一次统一合成，也不等价于真实光圈逐视角积分。

**价值（判断）：** 比单个扩大后的高斯更灵活；可参考如何表达不同形状的离焦响应。完整移植可能涉及训练离焦参数预测模块，成本及几何兼容性待验证。优先级低于确定性解析卷积和 3DGRT。

[论文](https://arxiv.org/abs/2412.16028) · [项目](https://jho-yonsei.github.io/CoCoGaussian/) · [代码](https://github.com/Jho-Yonsei/CoCoGaussian)

## 7. DoF-Gaussian：中心视角 RGB-D 散射

CVPR 2025，与 DOF-GS 是不同论文。其离焦渲染算法由清晰 RGB 与渲染深度计算每个像素的 CoC，执行空间变化的像素散射与归一化。论文的整体训练/相机优化不等于这个可单独提取的后处理步骤。

**价值（判断）：** 作为二维深度引导虚化对照，可测试三维光圈渲染的必要性。也可考虑固定可见/隐藏分层输入，再采用 Dr.Bokeh 风格的分层合成（这是本项目候选组合，不是 DoF-Gaussian 原方法）。单一中心 RGB-D 不保留所有隐藏表面，容易在遮挡处丢失真实光圈可见性。

[论文](https://arxiv.org/abs/2503.00746) · [项目](https://dof-gaussian.github.io/) · [CVPR 正式论文入口](https://openaccess.thecvf.com/content/CVPR2025/html/Shen_DoF-Gaussian_Controllable_Depth-of-Field_for_3D_Gaussian_Splatting_CVPR_2025_paper.html)

## 8. 候选比较安排（待用户另行批准执行）

原则：冻结同一个三维场景和权重，尽量只替换渲染器。保留当前数据划分与评价协议，不使用独立测试集调参。首先复用现有少量代表性场景（连续平面、直边卡片、细杆/窄缝、复杂遮挡及高光；选择具体 case 时再确认），不先扩大训练。

| 比较臂 | 目的 |
|---|---|
| K64 原渲染器 | 统一现有基准 |
| 同光圈采样的 3DGRT | 隔离高斯渲染机制差异 |
| DOF-GS 或 CineGS 卷积 | 检查速度收益及遮挡近似代价 |
| Closed-Form Convolution（条件允许） | 检查光斑形状及解析核收益 |
| RGB-D 散射（可选） | 检查三维可见性优势 |
| CoCoGaussian（可选） | 学习型离焦近似对照，若重训需单列 |

统一物理尺度、焦距、对焦距离（固定共同焦平面，避免自动选焦混淆）、光圈形状与半径、曝光/线性 RGB/截断、像素面积采样约定、背景 F2 合成、输入场景与评价掩码。零光圈检查是参数兼容性诊断；另可用更密光圈采样和 Cycles 参考检查积分误差。针对已知不纯净的 GT 隐藏层替身，不直接复用其远焦分数作为质量上界；依当前正式协议优先使用可核验的 F2/实际 Query 对照，若有正确表面颜色的 GT 再加入。

指标：全图 PSNR/SSIM/LPIPS，近焦焦内 PSNR、远焦可恢复区域 PSNR、边界 ±2 px、细结构 CoC、零光圈保真、有效隐藏光线及漏光透射率（如可比），逐场景配对与合并值并列。速度和显存统一 RTX 5090，区分固定场景的单次渲染与完整 E2E；记录 BVH/预处理开销、采样数量、动态逐帧更新成本，不能以论文异硬件 FPS 当本项目结果。

## 9. 优先级及结论

1. 3DGRT：优先作为质量诊断，检查当前 K64 光栅化与高斯表示的可见性差异。
2. DOF-GS / Cinematic Gaussians：优先作为效率方向，尽可能保留物理控制并减少重复投影。
3. Closed-Form Convolution：先取得全文、核实源码与合成方式，再判断移植。
4. CoCoGaussian、DoF-Gaussian：补充离焦核和二维后处理的比较。

以上是渲染器候选，不是新的网络设计或已验证实验结论。当前保留 K64，继续推进已经确认的前馈表面表示与隐藏层工作。若将来执行对照，再由用户确认范围及安排；本文不自动触发训练、代码改动或当前进度更新。
