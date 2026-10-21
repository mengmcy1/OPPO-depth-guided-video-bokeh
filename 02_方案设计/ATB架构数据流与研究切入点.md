# Any-to-Bokeh：完整数据流、关键模块与研究切入点

> 定位：对 ATB 原理与本项目研究问题的学习笔记。先讲清输入从哪里来、每个网络做什么、数据如何流动，再讨论可以在哪里改进。
>
> 依据：ATB 官方公开推理/预处理代码 + 本仓库实验记录；**不是新方法设计定稿，也不是新的实验结果**。
>
> 核对日期：2026-10-21。实验状态按本仓库截至 2026-10-15 的最近正式记录整理，具体更新以 [当前实验进度](../03_实验记录/当前实验进度与方法论.md) 为准。

## 1. 先记住三个结论

1. **ATB 不是「图像 → ViT → VAE → UNet」的串行模型。** RGB 帧分别进入 CLIP-ViT 与 RGB VAE Encoder；另一条外部几何路径由 VDA 和目标区域生成 CoC/MPI 条件。各路特征最后在 UNet 中融合。
2. **RGB 和 CoC 只走 VAE 的编码部分；最终的 Temporal VAE Decoder 处理的是 UNet 生成的虚化 Latent。** 解码时还会用到 RGB 编码器保存的多尺度特征；ViT 特征不是直接输入 Decoder。
3. **目标 Mask 的主要作用是确定焦平面，不是「白色像素强制清晰」。** 官方公开流程中，VDA 提供几何、Grounding DINO/SAM2 或外部标注提供目标区域，两者结合后生成 CoC 与 MPI Mask。

## 2. 完整推理数据流

~~~mermaid
flowchart TD
    V["输入 RGB 视频"] --> A["CLIP 图像编码器：ViT"]
    V --> B["RGB VAE Encoder"]
    V --> D["Video Depth Anything<br/>逐帧几何估计"]
    V --> P["选定对焦目标<br/>Grounding DINO + SAM2，或外部 Mask"]
    D --> Q["整段视频归一化、量化<br/>q：0–255"]
    P --> M["主体二值 Mask"]
    Q --> F["Mask 内平均视差<br/>焦平面 f"]
    M --> F
    Q --> C["构造相对离焦图<br/>c = |q/255 - f|"]
    F --> C
    Q --> MM["构造 MPI 分区 Mask"]
    F --> MM
    C --> CV["CoC VAE Encoder<br/>与 RGB 使用同一 VAE"]
    A --> U["时空 UNet + 注意力"]
    B --> U
    CV --> U
    MM --> U
    F -. "用于条件构造" .-> U
    K["K：虚化强度"] --> U
    U --> Z["生成 Bokeh Latent"]
    Z --> W["WOIS：窗口重叠加权融合"]
    W --> VD["Temporal VAE Decoder"]
    B -. "多尺度 RGB 特征缓存<br/>Skip Connection" .-> VD
    VD --> OUT["输出 RGB 虚化视频"]
~~~

**说明：** 流程图把「目标选取/分割」放在外部预处理层。官方公开脚本可使用 Grounding DINO + SAM2；本项目 DAVIS 复现使用数据集现成的主体标注确定焦点。VDA、目标分割、编码、生成不一定在同一个 Python 推理调用中顺序执行。

### 2.1 先处理原始视频：VDA + 主体 Mask

**Video Depth Anything（VDA）** 是外部视频深度估计模型。官方预处理使用 ViT-Large 配置，先对视频逐帧输出几何值，再以**一段视频统一的 min/max** 归一化到 0–255，并存成图像：

\[
q_t(x,y)=\operatorname{uint8}\left(
255\frac{d_t(x,y)-d_{\min}}{d_{\max}-d_{\min}}
\right).
\]

- 官方变量将该结果称为 disparity（视差），但公开预处理里是将 VDA 的输出直接归一化，**没有显式进行「1/物理深度」转换**；不要把它解释为有真实尺度的相机视差。
- 使用一段视频共同的最大/最小值，有助于不同帧采用相同数值范围，但不等于保证每个结构的深度正确或时序稳定。
- 若几何范围为零、输出异常或主体区域为空，必须明确报错，而不是在分析中臆造可用视差。具体输入规则见仓库评价协议。

**主体 Mask** 是二值分割图：\(M_t(x,y)\in\{0,1\}\)。它说明选中的对焦物体在哪里，不包含物理深度。

公开示例有两种准备思路：

- **自动定位：** 给 Grounding DINO 一段文本（如「goat」）→ 找到目标检测框 → SAM2 根据检测框进行精确分割，并向视频其他帧传播。
- **外部指定：** 用户点击/框选，或者直接给现成标注 Mask。SAM2 本身不会决定「应该对谁对焦」。

注意：官方示例脚本把文件夹名转换成文本提示，并选置信度最高的一个候选框；这**不等同于任意场景都能自动、正确地选择用户想对焦的实例**。

对焦值来自 Mask 内的预测视差平均值：

\[
f_t=\frac{\sum_{x,y}M_t(x,y)\,[q_t(x,y)/255]}
{\sum_{x,y}M_t(x,y)}.
\]

**重点：** Mask 是用来计算 \(f_t\) 的；生成网络的输入不是「Mask 内强制保持清晰」。一个主体内部如果深度预测差异很大，局部仍可能有较大离焦。

### 2.2 用视差和焦点构造两路几何控制条件

**第一路：CoC / 相对离焦图。** 公开数据加载器按以下关系构造：

\[
c_t(x,y)=|q_t(x,y)/255-f_t|.
\]

单通道 \(c_t\) 复制为 3 通道后，经 **同一个 VAE Encoder** 压缩成 CoC Latent，再和 RGB Latent 拼接输入 UNet。

- 这一步的代码**没有先乘 K**；K 作为独立强度参数传入 UNet。
- 分析时常使用 \(r_{\rm nominal}=Kc_t\) 说明名义离焦大小，但它**不是最终输出中实测的光学模糊半径**。

**第二路：MPI Mask。** 根据 \(q/255\)、\(f\) 把像素分配到若干离焦分区，作为 UNet 中 MPI Attention 的控制信息。

- **主体 Mask：** 表示「目标是谁」，主要参与算焦点。
- **MPI Mask：** 表示「像素属于哪个相对离焦区」，直接参与生成网络的几何注意力。
- 二者名称相似，作用和进入模型的位置不同。

### 2.3 RGB 帧同时进入 ViT 和 VAE：不是先后串联

| 分支 | 实际网络/操作 | 输入 | 输出 | 进入哪里 |
| --- | --- | --- | --- | --- |
| 全局视觉 | CLIP 视觉编码器（ViT） | RGB 缩放至约 224×224 | 每帧的整图 image embedding（扩成长度为 1 的条件序列） | UNet 的条件注意力 |
| 原图内容 | 卷积型 VAE Encoder | RGB 3×576×1024 | RGB Latent，通常为 4×72×128；另缓存多尺度特征 | Latent 进 UNet；缓存供最终 Decoder |
| 几何条件 | **同一个**卷积型 VAE Encoder | CoC/相对离焦图复制成 3 通道 | CoC Latent，通常为 4×72×128 | 与 RGB Latent 通道拼接进 UNet |
| 离焦分区 | 规则构造 | 归一化视差 + 焦点 | MPI Mask | UNet 内部几何注意力 |
| 额外参数 | 条件嵌入 | K、FPS、运动等 | 条件向量 | UNet |

**ViT 的 Token 容易理解错：** ViT 内部确实把图像切成 Patch Token，但 ATB 公开 Pipeline 取的是视觉模型输出的整图 **image_embeds**，然后执行 unsqueeze(1)。它不是把整串原始 Patch Token 直接喂给 UNet。

**VAE 是什么模型？** 这里使用 SVD/ATB 的 **AutoencoderKLTemporalDecoder**。Encoder 以卷积、ResNet Block、下采样为主；Decoder 有上采样和时序模块。它不是另一个 ViT。以 576×1024 输入为例，空间尺寸压缩为 72×128，通道从 RGB 的 3 变为潜在表示的 4：

\[
I_t\in\mathbb R^{3\times576\times1024}
\quad\xrightarrow{E_{\rm VAE}}\quad
z_t\in\mathbb R^{4\times72\times128}.
\]

4 个潜在通道并不对应 RGB 或某四种人工语义，而是网络学习到的表示。VAE 编码还先构造高斯潜在分布，再从中采样，官方 Pipeline 的接口是 latent_dist.sample()。

**为什么 RGB 要「编码与缓存」？** 官方在 RGB 路径设置 cache_flag=True，缓存 VAE Encoder 各下采样阶段的多尺度卷积特征；最终 VAE Decoder 会通过 Skip Connection 复用。CoC 路径走同一个 VAE Encoder，但不保存这一组 RGB 解码缓存。保存较高分辨率的特征是显存的重要来源，不能把「缓存」理解成仅保存一个 4×72×128 的小张量。

### 2.4 U-Net 不只是纯 CNN

传统 U-Net 是卷积网络的 U 形编码—解码结构。ATB 沿用 SVD 的 **UNetSpatioTemporalConditionModel**：仍有 CNN/ResNet、下采样、上采样，同时增加空间/时间注意力和条件交互，并加入 MPI 相关模块。

主要接口可以这样记：

1. RGB Latent 和 CoC Latent 沿通道拼接，得到 **8 通道**的空间潜在输入。
2. ViT 的 image embedding 作为 **encoder_hidden_states** 参与注意力条件；这不是进入最终 VAE Decoder。
3. RGB Latent 还作为 **mpi_cond** 的条件来源，配合 MPI Mask、K 等作用于几何注意力。
4. UNet 单步预测目标 Bokeh Latent，输出通常为 **4 通道**。

省略 batch/time 维度：

\[
[4,72,128]_{\rm RGB} \oplus [4,72,128]_{\rm CoC}
\longrightarrow [8,72,128]
\xrightarrow{\mathrm{UNet}}
[4,72,128]_{\rm Bokeh}.
\]

UNet 内部也有 U 形「Decoder/上采样」，但**这不是将 Latent 还原为 1024×576 RGB 的 Temporal VAE Decoder**。

### 2.5 长视频窗口融合与最终解码

ATB 采用 **WOIS（Weighted Overlap Inference Strategy，加权重叠推理策略）**：

- 把长视频分成有重叠的窗口；
- 对重叠的预测 Latent 加权融合；
- 将融合后的 Latent 送入 Temporal VAE Decoder；
- Decoder 借助 RGB Encoder 缓存的多尺度特征还原高分辨率 RGB，并进行时序处理。

简化表示：

\[
I^{\rm Bokeh}=D_{\rm temporal\,VAE}
\left(\mathrm{WOIS}\{z^{\rm Bokeh}_{\rm window}\},\,F^{\rm RGB}_{\rm cache}\right).
\]

**这里没有「最后再把 ViT Token 和 VAE Latent 直接拼给 Decoder」的步骤**；ViT 条件已经在 UNet 内发挥作用。

## 3. 一个视频到底对应哪些输入文件？

官方公开演示将部分输入预先制作好，再交给推理入口：

| 输入/中间数据 | 典型来源 | 作用 |
| --- | --- | --- |
| RGB 帧 | 原视频拆帧 | 供 ViT、RGB VAE 和 VDA |
| 目标 Mask | Grounding DINO + SAM2、用户指定或数据集标注 | 计算对焦位置 |
| 深度/视差 PNG | VDA 输出，经全视频归一化量化 | 生成 CoC 与 MPI 分区 |
| 焦点值 f | 目标 Mask 内的归一化视差均值；官方公开流程可写入文件名 _zf_ 字段 | 控制对焦平面 |
| K | 配置或逐帧参数 | 控制虚化强度 |
| CoC 与 MPI Mask | 加载 RGB/视差时计算 | ATB 生成条件 |

**本项目 DAVIS 的特殊性：** 目标区域使用 DAVIS 已有标注，不是每次都重新运行 Grounding DINO/SAM2；VDA 则用于公共几何输入。不能把这两条数据来源混为一谈。

**效率评价边界：** 本项目已测的 ATB Core 与 E2E-given-inputs 是**已有几何/控制输入之后**的执行口径，不包含从原始视频调用 VDA 和目标分割的全部开销。若以后要比较原始视频到完整输出的真实系统成本，需另立一致的端到端边界。

## 4. 讨论形成的研究切入点（重点）

下面按「现象/原因假设 → 可以改哪里 → 如何验证 → 当前证据」整理。它们是**候选方向**，不是已经实现的 Ours。与既有 [几何可靠性与时序推理方案](ATB改进候选方案-几何可靠性与时序推理.md) 配合阅读；本文件重点是解释各改动插在数据流的什么位置。

### 方向 A：VDA 几何可靠性和局部细结构修正（目前证据最直接）

**问题：** VDA 在主体内部、细线或遮挡边缘的几何预测可能不合适，经过焦点/CoC/MPI 构造后被转化为不期望的虚化。

**改动位置：** VDA 输出 \(q\) → 局部几何可靠性判断/修正 → **重新构造 CoC 与 MPI** → 冻结的 ATB。

**研究候选：**

- 识别主体内部不合理的深度跳变，尤其是细结构或连续物体；
- 用 RGB 结构线索、邻近可靠部位、跨帧对应建立局部修正参考；
- 进一步研究任务相关可靠性，而非只优化一般深度误差；考虑焦点和 K 如何放大偏差；
- 先冻结 UNet/VAE，验证仅改变输入几何是否足以带来有意义的视觉改善。

**已有证据与限制：**

- **B36 鸭头：已完成受控局部几何干预。** 原始第 0 帧鸭头名义离焦约 2.400、颈部约 0.204、身体约 0.580。将头部的视差局部替换为颈部参考值、固定 RGB/focus/K/权重后，鸭头对原 RGB 的区域 MAE **7.417 → 2.575**、梯度能量比 **0.429 → 0.836**，用户也偏好这一局部调整；邻近水面同幅度干预未产生相同的鸭头恢复。支持**当前几何条件对该输出具有因果贡献**。但参考值不是几何真值，诊断用标注辅助了区域选择，也尚未获得跨来源的通用修正方法。
- **B33 细绳：比「视差根本没识别细线」复杂。** RGB 对齐位置上仍有较弱的独立视差轮廓；帧 1→22 的离焦增长主要由汽车跟焦的焦点变化贡献。针对帧 22 的 3-native-pixel 窄带视差调整效果很弱；将焦点改回早期数值会影响汽车，不能作为有效的局部修复。不能把 B36 的成功外推到 B33。
- **B19 伞绳反例：** 六方法局部人工对照中 ATB 反而相对保留得更清楚。不能只搜集「ATB 不好」的例子。

**最小验证问题：** 当局部几何被修正且其他条件不变，错误虚化是否选择性改善？是否同时伤害背景、其他帧或正常场景？如果无稳定改善，不能继续把深度修正当成已证实的主路线。

### 方向 B：主体 Mask / 焦平面选择的鲁棒性（几何方向的独立子问题）

**问题：** Mask 只决定平均焦点，不直接保护主体。主体可能包含多个深度层次；多个同类实例时文本检测也可能选错主体。

**改动位置：** 对焦目标选择 → 主体 Mask → 焦点 \(f_t\) → CoC/MPI。应和直接修正 VDA 分开测。

**可以考虑：**

- 将「目标检测是否选对实例」与「已选目标内焦点是否合适」分开；
- 对比不同事先固定的焦点定义（主体均值、代表点、稳健统计等），但不能根据测试效果事后选值；
- 利用跨帧目标轨迹判断焦点跳变，重点检查场景运动和主体自身深度跨度；
- 如果确有必要，再考虑前景局部保护或多焦面表示，不能仅凭 Mask 与平均值不同就推断需要新网络。

**重要反证：** B36 的主体中位数与原平均焦点差很小，**简单将平均值改为中位数不足以解释/解决鸭头问题**；只把全局焦点改到鸭头虽能影响头部，也同时明显改变身体等区域。故焦点优化不能代替局部几何可靠性研究。

**最小验证问题：** 保持 VDA 的视差、UNet、K 不变，只改焦点构造，是否有稳定收益且不会损害其他主体区域？

### 方向 C：区分 CoC 与 MPI 条件各自的贡献

**问题：** VDA 的同一误差会同时改变 CoC Latent 和 MPI Mask。直接改视差后观察到画质变化，**还不能区分究竟是哪一路条件引起的**。

**改动位置：** VDA 输出后的条件构造；保持主模型冻结，在合适的受控实验中分别改变两条条件路径。

**建议诊断顺序：**

1. 原始 CoC + 原始 MPI；
2. 仅调整 CoC 条件，MPI 保持原始；
3. 仅调整 MPI Mask，CoC 保持原始；
4. 两条路径同时根据同一几何调整。

同时记录 RGB、focus、K、随机流、窗口划分、原始输出以及其他区域变化。对于模型训练时未见过的矛盾条件组合，需明确它是**机理探针**而不是自然场景的公平质量比较。

**当前状态：** B36 已验证「改视差、同时重建 CoC 和 MPI」可以影响输出；**尚未拆开证明两条条件路径谁起主要作用**。如果只需要解释现象，这一步可以先做诊断，不必立刻设计新注意力网络。

### 方向 D：VAE 缓存、编码冗余与 WOIS 效率

**问题：** UNet 是核心生成器，却不一定最耗时。本项目 4090 D 的 25 帧剖析显示：

| 模块计时阶段 | 占带插桩 Core |
| --- | ---: |
| CLIP 图像编码 | 0.79% |
| RGB VAE 编码与缓存 | 22.71% |
| CoC VAE 编码 | 18.78% |
| UNet 时空条件处理 | 18.55% |
| 重叠拼接与时序解码 | 38.87% |
| 其他 | 0.29% |

VAE 相关的两个编码阶段和最后「拼接 + 解码」阶段合计 **80.36%**。注意最后一项还包含窗口融合，因此**不能把 80.36% 全部说成 VAE 神经网络算子耗时**。

已有第一个实现级优化：VAE Encoder 的某 ResNet Block 在 output_scale_factor=1 时原地执行了无必要的除法，跳过它后，在固定四段测试中输出逐位一致；**设备峰值显存约减少 4.50 GiB，Core 用时约减少 1.10%**。这是已经实测的工程优化，不自动视为新算法贡献，也不据此推断 32 GiB 设备已能运行。

可以研究的下一步：

- **Encoder 缓存：** 哪些高分辨率缓存必须保留，哪些可以安全复用、压缩或分阶段释放？
- **WOIS 重复计算：** 重叠窗口导致同一实际帧在多个输入帧槽中被编码；能否复用 ViT/CoC/RGB 特征？需验证随机采样、特征缓存、窗口上下文是否影响等价性。
- **Decoder/窗口融合：** 分离 WOIS 权重相加与真正 VAE 解码的成本，再决定优化重点；不能把重叠拼接与时序解码混成一个被认为全可省的模块。
- **时序稳定性：** 如果要改变重叠长度或做跨窗口特征记忆，必须同时检查视频质量、帧间连续性、峰值内存及完整 E2E；不能拿单帧耗时估算视频收益。

这条方向已有明确性能成本证据，**但高效 WOIS / 新跨窗口算法尚未验证成功**。

## 5. 研究顺序建议：先做可归因的事

| 优先任务 | 已有依据 | 下一个需要回答的问题 | 不应提前宣布 |
| --- | --- | --- | --- |
| A：局部几何可靠性 | B36 受控干预有选择性响应，B33 有反例/复杂条件 | 在更多帧和独立来源是否能持续改善，且可自动检测？ | 真实深度已恢复、通用 Ours 已成立 |
| B：Mask/焦点因素 | Mask 只决定焦点；B36 全局改焦点有旁效应 | 能否在不改几何的情况下获得局部且稳定的收益？ | 平均值就是错误根因 |
| C：CoC vs MPI 归因 | 改几何会同时影响两路条件 | 哪条路径造成主要响应，两者是否互补？ | 已锁定具体神经模块错误 |
| D：效率与特征复用 | VAE 阶段高成本，scale=1 优化已实测 | 是否有更大的安全复用空间，效果与时间如何权衡？ | 理论帧槽减少=实测等比例加速 |

原则：

1. 一次只改变一个可解释因素；原版、匹配对照、反例必须保留。
2. 区分「输入几何与结果相关」「受控修改确实影响结果」「模型机制已解释」「方法跨来源有效」四种不同证据层次。
3. 诊断所用 GT/主体标注可以帮助**分析现象**，但不能未经说明就作为未来实际 Ours 的必需输入或当成公平推理权限。
4. DAVIS 无配对虚化 GT，局部 MAE/梯度反映变化与细节保留，不等于真实镜头虚化准确率；B19 反例与正常场景都要保留。
5. 如选择效率线，继续保留 Table 3A（模型成本）与 Table 3B（25 帧 Core/E2E）；VDA/SAM2 前处理不能悄悄算进一部分方法、却对其他方法不算。
6. 当前不自动启动新的 GPU 实验、训练或更换已冻结的 baseline；正式实施前再依据最新进度确定实验范围和输入划分。

## 6. 术语速查

| 名称 | 一句话理解 |
| --- | --- |
| **ViT**（Vision Transformer） | 用 Transformer 建模图像特征；ATB 的 CLIP 视觉编码器使用 ViT |
| **CLIP**（Contrastive Language–Image Pre-training） | 图文对比预训练模型；ATB 使用它的图像编码分支 |
| **VAE**（Variational Autoencoder） | 将图像编码到较小的潜在空间，并可再解码还原 |
| **Latent**（潜在特征） | 不再是 RGB 颜色值、而是神经网络学习到的低分辨率表示 |
| **UNet** | 具有下采样—上采样和跳跃连接的网络；ATB 的时空 UNet 内还含注意力 |
| **Self-/Cross-Attention**（自/交叉注意力） | 特征之间建立关联；或让当前特征参考外部条件 |
| **VDA**（Video Depth Anything） | 上游视频深度估计模型；不在冻结 ATB Core 计时边界内 |
| **Grounding DINO** | 根据文本提示寻找目标并返回检测框 |
| **SAM2**（Segment Anything Model 2） | 根据点/框/Mask 提示分割、跟踪指定目标；不自行决定该对谁对焦 |
| **主体 Mask** | 目标区域二值图，主要用于计算焦点 |
| **MPI Mask** | 按离焦条件构造的分区图，用于 UNet 几何注意力 |
| **CoC**（Circle of Confusion） | 弥散圆概念；ATB 输入使用归一化相对离焦图 |
| **WOIS**（Weighted Overlap Inference Strategy） | 重叠窗口生成结果加权融合 |

## 7. 代码和记录索引

**作者官方实现：**

- [README：Grounding DINO/SAM2、VDA 与推理输入](https://github.com/vivoCameraResearch/any-to-bokeh/blob/main/README.md)
- [any2bokeh_sam.py：文本选目标 → 检测框 → SAM2 跟踪](https://github.com/vivoCameraResearch/any-to-bokeh/blob/main/any2bokeh_sam.py)
- [utils/pre_process.py：VDA 推理、整段归一化、Mask 内取焦点](https://github.com/vivoCameraResearch/any-to-bokeh/blob/main/utils/pre_process.py)
- [utils/dataset.py：CoC 与 MPI Mask 计算](https://github.com/vivoCameraResearch/any-to-bokeh/blob/main/utils/dataset.py)
- [pipelines/any2bokeh_pipe.py：ViT / VAE / UNet / WOIS / 解码数据流](https://github.com/vivoCameraResearch/any-to-bokeh/blob/main/pipelines/any2bokeh_pipe.py)
- [models/vae.py：带 RGB Skip 缓存的 VAE Encoder / Temporal Decoder](https://github.com/vivoCameraResearch/any-to-bokeh/blob/main/models/vae.py)
- [models/unet.py：时空 UNet、MPI 条件入口](https://github.com/vivoCameraResearch/any-to-bokeh/blob/main/models/unet.py)

**本仓库：**

- [当前实验进度与方法论](../03_实验记录/当前实验进度与方法论.md)
- [Benchmark 与 Baseline 统一评价协议](../03_实验记录/Benchmark与Baseline统一评价协议.md)
- [ATB 改进候选方案：几何可靠性与时序推理](ATB改进候选方案-几何可靠性与时序推理.md)
- [ATB 模块分析记录](../03_实验记录/records/any2bokeh-native-module-profile-4090d-20261001.json)
- [Encoder scale=1 优化记录](../03_实验记录/records/any2bokeh-encoder-scale1-experiment-4090d-20261008.json)
- [B33 受控诊断](../03_实验记录/records/any2bokeh-B33-causal-diagnosis-20261015.json)
- [B36 受控诊断](../03_实验记录/records/any2bokeh-B36-causal-diagnosis-20261015.json)

---

**一句话概括：** VDA 提供场景几何，主体 Mask 决定焦点；ViT 提供全局视觉条件，VAE 将 RGB/CoC 压到潜在空间，时空 UNet 生成虚化 Latent，WOIS 与 Temporal VAE Decoder 恢复最终视频。当前最值得验证的是**局部几何可靠性、焦点选取与两条几何条件的归因**，以及已有性能证据支撑的**VAE/WOIS 效率优化**。