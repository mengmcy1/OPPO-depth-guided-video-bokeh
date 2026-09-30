# Benchmark、Baseline 与统一评价协议

> 当前版本：VBS 移除后的统一实验协议  
> 适用项目：OPPO Depth-Guided Video Bokeh  
> 目的：在开始大规模复现和自研模型前，先固定 Benchmark、Baseline、评价指标、输入公平性和结果表格，后续所有算法尽量按同一协议填写结果。

---

# 1. 当前实验目标

本项目最终关注的是：

\[
\text{High-FPS RGB Video}
+
\text{Low-FPS Coarse Depth}
\rightarrow
\text{High-FPS Stable Video Bokeh}
\]

并计划同时涉及：

- High-FPS Dense Depth；
- Alpha Matte；
- Physical Bokeh Rendering；
- Neural Residual / Refinement；
- Temporal Modeling。

实验需要回答四类问题：

1. 最终 Bokeh 画质是否好？
2. 视频时序是否稳定且保留真实变化？
3. 低帧率、粗糙、带误差的 Depth 是否会导致明显性能下降？
4. 模型的效率是否适合后续工程化？

---

# 2. 当前 Benchmark 总体结构

VBS 暂时从正式实验方案中移除，原因是当前尚未找到可靠、可核实的公开下载入口，不让不可获取的数据集阻塞实验主线。

当前保留三层：

~~~text
Benchmark Suite
│
├── 1. Paired Synthetic Video Bokeh Benchmark
│      └── Any-to-Bokeh Synthetic
│          ├── Official Test Set（若后续可获得）
│          └── Reproduced Any-to-Bokeh-style Set（Plan B）
│
├── 2. DAVIS
│      └── Real-world Generalization
│
└── 3. OPPO-style Low-FPS Coarse Depth Benchmark
       └── Project-specific Benchmark
~~~

说明：

- **Any-to-Bokeh Synthetic**：负责有 GT 的主定量评价；
- **DAVIS**：负责真实视频泛化；
- **OPPO-style**：负责 Low-FPS / Coarse / Noisy Depth 条件下的专项评价；
- **VBS**：仅保留为未来可选补充项，若后续官方公开且容易获取，再考虑加入，不进入当前执行主线。

---

# 3. Benchmark 1：Any-to-Bokeh Synthetic

## 3.1 定位

这是当前最重要的 paired video Bokeh 主定量 Benchmark。

参考 Any-to-Bokeh 论文测试协议：

- synthetic paired video；
- 有 Bokeh GT；
- 有 Geometry / disparity；
- 有 focus plane；
- 有 blur strength；
- 视频级测试；
- 可进行完整的全参考定量评价。

论文参考规模：

- 200 videos；
- 25 frames / video；
- 1024 × 576。

当前官方完整测试包尚未确认可获得，因此设置两种情况：

### 情况 A：官方 Test Set 后续可获得

直接按照作者测试集和评价脚本进行正式复现。

### 情况 B：官方 Test Set 仍无法获得

按照官方论文和公开的 \`dataset_synthsis.py\` 生成逻辑构建固定的：

> **Reproduced Any-to-Bokeh-style Synthetic Benchmark**

必须明确标注为自建重现版本，不能声称与作者原 200-video test set 完全一致。

需要固定：

- 前景 / 背景数据来源；
- 随机种子；
- 视频数量；
- 视频长度；
- 分辨率；
- object 数量；
- geometry / disparity 生成方式；
- focus 设置；
- blur strength；
- renderer；
- train / test 隔离规则。
### 3.1.1 E7 的 VideoMatte 动态源策略与 Any-to-Bokeh 公开代码证据

Any-to-Bokeh 的公开材料不足以证明论文最终 200-video synthetic test set 使用了怎样的 source-level split，因此本项目不把“复现其公开生成脚本”与“复现其最终测试源划分”混为一谈。

当前可核实的官方代码证据来自 `vivoCameraResearch/any-to-bokeh` 的 `dataset_synthsis.py`：

- test-generation 配置中启用了 `BG_test`、`merge_f25_test.json` 和 `gen_test_flag=True`；
- 同一文件给出的 `merge_f25_test.json` 格式示例明确出现 `AM-2k/train/fg`、`AM-2k/train/mask`；
- 视频 matting 示例明确出现 `VideoMatte240K_JPEG_HD/train/fgr` 与 `train/pha`；
- 文件中还保留了直接指向 `VideoMatte240K_JPEG_HD/test/clip/clip_f25.json` 的注释路径，但实际启用的是未公开的 `merge_f25_test.json`；
- 公开脚本当前写有 `synthsis_images_num=300`，而论文报告的是 200 个 test videos，进一步说明该脚本是生成工具的公开版本，不能视为论文最终 test manifest 的逐项冻结副本。

因此，目前只能得到以下有边界的判断：

> Any-to-Bokeh 公开 test-generation 代码存在使用原始数据集 `train` source 生成 synthetic test 的明显痕迹；但关键的 `merge_f25_test.json` 未公开，不能据此断言论文最终 200 个 test videos 的具体 source 列表，也不能证明其训练/测试在 VideoMatte 或 AM-2k 层面实现了 source-level disjoint。

基于这一不确定性，E7 的动态前景建议采用两层设计，**当前仅作为待冻结方案，不代表正式场景清单已经确定**：

1. **Strict Held-out Dynamic Subset（主测试动态子集）**
   - 使用 VideoMatte240K 官方 `test` split 的 5 个独立源视频；
   - 每个源视频可以生成多个 25-frame clip、背景、Focus、Depth 和 \(K\) 条件，但必须始终保留原始 `source_video_id`；
   - 同一源视频的多个 clip / 多个光学条件不能当作新的独立动态源；
   - 动态子集统计至少保留 per-source-video grouping，必要时先在同一 source 内聚合，再对 5 个 source 求均值，避免某个源因派生 scene 更多而获得更高权重；
   - 这提供更清晰的官方 split provenance，但**不等价于证明任一预训练 baseline 从未见过这些人物或素材**。

2. **Dynamic Extended Stress Set（可选扩展压力测试）**
   - 若 5 个独立动态 source 对运动类型、发丝、衣物、遮挡覆盖不足，可从 VideoMatte `train` 中选择 E1–E6 未使用的额外源视频扩展动态场景；
   - 该扩展集必须与主 held-out 动态子集分开报告，并明确标注：Any-to-Bokeh 的公开 synthetic generation 使用过 VideoMatte `train` source，因此无法保证其预训练阶段未见过这些源；
   - 扩展集主要用于增加动态/边缘/遮挡压力覆盖，不作为最严格 source-held-out 结论的唯一依据；
   - 不按 Any-to-Bokeh 或其他 baseline 的测试分数选择 train source。

如果后续需要更强的“独立真实动态源”覆盖，优先考虑增加另一个具有明确 held-out test split、可提供前景与 Alpha 的视频 matting 数据源，而不是简单把大量 VideoMatte train source 混入主测试并当作独立 held-out 样本。

最终 E7 冻结时必须明确记录：

- 哪些结果属于 Strict Held-out Dynamic Subset；
- 哪些属于可选 Dynamic Extended Stress Set；
- 每个派生 scene 对应的原始 source ID；
- source-level 统计与 scene-level 统计的区别；
- Any-to-Bokeh 公开代码只能支持上述“生成方式迹象”，不能支持对其论文最终 200-video source split 的更强断言。


---

## 3.2 主指标

统一使用六个核心指标：

\[
\boxed{
PSNR,\ SSIM,\ LPIPS,\ FVD,\ FD,\ RM
}
\]

| Metric | 趋势 | 主要含义 |
|---|---:|---|
| PSNR | ↑ | 像素级重建准确性 |
| SSIM | ↑ | 结构相似度 |
| LPIPS | ↓ | 感知相似度 |
| FVD | ↓ | 视频整体分布质量 |
| FD | ↓ | 预测与 GT 的运动 / 光流关系差异 |
| RM | ↓ | 预测帧间变化与 GT 帧间变化的一致性 |

Any-to-Bokeh 原论文还使用 VFID-I。完整复现原论文时可以继续计算，但当前主表优先保留以上六个指标。

---

# 4. Benchmark 2：DAVIS

## 4.1 定位

DAVIS 负责 Real-world Generalization。

它是真实自然视频，但没有严格 paired 的真实 Bokeh GT，因此不能像 synthetic benchmark 一样直接使用 against-GT 的：

- PSNR；
- SSIM；
- LPIPS。

Any-to-Bokeh 官方仓库已经提供 DAVIS demo 和 VEPI 评价代码，因此 DAVIS 是当前最容易直接落地的真实视频 Benchmark。

---

## 4.2 主要指标

第一阶段使用：

\[
\boxed{
VEPI
+
Qualitative\ Video
+
Fixed\ Key\ Frames
}
\]

其中 VEPI 主要评价：

- focused subject edge；
- Hair；
- Thin Structure；
- 主体边缘是否被背景 Blur 吞噬。

后续如需要更正式的真实视频评价，再增加：

- blinded user study；
- MOS / preference study。

---

## 4.3 固定定性观察项

DAVIS 每段视频固定观察：

- Hair / Thin Structure；
- Foreground Halo；
- Background Leakage；
- Color Bleeding；
- Focus Error；
- Temporal Flicker；
- Occlusion / Disocclusion；
- Camera Motion；
- Fast Motion。

不自行构造没有文献依据的“1～5 分稳定性指标”。

## 4.4 DAVIS v2：公共数据规则固定，方法预检完成后再批量评价

用户已确认本轮先完成Block A公共预处理，并单独检查DeepLens native DepthNet；配置为`configs/davis-eval-v2.json`。这是**Reproduced DAVIS-2016 25-frame evaluation protocol**，不声称获得缺失的作者`davis_k16.csv`。正式版§4.1提到25帧评价/计时、A.4写DAVIS所有视频，但不能据此恢复确切作者窗口。

- **数据**：沿官方2016 train30+val20清单复用本机TrainVal包的50段，全源3455帧；不重新下载。原始JPEG和索引标注只读。按[DAVIS作者single-object loader](https://github.com/fperazzi/davis-2017/blob/main/python/lib/davis/dataset/base.py)的`annotation != 0`合并前景，派生PNG值域必须为`{0,255}`；不能保存为0/1再交给ATB灰度阈值。前25帧1250张必须具备非空ROI；后续上下文帧可因主体出画而mask为空，仍全部送入VDA，不给这些非评价帧强行定义focus。
- **几何**：VDA-Large/ViT-L相对深度，`input_size=518, fp32=False`。50个sequence**各自**完整推理、各自求一次T/H/W联合min/max；不跨视频合并，也不逐帧归一化。复用[ATB公开预处理](https://github.com/vivoCameraResearch/any-to-bokeh/blob/main/utils/pre_process.py)的截断量化`q=((d-min)/(max-min)*255).astype(uint8)`；前25帧focus由量化q在原分辨率二值mask内求均值/255，再以`{:.5f}`写入文件名。非finite、零span、缺帧/错配或评价ROI为空即报错，不静默跳过。
- **评价输入**：各段前25连续帧，K16，目标1024×576。普通geometry-driven方法共享RGB/q/f/K来源；BokehMe/BokehDiff保持已冻结Normalized行为，记录逐帧span及控制缩放，不临时补偿K。因此共享外部输入**不等于**各内部adapter的CoC物理等价。所有方法仍需相应real-world输入检查，不能把Synthetic通过自动当作DAVIS通过。
- **Dr.Bokeh**：DAVIS使用Full-public LDF+LaMa及共享VDA；GT mask只定义focus与VEPI ROI，不作为alpha/hidden layers。Synthetic reproduction主行仍是GT-alpha，不更改其历史配置。
- **DeepLens**：native TF1.15 CPU，原DepthNet→pixel focal depth→LensBlur→single SR。固定8段bear/blackswan/bmx-trees/breakdance/dog/horsejump-high/kite-surf/soapbox，前25帧；RGB在本预检用OpenCV线性resize到1024×576，AREA到512×288，原生BGR/uint8 SR行为保留。mask以nearest映射到原生512×288深度网格；选前景内最接近前景平均预测深度的真实像素，同值取row-major第一个，保持原graph的像素取焦接口。该自动选点是本项目reproduction adapter，非ATB私有协议证据；mask只参与控制选点，不输入DepthNet/renderer分割分支。
- **DeepLens aperture已冻结（用户确认）**：固定`aperture=0.5`，命名**native-default**，来源为官方UI的`valinit=5`再除10；不声称与共享K16物理等价或恢复ATB私有映射。8段预检已检查`0.25/0.5/1.0`及原生前/背景深度、离焦响应，没有计算VEPI选参数；另两档仅保留诊断，0.5结果可复用。官方DepthNet自身逐帧minmax，实际最大值接近1；不使用VDA span或把K/32当作有证据的映射。其他方法DAVIS预检完成后，再开始50段正式评价。
- **VEPI入口**：主列`VEPI-public-mp4`，按[demo](https://github.com/vivoCameraResearch/any-to-bokeh/blob/main/test/inference_demo.py)采用imageio/libx264、quality8、20fps，显式固定yuv420p、macro_block_size16、2个编码线程及imageio-ffmpeg0.4.7。OpenCV解码25帧后调用[原edge_batch.py](https://github.com/vivoCameraResearch/any-to-bokeh/blob/main/metrics/edge_batch.py)：BGR灰度、Sobel3幅值、参考RGB/mask线性resize、mask矩形5×5膨胀后>127、ROI Pearson、帧均值再视频等权均值。另报`VEPI-png`和MP4−PNG的codec delta作诊断，主列不因数值不同更换；不采用论文文字版Laplacian另选高分。
- **解释边界**：AIF原样输出可以在PNG自比达到VEPI=1，故VEPI只衡量前景边缘保留，不能单独决定虚化质量。Table2保留固定场景/关键帧与连续视频定性复核：背景失焦、焦点泄漏、halo/边界和时序闪烁；本轮不扩展BER等新指标。

---

## 4.5 DAVIS五方法预检与正式复用

用户确认顺序为Any-to-Bokeh→MPIB→BokehMe→Dr.Bokeh→BokehDiff，每种固定复用上述8段的前25帧及0/6/12/18/24关键帧。配置`configs/davis-baselines.json`，入口`scripts/run_davis_baseline.py`；不重做Block A，不计算VEPI选参数。共同检查200帧完整、1024×576、float finite、非空PNG、RGB/帧序、focus/K16、GT mask权限、冻结adapter及40组视觉实现检查；效果不佳不构成调参理由。

Any-to-Bokeh直接用原JPEG和原分辨率q进入作者WholeVid loader，CoC/MPI先在原分辨率构造，再走作者resize；对实际RGB/CoC/MPI张量和文件名focus逐帧核对。25帧分为起点0/4/8/12/16/20的六组，最后一组5帧加3个零padding；用作者实际decode_latents方法加identity VAE检查索引拼接，三次真实调用输出8+8+9帧，对应0–24连续顺序。

其他四种逐帧方法沿用冻结模型/控制：原JPEG以OpenCV线性resize到1024×576再转RGB float32；q/255以float64线性resize、不再次量化，focus保持源分辨率五位小数值。MPIB/BokehMe网络输入转换为float32；Dr.Bokeh/BokehDiff控制映射保留已有float64→网络float32规则。BokehMe/BokehDiff记录resize后每帧span和K/span，保持K16、不补偿；Dr.Bokeh走Full-public LDF+LaMa，逐帧记录LDF和实际repair/layer结果，GT mask不作为alpha。分割失败、细节重绘、局部强虚化等保留，除客观实现错误外不改协议。

**连续随机流与复用**：DAVIS处理顺序在输出前固定为这8段在前、其余42按名称排序；评价集合和视频等权平均不变。BokehDiff在加载前仅一次全局seed42，保留公开每次scheduler generator，PISA随机流不逐帧/逐段重置。Any-to-Bokeh保留项目既有全局seed0与公开每次CPU generator0；VAE latent sampling还会消耗全局随机流，也保存并恢复CPU/CUDA状态。两者都在每段结束保存RNG checkpoint，正式42段接续第8段状态，因此8段预检可作为同一正式流的前缀复用；不套用Synthetic历史独立candidate/preflight流的复用结论。

五种预检及视觉复核全部通过后，再执行六方法×50×25。DeepLens固定native-default0.5，复用已有200帧；其余五种同样只新增42段。全部方法完整输出后统一编码H.264并计算MP4主VEPI与PNG codec诊断；正式并发吞吐不填写Efficiency表。

# 5. Benchmark 3：OPPO-style Low-FPS Coarse Depth

## 5.1 定位

这是后续最能体现本项目核心创新的专项 Benchmark。

输入模拟：

\[
High\text{-}FPS\ RGB
+
Low\text{-}FPS\ Coarse\ Depth
\]

如果基于有完整 GT 的 synthetic paired dataset 构造，可以同时拥有：

- Full-FPS Bokeh GT；
- Full-FPS Depth GT；
- 可选 Alpha GT。

Any-to-Bokeh-style Synthetic 数据可以作为构造该 Benchmark 的重要基础。

---

## 5.2 Final Bokeh 指标

尽量与 Any-to-Bokeh Synthetic 保持一致：

\[
\boxed{
PSNR,\ SSIM,\ LPIPS,\ FVD,\ FD,\ RM
}
\]

---

## 5.3 Depth 指标

如果有 Full-FPS Depth GT：

| Metric | 趋势 | 主要含义 |
|---|---:|---|
| AbsRel | ↓ | 相对深度误差 |
| RMSE | ↓ | 均方根深度误差 |
| \(\delta_1\) | ↑ | 阈值内的深度像素比例 |
| Depth Edge Error | ↓ | 深度突变边缘恢复 |

---

## 5.4 Alpha 指标

如果有 Alpha GT：

| Metric | 趋势 | 主要含义 |
|---|---:|---|
| SAD | ↓ | Alpha absolute difference |
| MSE | ↓ | Alpha 均方误差 |
| Gradient Error | ↓ | 发丝 / 软边缘恢复 |
| Connectivity Error | ↓ | 前景连通与拓扑结构 |

Depth 和 Alpha 指标单独建表，不与最终 Bokeh 主表混在一起。

---

## 5.5 Low-FPS / Noise 专项变量

Depth FPS 建议至少测试：

\[
30,\ 15,\ 10,\ 5,\ 2\ fps
\]

Depth Quality 可分：

- Clean；
- Mild；
- Medium；
- Severe。

候选退化：

- smooth / Gaussian perturbation；
- local holes；
- boundary corruption；
- dilation / erosion；
- temporal missing frames；
- confidence degradation。

最终退化模型应尽量依据真实 OPPO Stereo Depth 的统计特征，而不是只使用任意人工噪声。

---

# 6. 当前核心 Baseline

| Method | 代表路线 | 原生输入类型 | 原生视频模型？ |
|---|---|---|:---:|
| DeepLens | 单帧学习式 Neural Bokeh | RGB + Depth + Focus/Aperture | ❌ |
| Dr.Bokeh | Physical RGBD / Occlusion-aware Rendering | RGB + Depth + Lens/Focus Parameters | ❌ |
| BokehMe | Physical + Neural Hybrid | RGB + Disparity + Focus Disparity + \(K\) | ❌ |
| MPIB | MPI / Explicit Layered Geometry | RGB + Disparity + Focus Disparity + \(K\) | ❌ |
| BokehDiff | One-step Diffusion / Generative Bokeh | RGB + Geometry / Defocus Control | ❌ |
| Any-to-Bokeh | Video + Geometry + Temporal Modeling | RGB Video + Per-frame Geometry + Focus + \(K\) | ✅ |
| BokehCrafter* | Video Diffusion | AiF Video + Reference Bokeh Image + Text | ✅ |
| Ours | Low-FPS Depth-guided Video + Hybrid Rendering | High-FPS RGB + Low-FPS Coarse Depth + Mask/Confidence + Camera/Focus | ✅ |

\* BokehCrafter 作为方法 baseline 可以保留；是否进入正式 quantitative 主表，取决于能否获得可运行实现 / checkpoint，并在当前 Benchmark 上公平运行。它不再依赖 VBS 才能保留。

---

# 7. 为什么选这些 Baseline

## DeepLens

代表较早期学习式单帧 Bokeh，主要观察单帧质量与逐帧处理后的视频稳定性。

## Dr.Bokeh

代表强物理 RGBD Renderer 和显式遮挡处理，重点比较 Depth Boundary、Partial Occlusion、Color Bleeding 和物理可控性。

## BokehMe

代表：

\[
Physical + Neural
\]

与本项目：

\[
Physical\ Renderer + Neural\ Residual
\]

的整体思想较接近。

## MPIB

代表 Explicit Layered Geometry / MPI，适合比较 layered geometry、occlusion 和前后景关系。

## BokehDiff

代表现代 One-step Diffusion / Generative Bokeh，重点观察感知质量、复杂边缘与生成式先验的收益和成本。

## Any-to-Bokeh

当前最重要的直接 baseline，代表：

\[
Video + Explicit\ Geometry + Temporal\ Modeling
\]

重点比较：

- Temporal Consistency；
- Geometry Condition；
- Dynamic Focus；
- Geometry Noise Robustness；
- 与 Low-FPS Geometry 输入条件的差异。

当前 Any-to-Bokeh 已在 96 GB GPU 上完整跑通。

## BokehCrafter

代表 Video Diffusion / Temporal Prior。若后续能获得稳定可运行实现，可作为额外 video baseline；否则不阻塞主实验。

---

# 8. 单图方法 vs 原生视频方法

当前核心 baseline 可以分为：

\[
5\ 个单图/逐帧方法
+
2\ 个原生 Video 方法
+
Ours
\]

单图方法用于视频时：

~~~text
Frame t-1 → Model → Output t-1
Frame t   → Model → Output t
Frame t+1 → Model → Output t+1
~~~

没有显式跨帧信息交互。

原生视频方法：

~~~text
Frame t-1
Frame t
Frame t+1
   ↓
Temporal Module / Temporal Attention
   ↓
Video Output
~~~

因此将两类方法同时放进 Video Benchmark，可以直接观察 Temporal Modeling 的作用。

---

# 9. 统一输入协议：Track A / Track B

为了把 Geometry 输入质量差异与 Bokeh / Video 模型能力差异尽可能分开，保留双 Track。

## Track A：Unified Per-frame Geometry

### 输入

对于需要 Geometry 的方法，尽可能提供统一质量、统一帧率的：

- Per-frame Depth / Disparity；
- Focus Plane；
- Blur Strength / \(K\)。

~~~text
RGB Video
+
Per-frame Geometry
+
Focus / Blur Control
        ↓
各方法
        ↓
Bokeh Video
~~~

### 回答的问题

> 当 Geometry 条件已经充分提供时，谁的 Bokeh Rendering / Video Modeling 本身更强？

重点比较：

- Renderer；
- Neural Prior；
- Occlusion Handling；
- Boundary Quality；
- Temporal Modeling；
- Video Quality。

---

## Track B：Low-FPS Coarse Geometry

### 输入

\[
High\text{-}FPS\ RGB
+
Low\text{-}FPS\ Coarse\ Depth
\]

对于原本必须使用每帧 Geometry 的 baseline：

~~~text
Low-FPS Coarse Depth
        ↓
统一 Depth Interpolation / Propagation
        ↓
Per-frame Geometry
        ↓
Baseline
~~~

Ours：

~~~text
High-FPS RGB
+
Low-FPS Coarse Depth
+
Mask / Confidence
        ↓
Learned Temporal Geometry Propagation
        ↓
Depth / Alpha / Bokeh
~~~

### 回答的问题

> 在 Low-FPS、粗糙、可能存在误差的 Geometry 条件下，谁更稳定、更准确？

这条 Track 直接对应本项目的核心创新。

---

# 10. 哪些 Benchmark 做双 Track

| Benchmark | Track A | Track B | 说明 |
|---|:---:|:---:|---|
| Any-to-Bokeh Synthetic / Reproduced Set | ✅ | ✅ | 最适合做双 Track |
| DAVIS | ❌ 不做严格 Track A | ✅ 真实条件泛化为主 | 无 paired Bokeh GT / Full Geometry GT |
| OPPO-style | 可作为 Upper-bound | ✅ 核心 | 本身就是 Track B 专项 |

Track A / Track B 不意味着所有 Benchmark 都机械重复两遍。

---

# 11. 最终结果表模板

## Table 0：GT Renderer Comparison（历史开发验证）

这张表记录 E1–E6 阶段对两种 Bokeh GT 生成方案的开发验证。**它不再代表当前 Baseline Benchmark Sweep 的 GT 选择。** 当前 200-video 公开代理复现与后续 baseline sweep 固定使用 **sRGB + MPIB-public proxy**；linear RGB + Cycles 分支暂停并保留为独立物理参考。

1. **Dr.Bokeh Renderer**：作为快速、易接入的物理渲染方案；
2. **Blender Cycles**：作为独立于参赛 baseline 的物理光线追踪方案。

比较时必须使用完全相同的：

- RGB / AiF；
- Disparity / Depth；
- Alpha；
- Focus；
- Blur Strength \(K\)；
- Scene；
- Frame sequence。

Any-to-Bokeh 对同一批输入只产生一份 prediction，再分别以两种 renderer 产生的结果作为 GT 进行评价。

| GT Renderer | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | FD ↓ | RM ↓ | Rendering Time ↓ | Visual Quality | Fairness |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Dr.Bokeh Renderer | 33.0620 | 0.97600 | 0.02769 | — | 0.02281 | 0.003588 | 3.24 s/帧* | 图层渲染通过，保留交叉检查 | 与参赛 baseline 同源 |
| Blender Cycles（OptiX，1024 spp） | 32.8592 | 0.97510 | 0.03120 | — | 0.02199 | 0.003942 | 2.43 s/帧* | 对齐、标定、采样收敛检查通过 | E1–E6 独立物理参考；当前暂停 |

**2026-09-26 首轮实测：5 个开发场景 × 25 帧，1024×576，K=16。** 这些数字衡量同一份 Any-to-Bokeh 预测与两种候选参考的差异，不是 renderer 本身的准确率，也不等于官方测试集成绩。逐场景结果、指标定义、输入来源和检查证据见 [小型结果记录](records/renderer-pilot-20260926.json)。

- **历史阶段结论**：E1–E6 曾选择 Blender Cycles 4.5.13 LTS / OptiX / 1024 spp 作为独立物理参考，Dr.Bokeh 作为交叉检查。该结论已被后续 Any-to-Bokeh 公开代理复现的实证结果覆盖：当前 baseline sweep 固定使用 MPIB-public/sRGB reference，Cycles 分支暂停但不删除。
- 已知 RGBA 图层绕过学习式深度/分割/修复，采用线性光照合成与 sRGB 输出。关闭景深对齐 PSNR 60.57 dB；参考深度组合为对焦 2 m、背景 6 m，K=16，50 mm 焦距 / 36 mm 传感器宽度，对应 f/2.10054。光圈由明暗边界扩散标定，半径差约 0.1%；此映射尚不外推到所有深度或 K。
- Cycles 同帧独立随机种子差异 PSNR 57.16 dB、MAE 0.000698；1024 与 4096 spp 差异 PSNR 59.21 dB。低采样初版和 CPU 回退初版均不作为本表最终 Cycles 结果。
- LPIPS 使用 squeeze v0.1；FD 使用 torchvision RAFT-large C_T_SKHT_V2 的完整分辨率光流差；RM 对全部 24 对相邻帧计算变化差。5 段视频不足以稳定比较视频分布，本阶段 **不报告 FVD**，不以替代指标冒名填入。
- *耗时为试验成本：Dr.Bokeh 包含核心渲染/传输、不含 PNG 写盘，部分帧与校验任务交错；Cycles 包含渲染与 PNG 写盘。二者口径不同，不能直接作为统一速度排名或 Table 3 成绩。Cycles 的 125 帧总计约 303 秒，当前生成成本可接受。
- 素材为 AM-2k 训练前景及 BG-20k 背景，运动为构造的平面仿射运动。该 5-scene 试验只保留为 renderer 开发证据。后续 E6 24-scene 验证与 200-video MPIB-public proxy 复现均已完成，不再以本表决定当前 sweep 的 renderer。

除数值指标外，还固定观察：

- Hair / Thin Structure；
- Foreground / Background Occlusion；
- Halo；
- Color Bleeding；
- Strong Bokeh realism；
- temporal stability；
- renderer speed；
- reproducibility；
- 是否与参赛 baseline 存在天然偏置。

需要特别注意：

> 与 Any-to-Bokeh 论文报告指标更接近，只能作为“数据分布可能更接近原论文”的参考，不能单独作为最终选择 renderer 的依据。

最终 GT Renderer 需要综合考虑：

1. 与 Any-to-Bokeh 论文指标量级的接近程度；
2. 视觉真实性；
3. 发丝和遮挡边界质量；
4. 时序稳定性；
5. 生成成本；
6. 可复现性；
7. Benchmark 公平性。

原则上：

\[
\boxed{GT\ Generator \neq Compared\ Baseline}
\]

因此本表保留用于解释早期 renderer 选型过程；**当前不再从 Dr.Bokeh / Cycles 两者中重新选主 GT**。Baseline Benchmark Sweep 统一沿用已经完成 200-video 定量验证的 MPIB-public/sRGB reference。


## Table 1：Any-to-Bokeh Synthetic / Reproduced Synthetic

建议分别报告 Track A 与 Track B。

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | FD ↓ | RM ↓ |
|---|---:|---:|---:|---:|---:|---:|
| DeepLens（单级SR，aperture饱和） | **28.1669** | **0.943710** | **0.06333** | **40.12120** | **0.58263** | **0.012364** |
| Dr.Bokeh（GT Alpha reconstruction adapter） | **31.5885** | **0.973475** | **0.02938** | **25.85901** | **0.47862** | **0.008393** |
| BokehMe（Normalized，γ=2.2） | **33.3340** | **0.979418** | **0.02470** | **18.54637** | **0.53953** | **0.007394** |
| MPIB（γ=2.2冻结） | **31.6195** | **0.974408** | **0.02835** | **24.95694** | **0.44985** | **0.008184** |
| BokehDiff-B（Normalized，主表） | **27.0627** | **0.843751** | **0.13202** | **65.99693** | **0.80225** | **0.021633** |
| Any-to-Bokeh | **39.7116** | **0.986305** | **0.01515** | **8.93752** | **0.44496** | **0.005645** |
| BokehCrafter* |  |  |  |  |  |  |
| Ours |  |  |  |  |  |  |

注：Dr.Bokeh 主行读取 GT union alpha，其余方法只接收 RGB + PNG8 disparity + focus + K，信息权限不同；不读取 GT alpha 的公开完整 pipeline（LDF+LaMa）为 PSNR 29.1077 / FVD 143.28969，见 Table V。

若使用自建版本，表题必须明确写：

> Reproduced Any-to-Bokeh-style Synthetic Benchmark

不能与原论文官方 200-video test set 的 paper-reported numbers 直接混为同一测试集结果。

### Table V：Benchmark Validation — 与论文 Table 1 的性能版图对照

这张表是**辅助验证表**，不替代 Table 1A / 1B。目的不是要求每个方法逐项复刻论文绝对值，而是检查当前固定 200-video MPIB-public benchmark 是否能够恢复论文中的**总体量级、相对排序和方法间性能间隔**。随着 baseline sweep 完成逐行填写。

| Method | Paper PSNR ↑ | Our Bench PSNR ↑ | ΔPSNR | Paper SSIM ↑ | Our Bench SSIM ↑ | Paper FVD ↓ | Our Bench FVD ↓ | Sweep Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| DeepLens（单级SR，饱和） | 24.574 | **28.1669** | **+3.5929 dB** | 0.819 | **0.943710** | 125.338 | **40.12120** | ✅ 200段及复核完成；K16/32饱和全部披露 |
| BokehDiff-A：Shared-Geometry | 27.525 | **27.1560** | **-0.3690 dB** | 0.834 | **0.841752** | 65.678 | **67.92665** | ✅ 200段及视觉复核完成；作为protocol-recovery alternative保留 |
| BokehDiff-B：Normalized | 27.525 | **27.0627** | **-0.4623 dB** | 0.834 | **0.843751** | 65.678 | **65.99693** | ✅ 200段及视觉复核完成；选为主表展示配置 |
| BokehMe（Normalized，γ=2.2） | 27.992 | **33.3340** | **+5.3420 dB** | 0.936 | **0.979418** | 39.102 | **18.54637** | ✅ 双200段完成；与MPIB排序差异保留 |
| Dr.Bokeh（GT union alpha+LaMa，Track A reproduction主表） | 31.273 | **31.5885** | **+0.3155 dB** | 0.950 | **0.973475** | 32.710 | **25.85901** | ✅ 200段完成；选为ATB reproduction reported configuration，不声称恢复私有adapter |
| Dr.Bokeh（Full-public：LDF+LaMa，诊断/未来公平benchmark候选） | 31.273 | **29.1077** | **−2.1653 dB** | 0.950 | **0.963207** | 32.710 | **143.28969** | ✅ 完成；公开完整pipeline保留，不作为当前复现主行 |
| Dr.Bokeh（Known 4FG+1BG，诊断） | 31.273 | **32.2599** | **+0.9869 dB** | 0.950 | **0.978147** | 32.710 | **21.10796** | ✅ 同200段输入协议实验完成；额外隐藏层信息，不进入主表 |
| MPIB（γ=2.2冻结） | 31.390 | **31.6195** | **+0.2295 dB** | 0.950 | **0.974408** | 35.766 | **24.95694** | ✅ 200段完成 |
| **Any-to-Bokeh** | **38.899** | **39.7116** | **+0.8126 dB** | **0.974** | **0.986305** | **9.005** | **8.93752** | ✅ 完成 |

**论文版本补充（2026-09-29）**：[ICLR2026正式版Table1](https://proceedings.iclr.cc/paper_files/paper/2026/file/a0dca56ad0aceeb4eab4d35630e9a0ed-Paper-Conference.pdf#page=7)另有LPIPS列，DeepLens/BokehDiff/BokehMe/Dr.Bokeh/MPIB/Any-to-Bokeh分别为0.183/0.127/0.060/0.046/0.040/0.019；arXiv v2未列出。上表原六项数值一致。补充引用不追溯修改已完成候选选择，不把项目LPIPS实现默称为作者入口逐值复现。DeepLens精确视差/自带深度两处叙述来自v1不同数据域，公开材料仍未披露完整私有adapter。按原始几何保持CoC时，minmax后的DeepLens控制需包含span补偿`alpha_req=K*span/c`；该映射后续已冻结为单级SR + `aperture=min(K*span/c,1)` 饱和（c=12.488），K16/32 全部饱和（2725/5000帧）并已披露。详见现有汇报8.6–8.7及`records/deeplens-protocol-source-review-20260929.json`。

**当前Track A阶段边界与冻结规则**：本表当前首先服务于 **Any-to-Bokeh Synthetic/Table 1 reproduction**，不是最终统一信息权限的独立benchmark。Dr.Bokeh因此采用完整200段protocol-recovery后选定的GT Alpha reconstruction adapter作为reported主行；Full-public公开完整pipeline保留为诊断证据，并计划在后续自建公平benchmark时重新优先评估。该选择只表示“当前复现展示配置”，不证明ATB私有实验使用GT alpha。

在Ours正式评价前，本表中DeepLens、BokehDiff、BokehMe、Dr.Bokeh、MPIB与Any-to-Bokeh的reported配置全部冻结。**Ours出现结果后不得为了扩大/缩小差距回调baseline adapter或benchmark构造。** 后续研究顺序为：先在当前ATB-style reproduction benchmark上比较Ours与Any-to-Bokeh；再设计独立的本项目benchmark，统一RGB/低帧率粗深度/focus/control的信息权限和退化协议，并重新确定适用于该公平设置的baseline adapter。两阶段结果分表、分名报告。

Validation 重点依次观察：

1. **绝对量级**：每种方法是否落在与论文可解释的性能区间；
2. **相对排序**：baseline performance landscape 是否总体保持；
3. **方法间 gap**：例如 Any-to-Bokeh 与 MPIB / Dr.Bokeh 的优势间隔是否仍然显著；
4. **难度趋势**：统一保留按 (K=4/8/16/32) 的分组结果，观察强虚化条件下各方法的退化趋势。

只修复客观实现错误；**不根据与论文数字的接近程度修改 scene、GT、Focus、K、素材或评价参数。**



### MPIB adapter recovery：双候选完成，gamma=2.2已冻结

固定200段public-proxy开发集，完整运行MPIB官方模型，仅比较gamma=1.0与2.2。两个值分别有当前GT积分工作空间与MPIB官方demo默认值的独立依据，均属于原论文训练支持范围。输入、K、focus、checkpoint、planes、dilation及所有指标保持不变。

本轮允许根据与论文MPIB行的整体指标一致性选择候选，并同时保留两套结果和逐项偏差；该授权是对“不得按论文分数选择设置”的限范围更新，不授权改场景/GT或连续gamma搜索。不按本数据集最高分选，不按Any-to-Bokeh行选。没有预设的跨单位加权loss；若各指标证据相互冲突，保留未决，不强行冻结。论文未发布by-K分组值，不能声称这些分组也被精确复现。

该选择属于reproduction protocol reconstruction的经验依据，不证明找回真实作者配置，也不把当前200段升级为held-out公平测试。冻结于Ours之前应如实记录，但不等于没有选择偏差。未来独立测试和不同工作空间的数据仍需声明适配条件。最终确定后，Table 1A放明确命名的冻结版本，Table V及实验记录保留双候选证据；未确定时不提前填写最终MPIB行。

**2026-09-28最终结果**：两个候选各200段全部完成。gamma=2.2在PSNR、SSIM、FD、RM、FVD、VFID-I六项上均比gamma=1更接近论文MPIB行，因此按已授权的复现选择原则冻结 **gamma=2.2**，配置为`configs/mpib-track-a-frozen.json`。这是复现配置选择，不是本数据集最高重建质量选择。

| 指标 | 论文MPIB | gamma=1 | gamma=2.2（冻结） | gamma=1差值 | gamma=2.2差值 |
|---|---:|---:|---:|---:|---:|
| PSNR ↑ / dB | 31.390 | 35.972438 | **31.619476** | +4.582438 | +0.229476 |
| SSIM ↑ | 0.950 | 0.984682 | **0.974408** | +0.034682 | +0.024408 |
| FD ↓ | 0.481 | 0.359335 | **0.449849** | -0.121665 | -0.031151 |
| RM ↓（24 transitions） | 0.011 | 0.005572 | **0.008184** | -0.005428 | -0.002816 |
| FVD ↓ / VideoMAE | 35.766 | 19.708852 | **24.956944** | -16.057148 | -10.809056 |
| VFID-I ↓ / I3D | 5.444 | 1.645553 | **3.208706** | -3.798447 | -2.235294 |

LPIPS（非论文选择项）：γ=1为0.016910，γ=2.2为0.028355。RM主列使用完整24 transitions；作者公开2-transition诊断分别为0.005603/0.008238，另行保留。FVD/VFID-I采用此前同一评价适配及相同GT特征缓存。

结果与by-K、检查及失败案例见`records/mpib-adapter-recovery-200-20260928.json`和现有复现汇报8.1。FVD/VFID-I仍未精确匹配论文，gamma=1结果完整保留。此MPIB行参与了候选选择，因此Table V在这里属于复现校准证据，不是独立held-out验证。后续不依据Ours成绩再调整该配置。
### BokehMe adapter recovery：双候选完成，Normalized已冻结

Any-to-Bokeh 公开材料确认 BokehMe 是依赖输入 geometry 的逐帧 image baseline，并在 noisy-depth / DAVIS 实验中与其他方法共享 geometry/control 信息；**但没有公开 Synthetic Table 1 的 BokehMe inference adapter**。因此当前不能直接断言其 disparity normalization、focus/K 映射、gamma、highlight 或 preprocessing 细节。

本轮只比较两个预定义的 **signed defocus convention**：

1. **Shared-Defocus**：直接使用当前 PNG8 disparity / 255、文件名 focus 和原 K，保持 `K(d-d_f)`；
2. **BokehMe-Normalized**：按官方 demo 对 disparity 做 per-frame min-max，同时对 focus 做同一变换，K 首轮不缩放。

共同固定：

- 完整 classical renderer + ARNet + IUNet + error-map fusion；
- gamma=2.2；
- highlight=false；
- defocus_scale=10；
- circular aperture；
- 官方 checkpoint；
- 逐帧独立推理；
- 输出 clamp [0,1] 后 PNG8。

公开依据补充：作者[BLB dataset说明](https://github.com/JuewenPeng/BokehMe#blb-dataset)指出物理视差无需归一化；A与这一口径一致，B仅代表demo归一化候选。B中focus使用同一仿射变换且不夹紧，K不缩放，因此`S_B=S_A/(d_max-d_min)`，两者并非纯坐标等价。只有同步取`K′=K(d_max-d_min)`才与A等价，本轮不把该等价写法作为第三候选。实际pipeline接收`S/10`，内部再乘`defocus_scale=10`，避免重复放大。

候选配置`configs/bokehme-adapter-recovery.json`保持原样。2026-09-28固定8段检查、双200段及全部指标均完成；B在六项与论文行的绝对偏差上均小于A，因此按原规则冻结B，执行配置为`configs/bokehme-track-a-frozen.json`。预检查记录见`records/bokehme-adapter-preflight-20260928.json`。

执行顺序为固定8段 preflight 后双候选200段。外部参考为论文 BokehMe 行：PSNR27.992、SSIM0.936、FD0.536、RM0.013、FVD39.102、VFID-I8.633。若两个预定义候选中有一个在整体 metric signature 上一致更接近论文，则冻结为 reproduction adapter；若证据冲突，保留未决，不扩展为无约束参数搜索。

Table 1A已填唯一冻结B行，Table V保留差值。完整双候选指标、by-K及视觉诊断见`records/bokehme-adapter-recovery-200-20260928.json`和现有复现汇报8.2。后续使用冻结版本时须声明新输入域的转换条件，不依据Ours结果调整。

| 指标 | 论文BokehMe | A：Shared-Defocus | B：Normalized（冻结） | A−论文 | B−论文 |
|---|---:|---:|---:|---:|---:|
| PSNR ↑ / dB | 27.992 | 33.842835 | **33.334003** | +5.850835 | +5.342003 |
| SSIM ↑ | 0.936 | 0.983270 | **0.979418** | +0.047270 | +0.043418 |
| FD ↓ | 0.536 | 0.415701 | **0.539533** | -0.120299 | +0.003533 |
| RM ↓（24 transitions） | 0.013 | 0.006061 | **0.007394** | -0.006939 | -0.005606 |
| FVD ↓ / VideoMAE | 39.102 | 15.366611 | **18.546370** | -23.735389 | -20.555630 |
| VFID-I ↓ / I3D | 8.633 | 2.321887 | **2.814896** | -6.311113 | -5.818104 |

B的LPIPS=0.024698，公开前2对RM=0.007504；主表RM采用24对。B虽逐项比A更接近论文，但PSNR仍高5.3420 dB，FVD低20.5556；BokehMe相对冻结MPIB的PSNR/SSIM/RM/FVD/VFID-I排序与论文相反，FD排序一致。故Table V仍显示未恢复的性能版图；这些差距是观察结果，既不作为实现通过的硬门槛，也不通过继续调参强行消除。当前集合参与候选选择，不能称为独立held-out验证。



### Dr.Bokeh adapter：单一协议200段完成，论文差异保留

**2026-09-28完整8段预检及单一协议200段、全部指标已完成。** 原规范文件`configs/drbokeh-track-a-preflight.json`；本轮执行配置`configs/drbokeh-track-a.json`；结果记录`records/drbokeh-adapter-specification-20260928.json`。BokehMe已关账，不通过回调其参数强行恢复论文方法排序。

**输入证据和权限**：

- [Any-to-Bokeh正文§4.2及附录](https://arxiv.org/html/2505.21593v2)没有披露Dr.Bokeh是否获得GT alpha、oracle layers、LDF/LaMa设置及K/focal传参。DAVIS masks作为VEPI评价ROI，不能据此推断它们也是baseline推理输入。已核对当前公开main、远程refs及公开issues；[Issue #3作者回复](https://github.com/vivoCameraResearch/any-to-bokeh/issues/3#issuecomment-3712982533)曾说明评价代码/数据尚待审批。结论限定为已查材料未披露，不是证明私有实验没有oracle。
- [Dr.Bokeh CVPR 2024正文§5.2](https://openaccess.thecvf.com/content/CVPR2024/papers/Sheng_Dr._Bokeh_DiffeRentiable_Occlusion-aware_Bokeh_Rendering_CVPR_2024_paper.pdf#page=6)自身合成定量实验，给所有方法相同GT depth，为MPIB/Dr.Bokeh把预测遮挡像素替换为GT，并搜索blur kernel后取最佳结果。不能把这一100-scene、rendering-step评价协议移植成Any-to-Bokeh的200-video协议。
- 当前采用讨论中的A：只给RGB、PNG8 disparity/255、focus和K；使用官方LDF预测alpha及官方inpainting。B（额外给benchmark alpha）不进入本轮主表/预检，也不默认为第二个全量候选。单独给alpha仍不能提供所有隐藏图层颜色，故那也不等于纯renderer真值测试。
- Any-to-Bokeh对固定深度层的概括不能替代具体代码：Dr.Bokeh官方前/背景RGBAD保留逐像素视差、允许层内深度变化，不人为将每层压成一个固定深度值。

**固定的完整处理链**：RGB→[LDF](https://github.com/ShengCN/DrBokeh-Src/blob/24d96e4fb76a698494e86835c84a7c45ae2ebc10/app/Render/salient.py)→[RGBAD2layers](https://github.com/ShengCN/DrBokeh-Src/blob/24d96e4fb76a698494e86835c84a7c45ae2ebc10/app/Render/image_preprocessing.py)→Multi_Layer_Renderer。前景深度用OpenCV Telea inpainting；背景RGB和深度都用LaMa。统一输入depth不等于内部修复后的depth逐像素不变，需要记录差异。提供视差时不重新预测MiDaS；原模块导入时仍会载入它，但不调用其预测。固定gamma=2.2、highlight=false、offset=0、官方threshold=0.1/fg_erode=5/fg_iters=2/inpaint_kernel=7、圆形镜头和CUDA遮挡分支。所有帧独立运行，不读GT alpha、分层资产或隐藏背景，不新增时序平滑。

**输入坐标与名义CoC**：官方[Inference.py](https://github.com/ShengCN/DrBokeh-Src/blob/24d96e4fb76a698494e86835c84a7c45ae2ebc10/app/Render/Inference.py#L235)会归一化视差，却不会替调用者变换CLI focal。令

\[
 s=d_{\max}-d_{\min},\quad
 d'=\frac{d-d_{\min}}s,\quad
 d_f'=\frac{d_f-d_{\min}}s,\quad
 K_{Dr}=K_{manifest}\,s.
\]

则分层预处理前

\[
 K_{Dr}(d'-d_f')=K_{manifest}(d-d_f).
\]

这是保持输入名义pixel CoC的换算，没有从论文分数反推参数。`s=0`时停止并记录，不静默取任意范围；focus不夹紧，保留连续scalar对PNG8量化范围的轻微越界。K不补偿将使CoC放大`1/s`倍，不与当前设置混称等价。补偿后也不保证整个pipeline坐标等价：LaMa/uint16深度修复和occlusion的固定视差阈值均不具仿射不变性。

**真实CUDA核与lens覆盖**：[scatter_cuda_kernel.cu](https://github.com/ShengCN/DrBokeh-Src/blob/24d96e4fb76a698494e86835c84a7c45ae2ebc10/app/cuda-src/scatter_cuda_kernel.cu#L250)使用`scatter_dis=K_Dr*abs(relative_disp)+1`、softstep及面积归一化；[distance_kernel/lens_shape_mask](https://github.com/ShengCN/DrBokeh-Src/blob/24d96e4fb76a698494e86835c84a7c45ae2ebc10/app/Render/DScatter/CPU_scatter.py#L30)的距离单位就是pixel。没有额外全局2倍或10倍因子，`+1`也不能简单当成最终有效硬圆盘半径。另一个`floor(scatter_dis/2)`用于层间alpha可见性邻域，不是颜色散射半径。

lens是窗口边长；41/71/101的圆形mask分别为`ρ<20/35/50`。softstep存在非零尾部，不能声称数学上绝对无截断。预先采用工程容差：71对101的单位脉冲L1差及被截径向权重质量均≤1e-4。这不是文献规定或论文指标门槛。

| 名义R / px | 71核的二阶矩等效圆盘半径 / px | 41遗漏径向权重 | 71遗漏径向权重 | 71对101的单位脉冲L1差 |
|---|---:|---:|---:|---:|
| 0 | 0.51352 | 8.32e-25 | 3.55e-44 | 0 |
| 1.33333 | 2.05323 | 6.09e-24 | 2.60e-43 | 0 |
| 16 | 16.28939 | 8.36e-7 | 3.56e-26 | 3.56e-26 |
| 32 | 32.26115 | 0.618699 | 6.35e-6 | 1.15e-5 |
| 32.1 | 32.36116 | 0.621057 | 8.51e-6 | 1.55e-5 |

上述是常深度不透明平面的原始线性renderer单像素脉冲测试；未执行LDF/LaMa、未用benchmark oracle层、未计算模型质量分数。等效半径由`√(2 E[ρ²])`给出，不声称为理想物理半径。原核在R=0仍有小残余扩散，因此不将“对焦输出逐像素等于输入”作为实现门槛。正负离焦测试一致。完整结果包含0/1.333/2/4/8/16/32/32.1 px。

固定8段原始PNG8输入共200帧的坐标换算已验证：float64最大误差7.11e-15，最大名义输入半径25.6 px。**这些不是200帧完整pipeline结果**。lens71可以进入预检，41不采用。预检还需逐帧检查实际修复后RGBAD层的disparity和CoC范围，并在固定K32场景对同一组预测图层比较71/101；近零归一化分母可能放大很小的尾权重，不能仅凭常平面结论保证全部场景。

**本轮预检范围**：运行原固定8段×25帧、1024×576；记录LDF mask及其覆盖、前/背景RGBAD、LaMa RGB/深度补全、实际CoC/lens覆盖、输出范围和有限值，检查0/6/12/18/24关键帧。分割差或对焦边缘差先作为baseline失败案例保留，不能按视觉/分数删场景或换成GT alpha。 本轮真实图层对照固定repro_000/001的全部25帧；同一保存RGBAD分别render71/101，检查整体MAE≤1e-6、边缘MAE≤1e-6、最大float差≤1e-4、PNG8最大差≤1，记录weight/visibility及小分母区域误差。若71未通过而101覆盖足够，则按实现覆盖修正选择101，重新从保存图层渲染预检结果，不重新分割、不看论文分数。固定8段检查通过后，用户已授权直接冻结单一adapter并跑200段，不扩展为alpha×normalization×gamma×lens组合搜索。届时按原指标补Table 1A/Table V，论文Dr.Bokeh六项外部参考依次为PSNR31.273、SSIM0.950、FD0.522、RM0.011、FVD32.710、VFID-I6.097，排序差异保留，不用于再调参。

**完整预检验收结果**：200帧输入映射、实际LDF/LaMa/layer调用、finite及完整性均通过；40组关键帧未见客观实现错误，原生分割/补全失败保留。修复后最大名义半径25.6 px；固定K32两段50帧同层71/101差异全部为0，冻结71。`configs/drbokeh-track-a-frozen.json`与`records/drbokeh-full-preflight-20260928.json`为执行/验收入口。

**实现加速**：只修改独立CUDA扩展occlusion-aware forward的线程块分配，由6块改为2304块，不改每像素算术、循环、模型或控制。对全部200帧相同保存图层，新旧float输出逐值一致；完整pipeline复查PNG一致。`records/drbokeh-cuda-launch-check-20260928.json`与`configs/drbokeh-grid2d-launch.patch`记录证据。200段复用8段原预检，其余使用等价加速扩展；并行计时不用于Efficiency主表。全量质量指标完成后再填Table 1A/V。


**200段终审**：5000帧完整、finite、全量半径覆盖通过，最大实际R31.87451 px，无空/全满LDF mask。FVD143.289688经同缓存特征独立SVD复算为143.289918；差0.000230，保留原公开计算。VFID-I5.760450；相对论文FVD32.710存在显著差异，未因此改协议或删数据。完整数值/by-K及困难场景见现有复现汇报8.3，机器证据`records/drbokeh-track-a-200-20260928.json`与`records/drbokeh-track-a-review-20260928.json`。Table 1A/V已填当前实测结果，不能称恢复了作者私有adapter或验证了独立held-out benchmark。

### Dr.Bokeh input-protocol recovery：同200段实验已收尾（2026-09-29）

用户已确认分阶段顺序：固定11段视觉终审 → GT-alpha补齐189段 → known-layer补齐189段 → 同200段统一评价；geometry/gamma/blur-strength只在这轮完整结果出来后另行决定。原full-public的200段与主表完整保留，不按11段分数替换。

55组（11×{0,6,12,18,24}）AIF/GT/LDF alpha/GT union alpha/三组prediction及对应FG depth repair、LaMa mask、BG depth/RGB已查看，未发现新的实现阻塞。明显漏检会让前景残留在背景RGB/depth中；GT alpha后仍有LaMa残影与光度/边缘差异。repro_000的PSNR改善、FD恶化反例保留。稀疏关键帧不能排除未采样的相邻帧闪烁，不能据11段直接宣称解释了完整200段FVD主因。证据`records/drbokeh-input-visual-review-20260929.json`。

执行配置`configs/drbokeh-input-recovery-200.json`。GT alpha是四个前景的union，只替换LDF，官方cv2/LaMa保留；known-layer直接输入四个FG+一个BG，同时改变隐藏信息和层结构，不是纯LaMa消融。复用11段原结果与25帧记录，剩余各4725帧独立写入`outputs/drbokeh-input-recovery-200-20260929/`。gamma=2.2、lens71、highlight=false、offset0、PNG8 disparity、Focus/K、scene/GT/metric不变；先GT-alpha、再known-layer，最后PSNR/SSIM/LPIPS/FD/RM/VFID-I/FVD与by-K。结果目标`records/drbokeh-input-recovery-200-20260929.json`。

**完整200段结果已收尾**：PSNR Full/Alpha/Known=29.107707/31.588505/32.259896，FVD=143.289689/25.859006/21.107962。Alpha相对Full的PSNR胜率99%、FD90.5%；K4/8/16/32的PSNR增益3.1561/3.4937/2.1775/1.3678dB，并未随K递增。Known相对Alpha仍有39段PSNR和44段FD退步，不是逐段上限。全部七项指标、配对/by-K和论文差值见现有复现汇报8.5。新增三个事后退化案例与原55组记录共同审查，未发现新增明显实现阻塞；15000张PNG和特征/控制检查通过。机器证据`records/drbokeh-input-recovery-200-20260929.json`、`records/drbokeh-input-effect-analysis-20260929.json`、`records/drbokeh-input-final-review-20260929.json`。

**当前决定**：本轮输入协议实验关账，不自动追加geometry/gamma/强度实验；残余差值不能证明需要哪一种修正。Table V并列三种输入权限，Table 1A原Full-public行不变。改善不证明Any-to-Bokeh私有adapter使用了oracle；Known共同改变多个因素，不能叫纯LaMa消融，也不能与无oracle方法不加说明地混成同一输入权限比较。

坐标关系明确为：令s=d_max−d_min，d′=(d−d_min)/s、df′=(df−d_min)/s，则K|d−df|=Ks|d′−df′|=R；若归一化后仍用K，则K|d′−df′|=R/s。后者改变散焦强度，不是纯坐标变换。当前只运行Ks补偿的固定版本；未来known-layer raw/归一化+Ks只先做小规模等价性检查，不立刻扩为全量候选。lens是覆盖窗口，不能拿它调虚化强度。

### DeepLens冻结协议：原生单级SR+aperture饱和（2026-09-29）

用户已确认：1024×576遵循官方interactive的scale2路径，512×288 LensBlur后一次guided SR，不为扩大blur范围改成两级SR。原生TF1.15 CPU、原checkpoint，外部PNG8 disparity与文件名focus；`s=max(d)-min(d)>0; D=(d-min)/s; Df=(df-min)/s; aperture=min(K*s/12.488,1)`。无DepthNet、oracle alpha/隐藏信息、焦点夹紧或aperture外推。c由独立control probe决定，与论文/benchmark分数无关。范围饱和作为方法约束披露，不删除K16/32场景、不反推作者私有adapter。

执行配置`configs/deeplens-track-a-frozen.json`；固定8段完整preflight核对输入与饱和、same-scene K16/32平台、有限输出和40关键帧，通过后直接200×25及全部指标/by-K。K4/8预期不饱和，K16/32在当前输入范围预计全部饱和，最终记录实际逐帧统计。原生SR uint8中间转换和最终PNG输出口径保留；CPU时间不直接对照论文GPU时间。

**预检实际通过**：200帧signed数值检查与40关键帧复核完成。K4/8控制等效半径约4.20–4.26/7.68–8.24px；固定同输入K16/32输出逐值一致，正确进入饱和平台。记录`records/deeplens-preflight-20260929.json`。随后按该单一协议完成原生CPU八路200段及全部指标，未按预检分数调整任何参数。

**200段与收尾均完成**：PSNR28.166875、SSIM0.943710、LPIPS0.063333、FD0.582632、RM0.012364、FVD40.121195、VFID-I6.962586。K4/8均不饱和，K16/32全部饱和（共54.5%帧），没有剔除任何档位。5000帧完整/finite/signed检查、GT特征缓存与分组均值核对通过；共56组关键帧/额外案例复核，虚化不足和亮度/边缘问题原样保留。Table1A/V已填实测值；实际数值更优不证明恢复私有adapter，尤其对BokehDiff的多数指标排序与论文不同，不据此调c或改数据。记录`records/deeplens-track-a-200-20260929.json`、`records/deeplens-track-a-final-review-20260929.json`；详细论文差值/by-K见现有复现汇报8.7。CPU生产耗时不填GPU效率列。

### BokehDiff adapter recovery：共享几何与min-max双候选

依据[原论文§3](https://arxiv.org/html/2507.18060v1#S3)、[inference_hf.py](https://github.com/FreeButUselessSoul/bokehdiff/blob/38ff9372ab146ab20468832a0424fa0b2888dffd/inference_hf.py)及[dataset.py](https://github.com/FreeButUselessSoul/bokehdiff/blob/38ff9372ab146ab20468832a0424fa0b2888dffd/dataset.py)，统一输入为RGB、PNG8/255 disparity、文件名focus、manifest K。mask在相关loader分支只参与focus估计；已知focus时不运行Depth-Anything/BiRefNet，不提供GT mask，不执行GT曝光校正或demo三焦点输出循环。完整RealVisXL V5.0 + 官方small LoRA/VAE + PISA单步推理，不能拿classical数据renderer代替baseline。

| 候选 | disparity / focus | K | 第二条件通道 |
|---|---|---|---|
| A Shared-Geometry | 原d、df | 不变 | abs(d−df)K/10 |
| B Official-style Normalized | (d−min)/span、(df−min)/span | 不变 | abs(d−df)K/(10span) |

B只表示借鉴公开推理的min-max适配；官方npy分支通常还有3×3形态学范围重估/clamp，不能写成完全复刻loader。span=0报错，df不夹紧；B同时改变PISA的occlusion坐标及CoC幅度，不是单纯坐标换算。限定A/B，不因不能恢复论文排序就自动加入第三候选或倍率搜索。

训练代码无逐图最终min-max，但`K_adapt=K×times/max(df,1-df)`生成GT，`K_strength=K×ratio×times`生成网络条件；故公开K/10支持输入约定，不证明我们的像素虚化半径或隐藏作者adapter等价。1024×576、upsample1、T499、guidance1、固定prompt、FP32/TF32-off、原TILE_SIZE64、PISA hard1e7/supersampling5/segment7不搜索。

**随机协议**：公开PISA的torch.rand及p=0.1、train=True dropout原样保留。每候选单进程，加载模型前全局seed42一次，manifest scene顺序、每段0..24顺序，不逐帧reset。公开log_validation中的独立scheduler generator每次seed42照旧；它不控制PISA全局随机流。A/B使用相同运行顺序。中断按已完成scene的CPU/CUDA RNG状态恢复，不能跳过已生成帧而漏掉随机数消耗。预检和正式200段分别从头开始，不复用随机流位置不同的PNG。旧seed0逐帧重置产物仅为容量证据。

**输出与验证**：直接捕获公开JPEG编码前postprocessed float，clamp/round保存PNG8。固定repro_003首帧，A/B分别与官方函数float逐值一致；相同随机状态复跑差0；42/43敏感性只记录、不选择seed。d=1会造成无效grid坐标，但实际probe的sampled输出、occlusion及RGBfinite，源列occlusion为0；保留公开行为并记录依赖当前runtime的数值边界，不据最终finite宣称所有几何正确，不做隐式clamp/公式修改。见`records/bokehdiff-stochastic-review-20260928.json`。

执行配置`configs/bokehdiff-adapter-recovery.json`，先A/B各8×25完整预检，固定0/6/12/18/24做视觉实现检查。通过后各200×25，统一PSNR/SSIM/LPIPS/FD/RM/FVD/VFID-I/by-K。选择看与[Any-to-Bokeh Table 1](https://arxiv.org/html/2505.21593v2#S4.T1) BokehDiff行的整体一致性；指标明显冲突则不强行冻结，不因某候选本地分数更好而选它。当前数据参与选择，是protocol reconstruction开发证据，不是独立benchmark验收；Ours未开始不消除这一选择偏差。

**2026-09-29执行状态**：A/B各200×25已完成，Table V并列填入实测值。完整结果`records/bokehdiff-adapter-recovery-200-20260928.json`；原8段加全量后repro_012/193/197共11段、55组/110张候选关键帧已复核，记录`records/bokehdiff-final-review-20260929.json`。193为两侧最低PSNR、012为最高FD、197为最大配对VideoMAE特征距离（不叫逐段FVD）。保留强虚化过度平滑、弱虚化不足及细节重绘等失败情况，没有新增实现阻塞。

A的PSNR/SSIM/FD/RM更接近论文，B的FVD/VFID-I更接近。完整200段与定性复核完成后，用户选择B（Normalized）作为主表展示配置：前四项A/B差距相对有限，而B的FVD=65.996926与论文65.678仅差0.318926，VFID-I也更接近。Table V继续并列A/B，A作为protocol-recovery alternative保留。该选择是复现展示决策，不等同于证明Any-to-Bokeh作者使用B；不新增C或K倍率搜索，本阶段BokehDiff不再模型计算。原预检随机流与控制验证证据继续保留。

### E7 AM-2k 前景颜色恢复：历史方案（当前停止）

> **当前决策已更新**：本项目当前 baseline sweep 不再推进 AM validation / foreground estimator 选择。AM-2k train/fg 直接作为固定 composition asset 使用，并明确不把它称为独立物理 foreground-color GT。下述三层验证协议与 60-source 开发实验保留为历史方法学证据。

当前改用三层证据后再决定是否冻结 foreground-color estimator：

1. **官方 FG 一致性评价（proxy consistency）**
   - 在 AM-2k train 上，以 `original + alpha` 作为 estimator 输入；
   - 与官方提供的 `fg` 比较，仅用于衡量与 AM-2k 官方代理前景的数值一致性；
   - 可报告全前景区和 soft-alpha 边界区的 MAE / PSNR / gradient 等，但必须标注为 **proxy-reference consistency**，不能称为真实 foreground-color accuracy；
   - 官方 `fg` / `bg` 的闭式估计误差可能同时影响该比较。

2. **已知 FG 构造输入的可控恢复实验（controlled synthetic recovery）**
   - 选定一个已知前景 `F_known`、Alpha `alpha` 和已知背景 `B_known`，按冻结的色彩空间/合成规则显式构造：
   - `I_syn = alpha * F_known + (1 - alpha) * B_known`；
   - estimator 只接收 `I_syn + alpha`，再恢复 `F_hat`；
   - 因为 `F_known` 是该合成输入的构造源，所以在**这一受控合成条件内**可以把 `F_known` 作为精确参考，量化恢复误差；
   - 如果 `F_known` 来自 AM-2k 官方 `fg`，结论仍只说明“对由该代理 FG 构造的数据的可恢复性”，不外推为真实自然图像的物理颜色真值。

3. **换背景视觉检查（re-compositing visual check）**
   - 将恢复的 foreground 与黑 / 白 / 中灰 / checker / 代表性 BG-20K 背景重新合成；
   - 固定检查发丝、毛发、半透明边缘、衣物细边及高对比区域的旧背景残色、halo、color bleeding；
   - 对 AM-2k validation 只能做结构/视觉合理性检查，不因没有独立颜色 GT 而伪造定量 accuracy。

候选方法（例如 naive original-as-FG、foreground estimation 的不同颜色空间/算法）应在同一开发子集和预先固定的合成/评价规则下比较。最终冻结依据必须综合以上三层证据，而不是只看与官方 `fg` 的单一 PSNR，也不能根据正式 E7 test 上 Any-to-Bokeh 或其他 baseline 的得分反向选择方法。

若最终冻结某一方法，报告中的结论必须限定为：

> 该 foreground-color estimator 在 AM-2k 官方代理 FG 的一致性检查、已知 FG 构造的可控恢复实验和换背景视觉检查中通过预定标准，因此作为 E7 synthetic construction 的固定工程步骤；这不等价于证明其恢复了真实场景中的物理前景颜色。

### Any-to-Bokeh 外部参考检查（200-video public-proxy 版本已完成）

正式 Reproduced Any-to-Bokeh-style Synthetic Benchmark 的场景清单、源素材、Geometry / Focus / \(K\)、Renderer 和指标配置**全部冻结后**，再使用官方公开的 Any-to-Bokeh 权重在该冻结测试集上运行一次，作为外部 **sanity check（合理性检查）**。

该检查回答的是：

> 我们冻结后的测试集上，Any-to-Bokeh 的指标量级、定性现象和失败案例是否处于可解释范围；与论文公开 synthetic 结果相比，差异是否能由测试集分布、GT Renderer、场景运动、Geometry / Focus / \(K\) 分布和评价实现解释。

执行原则：

- 论文公开结果与本项目冻结测试集结果必须分开报告，不能视为同一测试集上的直接复现；
- 只比较**指标量级、趋势和典型 failure case**，不要求数值接近或一致；
- 若指标定义、预处理或分辨率口径不同，必须先标明不可直接数值比较；
- 若差异异常大，优先审计 RGB/GT 对齐、帧顺序、色彩空间、Depth/Disparity、Focus、\(K\)、CoC、输出尺寸及 metric implementation；
- 只修复能够客观确认的实现错误。若冻结后发现必须修改 Benchmark 构造规则，应明确升级 Benchmark 版本并重新冻结，不能静默改动；
- **禁止根据 Any-to-Bokeh 与论文分数的接近程度反向调整**测试源素材、场景难度、Depth/Focus/\(K\) 分布、Blender 光圈、Renderer 参数或评价阈值；
- 该检查不是 Benchmark 选材标准，也不是 Renderer 选型依据。

推荐在报告中单独保留：

> **Comparison with Any-to-Bokeh reported synthetic results**

并明确写明：

> 本 Benchmark 受 Any-to-Bokeh 公开合成协议启发，但不是作者官方 Synthetic Test Set 的逐样本复现，因此 paper-reported numbers 仅作为外部参考，不与本项目 Table 1A / 1B 成绩混为同一评价结果。

当前 public-proxy sanity check 已完成，因此可以直接进入其他 baseline 的 Table 1A sweep；future held-out E7 仍需独立冻结和复跑。


---

## Table 2：DAVIS Real-world

**冻结状态：DAVIS_2016_V2_SIX_BASELINES_COMPLETE_AND_FROZEN（2026-09-30）。** 每方法50×25帧，六方法7500帧及300段统一H.264视频完整；预检8段均复用。主列采用MP4 VEPI，PNG和codec差值另列诊断。BokehCrafter继续deferred，不计入本轮六方法。

| Method | VEPI-public-mp4 ↑ | Background Defocus | Focus Leakage | Halo / Boundary | Temporal Flicker / Video |
|---|---:|---|---|---|---|
| DeepLens | 0.92896 | 轻虚化；breakdance旁观者仍较清晰 | 大主体保留；原生深度不能精确覆盖细结构 | 发丝/细边软化保留 | 25帧姿态连续；不作无闪烁保证 |
| Dr.Bokeh | 0.87682 | 岸边/坡面失焦可见；背景分层不均 | breakdance旁观者混入salient层 | 细线遗漏、坡边/遮挡补全痕迹保留 | 分割及清晰区随帧变化，保留失败案例 |
| BokehMe | 0.95371 | horsejump早帧虚化较强，随后转弱 | 大主体保留，细线/水花局部软化 | 细边过渡仍有偏差 | 随帧失焦强度变化；记录span，不补偿 |
| MPIB | 0.92987 | 岸边/近坡失焦可见；近焦背景仍保留 | 大主体保留，细线/水花局部软化 | 轮廓及遮挡边缘仍有软化 | 25帧顺序连续；不作无闪烁保证 |
| BokehDiff | 0.84722 | 背景失焦可见；部分近焦背景仍清晰 | 主体细节存在平滑/重绘 | 毛发、细绳及局部纹理变化保留 | 纹理变化保留，不作闪烁等级排名 |
| Any-to-Bokeh | 0.95011 | 岸边/近坡失焦可见 | 大主体保留，细结构仍有偏差 | 毛发/细线边缘残余软化 | 未见拼接处孤立错帧；不作无闪烁保证 |
| Ours | later | later | later | later | later |

定性列来自固定8段、每方法40组关键帧及全部25帧顺序接触图复核（共1200张prediction和200张input），不是实时播放盲评，不据此宣称无闪烁或给出细微时序质量排名。所有48段主视觉视频保存在`outputs/davis-baselines-20260929/{method}/evaluation/mp4/{scene}.mp4`；逐场景观察、方法限制及视频入口见[最终复核](records/davis-final-review-20260929.json)。Full-public分割/补全错误、Normalized强度变化和生成细节变化均保留；不再据论文排序调整adapter。

全量Normalized的K/span约16.0245–72.8571；Dr实际图层最大半径14.37024px，LDF阈值空/全满帧均为0，但这不代表分割质量正确。全量检查见[完整性记录](records/davis-six-method-integrity-20260929.json)，MP4/PNG逐帧值见[六方法指标](records/davis-six-method-50-20260929.json)。VEPI只能描述前景边缘保留，不能单独排序整体虚化质量。

---

## Table 3A：ATB-style Model-level Computational Cost

**定位**：回答“这个方法/模型本身算一次有多贵”，与 Any-to-Bokeh 论文 computational-cost table 使用同类维度，重点报告单帧/单位输出帧的模型级计算成本。该表用于**模型复杂度比较**，不是完整25帧视频流水的真实端到端耗时。

| Method | Device | Time / output frame (s) ↓ | Params (M) ↓ | GFLOPs / output frame ↓ | Peak VRAM (GiB, sampled) ↓ |
|---|---|---:|---:|---:|---:|
| Dr.Bokeh | RTX PRO 6000 Blackwell | 0.28808 | 50.976 | N/A† | ≈2.88 |
| BokehMe | RTX PRO 6000 Blackwell | 0.01503 | 1.164 | N/A† | ≈1.01 |
| MPIB | RTX PRO 6000 Blackwell | 0.25277 | 41.528 | N/A† | ≈2.43 |
| BokehDiff | RTX PRO 6000 Blackwell | 1.10397 | 3493.432 | N/A† | ≈18.37 |
| Any-to-Bokeh | RTX PRO 6000 Blackwell | 0.50464 | 2457.810 | N/A† | ≈38.47 |
| Ours | GPU | later | later | later | later |

**本轮3A已完成（2026-09-30）**：五种GPU方法各4个固定输入×5次正式重复，共100个计时样本；逐帧方法测第0帧，ATB保留25帧原生调用再除25。表中Time为已同步的模型段均值，不是3B的25帧E2E。不同方法具体Core起止点见§16.2；CPU分层/repair等被排除出Core的工作会进入后续3B。

† GFLOPs已执行独立算子覆盖检查，但卷积/矩阵乘法/attention计数未覆盖自定义CUDA、FFT、归一化、逐元素、插值及随机控制等完整路径。因此总GFLOPs按协议标N/A，部分计数仅保存在诊断记录，不冒充完整方法复杂度。显存为0.2秒监测及快速单帧结束点采样的设备占用峰值（包含CUDA/CuPy/context/缓存）；不是精确瞬时最大值，精确PyTorch allocated/reserved峰值另保留。Params以实际调用的神经模块中独立参数张量统计，排除未调用模块、buffer和固定核；ATB包含UNet/VAE/image encoder，Dr包含LaMa而不含未调用的LDF/MiDaS。

完整逐次时间、标准差、by-K、原生执行单元和硬件见[原始计时](records/efficiency-3a-20260930.json)，结论与计数边界见[3A最终复核](records/efficiency-3a-final-review-20260930.json)。**Table3B未启动**；3A单帧pipeline诊断时间不能填写3B。

说明：

- 分辨率统一为 **1024×576**，batch size=1；正式实测尽量使用同一 GPU、同一软件栈和空闲设备状态。
- 对原生逐帧方法，直接测单帧核心执行；对 Any-to-Bokeh / Ours 这类原生视频模型，**不能为了表格强行制造不存在的单帧入口**。应按其真实最小/固定视频执行单元测量，再除以实际输出帧数得到 `Time / output frame` 与 `GFLOPs / output frame`，同时在记录中保存原始执行单元大小。
- `Time / output frame` 属于模型/renderer 侧标准化时间：排除 checkpoint/model load、首次 CUDA context、磁盘写入和 MP4 编码；GPU 计时前后必须同步。该口径只称 **ATB-style reproduced cost protocol**，除非作者公开计时边界完全可恢复，否则不声称逐项复刻论文私有计时实现。
- `Params` 统计当前冻结配置**实际参与推理的 learned parameters**。没有参与当前执行路径的网络不计入。例如 Track A Dr.Bokeh 已给 GT alpha 时不把未调用的 LDF 参数算入；若某方法核心是非学习 renderer，则如实报告其 learned-parameter 情况，不为填表虚构参数。
- `GFLOPs` 只统计实际执行路径中可可靠计算的浮点算子；遇到 renderer、自定义 CUDA、稀疏/动态控制流等无法可靠映射为 FLOPs 的部分，标记 `N/A` 并解释原因，不能用不完整数字冒充全流程 GFLOPs。
- `Peak VRAM` 在模型已加载、warm-up 完成后重置峰值统计，再测一次标准执行；包含常驻模型参数、activation 与临时 buffer，不包含 checkpoint 加载时的瞬时复制峰值。
- Any-to-Bokeh 论文使用的硬件与本项目当前 RTX PRO 6000 Blackwell Server Edition 不同，因此论文 Time/VRAM只作外部参考；**不同 GPU 上的绝对时间不能直接当作同硬件加速比**。
- 本次3A优先测量：四段固定efficiency clips分别取第0帧作为逐帧方法输入，每帧预热1次、正式5次；ATB继续各25帧原生overlap执行，每段同样预热1次、正式5次。输入在计时结果出现前固定。入口`python scripts/measure_efficiency.py --all --table 3a`；3B另行执行，不用单帧pipeline耗时冒充25帧E2E。
- DeepLens 未强行加入 3A 主表：其当前可复现冻结实现为 native TF1.15 CPU，而 ATB 的 computational-cost 主表没有给出与本项目当前 CPU 路径可直接同口径的行。DeepLens 的实际执行成本放到 Table 3B，并显式标记 CPU。若后续需要，可另列 CPU diagnostic，不与 GPU 3A 主排序混合。
- BokehCrafter 继续 deferred，不阻塞本轮 3A。

---

## Table 3B：25-frame Video E2E Efficiency under Given Inputs

**定位**：回答“在与 Synthetic Track A 相同输入权限下，真正处理一段25帧视频要多久”。这是**系统/流水级实际效率**，与3A的模型复杂度分开。

| Method | Device | Core Time / 25f ↓ | E2E-given-inputs / 25f ↓ | FPS (E2E) ↑ | Peak Device Memory ↓ |
|---|---|---:|---:|---:|---:|
| DeepLens | CPU |  |  |  |  |
| Dr.Bokeh | GPU |  |  |  |  |
| BokehMe | GPU |  |  |  |  |
| MPIB | GPU |  |  |  |  |
| BokehDiff | GPU |  |  |  |  |
| Any-to-Bokeh | GPU |  |  |  |  |
| Ours | GPU | later | later | later | later |

统一输入与输出边界：

~~~text
Synthetic Track A 已给输入
RGB + Geometry + Focus/K
(+ 当前冻结配置明确允许的额外输入，如 Dr.Bokeh GT alpha)
        ↓
文件读取 / 解码
        ↓
方法需要的 resize / normalization / geometry adapter
        ↓
方法内部必要的 layer construction / repair / inpainting / model inference / renderer
        ↓
postprocess / clamp / color-range recovery
        ↓
25 × 576 × 1024 × 3 的 CPU RGB uint8 frames
        ← E2E-given-inputs 计时结束
        ↓
PNG write / MP4 encode
        不计
~~~

`E2E-given-inputs` **计入**：

- benchmark 已给输入的文件读取与解码；
- resize、normalization、focus/K/geometry adapter；
- 当前冻结方法真正需要的分层、repair、inpainting、模型推理、renderer；
- 必要 postprocess；
- 最终25帧转为统一 CPU RGB uint8。

`E2E-given-inputs` **排除**：

- checkpoint / model load；
- 首次 CUDA context 和一次性环境初始化；
- benchmark 外部的 Geometry/Depth 生成；
- GT alpha 的生成过程；
- 最终 PNG/MP4 写盘与编码；
- DAVIS VDA / DeepLens DAVIS DepthNet 等不属于 Track A 冻结输入权限的前置计算。

方法权限必须跟 **Synthetic Track A 冻结主表**一致：

- Dr.Bokeh：使用 GT-alpha reconstruction adapter；不把 DAVIS Full-public LDF 重新塞进 3B。
- DeepLens：使用 Track A external-geometry、native TF1.15 CPU、single-SR frozen path；不使用 DAVIS native DepthNet。
- BokehMe：Normalized + gamma=2.2。
- MPIB：gamma=2.2。
- BokehDiff：Normalized-B。
- Any-to-Bokeh：官方冻结视频推理配置。

正式测量规则：

- 统一 1024×576、25 frames、batch size=1。
- 固定当前已准备的 **4 段 Efficiency clips**，不根据速度结果更换场景；每段至少正式重复5次。
- 模型加载后至少完成1次完整25帧 warm-up；若第一、第二次 warm-up仍有明显初始化差异，可增加 warm-up，但不得按结果挑有利次数。
- GPU 方法采用 wall-clock + `cuda synchronize` 的方式界定开始/结束；不能直接用异步 Python `time()` 夹住 forward。
- 输入读取采用 **warm filesystem cache** 条件：warm-up 后正式重复，不人为 drop OS page cache。
- GPU 方法逐个串行执行，计时前必须通过连续空闲检查；不得与其他 GPU workload、DAVIS生产任务或并行 baseline 同时运行。
- `Core Time` 仅统计当前方法核心 model/renderer 执行段；外围读取、adapter、CPU/GPU transfer 与最终 CPU uint8 conversion 不计入 Core，但全部进入 E2E。
- `FPS (E2E) = 25 / E2E-given-inputs_seconds`。
- GPU 行记录 inference peak VRAM；DeepLens CPU 行记录 peak process RSS，并在表注中明确单位/设备，不能把 CPU memory 与 GPU VRAM 当作同质排名。
- 报告每方法跨固定 clips × repeats 的均值，并同时保留标准差和逐次原始记录；主表可展示 mean，机器可读记录保存完整分布。
- DeepLens CPU 与其余 GPU 方法不做无条件“快N倍/慢N倍”结论，只报告各自冻结可复现路径的实际成本。
- 本轮生产并发吞吐、历史跑全量数据耗时、后台多worker吞吐**不得**填入 Table 3B。

3A 与 3B 的一句话区别：

> **Table 3A = 模型级复杂度：这个方法算一次有多贵。**  
> **Table 3B = 视频流水级效率：给定 Track A 输入后，一段25帧视频实际多久产出。**

---

## Table 4：OPPO-style Low-FPS Depth Robustness

**当前执行边界**：Table 4完整OPPO-style corruption依赖真实或有依据的stereo depth退化统计，暂不作为Ours前置。Ours前先完成Table 1B的 **Clean Low-FPS Geometry（30/15/10/5/2 fps）** baseline；Mild/Medium/Severe、hole、confidence、boundary corruption在未来自建独立benchmark阶段再冻结。

| Depth FPS | Depth Quality | PSNR ↑ | LPIPS ↓ | FVD ↓ | FD ↓ | RM ↓ | Depth AbsRel ↓ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30 | Clean |  |  |  |  |  |  |
| 15 | Clean |  |  |  |  |  |  |
| 10 | Clean |  |  |  |  |  |  |
| 5 | Clean |  |  |  |  |  |  |
| 2 | Clean |  |  |  |  |  |  |

后续可进一步增加 Mild / Medium / Severe。

---

# 12. 效率测试统一规则

本项目 Efficiency 已拆成两个互补口径，**不能把二者的 Time 混为同一列**：

1. **Table 3A：ATB-style Model-level Computational Cost**  
   关注模型/renderer 本身的 Time / output frame、Params、GFLOPs、Peak VRAM，服务于与既有论文 computational-cost 表的同类比较。

2. **Table 3B：25-frame Video E2E Efficiency under Given Inputs**  
   关注与 Track A 相同输入权限下，完整25帧实际执行流水的 Core Time 与 `E2E-given-inputs`。

共同要求：

- 固定输入与 method configuration，不能为了速度重新选择 adapter、gamma、aperture、分辨率或视频长度；
- batch size=1；
- warm-up 后正式计时；
- 每个固定 clip 至少重复5次；
- GPU 方法计时前后显式同步；
- 正式测量期间独占已获准 GPU，不与其他任务并发；
- 模型加载和首次 CUDA context 不计；
- 保存逐次时间、均值、标准差、设备、软件环境和峰值内存；
- 生产流水的并发吞吐与正式 benchmark latency 分开记录。

## 12.1 Core Time

只统计当前冻结方法真正执行的核心模型 / renderer 段。Core 的起止点允许因方法结构不同而不同，但必须在配置/记录中逐方法写明，并满足：

- 不包含磁盘读取；
- 不包含 benchmark 外部 geometry / alpha 生成；
- 不包含最终写盘；
- GPU 核心段前后同步；
- 对视频模型按真实视频执行单元统计，不把不存在的单帧 forward 人为拆出来。

## 12.2 E2E-given-inputs

当前 Table 3B 正式采用：

> **从 benchmark 已提供的冻结输入开始，到统一 CPU RGB uint8 结果就绪为止。**

它不是“从原始 RGB 开始自动完成估深/分割的一切系统总耗时”。因此不再使用旧的“E2E必含Depth/Geometry Estimation”的通用描述作为当前主表定义。

统一链路见 Table 3B。特别地：

- Dr.Bokeh Track A 已给 GT alpha，因此 GT alpha 的生成不计；GT alpha 之后当前冻结配置实际调用的处理计入。
- DeepLens Track A 使用 external geometry，因此 DAVIS native DepthNet 不计。
- 若未来独立公平 benchmark 改成统一 RGB + low-FPS coarse depth 权限，应另建新的系统 E2E 定义，不能静默覆盖本轮 3B。

---

# 13. Any-to-Bokeh 当前实验状态与下一步

## A. Official Reproduction

当前状态：

> Any-to-Bokeh 已经在 96 GB GPU 上完整跑通。

接下来保存并统一记录：

- predicted video；
- comparison video；
- runtime；
- peak VRAM；
- config；
- 后续 Benchmark 对应 metrics。

## B. Geometry Noise Robustness

设置：

- Clean；
- Mild；
- Medium；
- Severe。

候选 corruption：

- blur / smooth perturbation；
- elastic perturbation；
- dilation / erosion；
- boundary corruption；
- missing geometry。

记录：

- PSNR；
- LPIPS；
- FVD；
- FD；
- RM。

## C. Dynamic Focus

控制：

\[
FocusDistance(t)
\]

观察：

- Focus transition；
- FD；
- RM；
- Temporal Flicker。

## D. Dynamic Blur Strength

控制：

\[
K(t)
\]

观察：

- blur change 是否连续；
- foreground 是否稳定；
- temporal artifact。

---

# 14. 输出目录建议

每个算法 / 实验统一保存：

~~~text
output/
├── videos/
│   ├── predicted.mp4
│   ├── gt.mp4
│   └── comparison.mp4
│
├── frames/
│   └── key_frames/
│
├── metrics.json
├── efficiency.json
└── config.yaml
~~~

## metrics.json

根据 Benchmark 保存：

- PSNR；
- SSIM；
- LPIPS；
- FVD；
- FD；
- RM；
- VEPI（DAVIS 时）。

## efficiency.json

- core_time；
- e2e_time；
- fps；
- peak_vram；
- params。

## config.yaml

- checkpoint；
- resolution；
- frames；
- focus；
- \(K\)；
- geometry source；
- preprocessing。

---

# 15. 当前已确定与待确认

## 已确定

- VBS 暂时移出正式实验方案；
- 核心 Benchmark：Any-to-Bokeh Synthetic / Reproduced Synthetic、DAVIS、OPPO-style；
- 核心 baseline：DeepLens、Dr.Bokeh、BokehMe、MPIB、BokehDiff、Any-to-Bokeh、BokehCrafter*、Ours；
- Any-to-Bokeh 已在 96 GB GPU 上完整跑通；
- Any-to-Bokeh Synthetic 当前以 Reproduced public-proxy 路径为执行主线；Official 路径仅在未来可验证数据真正公开时补充；
- DAVIS 负责真实泛化；
- OPPO-style 负责 Low-FPS Coarse Depth 专项；
- paired synthetic benchmark 统一 PSNR / SSIM / LPIPS / FVD / FD / RM；
- Track A / Track B 两条评价线同时保留；
- DAVIS 不机械做双 Track；
- 效率必须区分 Core Time 与 E2E Time。

## 仍需继续确认

### 1. Any-to-Bokeh Synthetic 的当前执行版本

官方 200-video test manifest、`BG_test` 列表和 Any-to-Bokeh 内部 `ModuleRenderRT` 文件仍未公开。当前已经不再等待这些缺失组件作为实验前置条件，而是固定使用：

> **200-video MPIB-public/sRGB public-proxy benchmark**

该版本已完成 Any-to-Bokeh 官方权重 200×25 全量评价，并补齐 TorchMetrics PSNR/SSIM、VideoMAE FVD 和 I3D VFID-I。Baseline Benchmark Sweep 期间**不修改 scene manifest、RGB、GT、Geometry、Focus、K 或指标口径**。

如果未来官方 test set 真正公开，可新增 Official 路径做额外复现；future strict held-out E7 也保留为独立版本，但二者都不阻塞当前 baseline sweep。

### 2. Baseline 输入适配细节

**Track A主要方法已经冻结**：DeepLens、Dr.Bokeh、BokehMe、MPIB、BokehDiff、Any-to-Bokeh均已有200×25结果和明确reported adapter；后续除客观实现错误外不再回调。BokehCrafter继续deferred。

当前真正需要继续冻结的是：

- DAVIS各baseline的real-world输入/控制适配；
- DAVIS评价ROI与baseline推理输入权限的边界；
- Track B统一的Low-FPS Geometry temporal adapter；
- Efficiency的Core/E2E计时边界。

尤其需要确认：

> Track B下所有baseline如何接入同一份30/15/10/5/2 fps Clean Geometry，并确保temporal补齐策略一致；未来OPPO-style真实噪声则另建独立阶段。

### 3. BokehCrafter 可复现性

若无法获得稳定官方实现 / checkpoint，则不进入正式 quantitative 主表。

### 4. OPPO-style Benchmark 的真实 Depth 退化模型

最终需要根据真实 OPPO Stereo Depth 的：

- FPS；
- hole；
- confidence；
- boundary error；
- scale / noise；

重新确定 synthetic degradation。

---

# 16. 当前执行顺序：Ours前的Baseline表格收尾

当前优先目标是：**把所有不依赖Ours、且已有依据可以冻结的重复性实验一次性完成。** Ours的工程实现条件已经具备，但正式方法迭代暂缓到这些baseline表格基本填满以后，以避免开发过程中反复返回补测并防止评价协议受Ours结果反向影响。

~~~text
1. Synthetic Track A — Reproduced Any-to-Bokeh-style Synthetic
   ├── Any-to-Bokeh：✅ 200×25完成
   ├── MPIB：✅ gamma=2.2冻结，200×25完成
   ├── BokehMe：✅ Normalized + gamma=2.2冻结，200×25完成
   ├── Dr.Bokeh：✅ GT Alpha reconstruction adapter作为当前复现主行
   │              Full-public / Known Layers保留诊断
   ├── BokehDiff：✅ Normalized-B主表，A保留protocol-recovery证据
   ├── DeepLens：✅ native TF1.15 CPU + single-SR saturation完成
   ├── BokehCrafter：deferred
   └── Ours：待后续填写

2. DAVIS Real-world — 下一主任务
   ├── 冻结DAVIS scene / frame / target-instance manifest
   ├── palette/object-ID mask显式转换为评价ROI
   ├── 优先采用ATB public-code VEPI operational definition
   │   （Sobel + dilated ROI + Pearson correlation）
   ├── 论文文字版Laplacian描述仅作差异记录，不按论文分数反选
   ├── 冻结各baseline的depth/focus/K/resize输入协议
   └── 先跑除Ours外所有可复现baseline

3. Efficiency — DAVIS后立即批量测
   ├── 统一1024×576 / 25 frames / batch=1
   ├── warm-up + ≥5次正式重复
   ├── Core Time
   ├── E2E Time
   ├── FPS
   ├── Peak VRAM
   ├── Params
   └── DeepLens native CPU单独标记，不与GPU FPS无条件硬排名

4. Synthetic Track B — Clean Low-FPS Geometry
   ├── 仅改变Geometry FPS：30 / 15 / 10 / 5 / 2
   ├── 暂不叠加hole/noise/boundary corruption
   ├── 先冻结统一、简单、非学习式temporal geometry adapter
   ├── 所有baseline使用同一补齐规则
   └── 除Ours外全部方法批量跑完

5. Baseline Table Freeze
   ├── Table 1A Track A：baseline完成
   ├── Table 1B Track B-Clean：填满baseline
   ├── Table 2 DAVIS：填满baseline
   ├── Table 3 Efficiency：填满baseline
   ├── Table V：保持冻结
   └── 之后除客观实现错误外不根据Ours成绩回调

6. Ours
   ├── 开始正式实现 / 训练 / 迭代
   └── 每个版本只需补上述冻结评价体系中的Ours行
~~~

### 16.1 DAVIS先做的原因

DAVIS数据本身已经齐全，当前缺的是协议而不是下载。它同时补足ATB论文中的真实视频评价，并且不依赖Ours，因此最适合先完成。公开 edge_batch.py 实际操作与论文文字版VEPI存在差异，当前优先冻结**作者公开代码的可执行定义**，并把差异作为复现限制记录；不能为了接近论文VEPI数值反选算法定义。

### 16.2 Efficiency统一口径

当前按Table3A/3B及§12的新定义执行。配置`configs/efficiency-track-a.json`、入口`scripts/measure_efficiency.py`；此前尚未产出正式计时，旧配置将说明文字混入状态值的启动检查问题已修复。模型、checkpoint和Track A输入权限均不变。

**Core与E2E**：删除旧版“host输入到host输出整体算Core”的边界。Any-to-Bokeh按已经位于CUDA的RGB/CoC/MPI/K进入公开视频pipeline、到CUDA float输出为Core，三次原生调用输出8/8/9帧；BokehMe为已经CUDA输入的官方demo pipeline；BokehDiff为CUDA输入的VAE.encode加TiledSDXL到CUDA float输出。MPIB分段累计inpnet/mpinet/aupnet/renderer的forward；Dr分段累计两次LaMa forward和一次完整Multi_Layer_Renderer.forward（内部两层），CPU分层/cv2修复及传输只进E2E。DeepLens为native TF1 Session.run，输入adapter和最终CPU uint8转换在Core外。各段采用同步+wall-clock；外围读取、适配、传输、后处理和最终25张CPU RGB uint8全部进入E2E，写盘/编码在计时后。

Table3A从实际单帧执行记录（逐帧方法）或原生视频调用记录（ATB）按实际输出帧数归一，不能使用3B的E2E除25冒充模型时间。固定repro_004/006/003/000分别覆盖K4/8/16/32，每段完整预热1次、正式5次；GPU方法串行独占，DeepLens TF1CPU8线程，其他Torch4线程。保留每次Core分段、原生执行单元、E2E、跨重复标准差和by-K，不重挑场景。

**内存与参数**：CUDA每个真实执行单元重置PyTorch峰值并记录allocated/reserved，另在正式重复内每0.2秒采样设备总显存，涵盖CUDA/CuPy常驻内存；设备值明确是采样峰值，统一GiB。DeepLens按同周期采样实际进程RSS，CPU内存与GPU显存分设备报告，均不混入加载阶段峰值。神经网络以实际调用的模块记录learned parameters，排除未调用的网络、buffer和固定renderer核；Dr不调用LDF/MiDaS，其已加载但未调用的常驻显存仍如实保留。

**GFLOPs**：在全部latency重复完成后，另走一次固定输入的dispatcher算子覆盖检查，避免profiling开销混进时间。保留可计卷积/矩阵乘法/attention算术及实际操作列表；自定义CUDA、FFT、归一化、逐元素、插值或随机控制等路径未完整覆盖时，3A总GFLOPs标N/A并注明缺口，部分计数只作诊断。不会把不完整算子计数标成整个方法的GFLOPs。

计时专用入口移除Dr/BokehDiff的中间诊断数组拷贝，原生产adapter默认仍保留诊断，数学路径不变；先以固定输入核对输出一致性。产物放`outputs/efficiency-track-a-3ab-20260930/`，日志放`logs/efficiency-track-a-3ab-20260930/`，汇总写`records/efficiency-track-a-3ab-20260930.json`。DAVIS最终冻结状态与连续空闲检查通过后才测量；GPU当前已通过最小CUDA计算并恢复空闲，未更改驱动或放宽启动条件。

### 16.3 Track B先只做Clean Low-FPS

Track B当前首先回答一个单一问题：**当每帧完整Geometry变成越来越低帧率的Clean Geometry时，各方法如何退化。** 因此第一版只使用30/15/10/5/2 fps，不同时叠加噪声、hole或边界错误。先冻结一个公共、简单、非学习式temporal adapter，再给全部baseline使用，避免每个方法各自引入不同补全能力。

完整OPPO-style Mild/Medium/Severe corruption不作为Ours前置门槛。真实hole、confidence、boundary error、scale/noise应依据OPPO stereo sensor或可说明的数据统计后再设计，属于未来本项目独立benchmark；不能为了把Table 4填满而现在主观制造参数。

BokehCrafter继续保持deferred：只有官方实现/checkpoint可稳定复现后才补入quantitative主表，不阻塞本轮收尾。

当前阶段最重要的是：

> **先把DAVIS、Efficiency和Track B Clean的baseline一次测完并冻结；随后再开始Ours。此后Ours只补自己的行，不允许模型结果反向改变baseline或benchmark协议。**

---

# 17. Any-to-Bokeh-style Synthetic Benchmark 的构建细节

这一节记录当前公开的 \`dataset_synthsis.py\` 所体现的数据生成逻辑，用于后续构建：

> **Reproduced Any-to-Bokeh-style Synthetic Benchmark**

这里的目标不是声称 100% 复刻论文官方 200-video test set，而是尽量复现其核心生成思想，并构建一套可固定、可复现、可供所有 baseline 公平比较的 paired video bokeh benchmark。

---

## 17.1 最终希望生成的数据

每个 synthetic scene 最终应至少包含：

~~~text
scene_xxxx/
├── aif/
│   └── 全清晰 RGB 视频帧
│
├── disparity/
│   └── 每帧视差
│
├── mask/
│   └── 每个前景对象的二值遮罩
│
├── alpha/
│   └── 每个前景对象的软透明度图（建议额外保存）
│
├── bokeh_gt/
│   └── 对应的 Bokeh 真值视频
│
└── metadata.json
    └── focus / K / depth / source / seed 等元信息
~~~

其中：

- **全清晰图（All-in-Focus，AiF）**：前景和背景都保持清晰；
- **视差（Disparity）**：表示场景相对远近的一种几何量，通常近处视差更大；
- **Alpha Matte**：0～1 的软透明度图，用于描述发丝、软边界等；
- **Mask**：0/1 二值遮罩；
- **Bokeh GT**：由渲染器生成的虚化真值。

最终核心 paired data 为：

\[
RGB + Geometry + Focus + K + Bokeh\ GT
\]

这套数据既可以用于 Track A，也可以作为 Track B 的基础。

---

## 17.2 前景数据来源

当前公开脚本中可以确认至少使用了两类前景数据。

### 1. VideoMatte240K

VideoMatte240K 是视频抠图数据。

它提供：

- 视频前景；
- 对应透明度 / Alpha；
- 连续时间序列。

公开脚本中能看到类似：

~~~text
VideoMatte240K_JPEG_HD/train/fgr/...
VideoMatte240K_JPEG_HD/train/pha/...
~~~

其中：

- \`fgr\`：foreground，前景 RGB；
- \`pha\`：pha / alpha，透明度图。

这类数据特别适合：

- 人物；
- 发丝；
- 衣服边缘；
- 连续运动前景。

---

### 2. AM-2k

脚本中也明确出现：

~~~text
AM-2k/train/fg/...
AM-2k/train/mask/...
~~~

AM-2k 属于图像抠图数据。

和 VideoMatte240K 的区别可以简单理解为：

~~~text
VideoMatte240K
→ 本身就是视频前景

AM-2k
→ 单张前景图片
~~~

对于单张图像素材，可以通过重复 25 帧并人为施加位置变化，转成 synthetic video foreground。

---

## 17.3 前景统一成 25 帧序列

作者并不是直接在生成脚本里扫描整个数据集，而是通过：

\`merge_f25_test.json\`

统一组织前景。

其逻辑可以理解为：

~~~text
Object 0
├── 25 张 foreground frames
└── 25 张对应 alpha / mask

Object 1
├── 25 张 foreground frames
└── 25 张对应 alpha / mask
~~~

JSON 中核心字段是：

~~~text
clip_paths
clip_mask_paths
~~~

即：

- 前景帧路径；
- 对应遮罩 / Alpha 路径。

因此后续我们自己构建时，也建议先把所有前景素材统一转换成：

> **每个对象固定 25 帧的标准序列。**

---

## 17.4 背景数据

当前脚本引用了作者服务器中的：

~~~text
BG_test
Backgrounds
BG_imgs
~~~

这些实际背景素材没有随仓库完整公开。

因此后续自建 Benchmark 时需要替换为公开、可重复获取的背景图像来源。

建议背景覆盖：

- 室内；
- 街道；
- 建筑；
- 公园；
- 自然景观；
- 办公环境；
- 复杂纹理背景。

背景数据必须固定版本与 split，避免测试集随时间变化。

---

## 17.5 当前脚本的基本生成参数

公开脚本当前默认值包括：

~~~text
synthsis_images_num = 300
obj_num = 4
resolution = 1024 × 576
frame_num = 25
~~~

也就是：

- 每个视频 25 帧；
- 分辨率 1024×576；
- 每个场景最多组合 4 个前景对象；
- 当前脚本默认生成 300 个 scene。

需要注意：

> 论文描述的 synthetic test set 为 200 videos，而当前公开脚本默认值为 300。

因此可以进一步确认：

> **公开 \`dataset_synthsis.py\` 是数据生成代码版本，不等同于论文官方 200-video test set 的完整一键复现包。**

---

## 17.6 前景如何放进画面

对于 \`obj_num = 4\`，脚本先把画布划分成若干区域，为每个对象分配一个初始位置。

之后再加入随机位置扰动，避免每个 scene 都呈现规则网格布局。

可以简单理解为：

~~~text
1024 × 576 Canvas
│
├── Object A
├── Object B
├── Object C
└── Object D
~~~

但每个对象会有独立的随机位置。

---

## 17.7 随机缩放与旋转

每个前景对象会做数据增强。

### 缩放

当前代码大致使用：

\[
0.5 \sim 0.8
\]

的随机比例。

### 旋转

随机范围大致为：

\[
-10^\circ \sim +10^\circ
\]

目的：

- 增加场景变化；
- 防止相同素材每次出现形式完全一致。

---

## 17.8 视频运动是怎么来的

脚本会对每帧的前景中心位置增加随机扰动，例如：

~~~text
x / y 方向约 -40 ～ +40 pixel
~~~

因此前景会随时间产生位置变化。

对于 VideoMatte240K：

- 原前景人物本身已经有动作；
- 再叠加整体位置移动。

对于 AM-2k：

- 原图本身静止；
- 主要依赖 synthetic position change 形成视频运动。

因此 synthetic video 中同时可能包含：

- 人体内部动作；
- 整体对象移动；
- 多对象相对运动。

---

## 17.9 前景的深度 / 前后关系

脚本会人为给每个前景对象和背景赋予不同的几何位置。

当前参数类似：

~~~text
foreground dof = (1, 5)
background dof = (100, 200)
~~~

可以简单理解为：

> 前景对象分布在较近的不同深度层；背景放在远处。

多个 foreground 的深度顺序还会随机打乱，因此可以生成不同的前后遮挡关系。

---

## 17.10 视差如何生成

代码中场景几何使用类似：

\[
d(x,y)=\frac{1-ax-by}{c}
\]

的形式。

当前生成设置中：

\[
a=0,\quad b=0
\]

因此近似为：

\[
\boxed{d=\frac{1}{c}}
\]

也就是：

- \(c\) 小 → 更近；
- disparity 大；
- \(c\) 大 → 更远；
- disparity 小。

例如：

\[
c=1 \Rightarrow d=1
\]

\[
c=5 \Rightarrow d=0.2
\]

背景由于 \`c\` 更大，因此 disparity 很小。

每一帧都会生成一张对应的 disparity map。

---

## 17.11 Alpha 与 Mask 的作用

### Alpha

Alpha 是 0～1 的软透明度。

例如发丝边缘：

\[
\alpha = 0.4
\]

可以表示该像素同时包含前景与背景。

合成公式：

\[
I=\alpha F+(1-\alpha)B
\]

因此 Alpha 对：

- Hair；
- Thin Structure；
- Soft Boundary；

尤其重要。

### Mask

脚本还会通过类似：

\[
\alpha > 0.5
\]

得到二值 Mask。

Mask 主要用于：

- 标出物体区域；
- 计算对象几何；
- 确定 focus target；
- 后续评价。

---

## 17.12 AiF 视频如何生成

**AiF（All-in-Focus，全清晰图）**：

> 所有深度层都保持清晰。

脚本根据对象的前后关系，把：

~~~text
Background
↓
Far Foreground
↓
...
↓
Near Foreground
~~~

利用 Alpha 逐层合成。

最终每帧保存为：

~~~text
aif/0.jpg
aif/1.jpg
...
aif/24.jpg
~~~

这可以作为 Bokeh 模型的 RGB 输入。

---

## 17.13 Bokeh GT 如何生成

作者使用：

\`ModuleRenderRT\`

完成虚化真值渲染。

可以把它理解成：

> **光线追踪式 Bokeh 渲染模块（Ray-Tracing Bokeh Renderer）**

输入主要包括：

~~~text
RGB Layers
+
Alpha Layers
+
Geometry
+
Focus
+
Blur Strength K
~~~

输出：

~~~text
Bokeh GT Video
~~~

这一步把前面构造的 synthetic scene 转成 paired Bokeh ground truth。

---

## 17.14 K：虚化强度

参数 \(K\) 用于控制虚化强度。

简单理解：

\[
K\uparrow
\Rightarrow
Blur\uparrow
\]

当前 test generation 分支会从：

\[
\{4,\ 8,\ 16,\ 32\}
\]

中选择不同的 Bokeh strength。

因此同一类场景可以覆盖：

- 轻微虚化；
- 中等虚化；
- 强虚化。

---

## 17.15 Focus：焦点如何设置

脚本会计算每个 foreground object 对应的焦平面值。

可以理解为：

> 当前希望保持清晰的对象，对应一个 focus disparity。

同一个 AiF scene 可以针对不同 object 分别生成 Bokeh：

~~~text
同一个 AiF Video
│
├── Focus Object 0 → Bokeh GT 0
├── Focus Object 1 → Bokeh GT 1
├── Focus Object 2 → Bokeh GT 2
└── Focus Object 3 → Bokeh GT 3
~~~

如果目标对象随时间发生深度变化，focus 也可以随对象对应 geometry 变化。

这对于：

- moving subject；
- dynamic focus；
- temporal evaluation；

都很有价值。

---

## 17.16 整个生成流程

完整逻辑可以概括为：

~~~text
VideoMatte240K / AM-2k
        ↓
Foreground + Alpha
        │
        ├──────────────┐
        │              │
公开 Background Images │
        │              │
        └──────┬───────┘
               ↓
随机选择多个前景
               ↓
随机：
位置
缩放
旋转
运动
深度
前后关系
               ↓
合成 25 帧 AiF Video
               ↓
生成每帧 Disparity
               ↓
保存 Alpha / Mask
               ↓
设置 Focus Object
+
Blur Strength K
               ↓
Bokeh Renderer
               ↓
Bokeh GT Video
~~~

最终形成：

\[
\boxed{
RGB + Disparity + Alpha/Mask + Focus + K + Bokeh\ GT
}
\]

---

## 17.17 当前公开脚本的三个主要缺口

### 1. \`ModuleRenderRT\` 缺失

公开脚本引用：

~~~python
from z_bokeh.data_process.bokeh_kernel import ModuleRenderRT
~~~

但当前公开仓库中没有找到对应完整实现。

因此：

> **作者原 Bokeh Renderer 目前不是直接可运行状态。**

这是自建 benchmark 当前最大的技术缺口。

---

### 2. 作者背景数据缺失

如：

~~~text
BG_test
Backgrounds
BG_imgs
~~~

均为作者内部服务器路径。

需要替换成我们自己选定的公开背景数据集。

---

### 3. \`merge_f25_test.json\` 未提供

公开代码给出了 JSON 格式示例，但没有提供作者实际使用的完整 test JSON。

因此我们需要自己：

~~~text
VideoMatte240K / AM-2k
↓
统一成 25-frame clips
↓
生成自己的 merge_f25_test.json
~~~

---

## 17.18 我们自己的版本建议额外保存什么

为了同时服务后续 Track A / Track B，建议比原脚本多保存：

~~~text
scene_xxxx/
├── rgb/
├── depth_gt/
├── disparity_gt/
├── alpha/
├── mask/
├── bokeh_gt/
└── metadata.json
~~~

其中 \`metadata.json\` 至少记录：

- random seed；
- foreground source；
- background source；
- object depth；
- object order；
- focus object；
- per-frame focus；
- \(K\)；
- resolution；
- number of frames；
- renderer version。

这样后续可以直接从 Full-FPS Geometry 构造：

~~~text
Track A
→ Full per-frame Geometry

Track B
→ Downsample / Corrupt Geometry
   30 → 15 → 10 → 5 → 2 fps
   + Noise / Hole / Boundary Error
~~~

而无需重新生成场景。

---

## 17.19 当前构建原则

当前建议遵循：

> **复现 Any-to-Bokeh 的生成逻辑，而不是强求复刻作者完全相同的 200 个测试样本。**

优先保证：

- 公开数据源；
- 固定 split；
- 固定随机种子；
- 固定生成脚本；
- 固定 renderer；
- 固定 metadata；
- 所有 baseline 使用同一测试数据。

这样才能把它真正作为一个：

> **可复现、可审计、可公平比较的 paired video bokeh benchmark。**


---

# 18. 当前最终表格数量

当前实验方案共保留 **5 张主表 + 1 张辅助 Benchmark Validation 表**：

1. **Table 0：GT Renderer Comparison**  
   用于选择正式 Synthetic Benchmark 的 Bokeh GT Renderer。

2. **Table 1：Any-to-Bokeh Synthetic / Reproduced Synthetic**  
   主 paired synthetic 定量 Benchmark。

3. **Table 2：DAVIS Real-world**  
   真实视频泛化。

4. **Table 3：Efficiency**  
   拆成 **Table 3A：ATB-style Model-level Computational Cost** 与 **Table 3B：25-frame Video E2E Efficiency under Given Inputs**；前者报告模型级计算成本，后者报告实际25帧流水效率。

5. **Table 4：OPPO-style Low-FPS Depth Robustness**  
   Low-FPS Coarse Depth 专项鲁棒性评价。

其中 Table 1 与 Table 3 在最终报告中分别拆成：

- **Table 1A：Track A — Unified Per-frame Geometry**
- **Table 1B：Track B — Low-FPS Coarse Geometry**
- **Table 3A：ATB-style Model-level Computational Cost**
- **Table 3B：25-frame Video E2E Efficiency under Given Inputs**

因此仍可理解为 **5 个主表主题 + 1 张辅助验证表**，但实际展示为：

\[
\boxed{7\ 个主结果表块 + 1\ 个 Benchmark\ Validation\ 表}
\]

即：

~~~text
Table 0   GT Renderer Comparison（历史开发验证）

Table 1A  Synthetic Track A
Table 1B  Synthetic Track B
Table V   Benchmark Validation vs Paper Table 1（辅助）

Table 2   DAVIS Real-world

Table 3A  ATB-style Model-level Computational Cost
Table 3B  25-frame Video E2E Efficiency under Given Inputs

Table 4   OPPO-style Low-FPS Depth Robustness
~~~
