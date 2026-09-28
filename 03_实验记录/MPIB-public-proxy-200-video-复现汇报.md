# Any-to-Bokeh Synthetic Evaluation 公开代理复现：200-video 定量验证

> 任务：使用公开可获得组件重建 Any-to-Bokeh synthetic evaluation pipeline，并验证官方权重的指标量级  
> 当前状态：全量任务运行中；本文中的 39.47 dB / 0.9865 / 0.4217 / 0.0058 为当前阶段性统计，最终以 200/200 汇总结果为准  
> 核心方案：Any-to-Bokeh 公开场景生成逻辑 + MPIB 官方 ray-tracing renderer proxy + Any-to-Bokeh 官方权重

---

## 一、核心结论

### 公开代理复现已取得高度一致的阶段性结果

基于 Any-to-Bokeh 公开的数据生成逻辑、论文明确引用的 MPIB ray-tracing 方法以及官方发布模型权重，我们已经构建并跑通 **200-video / 5000-frame** 规模的公开代理 synthetic evaluation pipeline。

当前阶段性结果显示，官方 Any-to-Bokeh 权重在该复现数据上的主要指标与论文 Table 1 处于高度一致的量级：

- **PSNR：39.47 dB**，论文报告 **38.899 dB**，绝对差异 **+0.57 dB**
- **SSIM：0.9865**，论文报告 **0.974**
- **FD：0.4217**，论文报告 **0.431**
- **RM：0.0058**，论文报告 **0.007**

更重要的是，这种一致性不是单一指标上的偶然接近，而是与以下三层证据同时吻合：

1. **公开参数高度对齐**：分辨率、帧数、4 对象布局、深度范围、随机缩放/旋转/运动、K 档位、101 ray sampling 等核心设置均与作者公开 generator 对齐；
2. **Renderer 证据闭环**：论文明确 following MPIB；MPIB 官方 `ModuleRenderRT` 与 Any-to-Bokeh 公开六参数调用形式吻合；独立 probe 中两份公开 MPI-style ray kernel 在同一工作色彩空间下达到 74.6–76.9 dB 的相互 PSNR；
3. **官方 checkpoint 行为对齐**：空间质量与视频时序指标同时落在论文报告的相近区间。

综合来看，当前结果有力支持：

- **MPIB-public `ModuleRenderRT` 是当前公开条件下最有依据的 Any-to-Bokeh renderer proxy；**
- **当前 scene construction 已成功恢复论文公开 synthetic generation 的主要统计特征；**
- **官方 Any-to-Bokeh 权重在该 pipeline 上表现出与论文报告高度一致的行为；**
- **200-video / 5000-frame 的 synthetic evaluation 工程链路已经具备稳定批量运行能力。**

本实验定位为基于公开资料恢复论文 synthetic evaluation 的主要行为；官方未公开的测试清单和内部 renderer 不属于本轮可恢复对象。

---

## 二、公开参数口径核对

### 2.1 场景生成核心参数

| 项目 | Any-to-Bokeh 公开代码 | 当前复现 | 状态 |
|---|---|---|---|
| 分辨率 | 1024×576 | 1024×576 | ✅ 对齐 |
| 每段帧数 | 25 | 25 | ✅ 对齐 |
| 前景对象数 | 4 | 4 | ✅ 对齐 |
| 前景深度范围 | 1–5 | 1–5 | ✅ 对齐 |
| 背景深度范围 | 100–200 | 100–200 | ✅ 对齐 |
| 前景缩放 | 0.5–0.8 | 0.5–0.8 | ✅ 对齐 |
| 旋转范围 | -10°–10° | -10°–10° | ✅ 对齐 |
| 每帧中心扰动 | -40–40 px | 同口径随机运动 | ✅ 对齐 |
| Test K 值 | {32,16,8,4} | {4,8,16,32} | ✅ 对齐 |
| Alpha crop | mask > 128 | mask > 128 | ✅ 对齐 |
| VideoMatte 时间采样 | 公共示例 stride=4 | stride=4 | ✅ 对齐 |
| 前景布局 | 2×2 grid + 随机扰动 | 保留 | ✅ 对齐 |
| Ray sampling | 101 | 101 | ✅ 对齐 |
| 随机种子基准 | NumPy seed 20 | seed 20 + scene_id 的确定性扩展 | ✅ 可复现 |
| 基础场景规模 | 公开脚本当前写 300；论文报告 200 test videos | 固定 200 | ✅ 对齐论文规模 |

### 2.2 模型推理参数

当前推理使用官方 checkpoint，并保持作者公开 demo 的关键设置：

| 项目 | 当前设置 |
|---|---|
| Resolution | 1024×576 |
| Group frames | 8 |
| Overlap frames | 4 |
| Motion bucket | 127 |
| FPS condition | 7 |
| Noise augmentation | 0 |
| Inference steps | 1 |
| Generator seed | 0 |
| Model code | 官方 Any-to-Bokeh，未修改模型主体 |

因此当前复现不是只对齐数据生成，**模型推理路径本身也与作者公开 inference protocol 高度一致。**

---

## 三、GT Renderer：MPIB-public proxy 的证据链

### 3.1 为什么选择 MPIB-public

Any-to-Bokeh 论文 Appendix A.2 明确说明 synthetic Bokeh generation **following MPIB** 使用 ray-tracing-based method。

作者公开 generator 中的调用形式为：

```python
module(images, alphas, coffs_input, k, 1 / zf, 101)
```

MPIB 官方仓库根目录 `bokeh_generator.py` 中：

```python
ModuleRenderRT(images, alphas, coffs, K, depth_focus, samples_per_side)
```

六参数结构吻合，第六参数正是 `samples_per_side`，当前复现固定为 **101**。

### 3.2 独立 renderer probe

在相同输入和工作色彩空间下：

| 对比 | PSNR | 解释 |
|---|---:|---|
| MPIB vs BokehDiff ray kernel | **74.6–76.9 dB** | 两份公开 MPI-style ray 实现数值高度接近 |
| MPIB/linear vs Cycles/linear | **53.9–59.8 dB** | 不同 renderer 在统一 linear 口径下仍保持较高一致性 |
| MPIB/sRGB vs Cycles/linear | **31.8–33.2 dB** | 工作色彩空间对最终参考域影响显著 |

这一结果说明当前选择不是根据 Any-to-Bokeh 得分反向挑选 renderer，而是由**论文引用、公开接口和独立 renderer probe**共同支持。

因此本项目统一使用名称：

> **MPIB-public proxy**

它强调“公开可复现的 MPIB 官方代理实现”，同时避免把未公开的 Any-to-Bokeh 内部文件与公开 MPIB 文件混为一谈。

---

## 四、阶段性定量结果

### 4.1 当前复现 vs 论文 Table 1

> 当前数值为运行中的阶段性汇总；最终 200/200 完成后更新同一表格。PSNR 使用 dB 绝对差异，不采用百分比差异表达。

| 指标 | 论文 Table 1 Any-to-Bokeh | 当前阶段性复现 | 差异 | 当前观察 |
|---|---:|---:|---:|---|
| **PSNR ↑** | **38.899 dB** | **39.47 dB** | **+0.57 dB** | 高度接近 |
| **SSIM ↑** | **0.974** | **0.9865** | +0.0125 | 同一高质量区间 |
| **FD ↓** | **0.431** | **0.4217** | -0.0093 | 高度接近 |
| **RM ↓** | **0.007** | **0.0058** | -0.0012 | 同一量级 |
| **VFID-I ↓** | **1.479** | 待补充 | - | 待同口径评价 |
| **FVD ↓** | **9.005** | 待补充 | - | 待同口径评价 |
| **LPIPS ↓** | Table 1 未直接报告 | **0.0165** | - | 本项目额外诊断指标 |

### 4.2 论文 Table 1 完整参考

| Method | PSNR ↑ | SSIM ↑ | FD ↓ | RM ↓ | VFID-I ↓ | FVD ↓ | VEPI ↑ | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepLens | 24.574 | 0.819 | 1.162 | 0.030 | 16.042 | 125.338 | 0.715 | 0.226 |
| BokehDiff | 27.525 | 0.834 | 0.660 | 0.021 | 7.395 | 65.678 | 0.859 | 0.799 |
| BokehMe | 27.992 | 0.936 | 0.536 | 0.013 | 8.633 | 39.102 | 0.937 | 0.103 |
| Dr.Bokeh | 31.273 | 0.950 | 0.522 | 0.011 | 6.097 | 32.710 | 0.863 | 2.729 |
| MPIB | 31.390 | 0.950 | 0.481 | 0.011 | 5.444 | 35.766 | 0.921 | 0.521 |
| **Any-to-Bokeh** | **38.899** | **0.974** | **0.431** | **0.007** | **1.479** | **9.005** | **0.944** | **0.363** |

---

## 五、当前最值得强调的发现

### 5.1 PSNR 已进入论文同一数值区间

当前：

[
39.47 mathrm{dB}
]

论文：

[
38.899 mathrm{dB}
]

绝对差异仅 **+0.57 dB**。

考虑到当前使用的是重新抽样的 foreground/background combination、公开 MPIB renderer proxy 和独立固定 manifest，这一结果对“当前 pipeline 已恢复论文 synthetic evaluation 的主要行为”提供了非常强的支持。

### 5.2 FD 与论文报告几乎重合

当前 FD：

[
0.4217
]

论文：

[
0.431
]

FD 基于相邻帧光流差异，与单帧 PSNR 属于不同评价维度。空间质量和时序行为同时接近论文结果，说明当前复现不仅恢复了单帧外观，还恢复了相近的 temporal evaluation behavior。

### 5.3 sRGB / renderer 口径解释了历史 Cycles 差异

历史开发结果：

| GT Renderer | Any-to-Bokeh PSNR | 用途 |
|---|---:|---|
| **MPIB-public / sRGB** | **39.47 dB** | 当前公开代理复现 |
| Cycles / linear RGB | 33.11 dB | 独立物理 reference 开发实验 |
| Dr.Bokeh | 33.06 dB | 早期交叉检查 |

这些结果对应不同 reference domain，不用于判定 renderer 谁“更准确”。

当前最重要的发现是：

> **Any-to-Bokeh 官方权重与 MPIB-public / sRGB reference domain 高度匹配，这与论文 synthetic data 的构造来源和公开调用方式一致。**

这也是当前复现指标能够进入论文同一数值区间的重要原因。

---

## 六、技术配置

| 项目 | 配置 |
|---|---|
| **GT Renderer** | MPIB-public (`bokeh_generator.py::ModuleRenderRT`) |
| **Renderer Commit** | `d77d93d` |
| **采样密度** | 101×101 网格中的单位圆 ray sampling |
| **工作色彩空间** | sRGB composition + sRGB-domain ray integration |
| **场景数量** | 200 |
| **帧数/场景** | 25 |
| **总帧数** | 5000 |
| **分辨率** | 1024×576 |
| **K** | 4 / 8 / 16 / 32 |
| **前景来源** | AM-2k train/fg（固定 composition asset）+ VideoMatte240K train |
| **背景来源** | BG-20k train |
| **输出格式** | PNG8 |
| **随机种子** | 固定 deterministic manifest |
| **模型** | Any-to-Bokeh 官方 checkpoint |

---

## 七、指标口径核对：哪些保留，哪些重算

这里的目标不是重跑实验，而是让最终“论文数字 vs 当前数字”的比较尽可能使用作者公开指标实现。

### 7.1 可以直接保留，不需要重新 render / inference

以下内容已经完成且可继续作为最终汇报证据：

- **200-scene manifest**
- 所有 foreground/background/source ID、运动、depth、Focus、K 配置
- sRGB 输入生成结果
- MPIB-public proxy GT
- Any-to-Bokeh 官方 checkpoint 预测结果
- Renderer probe（MPIB / BokehDiff / Cycles）
- 当前 PSNR / SSIM / LPIPS / FD / RM，作为本项目统一诊断指标
- 完整性检查、输入 CoC 量化误差检查
- 现有 timing 作为工程运行记录
- 按 K、AM/VM 类型进行分组分析的原始结果

因此后续指标核对：

[
oxed{	ext{不需要重新生成场景，不需要重新渲染 GT，也不需要重新跑 Any-to-Bokeh}}
]

只需要对已经生成的 GT / prediction 重新评价。

### 7.2 建议用作者公开 metric implementation 再算一次

#### PSNR

当前使用 **mean per-frame PSNR**。

作者公开 `vid_metrics.py` 使用 TorchMetrics `PeakSignalNoiseRatio(data_range=1.0)`。

最终建议同时保留：

- 当前 mean-frame PSNR：便于项目内部统一比较；
- **author-public PSNR**：用于和 Table 1 做最直接对照；
- 已保存的 `PSNR_clip_mse`：作为聚合方式诊断。

PSNR 的汇报统一使用 **dB 绝对差异**，不计算百分比差异。

#### SSIM

当前实现：

- skimage
- Gaussian sigma=1.5
- population covariance

作者公开实现：

- TorchMetrics `StructuralSimilarityIndexMeasure(data_range=1.0)`

因此最终 Table 1 对照应增加一列或直接采用 **author-public SSIM**。当前 0.9865 继续保留为项目内部指标，不作废。

#### FD

当前使用 torchvision RAFT-large，并对 24 个相邻 transition 计算 flow L1 difference。

作者公开实现同样采用 torchvision `raft_large` + RAFT weights transforms，因此当前 **0.4217** 已具有较强直接对照价值。

为彻底消除 torchvision / weight alias / preprocessing 版本差异，最终可以再用作者公开 `compute_flow_error` 路径重算一次；这属于低成本确认，不涉及模型重跑。

### 7.3 RM：保留我们的完整 24-transition 版本，并单独记录作者公开代码口径

当前 RM：

[
0.0058
]

计算全部 24 个相邻帧 transition，定义清楚、适合作为后续项目统一时序指标。

作者公开 `temporal_consistency` 代码中，transition 数量的维度读取存在与其 `B,C,T,H,W` 数据布局不完全匹配的迹象。因此最终建议：

- **主项目继续保留 24-transition RM**；
- 如需严格复刻 Table 1 数字，可额外运行作者 public code 原样口径并单列；
- 两者不互相覆盖。

这样既保留了当前更完整的时序评价，也能完成论文表格的公开代码对照。

### 7.4 LPIPS

当前 `LPIPS squeeze v0.1` 可直接保留。

论文 Table 1 未直接给 LPIPS，因此它属于本项目额外的 perceptual-quality 诊断指标，不需要为论文数值对齐而修改。

### 7.5 FVD / VFID-I

这是全量 200-video 完成后的主要补充项：

- **FVD：论文 9.005**
- **VFID-I：论文 1.479**

优先按照作者公开 `vid_metrics.py` / VFID 路径和对应权重配置计算。

如果这两项同样进入论文相近量级，将形成：

[
	ext{spatial fidelity}
+
	ext{optical-flow temporal consistency}
+
	ext{video distribution quality}
]

三类证据的完整闭环。

---

## 八、200/200 完成后的最终核对清单

### 第一优先级：直接更新最终成绩

1. 验证 200 场景 / 5000 帧 RGB、GT、prediction 完整；
2. 更新最终 PSNR / SSIM / LPIPS / FD / RM；
3. 输出 K=4/8/16/32 分组结果；
4. 输出 AM / VideoMatte 分组结果；
5. 检查代表性高分、中位和困难样例。

### 第二优先级：作者公开 metric 同口径重算

直接使用现有输出，不重新跑模型：

1. TorchMetrics PSNR；
2. TorchMetrics SSIM；
3. 作者公开 RAFT FD；
4. 作者 public-code RM 对照；
5. FVD；
6. VFID-I。

完成后，本汇报中的“当前阶段性结果”替换为“最终 200/200 结果”。

---

## 九、对后续 Benchmark 的价值

本轮最重要的价值不是简单得到一个接近 38.899 dB 的数字，而是已经验证了一套完整可复现链路：

[
	ext{公开 scene recipe}
ightarrow
	ext{MPIB-public ray GT}
ightarrow
	ext{Any-to-Bokeh 官方推理}
ightarrow
	ext{视频质量评价}
]

因此后续 E7 不需要再重新探索 renderer 方向，可以直接复用：

- 已验证的数据生成规则；
- MPIB-public proxy renderer；
- 固定 K / Focus / geometry 定义；
- 当前批量调度和完整性检查工具；
- 已验证的 Any-to-Bokeh 输入/推理链路；
- 后续完成同口径校准后的 metric implementation。

这使后续工作可以更快转向 **baseline 统一比较和 Ours 方法构建**。

---

## 十、复现范围说明

当前 200-video 集合按照作者公开生成思想重新构建，重点恢复**公开参数、renderer 类型和统计行为**。其中：

- 官方 `merge_f25_test.json` 未公开，因此 scene identity 使用本项目固定 manifest；
- 官方 `BG_test` 列表未公开，因此背景使用 BG-20k train 的固定公开替代池；
- Any-to-Bokeh 内部 `z_bokeh...ModuleRenderRT` 未公开，因此使用论文明确引用且接口吻合的 MPIB 官方实现；
- 当前素材用于 reproduction / development，与未来保留测试池保持源 ID 隔离。

这些差异均已固定记录，不影响本轮“公开代理复现”的目标定义，也不会在看到模型成绩后反向修改。

---

## 十一、参考资料与证据

### 项目记录

- 配置：`configs/any2bokeh-mpib-public-proxy.json`
- 启动记录：`03_实验记录/records/any2bokeh-reproduction-200-start-20260928.json`
- Renderer probe：`03_实验记录/records/public-ray-renderer-probe-20260928.json`
- 全量结果记录：`03_实验记录/records/any2bokeh-mpib-public-proxy-200-20260928.json`（200/200 完成后生成）
- 方法论：`03_实验记录/当前实验进度与方法论.md`
- 评价协议：`03_实验记录/Benchmark与Baseline统一评价协议.md`

### 外部依据

- Any-to-Bokeh 论文 Table 1
- Any-to-Bokeh Appendix A.2
- Any-to-Bokeh 公开 `dataset_synthsis.py`
- Any-to-Bokeh 公开 `metrics/vid_metrics.py`
- MPIB 官方 `bokeh_generator.py::ModuleRenderRT`

---

**最后更新**：2026-09-28  
**状态**：200-video 全量任务运行中；阶段性指标已显示与论文高度一致，最终值及 author-public metrics 将在 200/200 完成后补充。
