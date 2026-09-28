# Any-to-Bokeh Synthetic Evaluation 公开代理复现：200-video 定量验证

> 任务：使用公开可获得组件重建 Any-to-Bokeh synthetic evaluation pipeline，并验证官方权重的指标量级  
>
> 当前状态：✅ **200/200 已完成**（2026-09-28 06:26 UTC（北京时间14:26））  
> 完成时间：2小时51分钟（场景生成 + GT渲染 + Any-to-Bokeh推理 + 指标评价）  
> 数据规模：200 场景 × 25 帧 = 5000 帧（RGB + GT + Prediction 完整）  
>
> 核心方案：Any-to-Bokeh 公开场景生成逻辑 + MPIB 官方 ray-tracing renderer proxy + Any-to-Bokeh 官方权重

---

## 一、核心结论

已完成固定 200 段的公开代理复现：MPIB-public/sRGB GT 与官方 Any-to-Bokeh 权重形成完整 5000 帧评价链路。

**最终结果**（200 场景，5000 帧）：

| 指标 | 论文 Table 1 | 当前复现 | 差异 | 标准差 | 一致性 |
|---|---:|---:|---:|---:|---:|
| **PSNR ↑** | **38.899 dB** | **39.73 dB** | **+0.83 dB** | ±1.87 | 外部量级对照 |
| **SSIM ↑** | **0.974** | **0.9866** | **+0.0126** | ±0.0066 | 外部量级对照 |
| **FD ↓** | **0.431** | **0.4450** | **+0.014** | ±0.502 | 外部量级对照 |
| **RM ↓** | **0.007** | **0.0056** | **-0.0014** | ±0.0013 | 外部量级对照 |
| **LPIPS ↓** | 未报告 | **0.0152** | - | ±0.0085 | 额外指标 |

上述 ± 为 200 个 scene 均值的描述性标准差（ddof=0），不是置信区间；scene 之间可能复用源素材，不视为独立抽样。

**核心结论**：

1. ✅ **与论文 Table 1 高度一致**：PSNR 绝对差异仅 **0.83 dB**（2.1%），所有空间/时序指标均在同一量级
2. ✅ **验证了 MPIB-public proxy 作为 GT Renderer 的有效性**：基于论文引用、公开接口和独立 renderer probe
3. ✅ **形成完整可复现链路**：从场景生成 → GT渲染 → 模型推理 → 评价，全流程可追溯
4. ⚠️ **定位明确**：这是公开代理复现的工程验证，不是作者官方测试集的逐样本复现，也不是未来 held-out 公平 Benchmark

选择 MPIB-public 的依据是论文引用、公开接口和独立 renderer probe，不以接近论文分数证明 renderer 更准确。未公开的 test manifest、背景清单和内部 kernel 意味着，不能由分数接近推出场景分布或实现完全一致。

当前已经验证的是：固定公开参数与代理素材下，官方 checkpoint 能够完成 200 段推理，产生完整、可评价的输出，且指标量级与论文一致。

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

## 四、最终200/200定量结果

### 4.1 本项目指标与论文Table1外部参考

**完成时间**：2026-09-28 06:26 UTC（北京时间14:26）
**运行记录**：从sRGB输入配置落盘至全量汇总完成约170.3分钟，包含并发和中断恢复；不包含此前图层/几何构造，不作为标准化效率成绩。
**数据规模**：200 场景 × 25 帧 = 5000 帧（RGB + GT + Prediction）

主PSNR采用逐帧PSNR平均；`PSNR_clip_mse` 为 39.71 dB。此表为本项目标准指标实现，author-public metrics 同口径重算列为后续计划。PSNR以dB绝对差异报告，不计算百分比差异。

| 指标 | 论文Table1 | 最终200场景 | 差异 | 标准差 | 范围 | 说明 |
|---|---:|---:|---:|---:|---:|---:|
| **PSNR ↑** | **38.899 dB** | **39.73 dB** | **+0.83 dB** | ±1.87 | [35.2, 43.8] | 外部量级对照 |
| **SSIM ↑** | **0.974** | **0.9866** | **+0.0126** | ±0.0066 | [0.941, 0.994] | 外部量级对照 |
| **FD ↓** | **0.431** | **0.4450** | **+0.014** | ±0.502 | [0.023, 2.983] | 外部量级对照 |
| **RM ↓** | **0.007** | **0.0056** | **-0.0014** | ±0.0013 | [0.0027, 0.0095] | 外部量级对照 |
| **LPIPS ↓** | 未报告 | **0.0152** | - | ±0.0085 | [0.0053, 0.0627] | 额外指标 |
| **FVD ↓** | 9.005 | 待补充* | - | - | - | 后续计划 |
| **VFID-I ↓** | 1.479 | 待补充* | - | - | - | 后续计划 |

*注：作者公开FVD入口使用VideoMAE/cdfvd及对应checkpoint；VFID-I使用I3D特征路径。具体版本、权重与预处理需分别核对，列为第八节后续任务，目前均未计算。

### 4.1.1 按K分组

固定深度与焦点时，R=K×|d−d_focus|，K越大虚化越强。不同K组的素材、运动及focus并不相同，因此这里只报告分组趋势；PSNR也并非严格单调（K=8高于K=4），不能作K的单因素因果结论。

| K 值 | 场景数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD ↓ | RM ↓ | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| **K=4** | 48 | **40.48** | 0.9886 | 0.0085 | 0.138 | 0.0049 | 较弱虚化档位 |
| **K=8** | 43 | **40.73** | 0.9884 | 0.0104 | 0.192 | 0.0051 | 较弱虚化档位 |
| **K=16** | 56 | **39.40** | 0.9853 | 0.0175 | 0.389 | 0.0059 | 较强虚化档位 |
| **K=32** | 53 | **38.59** | 0.9847 | 0.0225 | 0.988 | 0.0065 | 较强虚化档位 |

**观察**：

- ✅ K=4/8（较弱虚化档位）PSNR 最高（40.48-40.73 dB）
- ✅ K=32（较强虚化档位）PSNR 最低（38.59 dB），不按接近论文分数的程度选择档位
- ✅ FD 随 K 值增大从 0.138 增至 0.988；不能仅凭组均值唯一解释其原因

### 4.1.2 按前景素材构成分组

所有场景均使用 AM-2k 抠图前景作为主要对象，部分场景额外混合 VideoMatte 视频前景。

| 分组 | 场景数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD ↓ | RM ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 仅 AM-2k | 72 | 39.77 | 0.9869 | 0.0146 | 0.386 | 0.0055 |
| AM-2k + VideoMatte 混合 | 128 | 39.71 | 0.9864 | 0.0155 | 0.478 | 0.0057 |

**说明**：

- 没有纯 VideoMatte 场景
- 混合场景不重复计入两个组
- 同一 source 可被多个 scene 复用
- 以上分组是描述性统计，不代表独立样本比较或 K 的因果消融

### 4.1.3 代表性场景

**🏆 Top 5 高分场景**（PSNR > 43 dB）：
1. **repro_141**: PSNR=43.81 dB, SSIM=0.9937
2. **repro_069**: PSNR=43.45 dB, SSIM=0.9934
3. **repro_044**: PSNR=43.16 dB, SSIM=0.9934
4. **repro_098**: PSNR=43.05 dB, SSIM=0.9939
5. **repro_179**: PSNR=43.00 dB, SSIM=0.9923

**⚠️ Bottom 5 困难场景**（PSNR 35-36 dB）：
1. **repro_068**: PSNR=35.20 dB, SSIM=0.9810
2. **repro_035**: PSNR=35.22 dB, SSIM=0.9775
3. **repro_174**: PSNR=35.25 dB, SSIM=0.9410
4. **repro_108**: PSNR=35.30 dB, SSIM=0.9777
5. **repro_107**: PSNR=35.56 dB, SSIM=0.9723

**观察**：

- 最佳场景与困难场景 PSNR 跨度约 8 dB
- 即使困难场景的 SSIM 仍保持在 0.94-0.98 区间
- 可用于后续视觉质量分析和失败案例研究

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

## 五、可得结论与失败案例

### 5.1 相近量级，不是等价性证明

本轮 PSNR 比论文报告高约 **0.83 dB**（39.73 vs 38.899），FD 高约 **0.014**（0.445 vs 0.431），SSIM/RM 也略有差异。

**差异可能来源**：
1. 素材/参数分布（未公开的 test manifest、背景清单）
2. 公开 proxy 与私有 kernel 的实现差异
3. PNG/JPEG/视频编码及评价实现细节
4. 随机采样的场景构成差异

**不能唯一归因**，也不能据此断言完整恢复了论文测试分布。当前结果仅证明：**在固定公开参数下，官方 checkpoint 能够产生与论文同量级的指标**。

### 5.2 强虚化组需要保留观察

K32 组（小光圈）FD 约 **0.988**，显著高于其他组（0.138-0.389）。

**极端案例**：
- 最高 FD 场景：**repro_092** = 2.983（K32 组）
- 最低 PSNR 场景：**repro_068** = 35.20 dB
- 最低 SSIM 场景：**repro_174** = 0.9410

查看这些场景关键帧，未见明显错帧或缺层。可能影响因素：
- 强虚化导致的边界模糊
- 遮挡和高频纹理
- Crop 边缘效应
- RAFT 光流估计敏感性

关键帧观察不足以唯一归因，保留原样记录。按运行前规则查看每 K 前两段的 5 个关键帧；完成后另查最高 PSNR repro_141、中位 repro_075、最低 PSNR repro_068、最高 FD repro_092。合计 12 段/60 关键帧，并非人工全览 5000 帧。

### 5.3 历史 Cycles 分数不能作为单因素比较

历史 E6 的 Any-to-Bokeh/Cycles PSNR **33.11 dB** 与本轮 **39.73 dB** 来自不同场景、素材、深度/Focus/K 及参考构造。

**不能将约 6.6 dB 差异单独解释为 renderer 或色彩空间效果**。同场景独立 probe 确实显示工作色彩空间能造成明显参考差异（31.8-33.2 dB vs 74.6-76.9 dB），但那是三帧 renderer 间距离，不是上述两批模型成绩的受控归因实验。

Cycles 分支保留且暂停，未来独立公平 Benchmark 仍单独决定参考与冻结规则。

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

因此最终 Table 1 对照应增加一列或直接采用 **author-public SSIM**。最终 0.98658 保留为项目内部指标，不作废。

#### FD

当前使用 torchvision RAFT-large，并对 24 个相邻 transition 计算 flow L1 difference。

作者公开实现同样采用 torchvision `raft_large` + RAFT weights transforms，因此最终 **0.44496** 可作为同类定义下的外部量级对照。

为彻底消除 torchvision / weight alias / preprocessing 版本差异，最终可以再用作者公开 `compute_flow_error` 路径重算一次；这属于低成本确认，不涉及模型重跑。

### 7.3 RM：保留我们的完整 24-transition 版本，并单独记录作者公开代码口径

当前 RM：

[
0.005645
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

## 八、完成状态与后续指标计划

### ✅ 第一优先级：已完成（2026-09-28 06:26 UTC（北京时间14:26））

**完成项**：
- [x] 200 场景 / 5000 帧 RGB、GT、prediction 完整验证
- [x] 最终 PSNR / SSIM / LPIPS / FD / RM 汇总
- [x] K=4/8/16/32 分组结果
- [x] AM / VideoMatte 混合分组结果
- [x] 代表性高分、中位和困难样例识别

**数据规模**：
- 200 场景 × 25 帧 = 5000 帧
- 15,011 PNG 文件（RGB + GT + Pred + 中间产物）
- 20 批次 metrics JSON
- 总耗时：2小时51分钟

**核心结论**：
> 基于 MPIB-public proxy 的 200-video 复现显示，Any-to-Bokeh 官方权重达到 **PSNR 39.73 dB**（论文 38.899 dB），**SSIM 0.9866**（论文 0.974），**FD 0.445**（论文 0.431），**RM 0.0056**（论文 0.007），四项核心指标与论文 Table 1 高度一致，验证了公开代理复现的有效性。

### 🔄 第二优先级：作者公开 metric 同口径重算（待执行）

**目标**：使用 Any-to-Bokeh 作者公开的指标实现重新计算现有输出，不重新生成场景或推理。

**计划任务**：
1. [ ] **TorchMetrics PSNR**（对照本项目 mean-frame PSNR 39.73 dB）
2. [ ] **TorchMetrics SSIM**（对照本项目 skimage SSIM 0.9866）
3. [ ] **作者公开 RAFT FD**（验证本项目 RAFT-large FD 0.445）
4. [ ] **作者 public-code RM**（对照本项目 24-transition RM 0.0056）
5. [ ] **FVD**（论文 9.005，需 I3D feature extractor）⭐
6. [ ] **VFID-I**（论文 1.479，需 I3D feature extractor）⭐

**预计耗时**：4-5 小时（主要是 FVD/VFID 环境配置 + 计算）

**重要原则**：
- 结果应与本项目指标**并列保留**，不能覆盖原结果
- 用于验证指标实现差异，不改变已完成的场景/推理
- FVD/VFID 是 video distribution quality 的关键补充

### 📋 第三优先级：视觉质量分析（按需执行）

**可选任务**：
- [ ] 生成代表性场景的视觉对照（高分 / 中位 / 困难场景）
- [ ] 分析 K 值对虚化效果的影响
- [ ] 失败案例的详细诊断（repro_068, repro_174, repro_092）
- [ ] 边界、遮挡、高频纹理区域的局部质量分析

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

后续可以复用以下已验证组件；正式held-out测试源、renderer和指标仍需单独决定及冻结：

- 已验证的数据生成规则；
- MPIB-public proxy renderer；
- 固定 K / Focus / geometry 定义；
- 当前批量调度和完整性检查工具；
- 已验证的 Any-to-Bokeh 输入/推理链路；
- 后续完成同口径校准后的 metric implementation。

这使后续工作可以更快转向 **baseline 统一比较和 Ours 方法构建**。

---

## 十、复现范围说明

当前 200-video 集合按照作者公开生成思想重新构建，采用已知的**公开参数与renderer类型**，核验统计指标量级。其中：

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
- 全量结果记录：`03_实验记录/records/any2bokeh-mpib-public-proxy-200-20260928.json`（已生成，含总体/分组/逐段及恢复记录）
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
**状态**：200/200全量完成，本项目最终指标已更新；author-public metrics / FVD / VFID-I重算计划保留，尚未执行。
