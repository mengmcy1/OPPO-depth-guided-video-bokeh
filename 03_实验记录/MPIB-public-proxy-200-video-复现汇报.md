# Any-to-Bokeh Synthetic Evaluation 公开代理复现：200-video 定量验证

> 任务：使用公开可获得组件重建 Any-to-Bokeh synthetic evaluation pipeline，并验证官方权重的指标量级  
> 当前状态：✅ **200/200 全部完成**（2026-09-28 14:25）；RGB、GT、prediction 各 5000 帧及最终完整性检查通过  
> 核心方案：Any-to-Bokeh 公开场景生成逻辑 + MPIB 官方 ray-tracing renderer proxy + Any-to-Bokeh 官方权重

---

## 一、核心结论

已完成固定 200 段的公开代理复现：MPIB-public/sRGB GT 与官方 Any-to-Bokeh 权重形成完整 5000 帧评价链路。

**最终结果**（200 场景，5000 帧）：
- **PSNR**: **39.73 dB** ±1.87（论文 38.899 dB，差异 +0.83 dB）
- **SSIM**: **0.9866** ±0.0066（论文 0.974，差异 +0.0126）
- **LPIPS**: **0.0152** ±0.0085（论文未报告）
- **FD**: **0.4450** ±0.5016（论文 0.431，差异 +0.014）
- **RM**: **0.0056** ±0.0013（论文 0.007，差异 -0.0014）

**核心结论**：
1. ✅ **与论文 Table 1 高度一致**：PSNR 差异仅 0.83 dB（2.1%），所有指标均在同一量级
2. ✅ **验证了 MPIB-public proxy 作为 GT 的合理性**：基于论文引用、公开接口和独立 renderer probe
3. ⚠️ **定位明确**：这是公开代理复现的工程验证，不是作者官方测试集的逐样本复现，也不是未来 held-out 公平 Benchmark

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

主PSNR采用逐帧PSNR平均；`PSNR_clip_mse` 为 39.7116 dB。此表尚不标作 author-public metrics；PSNR以dB绝对差异报告。

| 指标 | 论文Table1 | 最终200段 | 差异 |
|---|---:|---:|---:|
| PSNR ↑ | 38.899 dB | 39.7320 dB | +0.8330 dB |
| SSIM ↑ | 0.974 | 0.98658 | +0.01258 |
| FD ↓ | 0.431 | 0.44496 | +0.01396 |
| RM ↓ | 0.007 | 0.005645 | -0.001355 |
| LPIPS ↓ | 未报告 | 0.01515 | 不适用 |
| FVD ↓ | 9.005 | 未计算 | 后续计划 |
| VFID-I ↓ | 1.479 | 未计算 | 后续计划 |

### 4.1.1 按K分组

| 分组 | 段数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD ↓ | RM ↓ |
|---|---:|---:|---:|---:|---:|---:|
| K=4 | 48 | 40.4803 | 0.98857 | 0.00851 | 0.13775 | 0.004893 |
| K=8 | 43 | 40.7321 | 0.98835 | 0.01044 | 0.19174 | 0.005133 |
| K=16 | 56 | 39.3991 | 0.98526 | 0.01753 | 0.38908 | 0.005885 |
| K=32 | 53 | 38.5945 | 0.98471 | 0.02248 | 0.98766 | 0.006487 |

### 4.1.2 按前景素材构成分组

| 分组 | 段数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD ↓ | RM ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 仅 AM-2k | 72 | 39.7696 | 0.98694 | 0.01458 | 0.38635 | 0.005545 |
| AM-2k + VideoMatte 混合 | 128 | 39.7108 | 0.98637 | 0.01547 | 0.47792 | 0.005701 |

没有纯VideoMatte场景。混合场景不重复计入两个组；同一source可被多个scene复用。以上分组是描述性统计，不代表独立样本比较或K的因果消融。

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

本轮PSNR比论文报告高约0.833dB，FD高约0.01396；同时保留SSIM、RM及额外LPIPS结果。差异可能来自素材/参数分布、公开proxy与私有kernel、PNG/JPEG/视频编码及评价实现，不能唯一归因，也不能据此断言完整恢复了论文测试分布。

### 5.2 强虚化组需要保留观察

K32组FD约0.98766，高于其他组。最高FD场景repro_092为2.98291，最低PSNR场景repro_068为35.1970dB，均原样保留。查看这些场景关键帧，未见明显错帧或缺层；强虚化、遮挡、高频纹理、crop边缘以及RAFT估计敏感性可能影响结果，关键帧观察不足以唯一归因。

按运行前规则查看每K前两段的5个关键帧；完成后另查最高PSNR repro_141、中位repro_075、最低PSNR repro_068、最高FD repro_092。合计12段/60关键帧，并非人工全览5000帧。

### 5.3 历史Cycles分数不能作为单因素比较

历史E6的Any-to-Bokeh/Cycles PSNR33.1084dB与本轮39.7320dB来自不同场景、素材、深度/Focus/K及参考构造。不能将约6.6dB差异单独解释为renderer或色彩空间效果。

同场景独立probe确实显示工作色彩空间能造成明显参考差异，但那是三帧renderer间距离，不是上述两批模型成绩的受控归因实验。Cycles分支保留且暂停，未来独立公平Benchmark仍单独决定参考与冻结规则。

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

### 第一优先级：已完成最终成绩与检查

1. 验证 200 场景 / 5000 帧 RGB、GT、prediction 完整；
2. 更新最终 PSNR / SSIM / LPIPS / FD / RM；
3. 输出 K=4/8/16/32 分组结果；
4. 输出 AM / VideoMatte 分组结果；
5. 检查代表性高分、中位和困难样例。

### 第二优先级：作者公开 metric 同口径重算（待执行）

直接使用现有输出，不重新跑模型：

1. TorchMetrics PSNR；
2. TorchMetrics SSIM；
3. 作者公开 RAFT FD；
4. 作者 public-code RM 对照；
5. FVD；
6. VFID-I。

第一优先级已完成：200段/各5000帧、20批指标、按K及互斥素材构成分组与代表样例均已核对。第二优先级仍待执行，结果应与本项目指标并列保留，不能覆盖原结果。

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
