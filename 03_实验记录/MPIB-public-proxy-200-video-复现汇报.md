# Any-to-Bokeh Synthetic Evaluation 公开代理复现：200-video 定量验证

> 任务：使用公开可获得组件重建 Any-to-Bokeh synthetic evaluation pipeline，并验证官方权重的指标量级
>
> 当前状态：✅ **200/200 已完成**（2026-09-28 06:26 UTC（北京时间14:26））
> 运行记录：sRGB输入配置落盘至全量汇总约170.3分钟，含并发和中断恢复，不含此前图层/几何构造。
> 数据规模：200 场景 × 25 帧 = 5000 帧（RGB + GT + Prediction 完整）
> 补充指标：2026-09-28完成TorchMetrics PSNR/SSIM、VideoMAE FVD、I3D VFID-I，见下表及第七节。
>
> 核心方案：Any-to-Bokeh 公开场景生成逻辑 + MPIB 官方 ray-tracing renderer proxy + Any-to-Bokeh 官方权重

---

## 一、核心结论

已完成固定 200 段的公开代理复现：MPIB-public/sRGB GT 与官方 Any-to-Bokeh 权重形成完整 5000 帧评价链路。

**论文原文与本次结果逐项对照**（200场景，差值统一为“本次−论文”）：

| 指标 | 论文Table 1 | 本次结果 | 差值（本次−论文） | 结果来源 |
|---|---:|---:|---:|---|
| PSNR ↑ / dB | 38.899 | **39.711601** | +0.812601 | 本轮：作者公开函数 / TorchMetrics |
| SSIM ↑ | 0.974 | **0.986305** | +0.012305 | 本轮：作者公开函数 / TorchMetrics |
| FVD ↓ | 9.005 | **8.937519** | -0.067481 | 本轮：VideoMAE，200段集合 |
| VFID-I ↓ | 1.479 | **0.808557** | -0.670443 | 本轮：I3D，400个连续10帧窗口 |
| FD ↓ | 0.431 | **0.444956** | +0.013956 | 沿用原项目RAFT结果，未重算作者入口 |
| RM ↓ | 0.007 | **0.005645** | -0.001355 | 沿用原项目24-transition结果 |
| LPIPS ↓ | 未报告 | **0.015153** | 不适用 | 项目额外指标，论文Table 1未报告 |

↑指标的正差、↓指标的负差仅表示本次数值更优；由于测试素材、proxy GT及公开指标适配存在差异，**不能解释为同一测试条件下优于论文**。论文数值来自[原文v2 Table 1](https://arxiv.org/html/2505.21593v2#S4.T1)。

PSNR/SSIM主对照采用本轮TorchMetrics补算；原逐帧PSNR **39.73197 dB**、skimage SSIM **0.986575**继续保留在第四节。FVD/VFID-I为集合级距离，不填写逐scene标准差或范围。

**核心结论**：

1. ✅ **与论文Table 1逐项对照**：本轮TorchMetrics PSNR差 **+0.81260 dB**，SSIM差 **+0.012305**，FVD差 **-0.06748**，VFID-I差 **-0.67044**；这些是实际数值差，不能当作测试条件等价的证明
2. ✅ **验证了 MPIB-public proxy 的运行可行性**：论文引用、公开接口和独立 renderer probe 支持代理实现的选择，不能证明其等同于作者私有 renderer
3. ✅ **形成完整可复现链路**：从场景生成 → GT渲染 → 模型推理 → 评价，全流程可追溯
4. ⚠️ **定位明确**：这是公开代理复现的工程验证，不是作者官方测试集的逐样本复现，也不是未来 held-out 公平 Benchmark

选择 MPIB-public 的依据是论文引用、公开接口和独立 renderer probe，不以接近论文分数证明 renderer 更准确。未公开的 test manifest、背景清单和内部 kernel 意味着，不能由分数接近推出场景分布或实现完全一致。

当前已经验证的是：固定公开参数与代理素材下，官方checkpoint能完成200段推理和本轮四项补算。数值接近论文不证明恢复了官方测试集、私有renderer或历史评价环境；VFID-I更低也不能解释为模型优于论文。

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

### 4.1 保留的项目诊断指标与新增集合指标

**完成时间**：2026-09-28 06:26 UTC（北京时间14:26）
**运行记录**：从sRGB输入配置落盘至全量汇总完成约170.3分钟，包含并发和中断恢复；不包含此前图层/几何构造，不作为标准化效率成绩。
**数据规模**：200 场景 × 25 帧 = 5000 帧（RGB + GT + Prediction）

下面保留原五项诊断口径，其中PSNR为逐帧平均，`PSNR_clip_mse`为39.71159 dB。与论文的主对照表见第一节，采用本轮TorchMetrics PSNR/SSIM；这里的旧结果不被覆盖。±为原200个scene均值的描述性标准差（ddof=0），不是置信区间，scene可能共享素材。FVD/VFID-I不适用逐scene统计。PSNR仅报告dB绝对差异。

| 指标 | 论文Table1 | 最终200场景 | 差异 | 标准差 | 范围 | 说明 |
|---|---:|---:|---:|---:|---:|---:|
| **PSNR ↑** | **38.899 dB** | **39.73 dB** | **+0.83 dB** | ±1.87 | [35.2, 43.8] | 外部量级对照 |
| **SSIM ↑** | **0.974** | **0.9866** | **+0.0126** | ±0.0066 | [0.941, 0.994] | 外部量级对照 |
| **FD ↓** | **0.431** | **0.4450** | **+0.014** | ±0.502 | [0.023, 2.983] | 外部量级对照 |
| **RM ↓** | **0.007** | **0.0056** | **-0.0014** | ±0.0013 | [0.0027, 0.0095] | 外部量级对照 |
| **LPIPS ↓** | 未报告 | **0.0152** | - | ±0.0085 | [0.0053, 0.0627] | 额外指标 |
| **FVD ↓（VideoMAE）** | 9.005 | **8.93752** | -0.06748 | 不适用 | 不适用 | 200段集合级距离 |
| **VFID-I ↓（I3D）** | 1.479 | **0.80856** | -0.67044 | 不适用 | 不适用 | 400个10帧片段集合级距离 |

补充对照（同一200段、原始1024×576 PNG）：

| 指标 | 原项目口径 | 作者公开函数 / TorchMetrics 1.9.0 | 差异解释 |
|---|---:|---:|---|
| PSNR | 39.73197 dB（逐帧平均） | **39.71160 dB** | 作者按整段MSE计算再按scene平均；独立的PSNR_clip_mse=39.71159，与新值仅差0.00001 dB |
| SSIM | 0.986575（skimage） | **0.986305** | 同一图像，差异来自评价实现/边界处理；不表示模型变化 |

FVD使用VideoMAE/cdfvd，VFID-I使用I3D，不能都标为I3D。两者不是逐视频分数平均；模型权重、样本计数及PNG适配详见第七节。

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

以本轮补算口径比较：PSNR **39.71160 vs 38.899**（差+0.81260 dB）、SSIM **0.986305 vs 0.974**（差+0.012305）、FVD **8.93752 vs 9.005**（差-0.06748）、VFID-I **0.80856 vs 1.479**（差-0.67044）。FD/RM沿用原结果，完整差值见第一节。FVD接近原文、VFID-I较低均是本次测量事实，不说明代理测试集与原测试集等价。

**差异可能来源**：
1. 素材/参数分布（未公开的 test manifest、背景清单）
2. 公开 proxy 与私有 kernel 的实现差异
3. PNG/JPEG/视频编码及评价实现细节
4. 随机采样的场景构成差异

**不能唯一归因**，也不能据此断言完整恢复了论文测试分布。当前结果仅证明：**在固定公开参数下，官方 checkpoint 能够产生与论文同量级的指标**。

### 5.2 强虚化组需要保留观察

K32 组（固定深度与焦点时虚化更强）FD 约 **0.988**，数值高于其他组（0.138-0.389）；此处不作统计显著性判断。

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

## 七、作者公开指标补算：已完成及适配边界

### 7.1 本轮实际范围

200段GT/prediction保持原样，只读取已有PNG，未重新生成场景、渲染GT或推理Any-to-Bokeh。完成TorchMetrics PSNR/SSIM、VideoMAE FVD、I3D VFID-I；原五项诊断指标和按K/素材分组全部保留。

证据：`records/any2bokeh-author-public-metrics-200-20260928.json`；代码：`scripts/evaluate_author_public_metrics.py`。原始特征与阶段JSON保存在本机`outputs/author-public-metrics-20260928/`。

### 7.2 TorchMetrics PSNR/SSIM

直接执行Any-to-Bokeh提交`f7a16959`中`metrics/vid_metrics.py`的`psnr`、`ssim`函数体。上游模块顶层依赖未发布的`metrics.utils`和遗留`z_train`路径，因此仅按AST提取两函数及所需依赖；没有改动计算公式。

使用现有TorchMetrics **1.9.0**，`data_range=1.0`，batch_size=1 clip，25帧展开后计算，每段等权汇总。RGB PNG除以255，无缩放、无重新编码。结果为 **39.71160 dB / 0.986305**。PSNR与独立整段MSE结果差 **0.00000967 dB**，验证聚合一致。作者未锁定的历史依赖版本不能由本次结果反推。

### 7.3 FVD：VideoMAE，200段各计一次

调用cdfvd提交`a1e037ab`的VideoMAE模型、预处理、FeatureStats及Frechet距离公式；FP32，关闭TF32，seed=42，batch_size=16。

- 权重：OpenGVLab官方`VideoMAE2/mae-g/vit_g_hybrid_pt_1200e_ssv2_ft.pth`，约2.02GB。
- 先按公开video预处理缩放短边至256并中心裁剪，再按公开特征预处理缩放224×224；保留其byte转换和[0,1]输入口径。
- 200个GT和200个prediction分别得到 **200×1408** 特征，每段输入25帧。公开模型tubelet_size=2，末尾无法配对的一帧不会进入特征。
- 显式使用`n_real='full', n_fake='full'`。cdfvd默认n_fake=2048会重复这200段；本次不把重复采样当作新增样本。
- 使用原PNG解码替代MP4解码，避免额外有损编码；这与作者MP4入口存在明确适配差异。

最终 **FVD=8.93752**。这是VideoMAE特征空间中的集合距离，不是传统I3D-FVD，也不是200个逐scene距离的平均。

### 7.4 VFID-I：作者I3D内核，保持scene边界

使用作者指定的`zhengchong/VFID/i3d.pt`（约51MB），权重严格加载，无missing/unexpected keys；复用其`InceptionI3d.extract_features`、`ImageSequenceDataset`图像变换和`calculate_fid`。

- `sequence_length=10`、224×224、RGB、输入[-1,1]，batch_size=4。
- 每段取0–9、10–19两个连续不重叠窗口；最后5帧不参与本项。200段对应 **400×1024** 特征/侧。
- 公开图片loader会将多目录帧展平再切片，可能跨scene；本项目对每段分别构造公开dataset，再合并dataset。
- 公开VideoDataset的累计索引减法会产生负帧索引，且OpenCV路径未做BGR→RGB；本次使用其RGB图片路径，绕开该问题。

最终 **VFID-I=0.80856**。这是公开模型/公式下、明确采样适配后的结果，不能声称原封不动复刻作者视频loader或论文运行环境。相邻窗口共享源视频，也不能称为400个独立视频样本。

### 7.5 检查、运行成本与保留项

GT/预测场景ID与固定manifest一致；FVD两侧200×1408、VFID-I两侧400×1024特征全部有限。GT与自身比较距离接近0（浮点误差绝对值小于0.001）；已从保存特征独立重算FVD，与正式输出一致。

正式三阶段合计 **337.4秒（约5.6分钟）**，不含下载、小测试和文档整理；最高PyTorch allocated显存 **23.63 GiB**。该计时仅为本机指标补算成本，不是论文中的Any-to-Bokeh模型推理效率。

FD **0.44496**、完整24-transition RM **0.005645**及LPIPS继续使用原项目实现。作者FD/RM原入口本轮未重算，未将其标成完成；公开RM存在按通道数读取时序长度的问题，若后续核对应另列，不能覆盖现有完整24-transition结果。VFID-ResNeXt不属于本轮VFID-I任务，未下载该权重或计算。

### 7.6 环境、权重与复跑入口

复用`magicbokeh`（Python 3.10、torch 2.8.0+cu128），仅补装cd-fvd 0.1.0的固定Git版本（`--no-deps`），没有升级其他包。依赖记录在`configs/requirements-author-public-metrics.txt`。

| 用途 | 本机权重路径 | 来源 |
|---|---|---|
| VFID-I | `models/metrics/VFID/i3d.pt` | 作者指定zhengchong/VFID，revision `dca26ff3` |
| FVD | `models/metrics/VideoMAE2/mae-g/vit_g_hybrid_pt_1200e_ssv2_ft.pth` | OpenGVLab/VideoMAE2，revision `706cc172` |

```bash
conda activate magicbokeh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
python scripts/evaluate_author_public_metrics.py \
  --output outputs/author-public-metrics-20260928
```

上述目录已有完整结果，再运行会校验协议并复用。独立重复计算时指定新的output目录。数据、权重、特征数组和大型日志不进入Git。

## 八、完成状态与后续讨论

本轮用户要求的四项已完成并填表：

| 补充项 | 状态 | 最终值 |
|---|---|---:|
| TorchMetrics PSNR | 200段完成 | 39.71160 dB |
| TorchMetrics SSIM | 200段完成 | 0.986305 |
| FVD / VideoMAE | 200段集合，适配已声明 | 8.93752 |
| VFID-I / I3D | 400个10帧窗口，保持scene边界 | 0.80856 |

本轮完成不改变原有复现边界：测试素材、proxy GT、PNG编码及部分loader/样本计数与作者入口不同。FVD接近论文9.005、VFID-I低于论文1.479，只能作为外部参考，不能证明官方测试指标被严格复现。

后续可讨论作者FD/RM口径核对、困难案例分析和其他baseline统一比较；这些不是本轮已完成内容。未来公平Benchmark仍需单独确定源划分、renderer与指标，Cycles及AM validation恢复线保持暂停。

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
- 全量结果记录：`03_实验记录/records/any2bokeh-mpib-public-proxy-200-20260928.json`（原五项指标及恢复记录）
- 本轮补算：`03_实验记录/records/any2bokeh-author-public-metrics-200-20260928.json`（TorchMetrics、FVD、VFID-I及适配/验证）
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

补算实现及权重来源：[作者公开指标代码](https://github.com/vivoCameraResearch/any-to-bokeh/blob/f7a16959dd219080fb8eec0d32bb9602d8970c22/metrics/vid_metrics.py)、[cdfvd固定版本](https://github.com/songweige/content-debiased-fvd/tree/a1e037ab7cb087debd2221d14ae4a001ec054201)、[作者指定I3D权重](https://huggingface.co/zhengchong/VFID)、[官方VideoMAE2权重](https://huggingface.co/OpenGVLab/VideoMAE2)。
