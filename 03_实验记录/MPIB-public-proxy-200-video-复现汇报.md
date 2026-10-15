# Any-to-Bokeh Synthetic Evaluation 公开代理复现：200-video 定量验证

> 任务：使用公开可获得组件重建 Any-to-Bokeh synthetic evaluation pipeline，并验证官方权重的指标量级
>
> 当前状态：✅ Synthetic七个方法各200×25及全部指标完成；原六种配置保持冻结，2026-09-30新增MagicBokeh的Target-guided Depth-Medoid B与同配置Table 3A。Any-to-Bokeh首轮模块分析完成，下一步验证VAE/缓存效率机制及系统归纳失败场景。
> 运行记录：sRGB输入配置落盘至全量汇总约170.3分钟，含并发和中断恢复，不含此前图层/几何构造。
> 数据规模：200 场景 × 25 帧 = 5000 帧（RGB + GT + Prediction 完整）
> 补充指标：2026-09-28完成TorchMetrics PSNR/SSIM、VideoMAE FVD、I3D VFID-I，见下表及第七节。
>
> 核心方案：Any-to-Bokeh 公开场景生成逻辑 + MPIB 官方 ray-tracing renderer proxy + Any-to-Bokeh 官方权重
> Baseline：七个方法的主表与论文外部参考见1.1；原六方法候选、选择依据与失败案例见8.1–8.7，MagicBokeh详见中央进度及阶段归档。中间方法排序未完全恢复，不宣称整体性能版图已对齐。

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
| LPIPS ↓ | 0.019（ICLR2026版） | **0.015153** | -0.003847 | 项目lpips/squeeze；正式版新增列，未声称逐值复刻作者入口 |

↑指标的正差、↓指标的负差仅表示本次数值更优；由于测试素材、proxy GT及公开指标适配存在差异，**不能解释为同一测试条件下优于论文**。原六项论文数值来自[arXiv v2 Table1](https://arxiv.org/html/2505.21593v2#S4.T1)；2026-09-29独立核对[ICLR2026正式版Table1](https://proceedings.iclr.cc/paper_files/paper/2026/file/a0dca56ad0aceeb4eab4d35630e9a0ed-Paper-Conference.pdf#page=7)后补入LPIPS。正式版其余这些数值相同，历史实验与选择依据不改。

PSNR/SSIM主对照采用本轮TorchMetrics补算；原逐帧PSNR **39.73197 dB**、skimage SSIM **0.986575**继续保留在第四节。FVD/VFID-I为集合级距离，不填写逐scene标准差或范围。

**核心结论**：

1. ✅ **与论文Table 1逐项对照**：本轮TorchMetrics PSNR差 **+0.81260 dB**，SSIM差 **+0.012305**，FVD差 **-0.06748**，VFID-I差 **-0.67044**；这些是实际数值差，不能当作测试条件等价的证明
2. ✅ **验证了 MPIB-public proxy 的运行可行性**：论文引用、公开接口和独立 renderer probe 支持代理实现的选择，不能证明其等同于作者私有 renderer
3. ✅ **形成完整可复现链路**：从场景生成 → GT渲染 → 模型推理 → 评价，全流程可追溯
4. ⚠️ **定位明确**：这是公开代理复现的工程验证，不是作者官方测试集的逐样本复现，也不是未来 held-out 公平 Benchmark

选择 MPIB-public 的依据是论文引用、公开接口和独立 renderer probe，不以接近论文分数证明 renderer 更准确。未公开的 test manifest、背景清单和内部 kernel 意味着，不能由分数接近推出场景分布或实现完全一致。

当前已经验证的是：固定公开参数与代理素材下，官方checkpoint能完成200段推理和本轮四项补算。数值接近论文不证明恢复了官方测试集、私有renderer或历史评价环境；VFID-I更低也不能解释为模型优于论文。

### 1.1 七个方法的Synthetic主表与论文外部参考（200×25）

单元格为「本项目 / 论文」。PSNR/SSIM 为 TorchMetrics；LPIPS 为项目 squeeze v0.1 实现，论文值取 ICLR2026 正式版；FVD 为 VideoMAE 集合距离；RM 为完整 24-transition。本项目测试素材与 GT 为公开代理，论文值来自作者未公开的测试集，只作量级与排序参考。完整精度数值见协议文档当前 Table 1 / Table V；历史阶段仍称Track A。

| 方法（冻结主表配置） | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ | FD ↓ | RM ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Any-to-Bokeh（官方权重） | 39.712 / 38.899 | 0.9863 / 0.974 | 0.0152 / 0.019 | 8.94 / 9.005 | 0.445 / 0.431 | 0.0056 / 0.007 |
| MPIB（γ=2.2） | 31.619 / 31.390 | 0.9744 / 0.950 | 0.0284 / 0.040 | 24.96 / 35.766 | 0.450 / 0.481 | 0.0082 / 0.011 |
| Dr.Bokeh（GT alpha + LaMa）\* | 31.589 / 31.273 | 0.9735 / 0.950 | 0.0294 / 0.046 | 25.86 / 32.710 | 0.479 / 0.522 | 0.0084 / 0.011 |
| BokehMe（Normalized，γ=2.2） | 33.334 / 27.992 | 0.9794 / 0.936 | 0.0247 / 0.060 | 18.55 / 39.102 | 0.540 / 0.536 | 0.0074 / 0.013 |
| BokehDiff（Normalized） | 27.063 / 27.525 | 0.8438 / 0.834 | 0.1320 / 0.127 | 66.00 / 65.678 | 0.802 / 0.660 | 0.0216 / 0.021 |
| DeepLens（单级SR，aperture饱和） | 28.167 / 24.574 | 0.9437 / 0.819 | 0.0633 / 0.183 | 40.12 / 125.338 | 0.583 / 1.162 | 0.0124 / 0.030 |
| MagicBokeh（Target-guided Depth-Medoid B） | 25.120 / — | 0.7995 / — | 0.1977 / — | 97.79 / — | 0.871 / — | 0.0268 / — |

\* **Dr.Bokeh 主行读取 GT union alpha**（真实前景 alpha 的并集），原有其他方法接收RGB + PNG8 disparity + focus + K；新增MagicBokeh从RGB估深，并使用GT可见目标区域与自身预测深度辅助选择一个焦点像素，信息权限不同。不读取 GT alpha 的公开完整 pipeline（LDF+LaMa）为 PSNR 29.108 / FVD 143.29，保留在 8.5 与 Table V。该选择服务于当前 ATB 复现，不声称作者私有 adapter 使用了 GT alpha；未来自建公平 benchmark 优先重新评估公开完整 pipeline。

**可以得出的结论**：

- **已复现**：Any-to-Bokeh 在六项指标上均排第一，量级与论文接近；它领先 MPIB 的 PSNR 差距为 8.09 dB，论文为 7.51 dB。
- **未复现**：原六种有论文对应行的方法之间的排序。按 PSNR，论文为 ATB > MPIB > Dr.Bokeh > BokehMe > BokehDiff > DeepLens，本项目对应六法为 ATB > BokehMe > MPIB > Dr.Bokeh > DeepLens > BokehDiff；FVD 排序也只部分一致。BokehMe 和 DeepLens 的绝对值明显优于论文行。新增MagicBokeh B没有对应的ATB论文行，不参与六法论文偏差统计；其25.1202 dB在当前七法Synthetic主表中最低，结论限定于当前代理数据与冻结控制协议。
- **选择偏差**：MPIB、BokehMe、BokehDiff 的主表配置是在这 200 段上从预设候选中按“整体更接近论文行”选定的；Dr.Bokeh 的 GT-alpha 配置也在看过 200 段结果后选定。这些行接近论文，有一部分来自选择本身，不是独立验证。Any-to-Bokeh 行未经候选选择；DeepLens 的控制映射在正式分数出现前冻结，但 K16/32 全部触发 aperture 饱和（2725/5000 帧）。

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
| **LPIPS ↓** | 0.019（ICLR2026版） | **0.0152** | -0.0038 | ±0.0085 | [0.0053, 0.0627] | 版本补充对照，项目squeeze实现 |
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

以下按[ICLR2026正式版Table1](https://proceedings.iclr.cc/paper_files/paper/2026/file/a0dca56ad0aceeb4eab4d35630e9a0ed-Paper-Conference.pdf#page=7)补齐LPIPS列；旧arXiv v2缺此列，其余所列数值一致。

| Method | PSNR ↑ | SSIM ↑ | FD ↓ | RM ↓ | VFID-I ↓ | FVD ↓ | LPIPS ↓ | VEPI ↑ | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepLens | 24.574 | 0.819 | 1.162 | 0.030 | 16.042 | 125.338 | 0.183 | 0.715 | 0.226 |
| BokehDiff | 27.525 | 0.834 | 0.660 | 0.021 | 7.395 | 65.678 | 0.127 | 0.859 | 0.799 |
| BokehMe | 27.992 | 0.936 | 0.536 | 0.013 | 8.633 | 39.102 | 0.060 | 0.937 | 0.103 |
| Dr.Bokeh | 31.273 | 0.950 | 0.522 | 0.011 | 6.097 | 32.710 | 0.046 | 0.863 | 2.729 |
| MPIB | 31.390 | 0.950 | 0.481 | 0.011 | 5.444 | 35.766 | 0.040 | 0.921 | 0.521 |
| **Any-to-Bokeh** | **38.899** | **0.974** | **0.431** | **0.007** | **1.479** | **9.005** | 0.019 | **0.944** | **0.363** |

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

后续MPIB、BokehMe及Dr.Bokeh已完成，见8.1–8.3；BokehDiff双候选200段及视觉复核完成，结果见8.4；Dr.Bokeh新增输入协议200段及配对分析已收尾，见8.5；DeepLens单级SR+饱和200段已完成，见8.7；开发依据保留在8.6。未来公平Benchmark仍需单独确定源划分、renderer与指标，Cycles及AM validation恢复线保持暂停。

---

### 8.1 MPIB复现锚点：双候选已完成，冻结gamma=2.2

**完成时间：2026-09-28 11:06:59 UTC。** 两候选各200段×25帧=5000张prediction，RGB/GT及几何清单未变。MPIB走完整LaMa→MPINet→AUPNet→MPIBokehRenderer流程；同一帧的预测图层在两个gamma之间共享，跨帧不缓存，未读取构造GT的真实RGBA层。候选在运行前限定为1.0/2.2，没有增加连续搜索。

对照目标是论文 **MPIB** 行，差值均为“本次−论文”：

| 指标 | 论文MPIB | gamma=1 | gamma=2.2（冻结） | gamma=1差值 | gamma=2.2差值 |
|---|---:|---:|---:|---:|---:|
| PSNR ↑ / dB | 31.390 | 35.972438 | **31.619476** | +4.582438 | +0.229476 |
| SSIM ↑ | 0.950 | 0.984682 | **0.974408** | +0.034682 | +0.024408 |
| FD ↓ | 0.481 | 0.359335 | **0.449849** | -0.121665 | -0.031151 |
| RM ↓（24 transitions） | 0.011 | 0.005572 | **0.008184** | -0.005428 | -0.002816 |
| FVD ↓ / VideoMAE | 35.766 | 19.708852 | **24.956944** | -16.057148 | -10.809056 |
| VFID-I ↓ / I3D | 5.444 | 1.645553 | **3.208706** | -3.798447 | -2.235294 |

**选择结论：gamma=2.2在六项指标上的绝对偏差均更小。** 不需要跨单位加权或主观折中，符合预先规定的复现校准原则，因此冻结gamma=2.2。gamma=1在本项目sRGB-GT上的重建指标更好，但不是本轮“复现论文MPIB统计行为”的选择目标；两套原始结果全部保留。

这里仍有明显未对齐项：gamma=2.2的FVD为24.95694而论文为35.766，VFID-I为3.20871而论文为5.444，SSIM也高于论文。因此只能说它是**两个有依据候选中更接近论文的配置**，不能说已经找回作者未公开adapter。

原项目LPIPS为 **0.016910（γ=1）/0.028355（γ=2.2）**；当时采用的arXiv v2 Table1未报告该列，ICLR2026正式版已确认MPIB为0.040。保留原诊断与当时六项选择口径，不倒推更改冻结。公开代码实际前2对相邻帧的RM为 **0.005603/0.008238**，完整24对RM见主表；两种RM口径都指向gamma=2.2更接近0.011，但不能据此证明论文当时使用了哪一种。

**按K分组（两候选采用同一批配对scene）**：

| K | 段数 | PSNR γ=1 | PSNR γ=2.2 | SSIM γ=2.2 | FD γ=1 | FD γ=2.2 | RM γ=2.2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 48 | 35.2423 | 33.1374 | 0.972220 | 0.140297 | 0.156359 | 0.006747 |
| 8 | 43 | 37.0885 | 33.2165 | 0.978363 | 0.156392 | 0.187143 | 0.006977 |
| 16 | 56 | 36.8171 | 31.2349 | 0.976457 | 0.304282 | 0.393032 | 0.008362 |
| 32 | 53 | 34.8358 | 29.3554 | 0.971015 | 0.780529 | 0.988822 | 0.010278 |

跨K素材不同，不能当作只改变K的控制实验；同scene两gamma的差异则只有最终渲染的gamma不同。两个候选的全局PSNR相差 **4.35296 dB**，说明当前参考域下光度约定影响很大，不能把选择后的全部方法差距归因于网络能力。

Any-to-Bokeh与冻结MPIB的PSNR差为 **8.09213 dB**，论文差为 **7.509 dB**；FVD差为 **16.01943**，论文差为 **26.761**。排序在这些指标上保持，但性能间隔没有全部恢复。MPIB配置已用论文行辅助选择，因此Table V含有校准证据，不能又把同一行称为完全独立的benchmark验证。

**检查与视觉证据**：两侧10000个预测PNG名称/数量/尺寸/RGB模式/文件完整性通过；40个批次、400条scene结果及全部标量有限。GT特征与Any-to-Bokeh评价缓存逐值相同，FVD为200×1408、VFID-I为400×1024/侧；按K加权均值与总体均值一致。原始demo对照和固定8场景40关键帧检查通过。全量后额外查看repro_068（最低PSNR）、repro_012（最高FD）、repro_193（最高RM）的0/6/12/18/24帧，未见明显错帧或焦点对象错配；gamma=2.2在斑马和强对比背景的混合虚化区域更亮，相关失败案例保留。累计11段55关键帧，不声称已人工播放检查全部视频。

**冻结与复用**：配置`configs/mpib-track-a-frozen.json`；小型结果`records/mpib-adapter-recovery-200-20260928.json`；完整预测`outputs/mpib-adapter-recovery-20260928/gamma_2.2/clips/<scene>/pred_mpib/`。原候选协议`configs/mpib-adapter-recovery.json`保持原样。后续MPIB沿用冻结版本；若进入新的光度/输入域，提前声明转换条件，不依据Ours结果改配置。

```bash
conda activate magicbokeh
CUDA_VISIBLE_DEVICES=0 python scripts/run_mpib_recovery.py \
  --protocol configs/mpib-track-a-frozen.json \
  --output outputs/mpib-track-a-repeat
```

上述为独立复跑示例，当前5000帧冻结结果已齐全，无需重跑。未将本轮共享图层、并发评价的计时填入正式Efficiency表。该阶段GPU显存已释放；随后完成的BokehMe见8.2。

### 8.2 BokehMe：双候选200段完成，冻结Normalized / gamma=2.2

**自动汇总完成：2026-09-28 13:45:29 UTC。** A（Shared-Defocus）和B（Normalized）各5000张prediction，完整PSNR/SSIM/LPIPS/FD/RM、VideoMAE FVD、I3D VFID-I均完成。对照目标为论文BokehMe行；差值为“本次−论文”。

| 指标 | 论文BokehMe | A：Shared-Defocus | B：Normalized（冻结） | A−论文 | B−论文 |
|---|---:|---:|---:|---:|---:|
| PSNR ↑ / dB | 27.992 | 33.842835 | **33.334003** | +5.850835 | +5.342003 |
| SSIM ↑ | 0.936 | 0.983270 | **0.979418** | +0.047270 | +0.043418 |
| FD ↓ | 0.536 | 0.415701 | **0.539533** | -0.120299 | +0.003533 |
| RM ↓（24 transitions） | 0.013 | 0.006061 | **0.007394** | -0.006939 | -0.005606 |
| FVD ↓ / VideoMAE | 39.102 | 15.366611 | **18.546370** | -23.735389 | -20.555630 |
| VFID-I ↓ / I3D | 8.633 | 2.321887 | **2.814896** | -6.311113 | -5.818104 |

**选择B的依据与边界**：六项绝对偏差一致指向B，故按运行前约定冻结B，没有临时加权或增加gamma/highlight搜索。A的本地GT重建分数更好，仍完整保留。这个选择只说明B在两个预定候选中更像论文行，不证明找回未公开的作者adapter。

最终控制为`d′=(d−min(d))/span`、`df′=(df−min(d))/span`、K不变、pipeline输入`K(d′−df′)/10`，内部再乘10。gamma=2.2、highlight=false、圆形scatter、官方ARNet/IUNet及误差图融合、逐帧独立推理。仅调用renderer或使用GT分层均不属于本次baseline。A遵循共享物理defocus；B来自demo式归一化。官方BLB的物理视差说明支持A无需归一化，因此不能把B包装为普适物理约定。

**剩余复现差距很大**：B的PSNR比论文高5.3420 dB、FVD低20.5556、VFID-I低5.8181；只有FD数值非常接近（0.53953对0.536）。当前BokehMe PSNR33.3340高于冻结MPIB31.6195，而论文BokehMe27.992低于MPIB31.390；当前BokehMe FVD18.5464低于MPIB24.9569，而论文为39.102高于35.766。SSIM、RM、VFID-I的BokehMe/MPIB排序也相反，FD排序一致。**完整论文性能版图尚未恢复**，不能因为每个方法都挑了更接近论文的候选就宣布Benchmark已通过独立验证。

替代场景、代理GT、未公开adapter和评价适配均可能贡献差异，目前不能唯一归因。不把论文排序当硬门槛；不为匹配排序继续改K/gamma/GT或删除异常场景。Any-to-Bokeh仍在PSNR、SSIM、FVD等主项上优于本次BokehMe，但这不是全部方法相对差距已复现的证据。当前方法选择使用同一200段，属于reproduction development；未来公平held-out Benchmark独立处理。

补充指标：LPIPS A/B=0.021283/0.024698；公开代码前2对RM=0.006073/0.007504。主表使用完整24对RM，不混淆两种定义。FVD/VFID-I为整集合距离，不是逐scene均值。

**按K分组**：

| K | 段数 | PSNR A | PSNR B | SSIM B | FD A | FD B | RM B |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 48 | 38.1717 | 37.7150 | 0.990502 | 0.094215 | 0.099758 | 0.004302 |
| 8 | 43 | 35.6697 | 35.1165 | 0.984083 | 0.142203 | 0.160020 | 0.005607 |
| 16 | 56 | 32.2239 | 31.6873 | 0.976011 | 0.360245 | 0.425723 | 0.008272 |
| 32 | 53 | 30.1507 | 29.6601 | 0.969197 | 0.987347 | 1.365981 | 0.010715 |

两候选在同一scene上配对，B的全局PSNR比A低0.50883 dB；各K组包含不同素材，不能把跨K统计当作只改变K的控制实验。论文没有对应by-K结果，分组仅用于内部诊断。

**完整性与视觉检查**：10000张PNG名称/数量/1024×576/RGB模式/完整性通过；400条逐scene指标有限；FVD每侧200×1408、VFID-I每侧400×1024，GT特征与已有Any-to-Bokeh缓存逐值一致，by-K加权均值与总体一致。每帧均调用classical/ARNet/IUNet。与未改动官方demo的预检查最大float差异3.58e-7、PNG8最多1色阶，符合事先容差；scatter浮点atomicAdd不保证逐位一致。

固定8段40关键帧后，额外查看repro_068（最低PSNR，B=22.2461）、repro_012（最高FD，B=4.30462）、repro_193（最高RM，B=0.021732）的0/6/12/18/24帧。斑马高频条纹/混合虚化、暗场强虚化、运动遮挡分别存在差异；未见明显错帧或焦点对象反转，不能据静态关键帧唯一解释FD/RM异常。累计11段55关键帧，未声称人工检查全部视频，未删除任何场景。

全量span范围0.792157–0.996078，B对A的实际模糊放大约1.004–1.262倍。289帧变换后focus略高于1，最大1.002413，来自连续focus与PNG8上界量化差；按预先协议保留、不夹紧，不改变已有数据控制。

**复用入口**：`configs/bokehme-track-a-frozen.json`；小型记录`records/bokehme-adapter-recovery-200-20260928.json`；选定输出`outputs/bokehme-adapter-recovery-20260928/normalized/clips/<scene>/pred_bokehme/`。另一候选及原恢复配置保留。复跑时使用新输出目录：

```bash
conda activate magicbokeh
CUDA_VISIBLE_DEVICES=0 python scripts/run_bokehme_recovery.py \
  --protocol configs/bokehme-track-a-frozen.json \
  --output outputs/bokehme-track-a-repeat
```

冻结配置单帧复查（repro_004第0帧）与原候选PNG逐像素一致。当前5000帧选定结果已经齐全，无需重跑；后台进程全部结束，GPU显存已释放。本轮并发计时不填Efficiency表。后续Dr.Bokeh Track A全量与终审现已完成，见8.3。

### 8.3 Dr.Bokeh：单一协议200段完成，FVD显著差异保留

固定8段共200帧完整执行官方LDF、RGBAD2layers、前景cv2深度修复、背景LaMa RGB/depth、双层renderer。不给GT alpha或隐藏颜色层；40组关键帧已检查，分割漏主体、LaMa残影等方法失败保留。

| 实现检查 | 实测 |
|---|---:|
| 输入CoC恒等式最大误差，float64 / FP32 | 7.11e-15 / 1.58e-6 |
| 修复后层最大名义虚化半径 | 25.6 px |
| 空 / 全满LDF mask帧数 | 0 / 0 |
| 两段K32共50帧，同层lens71/101最大float差 | 0 |
| 加速启动与原版：200帧同保存图层最大float差 | 0 |
| 完整pipeline重复：PNG8最大差 | 0 |

据实现/覆盖检查冻结lens71、gamma2.2、highlight=false、offset0及既定K补偿，不看论文分数选择。CUDA作者启动网格在1024×576下只有6块，改为16×16线程二维网格2304块后利用更多SM；每像素运算不变，200帧同层输出位级一致。原版和加速版完整重算LaMa都出现约1.19e-7最终float变化，不能将整体网络称为位级确定性。历史GPU-util100%只代表kernel活跃时间，不证明吞吐饱和；原12–15小时估计撤回。

**全量18:13:36 UTC完成**：200段、5000张prediction、20批帧/时序指标及FVD/VFID-I全部齐全。5000张PNG完整性、所有指标finite、GT特征缓存一致性、by-K加权均值检查通过；修复后最大名义半径31.87451 px，5000帧覆盖检查均通过，无空/全满LDF mask。

| 指标 | 论文Dr.Bokeh | 本项目200段 | 本项目−论文 |
|---|---:|---:|---:|
| PSNR ↑ / dB | 31.273000 | **29.107707** | -2.165293 |
| SSIM ↑ | 0.950000 | **0.963207** | +0.013207 |
| FD ↓ | 0.522000 | **0.561774** | +0.039774 |
| RM ↓（24 transitions） | 0.011000 | **0.012391** | +0.001391 |
| FVD ↓ / VideoMAE | 32.710000 | **143.289688** | +110.579688 |
| VFID-I ↓ / I3D | 6.097000 | **5.760450** | -0.336550 |

补充LPIPS **0.037996**；主PSNR/SSIM沿用TorchMetrics，原逐帧PSNR均值29.282036。公开前2对RM=0.012609，不混入主列24对RM。

| K | scenes | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD ↓ | RM ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 48 | 29.9679 | 0.962776 | 0.031277 | 0.259706 | 0.010937 |
| 8 | 43 | 29.6828 | 0.964781 | 0.031756 | 0.289562 | 0.011562 |
| 16 | 56 | 28.9299 | 0.963707 | 0.038803 | 0.492783 | 0.012712 |
| 32 | 53 | 28.0500 | 0.961793 | 0.048292 | 1.129092 | 0.014041 |

FVD/VFID-I为200段分布距离，不把它们分摊成每scene分数或按K加权。K32组PSNR最低、FD/RM最高，保留强虚化困难情况。

**FVD核验**：公开FeatureStats/scipy sqrtm计算143.289688；相同缓存特征采用独立population covariance低秩SVD计算143.289918，差仅0.000230。均值项48.159635、协方差项95.130283；这是对聚合数值的检查，不是重新提取特征或换指标。200×1408两侧特征finite，GT缓存逐值一致。原数值保留，不替换成独立算法值。

终审查看repro_068（PSNR20.65848，最低）、repro_143（最大配对VideoMAE特征距离）、repro_012（FD3.25892，最高）的0/6/12/18/24关键帧。068斑马虚化区明显偏亮、条纹对比改变；143强虚化主体及边界与参考存在差异；012背景模糊纹理不同。未见明显错帧、整体颜色通道反转或空输出，但这些观察不能唯一解释FVD高值。配对特征距离仅为场景诊断，不叫单场景FVD或对总FVD的贡献。

**复现结论**：PSNR低于论文2.1653 dB，FVD高110.5797；当前Dr.Bokeh与BokehMe在六项论文可比指标上的两方法排序全部反转，Dr与MPIB的FVD排序也反转。已经完成当前单一Track A协议的评价，但论文方法性能版图仍未完整恢复。可能关联输入/renderer/光度约定及分割补全差异，不能据现有结果唯一归因。当前协议在查看指标前按实现/覆盖冻结，该历史200段协议与结果继续保留；用户随后授权的新输入协议诊断另列8.5，不改场景或GT。并发计时仍不填Efficiency表。

证据：`records/drbokeh-full-preflight-20260928.json`、`records/drbokeh-cuda-launch-check-20260928.json`；冻结配置`configs/drbokeh-track-a-frozen.json`；输出`outputs/drbokeh-track-a-20260928/`。

全量/终审证据：`records/drbokeh-track-a-200-20260928.json`、`records/drbokeh-track-a-review-20260928.json`。

### 8.4 BokehDiff：双200段完成，B选为主表、A保留

两侧各5000帧、全部逐段指标和FVD/VFID-I于2026-09-29 05:07北京时间完成；PNG完整性、GT特征一致性与随机恢复检查通过。A保留共享原始几何；B对d/df做min-max而K不变，控制幅度放大1/span。其他checkpoint/seed42连续随机流/PISA/输出PNG8规则相同。

| 指标 | 论文 | A Shared-Geometry | B Normalized | 更接近论文 |
|---|---:|---:|---:|---|
| PSNR | 27.525000 | 27.156036 | 27.062699 | A |
| SSIM | 0.834000 | 0.841752 | 0.843751 | A |
| FD | 0.660000 | 0.768883 | 0.802247 | A |
| RM | 0.021000 | 0.021068 | 0.021633 | A |
| FVD | 65.678000 | 67.926652 | 65.996926 | B |
| VFID_I | 7.395000 | 5.943235 | 6.501893 | B |

LPIPS A/B=0.134406/0.132024；主RM为24 transitions，作者公开前2对版本分别0.021282/0.022043。

| 候选 | K | scenes | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD ↓ | RM ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 4 | 48 | 27.3245 | 0.805400 | 0.120598 | 0.278166 | 0.020730 |
| A | 8 | 43 | 27.9153 | 0.826076 | 0.147148 | 0.340665 | 0.020600 |
| A | 16 | 56 | 27.1921 | 0.845329 | 0.150816 | 0.716362 | 0.020148 |
| A | 32 | 53 | 26.3494 | 0.883615 | 0.119233 | 1.616222 | 0.022727 |
| B | 4 | 48 | 27.3329 | 0.805695 | 0.120189 | 0.277920 | 0.020732 |
| B | 8 | 43 | 28.0600 | 0.832283 | 0.138133 | 0.329123 | 0.019694 |
| B | 16 | 56 | 27.1271 | 0.849632 | 0.145217 | 0.698400 | 0.021314 |
| B | 32 | 53 | 25.9408 | 0.881308 | 0.123847 | 1.770692 | 0.024359 |

**定性复核**：原8段及全量后repro_012/193/197，共55组/110张候选关键帧已查看。两侧最低PSNR均为193：较GT更强的背景/人物/动物平滑；最高FD均为012：强虚化及边界形态差异。197为最大配对VideoMAE特征距离，保持作为分布差异定位场景，不称逐段FVD。弱虚化场景004/006/007存在较GT更清晰的细节，017的A部分时刻虚化不足；两候选均有纹理重绘。未见需重跑的新增实现阻塞，不将少量关键帧当作视频无闪烁证明。

**当前结论**：论文贴近程度为4项偏A、2项偏B。预登记阶段没有设置指标权重，因此当时未用投票或相对误差平均强行冻结；在完整200段与定性复核均结束后，用户选择B（Normalized）作为主表展示配置。理由是PSNR/SSIM/FD/RM上A虽更接近论文，但A/B差距相对有限；B在FVD/VFID-I上更接近，尤其FVD=65.996926与论文65.678仅差0.318926。Table V继续并列A/B，A作为protocol-recovery alternative保留。该选择用于复现展示，不证明Any-to-Bokeh作者私有adapter就是B；本轮不新增模型推理、第三候选或K搜索。完整结果与复核：`records/bokehdiff-adapter-recovery-200-20260928.json`、`records/bokehdiff-final-review-20260929.json`。

### 8.5 Dr.Bokeh：同200段输入协议实验已收尾（2026-09-29）

三组均为同一200×25=5000帧，1024×576。Full-public复用原结果，GT-alpha与known-layer各新增/复用5000帧。全部PSNR/SSIM/LPIPS/FD/RM/FVD/VFID-I、by-K及逐段配对分析完成；没有运行geometry/gamma/blur-strength新候选。实验完成当时未自动替换主表；随后用户基于“当前阶段优先复现Any-to-Bokeh Table 1性能版图”的目标，选择GT-alpha作为Track A reproduction reported configuration。

| 同200段输入协议 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD ↓ | RM ↓ | FVD ↓ | VFID-I ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-public：LDF+LaMa | 29.107707 | 0.963207 | 0.037996 | 0.561774 | 0.012391 | 143.289689 | 5.760450 |
| GT union alpha+LaMa | 31.588505 | 0.973475 | 0.029383 | 0.478618 | 0.008393 | 25.859006 | 3.732668 |
| Known 4FG+1BG | 32.259896 | 0.978147 | 0.025926 | 0.443133 | 0.007250 | 21.107962 | 3.311429 |
| 论文Dr.Bokeh（ICLR2026，不同测试集） | 31.273 | 0.950 | 0.046 | 0.522 | 0.011 | 32.710 | 6.097 |

PSNR/SSIM为TorchMetrics；RM主列为全部24个transition，公开前2对诊断RM分别为0.012609 / 0.008483 / 0.007298（Full/Alpha/Known）。FVD、VFID-I为完整集合距离，不能给它们定义逐段胜率。Full-public复用原预测/逐段指标，同200段缓存特征重算FVD与旧记录仅有不足1e-6的数值差，历史成绩未改。

| 配对改善（正值表示改善） | PSNR增加/dB | FD减少 | FVD减少 | PSNR胜率 | FD胜率 |
|---|---:|---:|---:|---:|---:|
| Full→GT alpha | 2.480798 | 0.083156 | 117.430682 | 198/200（99.0%） | 181/200（90.5%） |
| GT alpha→Known | 0.671391 | 0.035485 | 4.751044 | 161/200（80.5%） | 156/200（78.0%） |
| Full→Known | 3.152189 | 0.118641 | 122.181727 | 193/200（96.5%） | 194/200（97.0%） |

GT-alpha相对Full-public的PSNR中位增益2.287867dB，PSNR/FD同时改善180/200段。因此改善在完整200段上成立，非只依赖原11段诊断。2段PSNR退步和19段FD退步均保留；Known相对Alpha仍有39段PSNR、44段FD退步，**不能称为逐场景性能上限**。这些是描述性配对统计，没有把源素材相关的场景当独立样本进行显著性检验。

| K | n | Full→Alpha ΔPSNR ↑ | Full→Alpha FD减少 | PSNR胜数 | FD胜数 | Alpha→Known ΔPSNR ↑ | Alpha→Known FD减少 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 48 | 3.156101 | 0.098549 | 48/48 | 48/48 | 1.177192 | 0.018622 |
| 8 | 43 | 3.493732 | 0.101116 | 43/43 | 43/43 | 0.579213 | 0.013328 |
| 16 | 56 | 2.177549 | 0.086635 | 56/56 | 50/56 | 0.485600 | 0.017811 |
| 32 | 53 | 1.367801 | 0.050968 | 51/53 | 40/53 | 0.484401 | 0.087408 |

**K趋势没有支持预设的简单解释**：Full→Alpha的PSNR/FD改善量没有随K单调增加，K32的平均改善反而最小。Alpha→Known的FD改善在K32较大，但整体也不是单调序列，且PSNR不呈同样趋势。不同K组内容并不相同；既不能据本表确认“强虚化必然放大输入修正收益”，也不能反向证明强虚化不会放大误差。完整三组by-K原始均值见机器记录。

| 本项目−论文（带符号差值） | Full-public | GT alpha | Known layers |
|---|---:|---:|---:|
| PSNR | -2.165293 | +0.315505 | +0.986896 |
| SSIM | +0.013207 | +0.023475 | +0.028147 |
| FD | +0.039774 | -0.043382 | -0.078867 |
| RM | +0.001391 | -0.002607 | -0.003750 |
| FVD | +110.579689 | -6.850994 | -11.602038 |
| VFID_I | -0.336550 | -2.364332 | -2.785571 |
| LPIPS（ICLR2026新增参考列） | -0.008004 | -0.016617 | -0.020074 |

**收尾判断**：GT alpha使原来FVD143.29的大幅偏离降到25.86，分割输入及其引起的分层、深度修复、背景补全过程是当前结果的重要影响因素；不能只归因renderer本体。与论文相比，GT-alpha的PSNR高0.3155dB、FVD低6.8510，Known分别高0.9869dB、低11.6020；新协议的SSIM/RM/VFID等也整体优于论文，仍有代理数据/渲染/输入及评价适配差异。Full-public的SSIM与VFID-I反而更接近论文，因此不存在“oracle后六项都更接近”的结论。更接近某几项或改善原排序，不证明作者使用了oracle。**本轮输入实验关账；没有新的客观实现错误要求继续搜索，暂不追加geometry/gamma/强度。后续用户明确将GT-alpha选为当前ATB reproduction主表配置；这是阶段性reporting选择，不是私有adapter归因。**

**后续主表报告选择（Ours开始前冻结）**：当前阶段的任务是尽可能恢复ATB Synthetic benchmark及其Table 1 performance landscape，而不是建立最终统一信息权限的公平benchmark。基于完整200段结果，GT-alpha被选为Dr.Bokeh的Track A主表 **reconstruction adapter**；Full-public继续保留为公开完整pipeline诊断结果，Known Layers继续作为额外隐藏层信息诊断。该选择发生在Ours正式评价前，之后不允许根据Ours成绩重新回调Dr.Bokeh协议。

当前先在已冻结的ATB-style reproduction benchmark上比较Ours与Any-to-Bokeh。未来若研究问题需要独立benchmark，应按最终任务统一baseline的信息权限，届时重新评估Dr.Bokeh Full-public等不读取GT alpha/hidden layers的公开完整pipeline，不能默认沿用本阶段GT-alpha权限。独立评价不预设低帧率粗深度，也不是当前前置；不同条件的结果分开命名和报告。

**实现与反例检查**：三组共15000张PNG通过完整性/尺寸/模式验证；新增两分支10000帧素材重建与lens覆盖全部通过，最大实际R31.87451px，遗漏径向权重质量最大4.3893e-6，CoC变换误差float64≤1.07e-14/float32≤3.55e-6，无输出裁剪。原11段55组视觉记录保留；追加查看GT-alpha最大PSNR退步193，以及Known相对Alpha最大PSNR退步160、最大FD退步101，各0/6/12/18/24帧和中间层。累计14段70组，未发现新的明显错层/通道/帧序实现阻塞；稀疏关键帧不能排除未采样时序问题。

- repro_193：Alpha相对Full的PSNR−0.337257dB、FD+0.105294；mask、前景深度修复、LaMa补全均改变，边界更完整不保证最终分数单调改善。
- repro_160：Known相对Alpha的PSNR−2.475994dB；人物保持清晰、狮子保持离焦，但羊/狮子的外观和边缘混合与参考仍有差异，不唯一归因某个组件。
- repro_101：Known相对Alpha的FD+0.154335，相对Full+0.456109；强虚化动物及遮挡/背景仍有差异，不能仅凭FD和5个关键帧认定时序模块错误。
- repro_000的Alpha FD恶化反例继续保留；不删除任何scene或调整参数。

**执行收尾**：推理使用原四路场景调度。逐段指标保留串行已完成46段后改四路互斥分片，函数不变，按manifest顺序float64汇总；FVD/VFID不拆分。最后检查曾因复用场景`repro_143`缺少Full-public目录软链接而停止，原25帧完整。补齐所有复用场景的索引后检查通过，未重跑推理/特征，原日志保留。当前所有后台计算已结束。

结果`records/drbokeh-input-recovery-200-20260929.json`，配对分析`records/drbokeh-input-effect-analysis-20260929.json`，收尾与反例记录`records/drbokeh-input-final-review-20260929.json`；入口`scripts/run_drbokeh_protocol_recovery.py`、`scripts/analyze_drbokeh_input_recovery.py`和`scripts/review_drbokeh_inputs.py`。自动结果中的REVIEW_PENDING保留生成时状态，最终收尾状态以final-review为准。Known同时改变隐藏RGB、alpha、几何和层数，不能叫纯LaMa消融；本集合仍是复现开发集，不是held-out验证。

**保留的历史11段诊断**（与上方200段不可混算）：

| 同11段 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD ↓ | RM ↓ | FVD ↓ | VFID-I ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| LDF+LaMa | 28.251015 | 0.958056 | 0.043572 | 0.978921 | 0.013668 | 293.721479 | 14.740562 |
| GT union alpha+LaMa | 30.212765 | 0.969182 | 0.034304 | 0.932069 | 0.009422 | 48.065288 | 7.087483 |
| Known 4FG+1BG | 30.816634 | 0.974313 | 0.029478 | 0.833600 | 0.008149 | 40.349377 | 6.159460 |

### 8.6 DeepLens：原生环境与独立控制标定（开发历史）

新增隔离Python3.7/TF1.15 CPU环境，与已有Python3.10/TF2.15 compat.v1在相同3帧、原checkpoint、外部几何、LensBlur+Feature+一次SR上对照。两侧均恢复210个变量；disparity resize及signed输入逐值一致；TF2新probe与此前adapter结果逐值一致。

| 对照帧 | low输出MAE | 最终float最大差 | PNG8最大差 | PNG通道差异比例 |
|---|---:|---:|---:|---:|
| repro_000/0 | 6.329e-7 | 0.046681 | 11 | 0.23075% |
| repro_004/12 | 3.622e-7 | 0.002045 | 1 | 0.00746% |
| repro_017/24 | 1.747e-6 | 0.003317 | 1 | 0.06996% |

预先设定的FP32对照范围未全部通过，原判定保留，不放宽阈值。定位到官方SR内部`tf.cast(low*255, uint8)`：本机原生TF1与TF2 CPU对越界值行为不同；同一个约256.195的值分别得到0和255。固定完全相同的native low输入后，仍有65个中间通道相差255，最终最大差仍约0.046681，故不只是上游浮点舍入噪声。

**运行选择**：后续adapter开发优先用已跑通的原生TF1 CPU，不给官方SR擅自加clamp、不把原兼容40帧冒充原生结果；不宣称与原论文GPU逐数值一致。代码`scripts/probe_deeplens_tf_parity.py`，依赖`configs/requirements-deeplens-tf1-cpu.txt`，证据`records/deeplens-tf-parity-20260929.json`和`records/deeplens-control-runtime-20260929.json`。

**该阶段控制口径尚未冻结**：[原论文§3.2.2](https://arxiv.org/pdf/1810.08100)定义的是目标/训练最大光圈的比值；[官方交互入口](https://github.com/scott89/deeplens_eval/blob/3dbc1b04ba082b3b6b1dc0d944b3d18fa542ae2c/core/evaluator_interactive.py)使用`(inverse_depth-focus)*aperture`及slider/10。proto默认的max_aperture_radius=16/max_kernel_radius=8没有在此forward中用于把本benchmark K变成输出像素半径，不能直接当映射依据。3帧×aperture{0,.25,.5,1}共12次原生CPU控制检查均finite、尺寸正确、输出随控制变化；这不是物理半径标定，也未依据论文分数挑参数。`K/32`仍仅是临时UI范围假设，训练最大光圈及分辨率尺度对应证据未确认，正式200段尚未启动。


**独立控制标定（2026-09-29）**：固定简单阶跃RGB/均匀平面，2方向×4有符号离焦差×4aperture，共32次原生TF1单级SR运行完成。初始二阶矩半径估计受振铃影响，出现负方差/增量，不能用截零后的拟合值冻结。初始协议同时记录的10–90%边缘宽度对八组控制均单调；按理想圆盘的1.37409765×R换算，aperture1/最大离焦差的等效半径为11.8425–13.1806 px。圆盘解析参考的像素积分验证误差最大0.0833 px。所有非零控制的描述性线性系数12.4880 px（RMSE0.2980 px）仅用于报告响应，不是已冻结mapping，也不证明该学习算子是圆盘卷积。

这说明当前原生单级路径不能覆盖K16/32的目标响应。**待明确采用上限饱和并披露覆盖不足，还是先检查官方递归SR**；后者涉及输出尺度策略，不能冒称1024默认入口。该标定时点正式8段预检与200段尚未启动；后续完成结果见8.7，未按论文PSNR/FVD选择。独立标定记录`records/deeplens-independent-control-20260929.json`；原生adapter三帧与此前native probe逐值相等，记录`records/deeplens-native-adapter-regression-20260929.json`。


**补充独立文献核验与尺度修正（2026-09-29）**：用户贴文中“precise disparity”依据为[ATB arXiv v1 §4.2](https://arxiv.org/html/2505.21593v1#S4.SS2)；“built-in depth model”依据为同版AppendixD。二者与[ICLR2026正式版](https://proceedings.iclr.cc/paper_files/paper/2026/file/a0dca56ad0aceeb4eab4d35630e9a0ed-Paper-Conference.pdf)新增LPIPS列不是同一版本文字。Synthetic采用外部精确视差有依据，但未公开具体DeepLens adapter；v1真实视频示例中自带DepthNet的描述，不能自动当作已确认的最新版完整DAVIS协议。DeepLens原论文§3.1/§3.2.2及官方代码支持normalized inverse depth、共同变换focus、signed×aperture和相对最大光圈比例；65×65的复杂度例子、未参与此forward换算的proto16/8，均不证明K/32。

用户贴文里的归一化与光圈公式还缺跨度补偿。若保持当前benchmark的原始disparity单位，令

\[
s=d_{\max}-d_{\min},\quad D=(d-d_{\min})/s,\quad D_f=(d_f-d_{\min})/s.
\]

固定输出尺度下，若独立测得近似关系为\(R_{eff}\approx c\alpha|D-D_f|\)，要对应\(R=K|d-d_f|\)，就应当使用

\[
\boxed{\alpha_{req}=Ks/c},\qquad (D-D_f)\alpha_{req}=(d-d_f)K/c.
\]

这是固定单位换算，不是逐scene按分数调参。只有不额外minmax、或s=1时才简化成K/c。未饱和时两个写法给网络的signed输入代数等价；不能把它们伪装成两个不同候选。实际float32/resize仍需预检检查；也不能把focus静默夹回[0,1]。若aperture饱和，等价性和目标强度覆盖都不再成立。

按现有描述性c=12.4880和5000帧已保存的几何范围推算（仅输入统计，无预测分数），K4/8/16/32所需aperture范围分别为0.2537–0.3191、0.5075–0.6381、1.0149–1.2762、2.0299–2.5524；K16的1400帧和K32的1325帧在此近似下均超过1。因此minmax本身没有消除超范围问题，范围处理仍待决定。这些值不是冻结映射，c也不是严格训练最大物理光圈。

现有32次标定直接控制canonical disparity/signed输入，测LensBlur/SR响应，没有对常量深度执行minmax。用户提出把颜色edge与两层depth跳变放在一起，会混入遮挡边界效应；测单平面响应仍优先使用当前均匀深度控制。正式adapter预检另核对s>0以及整条坐标链，不重复已完成的32次probe。证据`records/deeplens-protocol-source-review-20260929.json`。该次核验只修正文献与数学口径，当时尚未选择饱和/递归SR；后续用户决定和正式结果见下文。

**后续用户决定与启动（2026-09-29）**：采用官方1024×576单级SR，冻结c=12.488及`aperture=min(K*span/c,1)`，不再比较递归SR或外推；原生TF1CPU和原权重不变。此前“待范围决定”状态至此结束。预检为固定8×25帧和40关键帧；以相同控制画面/几何验证K4/8响应、K16/32饱和平台，统计实际饱和率，成功后直接200段。配置`configs/deeplens-track-a-frozen.json`、入口`scripts/run_deeplens_track_a.py`。不能把未饱和的理想K当实际达到的强度，也不能据范围受限单独解释论文DeepLens低分。

**预检与正式启动结果**：固定8段200帧数值检查通过，实际signed误差0；40关键帧未发现明显实现阻塞，强虚化不足和边缘差异保留。K4/K8控制等效半径分别4.2030–4.2635/7.6834–8.2393px；四组固定方向/正负离焦下K16与K32输出逐值相等。预检实际饱和率0%/0%/100%/100%，记录`records/deeplens-preflight-20260929.json`。随后直接启动200×25原生CPU八路推理（复用预检200帧），完成后GPU0统一七项指标和by-K；该启动时点尚无正式分数；完成结果见8.7。

### 8.7 DeepLens：单级SR+aperture饱和，200段及全部指标完成（2026-09-29）

用户在正式分数出现前选定原生TF1.15 CPU、1024×576官方scale2路径（512×288 LensBlur后一次guided SR），固定c=12.488和`aperture=min(K*span/c,1)`。D/focus同一minmax变换，无DepthNet、GT alpha/隐藏层、递归SR或aperture外推。独立标定、完整8段200帧预检和40关键帧通过后，正式200×25运行完成，预检200帧按完全相同协议复用。

| 指标 | 论文DeepLens | 本项目200段 | 本项目−论文 |
|---|---:|---:|---:|
| PSNR ↑ / dB | 24.574000 | **28.166875** | +3.592875 |
| SSIM ↑ | 0.819000 | **0.943710** | +0.124710 |
| LPIPS ↓ | 0.183000 | **0.063333** | -0.119667 |
| FD ↓ | 1.162000 | **0.582632** | -0.579368 |
| RM ↓（24 transitions） | 0.030000 | **0.012364** | -0.017636 |
| FVD ↓ / VideoMAE | 125.338000 | **40.121195** | -85.216805 |
| VFID-I ↓ / I3D | 16.042000 | **6.962586** | -9.079414 |

论文引用为ICLR2026正式版Table1（含LPIPS）。PSNR/SSIM使用公开TorchMetrics函数；LPIPS沿用项目squeeze v0.1实现；主RM为24对，公开前2对诊断RM=0.012409。FVD/VFID是200视频/400窗口的集合距离，GT特征与原缓存逐值相同，不赋予逐段FVD，也不将不同测试条件下的更优数值当作方法改进。

| K | scenes | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD ↓ | RM ↓ | 饱和帧率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 48 | 31.5369 | 0.972929 | 0.022356 | 0.151276 | 0.006941 | 0% |
| 8 | 43 | 29.1239 | 0.956802 | 0.036992 | 0.230638 | 0.009817 | 0% |
| 16 | 56 | 26.8638 | 0.952778 | 0.050477 | 0.456725 | 0.013037 | 100% |
| 32 | 53 | 25.7151 | 0.897043 | 0.135401 | 1.391907 | 0.018633 | 100% |

**实际范围披露**：K4/8分别1200/1075帧均未饱和；K16/32分别1400/1325帧全部饱和，共2725/5000帧（54.5%）。实际aperture范围依次0.253734–0.319051、0.507468–0.638103、1、1。所有场景都纳入评价，没有排除超范围K。独立同输入控制中K16/K32的float输出逐值相等；K4/K8的圆盘边缘等效半径约4.20–4.26/7.68–8.24px。c与估算有效K仅描述固定运行尺度的边缘响应，不保证所有纹理/遮挡均等价于物理圆盘卷积。

**结果解释**：当前PSNR比论文高3.5929dB，FVD低85.2168，其他五项数值也更优；仍不能称精确恢复ATB私有adapter。当前DeepLens除VFID-I外优于本项目BokehDiff-B六项指标，而论文相应排序相反，说明整体论文性能版图尚未完全恢复。Any-to-Bokeh对DeepLens的PSNR优势为约11.54dB，论文为14.325dB；差距仍不一致。既不把这些差异当硬门槛，也不追加c、gamma、外推或递归SR搜索。

K增大时PSNR下降，K32的LPIPS/FD/RM明显更大，与已知饱和限制相容；但K组内容不同，不能从分组均值唯一归因。未饱和K4/K8及定性案例也有亮度、纹理和边缘差异，因此不能声称所有误差均来自blur-range limitation。仍无法证明ATB私有adapter是否允许aperture>1。

**检查与失败案例**：5000帧RGB PNG完整、尺寸/模式一致，全部输出finite，实际signed输入与包含饱和的解析预期误差0；按K加权均值与全量均值一致。289帧normalized focus略超1（最大1.002413），按已冻结输入口径保留，无focus clamp。8个native worker均恢复相同210个LensBlur/Feature/SR变量；官方eval.config的use_moving_average=False已核对，未加载DepthNet。

- repro_068（最低PSNR16.5397dB，K32）：斑马条纹残留明显且偏亮，背景虚化不足；保留亮度/纹理失配，不唯一归因饱和。
- repro_012（最高FD5.56193，K32）：焦内人物保持清晰，但熊/狗/背景明显比GT清晰；没有从关键帧发现明显错帧/通道错误，不能将高FD简单等同闪烁。
- repro_197（最大配对VideoMAE特征MSE0.392973，非逐段FVD）：焦内长颈鹿颈部保留，人物、桥和其他对象的细节远多于强虚化GT。
- 另查看repro_067第22帧（K8未饱和），raw输出最高1.140146、通道越界比例4.6763%；最终PNG未见明显黑块或通道异常。原生SR内部uint8行为及最终clamp保留，没有为改善分数额外修图或改clamp。

原8段40组加事后3段15组、上述额外1帧，共56组检查；稀疏关键帧不证明无任何未采样时序问题。全部失败案例保留，未发现新的实现阻塞。

**收尾与产物**：Table1A/TableV已填此单一冻结版本；Dr.Bokeh不再实验，原主表输入协议不变。推理在原生TF1CPU八路运行，逐段GPU评价从单路按完整批次安全接为双路，前3批复用；每批10scene，分布指标仍全量统一。只有本任务CPU线程调度优先级调整，没有修改模型/线程数/精度。生产流水约42.4分钟（含推理和指标重叠），不填标准化Efficiency或对照论文Time列。全部后台已结束。

配置`configs/deeplens-track-a-frozen.json`；入口`scripts/run_deeplens_track_a.py`；完整结果`records/deeplens-track-a-200-20260929.json`，收尾`records/deeplens-track-a-final-review-20260929.json`。输出`outputs/deeplens-track-a-20260929/clips/<scene>/pred_deeplens/`；关键帧页在该目录的`visual-review`及原preflight目录。运行过程归档`历史阶段记录/2026-09-29_DeepLens单级SR饱和控制与200段评价.md`。自动结果中REVIEW_PENDING保留生成时状态，最终完成状态以final-review为准。

## 九、对后续 Benchmark 的价值

本轮最重要的价值不是简单得到一个接近 38.899 dB 的数字，而是已经验证了一套完整可复现链路：

```text
公开 scene recipe → MPIB-public ray GT → Any-to-Bokeh 官方推理 → 视频质量评价
```

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

**最后更新**：2026-09-30
**状态**：Any-to-Bokeh公开代理复现及author-public指标已完成；MPIB/BokehMe/Dr.Bokeh Track A已填真实结果，BokehDiff双200段及A/B复核完成且Normalized已选为主表展示配置；Dr.Bokeh新输入协议200段已收尾，DeepLens按用户冻结的单级SR+饱和协议完成200段，见8.7；范围与版本依据保留在8.6。

补算实现及权重来源：[作者公开指标代码](https://github.com/vivoCameraResearch/any-to-bokeh/blob/f7a16959dd219080fb8eec0d32bb9602d8970c22/metrics/vid_metrics.py)、[cdfvd固定版本](https://github.com/songweige/content-debiased-fvd/tree/a1e037ab7cb087debd2221d14ae4a001ec054201)、[作者指定I3D权重](https://huggingface.co/zhengchong/VFID)、[官方VideoMAE2权重](https://huggingface.co/OpenGVLab/VideoMAE2)。


## DAVIS v2 六方法收尾（2026-09-30）

六方法各50段×25帧完整性检查通过，300段视频按冻结H.264参数编码；MP4主VEPI与作者原始入口逐帧相等，PNG只作codec诊断。自动计算于北京时间00:26:51完成，随后固定8段的六方法关键帧及25帧顺序接触图完成定性复核，Table2冻结。没有重新生成预检、补偿Normalized控制、改gamma/aperture或按论文分数调参。

| Method | VEPI-public-mp4 ↑ | VEPI-png诊断 | MP4−PNG | 论文VEPI | 本轮−论文 |
|---|---:|---:|---:|---:|---:|
| DeepLens | 0.92896 | 0.93138 | -0.00242 | 0.715 | +0.21396 |
| Dr.Bokeh | 0.87682 | 0.88009 | -0.00327 | 0.863 | +0.01382 |
| BokehMe | 0.95371 | 0.95662 | -0.00291 | 0.937 | +0.01671 |
| MPIB | 0.92987 | 0.93235 | -0.00247 | 0.921 | +0.00887 |
| BokehDiff | 0.84722 | 0.84728 | -0.00007 | 0.859 | -0.01178 |
| Any-to-Bokeh | 0.95011 | 0.95400 | -0.00389 | 0.944 | +0.00611 |
| MagicBokeh（DAVIS union区域辅助选焦） | 0.66962 | 0.66919 | +0.00043 | — | — |

MagicBokeh新增量化：VEPI-public-mp4 **0.66962**，VEPI-png **0.66919**，MP4−PNG **+0.00043**。输入为DAVIS非零标注union辅助的自身深度单点选焦，非共享VDA数值条件；多实例/组件选择限制及批准规则见上文。预检200帧及全量1250帧的选点、坐标、原生条件、种子、RGB/finite核验通过，不按VEPI或观感回调。原六行数值与定性列保持原样；新增行不自动填视觉质量结论，也不把新全量输出用于来源隔离组失败发现。[新增逐帧/逐视频及七方法同口径记录](records/magicbokeh-davis-50-20261015.json)、[最终程序核验](records/magicbokeh-davis-final-check-20261015.json)。


论文参考沿用上文§4.2 ICLR2026正式版Table1。本轮五方法与论文差值绝对值约0.0061–0.0167，DeepLens差值为+0.21396；这些只是声明条件下的外部对照，不能据此声称复原了作者私有adapter或超过原论文整体质量。本轮BokehMe和Any-to-Bokeh的VEPI排序交换；DeepLens采用原生DepthNet+UI默认aperture0.5，部分背景虚化较轻，该分数不能单独证明其bokeh质量改善。主VEPI衡量前景边缘保留，AIF原样输出就能在PNG自比达到1。

### DAVIS VEPI 排序交换与指标边界

论文中 Any-to-Bokeh 的 VEPI 为 **0.944**，BokehMe 为 **0.937**，前者仅高 **0.007**；本轮 reproduced DAVIS-2016 protocol 下，BokehMe 为 **0.95371**，Any-to-Bokeh 为 **0.95011**，排序反转但差距只有 **0.00361**。相对论文，BokehMe 增加 **+0.01671**，Any-to-Bokeh 增加 **+0.00611**，两者相对移动约 **0.01060**，已经足以跨过论文原本很小的0.007间隔。

50段逐视频配对显示，这一反转不是单个异常视频造成：BokehMe 在 **36/50** 段的VEPI高于Any-to-Bokeh，14/50段相反；逐段差值均值为 **+0.003607**，中位数 **+0.003361**，两端各去除10%后的均值 **+0.004849**。两方法逐视频VEPI相关系数约 **0.893**，说明二者面对的场景难易趋势总体一致，更像是前景边缘保留上的小幅系统性偏移，而不是评价链路失常。

编码不是排序反转的主因。PNG直接评价时，BokehMe仍为 **0.95662**、Any-to-Bokeh为 **0.95400**，差 **+0.00263**；统一H.264后差距扩大到 **+0.00361**。因此codec只额外放大约0.001，不能解释排序本身。

这组结果应按VEPI的测量目标解释，而不能直接转化成整体bokeh质量排名。公开实现对输入和输出计算Sobel边缘，在膨胀后的foreground ROI内取Pearson相关；AIF原样输出在PNG自比可达到VEPI=1。因此，**越接近原清晰输入的焦内边缘越容易获得高VEPI**，而背景虚化是否充分、边界过渡是否自然、整体视频观感与时序质量并未被该单指标完整覆盖。BokehMe当前Normalized adapter的全量有效控制范围约为K/span=16.0245–72.8571，能够出现背景较强失焦而前景像素结构仍较接近输入的情况；Any-to-Bokeh作为生成式方法也可能对毛发、纹理和亚像素边缘作轻微重建，从而降低Sobel相关。后者属于与现有观察一致的机制解释，**不是已经通过消融证明的唯一因果来源**。

因此，本轮只报告“在固定reproduced protocol下BokehMe的VEPI略高于Any-to-Bokeh”这一事实，不把它解释为BokehMe整体虚化质量更优，也不据此回调任何adapter。固定8段定性复核仍保留BokehMe的时变失焦强度、细结构软化，以及Any-to-Bokeh的细边残余偏差。原作者缺失的`davis_k16.csv`、精确25帧窗口与baseline私有DAVIS adapter没有恢复，且论文文字版Laplacian描述与公开`edge_batch.py`的Sobel实现存在差异；这些均属于当前排序与论文排序不能逐值等价比较的复现边界。

全量Normalized K/span约16.0245–72.8571，内部失焦强度并不统一；Dr实际图层最大半径14.37024px，LDF阈值空/全满帧均为0，但仍保留轮廓与分层失败。Dr的DAVIS行使用Full-public，不能与Synthetic的GT-alpha主行混称同一输入权限。

定性终审保留了breakdance的旁观者清晰层与分割变化、horsejump-high的时变虚化、kite-surf的细绳/水花、soapbox坡缘补全等现象。连续25帧以PNG接触图逐帧复核，没有宣称实时播放盲评、无闪烁或完整时序质量排名；全部MP4保留供播放。固定集合内未发现新的客观实现错误，按既定规则冻结，不扩大候选或为了排名重跑。

证据：[最终复核](records/davis-final-review-20260929.json)、[六方法逐帧指标](records/davis-six-method-50-20260929.json)、[全量完整性](records/davis-six-method-integrity-20260929.json)、[Table2](Benchmark与Baseline统一评价协议.md#table-2davis-real-world)。Efficiency正式保留下文Table 3A模型成本与Table 3B完整25帧流水；硬件版本分开报告。

## Table 3A 模型级计算成本（按硬件分开报告）

### RTX 4090 D：2026-10-01复测

| Method | Device | 模型段秒/输出帧 ↓ | Params (M) | GFLOPs | 设备采样峰值 (GiB) ↓ |
|---|---|---:|---:|---:|---:|
| Dr.Bokeh | RTX 4090 D | 0.342006 | 50.976 | N/A | ≈2.63 |
| BokehMe | RTX 4090 D | 0.012936 | 1.164 | N/A | ≈0.75 |
| MPIB | RTX 4090 D | 0.235940 | 41.528 | N/A | ≈2.18 |
| BokehDiff | RTX 4090 D | 0.921041 | 3493.432 | N/A | ≈17.57 |
| MagicBokeh B | RTX 4090 D | 1.269188 | 1398.545 | N/A | ≈5.13 |
| Any-to-Bokeh | RTX 4090 D | 0.803312 | 2457.810 | N/A | ≈42.14 |
| Ours | 待测 | — | — | — | — |

六种GPU方法各4个固定输入×5次正式重复，共120个正式计时样本。逐帧方法测第0帧；ATB保留25帧真实8/8/9输出调用，模型段约20.083秒再除以25，不是独立单帧在线延迟。参数量及GFLOPs覆盖口径复用，计时/显存重新测量。原始记录中的附带E2E字段不是六方法完整25帧Table 3B复测，不能据此填新卡3B。

证据：[4090 D六方法原始结果](records/table3-4090d-20261001.json)。两台GPU结果分开保留，不计算跨硬件方法加速比。

### RTX PRO 6000 Blackwell：2026-09-30原始测量

以下为旧卡配置及原始结果。固定 K4/8/16/32 的四个输入，每个预热1次、正式5次；逐帧方法使用第0帧，Any-to-Bokeh使用真实25帧 overlap 路径（每次8/8/9帧），按实际输出帧数归一。现有五种GPU方法及新增MagicBokeh各20次正式记录，没有挑选有利重复或更换RGB输入。硬件为 RTX PRO6000 Blackwell Server Edition、590.44.01 驱动，GPU 独占串行。

| Method | Device | Time / output frame (s) ↓ | Params (M) ↓ | GFLOPs / output frame ↓ | Peak VRAM (GiB, sampled) ↓ |
|---|---|---:|---:|---:|---:|
| Dr.Bokeh | RTX PRO 6000 Blackwell | 0.28808 | 50.976 | N/A† | ≈2.88 |
| BokehMe | RTX PRO 6000 Blackwell | 0.01503 | 1.164 | N/A† | ≈1.01 |
| MPIB | RTX PRO 6000 Blackwell | 0.25277 | 41.528 | N/A† | ≈2.43 |
| BokehDiff | RTX PRO 6000 Blackwell | 1.10397 | 3493.432 | N/A† | ≈18.37 |
| **MagicBokeh（B）** | RTX PRO 6000 Blackwell | **1.20890** | **1398.545** | N/A‡ | **≈5.30** |
| **Any-to-Bokeh** | RTX PRO 6000 Blackwell | **0.50464** | **2457.810** | N/A† | **≈38.47** |
| Ours | GPU | later | later | later | later |

表中 Time 是同步的核心模型/renderer执行，不是完整视频 E2E。Any-to-Bokeh 的 25 帧真实模型段约 12.616 s，再按输出帧数归一，因此 0.50464 s/output-frame 不能称为“独立单帧在线延迟”。

MagicBokeh的1024×576原生输出由六个512×512重叠图块组成；B配置1.20890s包含六次生成器forward与作者GPU拼接。DAdepth及GT目标辅助选点、条件构造/传输、CPU PIL与AdaIN颜色校正不计入Time；其参数也不进入生成器Params。实际调用生成器参数包括UNet、ControlNet、VAE及每图块调用的text encoder。Torch allocated/reserved峰值为3.89219/4.62695GiB。该路径的显存低于ATB、全帧耗时高于ATB；其GT目标辅助点控及自身估深与其他方法的输入权限不同，随结果披露；原Auto-LDF/空提示词1.17280s留作历史对照。‡ 完整算子覆盖尚未建立，GFLOPs保持N/A。

参数量按实际调用的神经模块中独立参数张量计数：ATB包含UNet、VAE和image encoder，排除未被forward调用的ff_k等权重；Dr包含LaMa，不把未调用的LDF/MiDaS算入。设备显存采用0.2秒监测与快速结束点采样，包含上下文与缓存，故标注≈；精确PyTorch allocated/reserved峰值另保存在记录中。

† 独立dispatcher profiling已执行，但可计卷积/矩阵乘法/attention算术并不覆盖自定义CUDA、FFT、归一化、逐元素、插值及随机控制等全部路径，因此总GFLOPs继续记N/A，不用部分计数冒充完整复杂度。

跨4个K的时间均值/标准差：

| Method | Mean (s/output-frame) | Std |
|---|---:|---:|
| Dr.Bokeh | 0.288076 | 0.000281 |
| BokehMe | 0.015025 | 0.001908 |
| MPIB | 0.252766 | 0.004515 |
| BokehDiff | 1.103966 | 0.010659 |
| MagicBokeh（B） | 1.208896 | 0.036781 |
| Any-to-Bokeh | 0.504643 | 0.000449 |

论文硬件和私有计时边界没有完全复原，因此本表只称 ATB-style reproduced cost，不将本轮与论文 Time 直接相除宣称同硬件加速比。

证据：[原五方法计时](records/efficiency-3a-20260930.json)、[最终复核及覆盖边界](records/efficiency-3a-final-review-20260930.json)、[计时入口一致性](records/efficiency-entry-parity-20260930.json)、[MagicBokeh B同配置补测](records/magicbokeh-frozen-B-table3-20260930.json)、[原生Auto-LDF对照](records/magicbokeh-native-and-table3-20260930.json)。

## Table 3B：25-frame Video E2E Efficiency under Given Inputs

**正式保留，与Table 3A互补。** Table 3A回答模型计算成本，Table 3B回答给定输入后25帧实际流水耗时；它与低帧率几何Track B没有关系。2026-10-01按用户更正恢复本表，不重新生成或改写已有实测结果。

**旧硬件实测**：下表来自旧RTX PRO 6000及旧机CPU的2026-09-30测量，六方法各20次正式重复，原始结果保留。**本轮当前4090 D及Xeon 8481C的七方法完整25帧视频E2E实测已完成，见下方当前硬件表。** MagicBokeh B按下述完整边界实测，不将generator-only时间充作E2E，也不给旧数据更换设备标签。

| Method | Device（旧机） | Core / 25f (s) ↓ | E2E-given-inputs / 25f (s) ↓ | FPS (E2E) ↑ | 峰值内存 (GiB, sampled) ↓ |
|---|---|---:|---:|---:|---:|
| DeepLens | CPU / TF1.15 / 8 threads | 25.370 | 26.452 | 0.945 | ≈1.36 RSS |
| Dr.Bokeh | RTX PRO 6000 Blackwell | 7.178 | 20.621 | 1.212 | ≈2.88 VRAM |
| BokehMe | RTX PRO 6000 Blackwell | 0.351 | 1.186 | 21.082 | ≈1.01 VRAM |
| MPIB | RTX PRO 6000 Blackwell | 6.164 | 7.681 | 3.255 | ≈2.43 VRAM |
| BokehDiff | RTX PRO 6000 Blackwell | 27.384 | 28.166 | 0.888 | ≈18.34 VRAM |
| Any-to-Bokeh | RTX PRO 6000 Blackwell | 12.628 | 14.084 | 1.775 | ≈38.47 VRAM |
| MagicBokeh B | 待测 | — | — | — | — |
| Ours | 待测 | — | — | — | — |

### 当前4090 D及CPU视频端到端效率（2026-10-01，已完成）

七方法实际运行完整25帧，四固定输入各预热1次、正式5次，共140个正式样本、28次完整预热。下表为均值±标准差；std来自20个正式样本，包含四片段差异，单K五次重复统计另存。FPS=25/平均E2E秒数。

| 方法 | 当前设备 | Core / 25f (s) | E2E-given-inputs / 25f (s) | FPS | GPU采样峰值 / GiB | CPU RSS / GiB |
|---|---|---:|---:|---:|---:|---:|
| DeepLens | Xeon 8481C / TF1.15 / 8 threads | 46.733 ± 2.104 | 48.789 ± 2.151 | 0.512 | — | 1.524 |
| Dr.Bokeh | RTX 4090 D | 8.584 ± 0.005 | 21.789 ± 0.518 | 1.147 | 2.626 | — |
| BokehMe | RTX 4090 D | 0.293 ± 0.077 | 1.139 ± 0.075 | 21.953 | 0.755 | — |
| MPIB | RTX 4090 D | 5.956 ± 0.163 | 7.536 ± 0.213 | 3.317 | 2.181 | — |
| BokehDiff | RTX 4090 D | 23.151 ± 0.013 | 23.957 ± 0.053 | 1.044 | 17.569 | — |
| MagicBokeh B | RTX 4090 D | 32.571 ± 0.784 | 36.570 ± 0.879 | 0.684 | 6.786 | — |
| Any-to-Bokeh | RTX 4090 D | 20.113 ± 0.020 | 21.860 ± 0.130 | 1.144 | 42.136 | — |

GPU补充峰值使用PyTorch口径，CPU不填PyTorch显存：

| GPU方法 | Torch allocated / GiB | Torch reserved / GiB |
|---|---:|---:|
| Dr.Bokeh | 1.939 | 2.209 |
| BokehMe | 0.314 | 0.348 |
| MPIB | 1.108 | 1.764 |
| BokehDiff | 14.876 | 17.152 |
| MagicBokeh B | 5.139 | 6.361 |
| Any-to-Bokeh | 34.293 | 41.713 |

硬件为4090 D / 49140MiB / 驱动580.105.08，以及Intel Xeon Platinum 8481C；容器配额20核、90GiB。GPU沿用PyTorch2.8.0+cu128 / CUDA12.8环境。DeepLens固定原生TF1.15 CPU、intra-op8/inter-op1线程、oneDNN关闭。内存沿用0.2s采样，模型加载峰值排除。

MagicBokeh B的Core=DAdepth网络forward+六图块生成器/GPU拼接，E2E另计读取、每帧估深外围处理、GT目标辅助选点、条件/传输和PIL/AdaIN；与模型计算成本表的generator-only口径不同。CPU RSS不能当作GPU显存。原有模型成本、质量及DAVIS结果未重跑，模型/精度/adapter/batch/缓存未改。

证据：[当前服务器七方法原始逐次数据、std/by-K及旧新硬件对照](records/video-e2e-4090d-20261001.json)。完整每帧计时单元/日志保存在`outputs/video-e2e-4090d-20261001/`及对应`logs/`；已完成项目可按入口恢复而跳过，未完成尝试保留并整项重启以保持随机流。旧PRO6000及旧CPU表保留原硬件标注，旧机MagicBokeh E2E缺项仍留空。

E2E-given-inputs从已给输入的读取/解码开始，包括适配、必要分层/修复、推理、传输和后处理，直到25张1024×576 CPU RGB uint8就绪；排除模型加载、首次CUDA context、benchmark外部geometry/GT alpha生成、最终写盘及MP4编码。FPS=25/平均E2E秒数，Core同时单列。

证据：[3B原始计时](records/efficiency-track-a-3ab-20260930.json)、[3B最终复核](records/efficiency-3b-final-review-20260930.json)、[阶段归档](历史阶段记录/2026-09-30_Efficiency_3A与3B测量.md)。旧数据的std、by-K、内存采样与CPU/GPU边界保持原记录。MagicBokeh B本轮按下述完整方法边界补测。

**MagicBokeh B本轮边界**：沿用冻结原生模型与点控adapter，E2E包括每帧重新DAdepth估深、读取已给目标metadata/layer alpha并计算可见区域、B选点、PNG8与defocus/focus-mask条件构造/传输、六图块生成与GPU拼接、CPU PIL/AdaIN及25帧uint8 RGB输出。外部benchmark geometry/alpha原始生成、模型加载、写盘/编码与指标排除；LDF自动选焦不属于B。Core单列DAdepth网络forward及生成器GPU段之和，与模型计算成本表的generator-only边界明确区分。逐帧种子/精度从原冻结B完整入口继承，未复用旧深度/条件/输出。配置`configs/magicbokeh-video-e2e.json`，入口`scripts/run_video_e2e.py`；当前执行产物在`outputs/video-e2e-4090d-20261001/`，原始日志在对应`logs/`。

## Any-to-Bokeh 显存与用时拆解（4090 D，2026-10-01/2026-10-08）

以下为 Any-to-Bokeh 原生实现的模块级拆解，服务于效率机制分析，不替换 Table 3A/3B 的正式计时。四固定片段（repro_004/006/003/000，K4/8/16/32）、完整25帧、1024×576、原生8/8/9窗口调用。阶段耗时带同步插桩，只用于归因，不能当作正式单帧延迟。

### 阶段用时与显存（8次正式诊断）

| 阶段 | 时间均值 ± 标准差（s/25帧） | 占插桩Core | allocated峰值 / GiB | reserved峰值 / GiB | 调用次数／输入帧槽 |
|---|---:|---:|---:|---:|---|
| CLIP 图像编码 | 0.160048 ± 0.000212 | 0.793% | 11.9249 | 41.6934 | 3／48 |
| RGB VAE编码及采样/缓存 | 4.580892 ± 0.005578 | 22.711% | 34.2929 | 41.6934 | 3／48 |
| CoC VAE编码及采样 | 3.787406 ± 0.003694 | 18.777% | 34.2942 | 41.6934 | 3／48 |
| UNet 时空条件处理 | 3.741973 ± 0.005802 | 18.552% | 14.3912 | 41.6934 | 3／48 |
| 重叠拼接＋时序VAE解码 | 7.841134 ± 0.001810 | 38.874% | 29.6453 | 41.6934 | 拼接3／48；VAE decode4／32 |
| 未归入上述阶段的Core时间 | 0.059259 ± 0.001479 | 0.294% | 未测 | 未测 | 未单独测量 |
| **完整带插桩Core** | **20.170711 ± 0.011979** | **100.000%** | **34.2942** | **41.6934** | pipeline 3；RGB 48＋CoC 48 |

- VAE相关阶段（RGB编码＋CoC编码＋重叠/解码）合计 **16.2094秒，占80.36%**；旧RTX PRO 6000同口径为81.01%。新卡各阶段整体约为旧卡的1.5–1.7倍，不是VAE单独变慢。
- 阶段的allocated峰值含常驻权重与缓存，**不能相加**；reserved包含预热后已持有的预留池，不代表模块新增占用。
- 插桩Core（20.1707秒）比正式未插桩Core（20.1128秒）高0.0579秒（0.288%），差值含插桩与单独运行波动，不是同步开销的独立因果估计。

### 显存构成

| 观测点／对象 | allocated | reserved | active | inactive split | requested | RGB storage |
|---|---:|---:|---:|---:|---:|---:|
| 模型加载完成 | 4.8990 | 5.0977 | 4.8990 | 0.1987 | 4.8552 | 0.0000 |
| 预热结束 | 11.3769 | 41.6914 | 11.3769 | 2.8379 | 11.3319 | 6.4688 |
| RGB/CoC编码出口（阶段峰值） | 34.2890/34.2902 | 41.6914 | 11.7912 | 2.8728 | 11.7456 | 6.4688 |
| 重叠＋解码出口（阶段峰值） | 29.6123 | 41.6914 | 11.6642 | 2.8847 | 11.6197 | 6.4688 |
| 完整clip结束 | 11.3778 | 41.6914 | 11.3778 | 2.9445 | 11.3319 | 6.4688 |

- **持续RGB特征缓存6.46875 GiB**，四级（16×128×576×1024 / 16×128×288×512 / 16×256×144×256 / 16×512×72×128，均float32），存储彼此独立；跨CoC/UNet/解码保留到下一次RGB替换。另有latent cache逻辑0.000274658 GiB、底层storage 0.001098633 GiB。
- 参数去重storage（全部已加载对象，不等同效率Params）：UNet FP16 3.494874 GiB、CLIP FP16 1.177335 GiB、VAE FP16 0.182976 GiB；RGB/CoC编码时整个VAE转FP32为0.365952 GiB，进入UNet与解码前恢复FP16。
- 编码阶段高requested区间由**5笔各4.5 GiB并发分配**构成（入口以上请求增量22.5 GiB）；重叠＋解码为18.0021 GiB。该增量是阶段进入值以上的请求峰值，不命名为纯激活显存。
- 每25输出帧，RGB/CoC各处理48个编码帧槽（45次真实帧访问、3个padding，其中20次为重叠窗口重复），解码32个帧槽（29＋3，重复4次）。**重复和padding不是已证明可删除的计算量**，还涉及随机采样、缓存索引与时序解码上下文。
- 本机**设备采样峰值42.1162 GiB**，高于Torch allocated 34.2942 GiB，差值来自预留池、CUDA context及非PyTorch分配。未在32GB显卡实测，不宣称32GB可用。

### 首个单因素实验：encoder unit-scale除法消除（2026-10-08）

唯一算法变化：VAE encoder残差返回表达式中，仅当`output_scale_factor`严格等于1时跳过除法，保留非原地加法；其他scale保留原除法。decoder、UNet、窗口/overlap、采样、FP32 upcast、缓存生命周期、allocator设置均不变。

| 片段/K | 本轮原版Core（s） | 候选Core（s） | 耗时变化 | 原版/候选 |
|---|---:|---:|---:|---:|
| repro_004/K4 | 20.26435 ± 0.01294 | 20.03947 ± 0.01490 | −1.110% | 1.01122 |
| repro_006/K8 | 20.26618 ± 0.01772 | 20.05308 ± 0.02277 | −1.052% | 1.01063 |
| repro_003/K16 | 20.27302 ± 0.00942 | 20.04558 ± 0.01069 | −1.122% | 1.01135 |
| repro_000/K32 | 20.28723 ± 0.00744 | 20.05847 ± 0.01008 | −1.128% | 1.01140 |
| **四片段等权汇总** | **20.272695 ± 0.014724** | **20.049148 ± 0.016010** | **−1.103%** | **1.01115** |

| 指标 | 原版 | 候选 | 变化 |
|---|---:|---:|---:|
| 完整allocated峰值 / GiB | 34.294156 | 29.797452 | −4.496704（−13.11%） |
| 完整reserved峰值 / GiB | 41.693359 | 37.193359 | −4.500000（−10.79%） |
| 设备采样峰值 / GiB | 42.116211 | 37.616211 | −4.500000（−10.68%） |

- **正确性**：四片段及对应预热全部通过——浮点不同元素数、不同位模式数、RGB8不同字节数均为0；sample调用与顺序、RNG状态（40个对应位置）、四级RGB缓存与latent cache逐字节一致。
- **机制证据**：配对memory history显示encoder残差返回分配120→60次，其中4.5 GiB返回请求24→12次；编码高requested区间由5笔并发4.5 GiB降为4笔（约18.003296 GiB），内部高点从残差返回转至conv1附近。非encoder的ResNet返回位置244次分配不变，与decoder/UNet未应用该改动一致。
- **边界**：仅保证已测四片段及预热的等价性，未扩大为所有输入保证；时间收益是该设备/输入上的实测，不是普遍加速保证。候选用显式开关单列，**原正式效率表与原生基线不替换**；定位为ATB实现效率优化，不是Ours创新。16→8编码微批量保留为已被本轮替代的暂缓历史候选。

证据：[4090 D模块与allocator诊断](records/any2bokeh-native-module-profile-4090d-20261001.json)、[单因素实验](records/any2bokeh-encoder-scale1-experiment-4090d-20261008.json)；旧卡对照见[首轮模块分析](records/any2bokeh-native-module-profile-20260930.json)。产物`outputs/any2bokeh-scale1-experiment-20261008/`、`outputs/any2bokeh-allocator-supplement-20261001/`、`outputs/any2bokeh-memory-history-20261001/`。

## 当前研究主线与执行边界（2026-10-01更正）

经当前阶段讨论，低帧率 Depth 不再作为必须坚持的创新方向。原 Track B / OPPO-style Low-FPS 自建 benchmark 从当前前置实验中移除，不继续为了填表执行 30/15/10/5/2 fps sweep。

当前限定的既有性能评价已完成：新卡模型计算成本保持原已完成结果，本轮只补七方法完整25帧视频端到端效率。编码分批、缓存优化、失败分析等继续作为后续候选，不因下列研究顺序自动执行：

1. **MagicBokeh已收尾**：原生验证、40帧三候选、用户冻结B、200×25评价及B同配置Table 3A全部完成；额外GT目标辅助选点权限已披露，不重复选方案或推理；
2. **Any-to-Bokeh效率机制验证**：首轮VAE/UNet/解码时间与峰值归因已完成，下一步区分FP32编码activation、持续缓存和重叠重复计算的贡献；
3. **Any-to-Bokeh Failure Analysis**：基于现有200段Synthetic和DAVIS固定视频，系统筛选边界/遮挡/细结构、大运动、强虚化、focus/control变化等可重复失败；
4. **冻结 Ours**：只选择有明确问题证据、可做受控消融且具有实际价值的方向，不再为了符合旧计划强行加入低帧率Depth completion、Token Merging或三头结构。

MagicBokeh已补DAVIS50×25量化，原六方法结果与ATB机制分析材料保持独立；新增视觉判断待用户复核。首轮profile确认VAE编码/解码约占81%模型段时间，RGB/CoC编码峰值约34.29GiB，RGB FP32缓存6.47GiB；下一步验证缓存/重叠编码的优化假设及失败机制，Ours尚未冻结。证据：[模块分析](records/any2bokeh-native-module-profile-20260930.json)。
