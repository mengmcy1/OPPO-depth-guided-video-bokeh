# Any-to-Bokeh Synthetic Evaluation 公开代理复现：200-video 定量验证

> 任务：使用公开可获得组件重建 Any-to-Bokeh synthetic evaluation pipeline，并验证官方权重的指标量级
>
> 当前状态：✅ **200/200 已完成**（2026-09-28 06:26 UTC（北京时间14:26））
> 运行记录：sRGB输入配置落盘至全量汇总约170.3分钟，含并发和中断恢复，不含此前图层/几何构造。
> 数据规模：200 场景 × 25 帧 = 5000 帧（RGB + GT + Prediction 完整）
> 补充指标：2026-09-28完成TorchMetrics PSNR/SSIM、VideoMAE FVD、I3D VFID-I，见下表及第七节。
>
> 核心方案：Any-to-Bokeh 公开场景生成逻辑 + MPIB 官方 ray-tracing renderer proxy + Any-to-Bokeh 官方权重
> 新增MPIB与BokehMe baseline：各自双候选200段及全部指标已完成，分别冻结gamma=2.2与Normalized/gamma=2.2；详见8.1–8.2。BokehMe与MPIB的部分排序尚未恢复，不宣称整体性能版图已对齐。

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

下一阶段先在这套已冻结的ATB-style reproduction benchmark上比较Ours与Any-to-Bokeh；完成这一层证据后，再建立本项目自己的独立benchmark。后者将重新统一baseline的信息权限与低帧率粗深度输入，届时Dr.Bokeh应优先重新评估Full-public等不读取GT alpha/hidden layers的公开完整pipeline，而不能默认沿用本阶段GT-alpha权限。两套benchmark承担不同目的，结果必须分开命名和报告。

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

**最后更新**：2026-09-29
**状态**：Any-to-Bokeh公开代理复现及author-public指标已完成；MPIB/BokehMe/Dr.Bokeh Track A已填真实结果，BokehDiff双200段及A/B复核完成且Normalized已选为主表展示配置；Dr.Bokeh新输入协议200段已收尾，DeepLens按用户冻结的单级SR+饱和协议完成200段，见8.7；范围与版本依据保留在8.6。

补算实现及权重来源：[作者公开指标代码](https://github.com/vivoCameraResearch/any-to-bokeh/blob/f7a16959dd219080fb8eec0d32bb9602d8970c22/metrics/vid_metrics.py)、[cdfvd固定版本](https://github.com/songweige/content-debiased-fvd/tree/a1e037ab7cb087debd2221d14ae4a001ec054201)、[作者指定I3D权重](https://huggingface.co/zhengchong/VFID)、[官方VideoMAE2权重](https://huggingface.co/OpenGVLab/VideoMAE2)。
