# 2026-09-30 Efficiency：Table 3A与3B测量

本阶段已完成。此文件保存准备与测量过程，不作为实时进度入口。3A模型级成本与3B完整视频流水分别报告；协议、权重及输入权限不变。

## 准备与3A完成后的过程记录

以下内容从中央进度迁入，是3B运行期间的状态快照；其中“正在测量”等描述按当时理解，最终状态见下一节。

### Efficiency：Table 3A / 3B 已分开

Efficiency 不再用一张表混合“模型复杂度”和“实际视频流水”。统一评价协议已拆成：

- **Table 3A — ATB-style Model-level Computational Cost**：回答“模型/方法本身算一次有多贵”。统一1024×576、batch=1，报告 Time / output frame、Params、GFLOPs / output frame、Peak VRAM。逐帧方法直接测单帧；原生视频方法按真实执行单元计时，再按实际输出帧数归一，不制造不存在的单帧入口。DeepLens当前native TF1.15 CPU路径不进入3A GPU主表，保留到3B报告实际CPU成本。
- **Table 3B — 25-frame Video E2E Efficiency under Given Inputs**：回答“在与Synthetic Track A相同输入权限下，一段25帧视频实际多久产出”。当前已准备的4段Efficiency clips、warm-up与重复计时入口对应这张表。E2E从已给RGB/Geometry/Focus/K（以及冻结配置允许的Dr.Bokeh GT alpha）读取开始，计入adapter、方法内部必要的分层/repair/inpainting、模型/renderer、postprocess及最终转CPU RGB uint8；排除model load、首次CUDA context、benchmark外部Depth/Geometry/GT-alpha生成和最终写盘/MP4编码。FPS固定由25/E2E计算。

方法权限继续严格跟 Track A：Dr.Bokeh 用 GT-alpha reconstruction adapter，不改成 DAVIS Full-public；DeepLens 用 Track A external geometry + native single-SR，不跑DAVIS DepthNet；BokehMe Normalized+gamma2.2、MPIB gamma2.2、BokehDiff Normalized-B、Any-to-Bokeh官方冻结视频配置。Core Time 与 E2E-given-inputs 分开，最终统一到CPU RGB uint8作为E2E终点。

2026-09-30 13:09北京时间执行最小CUDA探针，逐元素计算与三次1024方阵乘法均正确；探针退出后GPU由持续100%恢复0%、P8、约35W，连续空闲检查通过。没有更换驱动、重置GPU、改变功率/时钟或结束其他进程，具体驱动原因未唯一确定；证据见[GPU运行准备记录](../records/gpu-efficiency-readiness-20260930.json)。Table 3B GPU方法逐个串行执行，方法切换时最多等待30秒取得连续三个空闲读数，不放宽空闲条件；Table3A正式结果现已完成并复核，3B已启动、尚未完成；两者不能混为同一Time列。 2026-09-30实现对齐新版Core边界：外围传输/adapter从Core移出、Dr/BokehDiff移除计时内诊断拷贝，DeepLens补RSS，参数按实际调用模块统计；算子覆盖检查与latency重复分开。最小一致性检查已通过（Dr最大浮点差1.2e-7，BokehDiff逐值及随机状态一致）；本轮Table3A已完成：逐帧方法取固定4段各第0帧，ATB保留各完整25帧原生调用，均每单元预热1次、正式5次。3A产物`outputs/efficiency-3a-20260930/`，日志`logs/efficiency-3a-20260930/`。3B于14:12启动独立后台任务，使用`outputs/efficiency-track-a-3ab-20260930/`和同名logs目录；固定4段分别预热1次、正式5次，每次真实输出25帧，不用3A单帧结果外推。

### Table 3A 已完成，可用于本次汇报

五方法共100个正式计时样本，GPU串行独占；Dr/BokehDiff计时入口输出一致性验证通过。结果为：

| Method | Device | Time / output frame (s) ↓ | Params (M) ↓ | GFLOPs / output frame ↓ | Peak VRAM (GiB, sampled) ↓ |
|---|---|---:|---:|---:|---:|
| Dr.Bokeh | RTX PRO 6000 Blackwell | 0.28808 | 50.976 | N/A† | ≈2.88 |
| BokehMe | RTX PRO 6000 Blackwell | 0.01503 | 1.164 | N/A† | ≈1.01 |
| MPIB | RTX PRO 6000 Blackwell | 0.25277 | 41.528 | N/A† | ≈2.43 |
| BokehDiff | RTX PRO 6000 Blackwell | 1.10397 | 3493.432 | N/A† | ≈18.37 |
| Any-to-Bokeh | RTX PRO 6000 Blackwell | 0.50464 | 2457.810 | N/A† | ≈38.47 |
| Ours | GPU | later | later | later | later |

Core边界逐方法披露，Time单位为秒/输出帧；ATB真实25帧模型段约12.616秒，按输出帧归一，不能称为单帧在线延迟。GFLOPs已检查但完整路径未覆盖，主表记N/A，部分算术计数仅作诊断。采样显存与精确Torch峰值均保留，不能直接与不同GPU/计时边界的论文Time计算加速倍数。

[3A原始记录](../records/efficiency-3a-20260930.json)、[3A复核](../records/efficiency-3a-final-review-20260930.json)。用户优先完成的3A已用于汇报；**3B六方法25帧E2E-given-inputs正在串行测量**，完成后复核并填写统一协议及复现汇报，再进入Track B。


## 3B完成与阶段结论

| Method | Device | Core / 25f (s) ↓ | E2E-given-inputs / 25f (s) ↓ | FPS (E2E) ↑ | Peak memory (GiB, sampled) ↓ |
|---|---|---:|---:|---:|---:|
| DeepLens | CPU / TF1.15 / 8 threads | 25.370 | 26.452 | 0.945 | ≈1.36 RSS |
| Dr.Bokeh | RTX PRO 6000 Blackwell | 7.178 | 20.621 | 1.212 | ≈2.88 VRAM |
| BokehMe | RTX PRO 6000 Blackwell | 0.351 | 1.186 | 21.082 | ≈1.01 VRAM |
| MPIB | RTX PRO 6000 Blackwell | 6.164 | 7.681 | 3.255 | ≈2.43 VRAM |
| BokehDiff | RTX PRO 6000 Blackwell | 27.384 | 28.166 | 0.888 | ≈18.34 VRAM |
| Any-to-Bokeh | RTX PRO 6000 Blackwell | 12.628 | 14.084 | 1.775 | ≈38.47 VRAM |
| Ours | GPU | later | later | later | later |

**3B已完成并复核（2026-09-30）**：固定repro_004/006/003/000分别覆盖K4/8/16/32，每段完整25帧预热1次、正式重复5次；六方法共120次正式计时、3000帧输出，另有24次完整预热。14:12–14:54北京时间串行执行，未更改冻结adapter、模型、控制值或输入清单。表中为20次均值，FPS=25/平均E2E；逐次时间、样本标准差、by-K及原生调用均已复算。Any-to-Bokeh每段真实调用输出8/8/9帧，其余方法逐帧执行25次，未用3A单帧结果外推。

GPU为RTX PRO 6000 Blackwell Server Edition（590.44.01，Torch2.8/cu128）；DeepLens为Xeon Platinum8470Q、native TF1.15 CPU8线程，容器CPU配额22核。GPU显存和CPU RSS均为正式重复期间0.2秒采样峰值，不是精确瞬时峰值；精确Torch allocated/reserved另存。CPU内存与GPU显存不作同质排序，CPU与GPU时间不作无条件加速比。

E2E终点为25张1024×576的CPU RGB uint8；输入解码、adapter、必要分层/修复、推理、传输和后处理均计入，模型加载、外部geometry/alpha生成、最终写盘/编码不计。采用warm filesystem cache；同步Core计时的开销包含在E2E中。Dr.Bokeh继续使用Track A允许的GT-alpha reconstruction，DeepLens继续external geometry+single-SR+aperture saturation。

| Method | Core mean ± std (s/25f) | E2E mean ± std (s/25f) |
|---|---:|---:|
| DeepLens | 25.370 ± 1.239 | 26.452 ± 1.256 |
| Dr.Bokeh | 7.178 ± 0.014 | 20.621 ± 0.814 |
| BokehMe | 0.351 ± 0.045 | 1.186 ± 0.113 |
| MPIB | 6.164 ± 0.182 | 7.681 ± 0.207 |
| BokehDiff | 27.384 ± 0.029 | 28.166 ± 0.084 |
| Any-to-Bokeh | 12.628 ± 0.011 | 14.084 ± 0.062 |

| Method | K4 E2E (s) | K8 E2E (s) | K16 E2E (s) | K32 E2E (s) |
|---|---:|---:|---:|---:|
| DeepLens | 27.109 | 26.153 | 25.633 | 26.913 |
| Dr.Bokeh | 21.312 | 21.273 | 20.515 | 19.385 |
| BokehMe | 1.282 | 1.209 | 1.052 | 1.200 |
| MPIB | 7.783 | 7.571 | 7.448 | 7.924 |
| BokehDiff | 28.232 | 28.229 | 28.105 | 28.098 |
| Any-to-Bokeh | 14.089 | 14.144 | 14.008 | 14.097 |

证据：[3B逐次计时](../records/efficiency-track-a-3ab-20260930.json)、[3B最终复核](../records/efficiency-3b-final-review-20260930.json)。六方法均无失败或重跑，所有正式重复保留；既有3A数值不变。表格及讨论已写回统一协议和复现汇报，下一步为Track B Clean公共时间补齐协议。
