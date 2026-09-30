# MagicBokeh 原生复现、目标点选焦与Synthetic评价

本阶段已结束（2026-09-30）。当前实时任务见中央进度，以下保留按时间发生的原生验证、候选预检和全量过程；过程中的“待预检”“运行中”是当时状态，不是重复启动指令。

### MagicBokeh 原生复现与选焦问题（2026-09-30 16:56，北京时间）

官方源码`c672e63`保持未修改，复用已有环境、SD2.1与lora.pkl，补下载官方DAdepth.pth。三张完整官方示例分别为1562×1024、1564×1024、4032×3024；附带条件与从RGB重新预测条件两条路径各201个图块，全部finite、输出尺寸与拼接覆盖通过。重新预测深度与附带文件最大差≤0.000262，mask最大差1、最终PNG通道最大差2；同条件/同种子的独立适配器在第一张完整示例上与官方CLI输出逐值一致。完整图像对照已查看，没有配对GT，不报告画质分数或视频时序结论。

**已验证的适配问题**：官方batch路径以LDF非零mask内的深度中位数自动选焦，不能默认对应benchmark指定焦平面。四个固定计时场景的目标点与自动选焦在原生预测深度坐标中相差约0.180/0.353/0.192/0.667；K32的repro_000目标平面最大连通区域约4.23%像素满足原生清晰条件。这个数是条件覆盖诊断，不是最终图像质量分数。共享geometry仅用于标出诊断目标/点，没有输入本轮生成器。

**当时待解决问题（已由下节推进）**：验证官方点选焦对目标平面的适配。后续已完成预检并由用户冻结B；此处原生记录不作为重复启动指令。

证据：[原生复现、Table 3与选焦诊断](../records/magicbokeh-native-and-table3-20260930.json)。入口`validate_magicbokeh_native.py`、`measure_magicbokeh.py`及`configs/magicbokeh-native.json`；大型输出与日志在`outputs/magicbokeh-native-20260930/`及对应`logs/`。该原生验证轮任务已结束，正在运行的全量B见下节。

### MagicBokeh 点选焦预检、B冻结与全量执行（2026-09-30）

**可见性**：按目标层alpha乘所有更近层的透射率构造可见alpha，明确扣除遮挡；5000/5000帧均有alpha>0.9的安全区域，没有不可见帧、回退或目标实例切换。A/B都使用最大8连通域、图像边界零填充与欧氏距离；多解固定row-major。

**预检**：固定8段×0/6/12/18/24帧，Auto-LDF、A最内部点、B自身预测深度中位数的代表像素共120张输出。三候选统一官方demo提示词`high quality,`、同一帧种子、K与阈值；后续逐帧、无平滑。40帧的80个点均在可见目标内、所有输出finite/尺寸正确，8张接触图均已查看；没有读取Bokeh GT或计算PSNR等来选方案。

| K | Auto目标清晰区覆盖均值 | A覆盖均值 | B覆盖均值 | A非目标清晰区比例 | B非目标清晰区比例 |
|---|---:|---:|---:|---:|---:|
| 4 | 1.0000 | 1.0000 | 1.0000 | 0.9049 | 0.9226 |
| 8 | 0.9635 | 0.9927 | 0.9963 | 0.5306 | 0.5356 |
| 16 | 0.8424 | 0.9490 | 0.9570 | 0.3350 | 0.3183 |
| 32 | 0.2222 | 0.4379 | 0.7214 | 0.0784 | 0.0824 |

上述指标仅检查原生focus-mask条件，target定义为可见alpha>0.5，不能代替最终画质/时序分数。K4条件容差为0.8，非目标清晰区比例高与原生阈值有关，不搜索K/阈值。K32 repro_000的A点击位于狮子脸部、margin约160px，但目标覆盖平均6.50%；B约61.98%。RGB估深产生目标内部深度起伏，而proxy为对象指定平面，局部正确点击不保证整个目标符合清晰条件；B也保留部分目标失焦。

**用户确认**：已明确选择“改用B，披露目标区域深度辅助选点”。正式定义为 **Target-guided Depth-Medoid Point-Focus Adapter**：GT可见目标区域和原生预测深度辅助选出一个实际像素，focus仍读取该像素深度/255；它具有额外GT目标控制信息，不称automatic autofocus或同输入权限公平比较。A及Auto诊断输出保留。新进程入口重放B与预检PNG逐值一致，原生深度预测也一致。

**全量**：配置`configs/magicbokeh-track-a-frozen.json`，入口`run_magicbokeh_track_a.py --workers 4`；四路scene分片互不重叠，每帧种子`42+25*manifest_index+frame_index`，不依赖调度顺序。复用40张B预检输出，其余写入`outputs/magicbokeh-point-focus-20260930/full/`，日志在对应`logs/magicbokeh-point-focus-20260930/`；不覆盖既有六方法。`evaluate_magicbokeh_track_a.py`按10段批次调用已有TorchMetrics/LPIPS/RAFT/RM入口，末尾计算FVD与补充VFID-I；主Table 1仍是六项指标。全量数值未完成，不提前填表。

**下一步**：完成5000帧与200段指标复核；GPU独占后使用`measure_magicbokeh.py --focus-control frozen-b`测同一B配置Table 3，保留原Auto-LDF成本；随后转ATB profiling。证据：[目标统计、三候选预检、用户选择与冻结入口](../records/magicbokeh-point-focus-preflight-20260930.json)。


## 全量收尾与冻结结果

200段×25帧、1024×576、5000张PNG完整。四路互斥scene分片运行，40张预检B输出及其自身深度复用，其余帧写入新的派生目录。所有PNG数量/尺寸/模式/CRC、native depth finite、点在目标内、focus等于自身PNG8深度点击值、帧种子与K检查通过；GT的FVD/VFID-I特征与原缓存逐值一致，六项指标finite、by-K加权均值一致。

| 指标 | 实测 |
|---|---:|
| PSNR / TorchMetrics | 25.120155 |
| SSIM / TorchMetrics | 0.799536 |
| LPIPS / squeeze | 0.197734 |
| FVD / VideoMAE | 97.789874 |
| FD / RAFT | 0.870974 |
| RM / 24 transitions | 0.026840 |
| VFID-I / 补充 | 14.985397 |

这些是原生RGB估深、GT目标辅助点控与生成器整条路线在当前平面图层proxy上的结果。输入权限及控制坐标和外部geometry方法不同；保留质量差距及部分目标失焦，不按分数搜索K、gamma、阈值或换深度模型。B继续逐帧推理，无平滑或传播。

### 冻结B的模型成本

同四个固定RGB第0帧、K4/8/16/32，各预热1次、正式5次；每个1024×576输出用六个512×512图块。B提示词为官方demo的`high quality,`，与正式质量配置一致。模型段1.208896s/output-frame，实际调用Params1398.545M，采样峰值5.299805GiB；Torch allocated/reserved3.892187/4.626953GiB，GFLOPs=N/A。DA-depth、GT目标选点、输入构造/传输和CPU PIL/AdaIN不计入模型段；对应参数也不算生成器参数。

原生Auto-LDF/空提示词的1.172797s成本是较早的原生验证结果，保存在原记录，不覆盖或混入B主行。

### 证据与复现入口

- [原生验证与Auto-LDF成本](../records/magicbokeh-native-and-table3-20260930.json)；
- [三候选预检与用户选择B](../records/magicbokeh-point-focus-preflight-20260930.json)；
- [全量200段与完整性复核](../records/magicbokeh-track-a-200-20260930.json)；
- [B同配置20次正式成本测量](../records/magicbokeh-frozen-B-table3-20260930.json)；
- 配置`configs/magicbokeh-track-a-frozen.json`；入口`run_magicbokeh_track_a.py`、`evaluate_magicbokeh_track_a.py`、`summarize_magicbokeh_track_a.py`；
- 大型产物`outputs/magicbokeh-point-focus-20260930/`及对应`logs/`，源数据及既有六方法不覆盖。

阶段完成后转入Any-to-Bokeh的模块级时间/显存分析。未扩展MagicBokeh DAVIS，也未恢复Low-FPS专项。
