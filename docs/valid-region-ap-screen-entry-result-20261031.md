# 两组3轮AP对照：启动前验收结果（2026-10-31）

本轮列出的启动前验收已通过。两组真实训练均未启动，optimizer更新次数0；没有运行20步训练或重新训练baseline。新增执行入口可在明确授权后运行冻结A/B各3轮，不自动续训。

## 已完成项目

| 项目 | 实际检查 | 结果 |
|---|---|---|
| 两组真实配置加载 | MMEngine加载冻结control/candidate；除开关和输出路径外配置一致 | 通过 |
| 原模型起点 | 两组加载E0，完整模型状态与原checkpoint一致 | 通过 |
| 新试验起始状态 | epoch/iter均0，优化器状态为空；基础LR1e-5、骨干按0.1倍率 | 通过 |
| 数据与迭代数 | 3937训练图，每轮1969批，官方691评价图，原重复图保留 | 通过 |
| 跨进程保存恢复 | 两组分别原生Runner保存，在新Python进程恢复模型、345组非空优化器状态、空调度列表、计数、开关及显式RNG | 精确一致 |
| 3轮停止 | 原生EpochBasedTrainLoop+无参数假模型，5907次控制流迭代后停止；额外迭代拒绝 | 通过，无真实更新 |
| 总时间预算 | 注入到期时间，在下一更新前触发保存并停止 | 通过 |
| 非有限数值 | NaN loss在backward前拒绝，Inf梯度在step前拒绝，CPU标量及优化器状态不变 | 通过 |
| OOM中断 | 向共用编排注入OOM，失败结果落盘、不评价、不进入第二组 | 通过，无真实OOM |
| 固定末轮评价 | 仅完整epoch3/iter5907、正确开关/配置、非诊断/非partial检查点允许最终评价 | 正负对照通过 |
| 未授权调用 | 不带--execute退出，不启动训练 | 通过 |
| 评价回归 | 新入口完整评估未更新E0的691图，预测与原结果逐项比较 | 69100条精确相同 |

详细范围见[入口验收范围](valid-region-ap-screen-entry-checks-20261031.md)。训练预算、收益门槛保持[AP短程计划](valid-region-ap-screen-plan-20261031.md)，没有因本轮检查修改。

## 保存恢复没有偷偷训练

为了覆盖非空AdamW动量，又不新增参数更新，使用原E0已有优化器状态作序列化样本。诊断检查点的epoch/iter是明确设置的测试值，标记diagnostic_only=True；它不是新跑完一轮的权重，也绝不能作为本次微调起点或最终评价输入。

实际试验从原E0模型权重开始，新建优化器；验收确认初始优化器为空。开关不在state_dict中，因此另存screen_contract（组别、开关、配置哈希、head哈希和预算），恢复与评价均核验。Python/NumPy/Torch/CUDA随机状态显式随检查点保存并恢复，原生Runner之外的这一步有单独检查。

未承诺中途worker预取队列/数据游标的逐位续接，当前执行入口不自动恢复中断任务。若将来中断，先检查现场再决定恢复；不能将状态序列化通过等同于真实更新轨迹已经验证。

## 同一评价尺子的回归结果

新入口使用原数据、推理尺度和COCO设置，只对未更新的E0做回归：

| 指标 | 未四舍五入值 |
|---|---:|
| Mask AP | 32.62221771799669 |
| AP50 | 47.63731730870105 |
| AP75 | 35.254971962831405 |
| AP_small | 8.156236762716086 |
| AP_medium | 25.053321386655504 |
| AP_large | 47.76955952322051 |

全部69100条segm预测与旧文件内容逐项精确一致，本地下载后再次核对。不是新方案得到的AP，也不是对候选效果的验证。正式两组只有各自完整epoch3才可进入该评价路径。

## 执行入口与限制

新增scripts/run-valid-region-ap-screen.py，默认拒绝执行，只有显式--execute且验收报告passed、源码/配置哈希未变时才可开始。启动时重新检查原权重与数据哈希，获取项目训练锁，拒绝已有工作/评价目录，防止重复启动或覆盖旧结果。

运行时对两组同样增加ScreenFiniteOptimWrapper（有限性检查）与APScreenGuard（预算、开关/配置元数据、随机状态保存），不改变优化数学参数。固定先A后B，每组3轮后评价，再统一比较AP。异常写completed=False并停止，不自动重试/降参数/追加预算。4小时预算在训练批次及评价图片边界检查，不是强杀正在执行的GPU算子的精确秒表；初始化、保存及COCO汇总也有耗时。

若之后明确授权，操作入口为：

```bash
cd /root/autodl-tmp/underwater-segmentation
.venv/bin/python -B scripts/run-valid-region-ap-screen.py --execute
```

本轮没有执行此命令。执行前仍应按AGENTS检查当时服务器空闲和可用空间，这是运行时检查，不是另一轮模型实验。预计两组总计2.5–3.5小时，真实更新后的稳定性和最终AP需由该对照检验，不能保证提高AP。

## 首次失败与修正

首次验收在比较骨干LR时失败：脚本将计算得到的1e-5×0.1与字面量1e-6作集合精确比较。前者在Python中是1.0000000000000002e-6，实际配置正确。修改为按冻结配置的LR和倍率计算期望值；没有改学习率、配置哈希或放宽模型数值验收阈值。

失败报告、日志及OOM注入记录保留为attempt1-*；原诊断工作目录保留。修正后使用独立retry1诊断目录完成两组验收。第一轮失败不曾创建实际训练工作目录，完整验收后也确认两组正式工作目录均不存在。

## 归档和同步

运行源码：scripts/ap_screen_runtime.py、scripts/run-valid-region-ap-screen.py；验收脚本：scripts/verify-ap-screen-entry.py。它们及冻结两组配置已同步服务器；acceptance.json保存执行所需源码/配置哈希。

reports/valid-region-ap-screen-entry-20261031/包含acceptance.json、control/candidate-restore.json、子进程日志、停止/异常负对照记录、baseline-evaluation-replay/、baseline-raw-ap.json、失败记录、remote-hashes.txt、local-check.json。关键报告下载哈希、本地执行源码哈希和69100条预测再次核验通过。

诊断fixture仅保留在服务器work_dirs/valid_region_entry_probe_retry1/，不与真实试验目录混用。模型和优化器真实训练更新0，结束GPU占用0 MiB，未关机。README/实验计划与本报告本地提交，Git推送由用户负责。
