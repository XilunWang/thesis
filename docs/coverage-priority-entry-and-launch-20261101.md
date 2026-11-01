# 覆盖优先300步：启动前检查与执行记录（2026-11-01）

北京时间21:39:45已启动服务器独立后台任务，worker PID2606。当前处于初始化/完整step0核验阶段，尚无训练完成或收益结论；最多300步，最终统一评价后停止。不重跑两组已有对照，不追加步数、不自动重启。

## 授权与冻结条件

用户在冻结计划后明确“开始”，本轮据此执行“准备检查→通过后启动”。计划JSON里的training_authorized=false和execution_entry_ready=false保留其制定时状态，实际后续授权与启动记录见`reports/coverage-priority-pilot-execution-20261101/launch.json`，不回改冻结计划。

计划SHA：7085c00bca2e97beceead21b2af7c8858a712d2c4d65f3fc0897fd5851210602。

## 已完成检查

- 本地旧代码/候选规则/已有权重和评价哈希核验；600次访问顺序按同算法和seed重算，摘要与冻结aaed2ed6…一致。
- 新评价程序复算上一版已有结果，完整重现其535覆盖、4706无GT等计数，并按新计划准确判定覆盖通过、no_touch和无GT失败。没有把旧失败改成成功。
- 服务器READY，V100初始空闲，磁盘余量约16GB。首次同步后的SSH启动命令超时；重查READY后重试成功，未发生训练重复启动。
- 独立入口默认仅元数据检查；`--preflight`禁止optimizer.step；`--execute`必须有匹配计划/源码的验收证明和明确授权启动记录。
- GPU检查从共同实际旧step0恢复权重、空Adam和RNG，没有使用上一轮诊断的已训练头。
- 固定两图2971/21263983、3151/20264163，仅反传不更新；原模型和位置头不变。
- 记录直接取自本次objective返回的实际labels/layout：GT ID、cell、该GT在格内的像素数、label SHA。由记录重建的标签与损失所用标签精确一致。
- 两图step0含随机对照的全部评价记录与既有step0对应记录精确一致；完整分数的top200与本次评价一致。
- 同进程保存恢复、原输出不变通过。下载的验收checkpoint、分数和源码SHA均通过。先前独立进程无更新恢复验收也已保留；本轮没有宣称已验证更新后动量恢复。

## 正式入口及记录

`run-coverage-priority-pilot.py`继承原Pilot的数据、模型、评价和顺序，覆盖标签计算与新增记录，不改旧文件。

每步在更新前写`training-labels.jsonl`（attempted_update字段，只有对应steps.jsonl完成记录存在才表示该次更新完成），每步更新后原子保存latest及步日志。记录不参与梯度，不改变学习率或优化器。

评价期间通过只读forward hook保存128个holdout视图的FP32完整logits和有效区；不会额外前向，不改变随机流。每份分数绑定图ID、seed、输入/GT/checkpoint哈希，并复核top200。原评价384视图及随机对照不变。

第一次更新前仍会完成**全部384视图step0评价**，逐条对照原记录；不一致立即停止，不允许带差异开始更新。该全量检查是后台任务的首阶段，此刻不能标成已通过。

新checkpoint kind=`coverage_priority_pilot`，绑定计划、全部原/新增源码及开关设置哈希。正式输出`work_dirs/position_coverage_priority_pilot`必须原先不存在，防重复启动。本入口没有自动续跑选项，中断后保留现场。

最终按冻结7项门槛评价，全部通过才可讨论后续。原E0冻结，仍为位置头300步实验，不做691图Mask AP评价。

## 后台执行

`launch-coverage-priority-once.py --launch-authorized`使用独立session启动worker，SSH断开不会主动结束该任务。worker只调用一次execute，记录退出码、开始/结束时间，不重启，也不关服务器。

启动记录：北京时间2026-11-01 21:39:45，worker2606。启动后确认worker仍存活、process-result尚无结束或退出字段；GPU已进入初始化。预计约30–45分钟属于按历史耗时的估计，实际以状态文件为准。

## 入口

- 冻结计划：`docs/coverage-priority-pilot-plan-20261101.md`及同名JSON。
- `scripts/run-coverage-priority-pilot.py`、`coverage_pilot_metrics.py`、`launch-coverage-priority-once.py`。
- `reports/coverage-priority-entry-20261101/result.json`与其label-records/full-scores-0：无更新启动前证明。
- `reports/coverage-priority-entry-local-20261101/metric-control.json`：已有对照复算。
- 服务器`reports/coverage-priority-pilot-execution-20261101/{launch.json,process-result.json,console.log}`：任务状态。
- 服务器`work_dirs/position_coverage_priority_pilot/`：实际检查点、训练/标签日志、三次评价及完整分数。

本地Git记录与服务器文件同步分别完成，不自动git push。下一次检查必须读取实际步数/退出状态，不能仅凭计划时间断言完成。
