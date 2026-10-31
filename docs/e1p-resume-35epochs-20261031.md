# E1-P从第一轮续训至36轮并统一评价

2026-10-31，用户明确授权：续训前检查通过后，从首轮继续剩余35轮，保持参数不变，完成后统一评价并比较E0/E1。此授权取代此前“后35轮暂停”；不授权额外预算、调参或失败后自动重训。

## 不变的部分

恢复源是E1-P epoch_1.pth，SHA256 `ada5705de4ad9a7a57cb35e4a1862e81f7cf84fc013830a05d52dcbcdb159353`，epoch1/iter1969，610个模型状态张量、345组AdamW状态。完整恢复模型、优化器、学习率调度及计数；不从COCO重新起跑、不重复第一轮。

保持原GT像素监督公式、weight1、width3、全部原损失、单卡FP32、batch2、workers2、seed20261012、官方3937/691及重复图、训练增强、优化器、27/33轮衰减和总预算36轮70884步。首轮源码与run-protocol不改写。

## 新增执行程序

- check-e1p-resume.py：第一轮文件/模型/优化器/调度、冻结源文件及完整数据manifest、预训练、既有对照输入、资源检查；禁止重启已有续训尝试。
- resume-uiis-pixel-bll.py：复用首轮原BLLGuard，新增独立续训入口，原首轮正式resume拒绝保留。check-only在before_train核对加载状态后主动结束，0次更新；execute-remaining严格从原首轮检查点继续到36轮。
- evaluate-e1p-final.py：只在36轮70884步完整完成后评价epoch36全部691图，核对模型/优化器、同原版推理配置、原始预测/权重哈希。
- run-e1p-resume-and-evaluate.py：独立锁及一次性launch标记，依次训练→评价→compare-pixel-bll-offline.py；任何阶段失败立即记录并停止，不重训或自动换参数。训练在服务器独立进程运行，不依赖本地SSH保持连接，不自动关机。

输出分别为reports/e1p-resume-35epochs、reports/uiis-e1p-pixel-bll-36e-evaluation、reports/e1p-final-comparison。比较复用原E0/E1最终预测，核对它们未改；保存未四舍五入的三组COCO AP表、共同GT边界、多涂/漏涂、分尺寸匹配/漏检、覆盖率和固定案例GT ID。条件Boundary IoU不是Boundary AP。评价程序依赖已在服务器导入检查通过，比较数学口径沿用首轮前已通过的两组回归及三组重复预测测试。

## 状态与解释边界

续训前检查现已通过：完整数据/源文件/配置/首轮checkpoint核验，以及Runner实际恢复610个模型张量、非空优化器/调度器和epoch1/iter1969准确，0次参数更新。当前状态以reports/e1p-resume-35epochs/preflight.json、restore-check.json、training-process.json及process-result.json为准。只有process-result的completed=true且训练、评价、比较均成功，才能称整个流程完成。

首轮耗时25分55秒，因此剩余35轮粗估约15小时，另加评价/诊断时间；实际取决于批次与服务器速度。保存/恢复验证不承诺多worker随机轨迹与不中断训练逐位一致，保持与原E1同类epoch边界恢复方式。已有测试集诊断影响设计选择的探索性限制继续披露。

后续读取结果时先查错误和完成标记；评价或对比失败只修相应阶段，不能重跑训练。完成后需取回最终权重、报告、预测及哈希，更新README并本地提交；Git push由用户负责。

## 已启动状态

2026-10-31 01:37:33（Asia/Shanghai）独立后台流程启动，源码提交7fdff6c；编排PID7051、训练PID7053。实际训练日志确认RESUME_STATE_VERIFIED并进入Epoch 2，前50步正常，监督vanished=0。训练、评价及对比尚未完成，不预填结果。无训练参数或原首轮来源文件变化。预检/恢复检查及启动记录已取回本地，原始运行日志留在服务器；Git push由用户负责。

## 2026-10-31 09:20巡检

已完成19轮，第20轮100/1969步；编排7051和训练7053仍运行。最新日志时间09:19:45，约0.7747秒/步、ETA 7:12:56，预计训练约16:33结束，随后评价/比较另需时间。全续训日志异常标记及非有限loss/梯度记录均0，711条抽样监督记录vanished均0；这不是逐步效果评价。epoch17/18/19检查点存在，GPU使用21352/32768 MiB、磁盘空闲26 GiB。未重启、未改参数，尚无最终AP/边界结果。原始只读快照reports/e1p-resume-35epochs/status-20261031-0920.json。

## 2026-10-31 11:07巡检

已完成23轮，最新日志第24轮450/1969步（11:06:45），最新监督记录11:06:56、vanished=0。process-result仍为training/completed=false，未开始最终评价。日志ETA 5:26:17，预计训练约16:33完成，随后自动评价和比较。最新loss/grad_norm有限，GPU使用21352/32768 MiB，磁盘空闲26 GiB。本轮只读巡检，没有改参数或重启；快照status-20261031-1107.json。

## 2026-10-31 16:13巡检

已完成35轮，正在第36轮，最新训练日志100/1969步（16:12:31），ETA 24分20秒，预计约16:37训练完成，随后自动评价及三组比较，当前尚无最终结果。编排7051和训练7053仍运行，全续训日志异常标记0、非有限loss/梯度记录0，1341条抽样监督记录vanished均0；GPU使用21352/32768 MiB，磁盘空闲25 GiB。快照reports/e1p-resume-35epochs/status-20261031-1613.json。首次连接因网络权限无法解析域名，授权后SERVER_READY；一次只读日志汇总因远程相对Python路径错误失败，改为绝对路径后成功，不影响训练。本轮未修改参数或重启。
