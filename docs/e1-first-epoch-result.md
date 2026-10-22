# E1：BLL正式第1轮完成并停止

2026-10-22。用户授权最后检查通过后只运行第1轮、完成后停止检查。该轮属于正式36轮预算，不是技术短试跑；未继续第2轮，未运行其他模块实验。

## 启动依据

scripts/check-e1-first-epoch.py确认全部4631个官方文件哈希正确，3937/691划分及重复图片保留；COCO预训练、E0最终权重、训练源码及已通过的完整模型报告一致。除已约定的BLL/head注册/目录外，配置与实际E0关键字段一致。GPU空闲、磁盘剩余约35.5 GiB、训练锁空闲且E1目录不存在。preflight.json为passed=true。

从原COCO预训练初始化，608个兼容状态张量值核对通过；仅分类层weight/bias由81输出改8输出，保留新初始化，不是加载故障。不使用E0最终权重或技术试跑checkpoint初始化。配置仍为36轮、70884步、batch2、workers2、FP32、seed20261012，BLL weight1/width3和块内前景占有规则不变。

## 实际运行与停止

2026-10-22 17:05:18启动，17:30:01进程结束，包含启动和保存共1482.79秒（约24分43秒）。入口scripts/train-uiis-bll.py使用--stage first --execute-formal；后台监督进程记录退出状态，控制台输出落文件，避免SSH断开影响训练。FIRST_EPOCH_GATE在epoch_1.pth保存后停止，退出码0；result-first.json确认epoch=1、iter=1969、verification=false。没有自动resume，没有第2轮记录。

| 检查项 | 实测 |
| --- | --- |
| 正式完成预算 | 1/36轮，1969/70884步 |
| 总损失日志 | 首条44.2299，末条26.9223；范围23.7323–44.2299 |
| BLL损失日志 | 首条0.9119，末条0.7150；范围0.6823–0.9783 |
| 日志采样 | 39条loss记录（第50–1950步的窗口统计），40条BLL诊断 |
| BLL选区采样范围 | 1752–24367像素，采样标注消失数0 |
| 每步停止保护 | 未触发非有限损失/模型参数、GT消失、连续空选区保护；未见OOM |
| 结束空选区连续计数 | 0 |
| 日志最高memory | 14185 MiB；不是精确增量显存或效率对比 |
| 优化器学习率 | backbone组1e-5，其余相关组1e-4，尚未到27/33轮衰减 |

日志是窗口统计，不是每一步loss明细；第1969步完成由Runner结果和checkpoint/优化器计数交叉证明。训练loss下降不等于AP提升；没有在691张测试集上评估或选best。

## 停止后只读验收

scripts/inspect-e1-first-epoch.py运行通过，inspection.json passed=true、training_resumed=false：

- 唯一轮次检查点为epoch_1.pth，last_checkpoint指向它，无result-resume.json或epoch2文件。
- checkpoint的610个模型状态张量全部有限；345组AdamW状态全部有限，全部step=1969。优化器、调度器和BLL计数已保存，轮次/步数/学习率一致。
- 完整日志有首轮自动停止和BLL_RUNNER_STAGE_OK标记，无所列异常；原COCO加载核验通过。
- 核心源码、训练/测试标注、E0权重哈希未变。结束GPU显存0 MiB，无训练进程，训练锁已释放。

这次仅做checkpoint内容完整性检查，不启动新的Runner恢复或额外更新；此前epoch边界恢复已验收，不据此声称任意步中断或GPU轨迹逐位等价。

服务器目录work_dirs/uiis_e1_bll_r50_36e；epoch_1.pth大小529804636字节，SHA256为4a40dc4a90b1e8e3e90e1ab538c0c74ae3081e186630e27603a6d84887e6e356。报告、启动记录、控制台、协议、配置及Runner日志取回reports/e1-first-epoch；这些文件和权重被Git忽略，不能只靠Git保存。

## 结论与下一步

正式首轮完成并通过工程验收，可以作为后续同一次E1续训的起点。当前保持停止，不自动继续。下一步需要用户明确授权后，先核对现场资源及同一协议，再从epoch_1恢复到第2轮及后续约定预算；不重新初始化、不重复第一轮、不改BLL参数、不提前看测试AP。E2/MFRM、E3/DSGAT和组合均未开始。


首轮权重已备份至checkpoints/e1-backup/epoch_1.pth，大小与SHA256均与服务器验收报告一致，证明在reports/e1-first-epoch/local-backup.json。配置/协议/首轮结果一并备份；代码、文档和路线图状态已更新。
