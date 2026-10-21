# BLL正式入口准备与Runner集成验收

2026-10-21。入口已准备，集成检查通过，正式36轮训练未启动。

## 入口

- scripts/prepare-uiis-bll.py：继承原E0配置，只改head类型、BLL权重1.0/width3、注册导入和输出目录。
- scripts/train-uiis-bll.py：独立Runner入口，复用原TrainingGuard首轮停止、检查点归档与优化器恢复核对；增加预训练/恢复张量核对、代码/配置/数据/预训练哈希保护、BLL目标消失和连续空选区停止检查。
- scripts/verify-bll-training-entry.py：仅运行4张训练图/5个迷你轮次（每轮2步，共10步），独立目录work_dirs/e1_entry_verification_coco，先1轮停止，再新进程恢复至第5轮。

入口必须二选一显式传--verification或--execute-formal。验证目录必须以e1_entry_verification_开头；正式目录只接受work_dirs/uiis_e1_bll_r50_36e。正式flag只是防误操作开关，不代表本轮获准或已经执行。原E0训练入口未修改。

验证的epochs=5、milestones=[2,4]只为在10步内检查调度和保留机制；正式配置仍是36轮、27/33轮衰减，batch2/workers2、原数据/增强/优化设置。验证不访问测试集评估。

## 实际通过项

1. 解析E1配置，除5类允许差异外与基线构建配置完全一致；同时对照服务器真实E0 config-first.py中的model、train_dataloader、train_cfg、optim_wrapper、param_scheduler、randomness、env_cfg、load_from，全部相同（此对照针对改动前基线配置）。没有把诊断cuDNN设置静默带入正式训练。
2. COCO预训练SHA256与固定记录相同。加载前后核对608个兼容状态张量的实际数值，逐位一致。仅panoptic_head.cls_embed.weight/bias从81输出变8输出（含背景），尺寸不兼容不加载，确认保留新初始化。日志中的这两条size mismatch为预期类别变更，其余缺失/多余/尺寸不符会直接拒绝。
3. 从COCO起点完成真实前后向。第1迷你轮完成2步并保存退出；新进程恢复的610个模型状态张量（含分类层）完全一致，原TrainingGuard同时逐项验证优化器参数组/动量等、调度器、epoch与iter。恢复后继续完成第5迷你轮，共10步；没有把第1轮重复训练。
4. 迷你测试学习率按2/4轮衰减，最大lr最终约1e-6；根目录仅保留epoch3/4/5，里程碑保存1/3/5。
5. 项目训练锁冲突、未明确选择模式、覆盖已有目录、预算已满后再次resume，均被实际执行的拒绝测试拦截。

报告reports/bll-training-entry/validation.json中passed=true、formal_training_started=false，全部日志、配置、协议和阶段结果已取回本地。验证checkpoint仍在服务器，不声称已本地备份。正式E1目录不存在，E0最终权重SHA256仍为53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f，结束显存0 MiB。

## 失败记录与边界

首次配置比较脚本对mmengine Config使用del触发AttributeError: __delitem__，发生在构建/训练模型之前。改用底层ConfigDict的pop后复测通过；原报告保留于reports/bll-training-entry-config-api-failure/validation.json，没有覆盖或隐瞒。

本次验证epoch边界的Runner恢复，workers保持2；没有验证任意步中断、多worker随机增强游标逐位恢复，也不声称中断/不中断真实GPU训练轨迹完全一致。Runner沿用E0恢复语义，不将此前直接wrapper的RNG验收扩大为本入口的全轨迹保证。

入口具备基础停止/日志保护，但仍需正式训练前的全训练集数据处理审计（尤其GT缩小后消失）、资源/环境/源文件最终核验。4张COCO起点检查和16张热启动短试跑都不能证明3937张所有随机增强无问题。本轮没有完成全量审计，也没有启动首个正式训练轮次。

下一步先对官方训练集做不更新权重的数据处理审计；通过并再次获准正式启动后，才使用该入口开始第1正式轮检查。不要重复E0或从验证checkpoint续入E1。
