# E1-P正式入口验收与首轮授权

2026-10-31。用户授权“入口检查通过后只跑第一轮，跑完停止”，不授权后35轮或测试集评价。当前新增独立入口，不改旧E0/E1及冻结PixelBLL公式。

## 已通过的入口检查

独立prepare-uiis-pixel-bll.py/train-uiis-pixel-bll.py沿用原Runner及TrainingGuard，正式模式仅允许stage=first，显式拒绝resume；保留36轮配置和27/33轮调度，但第1轮checkpoint保存后停止。锁、禁止覆盖和模式门槛保留，新增更新前loss/梯度有限性守卫。

verify-pixel-bll-training-entry.py在4张真实图、workers2、5个迷你epoch共10步中完成：

- 与已保存E0实际配置逐项一致（仅head/额外loss设置、导入及目录白名单不同）；与E1实际配置除head外一致。
- COCO加载608个兼容张量值精确一致；仅cls_embed.weight/bias由81行变8行，保留原初始化规则。不是从E0/E1最终权重起跑。
- 第一个迷你epoch2步后退出，新进程恢复模型610张量、优化器/调度器、epoch/iter一致，再到5个迷你epoch10步。workers2与原采样配置运行成功；不承诺跨进程worker RNG或中途采样轨迹逐位等价。
- 两次学习率衰减、最近3份checkpoint及1/3/5里程碑保留、锁拒绝/未指定模式/禁止覆盖/满预算拒绝通过。正式resume拒绝也通过。

原始结果reports/pixel-bll-training-entry-20261031/validation.json，passed=true。该10步属于独立入口验收，不计入正式1969步。

## 评价代码冻结

compare-pixel-bll-offline.py继承既有配对/COCO/边界计算，增加E1-P输入、两两及三组共同GT统计和固定案例GT ID。没有读取新模型预测或评价首轮。

旧E0/E1全691图回归的COCO、阈值计数、边界汇总及逐实例记录与原报告完全一致。用既有E1预测临时充当第三组的测试中，E1与该第三组AP/边界完全相同，新增漏检/检出/改善/退化均0；三组共同集合复现原E0/E1统计。该测试不是E1-P成绩。

产物reports/pixel-comparison-regression-20261031与reports/pixel-comparison-three-arm-fixture-20261031，后者fixture-check.json明确标记。数值口径与案例选择在见到新结果前冻结；图片排版可后做，不改选择规则。首次生成脚本遇到GBK解码及字符串定位错误，均在评价执行前失败，随后修正为UTF-8/明确函数边界；未改变评价数学口径，相关日志保留。

## 正式首轮执行边界

启动包装run-e1p-first-epoch.py先核对来源哈希、完整官方数据manifest/重复图、预训练、既有E0、空闲GPU及磁盘，再仅运行stage=first。最多3小时安全超时，不自动重启。结束写process-result.json并调用只读inspect-e1p-first-epoch.py检查1轮1969步、345组AdamW计数、610模型张量、保存文件、有限数值、日志和源码/数据未改。

正式目录work_dirs/uiis_e1p_pixel_bll_r50_36e，首轮报告reports/e1p-first-epoch。尚未完成的实际首轮结果以该目录inspection.json及后续报告为准，不把本入口验收通过当首轮已完成。

最终预检已通过：4631个manifest条目核对，官方3937/691及重复图保留，预训练与E0哈希一致，V100空闲、可用磁盘约27.3 GiB。2026-10-31 01:03:54预检完成，随后首轮启动；源码基于b273611。当前仅首轮运行，完成后另记实际结果。
