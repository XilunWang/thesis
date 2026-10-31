# 有效区域匹配开关接入验收协议

独立ValidRegionMask2FormerHead，match_valid_region默认False。关闭直接调用父类，开启只改变_get_targets_single匹配采样坐标；无padding和空GT走父类路径。推理、网络参数、mask loss采样和损失权重不变，原E0/E1/E1-P源码配置不改。空GT没有匹配正样本需求，因此保持父类空GT处理，而不是额外改变它的点范围。

固定E0 epoch36，原增强seed20261012+image_id，batch2。真实混合有效尺寸[345,620]、无padding[8,12]、[345,620空GT]及双空GT控制。空GT只在诊断内存副本清空标签，不写标注。每批一次原前向，同一输出/RNG比较父类target与关闭开关的全部30项loss和RNG精确一致；无padding开启也必须精确一致。

观察每批全部10阶段×2图片的实际匹配输入，独立构造当前图片有效坐标并验证采样的GT/pred tensor；验证分配与target标签掩码一致，原mask loss_by_feat/loss单层函数继承不变。开启执行一次完整反传，梯度/loss有限、参数状态不变，不创建优化器或执行step。空GT mask loss应为零、分类损失有限。检查关闭/开启推理输出一致、权重状态严格加载，不加入新参数。错误尺寸需显式失败。

这不是训练效果验收、GPU反传逐位等价或正式保存恢复验收。无训练、无长任务，检查失败保留日志并如实报告。
