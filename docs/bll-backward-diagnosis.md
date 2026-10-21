# 完整模型梯度超差：原版GPU反传重复性诊断

2026-10-21。结论：已实测确认原有CUDA grid_sample反传是非逐位重复性的一个来源。此前完整模型验收失败，不能直接解释成BLL接错；也尚不能断言所有误差只来自这一个算子。原passed=false报告保留，未修改容差。

## 排除输入和新增模块干扰

scripts/diagnose-bll-backward.py只构建原版E0，不导入BLL。使用先前0、1号训练图、相同增强种子和E0权重，cuDNN确定性开启、benchmark关闭、CUBLAS_WORKSPACE_CONFIG=:4096:8。构建一次计算图后，两次autograd.grad对同一损失、同一图反传；没有重新增强、随机采样、匹配或参数更新。因此观察到的变化不能归因于两次输入不同或BLL新增计算。

实际同图反传：最终mask logits梯度最大差5.82077e-11；四层backbone输出梯度最大差约7.08e-8至3.28e-7；backbone.conv1.weight梯度最大差4.29153e-6，其最大梯度6.54797、相对L2误差4.58918e-7。说明微小差异已在损失向mask logits反传处出现，并沿网络传播到参数。该次诊断量级不等于此前全部重新前后向的误差，不能混用两种实验。

## 算子证据

严格torch.use_deterministic_algorithms(True)首先拒绝cumsum_cuda_kernel；这是全流程中首先碰到的不支持严格确定性的操作，并不证明此前输出的差异来自cumsum。warn_only模式进一步明确报告grid_sampler_2d_backward_cuda没有确定性实现。

服务器已安装源码链：mmdet/models/dense_heads/mask2former_head.py第318行在原掩码损失中调用point_sample；mmcv/ops/point_sample.py第245行定义该包装，第270行调用F.grid_sample。因此它属于原Mask2Former损失路径，非BLL新增代码。

单算子隔离：使用真实最终mask logits的子集作为输入，固定合成坐标与上游梯度，对同一grid_sample图反传3次。CUDA梯度不完全一致，最大差2.38419e-7；CPU三次完全一致，最大差0。这直接确认grid_sample的CUDA反传是一个可复现的误差来源。没有替换服务器框架、生产算子或模型设计。

只设置cuDNN确定性无法覆盖所有PyTorch CUDA算子和MMCV自定义算子，因此此前开启该选项仍超差并不矛盾。尚未隔离量化可变形注意力等其他算子的贡献，不把该诊断写成全模型所有差异的唯一根因。

## 对当前验收的含义

先前门槛是每个梯度元素都满足abs差≤1e-6+1e-5×abs参考值。它在合成head上通过，但在真实图完整网络上连原版自身重复也未全部通过，故不能单凭该失败判断BLL改动错误。原数据/输出/损失/RNG一致及功能检查仍成立；“数值验收尚未通过”状态保留。

下一步应先制定适用于该GPU环境的验收方法：增加多次原版重复的误差分布、逐层/归一化误差，结合关闭BLL对照和可检测错误的负对照，预先固定判据再独立验证；需要严格参考时单独使用CPU或确定性诊断实现。不能不断调大阈值直到通过，也不应为测试静默改变正式模型的算子。确认标准前不开展训练。

报告reports/bll-backward-diagnosis/diagnosis.json已取回，completed=true仅表示诊断完成，不表示完整模型验收通过。脚本退出码0；检查结束显存0 MiB，未训练、未改权重、未新增依赖。
