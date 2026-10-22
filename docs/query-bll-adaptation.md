# BLL到Mask2Former：跨层对应与独立辅助损失

当前目标生成已于2026-10-22修订，见 [小目标保留修订](bll-target-preservation.md)：块内前景占有、ceil有效半格，替代下文历史nearest/floor。下文记录2026-10-21原型及当时结果，不作为当前空间规则。

2026-10-21。本步固定一个可测试的适配候选并完成独立CPU测试。尚未注册自定义head、未运行真实Hungarian匹配或完整Mask2Former，不能称为框架接入成功。

## 对应规则

以最终decoder层**原损失已使用的同一次匹配**为准。对每张图，取sampling_result.pos_inds和pos_assigned_gt_inds成对使用。例如最终层query3匹配GT1，就用倒数第二层query3、最终层query3、GT1这一组三者生成边界辅助项。不能另做一次Hungarian，也不能拿倒数第二层独立匹配出的GT拼接。

同一query编号来自同一decoder迭代路径，但其上一层语义可能尚未正确；该方案只是用最终层匹配作为监督锚点，不声称跨层天然具有正确实例身份。错误的上一层轮廓也会进入边界并集，这是候选规则的实际含义。

`scripts/query_bll.py`接收已有索引，不含匹配器或随机采样，逐图明确收集而不依赖GT排序。检查同图query和GT索引一一对应，重复/越界会拒绝。原分类、原mask BCE/Dice及其他decoder层的原有监督均应保留。

## 当前候选空间和损失规则

| 项目 | 本步独立原型的选择 | 与原WaterMask的区别/限制 |
| --- | --- | --- |
| 输出 | 倒数第二层与最后一层[B,Q,h,w]，必须同尺寸 | 不再是14→28→56的RoI细化链 |
| GT | 使用已补齐至同一输入画布的GT，用nearest缩到原生mask logits网格 | 整数stride、二值GT；细小或细线物体可能在缩小中消失，后续需评估，不能以合成通过掩盖该限制 |
| 边界来源 | 前层同query的sigmoid≥0.5与对应GT，各自提取边界后取并集 | 仍保留预测/GT并集，不是先前GT-only简化；同网格不需要原来的2倍区域插值 |
| 带宽 | 当前探针使用width=3、阈值0.15 | 这是mask网格单位，不等于RoI上相同实际尺度，更不是原图对角线2%评价带；不是E1已选超参数 |
| padding | 假设图像在补齐画布左上角；只保留完全落在有效图像内的网格单元，边缘不完整单元也排除 | 在提边界前把有效区域外预测/GT置零，最终选区再裁去padding；有效图像外缘仍可能形成边界 |
| loss | 最终层选区内BCE求和/全批选中像素数，分母至少1 | 额外加入原损失，不替换原BCE/Dice；仅单进程CPU FP32，目前不承诺DDP/AMP等价 |
| 权重 | 调用时必须显式传loss_weight；合成测试用1，关闭测试用0 | 尚未确定正式训练权重；0时跳过对应/边界计算，不消费随机数 |
| 空情况 | 无匹配、无边界或无完整有效网格时返回连接最终logits计算图的0 | 明确健壮性处理，未修改原版参考实现 |

辅助损失只向最终logits反传。前层预测仅用于硬边界选区，选区detach；前层继续从原有辅助损失获得梯度。这个独立函数没有模型参数，不改变推理；不等于已经实现WaterMask的BMS推理融合。

## 实际测试

入口：`scripts/validate-query-bll.py`，环境沿用`.local/bll-libs`、bundled Python3.12、torch2.6.0+cpu；无新增安装。运行命令：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.local/bll-libs'
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B scripts/validate-query-bll.py
```

首次运行退出码0，19项检查通过，报告在`reports/query-bll-synthetic/validation.json`（Git忽略、拒绝覆盖）。包括：

- 2张图、3个最终匹配实例，索引故意乱序；266个选中像素；独立逐像素邻域计数、GT整数切片及NumPy BCE/解析梯度对照通过。
- query和GT重新编号后，只要匹配索引相应更新，损失不变；故意使用错误关联会产生可检测的不同损失。
- 最终层未匹配query、选区外及padding梯度为0；前层不接收硬选区梯度；输入未被修改。
- 任意改变padding预测值不改变loss，包括最后不完整有效网格的处理。
- loss_weight=0时，合成基准损失的值、梯度及随机状态一致。**这不是完整Mask2Former原BCE/Dice的lambda=0验收。**
- 无匹配、空选区、图像小于一个网格单元及非法索引/形状/权重/非零GT padding处理通过。

没有读取UIIS进行选择或试参。前一轮24组原版三阶段测试和本轮19项跨层规则测试是不同验收层级，不混写成完整模型测试。

## 下一步：真正接入head前的门槛

1. 新建独立注册的head及候选配置，E0入口默认配置保持不变。
2. 在最终层原有损失调用中获得同次sampling_result；如需重构单层损失返回值，应保持原计算顺序和随机数消费，不为BLL重复调用get_targets。上游已读取的get_targets可选返回分支存在tuple/list拼接风险，接入时不能未经运行验证直接依赖该分支；可在明确的单层损失实现中显式返回原采样结果。
3. 从实际模型取得两层logits、padded GT、img_shape并核验形状；本步人为提供的matches不能代替此项。
4. 固定同种子、输入、权重，先检验lambda=0时完整原head输出、所有损失和梯度一致；再启用辅助项检查有限梯度、额外显存/耗时及保存恢复。
5. 实际服务器使用Python3.10/torch2.1.2+cu121和mmdet3.3.0，须在该环境再验。服务器连接前按既定流程只读检查并备份E0；不因小测试通过直接启动正式训练。

E1、正式带宽与权重、是否最终保留该适配均未定案。本轮未连接服务器、未训练、未改数据或E0成绩。
