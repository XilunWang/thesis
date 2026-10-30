# PixelBLL同尺度接入验收结果

2026-10-31。方案已冻结在[bll-pixel-protocol-20261031.md](bll-pixel-protocol-20261031.md)。本轮仅实现独立候选与无更新验收，没有训练。

## 改了什么

旧BLL将GT缩到网格后，把有任意前景的格子整格标1；新候选保留增强后原GT，最终mask logits双线性插值到GT尺寸，在有效选区内逐像素算BCE。网络输出仍是原stride4，没有新增细化模块、参数或独立预测头。

旧占有GT仍用于确定检查区域，但不再作为BCE真值。前层/GT边界并集、最终原匹配、width3、weight1、全批选中像素平均均固定。不引入前景/背景平衡、错误加倍或只选错误。每个匹配实例必须有被选中前景，否则报错；这是保护性检查，不保证训练后召回。

独立文件：scripts/pixel_bll.py、scripts/pixel_bll_head.py，注册PixelBLLMask2FormerHead；configs/bll-pixel-head-probe.py仅head overlay，不是正式训练Runner。原scripts/query_bll.py、bll_core.py、bll_mask2former_head.py与E1记录SHA256一致，旧训练配置/入口、权重及预测未改。

## 本地验收

- 7份保存真实输出（32匹配实例）的新helper loss及最终logit梯度，与之前独立原GT像素候选公式一致。
- 122个单像素位置，含4×4相位、半格、小于stride图像，全部有正监督，损失与梯度有限；前层硬选区无直接梯度。
- 两张有效大小不同的合成输入按总选中像素归一，而非每图均值再平均，验证通过。
- 空匹配为图连通零；weight0旁路；负权重、非有限权重、非零padding、重复匹配四项负对照均拒绝。
- 验收脚本scripts/verify-pixel-bll.py；reports/bll-pixel-helper-20261031/result.json，passed=true。

## 真实框架验收

现有服务器MMDet环境，真实训练图ID22、seed20261043，原训练数据处理，加载E1 epoch36严格检查全部状态键。使用同一个完整模型前向输出，恢复相同CPU/CUDA RNG分别调用父类、关闭候选、开启候选的loss流程，以确保比较使用同一匹配/采样随机序列。

| 项目 | 实际结果 |
|---|---|
| weight0原loss键和值 | 与父类精确一致，没有新loss键 |
| weight1原loss键和值 | 与父类精确一致，额外增加loss_bll_pixel |
| 新loss | 0.9297927022，与独立公式一致 |
| 新loss对最终logits梯度 | 与独立公式在预设FP32误差范围内一致 |
| 当前匹配 | 3实例，面积136742/617/326 |
| 选中原前景像素 | 12856/617/326；两个小目标完整保留，大实例只检查边缘部分属预期 |
| 选中像素总量 | 37838，排除有效图像外padding |
| 总loss反传 | 345组参数梯度有限 |
| 模型状态与权重文件 | 全部状态张量不变，checkpoint SHA256不变 |
| 优化器 / 更新次数 | 未创建 / 0 |
| 本次峰值已分配显存 | 5002563072字节，仅此验证图与流程，不是训练上限 |

没有比较多次完整GPU反传的逐位等价，也没有声称完整模型新旧总梯度相同（新增损失本来就应改变总梯度）。关闭等价检查针对同前向/RNG下的loss；原网络forward未重写，结构/状态键相同，未进行新AP推理。

脚本scripts/verify-pixel-bll-framework.py；reports/bll-pixel-framework-20261031/result.json，passed=true。数据与权重核对通过，服务器结束GPU无计算进程。

## 当前能说什么、不能说什么

**可以说：独立候选公式固定、真实框架接入与本轮前后向验收通过。**

不能说边界AP已改善、小目标绝不会漏检、全部随机增强已审计或已可直接启动36轮。原局部梯度冲突并未因工程接入而消失。尚未为PixelBLL建立正式训练Runner/保护hook与保存恢复验收；旧train-uiis-bll.py仍属于E1旧算法，不能拿它直接训练新候选。

下一步如继续，是候选全训练数据处理/正监督保留审计及独立Runner生命周期验收，仍可先不训练；后续短试跑/正式训练需另行授权，先冻结对照协议。当前停止在验收完成状态。

## 同步与记录

候选helper/head、overlay和框架验收脚本已同步服务器，文件及下载结果SHA256逐项核对。协议、结果说明、本地helper验收脚本与README尚未同步服务器。所有代码/文档本地提交，远程Git由用户负责；没有安装新依赖或修改site-packages。
