# PixelBLL全训练集增强与保存恢复检查

日期：2026-10-31。授权范围为全3937图数据审计和保存恢复验证，无训练、无optimizer.step。冻结的原GT像素监督公式未改，旧E1核心、正式入口及权重未改。

## 数据检查方法

脚本：`scripts/audit-pixel-bll-data.py`。沿用原训练pipeline、batch2预处理，按image_id排序，每图固定seed=20261012+image_id，直接调用prepare_data，不以自动补取掩盖过滤。每图只做一次增强，不是枚举所有随机增强，也不是复现某一实际训练epoch的采样顺序。

对增强后原始二值GT检查标签、padding、空mask及原前景保留。仅用GT边缘部分选区，计算展开回原GT尺度后选中的原前景像素数；实际GT边缘与预测边缘并集只会增加选区，因此这是前景入选的保守下界，不需运行模型。每个非空实例必须至少有一个被选中的原前景像素；不要求目标内部所有像素进入边缘监督。

4×4块内的原前景数量用avg_pool×16精确计数，再乘GT选区，等价于展开选区后数原GT正像素。逐图图片哈希、增强后GT面积与历史同seed审计对照，检查数据规则一致；原始标注和E0权重末尾重验哈希。

全量结果通过，耗时644.91秒：

| 检查项 | 结果 |
|---|---:|
| 官方图片扫描/唯一ID | 3937/3937 |
| 进入监督检查的增强样本 | 3936 |
| 原pipeline过滤 | 1（ID976，XL_1142.jpg） |
| 增强后实例/保留正监督实例 | 22033/22033 |
| 增强后面积小于1024像素的实例 | 6391，均保留正监督 |
| 最小GT面积/最少入选前景 | 1像素/1像素 |
| 新监督前景完全丢失/处理错误/空实例mask | 0/0/0 |
| 无实例的合法增强样本 | 14 |
| 有padding/有效边缘半格的图片 | 2542/2197 |

ID976在相同seed的原pipeline中返回None，与历史检查一致；训练loader会按原规则补取，本次直接记录，不替换、不删除数据。14个合法无实例样本与“非空实例变空”不同，空匹配loss已在上一轮helper验收。历史nearest/floor造成的183个消失实例，本次均有正监督。这里的6391按增强后mask面积划分，仅用于数据诊断，不是官方AP_small分组或召回率。

本地另用16个合成mask（含有效半格）核对块计数与直接展开计数完全一致。取回原始结果后独立核对3937唯一ID、22033组GT/入选前景数量、所有0<入选前景≤GT面积、原数据配对一致及源码哈希；原始报告5个文件服务器/本地SHA256一致。标注与E0权重未变，旧E1由保存恢复检查确认未变。

## 保存恢复结果：通过

脚本：`scripts/verify-pixel-runner-checkpoint.py`。在隔离目录调用MMEngine原生Runner.save_checkpoint/resume，再启动新Python进程恢复。使用已有E1第36轮非空AdamW状态作为序列化样本；这不代表批准从E1权重开始新候选训练。

- 模型全部state_dict、345组优化器状态、学习率调度器、epoch=36、iter=70884，保存前后及新进程恢复后一致。
- Python、NumPy、Torch、CUDA随机状态通过显式rng-sidecar.pth保存恢复，指纹一致；不能说原生checkpoint自动保存了这些随机状态。
- 恢复后对真实ID22、seed20261043执行前向及反传，loss_bll_pixel=0.9297927021980286，参数梯度均有限；反传前后模型、优化器、调度器及步数不变。
- optimizer更新次数为0，没有调用train()。旧E1 checkpoint SHA256仍为`6d89f04575ec2a28b441c3f681e38627cf2273638de3ee6394bedb03279e8176`。
- 隔离checkpoint SHA256：`857300aa7cfbb2595d17638b0157ac5da447603e447e6a9b9bdfc5061370beaf`。仅保留服务器隔离目录，不作为正式候选权重。

首次尝试在模型创建前因内存Config未执行custom_imports，报PixelBLLMask2FormerHead未注册。检查脚本增加显式import后，在新目录重试通过。失败日志与目录保留，没有覆盖原产物。该修正只影响检查入口，不改损失公式。

本次验证的是状态保存/恢复和恢复后可计算；没有验证optimizer更新后的连续训练等价、数据worker游标恢复、多卡/AMP或训练资源峰值。不能据此承诺中断续训逐位一致。

## 证据与后续

- 数据原始结果：`reports/uiis-pixel-bll-data-20261031/`。
- 保存恢复结果：`reports/bll-pixel-runner-checkpoint-verified-20261031/result.json`及child.json。
- 失败日志：`reports/bll-pixel-runner-checkpoint-20261031-console.log`；成功日志为同名前缀加`-verified`。
- 隔离状态文件：服务器`work_dirs/pixel_bll_save_restore_verified/`；失败目录`work_dirs/pixel_bll_save_restore_check/`。

以上都是工程检查，不是AP或边界改善证据，也不排除原有局部前景/背景梯度冲突。下一步如要验证实际更新稳定性，需要另行授权固定少量图片的短试跑，并提前固定更新次数、保存恢复对照与停止条件；本轮不启动。

两份新检查脚本已同步服务器并核对哈希；报告原始产物已取回本地。文档仅本地更新，Git推送由用户负责。结束时GPU无计算进程。
