# WaterMask论文Mask2Former训练协议核查

核查日期：2026-10-29。仅检查公开原文、代码、作者公开压缩包及本地E0配置，不训练、不连接服务器。

## 结论

论文表3的Mask2Former是ResNet-101，Mask AP/AP50/AP75为25.7/38.0/27.7，标注为3×、多尺度训练。本地E0为ResNet-50，36轮，32.6222/47.6373/35.2550。AP差约6.9点，但尚不是同协议复现差值。

我们确定加载了完整COCO实例分割预训练模型。作者公开Mask2Former模板默认只初始化ImageNet骨干，但模板仍是COCO数据、80类和50e训练预算，不能证明论文UIIS对照实际沿用了该初始化。公开权重配置包也没有Mask2Former实验，因此目前只能将预训练作为优先验证假设，不能认定6.9点全部由此造成。

## 证据来源与范围

- [论文原文](https://openaccess.thecvf.com/content/ICCV2023/papers/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.pdf)：表3和5.1节。论文说明3937/691划分；不能把WaterMask自身的SGD描述直接套到Mask2Former。
- [原始mmdet2分支](https://github.com/LiamLian0727/WaterMask/tree/07042d254ffbdbf584ee0a5725e3c5e5e57e27c5)：锁定提交07042d254ffbdbf584ee0a5725e3c5e5e57e27c5；不使用后续mmdet3升级作为2023实验依据。
- [R101配置](https://github.com/LiamLian0727/WaterMask/blob/07042d254ffbdbf584ee0a5725e3c5e5e57e27c5/configs/mask2former/mask2former_r101_lsj_8x2_50e_coco.py)：继承R50实例配置，再继承R50 panoptic基础配置和default_runtime。
- [作者公开权重与配置包](https://drive.google.com/file/d/1SlgkcOzYjKku15ldezNRe4O4OsqdprIB/view)：WaterMask all pth and config.zip，2152994971字节。HTTP Range读取完整ZIP中央目录，并提取9个配置/日志文件，逐文件校验解压大小和ZIP CRC；没有下载大权重。完整目录及文本SHA256记录在[证据索引](watermask-protocol-evidence-20261029.json)。
- 本地实际展开配置：checkpoints/e0-backup/config-first.py及config-resume.py，而非仅查看尚未运行的配置模板。

## 逐项比较

| 项目 | 作者公开Mask2Former模板（不等于论文最终配置） | 本地E0实际配置 |
|---|---|---|
| 骨干 | ResNet-101 | ResNet-50 |
| 骨干初始化 | torchvision://resnet101 | 完整COCO权重覆盖兼容参数 |
| 整模型加载 | 默认load_from=None、resume_from=None | mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth |
| 数据/类别 | data/coco，80类 | UIIS官方划分，7类 |
| 优化器 | AdamW，lr=1e-4，weight_decay=0.05 | 相同 |
| 骨干学习率倍数/梯度裁剪 | 0.1 / max_norm=0.01 | 相同 |
| 训练增强 | LSJ 0.1–2.0，1024方形裁剪，随机翻转 | 核心策略相同 |
| 测试尺度 | 1333×800，保持比例，无翻转 | 相同 |
| 标准化 | ImageNet mean/std | 相同 |
| 训练预算 | 368750迭代，里程碑327778/355092 | 36轮70884步，第27/33轮衰减 |
| 批量 | 每GPU 2，文件名8x2 | 单GPU 2，自动LR缩放关闭 |

模板50e与论文表3的3×不一致，说明最终UIIS配置必然有改动；无法从模板确认论文实际迭代数、GPU数量、全局batch、命令行覆盖或初始化。文件名8x2也不能证明论文实际用了8张卡。旧版图像填充与新版预处理有细节区别，但未验证其AP影响。

仓库_base_/datasets/coco_instance.py确实包含UDW路径和专用均值方差，但Mask2Former继承链用_delete_=True重新定义COCO数据，不能把该文件的水下标准化直接当成Mask2Former配置。

## 公开压缩包能确认什么

目录仅有WaterMask/1x/r101、WaterMask/3x/r101、WaterMask/3x/r50、WaterMask/3x/cr101四组，无Mask2Former文件。R50首段日志20230118_104531.log记录torchvision://resnet50、load_from=None、resume_from=None，AdamW并未用于该WaterMask实验，而是SGD lr0.0025、36轮、27/33衰减。续训日志20230118_153505.log第36轮指标为0.264/0.436/0.288，与论文26.4/43.6/28.8一致。

R101和Cascade R101包内日志第36轮分别为27.2和27.1 AP，与论文一致；其日志是恢复训练片段，不能将resume_from误认为额外外部预训练。以上证明WaterMask本身的公开记录与论文对得上，不能据此外推缺失的Mask2Former对照日志。

Mask2Former目录的公开路径历史只查到2023-07-13的init提交；检查点目录与.mim副本没有提供UIIS最终版本。已查公开issues及comments，未发现解释25.7或Mask2Former实际预训练的有效记录。Google预览接口请求失败后改用正常公开下载的Range请求成功；没有绕过认证。原work/p3-reference/watermask-mmdet2.zip无法解压，本轮证据来自重新读取的固定提交单文件及公开权重包，不使用损坏缓存作证。

## 如何解释结果、下一步怎样验证

完整COCO预训练意味着分割相关模块已学习过实例分割，再适应UIIS；只初始化ImageNet骨干时，分割模块需要从UIIS开始学习。该区别有可能显著影响小数据集表现，但本轮未测因果效应。优化器、主要增强等在公开模板中与我们相近，不应笼统声称两边所有设置都不同。

既有AP审计已验证已保存预测的COCOeval计算；唯一已知完全重复图排除后AP仅改变+0.0016点。因此计分错误和这一重复图不能解释6.9点差异，但未重新推理，也未排除近重复或所有潜在数据重叠。

最有价值的受控实验是固定R50、现有UIIS划分、36轮、增强、优化器和种子，仅把整模型COCO初始化替换成ImageNet骨干初始化，并核对训练前实际加载参数。这只能衡量我们协议内预训练的影响，不能命名为WaterMask论文Mask2Former复现。要严格复现25.7还需作者实际R101配置、加载权重与运行日志。训练尚未授权或启动；不改变现有E0/E1和官方划分。

对外只能写“论文公开结果参考对比，预训练及实现协议未完全对齐”。不能将32.6减25.7或减26.4当成自研模块收益；模块收益应与同协议E0比较。

本地原始材料：work/watermask-protocol-audit（Git忽略）。新报告/证据索引/README本地提交，未同步服务器，Git远程由用户负责。
