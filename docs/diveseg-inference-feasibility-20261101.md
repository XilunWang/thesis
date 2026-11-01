# DiveSeg只推理复测可行性审查

日期2026-11-01。结论：有条件可行，尚未达到全量推理准入。没有连接服务器、安装依赖、下载完整权重、运行模型或训练。

## 固定证据

作者仓库https://github.com/ettof/Diveseg，固定提交e96d41e884397a0a107c8d3a8e743afc26b8b874。只读副本work/diveseg-reference-audit/source，下载探测记录同目录。GitHub API遭遇匿名频率限制，改为正常浅克隆成功，未使用凭据。目录受.gitignore排除，不提交第三方源码。

UIIS发布权重：https://drive.google.com/file/d/1hrho8Jg6SNypazx_8xg486rRfyUOaCtK/view 。正常公开下载确认后Range请求返回206、bytes 0-1023/2246056303、UIIS.pth、ZIP文件头。总大小2246056303字节（2.246GB/2.092GiB），仅读取1024字节。证明公开文件可获取，不证明权重内容、训练轮数、键完整性或35.6已复现；没有加载pickle。初次请求仅返回下载确认HTML，不能将该HTML误认为权重。

## 逐项源码检查

| 项目 | 已查事实 | 后续要求 |
|---|---|---|
| UIIS入口 | eval.sh指向dinov2_vit_large_bs.yaml，eval-only，MODEL.WEIGHTS=out_put/UIIS.pth | 禁止使用默认USIS10K train.sh；resume必须False |
| 继承链 | dinov2配置→maskformer2_R50_bs16_50ep.yaml→Base-COCO-InstanceSegmentation.yaml | 固定三文件，不能被R50文件名误导，实际骨干DiveSeg_Backbone/vitl |
| 类别 | register_UIIS.py中ID1–7转0–6，顺序与本地val.json七类完全一致 | 使用本地同一份691图标注，记录哈希；作者原标注哈希未公布，尚未确认逐文件相同 |
| 测试变换 | build_test_loader使用自定义mapper；is_train=False时几何变换为空，原图RGB，构造傅里叶风格图image_a；模型归一化并补齐32倍数 | 不是本项目800/1333测试缩放；仅改MIN_SIZE_TEST不会生效 |
| 输出 | 100 query，类别概率选topk，硬mask logits>0，类别分数乘mask内部平均sigmoid；无score0.5预筛 | 检查展开配置topk100，并保存全部输出 |
| 评价 | UIIS分支调用Detectron2 COCOEvaluator，未指定maxDets，未使用仓库另一个InstanceSegEvaluator | 作者未固定Detectron2提交；当前官方默认maxDets=[1,10,100]、topk100，运行前须固定版本再验 |
| 初始权重 | 构建骨干先加载checkpoints/dinov2_vitl14_pretrain.pth，再由Checkpointer载入UIIS整模型；14核插值至16核 | 当前入口需要两份权重，不能只下UIIS.pth直接运行；不能未经完整键检查就跳过骨干加载 |
| 重复建模 | train_net.py main开头build_model用于参数统计，eval_only分支又build_model | 第二次构建期间旧模型仍存活，可能增加峰值；复测入口应只构建一次并登记这一运行修订 |
| 依赖 | Detectron2、timm和自编译MSDeformAttn；xFormers缺失有普通attention回退 | 独立环境；不更改现有MMDetection环境。requirements和Detectron2未锁版本，运行兼容性未验证 |

源码永久入口：[配置](https://github.com/ettof/Diveseg/blob/e96d41e884397a0a107c8d3a8e743afc26b8b874/configs/UIIS/instance-segmentation/dinov2/dinov2_vit_large_bs.yaml)、[测试mapper](https://github.com/ettof/Diveseg/blob/e96d41e884397a0a107c8d3a8e743afc26b8b874/mask2former/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py)、[运行入口](https://github.com/ettof/Diveseg/blob/e96d41e884397a0a107c8d3a8e743afc26b8b874/train_net.py)、[骨干加载](https://github.com/ettof/Diveseg/blob/e96d41e884397a0a107c8d3a8e743afc26b8b874/mask2former/modeling/backbone/backbones.py)。Detectron2默认值参考当前main的config/defaults.py和evaluation/coco_evaluation.py，仅作依赖参考，不能当作者原环境证据。

## 公平比较的边界

作者配置训练batch10、衰减20000/26000，与论文batch8、23000/27000不同。因此当前代码加公开权重只能叫“发布版本复测”，不能未经日志证明就叫严格论文复现。此差异不授权重训。

E0计分maxDets=[100,300,1000]，该入口默认[1,10,100]。如果确实每图总共只输出100个，两者最大截断通常不会改变AP；仍需用同一份预测实际双算确认。AR子项不能直接混比。图像输入尺度不同即使计分一致也不是同条件消融。

复测拟先保留作者原图输入、原归一化和后处理，所有预测还原原图后用本地同一份GT计算原默认和项目协议指标。不为追35.6改尺度、挑checkpoint或调阈值。不能把它与E0的差归因于单模块，也不能自动把先验分支移植到R50。

## V100资源判断

本地691图尺寸已清点，最大图ID651为2048×1305（另有1800×1350）。原图补齐后最大图为2048×1312。ViT-L有16头、16像素patch，约10497 tokens（含CLS）。普通FP32 attention单个16×N×N矩阵约6.57GiB，softmax和其他中间量还会增加峰值；风格分支亦有1024通道高分辨率特征。此为源码推算，不是实测显存，不能承诺V100 32GiB全量可运行，也不能只测一张小图就通过。

当前无证据要求A100专属算子；没有证明V100编译、数值或完整前向可用。作者训练AMP=True不代表eval自动AMP，后续不能悄悄改精度。服务器当前磁盘/GPU状态本轮未检查，历史空闲不能代替实际检查。

## 下一项及停止条件

下一项是发布权重与隔离推理环境准备：服务器只读检查磁盘/GPU后，核验两份权重来源/大小/哈希，固定依赖；准备仅构建一次且无优化器的评估入口。在用户继续授权前本轮不安装或下载大权重。

环境就绪后先固定一张普通图及最大图ID651做无梯度前向，核对权重缺失/多余键、7类输出、原图恢复、峰值显存和耗时。通过才制定/执行691图全量只推理；若加载不完整或最大图OOM，停止并报告，不自动缩图、改精度、改网络或转训练。

本轮产出是可执行条件及具体缺口，没有新的模型成绩。BLL、简单有效区域采样和单格邻居候选继续暂停/结束，保留全部历史证据。
