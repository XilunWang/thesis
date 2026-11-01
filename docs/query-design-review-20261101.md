# 两候选源码与设计审查：细节反馈还是实例引导

日期2026-11-01。接续research-decision-20261101.md。本次只核对本地冻结配置、既有源码记录及上游原始代码/论文，无服务器、新模型前向、模块实现或训练。网络main链接仅代表访问时内容，未固定提交SHA；在任何复现前须固定版本。

## 结论先行

优先保留B“实例引导的query训练”作为研究参照，A“局部高分辨率反馈”暂缓。这里的优先级是接口、问题覆盖和推理成本的判断，不是UIIS上验证有效。当前还没有足以立项训练的原创方案；直接照搬MP-Former只能算已有方法复现/对照，不能把改名或移植到UIIS当新贡献。

不能因为BLL/采样失败就宣布Mask2Former不适合改进，也不能把已有多尺度融合、self-attention重新加一遍就称解决缺口。

## 当前原版已经有什么

本地reports/uiis-baseline-36e-evaluation/evaluation-config.py：R50 strides=[4,8,16,32]，pixel decoder num_outs=3，query decoder 9层、100query、256维、8头。保存的reports/baseline-decoder-audit-20261031/head-source.txt确认原forward实际调用链。

对照[MMDetection v3.3.0 head](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/mask2former_head.py)和[pixel decoder](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/layers/msdeformattn_pixel_decoder.py)：

- pixel decoder已有多尺度deformable attention及FPN式融合。该配置给query循环使用stride32/16/8 memory，stride4用于mask_features。stride4未直接作为第四层cross-attention memory，不等于query完全没有细节信息。
- _forward_head从同一个query状态分别算cls_embed与mask_embed，再与mask_features点积形成mask。
- 前层mask插值到下一memory尺寸，sigmoid<0.5生成detach硬注意力门；全禁止时回退为全开放。不是所有小目标一旦门内为空就永远无法看到图像。
- 每层已有cross-attention和query self-attention。因此“加注意力使query交互”并非现有结构缺少的能力。

具体缺口只能是如何选择/利用信息或如何训练这些能力，不能依据名称断言没有融合、没有交互。

## 候选A：局部细节反馈到query和mask

接入位置：pixel decoder输出之后、某次decoder更新内部。候选需让query以当前预测定义的区域读取stride4或新增更细特征，更新后的query再共同用于类别及mask预测。与先前MFRM末端适配不同：仅在最终mask logits后做残差修边，不能直接改善分类query。

依据：既有尺度对照显示部分目标有尺度敏感性。反证/限制：放大后45个几何失败中18有合格形状、仅7实际检出；32个全层失败中9有形状、仅2检出。分数、类别及重复匹配仍影响输出，不能用这些数字证明缺细特征。

成本与覆盖：全图stride4 memory相对stride8有4倍token，仅是该尺度交叉注意力项的比例，不是全模型显存/耗时4倍；局部采样可降低预算但会依赖初始区域，空/巨大/错位mask可能选不到目标。全图回退、局部坐标和新增可学习自由度都未确定。不能用GT框作为推理输入，也不能声称插值本身创造细节。

决策：暂缓实现。原模型已有融合，尚未定位有效细节入口/范围策略；目前欠缺的不是一个可直接添加的MFRM接口。现有原MFRM审查见[mfrm-feasibility-20261031.md](mfrm-feasibility-20261031.md)，不恢复该候选的历史实施待办。

## 候选B：实例引导的query训练

最相关原始参考：[MP-Former论文](https://arxiv.org/abs/2303.07336)。它针对Mask2Former层间mask不一致，以GT掩码引导的训练分支帮助重建目标，原论文不增加推理计算。论文其他数据集成绩不作为UIIS收益预测。

官方运行入口[run_50ep_no_noise_all_ly.sh](https://github.com/IDEA-Research/MP-Former/blob/main/run_50ep_no_noise_all_ly.sh)指定MultiScaleMaskedTransformerDecoderMaskDN、points模式、all-layer；该示例mask noise为0，但label noise为0.2，不能笼统称所有噪声都关闭。[配置](https://github.com/IDEA-Research/MP-Former/blob/main/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep_DN_query.yaml)的默认值会被入口覆盖，不能只读yaml认定执行参数。

[decoder源码](https://github.com/IDEA-Research/MP-Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py)：准备训练引导query/attention，拼接到正常query前；self-attention屏蔽普通query读取引导query；拆分两支输出。[criterion](https://github.com/IDEA-Research/MP-Former/blob/main/mask2former/modeling/criterion.py)：普通支仍Hungarian匹配，引导支依据已知GT索引计算分类及mask/Dice，包括辅助层；不是把普通重复query全部变正样本。具体实现包含多个变体，复现不能混用不同prepare函数。

映射到当前MMDetection接口，需要独立head在forward训练路径准备额外query及两个attention mask，分类/mask预测后拆开；loss路径给引导支明确配对，原100query普通匹配保留；推理只走原100query且不得读取GT。关键张量为query[B,Q+P,256]、self mask[Q+P,Q+P]及逐尺度cross mask[B*heads,Q+P,Hl*Wl]。这只是接口设计，未验证运行或冻结P/噪声/损失权重。

训练可能增加显存与计算；原论文不增加推理开销不等于当前移植已验证零开销。额外监督仍可能与普通支冲突。小目标引导区域下采样后是否可用、邻接实例是否混合，也尚无本项目验证。不能宣称它比BLL一定有效。

为什么比A优先：它在query形成/更新阶段同时影响类别和掩码，有具体原始实现可核对，且不必先依赖高分预测裁出小目标区域。只说明研究假设更完整，不证明“低分就是错误注意力”或32例全层失败都可解决。

## 防止再次走到训练才发现没有创新

本次没有找到已成立的原创改动。下一份设计必须以原MP-Former为对照对象，提出一个精确的剩余缺口，而不是再说“加实例监督”。可讨论的假设是相邻实例干扰下的实例范围与类别确认，但目前只是问题表述，尚未证明原方法遗漏它，也未查明文献是否已有对应方法。

设计准入要求：写清普通query/引导query各自读什么、训练和推理是否一致、相对参考方法哪一个机制不同、为什么现有负结果不直接否定它、哪种观测会推翻假设。只改名称、噪声常数、损失权重或增加层数，不足以凭此宣称论文创新。若无法提出可区分差异，就停止在设计阶段，不安排移植+几天训练。

MP-Former可作为后续引用或强对照，不自动复现；MaskDINO虽属于相关统一检测分割框架，本次仅核对论文入口、没有完成其源码适配审查，不能推荐直接整体替换baseline。为避免无边界扩大，本次不新增第三个工程候选。

## 本步产出及状态

完成原版能力核对、A/B接入位置及风险对照、原始方法来源与创新边界。优先B设计研究，A暂缓；未选定可训练的新模块。README/计划更新，本地Git提交，用户负责远程推送。官方3937/691及重复图不变，E0/E1/E1-P/A/B既有结果不改，不连接服务器、不训练。
