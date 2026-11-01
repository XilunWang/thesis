# 小目标文献与源码核对：把独立补漏改成有位置依据的查询

检索日期2026-11-01。范围为与当前R50/Mask2Former问题相关的代表性论文和官方实现，不是工业部署普查或穷尽SOTA榜单。未连接服务器、未改模型、未训练。优先采用论文原文/CVF及作者代码；检测框AP不等于实例分割Mask AP，航空数据上的收益不外推UIIS。

## 代表路线与本项目取舍

| 路线 | 原始来源及主要机制 | 对本项目的判断 |
|---|---|---|
| 放大/切片 | [SAHI，ICIP2022](https://arxiv.org/abs/2202.06934)：切片推理及切片微调，提高小目标在输入中的有效尺度；[官方实现](https://github.com/Small-Object-Detection/SAHI) | 可作为工程参照，非首选结构改动。我们全图尺度试验已有小目标增益及FP/成本代价；它不等于切片试验，不能说已否定SAHI。切片还需掩码拼接、跨片去重、公平总输出及算力预算 |
| 从图像特征选择初始查询 | [FastInst，CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/He_FastInst_A_Simple_Query-Based_Model_for_Real-Time_Instance_Segmentation_CVPR_2023_paper.html)：实例激活引导查询；[Mask DINO，CVPR2023](https://arxiv.org/abs/2206.02777)：编码器输出参与查询选择和定位初始化 | 最高相关。当前缺的是自动发现/读出实例的机制，不只是把特征交给原query。借鉴初始化机制，不整体换模型，不把相关论文当小目标专用算法 |
| 密度与查询预算/空间分配 | [DQ-DETR，ECCV2024](https://arxiv.org/abs/2404.03507)：计数、密度线索与动态query；[Dome-DETR，ACM MM2025](https://arxiv.org/abs/2505.05741)：密度引导特征/注意力与渐进查询初始化 | 支持研究“查询放在哪里”，不支持立刻增加query数。当前固定案例没有证明100查询总容量不足，先固定新增预算，隔离数量与位置的影响 |
| 训练中引导查询关注目标 | [MP-Former，CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_MP-Former_Mask-Piloted_Transformer_for_Image_Segmentation_CVPR_2023_paper.html)：带噪GT mask引导训练中的masked attention，推理无额外分支 | 与既有早期形状丢失证据相关，但只有部分案例属于此类；保留为机制参照，不自动恢复之前邻居辅助query路线，不把直接迁移称原创 |
| 更高分辨率、抑制干扰 | [DM-EFS，ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Sharma_DM-EFS_Dynamically_Multiplexed_Expanded_Features_Set_Form_for_Robust_and_ICCV_2025_paper.html)：浅层高分辨率特征及动态复用；[SET，CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html)：背景频率平滑与训练期显著性增强 | 细节和背景干扰都值得考虑，但我们的R50并非没有多尺度，直连也没普遍奏效。未验证水下频率噪声因果，不能仅凭任务名称添加频域/超分模块 |

另检索到2026-10-24新预印本[Dynamic Query as Budgeting](https://arxiv.org/abs/2610.33430)，讨论查询数量预算与查询选择的混杂；本次只核对摘要，未独立复现，不将其当成成熟共识或模型选择依据。它提醒要做同query数量对照，不能由“新增query有效”推断位置选择机制有效。

## 已实际核对的官方代码

- [FastInst QueryProposal](https://raw.githubusercontent.com/junjiehe96/FastInst/main/fastinst/modeling/transformer_decoder/utils.py)：空间类别预测、局部极值及top-k选择；[decoder](https://raw.githubusercontent.com/junjiehe96/FastInst/main/fastinst/modeling/transformer_decoder/fastinst_decoder.py)使用query_proposal产生的内容、位置和查询位置。不是只定义一组与当前图片无关的额外向量。
- [MaskDINO decoder](https://raw.githubusercontent.com/IDEA-Research/MaskDINO/main/maskdino/modeling/transformer_decoder/maskdino_decoder.py)：two_stage分支从编码器类别输出选top-k，并gather内容和位置；包含不同的定位、去噪及解码设计，不能把其中几行作为Mask2Former可直接加载的模块。
- 已查看[MP-Former官方仓库](https://github.com/IDEA-Research/MP-Former)、[DQ-DETR官方仓库](https://github.com/hoiliu-0801/DQ-DETR)的说明；未做这两个仓库的全源码审计。网页main分支读取仅代表本次查看版本，未下载或封存外部代码，接入前需固定版本及许可证。

以上机制已有公开工作。任何迁移成功首先是复现/适配证据，不自动构成论文创新，也没有来源承诺保护UIIS原结果。

## 修正我们上一版建议

上一版“冻结原版＋独立补漏查询”能隔离原模型参数，却没有充分回答两个问题：新query如何找到漏检位置，以及冻结的mask_features能否表达缺失实例。不能因为隔离了参数，就认为解决了小目标。

保留冻结原版的实验隔离，明确推荐候选为：**图像特征引导的独立小目标查询，先验证查询位置和内容初始化是否带来有效补漏。**

```text
图像 → 原R50 → 原像素解码器 → 原查询解码器 → 原始预测（冻结保留）
          │          │
          C3/C4      原mask_features
          │          │
          ├→ 小目标位置/目标性预测 → 少量空间分散的候选位置
          │                             │
          └────────取对应特征＋位置编码 → 独立查询解码 → 新掩码及类别
                                                  │
                           与原始预测按冻结协议合并 → 最终100输出
```

1. **先给query位置依据。** 小头从C3/C4输出小实例候选位置，选少量分散位置，以位置处特征初始化内容、坐标初始化位置编码。不能继续仅用随机/固定新增query；也不按原模型低分筛出位置，否则真正没画出的目标仍无入口。局部峰值和空间分散只是候选规则，可能漏掉相邻极小实例，不声称解决拥挤问题。
2. **原模型保持不动，新分支独立。** 原参数/缓冲/原100查询输出保持，新增支路不反写共享memory。冻结是用于隔离副作用的实验选择，不是文献已证明的最佳训练方式；必要共同适配也可能被限制。
3. **把“找位置”与“补漏”监督分开。** 候选位置头由训练集全部保留下来的小GT提供正例，不能只用每批变化的漏检集合形成不稳定种子标签；新实例分支再按冻结原版的覆盖情况承担残余目标。无残余图片及背景/重复候选需要监督。训练GT可生成位置监督，推理时只能用预测位置；不允许用val GT挑query。
4. **承认mask表达的独立限制。** 历史GT辅助探针显示少数目标C3/C4可读而原mask_features未读出，但探针有失败校准，不能证明mask_features必然缺信息。因此先测“有引导query、原mask_features”；只有该因素得到明确证据，才讨论新支路自己的mask特征投影/局部读出，原支路仍不动。不把C3/C4注入、密度头、新loss、局部高分辨率全部一起上。

这是一项待验证的组合设计，不是已实现的模块或必然提升方案。也不是“从最后一层修一下边缘”：入口在新增query形成之前；原计算分支保持原样。

## 与已有证据如何对应

- 原查询排查中45个固定漏检无合格同类几何：需要改善候选形成，而非只提高分数；但该固定集不代表全量比例。
- C3/C4少数实例有GT辅助可读线索：支持尝试从特征产生查询，不证明无GT时能够定位。
- 直连新增24又丢24：隔离原支路可以排除原参数/原输出直接被改坏，但不能保证新增FP或top100挤出不降AP。
- APs收益集中水草且新增集中单图：验收必须看小鱼、小珊瑚、类别分布及稳定性，不能只报APs总数。
- 未证明100query容量不足：暂不引入动态计数/大幅增加查询，也不以提高最终输出上限换取表面召回。

## 建议的最小验证与停止条件

当前只做设计。下一步先核对官方QueryProposal与本地特征接口，并在训练标注上核对小目标代表位置、stride8网格碰撞和候选预算。该检查不证明学到的候选召回，不把GT oracle结果当AP，也不重跑已完成的全部基线诊断。

若以后明确授权学习验证，在同一冻结E0上比较固定额外query预算的两组：A为同样分支/容量但无图像位置引导的查询，B为图像特征引导查询。目标性头、损失和参数预算要控制并如实列出不可完全对齐项；另有E0原输出作为保护参照。不是立即跑两个完整模型，也不是把旧全模型续训A当该机制的唯一公平对照。

先检查候选覆盖残余小GT、非目标位置比例和近邻重复；成功后才评价mask AP、原正确结果被挤出、各类新增/丢失以及显存/延迟。最终100输出、原3937/691划分、重复图及评价方式不改。不得在691图反复选阈值后再称独立泛化，位置阈值/预算用训练数据或训练内留出确定并冻结。

不能保证不降AP；若候选定位本身不比等预算对照有用，停止，不做长训。若只改善水草少数图、总AP未达事先门槛或原正确结果明显被挤出，也不追加36轮。当前未冻结新训练预算或批准实现，无训练启动。

## 当前决策

优先借鉴FastInst/MaskDINO的图像引导query形成；密度分配仅作约束参考，MP训练引导为后备，SAHI为可选工程参照。暂停上一版未指定查询来源的直接接入，不堆叠所有模块。论文贡献尚未确定，先验证本项目假设及是否超出已有方法。README和实验计划更新，本地提交，Git推送由用户负责。
