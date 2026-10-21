# P3首次接口核对：WaterMask → Mask2Former

日期：2026-10-21。完成的是官方源码静态核对与最小候选设计，不是模块实现、前后向验收或E1正式定案。E0仍为36轮/AP32.6，没有训练新模型。

## 来源和核对边界

- [WaterMask官方仓库](https://github.com/LiamLian0727/WaterMask/tree/f1acaafd4a2f3bbdbec20726f475f429d6bb95c5)：固定mmdet3提交`f1acaafd4a2f3bbdbec20726f475f429d6bb95c5`，读取WaterMaskHead、WaterRoIHead、LaplacianCrossEntropyLoss及UIIS配置。它是后续MMDet3迁移版本，不能直接当作2023论文原始实现。
- mmdet2分支固定引用查得`07042d254ffbdbf584ee0a5725e3c5e5e57e27c5`；本轮尚未读取和逐项比较该分支实现。
- [论文公开摘要](https://openaccess.thecvf.com/content/ICCV2023/html/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.html)确认DSGAT、MFRM及边界学习的总体角色。全文获取返回403，本轮没有完成公式/图示核对，不根据摘要补写公式。下一步仍须核对全文和原始分支。
- [MMDetection v3.3.0](https://github.com/open-mmlab/mmdetection/tree/v3.3.0)：标签解析为`44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`，读取Mask2FormerHead、继承的MaskFormerHead和MSDeformAttnPixelDecoder。结合本地保存的evaluation-config.py及prepare-uiis-baseline.py核对实际配置；未连接服务器，未验证已安装文件与上游源码逐字节一致。
- [来源哈希清单](p3-source-manifest.json)记录8个读取文件的SHA256和来源；源码位于被Git忽略的work/p3-reference，未导入执行，未并入训练代码。GitHub API目录查询限流后通过公开固定提交归档取得源码。归档及展开副本精简命令被执行策略拒绝，当前仍保留；不将精简记为完成。

## 三个候选为什么不能直接搬

以下WaterMask路径均相对于`project/WaterMask/`；函数位置按下载源码行号。

| 候选 | 官方代码实际接口 | Mask2Former的差异 | 适配判断 |
| --- | --- | --- | --- |
| DSGAT | `watermask/water_mask_head.py:19,74,301`：GAT加入gff分支；WaterRoIHead把FPN的x[0]送给它。输入[B,C,H,W]，token卷积后展平节点，构建图，再反卷积/插值、残差输出同尺寸特征 | 当前像素解码器输出mask_features和3层memory，没有独立gff分支 | 可以研究放在mask_features之后，但语义、开销和收益均需验证，不能仅因尺寸一致认定正确 |
| MFRM相关多阶段细化 | `water_mask_head.py:146,318,345`：Stage组合RoI局部特征、从全局特征RoIAlign得到的特征、上一阶段掩码概率；输出14→28→56的逐实例RoI掩码 | Mask2Former输出全图query掩码，无RoI局部特征、rois、roi_labels和同构的逐级RoI缩放流程 | 直接迁移会引入新的实例裁剪/细化路径，改动面最大，不作为第一项最小验证 |
| BLL对应实现 | `watermask/cross_entropy_loss.py:198,216`：LaplacianCrossEntropyLoss接收多阶段预测/GT列表；前两阶段普通BCE，后阶段只在前阶段预测和GT边界并集内算BCE | 现有loss_mask接收点采样logits/targets，原函数需要二维邻域和多阶段信息 | 不能把配置loss_mask.type直接换为该类；需在匹配之后、点采样之前设计独立分支 |

重要细节：

1. DSGAT代码`torch.topk(dist,k,dim=2)`未设置largest=False，实际选的是较大距离节点，不能把它写成“最近邻图”。固定配置top_k=11；节点不足11时需明确行为。全局距离/注意力仍是N×N，top-k不意味着代码已用稀疏存储。以1024输入、stride4、token3无padding为尺寸示例，节点85×85=7225，单个FP32方阵约199 MiB；这不是实测峰值，前后向还会有其他矩阵和梯度。
2. BLL代码用宽度3的7×7拉普拉斯核及0.15×核面积阈值，得到背景/边界/前景标签。它不是本项目评价所用“图像对角线2%内边界带”；训练监督和评价尺子必须分别定义。
3. 官方UIIS配置阶段权重为[0.25,0.65,1]、start_stage=2；后阶段边界区域来自前阶段预测与GT，并经插值。WaterRoIHead推理还有边界区域融合策略。单加一个辅助损失不等于完整BLL/BMS复现。
4. 官方当前配置是ResNet101、Mask R-CNN基础配置且有save_best行为。不能连同这些训练/选模设置移入E0；本项目仍R50、原预算、固定最后权重和官方原始划分。

## Mask2Former可用接口

以本地保存配置为准：R50输出通道[256,512,1024,2048]、stride[4,8,16,32]；像素解码器统一256通道，输出stride4 mask_features及从低分辨率到高分辨率的3层memory；100个query、7类+背景、9层Transformer decoder。训练1024×1024裁剪时，以下是源码/配置推导尺寸，尚未做运行时hook验证。

| 位置 | 张量/作用 |
| --- | --- |
| MSDeformAttnPixelDecoder.forward | mask_features为[B,256,256,256]；memory预计[B,256,32,32]、[B,256,64,64]、[B,256,128,128] |
| Mask2FormerHead._forward_head | query[B,100,256]，经MLP得到mask embedding，与mask_features做einsum，产生[B,100,256,256]掩码logits；类别[B,100,8] |
| Mask2FormerHead._get_targets_single | 随机采样12544点用于匹配，返回pos_inds及按匹配顺序排列的GT mask；保留原Hungarian分配 |
| Mask2FormerHead._loss_by_feat_single（249行起） | 先提取正匹配预测[Npos,h,w]和GT，随后不确定点采样，计算原mask BCE与Dice；二维辅助损失应在点采样前取数据 |
| MaskFormerHead.loss_by_feat（324行起） | 对初始预测和各decoder层调用单层损失，再聚合最终与辅助loss。若只监督最终层，应显式控制这一层的附加分支，而不是无意给全部10组输出加权 |

注册自定义head应位于项目源码，通过custom_imports配置加载；不改服务器site-packages，也不修改E0入口的默认行为。不能把“重新调用get_targets得到另一组随机匹配”当作复用原匹配：应共用同次assignment结果，或显式记录方案差异。

## 当前最小候选：先做边界监督可行性检查

**建议优先验证损失分支的接口，暂不加DSGAT/MFRM。** 理由是能保持推理结构和参数量，单独检查边界监督是否可接入；这是工程改动最小的候选，不是已有证据证明它最能提高AP。已有小目标漏检也不能指望仅靠边界监督解决。

候选H1（待原文核对后登记）：最终decoder层的原匹配正实例，在GT边界附近增加辅助BCE，原分类/BCE/Dice、匹配器、训练预算、数据和推理均保持。GT-only版本不使用上一阶段预测边界，因此只能称“WaterMask启发的边界区域辅助监督”，不能写成原版BLL或新创新点。若要忠实迁移原BLL，还需定义跨层同一query/GT对应关系及前阶段边界传播，单独评估其必要性。

最小验证只需合成张量，不做训练：

1. 先与原始分支和论文核对边界目标实现。固定生成目标的网格分辨率、padding排除、边界宽度、插值及归一化方法；不把RoI上的宽度3直接解释成全图上的相同尺度。
2. 做普通/空GT、全前景、细线、图像边缘和变尺寸输入案例；确保形状正确、空集合返回连接计算图的零、无NaN。
3. 检查边界内梯度存在、区域外该辅助项梯度为零，原损失分支仍正常；lambda=0时输出、原损失与梯度对齐原版，并控制相同随机状态。
4. 在实际训练环境做一批前后向、预训练权重加载及保存恢复检查；记录额外耗时/显存。此时仍不启动长训练。
5. 通过后再登记E1：固定lambda、带宽、分辨率、层范围、损失归一化及验收指标。当前没有选定这些超参数，不能直接进入正式训练。

现存阈值报告说明小目标匹配率54.9%、中大目标72.1%/73.1%；这些是问题定位而非因果解释。README第21节边界汇总仅作已有历史背景，本轮未重新验证逐实例边界结果，不据其单独定案BLL。最终效果必须由独立配置的实际对照实验回答。

## 本步验收与下一入口

本步交付：固定来源与哈希、三模块接口差异、最小候选与验收条件。源码静态核对不等于模型运行成功。

P3进行中：下一小步先补齐论文方法节及mmdet2原始实现对照，再固定H1的边界目标定义，写独立合成张量检查。P4尚未开始；没有连接服务器、启动训练或改写实验结果。服务器重新连接后须先按既定流程检查并备份E0最终权重。
