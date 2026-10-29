# E0基线AP独立复核（2026-10-29）

## 结论与范围

直接读取已保存预测，绕过MMDetection Runner，使用pycocotools 2.0.11 COCOeval重算。未训练、未重新模型推理、未连接服务器，未改变数据、权重或既有结果。默认Python3.8无pycocotools；改用已有bundled Python与项目.local/analysis-libs，未安装依赖。

| 评估范围 | Mask AP | AP50 | AP75 |
|---|---:|---:|---:|
| 官方691图 | 32.622218 | 47.637317 | 35.254972 |
| 敏感性检查：排除完全重复的1张测试图（690图） | 32.623851 | 47.643172 | 35.253592 |

原报告六项掩码指标经三位小数舍入全部一致。标准useCats=1，IoU=0.50:0.05:0.95，maxDets=[1,10,100]。删除内存预测中的bbox字段后loadRes，按mask面积计算尺度指标；磁盘原预测未改。使用全部69100条预测，没有额外score0.5筛选。

## 数据核对

- train3937图24626标注，val691图3784标注；图像/标注ID唯一，引用合法，两组类别表一致（ID1–7）。预测覆盖全部691个图像ID，类别ID合法、分数有限且在0–1内，RLE尺寸与对应标注图像尺寸一致。
- 标注SHA256与已保存的作者数据源manifest（revision 44ca5db46599f14faf8e57643014664435823738）一致。本轮未重新下载远端标注，不声称重新验证最新远端版本。
- 对全部4628张本地图片逐文件SHA256，跨划分完全相同文件仅train/XL_266.jpg与val/XL_679.jpg。仅排除对应测试图的评估敏感性差值为+0.001633 AP点，不能解释6.9点文献差异。原划分仍保留；没有检查近重复/相邻帧，也不等于无泄漏证明。
- 结果JSON：reports/baseline-ap-audit-20261029/result.json；可重复脚本scripts/audit-baseline-ap.py。输出目录按项目规则不入Git。

## 训练协议与文献证据

本地checkpoints/e0-backup/config-first.py和run-protocol.json确认R50、7类别、整模型COCO预训练mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth、AdamW lr1e-4、batch2、36轮；1024随机尺度裁剪训练，1333×800等比测试。配置顶部残留的80类变量不是实际head配置，实际两个head均7类别。

[WaterMask原论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.pdf)表3报告Mask2Former R101为25.7/38.0/27.7；5.1给出3937/691划分和标准mask AP。其WaterMask使用SGD的说明不能自动推断为表3 Mask2Former训练配置。本文本不足以确定该对照是否使用整模型COCO权重，因此不得断言差异由预训练造成。

本次只确认32.6可从已有预测重现，以及一个完全重复图影响极小；没有重新验证权重生成预测的过程，也没有完成与论文对照的完整协议对齐。下一步优先取得作者Mask2Former对照具体配置/日志，核对预训练、数据版本、增强和训练预算；如需新推理/训练，另行准备执行，不自动重训。

## 当前进度纠正

本项目实际已完成E1 BLL36轮（AP33.0）及离线边界诊断；最近聊天中“尚未运行改进实验”的表述过时。以README46–49节为准，本次审计不恢复单图BLL暂停任务，也不更改研究路线。
