# P2 边界评价协议 v1

登记日期：2026-10-21。状态：协议、合成/六图验证和官方实现运行对照均已通过（README第20节）；不是已完成的全量边界评价。研究问题是已检出实例的轮廓质量，不预设边界是主要误差或 BLL 有效。

## 来源与版本

- [Boundary IoU，CVPR 2021，arXiv v1](https://arxiv.org/html/2103.16562v1)，第5、6节：分别考察掩码内部边界带的交并比，并用 min(Mask IoU, Boundary IoU) 构成实例评价的匹配质量。小物体可能整体落入边界带，因此两种 IoU 可以相同。
- 官方实现固定到 [37d25586a677b043ed585f10e5c42d4e80176ea9](https://github.com/bowenc0221/boundary-iou-api/commit/37d25586a677b043ed585f10e5c42d4e80176ea9)。本轮浏览核对 master 的 [boundary_utils.py](https://github.com/bowenc0221/boundary-iou-api/blob/37d25586a677b043ed585f10e5c42d4e80176ea9/boundary_iou/utils/boundary_utils.py)、[cocoeval.py](https://github.com/bowenc0221/boundary-iou-api/blob/37d25586a677b043ed585f10e5c42d4e80176ea9/boundary_iou/coco_instance_api/cocoeval.py) 及提交历史；后续已下载该固定提交原始函数并完成OpenCV运行对照，详见文末入口及README第20节。

## 固定输入和范围

- 原始 UIIS 划分、重复图片和所有标注不变。输入 `data/UIIS/annotations/val.json` 与 `reports/uiis-baseline-36e-evaluation/predictions.segm.json`，哈希必须与已有 diagnostics.json 一致。
- 本轮只选标注列表前六张：L_1000、L_1001、L_1025、L_1029、L_1032、L_1046；沿用既有案例，不根据结果另挑图片。不启动训练、推理或全量分析。
- 按原始图像分辨率解码二值掩码，不缩放、不裁剪。使用 pycocotools 的 polygon/RLE 解码，不以自写解码器替换。维度不符、非二值数据或异常输入应失败，不能静默修复。

## 边界度量

设原图高宽为 H、W，固定 r=0.02，d=max(1, round(r×sqrt(H²+W²)))，采用 Python round。原掩码减去以3×3全一核腐蚀d次后的掩码，得到内部边界带；图像外为背景，因此触及画面边缘的物体也有边界。Boundary IoU 为两个边界带的交集/并集。另保存 Mask IoU 和两者最小值。

2% 是参考设置，不是 UIIS 标注一致性已经验证的最优容差。本轮不依据结果挑选带宽。局部实现使用 NumPy 方形窗口计数计算腐蚀，需分别通过手算/独立逐像素判定，以及后续官方 OpenCV 实现的逐位比较。

一空一非空的 IoU 为0；双空返回 null，并从条件均值中排除、保留记录；不将双空计为完美实例。原标注空掩码视为异常。小物体/细线腐蚀后消失时边界带可等于整个掩码，这是定义的结果。

## 固定配对与解释范围

- 沿用 `analyze-uiis-errors.py`：score≥0.5，置信度降序、同类 Mask IoU≥0.5、一对一最高 IoU 配对；分数相同保留输入顺序，IoU并列沿用标注顺序。不根据 Boundary IoU 重新挑选配对。
- 普通 GT 计入匹配/漏检。crowd 不计漏检，未匹配预测与同类 crowd 的原掩码 IoA≥0.5 时忽略。匹配数、漏检、预测数及 crowd 忽略数须逐图吻合既有诊断。
- 保存 GT ID、预测排序、类别、面积、置信度和三种 IoU；汇总为匹配实例等权平均，分母明确列出，不按像素面积加权。无匹配组输出 null。
- 必须同时列出未匹配普通 GT、未匹配且未忽略的预测数量。条件均值不能覆盖漏检/误报，不能称为 Boundary AP、整体分割准确率或 AP 损失归因。六图结果不能外推691张，也不能证明某模块有效。
- 后续全量条件统计按类别及 GT 面积分组，沿用现有诊断 small<32²、medium<96²、其余large 的互斥分组，明确其边界值处理与 COCO AP 分组口径可能不同。

## 正式 Boundary AP 的后续门槛（本轮不实现）

如后续计算 Boundary AP，应使用固定版本官方评价器，普通 GT 的匹配质量取两种 IoU 的最小值；crowd 沿用原掩码 IoA。保持官方类别、置信度排序、maxDets、面积组、IoU阈值及忽略规则，不先用0.5分数筛掉预测。先在同一预测上重算 Mask AP 对齐32.6，再算 Boundary AP。不得将两者差值解释成可被某模块恢复的因果损失。

## 验收与执行

1. 合成案例：相同、平移、不相交、一空/双空、全图前景、细线、非法输入；检查对称性、手算面积和独立慢速窗口判定。
2. 官方 OpenCV 腐蚀逐位比较通过后，才认为边界提取运行一致性充分验证；2026-10-21已通过，详见reference-comparison.json。
3. 六图通过官方解码、既有计数对齐及 Mask IoU 对齐，并检查逐实例输出；之后才能进入全量离线分析。

从项目根目录运行（Python 环境含 numpy、Pillow、pycocotools）：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.local/analysis-libs'
python scripts/validate-uiis-boundary.py
```

默认结果目录 `reports/uiis-boundary-pilot/validation.json`，已存在则拒绝覆盖；如需重验，通过 `--output` 显式指定新的独立目录。JSON记录源码哈希、基础提交、运行环境、输入哈希（六图成功时）和失败信息。依赖不可读退出码2，不能当作整体验收通过。结果目录被Git忽略，需独立保管；协议和脚本随本地提交保留。

## 官方实现运行对照入口

脚本 `scripts/verify-boundary-reference.py` 直接加载固定提交的官方 `boundary_utils.py`，先校验SHA256为 `a495cc15fdbded38681edef30b29c62014e81e864f0e68ca6062ce42db51fd9c`。参考源码放在 `work/boundary-reference/boundary_utils.py`（本地临时材料，不纳入Git）；下载地址固定为：
https://raw.githubusercontent.com/bowenc0221/boundary-iou-api/37d25586a677b043ed585f10e5c42d4e80176ea9/boundary_iou/utils/boundary_utils.py

依赖：原numpy/Pillow/pycocotools，加本地 `opencv-python-headless==4.13.0.92`（Python≥3.9要求numpy≥2，与本地2.3.5匹配）。不修改服务器训练环境。通过下列命令运行，默认输出同一成功目录中的 `reference-comparison.json`，已有报告则拒绝覆盖；原 `validation.json` 不改写。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.local/analysis-libs'
python scripts/verify-boundary-reference.py
```

检查包括6种画布尺寸、6种掩码结构、3种带宽的108次逐像素对照，以及固定六图中全部标注与score≥0.5预测掩码。0.005/0.12仅用于实现边界情况测试，不作为UIIS带宽试参；真实图固定0.02。12对已匹配实例的Boundary IoU由官方边界掩码和pycocotools RLE交并比独立计算，容差1e-12。先重算既有六图诊断并要求与保存JSON一致，验证期间检查保存JSON哈希未变化。最终是否通过以报告passed和README最新记录为准，不以脚本存在作为验收通过。
