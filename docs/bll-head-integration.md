# BLL候选head：代码接入与验收边界

2026-10-21。P3进行中，E1未定案，没有启动训练。

## 本次实现

`scripts/bll_mask2former_head.py`声明独立注册类BLLMask2FormerHead，继承MMDetection 3.3.0原head。默认权重0，直接调用父类loss_by_feat，返回原损失字典，不增加零值BLL键。没有新参数，forward/predict继承原实现。原E0入口不导入这个类。

启用时仍只调用一次父类loss_by_feat。在原_get_targets_single返回后观察sampling_result，保持原匹配、随机采样和损失顺序。固定上游按层、再按图遍历，取最后一层的B组query/GT索引，传给既有query_boundary_loss；不调用第二次get_targets，不依赖上游可选返回分支。检查采样结果数量，异常时finally清除临时引用，不保存计算图。仅支持单进程FP32；多进程和非FP32 logits明确拒绝。

`configs/bll-head-probe.py`仅是head验收覆盖项，不是完整训练配置。权重1和width3仅作数值探针。没有修改E0、训练预算、数据划分或AP；没有实现MFRM/BMS推理融合。

为支持注册模块包导入，query_bll只调整了import：包调用和直接脚本调用均可使用。历史报告保留原哈希，本轮重跑报告记录当前哈希。

## 已执行的检查

- `scripts/validate-bll-head-contract.py`退出码0，6组接口检查通过。实际执行候选类代码及固定上游loss_by_feat遍历代码，但单层损失和匹配使用替身。检查关闭时值/梯度/RNG一致、启用时原损失/RNG不变、匹配调用数不增加、使用最后一层匹配、梯度仅额外进入最终logits、父类异常后的清理和重试。报告：reports/bll-head-contract/validation.json。
- 修改导入后的原跨层测试重跑：19项通过，3个匹配实例、266个像素。报告：reports/query-bll-package-check/validation.json。
- `scripts/verify-bll-head-framework.py --device cpu --output reports/bll-head-framework-local/validation.json`退出码1：ModuleNotFoundError: No module named 'mmdet'。在导入阶段停止，未构建或运行真实head。本地只安装了独立CPU测试依赖，当前不是pycocotools文件访问故障。

## 服务器验收入口（已编写，尚未运行）

按AGENTS完成只读连接检查、E0备份和代码同步后，在服务器原项目环境执行：

```bash
python -B scripts/verify-bll-head-framework.py --device cuda:0
```

脚本使用E0的head配置和训练匹配配置，构造同权重的原head与候选head；一张128×128画布的合成多尺度特征、两个合成实例，执行真实pixel decoder、Transformer decoder、Hungarian和损失。weight=0要求各层输出、全部原损失、参数梯度、随机状态逐项精确一致；weight=1要求原损失/RNG不变、新项正且有限、参数梯度有限且发生变化；再做state_dict内存保存/读取后的损失和梯度一致性验证。没有优化器步进，不调用Runner.train，不加载或覆盖E0权重。报告拒绝覆盖。

该入口尚未经真实框架运行，可能暴露接口或算子差异；失败必须修复并记录，不能跳过验收。记录的单次耗时/峰值显存受冷启动及两个head同时驻留影响，只是诊断信息，不能作为性能增减结论。它不包含backbone、数据预处理、真实UIIS图像、优化器恢复或训练收益验证。

通过后再进行完整模型与少量固定训练图的检查，核实实际padding、GT缩小与小物体丢失情况，并检查正式保存/恢复和开销；这些条件通过前不进入长训练。AMP/DDP、正式权重/带宽、是否选择BLL为E1均未定案。

## 2026-10-21服务器实测更新

此前“尚未运行”描述的是第31节完成时的状态。现已运行，详见README第32节及reports/bll-head-framework：CPU精确检查通过；GPU精确梯度检查失败，原版自身重复也不逐位一致，加入原版重复对照并明确采用atol=1e-6/rtol=1e-5后GPU数值检查通过。输出、原损失及RNG仍精确比较。默认梯度容差仍为0；可显式传--gradient-atol和--gradient-rtol。失败与成功报告均保留。尚未完成真实图/完整模型、优化器保存恢复与开销验收。
