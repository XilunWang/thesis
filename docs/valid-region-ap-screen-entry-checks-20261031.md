# AP短程对照入口验收范围（2026-10-31）

本轮只验收，不执行两组真实3轮或任何模型optimizer.step。使用冻结control/candidate配置、E0权重和官方3937/691数据规则，保留原重复图。最终结果单独记录于valid-region-ap-screen-entry-result-20261031.md及reports/valid-region-ap-screen-entry-20261031/acceptance.json。

## 配置与恢复

两组实际MMEngine加载、数据长度3937/批次数1969、原权重状态精确加载、fresh epoch/iter=0、初始优化器为空、LR按1e-5及backbone倍率0.1计算。有效区域开关及配置/head哈希写入screen_contract，恢复时独立核验，因为开关不在state_dict中。

为避免验收产生训练，借用E0现有345组非空AdamW状态，仅作序列化fixture；设置显式诊断计数后，用Runner原生save_checkpoint，另起Python进程resume，并比较模型、优化器、空调度列表、epoch/iter、开关及显式保存的Python/NumPy/Torch/CUDA RNG。fixture写diagnostic_only=True，实际训练/评价必须拒绝它。真实试验仍新建优化器，不继承这些fixture动量。

该检查不承诺中途数据worker预取队列/游标逐位续接；执行入口不自动重启或恢复中断试验。将来发生中断须先检查完整checkpoint与数据进度，不宣称可无缝重放。

## 停止和异常控制

使用原生EpochBasedTrainLoop配无参数假模型和1969批虚拟数据，验证正好3轮/5907次控制流迭代，并测试额外更新拒绝；这是循环测试，不是模型训练。deadline注入验证更新前保存并停止。

仅对CPU标量与空AdamW做负对照：NaN loss在backward前拒绝、Inf梯度在step前拒绝，标量未变且优化器状态为空。编排函数注入OOM，要求失败记录completed=False且不执行评价、不进入第二组。错误轮次、步数、开关、诊断checkpoint和不完整checkpoint均不得通过最终评价准入。

运行时两组统一加ScreenFiniteOptimWrapper和APScreenGuard，只做有限性、预算及协议检查，数学优化配置不变。原本冻结配置文件不改，运行时封装与入口源码哈希纳入验收封存。

## 评价回归

共用实际新入口的评价函数，对未更新E0权重在691图做一次完整重放。它明确是baseline replay，不冒称候选epoch3结果。要求691图ID覆盖完整、预测与原保存结果一致，并用原COCO maxDets重新核验未四舍五入AP。

实际执行入口必须先核验真实epoch3/5907及每个非空优化器状态step=5907，才调用评价；不接收epoch1/2、fixture、partial_stop或错误开关配置。固定epoch3，不选best。只有两组训练和评价均完成才输出比较成功，失败不重试、不降参数、不追加轮次。

## 实际运行入口

scripts/run-valid-region-ap-screen.py不带--execute立即拒绝，导入脚本不启动训练。执行时核验acceptance.passed和封存源码/配置哈希、争用训练锁、拒绝现有工作目录，固定先A后B，每组3轮后评价；总预算4小时，训练批次/评价图片边界检查deadline。由于当前算子调用不能被安全切断，边界停止并非操作系统级精确秒表硬杀。

当前阶段仍没有授权--execute，不因验收通过自动调用该入口。
