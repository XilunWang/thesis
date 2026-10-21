# BLL独立合成验证

本步骤只检查三阶段边界损失规则，不接入Mask2Former、不读取UIIS、不训练、不计算AP。原文和原始代码依据见[p3-boundary-loss-review.md](p3-boundary-loss-review.md)。

## 实现与参考的关系

- `scripts/bll_core.py`独立实现三阶段规则，固定阶段权重0.25/0.65/1、边界宽度3、阈值0.15、bilinear和align_corners=True。只接受二值GT和FP32 logits。
- `scripts/validate-bll.py`校验固定原始源码SHA256后，用AST只提取官方`generate_block_target`和`LaplacianCrossEntropyLoss`。类的注册装饰器被移除，不安装或执行整个MMDetection。
- 官方类使用的通用BCE辅助函数在此以PyTorch mean BCE适配，适用范围严格限于本测试的二值目标、单位权重、默认mean reduction；不宣称覆盖原框架ignore_index、类别展开和任意权重API。
- 本地实现给零实例增加连接计算图的零损失。这是明确的健壮性扩展；官方零实例的实际表现单独记录，不能因本地返回0就声称官方也通过。

## 检查内容

24组组合：6种图形（矩形、错位轮廓、细线、贴边、全前景、空前景）×2种空间比例×2种批量大小（1/3）。三阶段为14→28→56，长方形额外使用宽度21→42→84。

1. 使用独立Python循环逐位置数不相同邻居，验证本地及官方提边界结果。检查零padding导致的图像外缘边界。
2. 用独立NumPy双线性插值验证预测/GT边界并集的放大和二值化。
3. 用NumPy稳定BCE公式、逐像素权重及解析导数验证损失和实际autograd梯度；同时与提取的官方函数输出/梯度比较。
4. 确认最终层区域外该项梯度为零、区域内有梯度；前两层仍保留完整BCE梯度，没有来自硬区域选择的额外梯度。
5. 固定前两阶段和选区，仅把最终logits变得更符合GT，检查最终损失降低；空选区保持零。这不是任意几何变化下损失必定单调的声明。
6. 7/8个不同邻居、等于阈值的手算案例；零实例和非二值输入拒绝检查。空前景案例覆盖非空批次但无边界像素的情况。

损失容差为atol/rtol=1e-6，梯度atol=1e-7、rtol=1e-5。CPU、单线程、固定随机种子20261021。报告保留逐案例数值、环境、源码哈希、参考来源和基础Git提交；已有报告拒绝覆盖。

## 运行

本机测试使用项目内隔离CPU PyTorch环境`.local/bll-libs`，不修改服务器Python3.10/PyTorch2.1.2+cu121。项目测试解释器为bundled Python3.12，PyTorch固定2.6.0+cpu；后续必须在实际训练环境再次验证，不将本地通过等同于服务器通过。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.local/bll-libs'
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B scripts/validate-bll.py
```

默认报告`reports/bll-synthetic/validation.json`被Git忽略，需独立保管。重验必须用`--output`指定未存在的新文件。不要为重跑修改既有报告或反复训练基线。

## 后续门槛

这里没有任何Mask2Former实例分配、跨decoder层对应、padding有效区域处理或模型保存恢复测试。通过以后先固定如何复用同一次最终层匹配、读取前后层同一query，再实现独立适配入口。原有BCE/Dice、数据、推理和E0配置先保留；E1、适配带宽和权重仍未定案。

## 2026-10-21实际结果

首次运行退出码0，报告passed=true。24组案例全部通过；3个阈值手算检查、零实例扩展和非二值输入拒绝检查通过。24组与提取官方函数的总损失绝对差均为0，梯度亦通过设定容差与独立解析导数对照。

例如batch=1、14→28→56方形的错位轮廓案例，在固定选区下修正最终预测，最终边界项从0.33177096降到0.00247572。此数字仅是合成测试的行为验证，不是训练成绩或预计AP提升。

原始官方函数在本地CPU环境的零实例输入返回非有限值；本地实现返回连接三阶段计算图的0，反向后梯度存在且有限。此差异在报告的upstream_zero_instances中显式保存，不能宣称零实例路径与官方等价。

实际环境为Python3.12、PyTorch2.6.0+cpu，完整隔离依赖版本保存在报告。安装来自官方CPU索引，命令为：

```powershell
python -m pip install --target .local/bll-libs --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
```

下一步可以进入Mask2Former适配定义：先确定同一真实实例对应的前后层query及边界网格，不直接开始长训练，也不把当前三阶段RoI测试当作已接入query模型。
