# BLL完整模型和优化器保存恢复检查

2026-10-21，scripts/verify-bll-checkpoint.py服务器执行退出码0，reports/bll-checkpoint/validation.json中passed=true，报告已取回本地。

## 实际执行

使用E0模型参数初始化独立BLL完整模型（weight1、width3），按E0 optim_wrapper配置新建AdamW与调度器。对固定训练图0/1沿用既有增强种子做一次真实前后向及优化器更新，使动量等状态真实产生；这是隔离验收副本，非E1或短试跑。调度器显式推进一次虚拟tick以测试状态持久化，不代表完成一轮训练。

保存完整模型state_dict（含缓冲区）、optim_wrapper state_dict、调度器、wrapper内部计数、Python/NumPy/torch/CUDA RNG和元数据至独立work_dirs/bll_checkpoint_verification/checkpoint.pth。文件约505 MiB，SHA256为56c15787eb1e69222adb0500163226137060c99cbd6de501661b030f99e21599；仅此测试checkpoint在服务器，未声称已下载到本地。脚本拒绝覆盖既有目录或报告。

父进程对有梯度的345个参数张量施加固定1e-4梯度并按同一wrapper规则更新，保存下一状态的指纹。随后新Python进程重新构建模型、优化器和调度器，读取磁盘checkpoint：恢复后各状态及RNG的完整内容哈希均与保存前一致；用相同固定梯度推进后的模型、优化器、调度器、计数及RNG也与父进程逐项一致。

固定梯度对照隔离了GPU反传非确定性，证明当前配置下的状态恢复与优化器继续更新一致；不把它当成两次真实GPU训练轨迹逐位等价。子进程在恢复并完成受控更新后，再读取真实图0/1做一次新的前后向，所有损失及梯度有限，未执行该次梯度的优化器更新。

## 通过项与限制

- 345个参数的真实梯度生成AdamW状态，状态非空；恢复后的参数组、学习率、动量、step等由完整状态指纹核对。
- strict=True加载完整模型，参数/缓冲区、调度及随机状态恢复一致；固定梯度的下一步状态一致。
- 完整真实数据前后向在恢复后可运行。
- 父/子进程均核验E0源权重SHA256与原记录一致；结束显存0 MiB。

本次对独立副本有1次真实梯度更新，以及父/子两条对照路径各1次固定梯度更新；不能称为“完全没有优化器更新”，但未开展训练循环、短试跑或正式E1，也未覆盖E0。

使用直接model/optim_wrapper检查入口，尚未验证MMEngine Runner恢复、多worker数据游标、梯度累积、AMP、DDP及正式epoch恢复。调度tick测试也不证明已走到第27/33轮里程碑。下一步按docs/bll-short-pilot-protocol.md实现带明确游标/预算与中途恢复的短试跑入口，再按该协议执行；本轮只制定协议，没有启动试跑。
