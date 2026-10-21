# 项目文件用途与清理审计（2026-10-21）

扫描全部4852个项目文件（不遍历Git内部对象），逐文件清单见本地 work/file-audit-20261021.json。数据与第三方包按归属分类，非逐张图片语义检查；未读取SSH凭据内容。

已删除22个文件，释放161287字节；空缓存/重复配置目录9个。删除项的原路径、大小、SHA256与理由均已保存。保留的报告、源码、数据标注和官方重复图片哈希复核通过。

## 删除清单

| 文件 | 字节 | 原因 |
| --- | ---: | --- |
| .local/uiis-missing.json | 31820 | 已完成补传的临时缺失文件列表；正式下载清单和数据审计保留 |
| .local/uiis-tree.json | 484 | 下载时的顶层目录查询缓存；正式download-manifest.json已保留来源版本和文件列表 |
| .local/analysis-libs/cv2/data/__pycache__/__init__.cpython-312.pyc | 444 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/gapi/__pycache__/__init__.cpython-312.pyc | 25212 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/mat_wrapper/__pycache__/__init__.cpython-312.pyc | 2002 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/misc/__pycache__/__init__.cpython-312.pyc | 291 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/misc/__pycache__/version.cpython-312.pyc | 449 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/typing/__pycache__/__init__.cpython-312.pyc | 6905 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/utils/__pycache__/__init__.cpython-312.pyc | 693 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/__pycache__/__init__.cpython-312.pyc | 9630 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/__pycache__/config-3.cpython-312.pyc | 1285 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/__pycache__/config.cpython-312.pyc | 479 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/__pycache__/load_config_py2.cpython-312.pyc | 533 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/__pycache__/load_config_py3.cpython-312.pyc | 747 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/cv2/__pycache__/version.cpython-312.pyc | 344 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/pycocotools/__pycache__/__init__.cpython-312.pyc | 269 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/pycocotools/__pycache__/coco.cpython-312.pyc | 22864 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/pycocotools/__pycache__/cocoeval.cpython-312.pyc | 29205 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| .local/analysis-libs/pycocotools/__pycache__/mask.cpython-312.pyc | 1885 | Python可再生字节码缓存；保留库源码、二进制和安装元数据 |
| reports/readme-review.json | 592 | 旧版README的49链接/17脚本检查快照已过时；非实验结果 |
| reports/uiis-baseline-36e-evaluation/runner/config-resume.py | 12577 | 与evaluation-config.py逐字节相同的本地导出副本；保留正式评价配置，不涉及服务器训练目录 |
| reports/uiis-baseline-36e-evaluation/runner/20261013_141453/vis_data/config.py | 12577 | 与evaluation-config.py逐字节相同的可视化后端配置副本 |

## 保留文件的用途

数据集4633个文件按官方数据整体保留；第三方依赖按完整安装包保留，避免只删“看似未引用”的动态加载文件。Git内部对象整体保留。以下列出其他文件。

| 文件 | 保留理由 |
| --- | --- |
| .local/id_ed25519 | SSH连接凭据/主机校验；未读取内容 |
| .local/id_ed25519.pub | SSH连接凭据/主机校验；未读取内容 |
| .local/known_hosts | SSH连接凭据/主机校验；未读取内容 |
| docs/boundary-evaluation-protocol.md | 边界评价定义与验收协议 |
| reports/analysis-permission-diagnosis-20261021.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/baseline-readiness-audit.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/environment-check.txt | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/environment-freeze.txt | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/first-epoch-review.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/formal-training-preflight.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/preflight-code-manifest.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/result-resume.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/server-check.txt | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/training-completion-review.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/training-console.log | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/training-entry-verification.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/training-exit-code.txt | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-audit.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-candidate.py | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-profile-config.py | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-profile.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-profile.log | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-ground-truth-train.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-ground-truth-val.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-smoke-100.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-smoke-50.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-smoke-checkpoints.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-smoke-config-100.py | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-smoke-config-50.py | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-smoke-console.log | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-smoke-latest.txt | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/demo-mask2former/input.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/demo-mask2former/prediction.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/demo-mask2former/summary.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/console.log | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/evaluation-config.py | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/exit-code.txt | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/metrics.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/predictions.bbox.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/predictions.segm.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/report.md | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/examples/0.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/examples/1.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/examples/2.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/examples/3.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/examples/4.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/examples/5.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/runner/20261013_141453/20261013_141453.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-baseline-36e-evaluation/runner/20261013_141453/20261013_141453.log | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-boundary-pilot/reference-comparison.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-boundary-pilot/validation.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-error-analysis/diagnostics.json | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-error-analysis/report.md | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-error-analysis/examples/0.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-error-analysis/examples/1.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-error-analysis/examples/2.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-error-analysis/examples/3.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-error-analysis/examples/4.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| reports/uiis-error-analysis/examples/5.jpg | 环境、数据、训练或分析的原始证据/可视化；不是可随意删除的运行垃圾 |
| scripts/analyze-uiis-boundaries.py | 全量条件边界诊断；依赖六图、官方对照及误差匹配实现 |
| scripts/analyze-uiis-errors.py | 阈值误差诊断及边界分析复用的匹配逻辑 |
| scripts/demo-mask2former.py | COCO推理检查，且提供基线预训练权重下载与哈希校验入口 |
| scripts/download-uiis.py | 固定版本数据下载、哈希校验与audit-only入口 |
| scripts/evaluate-uiis.py | 固定最后权重全量评估及防覆盖 |
| scripts/evaluation-worker.sh | 评估后台日志和退出码记录 |
| scripts/install-env.sh | 服务器环境重建 |
| scripts/plot-uiis-boundaries.py | 可复用的条件边界诊断绘图工具 |
| scripts/prepare-uiis-baseline.py | 正式train-uiis.py直接导入build_config；非废弃临时脚本 |
| scripts/report-uiis-evaluation.py | 从真实预测与指标生成基线报告 |
| scripts/run-training.sh | 训练启动、状态查询和并发保护 |
| scripts/run-uiis-smoke.sh | 短训练50到100步恢复验证入口 |
| scripts/test-training-entry.py | 真实训练入口锁、保存、恢复及保留规则验证 |
| scripts/train-uiis-smoke.py | 小规模训练管线验证，未来环境变化仍可用 |
| scripts/train-uiis.py | 正式基线训练、恢复与阶段保存实现 |
| scripts/training-worker.sh | 正式训练与入口验证的后台执行器 |
| scripts/validate-uiis-boundary.py | 被全量分析直接导入的边界核心与小规模协议验证 |
| scripts/validate-uiis.py | 官方数据结构、掩码、重复图审计 |
| scripts/verify-boundary-reference.py | 固定官方源码对照验证 |
| scripts/verify-env.py | GPU算子前后向与二进制兼容检查 |
| scripts/verify-uiis-error-analysis.py | COCO独立复核阈值匹配计数 |
| work/cleanup-20261021-183203.json | 已执行清理的逐项记录 |
| work/boundary-reference/boundary_utils.py | 官方对照验证实际加载的固定版本源码 |
| work/boundary-reference/opencv-wheel-manifest.json | OpenCV下载来源与校验哈希证据 |
| .gitattributes | 跨Windows/Linux行尾规则 |
| .gitignore | 防止数据、密钥、权重与运行输出误提交 |
| AGENTS.md | 协作与实验约束 |
| check-server.ps1 | 服务器只读连接检查 |
| connect.ps1 | SSH连接入口 |
| README.md | 项目状态、计划、协议与历史记录 |
| requirements.txt | 固定服务器依赖版本 |
| run-demo.ps1 | 调用COCO/预训练权重获取脚本并回传结果 |
| setup-ssh.ps1 | 重建连接时安装公钥；不是每轮运行 |
| ssh.config | SSH主机别名和密钥路径 |
| sync-code.ps1 | 按Git清单同步代码 |
| underwater-segmentation.code-workspace | VS Code项目打开入口 |

没有为了减少数量删除有效代码：21个脚本各有训练、环境、审计、评价或复现用途；入口间依赖和动态加载路径已核对。未执行训练、推理或服务器连接。包缓存会随以后运行重新产生，无需反复清理。

本清单及README纳入Git；详细逐文件JSON保存在work并被Git忽略。
