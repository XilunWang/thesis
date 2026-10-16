# 研究汇报 PPT 维护

当前版本：output/水下实例分割阶段汇报_20261014_v1.pptx（2026-10-14，16页）。
沿用用户提供的 4.27组会王雨乐.pptx 的母版、校徽、配色及字体；不沿用原作者身份。原模板保存在 templates/group-meeting-template.pptx，仅本地使用。

## 更新规则

- 仅在一个大阶段完成或用户明确要求时更新PPT：如基线阶段收尾、第一研究点完整实验/消融完成、第二研究点完成、开题/中期/答辩。日常小步骤只更新README及Git，不自动重建或渲染PPT，以减少token和时间消耗。
- build.mjs 为内容与版式源，status.json 为日期及输出文件名；指标从 reports/uiis-baseline-36e-evaluation/metrics.json 读取，不能手动编造成绩。
- 新增实验须先核对原始报告，再增加对应结果页和讲者备注来源；保留基线及局限，区分计划和完成。第二研究点仍待定。
- 修改日期及版本号后生成新文件，保留旧版本，不覆盖已汇报的 PPT。当前不知道汇报人姓名，封面未填姓名。
- 在 PowerPoint 手工改动后，应同步到生成源，否则下次重建不会保留手工修改。
- 模板、报告和 PPT 输出暂不提交 Git；源码、维护约定与 README 提交。需备份这些本地素材才能在其他电脑完整重建。

## 生成与检查

使用 Codex Presentations skill 的 bundled Node 运行 build.mjs；可通过 CODEX_PRESENTATION_RUNTIME 和 CODEX_PRESENTATION_SKILL 指定运行时及技能路径。
构建器导出 .build/candidate.pptx、逐页 PNG，再调用 finalizer 写入独立输出。每次均需核对所有页、数据、可编辑表格、模板样式、讲者备注来源。

早期检查记录：finalizer 在 Node realpath 阶段发生 EPERM，定向授权后仍失败，未宣称完整终检通过。已独立运行原生结构与几何检查器，均为零发现；13 页逐页检查完成，原生表格页 3/4/5/6/7/12 确认；Artifact Tool 重导入 13 页成功。检查后将候选副本交付到 output。尚未在 PowerPoint 原生程序中打开验证。

## 版本记录

- v1 / 2026-10-14：16页整合阶段汇报，包含正式基线、全量阈值诊断、固定案例、两类报告口径和后续路线。依赖 reports/uiis-error-analysis/diagnostics.json。
- 用户要求重新从v1编号：原整合版仅重命名，SHA256保持一致；此前13页和15页PPT已删除，output目录只保留此版本。后续大阶段或明确要求更新时从v2递增。
- 本版独立结构/几何检查及16页重导入通过；自动finalizer仍因realpath EPERM受阻，未在原生PowerPoint验证。此轮仅重命名、清理和更新引用，未重建或重渲染。
## 开题报告独立版本

- v1 / 2026-10-16：output/水下实例分割_硕士开题报告_20261016_v1.pptx，共23页；build-proposal.mjs 为生成源，templates/proposal-template.pptx 为用户新模板的本地副本。
- 与16页阶段汇报分别维护，不覆盖原汇报。姓名、专业、导师和日期待补，第二点明确为拟研究方案；研究计划使用相对月份。
- 23页逐页检查完成。finalizer的结构、布局、字体与Artifact Tool重导入检查已通过，最终硬链接发布遇Windows EPERM；普通复制候选文件并核对SHA256一致后交付。未在原生PowerPoint验证。当前生成脚本仍会在该发布步骤报错，重建时不得忽略更早的验证错误。
- 输出及模板按既有规则仅留本地；Git记录生成源码和文档。

### 当前开题版本：v2（10页，2026-10-16）

用户要求改为口语化、每页解释更完整的10页版本。文件：output/水下实例分割_硕士开题报告_20261016_v2.pptx。build-proposal.mjs为当前源，build-proposal-v1.mjs保留23页历史源。v1及阶段汇报未删除。

去掉目录和章节过渡页，使用海星案例解释研究问题，说明现成模型结果与自有改进的区别。第二点尚待论证，封面信息待补。10页均已渲染检查。最终硬链接发布受Windows限制时，脚本仅对该具体错误采用独占复制，其余验证错误仍终止。
