"""Write the offline comparison report from immutable diagnostic outputs."""
import json
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[1];O=R/'reports/e1-offline-comparison'
s=json.loads((O/'summary.json').read_text(encoding='utf-8'));rows=json.loads((O/'instances.json').read_text(encoding='utf-8'));probe=json.loads((O/'ranking-probe.json').read_text())
lines=['# BLL为何大目标AP下降：本地预测对照','','2026-10-23，全部691张图；无训练、无重新推理、无服务器连接。协议见docs/e1-comparison-protocol.md。','','## 结论','','当前BLL适配没有达到预期的大目标边界改善目标：共同检出的871个大目标中，边界IoU均值下降0.99个百分点，552个下降、319个提高。大目标AP下降并非固定0.5阈值下漏检变多；同一阈值的未匹配数反而少22个。类别不均衡的退化、严格IoU要求下的掩码质量和置信度排序变化共同出现，不能把AP差值全部归因于边界。','','这是对输出表现的定位，不是对BLL内部机制的因果证明；不能据此断言是weight1、width3或块占有规则单独造成，也未量化它们各自贡献。单种子不排除训练随机波动。','','## 漏检对照：score≥0.5、同类Mask IoU≥0.5','','这里“漏检”是固定匹配规则下未匹配普通GT，含低分、错类、掩码不合格等，并不都是模型完全没产生候选。','','| 尺寸 | GT数 | E0未匹配 | E1未匹配 | 变化 | E0匹配率 | E1匹配率 |','|---|---:|---:|---:|---:|---:|---:|']
for z in ['small','medium','large']:
 a=s['thresholds']['E0']['0.5'][z];b=s['thresholds']['E1']['0.5'][z];lines.append(f"| {z} | {a['gt']} | {a['missed']} | {b['missed']} | {b['missed']-a['missed']:+d} | {a['matched']/a['gt']:.2%} | {b['matched']/b['gt']:.2%} |")
lines+=['','小目标AP上涨不等于固定阈值漏检减少：0.5时两版都是539个未匹配，但具体实例有41个新增、41个找回。','','### 预定阈值敏感性（未匹配数 E0→E1）','','| 分数阈值 | small | medium | large |','|---|---:|---:|---:|']
for t in ['0.3','0.5','0.7']:lines.append('| '+t+' | '+' | '.join(f"{s['thresholds']['E0'][t][z]['missed']} → {s['thresholds']['E1'][t][z]['missed']}" for z in ['small','medium','large'])+' |')
lines+=['','大目标0.7阈值下未匹配反而增加17个，方向对分数阈值敏感，不能只报对BLL有利的0.5。大目标新增未匹配35个、找回57个；新增35个中32个仍有同类IoU≥0.5的低分候选，3个同类掩码重合不足。','','## 同一GT的边界对照','','固定图像对角线2%边界带，原分辨率；不按边界挑配对。不是Boundary AP，不能覆盖未匹配GT。','','| 尺寸 | 共同匹配/全部GT | E0边界IoU | E1边界IoU | 差值（百分点） | 下降/提高 |','|---|---:|---:|---:|---:|---:|']
for z,d in s['summary'].items():
 a=d['common_means']['E0']['boundary_iou'];b=d['common_means']['E1']['boundary_iou'];lines.append(f"| {z} | {d['common']}/{d['gt']} ({d['common']/d['gt']:.1%}) | {a*100:.2f}% | {b*100:.2f}% | {(b-a)*100:+.2f} | {d['boundary_worsened']}/{d['boundary_improved']} |")
lines+=['','小目标3例边界IoU相等。共同匹配有选择偏差，必须结合覆盖率。大目标各自匹配集合边界均值为70.00%→68.63%（906→928个），集合不同，不作为配对差值。','','共同大目标Mask IoU为87.95%→87.70%。按GT面积归一的漏涂比例5.96%→5.30%，多涂比例8.04%→9.12%；表现为平均多涂增加、漏涂减少，支持存在外扩倾向，但不能从输出证明由哪项BLL实现导致。','','## 大目标AP按类别分解','','COCO按类别等权平均，不按实例数加权。下表差值/7为各类别对大目标总AP差值的算术贡献，不是错误机制的因果归因。','','| 类别 | 大GT数（诊断分组） | E0 AP | E1 AP | 差值 | 对总AP贡献 |','|---|---:|---:|---:|---:|---:|']
counts=Counter(r['category'] for r in rows if r['size']=='large')
for cat,a in s['coco']['E0']['large_by_class'].items():
 b=s['coco']['E1']['large_by_class'][cat];delta=(b['ap']-a['ap'])*100;lines.append(f"| {cat} | {counts[cat]} | {a['ap']*100:.2f} | {b['ap']*100:.2f} | {delta:+.2f} | {delta/7:+.2f} |")
lines+=['','机器人大目标仅6个，类别AP波动影响较大；潜水员、水草也下降，沉船/遗迹明显提高，不能按多数实例的表现解释类别宏平均。','','### 大目标各IoU门槛AP','','| IoU | E0 AP | E1 AP | 差值 |','|---|---:|---:|---:|']
for t,a in s['coco']['E0']['large_by_iou'].items():
 b=s['coco']['E1']['large_by_iou'][t];lines.append(f"| {t} | {a['ap']*100:.2f} | {b['ap']*100:.2f} | {(b['ap']-a['ap'])*100:+.2f} |")
lines+=['','大目标IoU0.95门槛AP下降约4.56点，AR由29.61%降到25.34%；大目标平均AR反而由73.91%升到75.70%。因此不是单纯召回全面下降，也不能用整体AP75上涨替代大目标边界结论。','','## 排序与最佳候选补充探针（探索性）','','看到主诊断案例后额外进行：对同一871个GT，用人工标注从全部已保存同类候选中选Mask IoU最高者，不设分数阈值、不做一对一。它使用答案，不能部署、不是正式评价，不能替换上述固定配对结果。','','最佳候选Mask IoU均值90.08%→90.26%，边界IoU72.94%→72.55%，最佳候选平均分数0.656→0.577；与按分数实际匹配的候选不同的GT从268增至333。支持“较好掩码不一定获得较高分”的现象，并且解除分数限制后边界均值仍略降；不能把两个均值之差当作严格的排序AP损失分解。','','## 固定规则选取的案例','','只显示指定GT及其配对预测，其他实例省略；未匹配案例显示最高Mask IoU同类候选，明确标注not matched，不伪装为正确检测。案例按协议选取极端下降、提高、近似不变和新增未匹配，不能用于估计总体发生率。']
notes={3054:'BLL高分候选缺少约48%的GT区域。另有IoU0.952的完整候选，但分数仅0.100；原版配对IoU0.963。这里不仅是局部轮廓偏移，也有缺块候选排在前面的现象。',2886:'BLL高分候选缺块，配对IoU0.700；另有IoU0.896候选，分数仅0.159。',3424:'改善反例：原版高分候选缺块，而更好候选分数0.481；BLL选中的完整候选使IoU0.506→0.983。',860:'边界基本不变的鱼实例，避免只展示极端案例。',955:'新增未匹配：存在高重合低分候选，不能解释为完全没有分割出来。',1487:'标注类别为fish的海龟实例：BLL有IoU0.959候选，但分数约0.000172；原版候选分数0.791。图中存在掩码不代表检测通过阈值。'}
for c in json.loads((O/'cases.json').read_text(encoding='utf-8')):lines+=['',f"### {c['type']} / GT {c['instance']['gt_id']}",notes[c['instance']['gt_id']],f"![案例]({c['path']})"]
lines+=['','## 验证与建议','','本地重算COCO结果按三位小数与两组服务器报告逐项一致；未舍入整体AP32.6222→32.9595，实际差0.3372点（展示值32.6→33.0）；大目标47.7696→46.5519，差-1.2176点。固定阈值匹配计数逐项守恒；原标注/预测哈希未改；边界OpenCV实现与既有NumPy实现随机掩码逐像素一致，既有官方实现对照记录继续适用。尺寸诊断互斥阈值与COCO边界值可能不同。','','建议暂不把当前BLL当作成功的边界模块直接加入组合。先记录此负面边界证据，若继续改进应另立假设和实验编号；当前结果不足以决定该改weight、带宽还是目标构造，不直接启动下一轮长训练。']
(O/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('REPORT_WRITTEN')
