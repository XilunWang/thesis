process.on('uncaughtException',e=>{console.error(e.message);process.exit(1)});
import fs from 'node:fs/promises';import path from 'node:path';import crypto from 'node:crypto';import {pathToFileURL} from 'node:url';
const ROOT=path.resolve(import.meta.dirname,'..'),R='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies',SK='C:/Users/Lenovo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(R+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
process.env.RUNTIME_NODE_MODULES=R+'/node/node_modules';
const B=ROOT+'/presentations/.build/proposal-v6',T=ROOT+'/presentations/templates/proposal-template.pptx',F=ROOT+'/presentations/output/水下实例分割_硕士开题报告_20261016_v6.pptx';
const p0=await PresentationFile.importPptx(await FileBlob.load(T));const proto=p0.toProto();
const text=e=>(e.paragraphs||[]).flatMap(x=>x.runs||[]).map(x=>x.text||'').join('');
for(const i of [0,13]){proto.slides[i].elements=proto.slides[i].elements.filter(e=>e.type===7||(e.type===1&&!text(e)));proto.slides[i].speakerNotes=undefined;}
for(const i of [3,6,9,12]){proto.slides[i].elements=proto.slides[i].elements.filter(e=>{const b=e.bbox||{};return (b.xEmu<150*9525&&(b.widthEmu||b.wEmu||0)<160*9525)||(b.yEmu<105*9525&&e.id!=='21')});proto.slides[i].speakerNotes=undefined;}
for(const sl of proto.slides)for(const e of sl.elements)for(const para of e.paragraphs||[])for(const run of para.runs||[])if(run.text==='Doctoral Research Proposal')run.text='Research Content';
const p=Presentation.load(proto);p.slides.keep([0,1,2,3,5,6,8,9,11,12,13]);
const [cover,toc,d1,b1,d2,b2,d3,b3,d4,b4,end]=p.slides.items;const ordered=[],used=new Set(),tables=[];const green='#19652B',ink='#333333';
const refs={mask:'https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html',m2f:'https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html',water:'https://openaccess.thecvf.com/content/ICCV2023/html/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.html',dino:'https://arxiv.org/abs/2304.07193',dive:'https://arxiv.org/abs/2511.08334',usis:'https://proceedings.mlr.press/v235/lian24c.html',boundary:'https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html'};
function tx(s,t,x,y,w,h,size=26,o={}){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:'微软雅黑',fontSize:size,color:o.color||ink,bold:o.bold||false,alignment:o.align||'left',verticalAlignment:'middle',wrap:true,insets:{left:0,right:0,top:0,bottom:0}};return a;}
function notes(s,t,src=[]){s.speakerNotes.text=t+'\n\n来源：\n'+src.join('\n');}
function take(base,title){let s=base.duplicate();ordered.push(s);tx(s,title,200,38,730,49,32,{bold:true,color:green});return s;}
function lines(s,items){items.forEach((a,i)=>{tx(s,a[0],210,145+i*155,990,42,29,{bold:true,color:green});tx(s,a[1],210,194+i*155,990,91,25)});}
function table(s,v,widths,y=160,h=350,size=24){const t=s.tables.add({rows:v.length,columns:v[0].length,left:205,top:y,width:1020,height:h,values:v,columnWidths:widths});for(let r=0;r<v.length;r++)for(let c=0;c<v[0].length;c++){let a=t.getCell(r,c);a.fill=r===0?'#E9F1E9':'#FFFFFF';a.text.style={typeface:'微软雅黑',fontSize:size,color:r===0?green:ink,bold:r===0,verticalAlignment:'middle'};}t.borders.assign({style:'solid',fill:'#CAD7CB',width:.6});tables.push(s);}
function foot(s,t){tx(s,t,210,651,1000,35,18,{color:'#666666'});}
async function img(s,file,x,y,w,h){s.images.add({blob:new Uint8Array(await fs.readFile(ROOT+'/'+file)).buffer,contentType:'image/jpeg',alt:file,fit:'contain',position:{left:x,top:y,width:w,height:h}});}




await fs.mkdir(B,{recursive:true});
ordered.push(cover);
tx(cover,'面向水下场景的\n实例分割方法研究',80,226,1120,185,62,{bold:true,color:'#FFFFFF',align:'center'});
tx(cover,'拟研究方向：细节增强与边界约束的水下实例分割',180,450,920,50,28,{color:green,align:'center'});
tx(cover,'姓名：待填写      专业：待填写\n导师：待填写      汇报日期：待填写',340,525,650,100,26);
notes(cover,'【开场】本课题研究水下实例分割，已完成Mask2Former基线训练。接下来拟研究如何改善目标细节与轮廓。今天重点说明现有问题、准备改动的位置以及验证方案。\n【状态】这是方案讨论稿。第一点结构尚未实现，第二点待论证，题目暂定。');
let s=take(b1,'1.1 研究任务与目标');
lines(s,[['研究对象：水下图像中的独立目标','输入单张水下图像，输出每个目标的类别与像素区域。以UIIS数据集的7类目标为对象，研究不同实例的识别与分割。'],['研究问题：目标细节与轮廓不完整','水下图像的低对比度、遮挡及目标尺度差异增加分割难度。结合已有基线案例，重点分析细长结构缺失与局部区域偏差。'],['研究目标：形成可验证的模型改进','以Mask2Former-R50为基线，研究细节特征细化与边界监督的适配方法，评价目标轮廓质量、整体分割性能及新增计算开销。']]);
foot(s,'应用意义：为水下生物计数、形态分析与巡检图像处理提供目标区域信息。');
notes(s,'【讲稿】这篇论文要做的东西是一个改进后的水下实例分割模型。输入一张水下图片，输出每只动物各自的区域。研究不涉及重新制作数据集，也不是使用大语言模型分析图片。当前主线是基于已训练的Mask2Former，研究目标细节和轮廓的改进。\n【解释】实例分割要把每个目标分开，例如三只海星输出三个区域。细节与轮廓是否是主要问题，需要后续全量分析支持。',[refs.water]);
s=take(b2,'2.1 研究现状与研究切入点');
table(s,[['技术方向','已有研究特点','本课题关注问题'],['通用实例分割 [1]','通过统一框架完成实例级区域预测','复杂水下场景的目标细节表达'],['水下细粒度分割 [2]','通过细节建模与边界学习改善区域表达','细节信息与轮廓监督的协同'],['预训练特征适配 [3,4]','利用预训练视觉信息开展水下任务适配','水下特征适应与分割性能的关系']],[240,430,350],155,320,24);
tx(s,'研究切入点：面向细长结构与局部轮廓，\n研究细节增强和边界约束的联合建模方法。',210,520,1000,95,27,{color:green});
foot(s,'参考：[1] Mask2Former  [2] WaterMask  [3] DINOv2  [4] DiveSeg。文献链接见备注。');
notes(s,'【讲稿】Mask2Former是起点，WaterMask是借鉴对象。我们不是把两个模型直接串在一起，而是研究WaterMask中的细节和边界思想如何适应Mask2Former。DINOv2和DiveSeg暂时只作为第二点候选，不影响第一主线。\n【解释】现有论文给出的是思路和实现参考，自有贡献必须落在具体设计及其作用证据上。',[refs.m2f,refs.water,refs.dino,refs.dive]);
s=take(b3,'3.1 问题证据：基线预测案例');
await img(s,'reports/uiis-baseline-36e-evaluation/examples/1.jpg',205,140,490,280);
await img(s,'reports/uiis-baseline-36e-evaluation/examples/2.jpg',730,140,490,280);
tx(s,'目标已识别，细长结构未完整保留',205,430,490,45,25,{bold:true,color:green});
tx(s,'部分小目标未显示预测结果',730,430,490,45,25,{bold:true,color:green});
tx(s,'研究关注：在保持目标识别能力的同时，改善区域完整性与轮廓质量。\n下一步需量化边界表现，并将漏检、类别错误与区域偏差分别分析。',210,505,1000,110,26);
foot(s,'每组左侧为标注、右侧为预测，显示阈值0.5。案例用于提出问题，不代表全量结论。');
notes(s,'【讲稿】这页让老师看到我们为什么准备研究细节。海星被识别出来了，但细长部分没有完整分出。右侧小目标未显示，也可能与置信度阈值有关。因此两类问题不能混为一谈。第一研究点优先关注区域细节与轮廓，漏检原因需要单独诊断。\n【解释】阈值0.5表示只展示得分达到0.5的结果。没有显示不等于模型完全没有输出。边界专用评价还没完成，不能先断言边界是所有错误的主要来源。',['reports/uiis-baseline-36e-evaluation/report.md','reports/uiis-error-analysis/report.md']);
s=take(b3,'3.2 总体研究方案（拟）');
table(s,[['模型环节','原始流程','拟研究改动'],['输入','UIIS水下图像','保持官方数据及标注规则'],['特征提取','ResNet-50与像素解码器','提取并融合多尺度图像信息'],['掩码特征','生成用于实例区域预测的特征','拟设计：细节增强单元'],['分割预测','输出目标类别与实例掩码','保留原实例预测流程'],['训练监督','原有分类与掩码损失','拟设计：边界约束分支']],[215,420,385],145,385,23);
tx(s,'输出保持一致：目标类别、置信度与实例区域。\n通过细节增强与轮廓约束改善实例区域表达。',210,555,1000,85,26,{color:green});
foot(s,'研究方案拟作用于掩码特征与训练监督，具体结构将在实现与实验中细化。');
notes(s,'【讲稿】这一页回答模型到底改哪里。输入仍是水下图片，输出仍是每个目标的类别和区域。首先保持骨干和原预测过程，拟在掩码特征附近研究两个候选改动：一个补充细节，一个在训练时关注边界。\n【解释】特征是模型内部的图像信息。损失是训练时衡量预测与标注差距的规则。表中自上而下是处理环节，最后一行是训练时的约束，不是推理输出之后的处理模块。\n【注意】这是本轮具体化的讨论方案，不代表已确定接口。WaterMask模块需核对，不能声称即插即用。',[refs.m2f,refs.water]);
s=take(b3,'3.3 边界约束分割方法（拟）');
lines(s,[['研究目标','针对细长目标轮廓缺失与局部区域偏差，拟引入实例边界约束，增强模型对目标轮廓的学习能力，并兼顾整体分割性能。'],['方法设计','利用实例标注构建边界监督信号，拟在掩码特征端设置边界预测分支，将区域分割目标与边界学习目标联合优化。'],['验证方案','分析实例对应、边界尺度与监督权重的影响，通过边界质量、实例AP及固定案例对照，验证轮廓约束的效果与适用范围。']]);
foot(s,'预期作用：改善局部轮廓表达。效果将通过统一条件下的对照实验验证。');
notes(s,'【讲稿】为了把方案落地，先提出一个足够具体的实验：让模型训练时不仅学习目标内部区域，也学习轮廓位置。可以考虑从现有人工标注生成边界，不必重新人工标注，再增加一个训练用的边界辅助分支。\n【解释】辅助分支可以理解为训练时的额外练习。它是否能改善主分割结果要靠实验。边界的宽度、预测分辨率和损失权重都要明确，参数选择应使用合适的开发验证规则。\n【重要】这是待核对的候选结构，不是WaterMask原代码的复刻，也不是已证明的新颖方法。若源码或问题分析不支持，应调整方案并更新记录，不强行实施。',[refs.water,refs.m2f,refs.boundary]);
s=take(b3,'3.4 细节增强与协同优化（拟）');
lines(s,[['局部细节信息增强','针对细长结构信息不足，拟在掩码特征中引入局部细节增强单元，改善目标细部信息的利用，研究其对区域完整性的影响。'],['区域表达与轮廓学习协同','以细节增强改善特征表达，以边界监督约束轮廓学习，拟构建两者协同的实例分割方案，分析其在细长目标和复杂背景中的作用。'],['模块贡献与效率分析','设置细节增强、边界约束及联合方案的消融实验，分析各设计对分割性能的贡献，并结合显存和训练成本评价方法效率。']]);
foot(s,'预期作用：提高细部区域完整性，并检验细节增强与边界约束的互补性。');
notes(s,'【讲稿】边界辅助实验是首个候选，后续再选择一种细节细化方式。前者主要改变训练要求，后者改变模型内部处理信息的方式。要分别跑原始模型、边界候选、细节候选，最后才看组合。\n【解释】DSGAT和MFRM来自WaterMask，不能在尚未理解接口时直接全部照搬。单项和组合的比较就是消融实验。\n【注意】这仍是第一主线内部的设计，不应为了满足两点要求把两个小模块直接包装成两个独立研究点。',[refs.water]);
s=take(b3,'3.5 前期结果与正式对照设计');
table(s,[['实验或评价','设置与状态'],['基线：已完成','Mask2Former-R50，COCO预训练，UIIS 36轮。\n实例AP 32.6、AP50 47.6、AP75 35.3'],['边界约束：拟开展','固定原训练条件，增加经核对的边界辅助监督'],['细节增强：拟开展','固定原训练条件，加入一种细节特征细化设计'],['联合方案：拟开展','基于单项结果判断是否组合，验证互补性'],['统一评价','实例AP、尺寸分组、边界质量、固定案例与计算开销']],[280,740],150,390,23);
foot(s,'UIIS训练3937张、测试691张、7类。保留既定划分与重复图片规则，关键实验重复验证。');
notes(s,'【讲稿】这一页把已完成和未来实验明确分开。目前只有基线完成，AP32.6不是本课题改进后的成绩。以后每一项改动都与这个起点在一致规则下比较，同时看边界和整体性能，防止只改善一个指标却损害其他方面。\n【解释】AP综合不同匹配严格程度的表现，AP50和AP75分别要求重合度至少为0.5和0.75。32.6不是像素准确率。\n【公平性】预训练、优化器与增强等条件需要控制。WaterMask公布分数只能作参考，不能据其差值宣称本方法贡献。尚未完成开发验证方案，不应使用测试集反复选参数。',['README.md','reports/uiis-baseline-36e-evaluation/metrics.json']);
s=take(b3,'3.6 研究内容二：水下特征适配（拟）');
lines(s,[['研究问题','针对水下图像与通用图像的外观差异，拟研究预训练视觉特征的水下适应方法，提升特征对水下目标的表达能力。'],['拟研究方法','以DINOv2为特征提取基础，探索轻量适配与特征融合方式，构建适合水下实例分割的特征表示，并与分割预测过程衔接。'],['对照与联合验证','固定预训练骨干与训练条件，比较适配前后的分割性能及开销。在独立验证后，分析其与细节增强、边界约束的联合效果。']]);
foot(s,'该方向为拟研究内容，具体适配结构与创新性将在文献论证和预实验中确定。');
notes(s,'【讲稿】论文希望有两个研究点，但第二点目前确实没有具体定下来。DINOv2和DiveSeg只是候选参考，不能仅凭一个模型名称就说研究方案完整。需要明确独立问题、具体结构和公平对照。\n【说明】本页保留真实状态，避免让导师误以为已有两套确定的方法。第一主线可以继续细化，第二点还需要文献和可行性依据。',[refs.dino,refs.dive]);
s=take(b4,'4.1 研究计划与预期成果');
table(s,[['阶段','工作内容','验收依据'],['问题与接口核对','补充边界评价，阅读两模型相关源码，\n明确方法结构与实施位置','问题证据与明确实现方案'],['边界约束研究','实现边界约束分支，完成训练检查，\n再开展统一条件的正式对照','训练正确性与指标变化'],['第一主线完善','实现细节增强，完成单项与组合实验，\n分析效率、稳定性及失败案例','完整消融与方法结论'],['第二点与论文整理','论证第二点，确定后再实施，\n同步完善方法章节与实验材料','独立贡献与学位论文']],[230,475,315],150,365,23);
tx(s,'预期成果：形成方法设计、对照与消融结果，\n整理可复现研究材料并完成学位论文。',210,545,1000,85,27,{color:green});
foot(s,'时间节点待开题与毕业安排确认。当前仍处于问题分析阶段，未启动新增训练。');
notes(s,'【讲稿】下一步不是继续增加模块名称，而是先交付一个能实施、能比较的方案。先查问题和代码接口，再做短训练检查，之后才正式训练。第一主线形成证据后，再推进第二点及论文整理。\n【验收】短训练需要确认损失和梯度正常、预测形状和实例对应正确。正式验证需要同时看整体和边界表现。所有结论以实际结果为准，不预先承诺精度提升。');

const qa = [
'【此前问题：论文究竟做什么？】研究产物是改进的水下实例分割模型。输入单张图片，输出每个物体的类别和像素区域。主线是Mask2Former上的细节增强与边界约束，第二点考虑水下特征适配。页面上的“拟”表示计划，不能讲成已完成成果。',
'【任务区分】目标检测通常画矩形框；语义分割区分每个像素的类别；实例分割还要把同类的不同物体分别分开。本课题做实例分割。\n【Mask2Former是通用模型吗？】是，可用于语义、实例、全景分割。我们采用其中的实例分割任务。\n【UIIS是什么？】WaterMask论文发布的水下实例分割数据集，本项目沿用官方3937张训练、691张测试、7类的规则。',
'【方法来源】WaterMask的DSGAT、MFRM和边界学习设计是相关工作的技术来源，必须引用。正文改用功能描述，不代表这些既有模块变成原创。最终创新性需要具体机制、与已有工作差异及实验共同证明。\n【和Mask R-CNN的关系】Mask R-CNN采用候选区域上的分割，Mask2Former以查询组织实例预测。WaterMask原有模块不能直接假定可插入，两者的实例表示和监督对应需要核对。\n【DINOv2／DiveSeg是否利用大语言模型？】这里是预训练视觉特征提取与实例分割路线，不是调用聊天模型阅读图片。',
'【为什么示例不好，但AP可能更高？】少量图片与全量指标反映的范围不同。固定案例用于发现问题，性能结论应结合691张全量评价。\n【置信度与IoU】置信度是模型对预测的评分；IoU是预测区域与人工标注的交集面积除以并集面积。显示阈值0.5筛选置信度，与IoU≥0.5不是一回事。\n【误差报告口径】按置信度降序进行同类别、IoU≥0.5的一对一匹配。普通标注未匹配算漏检，crowd标注不算漏检；同类crowd对未匹配预测的覆盖比例IoA≥0.5时忽略。\n【未匹配预测的分组】依次检查同类IoU≥0.5（重复）、异类IoU≥0.5（类别混淆）、同类IoU≥0.1（定位不足）、仅异类IoU≥0.1（类别与定位混合），其余低重合背景。优先级意味着一个预测只进入一个组。定位不足不等于边界误差，重复也不能直接证明实例拆分。\n【TIDE是什么？】是错误归因分析工具。本项目阈值统计不是TIDE，也不是各类错误导致AP下降多少的分解。',
'【ResNet-50是什么？】简称R50，是提取图像特征的骨干网络。它与负责分割预测的其他模块一起组成模型，不等同于完整Mask2Former。\n【具体改在哪一步？】第一点拟作用于像素解码器产生的掩码特征及训练监督，输出仍为实例掩码。DINOv2主要改变前端特征提取及适配路线，二者位置和作用不同。\n【代码为什么少？】本地项目主要是配置、训练入口和实验管理，核心网络实现来自所依赖的框架仓库。不能用本项目文件数判断完整模型大小。',
'【边界标签怎么来？】候选方案从已有实例掩码提取轮廓，不需要生成新照片。需要确定边界宽度、分辨率和每个实例的对应关系。\n【BLL是不是直接加上即可？】不是。损失函数依赖预测和标签的组织方式，需核对实例匹配、尺度与归一化。当前边界分支是待核对的具体方案，不是已验证的BLL迁移实现。\n【训练与推理的区别】辅助监督只在训练时使用标签。拟议分支若仅用于辅助训练，可考虑推理时移除，但必须在实现后验证推理路径和实际成本，不能提前保证没有开销。',
'【为什么不能全加再看分数？】全加后无法分辨哪项有贡献。应对比基线、仅细节增强、仅边界约束及联合方案，这叫消融实验。\n【两个模块就是两个研究点吗？】不自动成立。本版细节增强和边界约束仍属于第一主线，需要共同回答明确问题。第二点需有独立研究问题。\n【收益有多大、能否保证毕业？】目前没有改进模型结果，不能给出可信涨点概率或保证毕业。学校要求、创新性论证和实验质量共同影响结果。页数和模块数量不能替代证据。',
'【AP、AP50、AP75】AP评价不同置信度下精确率与召回率的关系；COCO式mask AP还在IoU 0.50至0.95、步长0.05的阈值上取平均。AP50只在IoU=0.50下评价，AP75在0.75下评价。32.6表示按百分制展示的AP，不是像素正确率。\n【COCO是什么？】它是通用视觉数据集，也常指其标注格式和评价规范。本项目用UIIS训练测试，使用COCO预训练权重和COCO式评价，不代表测试图片来自COCO。\n【36轮与3×是否相同？】36轮表示遍历训练数据36次。3×通常表示相对基础训练日程的延长标记，在常见设置中与36轮有关，但必须检查具体配置，不能仅凭名称认定公平。\n【已经改了哪些条件？】从通用任务迁移到UIIS需要类别和数据配置等调整；目前未加入自己的网络改进。基线为R50、COCO预训练、36轮70884步，已有AP32.6/AP50 47.6/AP75 35.3。\n【与WaterMask差多少？】已记录的论文R50参考AP为26.4，与本次32.6相差6.2个AP点。未同环境复现，预训练、优化器、增强等未完全对齐，只能作参考对比，不能声称是自己方法带来的提升。\n【100步和早期4.7】100步仅用于检查流程，不能推算WaterMask的100步表现，也不能与完整训练结果比较。早期4.7的准确指标需查当时日志，不能仅凭聊天把它认定为最终AP。\n【重复图片为什么保留？】按用户要求沿用官方划分，所有对照保持相同规则，并披露已知重复，不利用它宣称方法收益。\n【权重规则】现有报告固定使用最终第36轮权重，没有按测试分数选最优。未来选择规则需先确定，不能用测试集反复调参。',
'【DINOv2与DiveSeg分别做什么？】DINOv2提供预训练视觉特征，DiveSeg是一项利用这些特征并进行水下适配的已有研究。Mask2Former承担分割预测。第二点应研究自己的适配设计，而不是只换一个大模型。\n【第一点与第二点的区别】第一点侧重细节与边界，第二点侧重通用特征对水下外观的适应。必须固定相同DINOv2比较适配前后，才能隔离适配收益。\n【当前状态】第二点尚未完成结构和新颖性论证，正文以拟研究方向表述，不应在答辩中宣称已有创新模型。',
'【已完成与下一步】已完成环境、数据审计、短训练、36轮正式训练、全量评价及阈值诊断。尚未完成边界专用评价、改进模块实现、正式消融及第二研究点。\n【训练时电脑能关吗？】远程训练正确以后台进程运行并保存日志后，本地电脑可以关闭；服务器必须持续运行。当前没有因修改PPT启动训练。\n【运行时间与费用】要根据实际单轮速度和验证开销估算，不能沿用早期不准确估计。完整训练并不等于已经完成所有论文实验。\n【文件同步】本地生成源和README进行Git记录，PPT按当前仓库规则仅本地保存。训练服务器同步与GitHub推送是两件事。日常步骤更新README，PPT只在阶段节点或明确要求时更新。'
];
for(let i=0;i<ordered.length;i++) ordered[i].speakerNotes.text += '\n\n【历史提问与答辩备答】\n'+qa[i];

p.slides.keep(p.slides.items.map((s,i)=>ordered.includes(s)?i:-1).filter(i=>i>=0));p.slides.reorder(ordered.map(s=>'sl/'+s.id));
const candidate=B+'/candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<ordered.length;i++)await fs.writeFile(B+`/slide-${i+1}.png`,new Uint8Array(await(await p.export({slide:ordered[i],format:'png',scale:1})).arrayBuffer()));
const {finalizePresentation}=await import(pathToFileURL(SK+'/container_tools/artifact_tool_utils.mjs'));
try{console.log(await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:F,pythonExecutable:R+'/python/python.exe',integrityValidatorPath:SK+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SK+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--expected-slide-count','10','--validate-heading-fit',...tables.flatMap(s=>['--require-native-table-slide',String(ordered.indexOf(s)+1)])],requiredNativeTableOwnerSlides:tables.map(s=>ordered.indexOf(s)+1),fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:T,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(T)).digest('hex')},verifyArtifactToolImport:true,receiptPath:B+'/validation.json'}));}
catch(e){if(e.code!=='EPERM'||e.syscall!=='link')throw e;await fs.copyFile(candidate,F,1);const hash=crypto.createHash('sha256').update(await fs.readFile(F)).digest('hex');await fs.writeFile(B+'/copy-receipt.json',JSON.stringify({finalPath:F,sha256:hash,limitation:'finalizer reached final hard-link publication after validation, Windows EPERM; delivered using exclusive copy',slideCount:10},null,2));console.log('COPIED_AFTER_LINK_EPERM',F);}
