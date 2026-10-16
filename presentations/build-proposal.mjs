process.on('uncaughtException',e=>{console.error(e.message);process.exit(1)});
import fs from 'node:fs/promises';import path from 'node:path';import crypto from 'node:crypto';import {pathToFileURL} from 'node:url';
const ROOT=path.resolve(import.meta.dirname,'..'),R='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies',SK='C:/Users/Lenovo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(R+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
process.env.RUNTIME_NODE_MODULES=R+'/node/node_modules';
const B=ROOT+'/presentations/.build/proposal-v5',T=ROOT+'/presentations/templates/proposal-template.pptx',F=ROOT+'/presentations/output/水下实例分割_硕士开题报告_20261016_v5.pptx';
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
tx(cover,'研究主线：Mask2Former的细节与边界适配',180,450,920,50,28,{color:green,align:'center'});
tx(cover,'姓名：待填写      专业：待填写\n导师：待填写      汇报日期：待填写',340,525,650,100,26);
notes(cover,'【开场】本课题研究水下实例分割，已完成Mask2Former基线训练。接下来拟研究如何改善目标细节与轮廓。今天重点说明现有问题、准备改动的位置以及验证方案。\n【状态】这是方案讨论稿。第一点结构尚未实现，第二点待论证，题目暂定。');
let s=take(b1,'1.1 研究任务与目标');
lines(s,[['研究对象：水下图像中的独立目标','输入单张水下图像，输出每个目标的类别与像素区域。以UIIS数据集的7类目标为对象，研究不同实例的识别与分割。'],['研究问题：目标细节与轮廓不完整','水下图像的低对比度、遮挡及目标尺度差异增加分割难度。结合已有基线案例，重点分析细长结构缺失与局部区域偏差。'],['研究目标：形成可验证的模型改进','以Mask2Former-R50为基线，研究细节特征细化与边界监督的适配方法，评价目标轮廓质量、整体分割性能及新增计算开销。']]);
foot(s,'应用意义：为水下生物计数、形态分析与巡检图像处理提供目标区域信息。');
notes(s,'【讲稿】这篇论文要做的东西是一个改进后的水下实例分割模型。输入一张水下图片，输出每只动物各自的区域。研究不涉及重新制作数据集，也不是使用大语言模型分析图片。当前主线是基于已训练的Mask2Former，研究目标细节和轮廓的改进。\n【解释】实例分割要把每个目标分开，例如三只海星输出三个区域。细节与轮廓是否是主要问题，需要后续全量分析支持。',[refs.water]);
s=take(b2,'2.1 相关方法与本课题定位');
table(s,[['方法','已有工作','本课题中的角色'],['Mask2Former','通用图像分割框架，支持实例级掩码预测','已训练的基础模型，作为主要对照'],['WaterMask','面向水下图像的细节恢复、特征细化与边界学习','提供候选设计思路，需重新适配接口'],['DINOv2 / DiveSeg','预训练视觉特征及水下任务适配','第二研究点的候选参考，尚未定案']],[240,430,350],155,320,24);
tx(s,'本课题的核心工作：在Mask2Former中设计并验证水下适配，\n明确改动位置、训练约束和性能变化的来源。',210,520,1000,95,27,{color:green});
foot(s,'已有模块需引用来源，单纯组合模块不构成已经成立的创新。');
notes(s,'【讲稿】Mask2Former是起点，WaterMask是借鉴对象。我们不是把两个模型直接串在一起，而是研究WaterMask中的细节和边界思想如何适应Mask2Former。DINOv2和DiveSeg暂时只作为第二点候选，不影响第一主线。\n【解释】现有论文给出的是思路和实现参考，自有贡献必须落在具体设计及其作用证据上。',[refs.m2f,refs.water,refs.dino,refs.dive]);
s=take(b3,'3.1 问题证据：基线预测案例');
await img(s,'reports/uiis-baseline-36e-evaluation/examples/1.jpg',205,140,490,280);
await img(s,'reports/uiis-baseline-36e-evaluation/examples/2.jpg',730,140,490,280);
tx(s,'目标已识别，细长结构未完整保留',205,430,490,45,25,{bold:true,color:green});
tx(s,'部分小目标未显示预测结果',730,430,490,45,25,{bold:true,color:green});
tx(s,'研究关注：在保持目标识别能力的同时，改善区域完整性与轮廓质量。\n下一步需量化边界表现，并将漏检、类别错误与区域偏差分别分析。',210,505,1000,110,26);
foot(s,'每组左侧为标注、右侧为预测，显示阈值0.5。案例用于提出问题，不代表全量结论。');
notes(s,'【讲稿】这页让老师看到我们为什么准备研究细节。海星被识别出来了，但细长部分没有完整分出。右侧小目标未显示，也可能与置信度阈值有关。因此两类问题不能混为一谈。第一研究点优先关注区域细节与轮廓，漏检原因需要单独诊断。\n【解释】阈值0.5表示只展示得分达到0.5的结果。没有显示不等于模型完全没有输出。边界专用评价还没完成，不能先断言边界是所有错误的主要来源。',['reports/uiis-baseline-36e-evaluation/report.md','reports/uiis-error-analysis/report.md']);
s=take(b3,'3.2 总体技术路线与拟改动位置');
table(s,[['模型环节','原始流程','拟研究改动'],['输入','UIIS水下图像','保持官方数据及标注规则'],['特征提取','ResNet-50与像素解码器','首先保留原结构，建立一致对照'],['掩码特征','生成用于实例区域预测的特征','候选A：细节特征细化'],['分割预测','输出目标类别与实例掩码','保留原实例预测流程'],['训练监督','原有分类与掩码损失','候选B：实例边界辅助监督']],[215,420,385],145,385,23);
tx(s,'输出保持一致：目标类别、置信度与实例区域。\n研究变化集中于掩码特征及其训练监督。',210,555,1000,85,26,{color:green});
foot(s,'概念方案，尚未实现。实际插入位置与实例匹配方式以源码核对结果为准。');
notes(s,'【讲稿】这一页回答模型到底改哪里。输入仍是水下图片，输出仍是每个目标的类别和区域。首先保持骨干和原预测过程，拟在掩码特征附近研究两个候选改动：一个补充细节，一个在训练时关注边界。\n【解释】特征是模型内部的图像信息。损失是训练时衡量预测与标注差距的规则。表中自上而下是处理环节，最后一行是训练时的约束，不是推理输出之后的处理模块。\n【注意】这是本轮具体化的讨论方案，不代表已确定接口。WaterMask模块需核对，不能声称即插即用。',[refs.m2f,refs.water]);
s=take(b3,'3.3 首个验证方案：边界辅助监督');
lines(s,[['拟验证的假设','在原有实例分割训练中增加轮廓约束，能否改善细长目标的局部边界，同时保持整体分割性能？'],['候选实现','从人工实例掩码提取边界作为监督信号。在掩码特征端设置候选边界预测分支，并通过辅助损失参与训练，主分割输出仍沿用原流程。'],['关键实现与验收','核对目标匹配、边界尺度和损失权重，先完成短训练与梯度检查。正式比较原基线与边界辅助方案，联合检查实例AP、边界指标及固定案例。']]);
foot(s,'首个可检验候选，不等同于直接移植BLL。需先确认边界问题证据及源码可行性。');
notes(s,'【讲稿】为了把方案落地，先提出一个足够具体的实验：让模型训练时不仅学习目标内部区域，也学习轮廓位置。可以考虑从现有人工标注生成边界，不必重新人工标注，再增加一个训练用的边界辅助分支。\n【解释】辅助分支可以理解为训练时的额外练习。它是否能改善主分割结果要靠实验。边界的宽度、预测分辨率和损失权重都要明确，参数选择应使用合适的开发验证规则。\n【重要】这是待核对的候选结构，不是WaterMask原代码的复刻，也不是已证明的新颖方法。若源码或问题分析不支持，应调整方案并更新记录，不强行实施。',[refs.water,refs.m2f,refs.boundary]);
s=take(b3,'3.4 细节适配与组合实验');
lines(s,[['细节特征细化候选','参考DSGAT与MFRM的处理思路，选择一种与掩码特征兼容的细化设计，研究局部细节信息对目标区域预测的影响。'],['区分两类设计的作用','边界监督改变训练约束，细节细化改变特征表达。分别比较两项候选与原始基线，避免一次加入全部模块后无法判断性能变化来源。'],['检验是否值得组合','只有单项实验具有证据支持时，再比较组合方案的整体分割、边界质量和开销。对无稳定收益的设计分析原因，不以模块数量代替研究贡献。']]);
foot(s,'细节细化与边界监督属于同一主研究线中的候选设计，不自动视为两个独立研究点。');
notes(s,'【讲稿】边界辅助实验是首个候选，后续再选择一种细节细化方式。前者主要改变训练要求，后者改变模型内部处理信息的方式。要分别跑原始模型、边界候选、细节候选，最后才看组合。\n【解释】DSGAT和MFRM来自WaterMask，不能在尚未理解接口时直接全部照搬。单项和组合的比较就是消融实验。\n【注意】这仍是第一主线内部的设计，不应为了满足两点要求把两个小模块直接包装成两个独立研究点。',[refs.water]);
s=take(b3,'3.5 前期结果与正式对照设计');
table(s,[['实验或评价','设置与状态'],['基线：已完成','Mask2Former-R50，COCO预训练，UIIS 36轮。\n实例AP 32.6、AP50 47.6、AP75 35.3'],['边界候选：未实施','固定原训练条件，增加经核对的边界辅助监督'],['细节候选：未实施','固定原训练条件，加入一种细节特征细化设计'],['组合实验：未实施','基于单项结果判断是否组合，验证互补性'],['统一评价','实例AP、尺寸分组、边界质量、固定案例与计算开销']],[280,740],150,390,23);
foot(s,'UIIS训练3937张、测试691张、7类。保留既定划分与重复图片规则，关键实验重复验证。');
notes(s,'【讲稿】这一页把已完成和未来实验明确分开。目前只有基线完成，AP32.6不是本课题改进后的成绩。以后每一项改动都与这个起点在一致规则下比较，同时看边界和整体性能，防止只改善一个指标却损害其他方面。\n【解释】AP综合不同匹配严格程度的表现，AP50和AP75分别要求重合度至少为0.5和0.75。32.6不是像素准确率。\n【公平性】预训练、优化器与增强等条件需要控制。WaterMask公布分数只能作参考，不能据其差值宣称本方法贡献。尚未完成开发验证方案，不应使用测试集反复选参数。',['README.md','reports/uiis-baseline-36e-evaluation/metrics.json']);
s=take(b3,'3.6 第二研究点及当前方案边界');
lines(s,[['候选问题：预训练特征的水下适应','参考DINOv2与DiveSeg，考虑针对水下外观差异进行特征适配，并分析其与细节及边界设计的区别。'],['立项依据：独立问题与可控对照','需明确具体适配结构和已有工作的差异。实验应固定相同DINOv2骨干，比较适配前后表现，排除单纯更换模型规模带来的影响。'],['当前安排：完成论证后再启动','第二研究点尚未确定，不作为已完成创新。当前优先补齐第一主线的问题证据、实现接口与首个验证实验，再决定第二点的具体范围。']]);
foot(s,'完整开题方案仍需明确第二研究点的独立贡献，现有材料不能替代该论证。');
notes(s,'【讲稿】论文希望有两个研究点，但第二点目前确实没有具体定下来。DINOv2和DiveSeg只是候选参考，不能仅凭一个模型名称就说研究方案完整。需要明确独立问题、具体结构和公平对照。\n【说明】本页保留真实状态，避免让导师误以为已有两套确定的方法。第一主线可以继续细化，第二点还需要文献和可行性依据。',[refs.dino,refs.dive]);
s=take(b4,'4.1 研究计划与阶段验收');
table(s,[['阶段','工作内容','验收依据'],['问题与接口核对','补充边界评价，阅读两模型相关源码，\n确认首个候选的实施位置','问题证据与明确实现方案'],['首个候选验证','完成边界辅助候选，先短训练，\n再开展统一条件的正式对照','训练正确性与指标变化'],['第一主线完善','确定细节候选，完成单项与组合实验，\n分析效率、稳定性及失败案例','完整消融与方法结论'],['第二点与论文整理','论证第二点，确定后再实施，\n同步完善方法章节与实验材料','独立贡献与学位论文']],[230,475,315],150,365,23);
tx(s,'近期交付：一份可执行的适配方案及首个对照实验，\n以实验结果决定后续设计。',210,545,1000,85,27,{color:green});
foot(s,'时间节点待开题与毕业安排确认。当前仍处于问题分析阶段，未启动新增训练。');
notes(s,'【讲稿】下一步不是继续增加模块名称，而是先交付一个能实施、能比较的方案。先查问题和代码接口，再做短训练检查，之后才正式训练。第一主线形成证据后，再推进第二点及论文整理。\n【验收】短训练需要确认损失和梯度正常、预测形状和实例对应正确。正式验证需要同时看整体和边界表现。所有结论以实际结果为准，不预先承诺精度提升。');
p.slides.keep(p.slides.items.map((s,i)=>ordered.includes(s)?i:-1).filter(i=>i>=0));p.slides.reorder(ordered.map(s=>'sl/'+s.id));
const candidate=B+'/candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<ordered.length;i++)await fs.writeFile(B+`/slide-${i+1}.png`,new Uint8Array(await(await p.export({slide:ordered[i],format:'png',scale:1})).arrayBuffer()));
const {finalizePresentation}=await import(pathToFileURL(SK+'/container_tools/artifact_tool_utils.mjs'));
try{console.log(await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:F,pythonExecutable:R+'/python/python.exe',integrityValidatorPath:SK+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SK+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--expected-slide-count','10','--validate-heading-fit',...tables.flatMap(s=>['--require-native-table-slide',String(ordered.indexOf(s)+1)])],requiredNativeTableOwnerSlides:tables.map(s=>ordered.indexOf(s)+1),fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:T,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(T)).digest('hex')},verifyArtifactToolImport:true,receiptPath:B+'/validation.json'}));}
catch(e){if(e.code!=='EPERM'||e.syscall!=='link')throw e;await fs.copyFile(candidate,F,1);const hash=crypto.createHash('sha256').update(await fs.readFile(F)).digest('hex');await fs.writeFile(B+'/copy-receipt.json',JSON.stringify({finalPath:F,sha256:hash,limitation:'finalizer reached final hard-link publication after validation, Windows EPERM; delivered using exclusive copy',slideCount:10},null,2));console.log('COPIED_AFTER_LINK_EPERM',F);}
