process.on('uncaughtException',e=>{console.error(e.message);process.exit(1)});
import fs from 'node:fs/promises';import path from 'node:path';import crypto from 'node:crypto';import {pathToFileURL} from 'node:url';
const ROOT=path.resolve(import.meta.dirname,'..'),R='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies',SK='C:/Users/Lenovo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(R+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
process.env.RUNTIME_NODE_MODULES=R+'/node/node_modules';
const B=ROOT+'/presentations/.build/proposal-v4',T=ROOT+'/presentations/templates/proposal-template.pptx',F=ROOT+'/presentations/output/水下实例分割_硕士开题报告_20261016_v4.pptx';
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
tx(cover,'专业学位硕士开题报告',200,450,880,50,30,{color:green,align:'center'});
tx(cover,'姓名：待填写      专业：待填写\n导师：待填写      汇报日期：待填写',340,525,650,100,26);
notes(cover,'【建议讲稿】各位老师好，我的拟定研究题目是“面向水下场景的实例分割方法研究”。下面从研究背景与意义、研究现状、研究内容和研究计划四个方面汇报。\n【说明】题目暂定，身份及正式日期待补。第二研究点仍需论证。本版参照原模板的论述结构，不沿用其研究结论。');
let s=take(b1,'1.1 研究背景');
lines(s,[['水下环境感知的应用需求','海洋生态监测与水下智能巡检需要获取目标类别、数量及形态信息。实例分割能够同时识别目标类别与像素级区域，为水下场景的精细化分析提供基础。'],['水下实例分割的技术发展','实例分割由候选区域预测逐步拓展至Transformer统一建模。水下专用方法引入细节恢复与边界学习，进一步探索复杂环境中的目标表达。'],['复杂水下场景的现存挑战','水下成像中的色偏、低对比度及目标遮挡增加了前景与背景的区分难度。小目标信息不足、细长结构缺失及局部轮廓偏差，仍是本课题需要分析和验证的重点问题。']]);
notes(s,'【讲解】这一页按照原模板“需求—技术演进—挑战”的逻辑展开。首先说明为什么需要分割，再说明技术发展到哪里，最后引出问题。实例分割就是把每个目标分别圈出，例如三只海星分别对应三个区域。Transformer可理解为一种能组织图像信息的模型结构，本页无需展开计算细节。\n【注意】这些问题是研究关注点，不能说所有现有模型都存在同样程度的不足。',[refs.mask,refs.m2f,refs.water]);
s=take(b1,'1.2 研究意义');
lines(s,[['理论意义','围绕水下图像中目标细节、轮廓与视觉特征的关系，分析通用分割模型的误差表现，探索特征适配与边界监督的作用机制，为水下实例分割方法研究提供实验依据。'],['技术意义','针对目标细节缺失与局部区域预测偏差，研究适配Mask2Former的特征细化和监督方式，并探索预训练视觉特征的水下适应方法，评估精度提升与计算开销之间的关系。'],['应用意义','研究成果可为水下生物数量统计、目标形态分析和巡检图像处理提供技术支持。通过提高目标区域识别质量，为后续环境感知与辅助分析提供更可靠的视觉信息。']]);
notes(s,'【讲解】参照模板分成理论、技术、应用三个层面。理论层面是弄清楚方法为什么有效，不代表已经建立了新的理论体系。技术层面是研究具体怎么改模型。应用层面是分割结果能支持什么任务，尚未完成实际系统部署。\n【建议讲法】本研究希望通过对误差和改进效果的分析，说明细节与边界设计在水下场景中的作用，并形成能够复查的实验结论。');
s=take(b2,'2.1 通用实例分割方法');
lines(s,[['基于候选区域的方法','以Mask R-CNN为代表，通过候选区域提取、目标分类与掩码预测完成实例分割，将目标定位与区域预测结合，为后续研究建立了重要的两阶段框架。'],['基于Transformer的方法','Mask2Former通过掩码注意力组织图像特征，以查询驱动的方式预测目标区域，并在统一框架下支持语义、实例和全景分割，为不同场景的分割研究提供基础模型。'],['细粒度质量评价方向','整体区域重合度能够反映分割性能，但对局部轮廓变化的描述仍需补充。Boundary IoU等评价方法关注边界区域，为分析细节恢复和轮廓改善提供了更有针对性的依据。']]);
foot(s,'代表工作：Mask R-CNN（ICCV 2017）、Mask2Former（CVPR 2022）、Boundary IoU（CVPR 2021）。');
notes(s,'【讲解】参照原稿按技术路线归纳研究现状，每段说明代表方法及其主要思路。Mask R-CNN可以粗略理解为先找位置再分区域。Mask2Former用查询来组织每个目标的预测，本课题已经用它跑通基线。Boundary IoU是评价工具，不是新的分割模型。\n【术语】掩码就是标记物体所在像素的区域。查询可先理解为模型用于组织候选目标的一组内部表示。\n【注意】没有在本项目计算Boundary IoU，正文是在介绍已有评价方向。',[refs.mask,refs.m2f,refs.boundary]);
s=take(b2,'2.2 水下实例分割方法');
lines(s,[['细节恢复与边界学习方向','WaterMask构建UIIS数据集，并通过DSGAT、MFRM及边界学习设计增强目标细节与分割区域表达。该研究为水下图像的细粒度分割提供了专门的方法参考。'],['基础模型适配方向','USIS-SAM围绕水下显著实例分割开展SAM适配研究。DiveSeg利用DINOv2视觉特征，并结合水下适配设计进行实例分割，体现了预训练模型在水下任务中的应用探索。'],['本课题的研究切入点','现有研究分别从细节建模与预训练表征等角度开展改进。本课题拟以Mask2Former为基础，研究相关设计的适配方式，并通过统一条件下的实验分析其有效性与适用范围。']]);
foot(s,'代表工作：WaterMask（2023）、USIS-SAM（2024）、DiveSeg（arXiv:2511.08334）。');
notes(s,'【讲解】这一页对应原模板对领域方法的分类归纳。WaterMask与我们目前的主线最相关，可以借鉴它处理细节和边界的思路。USIS-SAM面向显著实例，任务口径与多类别实例分割不同，不能直接比较数字。DiveSeg提供DINOv2路线的参考。\n【术语】预训练特征是模型先在大量其他图像中学习到的内部表示。适配就是让这些表示更适合水下任务。\n【注意】这不是完整文献综述，也没有证明本文方案已具备新颖性。模块名称和方法必须引用原工作。',[refs.water,refs.usis,refs.dino,refs.dive]);
s=take(b3,'3.1 研究动机');
lines(s,[['目标细节与局部轮廓表达不足','前期基线案例中出现细长结构分割不完整和目标区域偏差。此类现象提示模型仍存在细粒度表达的改进空间，需要进一步区分目标漏检与局部轮廓误差，确定针对性的研究目标。'],['已有设计与分割框架存在结构差异','WaterMask与Mask2Former的实例特征组织和预测流程不同，直接迁移模块难以保证接口一致及监督有效。需要结合查询式分割结构，研究特征细化与边界学习的适配方式。'],['预训练特征的水下适应能力有待验证','水下图像的颜色、对比度和目标外观与通用图像存在差异。拟进一步研究预训练特征的适配效果，区分特征提取器变化和适配设计本身带来的收益。']]);
notes(s,'【讲解】参照原稿“问题名称＋问题表现与影响”的写法。这三点是我们的研究动机，不是已解决的问题。第一点来自已有预测案例，但还需要全量和边界指标支持。第二点解释为什么不能简单复制WaterMask代码：两种模型内部组织目标的方式不同。第三点为候选第二研究点提供问题背景。\n【注意】定位错误不全是边界误差。不能仅凭图片断言Mask2Former普遍边界差，也不能提前断言DINOv2一定更好。',['reports/uiis-baseline-36e-evaluation/report.md','reports/uiis-error-analysis/report.md',refs.water,refs.m2f,refs.dive]);
s=take(b3,'3.2 研究内容一：细节与边界适配');
lines(s,[['目标细节信息的适配与细化','以Mask2Former-R50为基础，借鉴DSGAT与MFRM的细节恢复和特征细化思路，研究与掩码特征兼容的适配方式，增强细长结构及局部目标信息的表达。'],['面向实例轮廓的监督设计','参考WaterMask的边界学习损失，结合Mask2Former的实例预测与匹配过程，研究边界监督的构建及其与原有分割目标的结合方式，约束局部轮廓偏差。'],['单项设计与组合效果验证','在统一训练条件下设置原始基线、单项适配与组合适配实验，结合整体分割指标、尺寸分组与边界评价，分析各项设计的作用及其计算代价。']]);
foot(s,'拟研究方案，具体接口与监督形式尚待源码核对和实验确定。');
notes(s,'【讲解】这页从研究动机推进到具体准备研究的内容。先研究怎样补充目标细节，再研究训练时怎样关注轮廓，最后分别验证。DSGAT和MFRM来自WaterMask，BLL是边界学习损失，可以理解为训练时对边缘偏差施加额外约束。\n【实施顺序】先核对源码和接口，从一项开始，不同时堆入全部设计。消融实验指分别加入或移除某项设计，以确认它的作用。\n【注意】这里描述拟实现内容，未产生自有改进成绩。适配机制需具体论证，已有模块迁移本身不等同于创新。',[refs.water,refs.m2f]);
s=take(b3,'3.3 研究内容二：水下特征适配（拟）');
lines(s,[['预训练视觉特征的水下适应','参考DINOv2与DiveSeg的技术路线，分析预训练视觉特征在水下目标识别中的表现，围绕水下外观差异探索适配方式，形成与第一研究点相区分的研究内容。'],['特征适配与分割预测的衔接','拟研究轻量适配与特征融合机制，将预训练特征与Mask2Former分割预测过程衔接，考察目标细节与语义信息的利用方式，并控制新增参数及训练开销。'],['适配贡献与方法互补性分析','固定相同预训练骨干，比较无适配、基础适配与所研究方案的效果。在独立验证的基础上，进一步分析与第一研究点组合后的性能及开销变化。']]);
foot(s,'候选研究点，具体结构、创新性及实施范围仍需文献核对和可行性实验。');
notes(s,'【讲解】第二点关注模型是否能提取适合水下图像的信息。DINOv2是预训练视觉模型，DiveSeg是利用这类特征开展水下分割的已有方法。轻量适配希望用较少的新增参数调整特征。第一点关注细节和轮廓，第二点关注特征适应，两者必须分别验证。\n【比较原则】固定相同DINOv2规模和预训练条件，才能判断适配设计的作用。与R50比较只能看整套系统差异，不能隔离适配贡献。\n【注意】当前是候选方向，不声称已经确定新颖结构或完成实验。',[refs.dino,refs.dive]);
s=take(b3,'3.4 前期基础与实验方案');
table(s,[['工作环节','已有基础与拟开展工作'],['数据与基线','UIIS：3937张训练、691张测试、7类目标。\n完成Mask2Former-R50的36轮训练与全量评价'],['已有结果','实例AP 32.6、AP50 47.6、AP75 35.3。\n完成阈值诊断与固定案例复核，尚未加入改进模块'],['对照与消融','统一数据规则、预训练及训练设置，开展单项与组合实验，\n对关键结果重复验证'],['综合评价','比较整体分割、尺寸分组及边界质量，\n记录显存与训练成本，结合案例分析适用范围']],[245,775],155,365,24);
tx(s,'基线流程已完成，下一步开展边界评价与首个模块适配。',210,560,1000,70,27,{bold:true,color:green});
foot(s,'沿用官方划分与既定重复图片处理规则。跨论文数值仅作参考，需披露实验条件差异。');
notes(s,'【讲解】已有工作证明实验流程能够运行。当前AP为32.6，它是综合分割评分，不是32.6%的像素正确。AP50与AP75对应不同区域重合要求，75更严格。R50指ResNet-50骨干，现有模型使用COCO预训练。\n【公平性】各项对照要使用一致数据规则，已知重复图片按用户要求保留。预训练、优化器与增强条件不同，会影响分数，因此不能把与WaterMask公布分数的差值当作创新。后续开发验证规则需统一确认，不能反复根据测试分数选择超参数。\n【当前状态】仍在P2，边界专用指标与模块适配未完成，未复现WaterMask。',['README.md','reports/uiis-baseline-36e-evaluation/metrics.json']);
s=take(b4,'4.1 研究规划');
table(s,[['阶段安排','主要研究任务','预期产出'],['问题分析与文献研究','梳理相关方法，补充基线问题评价，\n明确研究内容与技术路线','问题分析与方案设计'],['第一研究点实施','开展模块适配、模型训练及单项和组合验证，\n分析方法的有效性与局限','方法实现与消融结果'],['第二研究点论证','进行特征适配的可行性实验，\n确定独立贡献并开展对照研究','第二项研究与实验结果'],['论文撰写与结果整理','复核实验数据，绘制结果图表，\n完善论文结构与正文内容','论文初稿与研究材料'],['论文修改与答辩准备','完善论证与实验分析，修改论文，\n整理答辩材料并完成汇报准备','论文定稿与答辩材料']],[250,500,270],145,420,23);
foot(s,'具体时间根据学校开题与毕业安排确定。研究实施与论文撰写同步推进。');
notes(s,'【讲解】参照模板按阶段说明工作，不加入未经确认的毕业月份。先阅读文献和查清问题，再完成第一研究点；第二点先做可行性论证，之后进入正式实验，最后整理论文与答辩。\n【预期成果】形成有实验支持的方法设计、可复现配置与日志、完整对照图表和学位论文。\n【结语】目前已经完成基线训练与评价，后续将围绕确定的问题开展适配验证。以上是我的开题汇报，恳请各位老师批评指正。');
p.slides.keep(p.slides.items.map((s,i)=>ordered.includes(s)?i:-1).filter(i=>i>=0));p.slides.reorder(ordered.map(s=>'sl/'+s.id));
const candidate=B+'/candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<ordered.length;i++)await fs.writeFile(B+`/slide-${i+1}.png`,new Uint8Array(await(await p.export({slide:ordered[i],format:'png',scale:1})).arrayBuffer()));
const {finalizePresentation}=await import(pathToFileURL(SK+'/container_tools/artifact_tool_utils.mjs'));
try{console.log(await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:F,pythonExecutable:R+'/python/python.exe',integrityValidatorPath:SK+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SK+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--expected-slide-count','10','--validate-heading-fit',...tables.flatMap(s=>['--require-native-table-slide',String(ordered.indexOf(s)+1)])],requiredNativeTableOwnerSlides:tables.map(s=>ordered.indexOf(s)+1),fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:T,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(T)).digest('hex')},verifyArtifactToolImport:true,receiptPath:B+'/validation.json'}));}
catch(e){if(e.code!=='EPERM'||e.syscall!=='link')throw e;await fs.copyFile(candidate,F,1);const hash=crypto.createHash('sha256').update(await fs.readFile(F)).digest('hex');await fs.writeFile(B+'/copy-receipt.json',JSON.stringify({finalPath:F,sha256:hash,limitation:'finalizer reached final hard-link publication after validation, Windows EPERM; delivered using exclusive copy',slideCount:10},null,2));console.log('COPIED_AFTER_LINK_EPERM',F);}
