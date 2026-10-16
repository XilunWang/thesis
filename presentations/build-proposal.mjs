process.on('uncaughtException',e=>{console.error(e.message);process.exit(1)});
import fs from 'node:fs/promises';import path from 'node:path';import crypto from 'node:crypto';import {pathToFileURL} from 'node:url';
const ROOT=path.resolve(import.meta.dirname,'..'),R='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies',SK='C:/Users/Lenovo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(R+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
process.env.RUNTIME_NODE_MODULES=R+'/node/node_modules';
const B=ROOT+'/presentations/.build/proposal-v3',T=ROOT+'/presentations/templates/proposal-template.pptx',F=ROOT+'/presentations/output/水下实例分割_硕士开题报告_20261016_v3.pptx';
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
notes(cover,'【建议讲稿】各位老师好，我的拟定研究题目是“面向水下场景的实例分割方法研究”。本次汇报包括研究背景与意义、相关研究、研究内容及研究计划。目前已经完成基础模型的训练和评价，后续准备围绕目标细节和水下特征开展方法研究。\n【说明】题目和第二研究点尚待进一步论证。姓名、专业、导师和日期待补充。');
let s=take(b1,'1. 研究背景与意义');
lines(s,[['研究背景','水下生态监测与智能巡检需要准确识别目标类别、区分目标实例，并获得目标的像素级区域。'],['主要挑战','水下图像存在色偏、低对比度及遮挡等问题，小目标和细长结构容易出现漏检与分割不完整。'],['研究意义','研究适用于水下场景的实例分割方法，为生物数量统计、目标形态分析及水下环境感知提供技术基础。']]);
notes(s,'【建议讲稿】实例分割既要识别物体是什么，也要区分每一个物体。比如图中有三只海星，就要分别得到三只海星的身体范围。水下光照和颜色会发生变化，小目标还容易与背景混在一起，所以现成模型仍可能漏掉目标或丢失身体细节。研究更准确的分割方法，可以为后续计数和形态分析提供基础。\n【术语解释】像素级区域指沿目标形状标出区域，比只画一个矩形框更细致。本课题暂以单张图像为研究对象，尚未开展机器人或视频部署。',[refs.water]);
s=take(b2,'2. 国内外研究现状');
table(s,[['代表工作','研究特点','本课题借鉴方向'],['Mask2Former\nCVPR 2022','采用掩码注意力，实现统一的图像分割框架','建立通用实例分割基线'],['WaterMask\nICCV 2023','面向水下图像的细节恢复、特征细化与边界学习','研究细节与边界设计的适配'],['DINOv2 / DiveSeg','利用预训练视觉表征，探索水下场景适配','研究水下特征适配方案']],[260,420,340],155,355,24);
tx(s,'研究切入点：面向水下图像特性，验证细节与边界适配、\n预训练特征适配对实例分割的作用。',210,555,1000,85,27,{color:green});
foot(s,'参考文献：[1] Mask2Former  [2] WaterMask  [3] DINOv2  [4] DiveSeg（完整来源见备注）');
notes(s,'【建议讲稿】相关研究主要有两条路线。第一条是通用分割模型，Mask2Former是本课题选择的基线。第二条是针对水下场景进行改进，WaterMask提供了细节恢复和边界学习思路。此外，DINOv2和DiveSeg为利用预训练特征提供了参考。我的工作需要在这些已有方法的基础上，明确适配机制，并通过实验说明具体作用。\n【术语解释】基线是后续比较的起点。特征是模型从图像中提取的内部信息。DINOv2提供图像特征，DiveSeg研究如何利用这类特征完成水下分割。\n【答辩提醒】现有模块不能直接当作自己的创新。此页是研究切入点，不声称已经证明文献空白或取得方法突破。',[refs.m2f,refs.water,refs.dino,refs.dive]);
s=take(b3,'3. 前期研究基础');
table(s,[['工作内容','完成情况'],['数据准备','UIIS官方划分：训练集3937张、测试集691张，共7类目标'],['基线训练','Mask2Former-R50，COCO预训练，完成36轮训练'],['全量评价','实例AP 32.6、AP50 47.6、AP75 35.3'],['误差诊断','完成全量阈值分析与固定案例复核，建立后续对照依据']],[255,765],155,350,24);
tx(s,'已完成基线训练与评价流程，具备开展方法适配实验的基础。',210,550,1000,90,27,{bold:true,color:green});
foot(s,'当前结果来自基线模型，尚未加入拟研究的改进方法。');
notes(s,'【建议讲稿】目前我已经完成数据准备、36轮训练和691张测试图像的评价。基础模型采用Mask2Former-R50，实例AP为32.6。在此基础上，我进一步查看了预测结果并进行了错误分析。现阶段已经有一个可以复现和比较的起点，但还没有加入自己的方法。\n【术语解释】R50指ResNet-50，是负责提取图像信息的网络。COCO是常用图像数据集，这里使用了在COCO上预训练的模型。AP综合多个匹配严格程度下的表现，AP50和AP75分别使用0.5和0.75的重合度要求，75更严格。32.6不是“32.6%的像素预测正确”。\n【答辩提醒】WaterMask公布分数与当前模型的训练条件不完全一致，差值不能算作本课题的贡献。',['reports/uiis-baseline-36e-evaluation/metrics.json','README.md']);
s=take(b3,'4. 基线问题分析');
await img(s,'reports/uiis-baseline-36e-evaluation/examples/1.jpg',205,140,490,280);
await img(s,'reports/uiis-baseline-36e-evaluation/examples/2.jpg',730,140,490,280);
tx(s,'细长结构分割不完整',205,430,490,45,26,{bold:true,color:green});
tx(s,'部分小目标未显示预测结果',730,430,490,45,26,{bold:true,color:green});
tx(s,'部分案例仍存在目标细节缺失与小目标识别不足。\n后续将结合全量统计和边界评价，区分漏检、类别错误与区域偏差，\n确定具有针对性的改进目标。',210,490,1000,145,26);
foot(s,'每组左侧为人工标注，右侧为预测。显示阈值0.5，案例不代表全部错误的成因。');
notes(s,'【建议讲稿】左边的例子中，模型找到了海星，但没有完整保留细长部分。右边部分较小目标没有出现在显示结果中。这提示我们需要关注细节与小目标，但几张图还不足以证明所有错误都来自边界。因此下一步要把目标没找到、类别判断错误、目标范围不准确区分开，再选择改进方案。\n【术语解释】显示阈值0.5表示只展示置信度达到0.5的预测。没有显示不一定表示模型完全没有输出，也可能是分数较低。边界评价用于检查轮廓附近的分割质量。\n【答辩提醒】现有阈值诊断不是各类错误对AP损失的归因，边界专用指标尚未计算。',['reports/uiis-baseline-36e-evaluation/report.md','reports/uiis-error-analysis/report.md']);
s=take(b3,'5. 研究内容一：细节与边界适配');
lines(s,[['研究目标','针对细长结构缺失和局部轮廓偏差，研究细节信息与边界监督在水下实例分割中的作用。'],['拟研究方法','以Mask2Former为基础，借鉴WaterMask的DSGAT、MFRM与边界学习损失，设计适配的特征细化与监督方式。'],['关键研究问题','分析两类模型的结构差异，明确模块接口与实例对应关系，通过单项及组合实验验证各项设计的贡献。']]);
foot(s,'拟研究方案：具体结构、插入位置及有效性仍需源码核对与实验验证。');
notes(s,'【建议讲稿】第一项研究关注目标是否分得完整、轮廓是否准确。我准备以现有Mask2Former为基础，借鉴WaterMask中的细节和边界处理思路。实际推进时先核对实现接口，再从一项改动开始，验证有效后再考虑组合。\n【术语解释】DSGAT是WaterMask中的细节恢复设计，MFRM用于特征细化。边界学习损失BLL可以理解为训练时增加对轮廓偏差的约束，不是一个独立的图像增强模型。监督指利用人工标注告诉模型哪些预测需要改正。\n【实现说明】WaterMask与Mask2Former组织实例特征的方式不同，因此不能直接复制后就保证可用，需要核对特征尺寸、实例对应及损失计算。\n【答辩提醒】目前没有实现，也没有改进成绩。模块来自已有论文，自有贡献必须由具体适配设计和实验论证。',[refs.water,refs.m2f]);
s=take(b3,'6. 研究内容二：水下特征适配（拟）');
lines(s,[['研究目标','针对通用视觉表征与水下图像外观差异，研究预训练特征对水下实例分割的适应能力。'],['候选技术路线','参考DINOv2与DiveSeg，探索预训练特征提取、轻量适配与分割预测的结合方式。'],['验证重点','固定相同预训练骨干，对比适配前后的精度与计算成本，并进一步检验与第一研究点的互补性。']]);
foot(s,'第二研究点仍为候选方案，具体结构与独立贡献有待进一步论证。');
notes(s,'【建议讲稿】第二项研究暂定关注水下图像的特征表达。第一项更关注细节和轮廓，第二项考虑模型如何从颜色偏、清晰度差的图像中提取有效信息。可参考DINOv2和DiveSeg，先固定一种预训练网络，再研究少量适配参数是否能改善水下分割。\n【术语解释】预训练表示模型已经在其他图像上学习过。轻量适配是尽量通过较少的新增或可训练参数，让这些信息更适合当前任务。骨干是负责提取图像特征的主要网络。\n【答辩提醒】这一点尚未最终确定，也未验证新颖性。对比时必须固定相同DINOv2，避免把更大网络或更多预训练数据的优势误认为适配方法的贡献。先独立验证第二点，再与第一点组合。',[refs.dino,refs.dive]);
s=take(b3,'7. 实验设计与评价方案');
lines(s,[['实验设置','沿用UIIS官方划分与既定数据规则，统一对照模型的训练与评价条件，明确预训练和数据增强设置。'],['对照与消融','设置原始基线、单项改进与组合改进，逐项验证作用，并对关键结果开展重复实验。'],['评价指标','采用实例AP、AP50、AP75及尺寸分组指标，补充边界质量与固定案例分析，同时记录训练成本和显存开销。']]);
foot(s,'跨论文指标仅作参考比较。严格方法对比需控制实验条件并披露差异。');
notes(s,'【建议讲稿】实验重点是让比较公平。首先，改动前后应使用一致的数据和训练规则。其次，不能一次加入很多设计后只看总分，而要分别测试每项改动，这就是消融实验。最后，不仅要看整体分数，也要看小目标、目标边界以及计算成本。\n【术语解释】消融实验可以理解为逐项添加或去掉一个设计，判断它是否有用。多次重复训练用于判断提升是否稳定。尺寸分组指标是分别看小、中、大目标的表现。\n【协议说明】已知重复图片按用户要求保留，所有对照遵循一致规则。后续开发验证规则仍需统一确认，不能反复用测试集挑选超参数。\n【WaterMask比较】可报告论文参考值，但若预训练、优化器和数据增强等不同，就不能把差值全部归因于方法。',['README.md',refs.boundary]);
s=take(b4,'8. 研究计划与阶段安排');
table(s,[['研究阶段','主要任务','阶段产出'],['问题分析与方案设计','补充边界评价，核对模块接口，确定首个适配方案','问题证据与技术方案'],['第一研究点验证','完成方法实现、正式训练及消融实验','方法说明与实验结果'],['第二研究点论证','开展文献核对和可行性实验，确定适配方案','独立研究内容及对照结果'],['论文撰写与答辩','复核实验材料，完善结果分析与论文正文','学位论文与答辩材料']],[245,465,310],150,390,24);
tx(s,'当前阶段：基线已完成，正在开展问题分析与方法设计。',210,575,1000,65,27,{bold:true,color:green});
foot(s,'具体时间节点根据学校开题、毕业安排确定，研究与论文写作同步推进。');
notes(s,'【建议讲稿】研究按四个阶段推进。第一阶段补充问题证据并确定首个改动。第二阶段完成第一研究点的实现与验证。第三阶段再对第二研究点进行正式论证，控制实验范围。最后统一整理结果、完成论文与答辩。目前处于问题分析阶段，基线已经完成。\n【说明】尚未获得正式开题和毕业时间，因此未编造具体月份。第二点先做可行性检查，再决定正式训练投入。原项目计划仍处于P2，本次制作PPT未启动任何训练。',['README.md 第2节']);
s=take(b4,'9. 预期成果');
lines(s,[['方法成果','围绕水下实例分割中的细节、轮廓及特征适配问题，形成具有明确机制与实验证据的改进方案。'],['实验成果','建立完整的对照与消融结果，分析不同目标尺寸、典型错误及计算开销，明确方法的适用范围与局限。'],['论文与研究材料','整理可复现代码、实验配置、训练日志和评价结果，完成学位论文及阶段汇报材料。']]);
foot(s,'预期成果以实际实验验证为依据。恳请各位老师批评指正。');
notes(s,'【建议讲稿】预期成果包括方法、实验和论文材料三个方面。方法上，希望围绕明确的问题形成有解释、有证据的改进。实验上，要有公平对照和逐项验证，同时说明不足。材料上，要保留配置、代码和日志，使结果可以复查。下一步最重要的是完成第一个适配方案并验证其效果。我的汇报到此结束，请各位老师批评指正。\n【答辩提醒】两项研究的创新性和有效性尚需证据支持，不预先承诺精度涨幅或论文发表。若某项改动无效，应保留负结果并调整假设。');
p.slides.keep(p.slides.items.map((s,i)=>ordered.includes(s)?i:-1).filter(i=>i>=0));p.slides.reorder(ordered.map(s=>'sl/'+s.id));
const candidate=B+'/candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<ordered.length;i++)await fs.writeFile(B+`/slide-${i+1}.png`,new Uint8Array(await(await p.export({slide:ordered[i],format:'png',scale:1})).arrayBuffer()));
const {finalizePresentation}=await import(pathToFileURL(SK+'/container_tools/artifact_tool_utils.mjs'));
try{console.log(await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:F,pythonExecutable:R+'/python/python.exe',integrityValidatorPath:SK+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SK+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--expected-slide-count','10','--validate-heading-fit',...tables.flatMap(s=>['--require-native-table-slide',String(ordered.indexOf(s)+1)])],requiredNativeTableOwnerSlides:tables.map(s=>ordered.indexOf(s)+1),fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:T,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(T)).digest('hex')},verifyArtifactToolImport:true,receiptPath:B+'/validation.json'}));}
catch(e){if(e.code!=='EPERM'||e.syscall!=='link')throw e;await fs.copyFile(candidate,F,1);const hash=crypto.createHash('sha256').update(await fs.readFile(F)).digest('hex');await fs.writeFile(B+'/copy-receipt.json',JSON.stringify({finalPath:F,sha256:hash,limitation:'finalizer reached final hard-link publication after validation, Windows EPERM; delivered using exclusive copy',slideCount:10},null,2));console.log('COPIED_AFTER_LINK_EPERM',F);}
