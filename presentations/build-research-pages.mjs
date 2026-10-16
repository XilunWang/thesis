import fs from 'node:fs/promises';import path from 'node:path';import {pathToFileURL} from 'node:url';
const ROOT=path.resolve(import.meta.dirname,'..'),R='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies',SK='C:/Users/Lenovo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(R+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
const B=ROOT+'/presentations/.build/research-pages',F=ROOT+'/presentations/output/开题报告_研究内容与框架_20261017_v2.pptx';
process.env.RUNTIME_NODE_MODULES=R+'/node/node_modules'; const imported=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Lenovo/Desktop/左怀琪-专业学位硕士开题PPT.pptx')); const proto=imported.toProto(); proto.slides[10].elements=proto.slides[10].elements.filter(e=>!(e.type===7 && e.bbox.yEmu>100*9525)); const p=Presentation.load(proto);const green='#19652B',ink='#333333';
function tx(s,t,x,y,w,h,size=25,bold=false,color=ink){let a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:'微软雅黑',fontSize:size,color,bold,verticalAlignment:'middle',wrap:true,insets:{left:0,right:0,top:0,bottom:0}};return a;}
let s=p.slides.items[9];
tx(s,'基于细节增强与边界学习的水下实例分割方法（拟）',205,125,1020,50,29,true,green);
const rows=[['多尺度细节增强','拟在Mask2Former像素解码器的特征融合环节适配DSGAT，增强局部细节表达，改善细长结构信息不足的问题。'],['实例掩码细化','拟将MFRM适配至实例掩码预测阶段，结合多尺度空间特征与实例查询信息，细化每个目标的前景区域和轮廓。'],['边界监督与联合优化','从已有实例标注生成边界标签，适配BLL边界学习损失，与原有分类、掩码损失共同优化，研究边界监督的作用。']];
rows.forEach((r,i)=>{tx(s,`${i+1}. ${r[0]}`,210,195+i*130,1000,38,28,true,green);tx(s,r[1],210,238+i*130,1000,75,25);});
tx(s,'验证方案：沿用UIIS数据规则，对比基线、单项适配与组合方案，\n同时评价实例AP、边界质量及计算开销。',210,604,1000,70,24);
s.speakerNotes.text='这页说明第一研究点，三个设计属于同一条主线。前面增强细节，后面细化每个实例的区域，再通过边界标签指导训练。DSGAT、MFRM和BLL来源于WaterMask，使用时应引用原文，不能把已有模块改名作为原创。本课题拟研究它们与Mask2Former的接口适配及实际作用。当前没有实现这些改动，也没有验证提升。MFRM依赖空间特征，不能直接塞入Query向量；BLL需明确预测边界、实例匹配、边界宽度、分辨率及权重。逐项实验须遵守结构依赖，例如边界损失需先有相应边界输出。资料：https://openaccess.thecvf.com/content/ICCV2023/html/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.html';
s=p.slides.items[10];
function box(t,x,y,w,h,added=false){let a=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:added?'#E8F2E9':'#F5F6F6',line:{fill:added?green:'#7A8580',width:1.5}});a.text=t;a.text.style={typeface:'微软雅黑',fontSize:22,color:added?green:ink,bold:added,alignment:'center',verticalAlignment:'middle',wrap:true,insets:{left:8,right:8,top:5,bottom:5}};return a;}
function link(a,b,from='right',to='left',dash=false){s.shapes.connect(a,b,{kind:from==='right'&&to==='left'?'straight':'elbow',fromSide:from,toSide:to,line:{fill:dash?'#A36A25':'#64776B',width:2,style:dash?'dashed':'solid'},tail:{type:'triangle',width:'sm',length:'sm'}});}
tx(s,'拟议框架：特征端增强细节，预测端细化实例轮廓',205,125,1020,45,28,true,green);
const input=box('水下图像',210,205,125,80),back=box('ResNet-50\n骨干网络',370,205,150,80),pixel=box('像素解码器\nDSGAT适配',565,205,190,80,true),decoder=box('Transformer\n解码器',810,205,200,80),cls=box('类别预测',1060,205,145,80);
link(input,back);link(back,pixel);link(pixel,decoder);link(decoder,cls);
const spatial=box('多尺度空间特征\n与掩码特征',565,365,190,85),coarse=box('初始实例掩码\n及实例查询',810,365,200,85);
link(pixel,spatial,'bottom','top');link(decoder,coarse,'bottom','top');link(spatial,coarse);
const refine=box('MFRM适配\n前景与边界细化',810,510,200,90,true),out=box('实例掩码',1060,510,145,90);
link(coarse,refine,'bottom','top');link(spatial,refine,'bottom','left');link(refine,out);
const gt=box('实例标注\n生成边界标签',210,510,155,90),loss=box('联合训练目标\n原有损失 + BLL',430,510,250,90,true);
link(gt,loss,'right','left',true);link(refine,loss,'left','right',true);
tx(s,'匹配后的实例监督',433,607,270,30,18,false,'#A36A25');
tx(s,'绿色：拟适配部分    实线：前向信息流    虚线：训练监督',210,650,1000,30,20);
s.speakerNotes.text='按图从左到右讲：输入图像经R50提取特征，在像素解码器的融合环节研究DSGAT适配。像素解码器同时提供供Transformer使用的多尺度特征和用于掩码预测的像素特征。查询经过Transformer解码后，预测类别，并与掩码特征结合得到初始实例掩码。随后拟通过MFRM适配结构利用空间特征及实例信息细化前景和边界。图中把掩码向量与像素特征的结合概括在“初始实例掩码及实例查询”框中。绿色部分是拟改动，灰色部分保留基线功能。虚线只表示训练所用监督信息，标签不进入推理。原有分类与掩码损失分别监督相应输出，图中为清楚展示而合并表示。BLL拟监督与标注匹配后的实例边界预测。该图是概念架构，未规定DSGAT具体层数与MFRM融合算子，默认画为末端细化候选，没有宣称修改每层注意力或阻止实例粘连。Mask2Former原有逐层掩码注意力反馈为简洁未展开，后续需要核对是否仅作用于最终预测。边界预测在推理时是否用于融合需要实验确定。来源：WaterMask https://openaccess.thecvf.com/content/ICCV2023/html/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.html ；Mask2Former https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py';
await(await PresentationFile.exportPptx(p)).save(B+'/candidate.pptx');
for(let i=0;i<p.slides.items.length;i++)await fs.writeFile(B+`/slide-${i+1}.png`,new Uint8Array(await(await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer()));
console.log('RENDERED');
const {finalizePresentation}=await import(pathToFileURL(SK+'/container_tools/artifact_tool_utils.mjs'));
try{await finalizePresentation({workspaceDir:ROOT,candidatePath:B+'/candidate.pptx',finalPath:F,pythonExecutable:R+'/python/python.exe',integrityValidatorPath:SK+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SK+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--expected-slide-count','14'],verifyArtifactToolImport:true,receiptPath:B+'/validation-v2.json'});}catch(e){if(e.code!=='EPERM'||e.syscall!=='link')throw e;await fs.copyFile(B+'/candidate.pptx',F,1);console.log('COPY_AFTER_LINK_EPERM');}
console.log(F);
