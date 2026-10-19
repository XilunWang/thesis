import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
const R='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies',SK='C:/Users/Lenovo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations';
process.env.RUNTIME_NODE_MODULES=R+'/node/node_modules';
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(R+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
const B=process.cwd()+'/presentations/.build/flow-readable',T=process.cwd()+'/presentations/output/开题报告_研究内容与框架_20261017_v2_研究现状更新.pptx';
await fs.mkdir(B,{recursive:true});await fs.copyFile(T,B+'/original.pptx');
const orig=await PresentationFile.importPptx(await FileBlob.load(T));
await fs.writeFile(B+'/before.png',new Uint8Array(await(await orig.export({slide:orig.slides.items[12],format:'png',scale:1})).arrayBuffer()));
const proto=orig.toProto();proto.slides[12].elements=proto.slides[12].elements.filter(e=>e.bbox.xEmu<180*9525||(e.bbox.yEmu<100*9525&&!(e.paragraphs||[]).length));
const p=Presentation.load(proto),s=p.slides.items[12];let id=0;
function shape(t,x,y,w,h,kind='text',size=22){let a=s.shapes.add({name:'flow-'+(++id),geometry:kind==='text'?'textbox':'rect',position:{left:x,top:y,width:w,height:h},fill:kind==='text'?'none':kind==='new'?'#E5F1E7':'#F1F3F3',line:{fill:kind==='text'?'none':kind==='new'?'#19652B':'#7A8580',width:kind==='text'?0:1.4}});a.text=t;a.text.style={typeface:'微软雅黑',fontSize:size,color:kind==='new'?'#19652B':'#333333',bold:kind==='new',alignment:kind==='text'?'left':'center',verticalAlignment:'middle',insets:{left:5,right:5,top:3,bottom:3}};return a;}
function arrow(a,b,from='right',to='left',dashed=false){let c=s.shapes.connect(a,b,{kind:(from==='right'&&to==='left')?'straight':'elbow',fromSide:from,toSide:to,line:{fill:dashed?'#AF7628':'#718078',width:1.7,style:dashed?'dashed':'solid'},tail:{type:'triangle',width:'sm',length:'sm'}});c.name='flow-'+(++id);return c;}
const heading=shape('3.2 模型框架与拟加入部分',200,38,735,48,'text',32);heading.text.style={typeface:'微软雅黑',fontSize:32,bold:true,color:'#19652B',verticalAlignment:'middle'};
shape('灰色：原有功能    绿色：拟适配部分    实线：信息流    虚线：仅训练监督',205,105,1030,32,'text',20);
let input=shape('水下图像',200,180,100,65,'old',21),bb=shape('ResNet-50\n提取图像特征',330,180,140,65,'old',20);
let px=shape('',500,155,210,135,'old');shape('像素解码器\n融合多尺度特征',510,164,190,58,'text',21);shape('① DSGAT细节增强',510,231,190,45,'new',19);
let tr=shape('Transformer解码器\n更新实例查询',760,155,210,65,'old',20),qu=shape('B  实例查询特征',1020,155,210,65,'old',21);
arrow(input,bb);arrow(bb,px);arrow(px,tr);arrow(tr,qu);
let feat=shape('A  空间与掩码特征',500,330,210,60,'old',20),pred=shape('掩码预测\n查询与掩码特征结合',760,330,210,60,'old',20),coarse=shape('C  初始实例掩码',1020,330,210,60,'old',21),cls=shape('类别预测',1020,250,210,45,'old',20);
arrow(px,feat,'bottom','top');arrow(feat,pred);arrow(qu,pred,'left','top');arrow(qu,cls,'bottom','top');arrow(pred,coarse);
shape('两路信息在掩码预测处汇合',205,329,270,65,'text',21);
shape('拟加入的细化路径（A、B、C对应上方输出）',205,407,1020,35,'text',21);
let inputs=shape('细化输入\nA 空间特征＋B 查询特征＋C 初始掩码',205,455,445,80,'old',20),ref=shape('② MFRM适配\n实例特征聚合与细化',700,455,230,80,'new',21),out=shape('细化后的实例掩码\n及边界预测',980,455,250,80,'new',21);
arrow(inputs,ref);arrow(ref,out);
shape('仅训练时：边界标签与边界预测共同计算损失',205,555,800,32,'text',21);
let gt=shape('人工实例标注',205,610,165,55,'old',20),edge=shape('提取边界标签\n按实例匹配监督',420,610,210,55,'new',20),loss=shape('③ BLL边界损失',980,610,250,55,'new',21);
arrow(gt,edge,'right','left',true);arrow(edge,loss,'right','left',true);arrow(out,loss,'bottom','top',true);
shape('原有分类与掩码损失保留；具体适配与融合待验证，原有掩码注意力反馈未展开。',205,680,1030,30,'text',17);
s.speakerNotes.text='读图说明：箭头代表信息依赖，不表示所有框按一条线逐一执行。灰色是基线已有功能，绿色是拟增加或适配的机制。先看上半部分：图像经ResNet-50提取特征，送入像素解码器融合不同尺度信息。DSGAT画在像素解码器内部，表示拟在融合环节适配，具体位置待核对，不是凭空输入的外部信息。像素解码器分两路：向右的多尺度特征供Transformer解码器更新实例查询B，向下提供空间与掩码特征A。B用于预测类别，同时经过掩码表示映射后与A中的掩码特征结合，产生初始实例掩码C。图中为简洁将映射归入掩码预测。原有Mask2Former走到C已能完成实例分割。A/B/C是输出的引用标记，不是执行顺序。\n再看细化路径：明确列出拟供细化模块使用的空间特征A、查询B、初始掩码C，避免交叉长线。MFRM适配需要研究如何把这些信息组织成每个实例的特征，并细化前景和边界；它不是只处理二值掩码的后处理。具体聚合方式及边界与前景的融合尚未确定，图为候选信息流。\n最后看下方虚线：人工标注生成边界标签，和匹配实例的边界预测共同用于计算BLL，不是把标签作为推理输入。原有分类和掩码损失仍然保留，损失通过反向传播指导训练。用新照片预测时无需人工标注或计算损失。\n拟期望：DSGAT改善特征细节保留；MFRM改善局部区域遗漏与轮廓偏差；BLL为边界学习提供明确约束。能否提高AP和边界质量需要统一条件下的消融验证。三个机制参考WaterMask，不能直接当作三个原创模块。原Mask2Former逐层掩码注意力反馈未画出以简化说明，不代表被移除。\n参考：WaterMask ICCV2023 https://openaccess.thecvf.com/content/ICCV2023/html/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.html ；Mask2Former CVPR2022 https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py';
await(await PresentationFile.exportPptx(p)).save(B+'/authored.pptx');
execFileSync(R+'/python/python.exe',['presentations/patch-flow-readable.py'],{stdio:'inherit'});
const checked=await PresentationFile.importPptx(await FileBlob.load(B+'/candidate.pptx'));
await fs.writeFile(B+'/after.png',new Uint8Array(await(await checked.export({slide:checked.slides.items[12],format:'png',scale:1})).arrayBuffer()));
const {finalizePresentation}=await import(pathToFileURL(SK+'/container_tools/artifact_tool_utils.mjs'));
try { await finalizePresentation({workspaceDir:process.cwd(),candidatePath:B+'/candidate.pptx',finalPath:process.cwd()+'/presentations/output/开题报告_流程图重绘与讲解_20261019.pptx',pythonExecutable:R+'/python/python.exe',integrityValidatorPath:SK+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SK+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--expected-slide-count',String(orig.slides.items.length)],verifyArtifactToolImport:true,receiptPath:B+'/validation.json'});

} catch(e) { if(e.code!=='EPERM'||e.syscall!=='link')throw e; await fs.copyFile(B+'/candidate.pptx',process.cwd()+'/presentations/output/开题报告_流程图重绘与讲解_20261019.pptx',1); console.log('COPY_AFTER_LINK_EPERM'); }
