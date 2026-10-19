import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
const R='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies',SK='C:/Users/Lenovo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations';
process.env.RUNTIME_NODE_MODULES=R+'/node/node_modules';
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(R+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
const B=process.cwd()+'/presentations/.build/flow-clarified',T=process.cwd()+'/presentations/output/开题报告_研究内容与框架_20261017_v2_研究现状更新.pptx';
await fs.mkdir(B,{recursive:true});await fs.copyFile(T,B+'/original.pptx');
const orig=await PresentationFile.importPptx(await FileBlob.load(T));
await fs.writeFile(B+'/before.png',new Uint8Array(await(await orig.export({slide:orig.slides.items[12],format:'png',scale:1})).arrayBuffer()));
const proto=orig.toProto();proto.slides[12].elements=proto.slides[12].elements.filter(e=>e.bbox.xEmu<180*9525||(e.bbox.yEmu<100*9525&&!(e.paragraphs||[]).length));
const p=Presentation.load(proto),s=p.slides.items[12];let id=0;
function shape(t,x,y,w,h,kind='text',size=22){let a=s.shapes.add({name:'flow-'+(++id),geometry:kind==='text'?'textbox':'rect',position:{left:x,top:y,width:w,height:h},fill:kind==='text'?'none':kind==='new'?'#E5F1E7':'#F1F3F3',line:{fill:kind==='text'?'none':kind==='new'?'#19652B':'#7A8580',width:kind==='text'?0:1.4}});a.text=t;a.text.style={typeface:'微软雅黑',fontSize:size,color:kind==='new'?'#19652B':'#333333',bold:kind==='new',alignment:kind==='text'?'left':'center',verticalAlignment:'middle',insets:{left:5,right:5,top:3,bottom:3}};return a;}
function arrow(a,b,from='right',to='left',dashed=false){let c=s.shapes.connect(a,b,{kind:(from==='right'&&to==='left')?'straight':'elbow',fromSide:from,toSide:to,line:{fill:dashed?'#AF7628':'#718078',width:1.7,style:dashed?'dashed':'solid'},tail:{type:'triangle',width:'sm',length:'sm'}});c.name='flow-'+(++id);return c;}
const heading=shape('3.2 模型框架与拟加入部分',200,38,735,48,'text',32);heading.text.style={typeface:'微软雅黑',fontSize:32,bold:true,color:'#19652B',verticalAlignment:'middle'};
shape('灰色：Mask2Former原有功能     绿色：拟加入或适配     虚线：训练监督',205,103,1030,35,'text',20);
let dsg=shape('① DSGAT细节增强',490,155,210,48,'new',21);
let input=shape('水下图像',200,230,110,65,'old'),bb=shape('ResNet-50\n提取图像特征',340,230,140,65,'old',20),px=shape('像素解码器\n融合多尺度特征',510,230,170,65,'old',20),tr=shape('Transformer解码器\n更新实例查询',730,230,210,65,'old',20),qu=shape('实例查询特征',1000,230,210,65,'old',21);
arrow(input,bb);arrow(bb,px);arrow(px,tr);arrow(tr,qu);arrow(dsg,px,'bottom','top');
let feat=shape('掩码特征',510,350,170,60,'old'),pred=shape('掩码预测\n查询与像素特征结合',730,350,210,60,'old',19),coarse=shape('原始实例掩码',1000,350,210,60,'old',21),cls=shape('类别预测',1000,155,210,48,'old',21);
arrow(px,feat,'bottom','top');arrow(feat,pred);arrow(qu,pred,'bottom','top');arrow(qu,cls,'top','bottom');arrow(pred,coarse);
let ref=shape('② MFRM适配\n前景与边界细化',1000,465,210,75,'new',20),out=shape('细化后的实例掩码',730,465,210,75,'new',20);
arrow(coarse,ref,'bottom','top');arrow(ref,out,'left','right');
shape('拟结合空间细节与查询信息\n具体聚合与融合方式待验证',210,455,460,75,'text',21);
shape('仅训练时使用',205,565,240,30,'text',22);
let gt=shape('实例标注',205,610,120,55,'old',21),edge=shape('匹配实例并\n提取边界标签',370,610,180,55,'new',19),loss=shape('③ BLL边界监督',1000,610,210,55,'new',21);
arrow(gt,edge,'right','left',true);arrow(edge,loss,'right','left',true);arrow(ref,loss,'bottom','top',true);
shape('原有分类与掩码损失保留',580,562,370,35,'text',21);
shape('概念方案：原有掩码注意力反馈未展开；绿色部分尚未完成实现。',205,682,1020,30,'text',18);
s.speakerNotes.text='灰色为Mask2Former原有功能：骨干提取图像特征，像素解码器融合多尺度信息；Transformer解码器更新查询，查询分别用于类别预测及掩码表示，与像素特征结合生成实例掩码。掩码特征不是原始照片，查询也不是分割图。绿色表示相对于基线拟加入或改造的部分，不能称为全部原创模块。①DSGAT拟在像素解码器特征融合中适配，图中的箭头表示适配位置，并非独立的额外输入。②MFRM需将原区域式方法适配为实例特征聚合与细化，拟利用空间特征、查询及初始掩码，但具体聚合算子与边界融合尚未确定。此图仅画主路径，补充特征输入用文字说明，未假定MFRM只接收二值掩码。细化后的掩码为改进路径输出，原Mask2Former已经能输出掩码。③从人工实例标注提取边界标签，在实例匹配后监督边界预测，BLL需有相应边界输出，此处包含在MFRM细化分支中。橙色虚线表示标签与预测共同用于计算训练损失，不是推理输入。原分类与掩码损失保留，预测新图片无需人工标注。原有逐层掩码注意力反馈为简洁未展开。来源：WaterMask ICCV2023；Mask2Former CVPR2022 https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py 。';
await(await PresentationFile.exportPptx(p)).save(B+'/authored.pptx');
execFileSync(R+'/python/python.exe',['presentations/patch-flow-clarified.py'],{stdio:'inherit'});
const checked=await PresentationFile.importPptx(await FileBlob.load(B+'/candidate.pptx'));
await fs.writeFile(B+'/after.png',new Uint8Array(await(await checked.export({slide:checked.slides.items[12],format:'png',scale:1})).arrayBuffer()));
const {finalizePresentation}=await import(pathToFileURL(SK+'/container_tools/artifact_tool_utils.mjs'));
try { await finalizePresentation({workspaceDir:process.cwd(),candidatePath:B+'/candidate.pptx',finalPath:process.cwd()+'/presentations/output/开题报告_流程图区分原有与新增_20261019.pptx',pythonExecutable:R+'/python/python.exe',integrityValidatorPath:SK+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SK+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--expected-slide-count',String(orig.slides.items.length)],verifyArtifactToolImport:true,receiptPath:B+'/validation.json'});

} catch(e) { if(e.code!=='EPERM'||e.syscall!=='link')throw e; await fs.copyFile(B+'/candidate.pptx',process.cwd()+'/presentations/output/开题报告_流程图区分原有与新增_20261019.pptx',1); console.log('COPY_AFTER_LINK_EPERM'); }
