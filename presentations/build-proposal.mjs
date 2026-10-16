process.on('uncaughtException',e=>{console.error(e.message);process.exit(1)});
import fs from 'node:fs/promises';import path from 'node:path';import crypto from 'node:crypto';import {pathToFileURL} from 'node:url';
const ROOT=path.resolve(import.meta.dirname,'..'),R='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies',SK='C:/Users/Lenovo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(R+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
process.env.RUNTIME_NODE_MODULES=R+'/node/node_modules';
const B=ROOT+'/presentations/.build/proposal-v2',T=ROOT+'/presentations/templates/proposal-template.pptx',F=ROOT+'/presentations/output/水下实例分割_硕士开题报告_20261016_v2.pptx';
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
tx(cover,'硕士开题报告：让水下目标分得更完整、更准确',200,450,880,50,28,{color:green,align:'center'});
tx(cover,'姓名：待填写      专业：待填写\n导师：待填写      汇报日期：待填写',340,525,650,100,26);
notes(cover,'题目暂定。这次汇报主要说明：为什么研究水下分割，已经做了什么，接下来准备怎样改。');
let s=take(b1,'1. 我研究什么，为什么值得做？');
lines(s,[['把每一个水下目标单独圈出来','比如一张照片里有三只海星，模型不仅要认出“海星”，还要把每只海星的身体范围分别圈出来。这就是实例分割。'],['水下图片比普通照片更难处理','水里的光线、颜色和清晰度会变化，动物还可能很小或互相遮挡。因此，模型容易漏掉目标，也可能把身体的一部分分到背景里。'],['做好之后有什么用','分得更准确，就更方便统计水下生物数量、观察目标形态，也能为水下巡检提供信息。本课题先研究单张图片，暂不做视频和机器人部署。']]);notes(s,'可以用“认出是什么，再把每一个分别圈出来”解释任务。应用是研究意义，尚未部署。',[refs.water]);
s=take(b2,'2. 别人做到了哪一步？');
table(s,[['已有方法','简单理解','我准备怎样借鉴'],['Mask2Former\n本课题的起点','一个通用分割模型，可以学习怎样把图片中的物体分开','先在水下数据上训练，作为后续比较的起点'],['WaterMask\n水下专用方法','针对水下图像，研究怎样补充目标细节、改善分割轮廓','借鉴其处理思路，逐项适配到Mask2Former'],['DINOv2 / DiveSeg\n第二点的候选方向','利用预训练视觉模型提供图像信息，并研究怎样适应水下场景','后续考虑改善模型对水下图像的理解']],[260,400,360],155,355,24);
tx(s,'我的研究需要回答：借鉴这些方法以后，究竟改善了什么？\n因此要做改动前后的比较，不能只把几个模块装在一起。',210,550,1000,95,26,{color:green});
foot(s,'参考：Mask2Former（2022）、WaterMask（2023）、DINOv2及DiveSeg。完整链接见备注。');notes(s,'Mask2Former是基线。WaterMask模块并不能直接照搬，具体接口还要核对。第二点未定。',[refs.m2f,refs.water,refs.dino,refs.dive]);
s=take(b3,'3. 我已经完成了哪些工作？');
table(s,[['已完成的工作','具体结果'],['准备数据','使用UIIS水下数据，3937张训练图片、691张测试图片，共7类目标'],['训练起点模型','已经完成Mask2Former-R50的36轮训练。R50指其中负责提取图像信息的网络'],['评价整体效果','AP 32.6、AP50 47.6、AP75 35.3。这些是分割评分，通常越高越好'],['检查错误','完成全量预测分析，并查看固定案例，寻找漏检和目标范围不准确的问题']],[255,765],155,350,24);
tx(s,'现在已经有一个能运行、能比较的起点模型。\n这些成绩属于现成模型在UIIS上的结果，自己的改进还没有开始。',210,550,1000,90,27,{bold:true,color:green});
notes(s,'AP综合不同匹配严格程度的结果，32.6不是“32.6%的像素正确”。模型使用COCO预训练。没有复现WaterMask，不能把跨论文分差归功于自己。',['reports/uiis-baseline-36e-evaluation/metrics.json','README.md']);
s=take(b3,'4. 现有结果还有什么问题？');
await img(s,'reports/uiis-baseline-36e-evaluation/examples/1.jpg',205,140,490,280);
await img(s,'reports/uiis-baseline-36e-evaluation/examples/2.jpg',730,140,490,280);
tx(s,'海星的细长部分没有完整分出来',205,430,490,45,25,{bold:true,color:green});
tx(s,'部分小目标没有出现在显示结果中',730,430,490,45,25,{bold:true,color:green});
tx(s,'每组图左边是人工标注，右边是模型预测。右图只显示模型较有把握的结果。\n这些例子说明还有改进空间，但仅看几张图，不能说明所有错误都来自边界。\n下一步要进一步检查：目标是没被找到，还是找到了、但身体范围没有圈准确。',210,490,1000,145,25);
foot(s,'显示阈值为0.5。全量统计和固定案例用于找问题，边界质量还需要专门评价。');
notes(s,'展示原始预测，不修改结果。小目标未显示可能与分数阈值有关。应区别漏检、分类错误及轮廓偏差。',['reports/uiis-baseline-36e-evaluation/report.md','reports/uiis-error-analysis/report.md']);
s=take(b3,'5. 第一个研究点：让目标圈得更完整');
lines(s,[['想解决什么问题','像海星的细长腕足、小目标的轮廓，可能在分割时丢掉。准备先确认这些问题有多普遍，再决定优先改哪里。'],['准备怎么改','借鉴WaterMask的DSGAT、MFRM和边界学习损失。简单说，就是尝试补充细节信息，并在训练时让模型更关注目标轮廓。'],['具体怎么推进','先核对模块能接在什么位置，再一次只改一项。只有实验显示有效，才尝试组合。最终要解释为什么这样改，而不只是搬用模块。']]);
foot(s,'这是拟研究方案。目前还没有接入这些模块，也没有得到改进后的分数。');notes(s,'DSGAT用于细节恢复，MFRM涉及特征细化。BLL是边界学习损失，属于训练监督。WaterMask与Mask2Former结构不同，需调整特征尺寸、实例对应及监督方式。',[refs.water,refs.m2f]);
s=take(b3,'6. 第二个研究点：让模型更懂水下图片');
lines(s,[['与第一点有什么不同','第一点关注目标细节和轮廓。第二点准备研究：水下图片颜色偏、清晰度差时，模型怎样提取更有用的图像信息。'],['可以考虑什么办法','DINOv2是已经从大量图像中学习过的视觉模型。可以借鉴DiveSeg，让它提供图像信息，再研究怎样调整这些信息，使其更适合水下分割。'],['目前准备做到哪一步','这一点还只是候选方向。先查清别人已经做过什么，再用小实验判断是否值得继续。比较时使用相同的DINOv2，才能看出适配方法的作用。']]);
foot(s,'需要先确定具体问题和自己的改动，不能把“换一个更大的模型”直接当作研究贡献。');notes(s,'不要承诺DINOv2一定优于现有基线。第二研究点的独立性、创新性和成本都尚需论证。',[refs.dino,refs.dive]);
s=take(b3,'7. 怎样证明改动确实有用？');
lines(s,[['先保证比较公平','改动前后使用同样的数据规则，尽量保持训练轮数、预训练权重和训练方式一致。官方划分和已知重复图片都按既定规则保留。'],['每次只检验一个变化','先跑原模型，再分别加入每项改动，最后看组合效果。这样才能知道是哪项起作用。这种逐项比较，在论文中叫“消融实验”。'],['既看分数，也看具体图片','比较AP等整体评分，还要检查小目标、轮廓和固定案例。同时记录训练时间和显存，判断提升是否值得。关键结果再重复验证。']]);
foot(s,'WaterMask的论文数字可作参考。严格比较还需要说明或对齐预训练、优化器和数据增强等条件。');notes(s,'不使用测试集反复选超参数。后续开发验证规则需先统一确认。边界指标尚未计算，不预设一定有提升。',['README.md',refs.boundary]);
s=take(b4,'8. 接下来按什么顺序做？');
table(s,[['阶段','主要工作','完成后应拿到什么'],['先把问题查清楚','补充边界检查，阅读WaterMask代码，选一个最先尝试的改动','明确“改哪里、为什么改”的方案'],['完成第一研究点','逐项改模型，做完整训练和改动前后的比较','方法说明、对照表和有代表性的图片'],['再确定第二研究点','核对DINOv2 / DiveSeg路线，先做小实验，再决定正式方案','可独立解释的第二项研究及实验结果'],['整理论文和答辩','统一实验记录，补必要的比较，边做边写论文','论文正文、结果图表和答辩材料']],[215,420,385],150,390,24);
tx(s,'现在处于第一阶段：基线已完成，正在查问题、准备第一个改动。',210,575,1000,65,27,{bold:true,color:green});
foot(s,'具体月份等待开题、毕业时间确认。按阶段推进，避免同时铺开过多实验。');notes(s,'仍处于P2，未进入改进模型正式训练。第二点不是本轮已确定的任务。',['README.md 第2节']);
s=take(b4,'9. 预期形成什么成果？');
lines(s,[['一个有证据支持的改进方法','说明现有模型哪里做得不足，自己的修改怎样解决问题，并用相同条件下的实验验证。如果改动无效，就记录原因并调整方案。'],['两项能够分别讲清楚的研究内容','第一项围绕细节和轮廓，第二项考虑水下图像特征。第二项需要进一步论证，最终各自解决什么问题、有什么贡献，都要有实验支持。'],['一套可复现的论文材料','保留代码、配置、训练记录和评价结果，整理对照表及图片，形成论文与答辩材料。当前最重要的是先完成第一个有效改动。']]);
foot(s,'现阶段是研究计划，不预先承诺精度提升或发表成果。恳请各位老师批评指正。');notes(s,'总结时强调已经完成基线，下一步是从证据出发设计和验证改动。');
p.slides.keep(p.slides.items.map((s,i)=>ordered.includes(s)?i:-1).filter(i=>i>=0));p.slides.reorder(ordered.map(s=>'sl/'+s.id));
const candidate=B+'/candidate.pptx';await(await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<ordered.length;i++)await fs.writeFile(B+`/slide-${i+1}.png`,new Uint8Array(await(await p.export({slide:ordered[i],format:'png',scale:1})).arrayBuffer()));
const {finalizePresentation}=await import(pathToFileURL(SK+'/container_tools/artifact_tool_utils.mjs'));
try{console.log(await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:F,pythonExecutable:R+'/python/python.exe',integrityValidatorPath:SK+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SK+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--expected-slide-count','10','--validate-heading-fit',...tables.flatMap(s=>['--require-native-table-slide',String(ordered.indexOf(s)+1)])],requiredNativeTableOwnerSlides:tables.map(s=>ordered.indexOf(s)+1),fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:T,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(T)).digest('hex')},verifyArtifactToolImport:true,receiptPath:B+'/validation.json'}));}
catch(e){if(e.code!=='EPERM'||e.syscall!=='link')throw e;await fs.copyFile(candidate,F,1);const hash=crypto.createHash('sha256').update(await fs.readFile(F)).digest('hex');await fs.writeFile(B+'/copy-receipt.json',JSON.stringify({finalPath:F,sha256:hash,limitation:'finalizer reached final hard-link publication after validation, Windows EPERM; delivered using exclusive copy',slideCount:10},null,2));console.log('COPIED_AFTER_LINK_EPERM',F);}
