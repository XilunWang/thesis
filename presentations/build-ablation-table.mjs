import fs from 'node:fs/promises';import path from 'node:path';import{pathToFileURL}from'node:url';
const R='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies';const{PresentationFile,FileBlob}=await import(pathToFileURL(R+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs'));
const B='presentations/.build/ablation-table';await fs.mkdir(B,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Lenovo/Desktop/左怀琪-专业学位硕士开题PPT.pptx'));const s=p.slides.items[9];
function tx(t,x,y,w,h,size=24,color='#333333',bold=false){let a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=t;a.text.style={typeface:'微软雅黑',fontSize:size,color,bold,verticalAlignment:'middle',insets:{left:0,right:0,top:0,bottom:0}};return a;}
tx('拟开展的消融与对照实验',205,125,1015,45,30,'#19652B',true);
const values=[['实验组别','基础架构','训练数据策略','DSGAT','MFRM\n+ BLL','验证目标'],['1. 基线','Mask2Former','真实数据微调','不使用','不使用','建立统一条件下的\n性能参照'],['3. 模块验证A','Mask2Former','真实数据微调','使用','不使用','验证细节增强对\n区域完整性的影响'],['4. 模块验证B','Mask2Former','真实数据微调','不使用','使用','验证掩码细化与\n边界学习的作用'],['5. 联合方案','改进的\nMask2Former','SynTIDE预训练\n+ 真实数据微调','使用','使用','评价数据策略与\n模块组合的整体效果']];
const t=s.tables.add({rows:5,columns:6,left:205,top:195,width:1020,height:355,values,columnWidths:[150,175,215,110,135,235]});
for(let r=0;r<5;r++)for(let c=0;c<6;c++){let a=t.getCell(r,c);a.fill=r===0?'#19652B':r%2===1?'#F0F5F0':'#FFFFFF';a.text.style={typeface:'微软雅黑',fontSize:21,color:r===0?'#FFFFFF':'#333333',bold:r===0,verticalAlignment:'middle'};}t.borders.assign({style:'solid',fill:'#CFDBD0',width:.7});
tx('统一评价：实例AP、AP50、AP75、边界质量及计算开销。',210,574,1000,40,23);
tx('第5组同时改变数据策略与模块配置，评价整体方案，不能单独归因于模块。\nSynTIDE属于待论证方案，以上实验尚未完成。',210,625,1000,60,20,'#666666');
s.speakerNotes.text='本页按用户指定保留图中第1、3、4、5组，编号沿用原图，不包含第2组。真实数据指UIIS，沿用官方划分与既定重复图片规则。真实数据微调不代表随机初始化，原基线使用COCO预训练权重，各组共同初始化和训练预算需明确。DSGAT、MFRM及BLL的技术来源为WaterMask，需要引用。第3组验证DSGAT适配，第4组验证MFRM与BLL这一组合，不能分别归因MFRM与BLL。第5组保留原图SynTIDE预训练加真实数据微调的设计，但SynTIDE的具体来源、数据构建方式与可行性尚未核对，不代表本项目已启用该路线。如果目标是模块组合消融，需要额外设置相同真实数据策略下同时使用两个模块的组；若要分析生成数据收益，还需恢复仅改变数据策略的对照组。不能据这四组直接宣称协同作用或SOTA。原图中的性能严重受限、攻克粘连及刷新SOTA等预期，改为待验证目标。来源：https://openaccess.thecvf.com/content/ICCV2023/html/Lian_WaterMask_Instance_Segmentation_for_Underwater_Imagery_ICCV_2023_paper.html';
await(await PresentationFile.exportPptx(p)).save(B+'/authored.pptx');await fs.writeFile(B+'/slide10.png',new Uint8Array(await(await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
