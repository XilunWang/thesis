"""CPU target geometry audit; fixed training subset, no augmentation/model/optimizer."""
import sys,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.query_bll import build_bll_targets
from scripts.bll_core import boundary
O=R/'reports/bll-design-audit'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 ann=R/'data/UIIS/annotations/train.json';coco=COCO(str(ann));ids=sorted(coco.imgs)[:128];rows=[];example=None
 # Every single-pixel phase within stride4: exact production helper versus independent NumPy pooling.
 for k in range(16):
  gt=torch.zeros((1,4,4));gt[0,k//4,k%4]=1;t,v=build_bll_targets(gt,(1,1),(4,4));assert t.item()==1
  assert F.avg_pool2d(gt[:,None],4).item()==1/16
 for count,iid in enumerate(ids):
  im=coco.imgs[iid];h,w=im['height'],im['width'];hp=(h+3)//4*4;wp=(w+3)//4*4
  for a in coco.imgToAnns[iid]:
   if a.get('iscrowd',0):continue
   mask=coco.annToMask(a);area=int(mask.sum());assert area>0
   gt=np.zeros((hp,wp),np.uint8);gt[:h,:w]=mask
   tensor=torch.from_numpy(gt)[None];target,valid=build_bll_targets(tensor,(hp//4,wp//4),(h,w))
   cells=gt.reshape(hp//4,4,wp//4,4).transpose(0,2,1,3);coverage=cells.mean(axis=(2,3));occupancy=cells.max(axis=(2,3));assert np.array_equal(target[0].numpy(),occupancy)
   up=np.repeat(np.repeat(occupancy,4,0),4,1)[:h,:w];assert np.all(up>=mask)
   region=boundary(target,3)[0].numpy()&valid.numpy()
   mixed=(coverage>0)&(coverage<1);minority=(coverage>0)&(coverage<.5)
   # Only full blocks for coverage/gradient probe: exclude partial valid-edge blocks.
   full=np.zeros_like(occupancy,dtype=bool);full[:h//4,:w//4]=True
   selected_fg=region&(occupancy>0)&full
   row=dict(image_id=iid,gt_id=a['id'],size='small' if a['area']<1024 else 'medium' if a['area']<9216 else 'large',source_area=area,expanded_area=int(up.sum()),ratio=float(up.sum()/area),mixed_cells=int((mixed&full).sum()),minority_cells=int((minority&full).sum()),selected_gt_boundary_cells=int(region.sum()),selected_fg_cells=int(selected_fg.sum()),selected_minority_cells=int((selected_fg&minority).sum()),selected_fg_coverage_sum=float(coverage[selected_fg].sum()),nearest_empty=not bool(gt[::4,::4].any()))
   rows.append(row)
   if row['size']=='large' and (example is None or row['ratio']>example[0]):example=(row['ratio'],row,mask,up,im)
  if count%32==0:print('AUDIT',count+1,flush=True)
 summary={}
 for size in ['small','medium','large']:
  rr=[r for r in rows if r['size']==size];n=sum(r['selected_fg_cells'] for r in rr)
  summary[size]=dict(instances=len(rr),global_expansion_percent=(sum(r['expanded_area'] for r in rr)/sum(r['source_area'] for r in rr)-1)*100,median_ratio=float(np.median([r['ratio'] for r in rr])),p95_ratio=float(np.percentile([r['ratio'] for r in rr],95)),nearest_empty=sum(r['nearest_empty'] for r in rr),occupancy_empty=0,selected_foreground_cells=n,selected_foreground_mean_coverage=sum(r['selected_fg_coverage_sum'] for r in rr)/n,selected_foreground_minority_fraction=sum(r['selected_minority_cells'] for r in rr)/n)
 result=dict(passed=True,images=128,instances=len(rows),ids=ids,annotation_sha256=sha(ann),summary=summary,scope='Native-resolution training masks; no resizing/crop/flip or model. stride4 production target helper. GT boundary only, not actual previous-prediction union. Coverage probes exclude partial blocks; expanded area clips valid image.',toy=dict(foreground_pixels=1,block_pixels=16,occupancy_target=1,coverage_target=.0625,gradient_at_logit_zero_occupancy=-.5,gradient_at_logit_zero_coverage=.4375),source_sha256={f:sha(R/f) for f in ['scripts/query_bll.py','scripts/bll_core.py','scripts/audit-bll-design.py']})
 (O/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(O/'instances.json').write_text(json.dumps(rows),encoding='utf-8')
 ratio,row,mask,up,im=example
 ys,xs=np.where(mask);x0=max(0,int(xs.min())-12);x1=min(mask.shape[1],int(xs.max())+13);y0=max(0,int(ys.min())-12);y1=min(mask.shape[0],int(ys.max())+13)
 canvas=Image.new('RGB',(1200,530),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
 for k,(title,data) in enumerate([('原始GT',mask),('占有规则展开（仅用于审查）',up),('红色：额外覆盖区域',None)]):
  if data is None:
   arr=np.full((*mask.shape,3),255,np.uint8);arr[mask>0]=[80,80,80];arr[(up>0)&(mask==0)]=[240,40,40]
  else:arr=np.repeat((255-data*200)[:,:,None],3,axis=2)
  tile=Image.fromarray(arr[y0:y1,x0:x1]);tile.thumbnail((390,420));canvas.paste(tile,(400*k,65));draw.text((400*k+5,35),title,font=font,fill='black')
 draw.text((6,5),f"训练图 {im['file_name']} / GT {row['gt_id']} / 面积增幅 {(ratio-1)*100:.2f}%",font=font,fill='black');draw.text((6,498),'固定前128张中面积膨胀比例最大的大目标；不是模型预测，也不是全体代表。',font=font,fill='black');canvas.save(O/'target-expansion.png')
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
