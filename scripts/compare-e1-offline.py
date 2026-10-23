"""Frozen offline E0/E1 diagnostics. No training, inference, or input modification."""
import ast,copy,hashlib,importlib.util,json,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/e1-offline-comparison'
def load_script(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
matcher=load_script('analyze-uiis-errors'); reference=load_script('validate-uiis-boundary')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def band(mask):
 d=max(1,round(.02*np.hypot(*mask.shape)))
 return mask.astype(bool)&~cv2.erode(mask.astype('uint8'),np.ones((3,3),np.uint8),iterations=d,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
def iou(a,b):return float(np.count_nonzero(a&b)/np.count_nonzero(a|b))
def size(a):return 'small' if a<1024 else 'medium' if a<9216 else 'large'
def mean(x):return float(np.mean(x)) if len(x) else None
def validmean(x):return mean(x[x>=0])
def main():
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'examples').mkdir()
 matcher.self_test()
 rng=np.random.default_rng(20261023)
 for shape in [(20,30),(80,100),(180,250)]:
  mask=(rng.random(shape)>.3).astype('uint8');assert np.array_equal(band(mask),reference.boundary(mask))
 ann=ROOT/'data/UIIS/annotations/val.json';assert sha(ann)=='5ed740baa1aa857058285d5947aa480774902af596e8b7effab50e631960e4a2'
 coco=COCO(str(ann));assert len(coco.imgs)==691
 names={k:v['name'] for k,v in coco.cats.items()};groups={};paths={};ap={}
 for label,folder in [('E0','uiis-baseline-36e-evaluation'),('E1','uiis-e1-bll-36e-evaluation')]:
  path=ROOT/'reports'/folder/'predictions.segm.json';paths[label]=dict(path=str(path),sha256=sha(path))
  preds=json.loads(path.read_text());group=defaultdict(list)
  for k,p in enumerate(preds):group[p['image_id']].append(dict(p,source_index=k))
  groups[label]={i:sorted(group[i],key=lambda x:-x['score']) for i in coco.imgs}
  clean=[{k:v for k,v in p.items() if k!='bbox'} for p in preds]
  ev=COCOeval(coco,coco.loadRes(clean),'segm');ev.params.maxDets=[100,300,1000];ev.evaluate();ev.accumulate();ev.summarize()
  old=json.loads((ROOT/'reports'/folder/'metrics.json').read_text())['metrics']
  for n,suffix in enumerate(['','_50','_75','_s','_m','_l']):assert round(float(ev.stats[n]),3)==old['coco/segm_mAP'+suffix],(label,suffix)
  precision=ev.eval['precision'];recall=ev.eval['recall'];idx=ev.params.areaRngLbl.index('large')
  ap[label]=dict(stats=ev.stats.tolist(),large_by_class={names[c]:dict(ap=validmean(precision[:,:,k,idx,-1]),ar=validmean(recall[:,k,idx,-1])) for k,c in enumerate(ev.params.catIds)},large_by_iou={f'{t:.2f}':dict(ap=validmean(precision[j,:,:,idx,-1]),ar=validmean(recall[j,:,idx,-1])) for j,t in enumerate(ev.params.iouThrs)})
  print(label,'COCO_MATCHED',flush=True)
 totals={m:{str(t):defaultdict(Counter) for t in [.3,.5,.7]} for m in groups};rows=[];pred_labels={m:Counter() for m in groups}
 for ix,(iid,im) in enumerate(coco.imgs.items()):
  gt=[a for a in coco.imgToAnns[iid] if not a.get('iscrowd',0)];crowd=[a for a in coco.imgToAnns[iid] if a.get('iscrowd',0)]
  gr=[coco.annToRLE(a) for a in gt];cats=[a['category_id'] for a in gt];local={};masks={};gb={}
  for model in groups:
   pp=groups[model][iid];ious=mu.iou([p['segmentation'] for p in pp],gr,[0]*len(gt)) if pp and gt else np.zeros((len(pp),len(gt)))
   at05={}
   for threshold in [.3,.5,.7]:
    n=sum(p['score']>=threshold for p in pp);labels,matched,used=matcher.match(ious[:n],[p['category_id'] for p in pp[:n]],cats)
    for j,a in enumerate(gt):
     c=totals[model][str(threshold)][size(a['area'])];c['gt']+=1;c['matched' if j in used else 'missed']+=1
    if threshold==.5:
     if crowd and n:
      ci=mu.iou([p['segmentation'] for p in pp[:n]],[coco.annToRLE(a) for a in crowd],[1]*len(crowd))
      for k,l in enumerate(labels):
       if l!='matched' and any(a['category_id']==pp[k]['category_id'] and ci[k,j]>=.5 for j,a in enumerate(crowd)):labels[k]='ignored_crowd'
     pred_labels[model].update(labels)
     at05={j:k for k,j in enumerate(matched) if j is not None}
   detail=[]
   for j,a in enumerate(gt):
    same=[k for k,p in enumerate(pp) if p['category_id']==a['category_id']];anybest=max(range(len(pp)),key=lambda k:ious[k,j]) if pp else None
    best=max(same,key=lambda k:ious[k,j]) if same else None
    k=at05.get(j);reason=None
    if k is None:
     high=[z for z in same if ious[z,j]>=.5]
     if any(pp[z]['score']>=.5 for z in high):reason='matching_competition'
     elif high:reason='below_score_threshold'
     elif anybest is not None and ious[anybest,j]>=.5:reason='wrong_class_overlap'
     elif best is not None and ious[best,j]>=.1:reason='mask_iou_below_05'
     else:reason='no_same_class_overlap_01'
    d=dict(matched=k is not None,pred_index=k,best_same_index=best,best_same_iou=float(ious[best,j]) if best is not None else 0,miss_reason=reason)
    if k is not None:
     gm=mu.decode(gr[j]).astype(bool);pm=mu.decode(pp[k]['segmentation']).astype(bool);assert gm.shape==pm.shape==(im['height'],im['width'])
     if j not in gb:gb[j]=band(gm)
     d.update(score=pp[k]['score'],mask_iou=float(ious[k,j]),boundary_iou=iou(gb[j],band(pm)),under_fraction=float(np.count_nonzero(gm&~pm)/gm.sum()),over_fraction=float(np.count_nonzero(pm&~gm)/gm.sum()))
    detail.append(d)
   local[model]=detail
  for j,a in enumerate(gt):rows.append(dict(image_id=iid,gt_id=a['id'],category=names[a['category_id']],size=size(a['area']),area=a['area'],E0=local['E0'][j],E1=local['E1'][j]))
  if ix%100==0:print('DIAGNOSTICS',ix+1,flush=True)
 summary={}
 for sz in ['small','medium','large']:
  rr=[r for r in rows if r['size']==sz];common=[r for r in rr if r['E0']['matched'] and r['E1']['matched']]
  lost=[r for r in rr if r['E0']['matched'] and not r['E1']['matched']];gained=[r for r in rr if r['E1']['matched'] and not r['E0']['matched']]
  summary[sz]=dict(gt=len(rr),common=len(common),lost=len(lost),gained=len(gained),lost_reasons=dict(Counter(r['E1']['miss_reason'] for r in lost)),conditional={m:{key:mean([r[m][key] for r in rr if r[m]['matched']]) for key in ['mask_iou','boundary_iou']} for m in groups},common_means={m:{key:mean([r[m][key] for r in common]) for key in ['mask_iou','boundary_iou','under_fraction','over_fraction']} for m in groups},boundary_improved=sum(r['E1']['boundary_iou']>r['E0']['boundary_iou']+1e-12 for r in common),boundary_worsened=sum(r['E1']['boundary_iou']<r['E0']['boundary_iou']-1e-12 for r in common))
  summary[sz]['common_by_class']={cat:dict(n=len(cc),delta_boundary=mean([r['E1']['boundary_iou']-r['E0']['boundary_iou'] for r in cc]),delta_mask=mean([r['E1']['mask_iou']-r['E0']['mask_iou'] for r in cc])) for cat in names.values() if (cc:=[r for r in common if r['category']==cat])}
 result=dict(images=691,inputs=paths,annotation_sha256=sha(ann),coco=ap,thresholds=totals,prediction_labels_at05=pred_labels,summary=summary)
 (OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');(OUT/'instances.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
 assert all(sha(Path(v['path']))==v['sha256'] for v in paths.values())
 render_cases(coco,groups,rows)
 print('OFFLINE_COMPARISON_COMPLETE',flush=True)

def render_cases(coco,groups,rows):
 # Predeclared case selection; selected GT only, same crop and overlay colors in every panel.
 common=[r for r in rows if r['size']=='large' and r['E0']['matched'] and r['E1']['matched']];delta=lambda r:r['E1']['boundary_iou']-r['E0']['boundary_iou']
 chosen=[('boundary_decline',r) for r in sorted(common,key=lambda r:(delta(r),r['gt_id']))[:2]]
 chosen += [('boundary_gain',sorted(common,key=lambda r:(-delta(r),r['gt_id']))[0]),('unchanged',min(common,key=lambda r:(abs(delta(r)),r['gt_id'])))]
 lost=[r for r in rows if r['size']=='large' and r['E0']['matched'] and not r['E1']['matched']]
 chosen += [('new_miss',r) for r in sorted(lost,key=lambda r:(-r['E0']['mask_iou'],r['gt_id']))[:2]]
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);cases=[]
 for typ,row in chosen:
  im=coco.imgs[row['image_id']];raw=np.array(Image.open(ROOT/'data/UIIS/val'/im['file_name']).convert('RGB'));gm=coco.annToMask(coco.anns[row['gt_id']]).astype(bool)
  canvas=Image.new('RGB',(1600,560),'white');draw=ImageDraw.Draw(canvas)
  for col,label in enumerate(['Original','GT','E0','E1']):
   arr=raw.copy();mask=None;desc=label
   if label=='GT':mask=gm
   if label in groups:
    d=row[label];idx=d['pred_index'] if d['matched'] else d['best_same_index']
    if idx is not None:
     pred=groups[label][row['image_id']][idx];mask=mu.decode(pred['segmentation']).astype(bool);desc+=f" score={pred['score']:.5f}"
    if not d['matched']:desc+=' [not matched]'
   if mask is not None:arr[mask]=(arr[mask]*.6+np.array([255,120,30])*.4).astype('uint8')
   # green GT contour, orange predicted contour; original is unmodified.
   if label!='Original':
    contours,_=cv2.findContours(gm.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(arr,contours,-1,(0,255,100),2)
   tile=Image.fromarray(arr);tile.thumbnail((390,400));canvas.paste(tile,(col*400,65));draw.text((col*400+5,35),desc,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16),fill='black')
   if label in groups and row[label]['matched']:draw.text((col*400+5,475),f"IoU {row[label]['mask_iou']:.3f} / BIoU {row[label]['boundary_iou']:.3f}",font=font,fill='black')
  draw.text((8,5),f"{typ} | {im['file_name']} | GT {row['gt_id']} | {row['category']}",font=font,fill='black');draw.text((8,520),'绿色线：人工轮廓；橙色填充：当前实例掩码。未匹配候选不代表正确检测。',font=font,fill='black')
  path=f"examples/{typ}-{row['gt_id']}.jpg";canvas.save(OUT/path,quality=94);cases.append(dict(type=typ,path=path,instance=row))
 (OUT/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
