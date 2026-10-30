"""Frozen offline E0/E1 diagnostics. No training, inference, or input modification."""
import argparse,ast,copy,hashlib,importlib.util,json,sys
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
 global OUT
 parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path);parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();OUT=args.output
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'examples').mkdir()
 matcher.self_test()
 rng=np.random.default_rng(20261023)
 for shape in [(20,30),(80,100),(180,250)]:
  mask=(rng.random(shape)>.3).astype('uint8');assert np.array_equal(band(mask),reference.boundary(mask))
 ann=ROOT/'data/UIIS/annotations/val.json';assert sha(ann)=='5ed740baa1aa857058285d5947aa480774902af596e8b7effab50e631960e4a2'
 coco=COCO(str(ann));assert len(coco.imgs)==691
 names={k:v['name'] for k,v in coco.cats.items()};groups={};paths={};ap={}
 for label,folder in [('E0','uiis-baseline-36e-evaluation'),('E1','uiis-e1-bll-36e-evaluation')]+([('E1-P',None)] if args.candidate else []):
  path=args.candidate if label=='E1-P' else ROOT/'reports'/folder/'predictions.segm.json';paths[label]=dict(path=str(path),sha256=sha(path))
  preds=json.loads(path.read_text());group=defaultdict(list)
  for k,p in enumerate(preds):group[p['image_id']].append(dict(p,source_index=k))
  groups[label]={i:sorted(group[i],key=lambda x:-x['score']) for i in coco.imgs}
  clean=[{k:v for k,v in p.items() if k!='bbox'} for p in preds]
  ev=COCOeval(coco,coco.loadRes(clean),'segm');ev.params.maxDets=[100,300,1000];ev.evaluate();ev.accumulate();ev.summarize()
  if folder:
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
  for j,a in enumerate(gt):rows.append(dict(image_id=iid,gt_id=a['id'],category=names[a['category_id']],size=size(a['area']),area=a['area'],**{label:local[label][j] for label in groups}))
  if ix%100==0:print('DIAGNOSTICS',ix+1,flush=True)
 summary={}
 for sz in ['small','medium','large']:
  rr=[r for r in rows if r['size']==sz];common=[r for r in rr if r['E0']['matched'] and r['E1']['matched']]
  lost=[r for r in rr if r['E0']['matched'] and not r['E1']['matched']];gained=[r for r in rr if r['E1']['matched'] and not r['E0']['matched']]
  summary[sz]=dict(gt=len(rr),common=len(common),lost=len(lost),gained=len(gained),lost_reasons=dict(Counter(r['E1']['miss_reason'] for r in lost)),conditional={m:{key:mean([r[m][key] for r in rr if r[m]['matched']]) for key in ['mask_iou','boundary_iou']} for m in groups},common_means={m:{key:mean([r[m][key] for r in common]) for key in ['mask_iou','boundary_iou','under_fraction','over_fraction']} for m in ['E0','E1']},boundary_improved=sum(r['E1']['boundary_iou']>r['E0']['boundary_iou']+1e-12 for r in common),boundary_worsened=sum(r['E1']['boundary_iou']<r['E0']['boundary_iou']-1e-12 for r in common))
  summary[sz]['common_by_class']={cat:dict(n=len(cc),delta_boundary=mean([r['E1']['boundary_iou']-r['E0']['boundary_iou'] for r in cc]),delta_mask=mean([r['E1']['mask_iou']-r['E0']['mask_iou'] for r in cc])) for cat in names.values() if (cc:=[r for r in common if r['category']==cat])}
 result=dict(images=691,inputs=paths,annotation_sha256=sha(ann),coco=ap,thresholds=totals,prediction_labels_at05=pred_labels,summary=summary)
 (OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');(OUT/'instances.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
 assert all(sha(Path(v['path']))==v['sha256'] for v in paths.values())
 panels={}
 if args.candidate:
  for labels in [('E0','E1-P'),('E1','E1-P'),('E0','E1','E1-P')]:
   panel={}
   for sz in ['all','small','medium','large']:
    universe=[r for r in rows if sz=='all' or r['size']==sz]
    common=[r for r in universe if all(r[m]['matched'] for m in labels)]
    panel[sz]=dict(gt=len(universe),common=len(common),coverage=len(common)/len(universe) if universe else None,means={m:{k:mean([r[m][k] for r in common]) for k in ['mask_iou','boundary_iou','over_fraction','under_fraction']} for m in labels})
    if len(labels)==2:
     a,b=labels;delta=lambda r:r[b]['boundary_iou']-r[a]['boundary_iou']
     lost=[r for r in universe if r[a]['matched'] and not r[b]['matched']]
     panel[sz].update(improved=sum(delta(r)>1e-12 for r in common),worsened=sum(delta(r)<-1e-12 for r in common),new_missed=len(lost),new_matched=sum(r[b]['matched'] and not r[a]['matched'] for r in universe))
     if sz=='large':
      panel[sz]['case_gt_ids']=dict(decline=[r['gt_id'] for r in sorted([r for r in common if delta(r)<-1e-12],key=lambda r:(delta(r),r['gt_id']))[:2]],gain=[r['gt_id'] for r in sorted([r for r in common if delta(r)>1e-12],key=lambda r:(-delta(r),r['gt_id']))[:1]],least_change=[r['gt_id'] for r in sorted(common,key=lambda r:(abs(delta(r)),r['gt_id']))[:1]],new_miss=[r['gt_id'] for r in sorted(lost,key=lambda r:(-r[a]['mask_iou'],r['gt_id']))[:2]])
   panels['/'.join(labels)]=panel
 else:
  old=json.loads((ROOT/'reports/e1-offline-comparison/summary.json').read_text())
  for k in ['images','annotation_sha256','coco','thresholds','prediction_labels_at05','summary']:assert json.loads(json.dumps(result[k]))==old[k],k
  assert rows==json.loads((ROOT/'reports/e1-offline-comparison/instances.json').read_text())
 (OUT/'paired-panels.json').write_text(json.dumps(dict(panels=panels,regression_passed=not bool(args.candidate)),indent=2))
 print('OFFLINE_COMPARISON_COMPLETE',flush=True)

if __name__=="__main__":main()
