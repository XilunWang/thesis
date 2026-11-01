"""Saved-prediction COCO matching/ranking audit. No inference or training."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
OUT=ROOT/'reports/direct-query-instances-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def avg(x):
 x=np.asarray(x);x=x[x>=0];return float(x.mean()*100) if x.size else None
def main():
 OUT.mkdir(exist_ok=False)
 ann=ROOT/'data/UIIS/annotations/val.json';gt=COCO(str(ann))
 assert sha(ann)=='5ed740baa1aa857058285d5947aa480774902af596e8b7effab50e631960e4a2'
 old=json.loads((ROOT/'reports/direct-query-ap-screen-run/comparison.json').read_text())
 paths={'control':ROOT/'reports/valid-region-ap-screen-control/predictions.segm.json','candidate':ROOT/'reports/direct-query-ap-screen-candidate/predictions.segm.json'}
 result={'models':{},'hashes':{'annotation':sha(ann)},'scope':'COCO same-class greedy matching, all saved predictions, no score cutoff; small area<=1024, maxDets1000 (100 outputs/image). Output associations, not training causality.'};instances={}
 for name,path in paths.items():
  result['hashes'][name]=sha(path);pp=json.loads(path.read_text());assert len(pp)==69100
  dt=gt.loadRes([{k:v for k,v in p.items() if k!='bbox'} for p in pp])
  ev=COCOeval(gt,dt,'segm');ev.params.maxDets=[100,300,1000];ev.evaluate();ev.accumulate();ev.summarize()
  assert np.allclose(ev.stats*100,old['coco'][name]['stats'],rtol=0,atol=1e-8)
  classes={};small={};matching={}
  for k,cid in enumerate(ev.params.catIds):
   cname=gt.cats[cid]['name'];c={'ap_by_area':{area:avg(ev.eval['precision'][:,:,k,a,-1]) for a,area in enumerate(ev.params.areaRngLbl)},'ranking':{}}
   for area_idx,area in [(0,'all'),(1,'small')]:
    ee=[e for e in ev.evalImgs if e and e['category_id']==cid and e['aRng']==ev.params.areaRng[area_idx]]
    for ti in [0,5]:
     thr=f'{ev.params.iouThrs[ti]:.2f}';key=area+'_'+thr
     ng=sum(int(np.count_nonzero(~np.asarray(e['gtIgnore'],bool))) for e in ee)
     scores=np.concatenate([np.asarray(e['dtScores']) for e in ee]);matched=np.concatenate([e['dtMatches'][ti]>0 for e in ee]);ignore=np.concatenate([e['dtIgnore'][ti] for e in ee])
     order=np.argsort(-scores,kind='mergesort');order=order[~ignore[order]];m=matched[order]
     tp=np.cumsum(m);fp=np.cumsum(~m);at={}
     for recall in [.25,.5,.75]:
      idx=np.flatnonzero(tp>=recall*ng) if ng else []
      at[str(recall)]=None if not len(idx) else {'tp':int(tp[idx[0]]),'fp':int(fp[idx[0]]),'score':float(scores[order[idx[0]]])}
     c['ranking'][key]={'gt':ng,'tp':int(tp[-1]) if len(tp) else 0,'fp':int(fp[-1]) if len(fp) else 0,'at_recall':at}
   classes[cname]=c
  for e in ev.evalImgs:
   if not e or e['aRng']!=ev.params.areaRng[1]:continue
   iid,cid=e['image_id'],e['category_id']
   rawgt=ev._gts[iid,cid];rawdt=sorted(ev._dts[iid,cid],key=lambda x:-x['score'])[:1000]
   overlaps=ev.ious[iid,cid]
   for j,gid in enumerate(e['gtIds']):
    if e['gtIgnore'][j]:continue
    gj=next(x for x,a in enumerate(rawgt) if a['id']==gid)
    col=overlaps[:,gj] if len(rawdt) else np.array([]);bi=int(np.argmax(col)) if col.size else None
    best=0. if bi is None else float(col[bi]);matches={}
    for ti in [0,5]:
     did=int(e['gtMatches'][ti,j]);thr=f'{ev.params.iouThrs[ti]:.2f}'
     matches[thr]=dict(matched=bool(did),score=float(dt.anns[did]['score']) if did else None)
    small[gid]={'best_same_iou':best,'best_same_score':float(rawdt[bi]['score']) if bi is not None else None,'matches':matches}
  instances[name]=small;result['models'][name]={'classes':classes};print(name,'DONE',flush=True)
 result['small_transitions']={};rows=[]
 assert set(instances['control'])==set(instances['candidate'])
 for gid,a in instances['control'].items():
  b=instances['candidate'][gid];g=gt.anns[gid]
  rows.append(dict(gt_id=gid,image_id=g['image_id'],file=gt.imgs[g['image_id']]['file_name'],category=gt.cats[g['category_id']]['name'],area=g['area'],control=a,candidate=b))
 for thr in ['0.50','0.75']:
  counts=Counter();byclass={};rescued=[];lost=[];reasons=Counter()
  for row in rows:
   a,b=row['control'],row['candidate'];am,bm=a['matches'][thr]['matched'],b['matches'][thr]['matched'];status='both' if am and bm else 'rescued' if bm else 'lost' if am else 'neither'
   counts[status]+=1;byclass.setdefault(row['category'],Counter())[status]+=1
   if status=='rescued':
    rescued.append(row['gt_id']);reasons['rescued_geometry_crossed' if a['best_same_iou']<float(thr) else 'rescued_matching_or_ranking']+=1
   if status=='lost':
    lost.append(row['gt_id']);reasons['lost_geometry_crossed' if b['best_same_iou']<float(thr) else 'lost_matching_or_ranking']+=1
  for name,field in [('control','lost'),('candidate','rescued')]:
   official=sum(c['ranking']['small_'+thr]['tp'] for c in result['models'][name]['classes'].values())
   assert official==counts['both']+counts[field]
  result['small_transitions'][thr]=dict(counts=counts,by_class=byclass,reasons=reasons,rescued_gt_ids=rescued,lost_gt_ids=lost)
 for name,path in paths.items():assert sha(path)==result['hashes'][name]
 result['passed']=True
 (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'instances.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result['small_transitions'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
