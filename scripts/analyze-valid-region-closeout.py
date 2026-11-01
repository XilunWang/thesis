"""Frozen prediction closeout; COCO size/category AP and matched GT transitions."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
OUT=ROOT/'reports/valid-region-closeout-20261101'
def mean_valid(x):
 x=x[x>=0];return float(x.mean()*100) if x.size else None
def main():
 assert not (OUT/'result.json').exists(),'preserve completed results'
 gt=COCO(str(ROOT/'data/UIIS/annotations/val.json'));reference=json.loads((ROOT/'reports/valid-region-ap-screen-run/comparison.json').read_text());models={};hashes={}
 for arm in ['control','candidate']:
  path=ROOT/f'reports/valid-region-ap-screen-{arm}/predictions.segm.json';hashes[arm]=hashlib.sha256(path.read_bytes()).hexdigest()
  preds=json.loads(path.read_text());assert len(preds)==69100
  for p in preds:p.pop('bbox',None)
  ev=COCOeval(gt,gt.loadRes(preds),'segm');ev.params.maxDets=[100,300,1000];ev.evaluate();ev.accumulate();ev.summarize()
  assert np.allclose(ev.stats*100,reference['coco'][arm]['stats'],rtol=0,atol=1e-9)
  model={'stats':(ev.stats*100).tolist(),'classes':{},'areas':{}}
  for k,cid in enumerate(ev.params.catIds):model['classes'][gt.cats[cid]['name']]={area:mean_valid(ev.eval['precision'][:,:,k,a,-1]) for a,area in enumerate(ev.params.areaRngLbl)}
  for a,area in enumerate(ev.params.areaRngLbl):
   entries=[x for x in ev.evalImgs if x and x['aRng']==ev.params.areaRng[a]]
   d={'gt':sum(int(np.count_nonzero(~np.asarray(x['gtIgnore'],dtype=bool))) for x in entries),'thresholds':{}}
   for threshold in [.5,.75]:
    t=int(np.argmin(abs(ev.params.iouThrs-threshold)))
    for cutoff in [0.,.5]:
     hits=set();tp=fp=0
     for x in entries:
      selected=(np.asarray(x['dtScores'])>=cutoff)&~x['dtIgnore'][t]
      matches=x['dtMatches'][t][selected];tp+=int(np.count_nonzero(matches));fp+=int(np.count_nonzero(matches==0));hits.update(int(v) for v in matches if v>0)
     assert tp==len(hits)
     d['thresholds'][f'iou{threshold}_score{cutoff}']=dict(tp=tp,fp=fp,miss=d['gt']-tp,matched_gt_ids=sorted(hits))
   model['areas'][area]=d
  models[arm]=model
 transitions={}
 for area in models['control']['areas']:
  transitions[area]={}
  for key,x in models['control']['areas'][area]['thresholds'].items():
   y=models['candidate']['areas'][area]['thresholds'][key];a=set(x['matched_gt_ids']);b=set(y['matched_gt_ids'])
   transitions[area][key]=dict(gained=sorted(b-a),lost=sorted(a-b),common=len(a&b),delta_fp=y['fp']-x['fp'])
 classes=models['control']['classes'];contributions={k:(models['candidate']['classes'][k]['all']-classes[k]['all'])/len(classes) for k in classes}
 assert abs(sum(contributions.values())-(models['candidate']['stats'][0]-models['control']['stats'][0]))<1e-9
 result=dict(models=models,transitions=transitions,class_contribution_to_total_ap=contributions,input_sha256=hashes,notes=['COCO size ranges retain official inclusive boundaries; area totals may overlap at endpoints.','score0 counts all saved outputs, not all pre-topk queries.','Matches are score-ordered one-to-one COCO matches; not geometry oracle or training causality.','Threshold TP/FP counts are diagnostics, not AP.'])
 (OUT/'result.json').write_text(json.dumps(result,indent=2))
 print('CLOSEOUT_COMPLETE',json.dumps(contributions))
if __name__=='__main__':main()
