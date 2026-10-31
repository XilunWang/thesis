"""Baseline-only error evidence; saved top100 candidates, no learned oracle or training."""
import hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sz(a):return 'small' if a<1024 else 'medium' if a<9216 else 'large'
def main():
    out=ROOT/'reports/baseline-priorities-20261031';out.mkdir(exist_ok=False)
    ann=ROOT/'data/UIIS/annotations/val.json';pred=ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json'
    digest=sha(pred);gt=COCO(str(ann));pp=json.loads(pred.read_text());groups=defaultdict(list)
    for p in pp:groups[p['image_id']].append(p)
    old=json.loads((ROOT/'reports/e1p-final-comparison/instances.json').read_text(encoding='utf-8'))
    old={r['gt_id']:r for r in old};records=[]
    for iid in gt.imgs:
        gg=[a for a in gt.imgToAnns[iid] if not a.get('iscrowd',0)];preds=sorted(groups[iid],key=lambda p:-p['score'])
        ious=mu.iou([p['segmentation'] for p in preds],[gt.annToRLE(a) for a in gg],[0]*len(gg)) if gg else np.zeros((len(preds),0))
        for j,a in enumerate(gg):
            same=[i for i,p in enumerate(preds) if p['category_id']==a['category_id']]
            best=max(same,key=lambda i:ious[i,j]) if same else None
            bs=float(ious[best,j]) if best is not None else 0.;ba=float(ious[:,j].max()) if len(preds) else 0.
            r=old[a['id']];assert abs(bs-r['E0']['best_same_iou'])<1e-10
            row=dict(gt_id=a['id'],image_id=iid,category=gt.cats[a['category_id']]['name'],area=a['area'],size=sz(a['area']),
                     matched=r['E0']['matched'],miss_reason=r['E0']['miss_reason'],best_same_iou=bs,best_any_iou=ba,
                     best_same_score=preds[best]['score'] if best is not None else None,
                     actual_mask_iou=r['E0'].get('mask_iou'),actual_boundary_iou=r['E0'].get('boundary_iou'),
                     over_fraction=r['E0'].get('over_fraction'),under_fraction=r['E0'].get('under_fraction'))
            records.append(row)
    def summarize(rr):
        matched=[r for r in rr if r['matched']]
        thresholds={}
        for t in [.5,.75,.85,.9]:
            thresholds[str(t)]=dict(no_geometric_candidate=sum(r['best_any_iou']<t for r in rr),
                 geometry_only_wrong_class=sum(r['best_any_iou']>=t>r['best_same_iou'] for r in rr),
                 same_class_candidate=sum(r['best_same_iou']>=t for r in rr))
            assert sum(thresholds[str(t)].values())==len(rr)
        return dict(gt=len(rr),matched=len(matched),missed=len(rr)-len(matched),miss_reasons=dict(Counter(r['miss_reason'] for r in rr if not r['matched'])),
             candidate_thresholds=thresholds,
             matched_means={k:float(np.mean([r[k] for r in matched])) if matched else None for k in ['actual_mask_iou','actual_boundary_iou','over_fraction','under_fraction','best_same_iou']},
             matched_with_alternative_iou_gain_01=sum(r['best_same_iou']-r['actual_mask_iou']>=.1 for r in matched),
             matched_below075_with_alternative_above075=sum(r['actual_mask_iou']<.75<=r['best_same_iou'] for r in matched))
    result=dict(scope='Per-GT candidate maxima are non-exclusive GT-assisted diagnostics, NOT AP, unique-query recall or attainable gains. Only saved top100 query-class predictions per image.',
                input_hashes={'annotations':sha(ann),'predictions':digest},
                by_size={s:summarize([r for r in records if r['size']==s]) for s in ['small','medium','large']},
                by_class={n:summarize([r for r in records if r['category']==n]) for n in [c['name'] for c in gt.cats.values()]},
                small_bins={n:summarize([r for r in records if lo<=r['area']<hi]) for n,lo,hi in [('under16x16',0,256),('16to32',256,1024)]})
    train=json.loads((ROOT/'data/UIIS/annotations/train.json').read_text())
    result['train_raw_counts']=dict(Counter(gt.cats[a['category_id']]['name'] for a in train['annotations'] if not a.get('iscrowd',0)))
    result['train_raw_counts_note']='Raw annotation counts, not realized augmented sampling or proof of imbalance causality.'
    assert len(records)==3748 and sha(pred)==digest
    result['passed']=True
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'instances.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
