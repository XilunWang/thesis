"""Paired fixed-sample diagnostics, not full test AP."""
import hashlib,importlib.util,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu
spec=importlib.util.spec_from_file_location('matcher',ROOT/'scripts/analyze-uiis-errors.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def size(a):return 'tiny' if a<256 else 'small' if a<1024 else 'medium' if a<9216 else 'large'
def main():
    out=ROOT/'reports/baseline-scale-pilot-20261031';sel=json.loads((out/'selection.json').read_text());run=json.loads((out/'run-result.json').read_text());assert run['passed']
    gt=COCO(str(ROOT/'data/UIIS/annotations/val.json'));records={};totals={};hashes={}
    for model in ['original','larger']:
        path=out/f'{model}-predictions.json';hashes[model]=hashlib.sha256(path.read_bytes()).hexdigest();pp=json.loads(path.read_text());groups=defaultdict(list)
        for p in pp:groups[p['image_id']].append(p)
        assert set(groups)==set(sel['image_ids']) and all(len(p)==100 for p in groups.values())
        totals[model]={};rr=[]
        for iid in sel['image_ids']:
            preds=sorted(groups[iid],key=lambda p:-p['score']);gg=[a for a in gt.imgToAnns[iid] if not a.get('iscrowd',0)];crowd=[a for a in gt.imgToAnns[iid] if a.get('iscrowd',0)]
            iou=mu.iou([p['segmentation'] for p in preds],[gt.annToRLE(a) for a in gg],[0]*len(gg)) if gg else np.zeros((len(preds),0))
            ci=mu.iou([p['segmentation'] for p in preds],[gt.annToRLE(a) for a in crowd],[1]*len(crowd)) if crowd else None
            matches={}
            for threshold in [.3,.5,.7]:
                for it in [.5,.75]:
                    key=f'score{threshold}_iou{it}';totals[model].setdefault(key,Counter());c=totals[model][key]
                    n=sum(p['score']>=threshold for p in preds);labs,matched,used=m.match(iou[:n],[p['category_id'] for p in preds[:n]],[g['category_id'] for g in gg],it)
                    for k,l in enumerate(labs):
                        if l!='matched' and ci is not None and any(g['category_id']==preds[k]['category_id'] and ci[k,j]>=.5 for j,g in enumerate(crowd)):labs[k]='ignored_crowd'
                    c.update(labs);c['fp']+=sum(l not in ['matched','ignored_crowd'] for l in labs)
                    c['predictions']+=n
                    for j,a in enumerate(gg):c[size(a['area'])+'_gt']+=1;c[size(a['area'])+'_matched']+=int(j in used)
                    matches[key]={j:k for k,j in enumerate(matched) if j is not None}
            for j,a in enumerate(gg):
                same=[k for k,p in enumerate(preds) if p['category_id']==a['category_id']]
                rr.append(dict(gt_id=a['id'],image_id=iid,category=gt.cats[a['category_id']]['name'],size=size(a['area']),
                    best_any=float(iou[:,j].max()),best_same=float(iou[same,j].max()) if same else 0.,
                    matched={key:j in v for key,v in matches.items()},
                    mask_iou=float(iou[matches['score0.5_iou0.5'][j],j]) if j in matches['score0.5_iou0.5'] else None))
        records[model]={r['gt_id']:r for r in rr}
    assert records['original'].keys()==records['larger'].keys()
    paired={}
    for s in ['tiny','small','medium','large','all']:
        ids=[i for i,r in records['original'].items() if s=='all' or r['size']==s];a=records['original'];b=records['larger'];key='score0.5_iou0.5'
        common=[i for i in ids if a[i]['matched'][key] and b[i]['matched'][key]]
        paired[s]=dict(gt=len(ids),candidate_coverage={v:{model:sum(records[model][i][v]>=t for i in ids) for model in records} for v,t in [('best_any',.5),('best_same',.5)]},
            same_candidate075={model:sum(records[model][i]['best_same']>=.75 for i in ids) for model in records},
            new_matched=sum(not a[i]['matched'][key] and b[i]['matched'][key] for i in ids),lost_matched=sum(a[i]['matched'][key] and not b[i]['matched'][key] for i in ids),
            recovered_missing_geometry=sum(a[i]['best_any']<.5<=b[i]['best_any'] for i in ids),lost_geometry=sum(b[i]['best_any']<.5<=a[i]['best_any'] for i in ids),
            common=len(common),common_mask_iou={model:float(np.mean([records[model][i]['mask_iou'] for i in common])) if common else None for model in records})
    perf={model:dict(mean_seconds=float(np.mean([r['seconds'] for r in run['measurements'] if r['scale']==model])),max_allocated_gib=max(r['peak_allocated_bytes'] for r in run['measurements'] if r['scale']==model)/1024**3) for model in records}
    result=dict(passed=True,totals=totals,paired=paired,performance=perf,prediction_sha256=hashes,scope=sel['selection_note'])
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(out/'instances.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
