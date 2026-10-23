"""Post-primary exploratory separation of score ordering and best stored mask quality."""
import importlib.util,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu
R=Path(__file__).resolve().parents[1];O=R/'reports/e1-offline-comparison'
spec=importlib.util.spec_from_file_location('cmp',R/'scripts/compare-e1-offline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
coco=COCO(str(R/'data/UIIS/annotations/val.json'));rows=json.loads((O/'instances.json').read_text(encoding='utf-8'));groups={}
for label,folder in [('E0','uiis-baseline-36e-evaluation'),('E1','uiis-e1-bll-36e-evaluation')]:
 g=defaultdict(list)
 for p in json.loads((R/'reports'/folder/'predictions.segm.json').read_text()):g[p['image_id']].append(p)
 groups[label]={i:sorted(v,key=lambda p:-p['score']) for i,v in g.items()}
common=[r for r in rows if r['size']=='large' and r['E0']['matched'] and r['E1']['matched']];detail=[]
for r in common:
 gm=coco.annToMask(coco.anns[r['gt_id']]).astype(bool);gb=m.band(gm);item=dict(gt_id=r['gt_id'],category=r['category'])
 for label in groups:
  d=r[label];p=groups[label][r['image_id']][d['best_same_index']];pm=mu.decode(p['segmentation']).astype(bool)
  item[label]=dict(mask_iou=d['best_same_iou'],boundary_iou=m.iou(gb,m.band(pm)),score=p['score'],different_from_selected=d['best_same_index']!=d['pred_index'])
 detail.append(item)
result=dict(exploratory=True,not_AP=True,limitation='Oracle best Mask IoU among saved same-class predictions without score cutoff or one-to-one uniqueness; uses GT, cannot deploy or replace primary comparison.',n=len(detail),means={label:{key:float(np.mean([d[label][key] for d in detail])) for key in ['mask_iou','boundary_iou','score']} for label in groups},different_from_selected={label:sum(d[label]['different_from_selected'] for d in detail) for label in groups},details=detail)
(O/'ranking-probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
for label in groups:print(label,result['means'][label],result['different_from_selected'][label])
for c in json.loads((O/'cases.json').read_text(encoding='utf-8')):
 r=c['instance'];print(c['type'],r['gt_id'],[(label,groups[label][r['image_id']][r[label]['best_same_index']]['score']) for label in groups])
