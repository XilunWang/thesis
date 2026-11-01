"""Explain saved matching deficits; do not change the labels or rerun augmentation."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools import mask as mu
spec=importlib.util.spec_from_file_location('assignment',ROOT/'scripts/audit-query-cell-assignment.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
SRC=ROOT/'reports/position-learning-preflight-20261101'


def main():
    proof=json.loads((SRC/'result.json').read_text())
    raw=json.loads((ROOT/'data/UIIS/annotations/train.json').read_text())
    annotations={a['id']:a for a in raw['annotations']};images={i['id']:i for i in raw['images']}
    cases=[]
    for e in proof['errors']:
        p=SRC/f"{e['split']}-{e['image_id']}-{e['seed']}.json"
        d=json.loads(p.read_text());assert 'Occupied-cell assignment deficit' in e['error']
        masks=mu.decode(d['gt_rles']).transpose(2,0,1);h,w=masks.shape[-2:]
        candidates={};byid={}
        for gid,area,mask in zip(d['annotation_ids'],d['original_areas'],masks):
            if not 0<area<=1024:continue
            yy,xx=np.nonzero(mask);j=np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2)
            cx,cy=int(xx[j])//8,int(yy[j])//8
            counts=mask.reshape(h//8,8,w//8,8).sum((1,3))
            candidates[gid]=sorted(map(int,np.flatnonzero(counts)),key=lambda c:((c//(w//8)-cy)**2+(c%(w//8)-cx)**2,-int(counts.flat[c]),c))
            byid[gid]=mask
        matched,witness=module.assign(candidates);assert witness['deficit']>0
        targets=[]
        for gid in witness['gt_ids']:
            yy,xx=np.nonzero(byid[gid]);a=annotations[gid]
            targets.append(dict(annotation_id=gid,category_id=a['category_id'],original_area=a['area'],
                                augmented_pixels=int(byid[gid].sum()),augmented_bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                                occupied_cells_xy=[[c%(w//8),c//(w//8)] for c in candidates[gid]]))
        overlaps=[]
        for j,a in enumerate(witness['gt_ids']):
            for b in witness['gt_ids'][j+1:]:
                overlaps.append(dict(a=a,b=b,mask_overlap_pixels=int(np.count_nonzero(byid[a]&byid[b]))))
        cases.append(dict(image_id=e['image_id'],file_name=images[e['image_id']]['file_name'],seed=e['seed'],
                          valid_shape=d['valid_shape'],small_targets=len(candidates),maximum_assignable=len(matched),
                          witness=witness,targets=targets,pair_overlaps=overlaps,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    out=dict(training_updates=0,cases=cases,scope='Exact occupied-cell capacity witnesses from saved failed augmentations; no label or model modification')
    (SRC/'failure-explanation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
