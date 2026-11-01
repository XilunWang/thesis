"""Paired ownership of saved K100 candidates; no models or new predictions."""
import collections
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT)]
import numpy as np
import cv2
from pycocotools import mask as mu
from scripts.label_only_pilot_metrics import primary


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    paths={'old':ROOT/'reports/position-grid4-execution-20261101/eval-300.json',
           'new':ROOT/'reports/label-only-pilot-execution-20261101/eval-300.json'}
    oldplan=json.loads((ROOT/'docs/label-only-pilot-plan-20261101.json').read_text())
    newhash=json.loads((ROOT/'reports/label-only-pilot-execution-20261101/artifact-sha256.json').read_text())
    assert sha(paths['old'])==oldplan['control_sha256']['eval-300.json']
    assert sha(paths['new'])==newhash['eval-300.json']
    data={k:{(r['image_id'],r['augmentation_seed']):r for r in primary(json.loads(p.read_text()))} for k,p in paths.items()}
    assert data['old'].keys()==data['new'].keys() and len(data['new'])==128
    frozen=json.loads((ROOT/'reports/position-learning-plan-grid4-20261101/views.json').read_text())
    views={(r['image_id'],r['seed']):r for r in frozen['runs']}
    classes=['fish','coral','aquatic_plants','wrecks_ruins','divers','robots','sea_floor']
    totals={name:collections.Counter() for name in ['old','new','entered','exited','common']}
    categories={name:collections.Counter() for name in totals}
    distance_bins={name:collections.Counter() for name in totals}
    rows=[];cache_hashes={}
    for key in sorted(data['new']):
        path=ROOT/'reports/position-learning-preflight-20261101'/f'holdout-{key[0]}-{key[1]}.json'
        d=json.loads(path.read_text());v=views[key]
        assert d['input_sha256']==v['input_sha256'] and d['masks_sha256']==v['masks_sha256']
        cache_hashes[path.name]=sha(path)
        h,w=d['padded_shape'];gh,gw=h//4,w//4;vh,vw=d['valid_shape'][:2]
        masks=mu.decode(d['gt_rles']).transpose(2,0,1) if d['gt_rles'] else np.zeros((0,h,w),dtype=np.uint8)
        areas=np.asarray(d['original_areas']);labels=np.asarray(d['labels']);ids=np.asarray(d['annotation_ids'])
        assert len(areas)==len(masks) and (areas>0).all()
        support=masks.reshape(len(masks),gh,4,gw,4).any(axis=(2,4)).reshape(len(masks),gh*gw)
        union=masks.any(axis=0)
        distance=cv2.distanceTransform((~union).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if union.any() else None
        selected={k:set(data[k][key]['selected_cells']) for k in ['old','new']}
        assert all(len(s)==100 for s in selected.values())
        rank={k:{c:i+1 for i,c in enumerate(data[k][key]['selected_cells'])} for k in ['old','new']}
        groups={'old':selected['old'],'new':selected['new'],'entered':selected['new']-selected['old'],
                'exited':selected['old']-selected['new'],'common':selected['new']&selected['old']}
        cellinfo={}
        for cell in sorted(selected['new']|selected['old']):
            assert cell//gw*4<vh and cell%gw*4<vw
            touched=np.flatnonzero(support[:,cell]);a=areas[touched]
            if (a<=1024).any():size='small_touch'
            elif len(a)==0:size='no_annotated_GT'
            elif (a<=9216).all():size='medium_only'
            elif (a>9216).all():size='large_only'
            else:size='medium_large_mixed'
            labs=set(map(int,labels[touched]))
            category='no_annotated_GT' if not labs else classes[next(iter(labs))] if len(labs)==1 else 'mixed_categories'
            x,y=(cell%gw)*4,(cell//gw)*4
            cx=min(int((x+min(x+4,vw))/2),vw-1);cy=min(int((y+min(y+4,vh))/2),vh-1)
            center_touches=[int(ids[i]) for i in np.flatnonzero(masks[:,cy,cx])]
            dist=float(distance[cy,cx]) if size=='no_annotated_GT' and distance is not None else None
            band=('image_has_no_GT' if distance is None else 'within4px' if dist<=4 else '4to8px' if dist<=8 else 'beyond8px') if size=='no_annotated_GT' else None
            cellinfo[cell]=dict(cell=cell,size=size,category=category,gt_ids=ids[touched].tolist(),
                center_gt_ids=center_touches,nearest_GT_pixel_distance=dist,distance_band=band,
                old_rank=rank['old'].get(cell),new_rank=rank['new'].get(cell))
        perview={}
        for name,cells in groups.items():
            counts=collections.Counter(cellinfo[c]['size'] for c in cells)
            totals[name].update(counts)
            cats=collections.Counter(cellinfo[c]['category'] for c in cells if cellinfo[c]['size']!='small_touch')
            categories[name].update(cats)
            distance_bins[name].update(cellinfo[c]['distance_band'] for c in cells if cellinfo[c]['size']=='no_annotated_GT')
            perview[name]=dict(counts)
            if name in ['old','new']:
                assert len(cells)-counts['small_touch']==data[name][key]['no_touch']
        rows.append(dict(image_id=key[0],seed=key[1],small_instances=int((areas<=1024).sum()),
                         counts=perview,cells=list(cellinfo.values())))
    sizes=['small_touch','medium_only','large_only','medium_large_mixed','no_annotated_GT']
    deltas={k:totals['new'][k]-totals['old'][k] for k in sizes}
    for k in sizes:
        assert deltas[k]==totals['entered'][k]-totals['exited'][k]
        assert totals['old'][k]==totals['common'][k]+totals['exited'][k]
        assert totals['new'][k]==totals['common'][k]+totals['entered'][k]
    assert sum(totals['old'].values())==sum(totals['new'].values())==12800
    assert sum(deltas[k] for k in sizes if k!='small_touch')==1456
    strata={}
    for label,empty in [('no_small',True),('has_small',False)]:
        selected=[r for r in rows if (r['small_instances']==0)==empty]
        strata[label]=dict(views=len(selected),counts={name:dict(sum((collections.Counter(r['counts'][name]) for r in selected),collections.Counter())) for name in totals})
    out=ROOT/'reports/label-only-no-touch-20261101';out.mkdir(exist_ok=True)
    result=dict(passed=True,training_updates=0,views=128,totals=totals,delta=deltas,
        no_small_touch_categories=categories,unannotated_distance_bands=distance_bins,strata=strata,records=rows,
        source_sha256={k:sha(p) for k,p in paths.items()},cache_sha256=cache_hashes,script_sha256=sha(Path(__file__)),
        limitation='Ownership uses any GT pixel in a stride4 cell. Original-area size bins are exclusive <=1024, (1024,9216], >9216, not COCO AP recomputation. No annotation is not confirmed semantic background; distance is augmented pixels at candidate centers. Net increase is arrivals minus departures, not 1456 uniquely identifiable new points.')
    (out/'result.json').write_text(json.dumps(result,indent=2))
    for name in ['totals','delta','no_small_touch_categories','unannotated_distance_bands','strata']:print(name,json.dumps(result[name]))


if __name__=='__main__':main()
