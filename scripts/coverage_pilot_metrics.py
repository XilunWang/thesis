"""Frozen coverage-priority gates and existing ownership/coverage rules."""
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
from scripts.label_only_pilot_metrics import primary,pair_coverage
from scripts.position_grid4_metrics import aggregate,bootstrap

def ownership(root,rows,views):
    viewmap={(r['image_id'],r['seed']):r for r in views['runs']};records=[];total=0
    for r in rows:
        key=(r['image_id'],r['augmentation_seed']);v=viewmap[key]
        d=json.loads((Path(root)/'reports/position-learning-preflight-20261101'/f'holdout-{key[0]}-{key[1]}.json').read_text())
        assert d['input_sha256']==v['input_sha256'] and d['masks_sha256']==v['masks_sha256']
        h,w=d['padded_shape'];m=mu.decode(d['gt_rles']).transpose(2,0,1) if d['gt_rles'] else np.zeros((0,h,w),np.uint8)
        union=m.any(axis=0).reshape(h//4,4,w//4,4).any(axis=(1,3)).ravel()
        count=int((~union[r['selected_cells']]).sum());total+=count
        records.append(dict(image_id=key[0],seed=key[1],no_annotated_gt=count))
    return dict(total=total,records=records)

def compare(root,candidate,adaptive,geometric,initial,plan,views):
    rows={name:primary(d) for name,d in [('candidate',candidate),('adaptive',adaptive),('geometric',geometric),('initial',initial)]}
    keys=lambda rs:[(r['image_id'],r['augmentation_seed'],r['n']) for r in rs]
    assert all(keys(rs)==keys(rows['candidate']) for rs in rows.values())
    summary={k:aggregate(rs) for k,rs in rows.items()}
    pairs={k:pair_coverage(root,rs,views) for k,rs in rows.items()}
    owners={k:ownership(root,rows[k],views) for k in ['candidate','adaptive','geometric']}
    assert owners['adaptive']['total']==4706 and owners['geometric']['total']==3626
    assert summary['adaptive']['occupied']==535 and summary['geometric']['occupied']==400
    a=summary['candidate'];g=plan['gates']
    checks=dict(overall_hits=a['occupied']>=g['overall_hits_min'],fish_hits=a['categories']['1']['hit']>=g['fish_hits_min'],coral_hits=a['categories']['2']['hit']>=g['coral_hits_min'],neighbor_hits=pairs['candidate']['hit']>=g['occupied_pair_hits_min'],no_touch=a['no_touch']<=g['no_touch_max'],redundant=a['redundant']<=g['redundant_max'],no_annotated_gt=owners['candidate']['total']<=g['no_annotated_gt_max'])
    assert a['occupied']+a['redundant']+a['no_touch']==12800
    return dict(position_gates_passed=all(checks.values()),gates=checks,thresholds=g,summaries=summary,pairs=pairs,ownership=owners,paired_bootstrap_vs_adaptive=bootstrap(rows['candidate'],rows['adaptive'],seed=20261301,repeats=1000),paired_bootstrap_vs_geometric=bootstrap(rows['candidate'],rows['geometric'],seed=20261301,repeats=1000))
