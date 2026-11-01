"""Frozen old metrics plus paired occupied-support coverage, for BOTH arms."""
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
from scripts.position_grid4_metrics import aggregate,bootstrap


def primary(data):
    return [r for r in data['records'] if r['split']=='holdout' and r['source']=='head' and r['k']==100]


def pair_coverage(root, rows, views):
    observations=views['frozen_neighbor_observations']
    lookup={(r['image_id'],r['augmentation_seed']):r for r in rows}
    details=[]
    for ob in observations:
        key=(ob['image_id'],ob['seed']);row=lookup[key]
        p=Path(root)/'reports/position-learning-preflight-20261101'/f'holdout-{key[0]}-{key[1]}.json'
        d=json.loads(p.read_text());h,w=d['padded_shape']
        masks=mu.decode(d['gt_rles']).transpose(2,0,1)
        chosen=set(row['selected_cells'])
        support={int(g):set(map(int,np.flatnonzero(m.reshape(h//4,4,w//4,4).any(axis=(1,3)))))
                 for g,a,m in zip(d['annotation_ids'],d['original_areas'],masks) if 0<a<=1024}
        owner={}
        def visit(g,seen):
            for cell in sorted(support[g]&chosen):
                if cell in seen:continue
                seen.add(cell)
                if cell not in owner or visit(owner[cell],seen):owner[cell]=g;return True
            return False
        for g in sorted(support):visit(g,set())
        matched=set(owner.values())
        assert len(matched)==row['occupied']  # Same matching as original measure().
        for a,b in ob['pairs']:
            details.append(dict(image_id=key[0],seed=key[1],ids=[a,b],hit=a in matched and b in matched))
    return dict(n=len(details),hit=sum(r['hit'] for r in details),details=details)


def compare(root,candidate,control,initial,plan,views):
    rows=primary(candidate);old=primary(control);start=primary(initial)
    assert [(r['image_id'],r['augmentation_seed'],r['n']) for r in rows]==[(r['image_id'],r['augmentation_seed'],r['n']) for r in old]
    a,b,c=aggregate(rows),aggregate(old),aggregate(start)
    pairs=pair_coverage(root,rows,views);oldpairs=pair_coverage(root,old,views)
    gates=dict(overall_no_regression=a['occupied']>=b['occupied'],
        fish_no_regression=a['categories']['1']['hit']>=b['categories']['1']['hit'],
        coral_protection=a['categories']['2']['hit']>=plan['minimum_coral_hits'],
        occupied_pairs_no_regression=pairs['hit']>=oldpairs['hit'],
        no_touch_no_increase=a['no_touch']<=b['no_touch'],
        redundant_no_increase=a['redundant']<=b['redundant'])
    return dict(position_gates_passed=all(gates.values()),gates=gates,candidate=a,control=b,initial=c,
        candidate_pairs=pairs,control_pairs=oldpairs,
        paired_bootstrap=bootstrap(rows,old,seed=20261301,repeats=1000),
        note='Fixed geometric seed metrics remain reported but are not the candidate selection objective; occupied pair metric applied identically to both arms. Bootstrap descriptive, no threshold tuning.')
