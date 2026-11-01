"""Hand-constructed matching, accounting and decision tests; no learned data."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.local/analysis-libs')]
import numpy as np
from scripts.position_grid4_metrics import measure,aggregate,random_mean,decide


def main():
    masks=np.zeros((2,8,12),np.uint8)
    masks[0,0,0]=masks[0,0,4]=1;masks[1,1,0]=1
    assignment={'10':[1,0],'20':[0,0]}
    r=measure([0],assignment,masks,[10,20],[2,1],[0,1],[[10,20]])
    assert r['n']==2 and r['occupied']==1 and r['seed']==1 and r['pair_hit']==0
    r=measure([0,1,5],assignment,masks,[10,20],[2,1],[0,1],[[10,20]])
    assert r['occupied']==r['seed']==2 and r['pair_hit']==1 and r['no_touch']==1 and r['redundant']==0
    assert r['categories']['1']['hit']==r['categories']['2']['hit']==1
    dup=measure([0,1],{'10':[0,0]},masks[:1],[10],[2],[0],[])
    assert dup['occupied']==1 and dup['redundant']==1
    empty=measure([0],{},np.zeros((0,8,12),np.uint8),[],[],[],[])
    assert empty['n']==empty['occupied']==0 and empty['no_touch']==1
    rows=[]
    for iid in [1,2,3]:
        for seed in [1,2]:rows.append(dict(image_id=iid,augmentation_seed=seed,split='holdout',**r))
    low=[]
    for row in rows:
        x=json.loads(json.dumps(row));x.update(occupied=0,seed=0,pair_hit=0)
        for c in x['categories'].values():c['hit']=0
        low.append(x)
    mean=random_mean([low,rows]);assert aggregate(mean)['occupied_recall']==.5
    p=dict(bootstrap_seed=1,bootstrap_repeats=1000)
    assert decide(rows,low,low,p)['position_gates_passed']
    assert not decide(low,low,low,p)['position_gates_passed']
    assert not decide(rows,rows,rows,p)['position_gates_passed']
    out=ROOT/'reports/position-grid4-metrics-check-20261101';out.mkdir(exist_ok=False)
    (out/'result.json').write_text(json.dumps(dict(passed=True,training_updates=0,tests=['one_to_one','augmenting_path','category_counts','empty','redundant','random_average','positive_and_negative_gate_controls']),indent=2))
    print('METRICS_CHECK_PASSED')


if __name__=='__main__':main()
