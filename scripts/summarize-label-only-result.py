"""Verify downloaded bounded run and recompute its primary held-out metrics."""
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT)]
import numpy as np
from pycocotools import mask as mu
from scripts.position_grid4_metrics import measure
from scripts.label_only_pilot_metrics import compare,primary


def main():
    out=ROOT/'reports/label-only-pilot-execution-20261101'
    hashes=json.loads((out/'artifact-sha256.json').read_text())
    for n,h in hashes.items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h
    status=json.loads((out/'process-result.json').read_text())
    result=json.loads((out/'result.json').read_text())
    steps=[json.loads(s) for s in (out/'steps.jsonl').read_text().splitlines()]
    assert status['exit_code']==0 and result['completed'] and result['step']==300 and result['original_preserved']
    assert [r['step'] for r in steps]==list(range(1,301))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm']) for r in steps)
    assert all(a['elapsed_seconds']<=b['elapsed_seconds'] for a,b in zip(steps,steps[1:]))
    initial=json.loads((out/'eval-0.json').read_text())
    old_initial=json.loads((ROOT/'reports/position-grid4-execution-20261101/eval-0.json').read_text())
    assert initial['records']==old_initial['records']
    del old_initial
    final=json.loads((out/'eval-300.json').read_text())
    old=json.loads((ROOT/'reports/position-grid4-execution-20261101/eval-300.json').read_text())
    views=json.loads((ROOT/'reports/position-learning-plan-grid4-20261101/views.json').read_text())
    lookup={(r['image_id'],r['seed']):r for r in views['runs']}
    neighbors={(r['image_id'],r['seed']):r['pairs'] for r in views['frozen_neighbor_observations']}
    checked=0
    for row in primary(final):
        key=(row['image_id'],row['augmentation_seed']);v=lookup[key]
        d=json.loads((ROOT/'reports/position-learning-preflight-20261101'/f'holdout-{key[0]}-{key[1]}.json').read_text())
        assert d['input_sha256']==v['input_sha256'] and d['masks_sha256']==v['masks_sha256']
        masks=mu.decode(d['gt_rles']).transpose(2,0,1)
        values=measure(row['selected_cells'],v['assignments'],masks,d['annotation_ids'],d['original_areas'],d['labels'],neighbors.get(key,[]))
        for k,value in values.items():assert row[k]==value,(key,k)
        checked+=1
    assert checked==128
    plan=json.loads((ROOT/'docs/label-only-pilot-plan-20261101.json').read_text())
    local=compare(ROOT,final,old,initial,plan,views)
    for k in ['gates','candidate','control','initial','candidate_pairs','control_pairs']:
        assert local[k]==result[k],k
    assert np.allclose(local['paired_bootstrap']['interval95'],result['paired_bootstrap']['interval95'],atol=1e-12,rtol=0)
    candidate,control=result['candidate'],result['control']
    for values in [candidate,control]:
        assert values['occupied']+values['redundant']+values['no_touch']==values['selected']==12800
    changes={k:candidate[k]-control[k] for k in ['occupied','redundant','no_touch']}
    assert sum(changes.values())==0
    trajectories={}
    for step in [0,100,300]:
        data=json.loads((out/f'eval-{step}.json').read_text())
        trajectories[str(step)]=data['summaries']['holdout/head/100']
    wall=(datetime.fromisoformat(status['finished_at'])-datetime.fromisoformat(status['started_at'])).total_seconds()
    proof=dict(verification_passed=True,experiment_gates_passed=result['position_gates_passed'],
        exit_code=0,updates=300,wall_seconds=wall,training_seconds=result['elapsed_training_seconds'],
        verified_artifacts=len(hashes),step0_records_exact=True,recomputed_primary_views=checked,
        nonfinite_training_values=0,loss_range=[min(x['loss'] for x in steps),max(x['loss'] for x in steps)],
        gradient_norm_range=[min(x['gradient_norm'] for x in steps),max(x['gradient_norm'] for x in steps)],
        comparison=local,changes=changes,trajectories=trajectories,
        interpretation='Fixed budget identity: more no-touch-small-GT cells partly replace redundant cells. Not a direct semantic-background/segmentation-FP count. Gate remains failed; do not train more.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'local-verification.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:v for k,v in proof.items() if k not in ['comparison','trajectories']},indent=2))
    print('gates',local['gates'])
    print('bootstrap',local['paired_bootstrap'])
    for step,a in trajectories.items():print('step',step,'coverage',a['occupied'],'fish',a['categories']['1']['hit'],'coral',a['categories']['2']['hit'])


if __name__=='__main__':main()
