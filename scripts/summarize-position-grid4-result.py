"""Independently recompute the completed pilot's gates from saved selections/counts."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.local/analysis-libs')]
import numpy as np
from scripts.position_grid4_metrics import aggregate,random_mean,decide,measure
from pycocotools import mask as mu
SRC=ROOT/'reports/position-grid4-execution-20261101'


def main():
    result=json.loads((SRC/'result.json').read_text())
    process=json.loads((SRC/'process-result.json').read_text());assert process['exit_code']==0
    assert result['completed'] and result['step']==300 and result['original_preserved']
    steps=[json.loads(x) for x in (SRC/'steps.jsonl').read_text().splitlines()]
    assert [x['step'] for x in steps]==list(range(1,301))
    assert all(np.isfinite(x['loss']) and np.isfinite(x['gradient_norm']) for x in steps)
    plan=json.loads((ROOT/'reports/position-learning-plan-grid4-20261101/manifest.json').read_text())
    files={s:json.loads((SRC/f'eval-{s}.json').read_text()) for s in [0,100,300]}
    for d in files.values():
        for n,h in d['codes'].items():assert hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest()==h
    def rows(step,source='head',split='holdout',k=100):
        return [r for r in files[step]['records'] if r['source']==source and r['split']==split and r['k']==k]
    # Rebuild primary holdout counts from actual selected cells and unchanged GT masks.
    frozen=json.loads((ROOT/'reports/position-learning-plan-grid4-20261101/views.json').read_text())
    observations={(r['image_id'],r['seed']):r['pairs'] for r in frozen['frozen_neighbor_observations']}
    by_view={(r['image_id'],r['seed']):r for r in frozen['runs'] if r['split']=='holdout'}
    records=rows(0)+rows(300)
    for s in plan['random_control_seeds']:records+=rows(0,f'random_{s}')
    recomputed=0
    for key,v in by_view.items():
        d=json.loads((ROOT/'reports/position-learning-preflight-20261101'/f"holdout-{key[0]}-{key[1]}.json").read_text())
        masks=mu.decode(d['gt_rles']).transpose(2,0,1) if d['gt_rles'] else np.zeros((0,*v['padded_shape']),np.uint8)
        for row in records:
            if (row['image_id'],row['augmentation_seed'])!=key:continue
            actual=measure(row['selected_cells'],v['assignments'],masks,d['annotation_ids'],d['original_areas'],d['labels'],observations.get(key,[]))
            assert all(actual[k]==row[k] for k in actual)
            recomputed+=1
    assert recomputed==128*12
    control=random_mean([rows(0,f'random_{s}') for s in plan['random_control_seeds']])
    rebuilt=decide(rows(300),rows(0),control,plan)
    assert rebuilt['gates']==result['gates'] and rebuilt['intervals']==result['intervals']
    assert rebuilt['position_gates_passed']==result['passed']
    assert all(len(rows(s))==128 and len(rows(s,split='learn'))==256 for s in files)
    summary=dict(passed=result['passed'],steps=300,training_seconds=result['elapsed_training_seconds'],
                 process=process,gates=result['gates'],bootstrap=result['intervals'],
                 random_holdout=aggregate(control),
                 trajectory={str(s):{split:aggregate(rows(s,split=split)) for split in ['learn','holdout']} for s in files},
                 holdout_k_curve={str(s):{str(k):aggregate(rows(s,k=k)) for k in [20,50,100,200]} for s in files},
                 loss_first25=float(np.mean([x['loss'] for x in steps[:25]])),loss_last25=float(np.mean([x['loss'] for x in steps[-25:]])),
                 verified=True,recomputed_primary_holdout_records=recomputed,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/'result.json',SRC/'steps.jsonl',*(SRC/f'eval-{s}.json' for s in files)]})
    (SRC/'local-comparison.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_sha256','holdout_k_curve']},indent=2))


if __name__=='__main__':main()
