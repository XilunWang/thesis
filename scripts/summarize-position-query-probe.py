"""Independently recompute matches and apply the preregistered stop rule."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.local/analysis-libs'), str(ROOT)]
import numpy as np
from pycocotools import mask as mu

OUT = ROOT / 'reports/position-query-probe-20261101'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('matching', ROOT / 'scripts/analyze-fixed12-e0.py')
    matching = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matching)
    proof = json.loads((OUT / 'attempt-2/result.json').read_text())
    assert proof['passed'] and proof['model_state_unchanged'] and not proof['optimizer_created']
    assert proof['script_sha256'] == sha(ROOT / 'scripts/probe-position-query.py')
    records = []
    for item in proof['runs']:
        path = OUT / 'attempt-2' / item['file']
        assert sha(path) == item['sha256']
        r = json.loads(path.read_text())
        original = json.loads((ROOT / f"reports/fixed12-e0-complementarity-20261101/attempt-2/case-{r['case_id']:02d}.json").read_text())
        assert r['input_sha256'] == original['input_sha256'] and r['masks_sha256'] == original['masks_sha256']
        qiou = mu.iou(r['query_rles'], original['gt_rles'], [0] * len(original['labels']))
        piou = mu.iou(r['prediction_rles'], original['gt_rles'], [0] * len(original['labels']))
        matched = matching.detection_match(piou, r['prediction_labels'], r['prediction_scores'], original['labels'], original['annotation_ids'])
        assert r['detected_gt'] == [original['annotation_ids'][j] for j in matched]
        for j, pergt in enumerate(r['per_gt']):
            assert pergt['best_iou'] == float(qiou[:, j].max()) and pergt['detected'] == (j in matched)
        assert r['unmatched_predictions_above_05'] == sum(s >= .5 for s in r['prediction_scores']) - len(matched)
        if r['branch'] == 'baseline':
            assert r['query_rles'] == original['query_rles']
            assert r['class_probabilities'] == original['class_probabilities']
            assert r['prediction_scores'] == original['selected_scores']
        records.append(r)
    assert len(records) == 23
    rows = []
    for r in records:
        if r['branch'] != 'hint':
            continue
        gid = r['target_gt']
        baseline = next(b for b in records if b['case_id'] == r['case_id'] and b['branch'] == 'baseline')
        controls = [c for c in records if c['target_gt'] == gid and c['branch'] not in ['hint', 'baseline']]
        target = lambda x: next(g for g in x['per_gt'] if g['gt_id'] == gid)
        specificity = (target(r)['detected'] and len(controls) == 4
                       and all(c['window']['valid_control'] and not target(c)['detected'] for c in controls))
        rows.append(dict(gt_id=gid, image_id=r['image_id'], baseline=target(baseline), hint=target(r),
                         controls={c['branch']:dict(**target(c), valid=c['window']['valid_control']) for c in controls},
                         location_specific_detection=specificity,
                         hint_gained_gt=sorted(set(r['detected_gt'])-set(baseline['detected_gt'])),
                         hint_lost_gt=sorted(set(baseline['detected_gt'])-set(r['detected_gt'])),
                         unmatched_predictions_baseline=baseline['unmatched_predictions_above_05'],
                         unmatched_predictions_hint=r['unmatched_predictions_above_05']))
    assert len(rows) == 4
    geometry = sum(r['hint']['best_iou'] >= .5 for r in rows)
    detected = sum(r['hint']['detected'] for r in rows)
    specific = sum(r['location_specific_detection'] for r in rows)
    result = dict(verified=True, training_updates=0, paths=23, rows=rows,
                  qualified_geometry=geometry, qualified_detections=detected, specific_detections=specific,
                  continue_gate=geometry >= 2 and detected >= 1 and specific >= 1,
                  timing_seconds={branch:dict(mean=float(np.mean([r['seconds'] for r in records if r['branch']==branch])),
                                               n=sum(r['branch']==branch for r in records)) for branch in ['baseline','hint','left','right','up','down']},
                  peak_memory_bytes=max(r['peak_memory_bytes'] for r in records),
                  source_sha256=sha(OUT / 'attempt-2/result.json'))
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
