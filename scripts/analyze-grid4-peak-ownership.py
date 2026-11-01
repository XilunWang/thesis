"""Offline ownership/distance audit; no model, training or candidate-rule changes."""
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.local/analysis-libs'))
import numpy as np
from pycocotools import mask as mu

SRC = ROOT / 'reports/position-grid4-coral-scores-20261101'
OUT = ROOT / 'reports/position-grid4-peak-ownership-20261101'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), median=float(np.median(a)) if len(a) else None,
                min=float(a.min()) if len(a) else None,
                max=float(a.max()) if len(a) else None,
                within4=int((a <= 4).sum()), within8=int((a <= 8).sum()))


def main():
    proof = json.loads((SRC / 'result.json').read_text())
    prior = json.loads((SRC / 'analysis.json').read_text())
    assert proof['passed'] and proof['training_updates'] == 0
    rows = {(r['image_id'], r['seed'], r['annotation_id']): r for r in prior['instances']}
    frozen = ROOT / 'reports/position-learning-plan-grid4-20261101/views.json'
    views = {(r['image_id'], r['seed']): r for r in json.loads(frozen.read_text())['runs']}
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [SRC/'result.json', SRC/'analysis.json', frozen]}
    output_views, output_targets = [], []
    for run in proof['runs']:
        path = SRC / run['file']
        assert sha(path) == run['sha256']
        key = (run['image_id'], run['seed'])
        v = views[key]
        cache = ROOT / 'reports/position-learning-preflight-20261101' / f'holdout-{key[0]}-{key[1]}.json'
        sources[str(cache.relative_to(ROOT))] = sha(cache)
        d = json.loads(cache.read_text())
        assert d['input_sha256'] == v['input_sha256'] and d['masks_sha256'] == v['masks_sha256']
        masks = mu.decode(d['gt_rles']).transpose(2, 0, 1)
        h, w = d['padded_shape']; gh, gw = h//4, w//4
        vh, vw = d['valid_shape'][:2]
        support = masks.reshape(len(masks), gh, 4, gw, 4).any(axis=(2, 4)).reshape(len(masks), -1)
        arrays = np.load(path, allow_pickle=False)
        valid = arrays['valid'].ravel(); cells = np.flatnonzero(valid)
        ids = list(map(int, d['annotation_ids']))
        labels = np.asarray(d['labels']); areas = np.asarray(d['original_areas'])
        assignments = {int(g): int(y)*gw+int(x) for g, (x, y) in v['assignments'].items()}
        seed_owner = {c:g for g,c in assignments.items()}
        assert len(seed_owner) == len(assignments)
        assert set(assignments) == {g for g, a in zip(ids, areas) if 0 < a <= 1024}
        for g, c in assignments.items():
            assert support[ids.index(g), c] and valid[c]
        def xy(c):
            x, y = (c % gw)*4, (c//gw)*4
            return np.array([(x+min(x+4,vw))/2, (y+min(y+4,vh))/2])
        def distance(a, b):
            return float(np.linalg.norm(xy(a)-xy(b)))
        def kind(c, center=False):
            if center:
                x, y = xy(c).astype(int)
                touched = np.flatnonzero(masks[:, min(y,vh-1), min(x,vw-1)])
            else:
                touched = np.flatnonzero(support[:, c])
            categories = set(map(int, labels[touched]))
            if not categories: return 'no_annotated_GT'
            if len(categories) > 1: return 'mixed_classes'
            return {0:'fish', 1:'coral'}.get(next(iter(categories)), 'other_GT')
        def describe(c, score):
            touched = np.flatnonzero(support[:,c])
            own = [ids[i] for i in touched if ids[i] in assignments]
            nearest = min(assignments, key=lambda g:(distance(c,assignments[g]),g)) if assignments else None
            return dict(cell=int(c), logit=float(score[c]), kind=kind(c), center_kind=kind(c,True),
                        gt_ids=[ids[i] for i in touched], small_gt_ids=own,
                        positive_seed_gt=seed_owner.get(int(c)),
                        own_seed_distances_px={str(g):distance(c,assignments[g]) for g in own},
                        nearest_seed_gt=nearest,
                        nearest_seed_distance_px=distance(c,assignments[nearest]) if nearest is not None else None)
        vr = dict(image_id=key[0], seed=key[1], stages={})
        for stage in [0,300]:
            scores = arrays[f'logits_{stage}'].ravel().astype(float)
            assert np.isfinite(scores[cells]).all()
            order = cells[np.lexsort((cells,-scores[cells]))]
            # Local maxima are diagnostic only; deterministic row-major tie break.
            rank = np.full(len(valid), len(valid)+1, dtype=int)
            rank[order] = np.arange(len(order))
            padded = np.pad(rank.reshape(gh,gw), 1, constant_values=len(valid)+1)
            is_peak = valid.copy()
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dy or dx:
                        is_peak &= rank < padded[1+dy:1+dy+gh,1+dx:1+dx+gw].ravel()
            selected = order[:100]
            vr['stages'][str(stage)] = dict(top100=[describe(c,scores) for c in selected],
                top100_local_peak_cells=[int(c) for c in selected if is_peak[c]],
                local_peaks_top100=[describe(c,scores) for c in order[is_peak[order]][:100]])
            for i,(gid,label,area) in enumerate(zip(ids,labels,areas)):
                if not(label==1 and 0<area<=1024):continue
                old = rows[(*key,gid)]['stages'][str(stage)]
                best = int(old['best_cell']); own_seed = assignments[gid]
                assert support[i,best] and rank[best]+1 == old['best_rank']
                external = order[~support[i,order]][:100]
                assert len(external)==100
                assert np.isclose(scores[external[-1]],old['external_top100_threshold'],rtol=0,atol=1e-8)
                record = dict(image_id=key[0],seed=key[1],annotation_id=gid,stage=stage,
                    transition=rows[(*key,gid)]['transition'],support_cells=int(support[i].sum()),
                    augmented_area=int(masks[i].sum()),seed_cell=own_seed,best_cell=best,
                    best_rank=int(rank[best]+1),seed_rank=int(rank[own_seed]+1),
                    best_to_seed_px=distance(best,own_seed),
                    best_to_seed_over_sqrt_area=distance(best,own_seed)/np.sqrt(masks[i].sum()),
                    best_is_positive=best in seed_owner,best_is_own_seed=best==own_seed,
                    best_positive_gt=seed_owner.get(best),seed_to_best_logit_gap=float(scores[best]-scores[own_seed]),
                    external_top100=[describe(c,scores) for c in external])
                output_targets.append(record)
        output_views.append(vr)
    summary = {}
    for stage in [0,300]:
        picks = [c for v in output_views for c in v['stages'][str(stage)]['top100']]
        peaks = [c for v in output_views for c in v['stages'][str(stage)]['local_peaks_top100']]
        owned = [c for c in picks if len(c['gt_ids'])==1 and len(c['small_gt_ids'])==1]
        bg = [c for c in picks if c['kind']=='no_annotated_GT']
        summary[str(stage)] = dict(top100_count=len(picks),
            top100_kinds=dict(collections.Counter(c['kind'] for c in picks)),
            top100_center_kinds=dict(collections.Counter(c['center_kind'] for c in picks)),
            local_peaks_top100_kinds=dict(collections.Counter(c['kind'] for c in peaks)),
            selected_local_peak_count=sum(len(v['stages'][str(stage)]['top100_local_peak_cells']) for v in output_views),
            positive_seed_count=sum(c['positive_seed_gt'] is not None for c in picks),
            own_seed_distance_by_kind={k:stats([next(iter(c['own_seed_distances_px'].values())) for c in owned if c['kind']==k]) for k in ['fish','coral','other_GT']},
            no_GT_nearest_seed_distance=stats([c['nearest_seed_distance_px'] for c in bg]),
            coral_by_transition={})
        summary[str(stage)]['unambiguous_small_instance_repetition'] = {}
        for k in ['fish','coral']:
            selected_pairs = [(v,c) for v in output_views for c in v['stages'][str(stage)]['top100']
                              if c['kind']==k and len(c['gt_ids'])==1 and len(c['small_gt_ids'])==1]
            unique = {(v['image_id'],v['seed'],c['small_gt_ids'][0]) for v,c in selected_pairs}
            summary[str(stage)]['unambiguous_small_instance_repetition'][k] = dict(
                candidates=len(selected_pairs), unique_view_instances=len(unique),
                additional_cells=len(selected_pairs)-len(unique))
        assert sum(summary[str(stage)]['top100_kinds'].values()) == 1800
        assert sum(summary[str(stage)]['top100_center_kinds'].values()) == 1800
        for transition in ['lost','retained','rescued','never_matched']:
            rr=[r for r in output_targets if r['stage']==stage and r['transition']==transition]
            summary[str(stage)]['coral_by_transition'][transition]=dict(n=len(rr),
                best_to_seed_px=stats([r['best_to_seed_px'] for r in rr]),
                normalized_distance=stats([r['best_to_seed_over_sqrt_area'] for r in rr]),
                best_at_own_seed=sum(r['best_is_own_seed'] for r in rr),
                best_negative=sum(not r['best_is_positive'] for r in rr),
                median_seed_to_best_logit_gap=float(np.median([r['seed_to_best_logit_gap'] for r in rr])),
                external_top100_kinds=dict(collections.Counter(c['kind'] for r in rr for c in r['external_top100'])))
    assert len(output_views)==18 and len(output_targets)==56
    result=dict(passed=True,training_updates=0,source_sha256=sources,script_sha256=sha(Path(__file__)),
                summary=summary,views=output_views,targets=output_targets,
                limitations=['No annotated GT is not proof of semantic background.',
                    'Support-cell and center-pixel ownership differ; mixed classes remain explicit.',
                    'Per-target external counts repeat the same view for multiple targets; use view counts for global totals.',
                    'Distances are augmented-image pixels using valid cell centers, not original-image pixels.',
                    'Held-out inference labels describe intended supervision, not direct training gradients on these views.',
                    'Local-maxima counts are descriptive; original top100 selection is unchanged.'])
    OUT.mkdir(exist_ok=True)
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
