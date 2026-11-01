"""GT-assisted supervision feasibility only; never modifies model or predictions."""
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.local/analysis-libs'))
import numpy as np
from pycocotools import mask as mu

SOURCE = ROOT / 'reports/position-grid4-coral-scores-20261101'
OUT = ROOT / 'reports/instance-representative-cache-20261101'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def allocate(support, score):
    """Score-preferred maximum-cardinality matching, NOT maximum total score."""
    candidates = {g: sorted(map(int, cells), key=lambda c: (-float(score[c]), c))
                  for g, cells in support.items()}
    owner = {}

    def visit(g, seen):
        for c in candidates[g]:
            if c in seen:
                continue
            seen.add(c)
            if c not in owner or visit(owner[c], seen):
                owner[c] = g
                return True
        return False

    for g in sorted(support, key=lambda g: (len(support[g]), g)):
        if not visit(g, set()):
            raise ValueError('Insufficient distinct support cells; never drop/merge a GT')
    result = {g: c for c, g in owner.items()}
    assert set(result) == set(support)
    return result


def duplicate_groups(support, reps, gw):
    """Remove ALL representatives before assigning each remaining cell once."""
    positive = set(reps.values())
    memberships = {}
    for g, cells in support.items():
        for c in cells:
            if c not in positive:
                memberships.setdefault(int(c), []).append(g)
    groups = {g: [] for g in support}
    for c, owners in sorted(memberships.items()):
        def key(g):
            r = reps[g]
            return ((c//gw-r//gw)**2+(c%gw-r%gw)**2, g)
        groups[min(owners, key=key)].append(c)
    union = set().union(*support.values()) if support else set()
    flattened = [c for cells in groups.values() for c in cells]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == union-positive
    assert not set(flattened) & positive
    return groups


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x))


def objective_parts(score, valid, reps, groups):
    """Three unweighted diagnostic terms; no final training coefficients chosen.

    Duplicate loss: instance-mean softplus(duplicate_logit-representative_logit).
    Selection/grouping is treated as detached. Report derivatives w.r.t. logits,
    NOT parameter gradients or actual optimization effects.
    """
    gradients = {k: np.zeros_like(score, dtype=float) for k in ['positive','duplicate','outside']}
    losses = {k: 0. for k in gradients}
    if reps:
        selected = np.array(list(reps.values()), dtype=int)
        losses['positive'] = float(np.logaddexp(0., -score[selected]).mean())
        gradients['positive'][selected] = (sigmoid(score[selected])-1.)/len(selected)
    nonempty = [g for g in groups if groups[g]]
    for g in nonempty:
        cells = np.asarray(groups[g], dtype=int); rep = reps[g]
        diff = score[cells]-score[rep]
        losses['duplicate'] += float(np.logaddexp(0., diff).mean())/len(nonempty)
        derivative = sigmoid(diff)/(len(nonempty)*len(cells))
        gradients['duplicate'][cells] += derivative
        gradients['duplicate'][rep] -= derivative.sum()
    inside = set(reps.values()) | {c for cells in groups.values() for c in cells}
    outside = np.array(sorted(set(map(int,np.flatnonzero(valid)))-inside), dtype=int)
    if len(outside):
        losses['outside'] = float(np.logaddexp(0., score[outside]).mean())
        gradients['outside'][outside] = sigmoid(score[outside])/len(outside)
    assert all(np.isfinite(x).all() for x in gradients.values())
    assert all(np.isfinite(x) for x in losses.values())
    return losses, gradients


def audit(support, score, valid, gw, pairs=()):
    reps = allocate(support, score)
    assert reps == allocate(dict(reversed(list(support.items()))), score)
    assert len(set(reps.values())) == len(support)
    assert all(c in support[g] and valid[c] for g,c in reps.items())
    groups = duplicate_groups(support, reps, gw)
    loss, gradients = objective_parts(score, valid, reps, groups)
    pos = list(reps.values()); dup = [c for group in groups.values() for c in group]
    combined = sum(gradients.values())  # Unit coefficients ONLY for sign audit.
    assert all(combined[c] < 0 for c in pos)
    assert all(combined[c] > 0 for c in dup)
    assert all(not x[~valid].any() for x in gradients.values())
    pairrows = []
    for a,b in pairs:
        assert a in reps and b in reps and reps[a] != reps[b]
        ca,cb = reps[a],reps[b]
        pairrows.append(dict(ids=[a,b],cells=[ca,cb],
            chebyshev_cells=max(abs(ca//gw-cb//gw),abs(ca%gw-cb%gw)),
            positive_logit_gradients=[float(combined[ca]),float(combined[cb])]))
    return dict(targets=len(support),representatives={str(g):c for g,c in reps.items()},
        duplicates=len(dup),duplicate_counts={str(g):len(c) for g,c in groups.items()},
        independent_argmax_collisions=len(support)-len({max(support[g],key=lambda c:(score[c],-c)) for g in support}),
        representatives_below_own_max=sum(int(score[c]<max(score[x] for x in support[g])) for g,c in reps.items()),
        naive_cross_positive_negative_pairs=sum(c in support[g] for g in support for h,c in reps.items() if g!=h),
        protected_positive_negative_conflicts=0,loss_parts=loss,pairs=pairrows)


def known_answer_checks():
    score=np.array([3.,2.,1.,0.,-1.,0.5,0.,0.,0.])
    valid=np.ones(9,dtype=bool);valid[-1]=False
    # Independent argmax would merge; reassignment preserves both instances.
    support={1:{0,1},2:{0}}
    a=audit(support,score,valid,3,[(1,2)])
    assert a['representatives']=={'1':1,'2':0}
    # Adjacent one-cell objects must both survive, even with no distance gap.
    b=audit({1:{0},2:{1}},score,valid,3,[(1,2)])
    assert b['pairs'][0]['chebyshev_cells']==1
    rejected=False
    try: allocate({1:{0},2:{0}},score)
    except ValueError: rejected=True
    assert rejected
    # Overlap: global representative protection removes contradictory labels.
    support={1:{0,1,3},2:{0,1,4}}
    reps=allocate(support,score);groups=duplicate_groups(support,reps,3)
    _,grad=objective_parts(score,valid,reps,groups)
    for term in grad:
        for c in range(len(score)):
            plus=score.copy();minus=score.copy();plus[c]+=1e-5;minus[c]-=1e-5
            p,_=objective_parts(plus,valid,reps,groups);m,_=objective_parts(minus,valid,reps,groups)
            assert np.isclose((p[term]-m[term])/2e-5,grad[term][c],atol=1e-8)
    # Exhaustive tiny feasibility check against enumeration, including deficits.
    sets=[set(c for c in range(3) if bits&(1<<c)) for bits in range(1,8)]
    checked=0
    for candidates in itertools.product(sets,repeat=3):
        possible=any(len(set(choice))==3 for choice in itertools.product(*candidates))
        try: allocate(dict(enumerate(candidates)),score);actual=True
        except ValueError: actual=False
        assert actual==possible;checked+=1
    empty=audit({},score,valid,3)
    assert empty['targets']==0
    return dict(passed=True,enumerated_matching_cases=checked,
        finite_difference_all_terms=True,adjacent_single_cells_preserved=True,
        impossible_shared_single_cell_rejected=True)


def main():
    checks=known_answer_checks()
    frozen_path=ROOT/'reports/position-learning-plan-grid4-20261101/views.json'
    frozen=json.loads(frozen_path.read_text())
    views={(r['image_id'],r['seed']):r for r in frozen['runs']}
    neighbors={(r['image_id'],r['seed']):r['pairs'] for r in frozen['frozen_neighbor_observations']}
    proof=json.loads((SOURCE/'result.json').read_text())
    assert proof['passed']
    sources={str(frozen_path.relative_to(ROOT)):sha(frozen_path),
             str((SOURCE/'result.json').relative_to(ROOT)):sha(SOURCE/'result.json')}
    def load(key):
        v=views[key]
        p=ROOT/'reports/position-learning-preflight-20261101'/f"{v['split']}-{key[0]}-{key[1]}.json"
        sources[str(p.relative_to(ROOT))]=sha(p)
        d=json.loads(p.read_text())
        assert d['input_sha256']==v['input_sha256'] and d['masks_sha256']==v['masks_sha256']
        masks=mu.decode(d['gt_rles']).transpose(2,0,1)
        h,w=d['padded_shape'];gw=w//4
        support={int(g):set(map(int,np.flatnonzero(m.reshape(h//4,4,gw,4).any(axis=(1,3)))))
                 for g,a,m in zip(d['annotation_ids'],d['original_areas'],masks) if 0<a<=1024}
        valid=np.zeros((h//4,gw),dtype=bool);vh,vw=d['valid_shape'][:2]
        valid[:(vh+3)//4,:(vw+3)//4]=True
        assert set(support)==set(map(int,v['assignments']))
        return v,d,support,valid.ravel(),gw
    scored=[]
    for run in proof['runs']:
        key=(run['image_id'],run['seed']);v,d,support,valid,gw=load(key)
        path=SOURCE/run['file'];assert sha(path)==run['sha256']
        arr=np.load(path,allow_pickle=False);assert np.array_equal(valid,arr['valid'].ravel())
        for stage in [0,300]:
            score=arr[f'logits_{stage}'].ravel().astype(float)
            result=audit(support,score,valid,gw,neighbors.get(key,[]))
            result.update(image_id=key[0],seed=key[1],stage=stage)
            result['coral']=[]
            for g,a,label in zip(d['annotation_ids'],d['original_areas'],d['labels']):
                if label!=1 or not 0<a<=1024:continue
                rep=result['representatives'][str(g)];sx,sy=v['assignments'][str(g)]
                result['coral'].append(dict(id=int(g),selected_cell=rep,old_seed_cell=sy*gw+sx,
                    selected_logit=float(score[rep]),old_seed_logit=float(score[sy*gw+sx]),
                    old_seed_becomes_duplicate=rep!=sy*gw+sx and sy*gw+sx not in result['representatives'].values()))
            scored.append(result)
    # Same fixed neighboring cases + original collision regression; no new images.
    stress=[]
    keys=sorted(set(neighbors)|{(2809,20263821)})
    for key in keys:
        v,d,support,valid,gw=load(key)
        pairs=neighbors.get(key,[(16538,16539)])
        for order in ['row_major','reverse_row_major']:
            # Deliberate synthetic preferences, NOT cached model scores.
            score=np.arange(len(valid),dtype=float)/len(valid)
            if order=='row_major':score=-score
            result=audit(support,score,valid,gw,pairs)
            result.update(image_id=key[0],seed=key[1],synthetic_priority=order)
            stress.append(result)
    summary=dict(scored_views=len(proof['runs']),scored_stage_checks=len(scored),
        targets_per_stage=sum(r['targets'] for r in scored if r['stage']==300),
        coral_records_per_stage=sum(len(r['coral']) for r in scored if r['stage']==300),
        score_stage_summary={str(stage):dict(
            independent_argmax_collisions=sum(r['independent_argmax_collisions'] for r in scored if r['stage']==stage),
            naive_cross_positive_negative_pairs=sum(r['naive_cross_positive_negative_pairs'] for r in scored if r['stage']==stage),
            representatives_below_own_max=sum(r['representatives_below_own_max'] for r in scored if r['stage']==stage),
            duplicate_cells=sum(r['duplicates'] for r in scored if r['stage']==stage),
            coral_representative_changed=sum(c['selected_cell']!=c['old_seed_cell'] for r in scored if r['stage']==stage for c in r['coral']),
            coral_old_seed_becomes_duplicate=sum(c['old_seed_becomes_duplicate'] for r in scored if r['stage']==stage for c in r['coral'])) for stage in [0,300]},
        geometry_views=len(keys),geometry_priority_checks=len(stress),
        frozen_neighbor_observations=sum(len(p) for p in neighbors.values()),
        frozen_distinct_pairs=len({tuple(p) for pairs in neighbors.values() for p in pairs}),
        geometry_targets_per_priority=sum(r['targets'] for r in stress if r['synthetic_priority']=='row_major'),
        geometry_pair_checks=sum(len(r['pairs']) for r in stress),
        geometry_independent_argmax_collisions=sum(r['independent_argmax_collisions'] for r in stress),
        geometry_naive_cross_positive_negative_pairs=sum(r['naive_cross_positive_negative_pairs'] for r in stress),
        neighboring_representative_pairs_within_one_cell=sum(p['chebyshev_cells']<=1 for r in stress for p in r['pairs']),
        protected_positive_negative_conflicts=0,missing_instances=0,training_updates=0)
    OUT.mkdir(exist_ok=True)
    result=dict(passed=True,scope='offline_GT_supervision_feasibility_not_inference_performance',
        known_answer_checks=checks,summary=summary,scored=scored,geometry_stress=stress,
        source_sha256=sources,script_sha256=sha(Path(__file__)))
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
