"""Opt-in representative objective, v1; original grid4 head/inference unchanged."""
import numpy as np
import torch
from torch.nn import functional as F
from scripts.isolated_position_head import position_loss

VERSION = 'instance-representative-v1'
WEIGHTS = {'positive': 0.5, 'duplicate': 0.25, 'outside': 0.25}


def make_layout(scores, masks, ids, areas, valid):
    """Detached score preference + one-to-one matching, not max-total-score."""
    scores = np.asarray(scores).reshape(-1)
    valid = np.asarray(valid, dtype=bool)
    gh, gw = valid.shape
    ids = list(map(int, ids))
    assert masks.shape == (len(ids), gh*4, gw*4)
    assert len(ids) == len(areas) and len(ids) == len(set(ids))
    assert np.isfinite(scores[valid.ravel()]).all()
    support = {g: set(map(int, np.flatnonzero(m.reshape(gh,4,gw,4).any(axis=(1,3)))))
               for g,a,m in zip(ids,areas,masks) if 0 < a <= 1024}
    for cells in support.values():
        if not cells or not all(valid.ravel()[c] for c in cells):
            raise ValueError('Missing or invalid small-instance support')
    candidates = {g: sorted(cells, key=lambda c: (-float(scores[c]),c)) for g,cells in support.items()}
    owner = {}
    def visit(g, seen):
        for c in candidates[g]:
            if c in seen: continue
            seen.add(c)
            if c not in owner or visit(owner[c], seen):
                owner[c] = g
                return True
        return False
    for g in sorted(support, key=lambda g:(len(support[g]),g)):
        if not visit(g,set()):
            raise ValueError('Insufficient distinct support cells; cannot discard or merge instances')
    reps = {g:c for c,g in owner.items()}
    positive = set(reps.values())
    memberships = {}
    for g,cells in support.items():
        for c in cells-positive: memberships.setdefault(c,[]).append(g)
    groups = {g:[] for g in support}
    for c, owners in sorted(memberships.items()):
        g = min(owners,key=lambda g:((c//gw-reps[g]//gw)**2+(c%gw-reps[g]%gw)**2,g))
        groups[g].append(c)
    union = set().union(*support.values()) if support else set()
    outside = sorted(set(map(int,np.flatnonzero(valid)))-union)
    dup = [c for cells in groups.values() for c in cells]
    assert len(reps)==len(support)==len(positive)
    assert len(dup)==len(set(dup)) and set(dup)==union-positive
    assert not positive & set(dup) and not union & set(outside)
    assert len(positive)+len(dup)+len(outside)==int(valid.sum())
    return dict(representatives=reps,groups=groups,outside=outside,shape=[gh,gw])


def terms_for_layout(logits, layout):
    """Per-image terms, per-instance duplicate mean; empty terms are exact zero."""
    z=logits.reshape(-1)
    zero=z[:0].sum()  # Finite zero even if invalid padding contains NaN.
    reps=layout['representatives'];groups=layout['groups']
    positive=F.softplus(-z[list(reps.values())]).mean() if reps else zero
    duplicate_terms=[F.softplus(z[cells]-z[reps[g]]).mean() for g,cells in groups.items() if cells]
    duplicate=torch.stack(duplicate_terms).mean() if duplicate_terms else zero
    outside=F.softplus(z[layout['outside']]).mean() if layout['outside'] else zero
    return dict(positive=positive,duplicate=duplicate,outside=outside)


def instance_term(logits,layout,gid):
    """This instance's contribution, excluding global outside loss."""
    z=logits.reshape(-1); reps=layout['representatives'];groups=layout['groups']
    value=WEIGHTS['positive']*F.softplus(-z[reps[gid]])/len(reps)
    nonempty=sum(bool(c) for c in groups.values())
    if groups[gid]:
        value=value+WEIGHTS['duplicate']*F.softplus(z[groups[gid]]-z[reps[gid]]).mean()/nonempty
    return value


def objective(logits, targets, valid, instances=None, *, enabled=False):
    """Default-off calls the original loss; enabled has no inference changes."""
    if not enabled:
        return position_loss(logits, targets, valid), None
    if instances is None or len(instances)!=len(logits):
        raise ValueError('Per-image instance masks/IDs/original areas required')
    if logits.shape!=valid.shape:
        raise ValueError('Logit/valid dimensions differ')
    losses=[];layouts=[];all_terms=[]
    for z, keep, item in zip(logits,valid,instances):
        if not bool(keep.any()): raise ValueError('No valid cells')
        layout=make_layout(z.detach().cpu().numpy(),item['masks'],item['ids'],item['areas'],keep.cpu().numpy())
        terms=terms_for_layout(z,layout)
        value=sum(WEIGHTS[k]*v for k,v in terms.items())
        if not bool(torch.isfinite(value)): raise ValueError('Nonfinite representative loss')
        losses.append(value);layouts.append(layout);all_terms.append(terms)
    return torch.stack(losses).mean(),dict(version=VERSION,weights=dict(WEIGHTS),layouts=layouts,terms=all_terms)
