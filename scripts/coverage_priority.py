"""Offline proposal only; no training entry imports this module."""
import numpy as np
VERSION='coverage-priority-v1-offline'
def representatives(scores,masks,ids,areas,valid):
    gh,gw=valid.shape;scores=np.asarray(scores).reshape(-1)
    assert masks.shape==(len(ids),gh*4,gw*4)
    assert len(ids)==len(set(ids))==len(areas)
    assert np.isfinite(scores[valid.ravel()]).all()
    counts={int(g):m.reshape(gh,4,gw,4).sum((1,3)).reshape(-1) for g,a,m in zip(ids,areas,masks) if 0<a<=1024}
    support={g:np.flatnonzero(v).tolist() for g,v in counts.items()}
    for cells in support.values():
        if not cells or not valid.ravel()[cells].all():raise ValueError('Missing/invalid support')
    candidates={g:sorted(cells,key=lambda c:(-int(counts[g][c]),-float(scores[c]),c)) for g,cells in support.items()}
    owner={}
    def visit(g,seen):
        for c in candidates[g]:
            if c in seen:continue
            seen.add(c)
            if c not in owner or visit(owner[c],seen):owner[c]=g;return True
        return False
    for g in sorted(support,key=lambda g:(len(support[g]),g)):
        if not visit(g,set()):raise ValueError('No full one-to-one assignment')
    reps={g:c for c,g in owner.items()}
    assert len(reps)==len(support)==len(set(reps.values()))
    return reps
