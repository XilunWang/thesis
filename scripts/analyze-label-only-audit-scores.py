"""Fixed-cell absolute and relative logits, plus algebraic rank decomposition."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT)]
import numpy as np
from pycocotools import mask as mu
OUT=ROOT/'reports/label-only-audit-scores-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(z,c,value=None):
    v=z[c] if value is None else value
    ids=np.arange(len(z));return int(1+np.sum((z>v)|((z==v)&(ids<c))))
def main():
    proof=json.loads((OUT/'result.json').read_text());request=json.loads((OUT/'request.json').read_text())
    assert proof['passed'] and sha(OUT/'request.json')==proof['request_sha256']
    reviews=json.loads((ROOT/'reports/label-only-visual-audit-20261101/visual-review.json').read_text(encoding='utf-8'))
    runs={r['case_id']:r for r in proof['runs']};results=[]
    for c in request['cases']:
        r=runs[c['case_id']];path=OUT/r['file'];assert sha(path)==r['sha256']
        a=np.load(path);valid=a['valid'].ravel().astype(bool);cell=c['cell'];assert valid[cell]
        z={s:np.where(valid,a[s].ravel().astype(float),-np.inf) for s in request['stages']}
        d=json.loads((ROOT/'reports/position-learning-preflight-20261101'/f"holdout-{c['image_id']}-{c['seed']}.json").read_text())
        assert d['input_sha256']==r['input_sha256'] and d['masks_sha256']==r['masks_sha256']
        h,w=d['padded_shape'];masks=mu.decode(d['gt_rles']).transpose(2,0,1)
        support=masks.reshape(len(masks),h//4,4,w//4,4).any(axis=(2,4)).reshape(len(masks),-1)
        assert not support[:,cell].any()
        background=valid & ~support.any(axis=0)
        stages={s:dict(logit=float(v[cell]),sigmoid=float(1/(1+np.exp(-v[cell]))),rank=rank(v,cell),valid_median=float(np.median(v[valid])),unannotated_median=float(np.median(v[background])),top100_threshold=float(np.sort(v[valid])[-100])) for s,v in z.items()}
        for s,v in stages.items():v['relative_to_valid_median']=v['logit']-v['valid_median']
        small=[]
        for i,(gid,area) in enumerate(zip(d['annotation_ids'],d['original_areas'])):
            cells=np.flatnonzero(support[i]&valid)
            if area>1024 or not len(cells):continue
            fixed=int(cells[np.argmax(z['initial'][cells])])
            small.append(dict(gt_id=gid,initial_best_cell=fixed,stages={s:dict(best_logit=float(v[cells].max()),fixed_logit=float(v[fixed]),candidate_minus_best=float(v[cell]-v[cells].max())) for s,v in z.items()}))
        comparisons={}
        for before in ['initial','old300']:
            b=z[before];n=z['new300'];delta=n[cell]-b[cell]
            others_old=b.copy();others_old[cell]=-np.inf
            others_new=n.copy();others_new[cell]=-np.inf
            comparisons[before+'_to_new300']=dict(candidate_logit_delta=float(delta),valid_median_delta=stages['new300']['valid_median']-stages[before]['valid_median'],relative_delta=stages['new300']['relative_to_valid_median']-stages[before]['relative_to_valid_median'],candidate_only_changed_rank=rank(others_old,cell,n[cell]),others_only_changed_rank=rank(others_new,cell,b[cell]))
        results.append(dict(case_id=c['case_id'],assessment=next(v['assessment'] for v in reviews['cases'] if v['case_id']==c['case_id']),stages=stages,comparisons=comparisons,small_targets=small))
    result=dict(training_updates=0,records=results,source_result_sha256=sha(OUT/'result.json'),script_sha256=sha(Path(__file__)),note='Mixed-score ranks are algebraic diagnostics, not model interventions or calibrated cross-model probabilities. Unannotated is not confirmed background.')
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in results:
        print(r['case_id'],[(s,round(v['logit'],3),v['rank'],round(v['valid_median'],3)) for s,v in r['stages'].items()],r['comparisons']['initial_to_new300'])
if __name__=='__main__':main()
