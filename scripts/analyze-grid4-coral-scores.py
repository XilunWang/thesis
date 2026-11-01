"""Fixed-region score/rank decomposition; descriptive, not training causality."""
import collections
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.local/analysis-libs')]
import numpy as np
from pycocotools import mask as mu
SRC=ROOT/'reports/position-grid4-coral-scores-20261101'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sigmoid(x):return float(1/(1+np.exp(-np.clip(x,-700,700))))


def main():
    proof=json.loads((SRC/'result.json').read_text());assert proof['passed'] and proof['training_updates']==0
    old=json.loads((ROOT/'reports/position-grid4-coral-analysis-20261101/result.json').read_text())
    previous={(r['image_id'],r['seed'],r['annotation_id']):r for r in old['instances'] if r['split']=='holdout'}
    frozen=json.loads((ROOT/'reports/position-learning-plan-grid4-20261101/views.json').read_text())
    views={(r['image_id'],r['seed']):r for r in frozen['runs']}
    rows=[];image_stats=[]
    for run in proof['runs']:
        p=SRC/run['file'];assert sha(p)==run['sha256']
        arrays=np.load(p,allow_pickle=False);valid=arrays['valid'].ravel();cells=np.flatnonzero(valid)
        logits={s:arrays[f'logits_{s}'].ravel().astype(np.float64) for s in [0,300]}
        key=(run['image_id'],run['seed']);v=views[key];h,w=v['padded_shape'];gw=w//4
        d=json.loads((ROOT/'reports/position-learning-preflight-20261101'/f'holdout-{key[0]}-{key[1]}.json').read_text())
        masks=mu.decode(d['gt_rles']).transpose(2,0,1)
        orders={s:cells[np.lexsort((cells,-logits[s][cells]))] for s in logits}
        ranks={s:np.zeros(valid.shape,dtype=np.int64) for s in logits}
        for s in logits:ranks[s][orders[s]]=np.arange(1,len(cells)+1)
        image_stats.append(dict(image_id=key[0],seed=key[1],stages={str(s):dict(valid_median=float(np.median(logits[s][cells])),global_top100_threshold=float(logits[s][orders[s][99]])) for s in logits}))
        for gid,area,label,mask in zip(d['annotation_ids'],d['original_areas'],d['labels'],masks):
            if not (label==1 and 0<area<=1024):continue
            support=np.flatnonzero(mask.reshape(h//4,4,w//4,4).sum((1,3)));assert valid[support].all()
            external=np.setdiff1d(cells,support);assert len(external)>=100
            x,y=v['assignments'][str(gid)];seed_cell=y*gw+x
            row=dict(image_id=key[0],seed=key[1],annotation_id=gid,transition=previous[(*key,gid)]['transition'],stages={})
            best={};external_order={}
            for s in logits:
                best[s]=int(support[np.argmin(ranks[s][support])])
                external_order[s]=external[np.lexsort((external,-logits[s][external]))]
                b=best[s];value=float(logits[s][b]);threshold=float(logits[s][external_order[s][99]])
                actual_rank=int(ranks[s][b]);prior_rank=previous[(*key,gid)]['stages'][str(s)]['first_occupied_rank_up_to200']
                assert (actual_rank if actual_rank<=200 else None)==prior_rank
                row['stages'][str(s)]=dict(best_logit=value,best_sigmoid=sigmoid(value),best_cell=int(b),best_rank=actual_rank,
                    seed_logit=float(logits[s][seed_cell]),seed_sigmoid=sigmoid(logits[s][seed_cell]),seed_rank=int(ranks[s][seed_cell]),
                    external_top100_threshold=threshold,external_threshold_sigmoid=sigmoid(threshold),margin=value-threshold,
                    support_mean=float(np.mean(logits[s][support])),support_median=float(np.median(logits[s][support])),
                    valid_median=float(np.median(logits[s][cells])))
            a,b=row['stages']['0'],row['stages']['300']
            ds=b['best_logit']-a['best_logit'];dt=b['external_top100_threshold']-a['external_top100_threshold']
            dm=b['margin']-a['margin'];assert np.isclose(dm,ds-dt,rtol=1e-10,atol=1e-10)
            def hybrid_rank(inside_step,outside_step):
                cell=best[inside_step];score=logits[inside_step][cell];other=logits[outside_step][external]
                return int(1+np.count_nonzero(other>score)+np.count_nonzero((other==score)&(external<cell)))
            row.update(delta_best_logit=ds,delta_competition_threshold=dt,delta_margin=dm,
                       delta_seed_logit=b['seed_logit']-a['seed_logit'],
                       fixed_old_best_cell_logits=[float(logits[s][best[0]]) for s in [0,300]],
                       fixed_final_best_cell_logits=[float(logits[s][best[300]]) for s in [0,300]],
                       only_self_changed_rank=hybrid_rank(300,0),only_external_changed_rank=hybrid_rank(0,300))
            eps=1e-6
            row['score_pattern']=('both_self_down_external_up' if ds < -eps and dt > eps else
                'self_down_external_not_up' if ds < -eps else 'external_up_self_not_down' if dt > eps else 'other')
            rows.append(row)
    assert len(rows)==28
    summary={}
    for group in ['lost','retained','rescued','never_matched']:
        rr=[r for r in rows if r['transition']==group]
        summary[group]=dict(records=len(rr),patterns=dict(collections.Counter(r['score_pattern'] for r in rr)),
            median_delta_best=float(np.median([r['delta_best_logit'] for r in rr])),
            median_delta_external_threshold=float(np.median([r['delta_competition_threshold'] for r in rr])),
            median_delta_seed=float(np.median([r['delta_seed_logit'] for r in rr])),
            only_self_changed_misses=sum(r['only_self_changed_rank']>100 for r in rr),
            only_external_changed_misses=sum(r['only_external_changed_rank']>100 for r in rr))
    result=dict(passed=True,training_updates=0,summary=summary,instances=rows,views=image_stats,
                source_result_sha256=sha(SRC/'result.json'),script_sha256=sha(Path(__file__)),
                limitation='Raw-score and rank decomposition across fixed checkpoints, not a causal training experiment; sigmoid values are not calibrated correctness probabilities. Hybrid maps use GT and are diagnostic only.')
    (SRC/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
    for r in rows:
        if r['transition']=='lost':print(r['image_id'],r['seed'],r['annotation_id'],r['score_pattern'],r['stages']['0'],r['stages']['300'],r['only_self_changed_rank'],r['only_external_changed_rank'])


if __name__=='__main__':main()
