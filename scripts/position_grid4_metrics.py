"""Position-only one-to-one metrics and the frozen pilot decision gates."""
import collections
import numpy as np


def measure(selected, assignment, masks, ids, areas, labels, neighbor_pairs):
    selected=list(map(int,selected));assert len(selected)==len(set(selected))
    chosen=set(selected);gw=masks.shape[-1]//4;h,w=masks.shape[-2:]
    support={};category={};seeds={}
    for gid,area,label,mask in zip(ids,areas,labels,masks):
        if not 0<area<=1024:continue
        counts=mask.reshape(h//4,4,w//4,4).sum((1,3))
        support[int(gid)]=set(map(int,np.flatnonzero(counts)))
        category[int(gid)]=int(label)+1
        x,y=assignment[str(gid)];seeds[int(gid)]=y*gw+x
    owner={}
    def visit(gid,seen):
        for cell in sorted(support[gid]&chosen):
            if cell in seen:continue
            seen.add(cell)
            if cell not in owner or visit(owner[cell],seen):owner[cell]=gid;return True
        return False
    for gid in sorted(support):visit(gid,set())
    matched=set(owner.values());touch=set().union(*support.values()) if support else set()
    categories={str(c):dict(n=sum(v==c for v in category.values()),hit=sum(category[g]==c for g in matched)) for c in range(1,8)}
    assert all(int(a) in seeds and int(b) in seeds for a,b in neighbor_pairs)
    return dict(n=len(support),occupied=len(matched),seed=sum(c in chosen for c in seeds.values()),
                selected=len(selected),no_touch=len(chosen-touch),redundant=len(chosen&touch)-len(matched),
                pairs=len(neighbor_pairs),pair_hit=sum(seeds[int(a)] in chosen and seeds[int(b)] in chosen for a,b in neighbor_pairs),
                oracle=min(len(selected),len(support)),categories=categories)


def aggregate(rows):
    keys=['n','occupied','seed','selected','no_touch','redundant','pairs','pair_hit','oracle']
    sums={k:sum(r[k] for r in rows) for k in keys}
    image=collections.defaultdict(lambda:[0.,0.])
    for r in rows:image[r['image_id']][0]+=r['occupied'];image[r['image_id']][1]+=r['n']
    sums.update(occupied_recall=sums['occupied']/sums['n'] if sums['n'] else None,
                seed_recall=sums['seed']/sums['n'] if sums['n'] else None,
                utilization=sums['occupied']/sums['selected'] if sums['selected'] else None,
                pair_recall=sums['pair_hit']/sums['pairs'] if sums['pairs'] else None,
                image_mean=float(np.mean([h/n for h,n in image.values() if n])) if any(n for h,n in image.values()) else None,
                empty_views=sum(r['n']==0 for r in rows))
    sums['categories']={}
    for cat in map(str,range(1,8)):
        n=sum(r['categories'][cat]['n'] for r in rows);hit=sum(r['categories'][cat]['hit'] for r in rows)
        sums['categories'][cat]=dict(n=n,hit=hit,recall=hit/n if n else None)
    return sums


def random_mean(groups):
    """Average counts for each fixed view, retaining shared denominators."""
    assert groups and all(len(g)==len(groups[0]) for g in groups)
    result=[]
    for index,first in enumerate(groups[0]):
        peers=[g[index] for g in groups]
        assert all((p['image_id'],p['augmentation_seed'])==(first['image_id'],first['augmentation_seed']) for p in peers)
        row={k:first[k] for k in ['image_id','augmentation_seed','split']}
        for k in ['n','occupied','seed','selected','no_touch','redundant','pairs','pair_hit','oracle']:
            row[k]=float(np.mean([p[k] for p in peers]))
        row['categories']={c:{k:float(np.mean([p['categories'][c][k] for p in peers])) for k in ['n','hit']} for c in map(str,range(1,8))}
        result.append(row)
    return result


def bootstrap(candidate,control,seed=20261301,repeats=1000):
    def grouped(rows):
        d=collections.defaultdict(lambda:np.zeros(2))
        for r in rows:d[r['image_id']]+=np.array([r['occupied'],r['n']])
        return d
    a,b=grouped(candidate),grouped(control);assert a.keys()==b.keys()
    ids=sorted(a);aa=np.array([a[i] for i in ids]);bb=np.array([b[i] for i in ids]);assert np.array_equal(aa[:,1],bb[:,1])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        ii=rng.integers(0,len(ids),len(ids));x=aa[ii].sum(0);y=bb[ii].sum(0)
        if x[1]:values.append(float((x[0]-y[0])/x[1]))
    if not values:raise ValueError('No target-bearing bootstrap samples')
    return dict(interval95=np.quantile(values,[.025,.975]).tolist(),valid_replicates=len(values))


def decide(final_rows,initial_rows,random_rows,plan):
    a=aggregate(final_rows);controls={'step0':initial_rows,'random_mean':random_rows};gates={};intervals={}
    for name,rows in controls.items():
        b=aggregate(rows)
        gates[name+'_occupied']=a['occupied_recall']-b['occupied_recall']>=.10
        gates[name+'_seed']=a['seed_recall']-b['seed_recall']>=.05
        gates[name+'_image_mean']=a['image_mean']-b['image_mean']>=.05
        for c in ['1','2']:
            x=a['categories'][c]['recall'];y=b['categories'][c]['recall']
            gates[name+'_category_'+c]=x is not None and y is not None and x-y>=.05
        gates[name+'_pairs']=a['pair_recall'] is not None and b['pair_recall'] is not None and a['pair_recall']>=b['pair_recall'] and a['pair_hit']>0
        intervals[name]=bootstrap(final_rows,rows,plan['bootstrap_seed'],plan['bootstrap_repeats'])
        gates[name+'_bootstrap']=intervals[name]['interval95'][0]>0
    return dict(position_gates_passed=all(gates.values()),gates=gates,intervals=intervals,final=a)
