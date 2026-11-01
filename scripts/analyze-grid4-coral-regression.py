"""Offline coral transitions and censored ranking from saved top200; no inference."""
import collections
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.local/analysis-libs')]
import numpy as np
from pycocotools import mask as mu
SRC=ROOT/'reports/position-grid4-execution-20261101'
OUT=ROOT/'reports/position-grid4-coral-analysis-20261101'


def match(support,cells):
    selected=set(cells);owner={}
    def visit(gid,seen):
        for cell in sorted(support[gid]&selected):
            if cell in seen:continue
            seen.add(cell)
            if cell not in owner or visit(owner[cell],seen):owner[cell]=gid;return True
        return False
    for gid in sorted(support):visit(gid,set())
    return set(owner.values())


def first_rank(cells,support):
    return next((i+1 for i,c in enumerate(cells) if c in support),None)


def main():
    OUT.mkdir(exist_ok=False)
    evals={s:json.loads((SRC/f'eval-{s}.json').read_text()) for s in [0,100,300]}
    lookup={s:{(r['image_id'],r['augmentation_seed'],r['k']):r for r in d['records'] if r['source']=='head'} for s,d in evals.items()}
    frozen=json.loads((ROOT/'reports/position-learning-plan-grid4-20261101/views.json').read_text())
    instances=[];views=[];source_hashes={}
    for view in frozen['runs']:
        p=ROOT/'reports/position-learning-preflight-20261101'/f"{view['split']}-{view['image_id']}-{view['seed']}.json"
        d=json.loads(p.read_text())
        coral=[gid for gid,a,label in zip(d['annotation_ids'],d['original_areas'],d['labels']) if label==1 and 0<a<=1024]
        if not coral:continue
        source_hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        masks=mu.decode(d['gt_rles']).transpose(2,0,1);h,w=masks.shape[-2:];gw=w//4
        all_support={};small={};categories={};metadata={}
        for gid,area,label,m in zip(d['annotation_ids'],d['original_areas'],d['labels'],masks):
            support=set(map(int,np.flatnonzero(m.reshape(h//4,4,w//4,4).sum((1,3)))))
            all_support[gid]=support;categories[gid]=label+1
            if 0<area<=1024:small[gid]=support
            metadata[gid]=dict(original_area=area,augmented_pixels=int(m.sum()),support_cells=len(support))
        key=(view['image_id'],view['seed']);by_stage={};records={}
        for step in evals:
            r100=lookup[step][(*key,100)];r200=lookup[step][(*key,200)]
            order=r200['selected_cells'];assert order[:100]==r100['selected_cells'] and len(set(order))==len(order)
            m100=match(small,order[:100]);m200=match(small,order)
            assert len(m100)==r100['occupied'] and sum(categories[g]==2 for g in m100)==r100['categories']['2']['hit']
            # Conservative GT oracle: remove repeat cells ONLY when each touches a
            # single annotated instance and that instance is an original-small fish.
            seen_fish=set();filtered=[];removed=[];cell_types=[]
            for rank,cell in enumerate(order,1):
                touched=[g for g,s in all_support.items() if cell in s]
                fish_only=len(touched)==1 and touched[0] in small and categories[touched[0]]==1
                duplicate=fish_only and touched[0] in seen_fish
                if duplicate:removed.append(rank)
                else:filtered.append(cell)
                if fish_only:seen_fish.add(touched[0])
                if rank<=100:
                    typ=('unambiguous_small_fish' if fish_only else 'coral_touch' if any(g in coral for g in touched)
                         else 'other_gt_touch' if touched else 'no_annotated_gt_touch')
                    cell_types.append(typ)
            oracle_match=match(small,filtered[:100])
            by_stage[str(step)]=dict(matched_coral_100=sum(g in m100 for g in coral),matched_coral_200=sum(g in m200 for g in coral),
                gt_fish_repeat_oracle_coral=sum(g in oracle_match for g in coral),removed_first100=sum(r<=100 for r in removed),
                first100_cell_types=dict(collections.Counter(cell_types)))
            records[step]=(order,m100,m200,filtered,oracle_match)
        views.append(dict(split=view['split'],image_id=key[0],seed=key[1],coral_targets=len(coral),stages=by_stage))
        for gid in coral:
            x,y=view['assignments'][str(gid)];seed_cell=y*gw+x
            row=dict(split=view['split'],image_id=key[0],seed=key[1],annotation_id=gid,
                     valid_shape=d['valid_shape'],valid_cells=((d['valid_shape'][0]+3)//4)*((d['valid_shape'][1]+3)//4),
                     **metadata[gid],stages={})
            for step,(order,m100,m200,filtered,oracle_match) in records.items():
                ranks=first_rank(order,small[gid]);seed_rank=first_rank(order,{seed_cell})
                row['stages'][str(step)]=dict(matched100=gid in m100,matched200=gid in m200,
                    first_occupied_rank_up_to200=ranks,seed_rank_up_to200=seed_rank,
                    occupied_cells_in100=len(set(order[:100])&small[gid]),
                    fish_repeat_oracle_match100=gid in oracle_match,
                    first_occupied_rank_after_oracle=first_rank(filtered,small[gid]))
            a=row['stages']['0']['matched100'];b=row['stages']['300']['matched100']
            row['transition']='retained' if a and b else 'lost' if a else 'rescued' if b else 'never_matched'
            row['final_miss_kind']=('matched' if b else 'ranking_101_200' if row['stages']['300']['first_occupied_rank_up_to200'] is not None and row['stages']['300']['first_occupied_rank_up_to200']>100 else 'no_support_in_top200' if row['stages']['300']['first_occupied_rank_up_to200'] is None else 'one_to_one_competition')
            instances.append(row)
    summaries={}
    for split in ['learn','holdout']:
        rows=[r for r in instances if r['split']==split];vv=[v for v in views if v['split']==split]
        stages={}
        for step in ['0','100','300']:
            stages[step]=dict(hit100=sum(r['stages'][step]['matched100'] for r in rows),hit200=sum(r['stages'][step]['matched200'] for r in rows),
                              seed_hit100=sum(r['stages'][step]['seed_rank_up_to200'] is not None and r['stages'][step]['seed_rank_up_to200']<=100 for r in rows),
                              fish_repeat_oracle_hit100=sum(r['stages'][step]['fish_repeat_oracle_match100'] for r in rows),
                              fish_repeat_cells_removed_first100=sum(v['stages'][step]['removed_first100'] for v in vv),
                              cell_types=dict(sum((collections.Counter(v['stages'][step]['first100_cell_types']) for v in vv),collections.Counter())))
            expected=evals[int(step)]['summaries'][f'{split}/head/100']['categories']['2']
            assert stages[step]['hit100']==expected['hit'] and len(rows)==expected['n']
        groups={}
        for name in ['retained','lost','rescued','never_matched']:
            rr=[r for r in rows if r['transition']==name]
            groups[name]=dict(records=len(rr),distinct_gt=len({r['annotation_id'] for r in rr}),
                             final_miss_kinds=dict(collections.Counter(r['final_miss_kind'] for r in rr)),
                             oracle_recovers=sum(not r['stages']['300']['matched100'] and r['stages']['300']['fish_repeat_oracle_match100'] for r in rr),
                             augmented_pixel_median=float(np.median([r['augmented_pixels'] for r in rr])) if rr else None)
        summaries[split]=dict(records=len(rows),distinct_gt=len({r['annotation_id'] for r in rows}),images=len({r['image_id'] for r in rows}),views=len(vv),
                               stages=stages,transitions=groups,final_miss_kinds=dict(collections.Counter(r['final_miss_kind'] for r in rows)))
    assert summaries['holdout']['records']==28 and summaries['learn']['records']==79
    result=dict(completed=True,training_updates=0,new_forward_passes=0,summary=summaries,instances=instances,views=views,
                limitation='Only ordered top200 cells saved. No actual logits, exact ranks beyond 200, or GT-free de-duplication claims. Oracle uses GT and is not a deployable method.',
                source_sha256={**source_hashes,**{f'eval-{s}.json':hashlib.sha256((SRC/f'eval-{s}.json').read_bytes()).hexdigest() for s in evals}},
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(summaries,indent=2))
    for r in instances:
        if r['split']=='holdout':print(r['image_id'],r['seed'],r['annotation_id'],r['transition'],r['augmented_pixels'],r['stages']['0']['first_occupied_rank_up_to200'],r['stages']['300']['first_occupied_rank_up_to200'],r['final_miss_kind'],r['stages']['300']['fish_repeat_oracle_match100'])


if __name__=='__main__':main()
