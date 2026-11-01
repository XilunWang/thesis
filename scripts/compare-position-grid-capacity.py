"""Fixed-cache capacity comparison; neither learned recall nor model training."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools import mask as mu
spec=importlib.util.spec_from_file_location('assignment',ROOT/'scripts/audit-query-cell-assignment.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
SRC=ROOT/'reports/position-learning-preflight-20261101'
OUT=ROOT/'reports/position-grid-comparison-20261101'
OPTIONS=[('stride8_single',8,1),('stride4_single',4,1),('stride8_double',8,2)]


def candidates_for(masks,ids,areas,stride):
    h,w=masks.shape[-2:];assert h%stride==w%stride==0
    result={}
    for gid,area,mask in zip(ids,areas,masks):
        if not 0<area<=1024:continue
        yy,xx=np.nonzero(mask);assert len(xx)
        j=np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2)
        cx,cy=int(xx[j])//stride,int(yy[j])//stride
        counts=mask.reshape(h//stride,stride,w//stride,stride).sum((1,3))
        result[gid]=sorted(map(int,np.flatnonzero(counts)),key=lambda c:((c//(w//stride)-cy)**2+(c%(w//stride)-cx)**2,-int(counts.flat[c]),c))
    return result


def main():
    OUT.mkdir(exist_ok=False)
    proof=json.loads((SRC/'result.json').read_text());assert len(proof['runs'])==384
    summaries={name:dict(targets=0,matched=0,deficit=0,failed_views=0,second_slots_used=0) for name,_,_ in OPTIONS}
    rows=[];source_hashes={};failure_case=None
    for i,r in enumerate(proof['runs']):
        p=SRC/f"{r['split']}-{r['image_id']}-{r['seed']}.json"
        d=json.loads(p.read_text());assert not d['filtered']
        source_hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        h,w=d['padded_shape'];vh,vw=d['valid_shape'][:2]
        masks=mu.decode(d['gt_rles']).transpose(2,0,1) if d['gt_rles'] else np.zeros((0,h,w),dtype=np.uint8)
        assert not masks[:,vh:,:].any() and not masks[:,:,vw:].any()
        row=dict(split=r['split'],image_id=r['image_id'],seed=r['seed'],options={})
        details={}
        for name,stride,slots in OPTIONS:
            c=candidates_for(masks,d['annotation_ids'],d['original_areas'],stride)
            matched,witness=module.assign(c,slots)
            assert len(set(matched.values()))==len(matched)
            assert all(cell in c[gid] and slot<slots for gid,(cell,slot) in matched.items())
            s=summaries[name];s['targets']+=len(c);s['matched']+=len(matched)
            s['deficit']+=witness['deficit'];s['failed_views']+=int(witness['deficit']>0)
            second=sum(slot>0 for cell,slot in matched.values());s['second_slots_used']+=second
            row['options'][name]=dict(targets=len(c),matched=len(matched),witness=witness,second_slots_used=second)
            if r['image_id']==2809 and r['seed']==20263821:
                details[name]={str(gid):dict(available_cells_xy=[[cell%(w//stride),cell//(w//stride)] for cell in c[gid]],
                    assigned=None if gid not in matched else dict(cell_xy=[matched[gid][0]%(w//stride),matched[gid][0]//(w//stride)],slot=matched[gid][1])) for gid in [16538,16539]}
        if details:
            failure_case=dict(image_id=r['image_id'],seed=r['seed'],options=details)
        rows.append(row)
        if (i+1)%64==0:print('CHECKED',i+1,flush=True)
    assert summaries['stride8_single']['failed_views']==1 and summaries['stride8_single']['deficit']==1
    result=dict(completed=True,training_updates=0,summary=summaries,runs=rows,failure_case=failure_case,
                source_hashes=source_hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Fixed 384 saved augmentations; bipartite GT capacity only. Two slots are capacity slots, not distinct learned coordinates.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries,indent=2));print(json.dumps(failure_case,indent=2))


if __name__=='__main__':main()
