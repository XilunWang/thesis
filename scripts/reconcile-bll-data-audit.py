"""Combine an immutable full audit with explicit BrokenPipe-only recovery rows."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    original=ROOT/'reports/uiis-bll-occupancy-audit'
    recovery=ROOT/'reports/uiis-bll-occupancy-recovery'
    output=ROOT/'reports/uiis-bll-occupancy-verified'
    if output.exists(): raise FileExistsError(output)
    load=lambda p:json.loads(p.read_text())
    rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
    a,b=load(original/'summary.json'),load(recovery/'summary.json')
    assert a['completed'] and b['completed'] and b['passed']
    for key in ['annotation_sha256','target_rule','reference_sha256','pipeline','preprocessor']:
        assert a[key]==b[key],key
    assert a['sources']['scripts/query_bll.py']==b['sources']['scripts/query_bll.py']
    old=rows(original/'images.jsonl');new=rows(recovery/'images.jsonl')
    errors=load(original/'findings.json')['processing_errors']
    ids={r['id'] for r in errors}
    assert len(ids)==len(errors)==a['processing_errors']==14
    assert all('BrokenPipeError: [Errno 32] Broken pipe' in r['error'] for r in errors)
    assert {r['id'] for r in old if not r['processing_ok']}==ids
    assert {r['id'] for r in new}==ids and len(new)==len(ids)
    replacement={r['id']:r for r in new}
    combined=[replacement.get(r['id'],r) for r in old]
    assert len(combined)==len({r['id'] for r in combined})==3937
    assert all(r['processing_ok'] and r['paired_reference_exact'] for r in combined)
    good=[r for r in combined if not r.get('pipeline_filtered')]
    filtered=[r['id'] for r in combined if r.get('pipeline_filtered')]
    assert filtered==[976] and len(good)==3936
    assert all(not r['vanished'] and r['empty_augmented_masks']==0 for r in good)
    previous=load(ROOT/'reports/uiis-bll-data-audit/findings.json')['vanished']
    byid={r['id']:r for r in combined}
    preserved=sum(byid[r['id']]['valid_grid_pixels'][v['index']]>0 for r in previous for v in r['vanished'])
    assert preserved==183
    assert a['annotation_unchanged'] and b['annotation_unchanged'] and a['e0_unchanged'] and b['e0_unchanged']
    source_area=sum(sum(r['gt_pixels']) for r in good)
    expanded_area=sum(sum(r['occupied_expanded_valid_pixels']) for r in good)
    ratios=sorted(x/y for r in good for x,y in zip(r['occupied_expanded_valid_pixels'],r['gt_pixels']) if y>0)
    result=dict(passed=True,completed=True,checked=3937,grid_checked=len(good),
        pipeline_filtered_ids=filtered,recovered_infrastructure_failure_ids=sorted(ids),
        augmented_instances=sum(r['augmented_instances'] for r in good),
        vanished_instances=0,previous_183_preserved=preserved,paired_reference_exact=True,
        annotation_unchanged=True,e0_unchanged=True,model_built=False,optimizer_built=False,
        target_rule=a['target_rule'],source_area=source_area,expanded_valid_area=expanded_area,
        area_ratio=dict(global_ratio=expanded_area/source_area,median=ratios[len(ratios)//2],
                        p95=ratios[int(.95*(len(ratios)-1))],maximum=max(ratios)),
        limitations=a['limitations']+['reconciled evidence; initial BrokenPipe failure report remains unchanged'],
        seconds=dict(initial=a['elapsed_seconds'],recovery=b['elapsed_seconds']))
    evidence=[original/'summary.json',original/'images.jsonl',original/'findings.json',
              recovery/'summary.json',recovery/'images.jsonl',Path(__file__).resolve()]
    result['evidence_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    output.mkdir()
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'images.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in combined))
    print(json.dumps(result))


if __name__=='__main__': main()
