"""Fixed visual-audit cases: saved rank trajectories only, no inferred logits."""
import json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    folder=ROOT/'reports/label-only-visual-audit-20261101'
    selection=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
    review=json.loads((folder/'visual-review.json').read_text(encoding='utf-8'))
    assessments={r['case_id']:r['assessment'] for r in review['cases']}
    paths={name:ROOT/'reports'/directory/f'eval-{step}.json' for name,directory,step in [
        ('initial','label-only-pilot-execution-20261101',0),
        ('new100','label-only-pilot-execution-20261101',100),
        ('new300','label-only-pilot-execution-20261101',300),
        ('old300','position-grid4-execution-20261101',300)]}
    data={}
    hashes={}
    for name,p in paths.items():
        hashes[name]=sha(p)
        expected=json.loads((ROOT/'reports/label-only-pilot-execution-20261101/artifact-sha256.json').read_text(encoding='utf-8')) if name!='old300' else json.loads((ROOT/'docs/label-only-pilot-plan-20261101.json').read_text(encoding='utf-8'))['control_sha256']
        assert hashes[name]==expected[p.name]
        rows=json.loads(p.read_text(encoding='utf-8'))['records']
        data[name]={(r['image_id'],r['augmentation_seed'],r['k']):r for r in rows if r['split']=='holdout' and r['source']=='head'}
    records=[]
    for c in selection['cases']:
        stages={}
        for name,rows in data.items():
            key=(c['image_id'],c['seed'])
            r=rows[key+(200,)];r100=rows[key+(100,)]
            assert r['selected_cells'][:100]==r100['selected_cells']
            cells=r['selected_cells'];rank=cells.index(c['cell'])+1 if c['cell'] in cells else None
            stages[name]=dict(rank=rank,rank_censored_above=200 if rank is None else None,small_instances=r100['n'],small_occupied_top100=r100['occupied'],no_touch_top100=r100['no_touch'])
        assert stages['new300']['rank']==c['new_rank']
        assert stages['old300']['rank'] is None or stages['old300']['rank']>100
        assert len({v['small_instances'] for v in stages.values()})==1
        records.append(dict(case_id=c['case_id'],image_id=c['image_id'],seed=c['seed'],cell=c['cell'],assessment=assessments[c['case_id']],stages=stages))
    out=ROOT/'reports/label-only-audit-ranks-20261101';out.mkdir(exist_ok=True)
    result=dict(training_updates=0,new_forward_passes=0,score_attribution_resolved=False,reason='Candidate eval stores ordered top200 cells but no logits. Rank cannot identify absolute score changes. Old coral score cache is a different arm.',source_sha256=hashes,selection_sha256=sha(folder/'selection.json'),script_sha256=sha(Path(__file__)),records=records)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in records:
        print(r['case_id'],r['assessment'],[(s,v['rank'],v['small_occupied_top100'],v['small_instances']) for s,v in r['stages'].items()])
if __name__=='__main__':main()

