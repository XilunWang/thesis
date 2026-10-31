"""Summarize original-instance exposure; repeated views are not independent samples."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def counts(rows):
    return dict(views=len(rows),retained=sum(r['retained'] for r in rows),
        resize_zero=sum(r['resized_area']==0 for r in rows),
        crop_zero_after_nonzero_resize=sum(r['resized_area']>0 and r['cropped_area']==0 for r in rows),
        filter_after_nonzero_crop=sum(r['cropped_area']>0 and not r['retained'] for r in rows),
        resized_1to15=sum(0<r['resized_area']<16 for r in rows),
        final_1to15=sum(0<r['final_area']<16 for r in rows),
        final_under256=sum(0<r['final_area']<256 for r in rows),
        crop_retains_under_half_nonempty=sum(0<r['cropped_area']<.5*r['resized_area'] for r in rows))
def main():
    out=ROOT/'reports/baseline-augmentation-audit-20261031';s=json.loads((out/'result.json').read_text())
    assert s['passed'] and len(s['runs'])==141 and all(r['parity_exact'] for r in s['runs'])
    rr=[r for r in s['records'] if r['source_area']>0];bins=[('tiny',0,256),('small',256,1024),('medium',1024,9216),('large',9216,float('inf'))]
    groups={}
    for r in rr:groups.setdefault((r['image_id'],r['instance_index']),[]).append(r)
    assert all(len(v)==3 for v in groups.values())
    result=dict(by_original_size={name:counts([r for r in rr if lo<=r['source_area']<hi]) for name,lo,hi in bins},
        by_class={str(c):counts([r for r in rr if r['category_id']==c]) for c in range(1,8)},
        unique_instances={name:dict(unique_instances=sum(lo<=v[0]['source_area']<hi for v in groups.values()),
        never_retained_in3=sum(lo<=v[0]['source_area']<hi and not any(r['retained'] for r in v) for v in groups.values())) for name,lo,hi in bins},
        source_empty_views=len(s['records'])-len(rr),pipeline_filtered_runs=sum(r['pipeline_filtered'] for r in s['runs']))
    for c in result['by_original_size'].values():
        assert c['views']==c['retained']+c['resize_zero']+c['crop_zero_after_nonzero_resize']+c['filter_after_nonzero_crop']
    (out/'summary.json').write_text(json.dumps(result,indent=2));print('AUGMENTATION_SUMMARY_CHECKED')
if __name__=='__main__':main()
