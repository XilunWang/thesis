"""Inventory existing local outputs and freeze a tiny diagnostic cohort; no model execution."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=ROOT/'reports/baseline-decoder-audit-20261031/instances.json'
    rows=json.loads(source.read_text(encoding='utf-8-sig'))
    chosen=[]
    for r in rows:
        if r['image_id'] not in [2,19,42]: continue
        group=('low_score_geometry_control' if r['group']=='low_score_control' else
               'never_good' if max(s['best_iou'] for s in r['stages'])<.5 else 'early_lost')
        chosen.append({'image_id':r['image_id'],'gt_id':r['gt_id'],'category_id':r['category_id'],
                       'group':group,'final_best_iou':r['stages'][-1]['best_iou'],
                       'all_stage_best_iou':max(s['best_iou'] for s in r['stages'])})
    assert len(chosen)==14 and len({r['gt_id'] for r in chosen})==14
    inventory=[]
    for p in sorted((ROOT/'reports').rglob('*.npz')):
        with zipfile.ZipFile(p) as z:
            inventory.append({'path':str(p.relative_to(ROOT)).replace('\\','/'),
                              'bytes':p.stat().st_size,'array_names':[n[:-4] for n in z.namelist() if n.endswith('.npy')]})
    inputs=[source,ROOT/'data/UIIS/annotations/val.json',ROOT/'scripts/audit-baseline-decoder.py']
    inputs += [source.parent/f'image-{i}.json' for i in [2,19,42]]
    out=ROOT/'reports/feature-readout-plan-20261101'; out.mkdir(exist_ok=True)
    result={'design_only':True,'model_run':False,'image_ids':[2,19,42],'cases':chosen,
            'groups':{g:sum(r['group']==g for r in chosen) for g in sorted({r['group'] for r in chosen})},
            'input_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in inputs},
            'npz_inventory':inventory,'npz_count':len(inventory)}
    (out/'cache-inventory-and-selection.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'images':result['image_ids'],'gt_ids':[r['gt_id'] for r in chosen],
                      'groups':result['groups'],'npz_count':len(inventory),'array_name_sets':sorted({tuple(x['array_names']) for x in inventory})}))
if __name__=='__main__': main()
