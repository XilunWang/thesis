"""Join existing decoder/scale evidence by GT id; no inference or training."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 paths=[ROOT/'reports/baseline-decoder-audit-20261031/instances.json',ROOT/'reports/baseline-scale-pilot-20261031/instances.json']
 decoder,scale=[json.loads(p.read_text(encoding='utf-8-sig')) for p in paths];rows=[]
 for d in decoder:
  if d['group']=='low_score_control':continue
  key=str(d['gt_id']);a,b=scale['original'][key],scale['larger'][key]
  assert a['image_id']==b['image_id']==d['image_id']
  assert d['stages'][-1]['best_iou']<.5 and a['best_any']<.5
  rows.append(dict(gt_id=d['gt_id'],image_id=d['image_id'],category=a['category'],early_good=any(s['best_iou']>=.5 for s in d['stages'][:-1]),large_geometry=b['best_any']>=.5,large_correct_geometry=b['best_same']>=.5,large_detected=b['matched']['score0.5_iou0.5'],large_best_iou=b['best_any']))
 assert len(rows)==45
 groups={}
 for name,rr in [('all',rows),('never_good',[r for r in rows if not r['early_good']]),('early_then_lost',[r for r in rows if r['early_good']]),('reefs',[r for r in rows if r['category']=='reefs'])]:
  groups[name]=dict(n=len(rr),larger_geometry=sum(r['large_geometry'] for r in rr),larger_correct_geometry=sum(r['large_correct_geometry'] for r in rr),larger_detected=sum(r['large_detected'] for r in rr))
 result=dict(groups=groups,rows=rows,input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},scope='Existing outputs only: original decoder all queries versus larger-scale final saved top100; asymmetric availability, no missing-query inference.')
 out=ROOT/'reports/research-decision-20261101/evidence-join.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
