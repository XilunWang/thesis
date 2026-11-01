"""Descriptive audit of saved logs/results only. No model or training imports."""
import hashlib,json,math,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={'control':'reports/valid-region-ap-screen-launch-20261031/console.log','candidate':'reports/direct-query-execution-20261101/training-console.log'}
def parse(path):
 rows=[]
 for line in path.read_text(encoding='utf-8').splitlines():
  m=re.search(r'Epoch\(train\) \[(\d+)\]\[\s*(\d+)/1969\]',line)
  if not m:continue
  epoch,step=map(int,m.groups())
  # Historical console contains control then old valid-region candidate.
  if rows and (epoch,step)<=(rows[-1]['epoch'],rows[-1]['step']):break
  fields={k:float(v) for k,v in re.findall(r'([\w.]+):\s+(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)',line)}
  assert all(math.isfinite(v) for v in fields.values())
  fields.update(epoch=epoch,step=step)
  fields['final_loss']=sum(fields[k] for k in ['loss_cls','loss_mask','loss_dice'])
  fields['aux_loss']=sum(v for k,v in fields.items() if re.fullmatch(r'd\d+\.loss_(cls|mask|dice)',k))
  assert abs(fields['loss']-fields['final_loss']-fields['aux_loss'])<0.003
  rows.append(fields)
 assert [(r['epoch'],r['step']) for r in rows]==[(e,s) for e in range(1,4) for s in range(50,1951,50)]
 return rows
def summary(rows):
 keys=['loss','final_loss','loss_cls','loss_mask','loss_dice','aux_loss','grad_norm','time']
 return {k:statistics.mean(r[k] for r in rows) for k in keys}
def main():
 rows={k:parse(ROOT/p) for k,p in SOURCES.items()}
 report={'scope':'Saved 50-step rolling log windows, 39 per epoch; excludes unlogged last 19 steps. Not exact epoch mean, not per-image paired statistics.', 'epoch':{},'third_epoch_windows':{},'paired_logged_loss':{},'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES.values()}}
 for arm,data in rows.items():
  report['epoch'][arm]={str(e):summary([r for r in data if r['epoch']==e]) for e in range(1,4)}
  third=[r for r in data if r['epoch']==3]
  report['third_epoch_windows'][arm]={'first_500_logged_steps':summary(third[:10]),'last_500_logged_steps':summary(third[-10:])}
  assert {r['base_lr'] for r in data}=={1e-5} and {r['lr'] for r in data}=={1e-6}
 for e in range(1,4):
  pairs=[(a,b) for a,b in zip(rows['control'],rows['candidate']) if a['epoch']==e]
  report['paired_logged_loss'][str(e)]={'mean_candidate_minus_control':statistics.mean(b['loss']-a['loss'] for a,b in pairs),'candidate_lower_windows':sum(b['loss']<a['loss'] for a,b in pairs),'windows':len(pairs)}
 comparison_path=ROOT/'reports/direct-query-ap-screen-run/comparison.json'
 c=json.loads(comparison_path.read_text())['coco'];a=c['control'];b=c['candidate']
 report['class_ap_change']={k:{'delta':b['class_ap'][k]-a['class_ap'][k],'contribution_to_overall_ap':(b['class_ap'][k]-a['class_ap'][k])/len(a['class_ap'])} for k in a['class_ap']}
 report['metric_change']={name:b['stats'][i]-a['stats'][i] for i,name in enumerate(['AP','AP50','AP75','APs','APm','APl','AR100','AR300','AR1000','ARs','ARm','ARl'])}
 assert abs(sum(v['contribution_to_overall_ap'] for v in report['class_ap_change'].values())-report['metric_change']['AP'])<1e-10
 report['missing_evidence']=['No epoch1/epoch2 validation AP for either arm','No per-branch gradient/update norm or activation ratio in saved training logs','No repeated seeds or fixed-batch loss trajectory','Optimizer steps prove updates were executed, not convergence or usefulness of added branch']
 out=ROOT/'reports/direct-query-learning-analysis-20261101';out.mkdir(exist_ok=True)
 (out/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
