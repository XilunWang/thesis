"""No-training tests for actual execution contracts and injected stage failures."""
import ast,copy,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/'.local/bll-libs'))
from scripts.direct_query_execution import check_next_update,check_output_paths,validate_final,sequence

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=ROOT/'reports/direct-query-execution-20261101';out.mkdir(exist_ok=True)
 for p in ['scripts/direct_query_execution.py','scripts/run-direct-query-screen.py']:ast.parse((ROOT/p).read_text(encoding='utf-8-sig'))
 code=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run-direct-query-screen.py')],capture_output=True,text=True)
 assert code.returncode==2 and 'explicit --execute required' in code.stderr
 (out/'no-execute.log').write_text(code.stdout+code.stderr)
 for i in range(5907):check_next_update(i//1969,i,0,1)
 for epoch,it,now,error in [(3,5907,0,RuntimeError),(2,5907,0,RuntimeError),(0,0,1,TimeoutError)]:
  try:check_next_update(epoch,it,now,1)
  except error:pass
  else:raise AssertionError('guard missed')
 expected=dict(head='fixture',enabled=True);final=dict(direct_query_contract=expected,meta=dict(epoch=3,iter=5907),optimizer={'state':{i:{'step':5907} for i in range(347)}})
 assert validate_final(final,expected)
 negative=[]
 for case in ['epoch','iteration','contract','diagnostic','partial','missing_state','wrong_step']:
  x=copy.deepcopy(final)
  if case=='epoch':x['meta']['epoch']=2
  if case=='iteration':x['meta']['iter']=5906
  if case=='contract':x['direct_query_contract']={}
  if case=='diagnostic':x['diagnostic_only']=True
  if case=='partial':x['partial_stop']=True
  if case=='missing_state':x['optimizer']['state'].pop(0)
  if case=='wrong_step':x['optimizer']['state'][0]['step']=5906
  try:validate_final(x,expected)
  except ValueError:negative.append(case)
  else:raise AssertionError(case)
 runs=[]
 for failed in [None,'train','eval','compare']:
  events=[];records=[]
  def train():
   events.append('train_candidate')
   if failed=='train':raise RuntimeError('injected OOM; no actual model')
   return 'fixed_epoch3'
  def evaluate(p):
   assert p=='fixed_epoch3';events.append('eval_candidate')
   if failed=='eval':raise TimeoutError('injected evaluation deadline')
   return {'fixture':True}
  def compare():
   events.append('compare_saved_E0_A_B')
   if failed=='compare':raise ValueError('injected bad predictions')
   return {'fixture':True}
  try:result=sequence(train,evaluate,compare,lambda r:records.append(copy.deepcopy(r)))
  except (RuntimeError,TimeoutError,ValueError):assert failed is not None and not records[-1]['completed']
  else:assert failed is None and result['completed']
  expected_events=['train_candidate','eval_candidate','compare_saved_E0_A_B'][:({'train':1,'eval':2,'compare':3}.get(failed,3))];assert events==expected_events
  runs.append(dict(injected_failure=failed,events=events,completed=records[-1]['completed']))
 with tempfile.TemporaryDirectory(dir=out) as tmp:
  root=Path(tmp);existing=root/'already_exists';existing.mkdir()
  check_output_paths([root/'new'])
  try:check_output_paths([root/'new',existing])
  except FileExistsError:pass
  else:raise AssertionError('existing output accepted')
 # Exercise actual finite-guard methods with a no-update base, without MMEngine imports.
 import torch
 tree=ast.parse((ROOT/'scripts/ap_screen_runtime.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ScreenFiniteOptimWrapper');node.decorator_list=[]
 class NoUpdateBase:
  def __init__(self):self.optimizer=SimpleNamespace(param_groups=[{'params':[torch.nn.Parameter(torch.tensor([1.]))]}]);self.base_calls=0
  def backward(self,*a,**kw):self.base_calls+=1
  def step(self,*a,**kw):self.base_calls+=1
 env={'torch':torch,'OptimWrapper':NoUpdateBase};exec(compile(ast.Module(body=[node],type_ignores=[]),'finite-guard-source','exec'),env);guard=env['ScreenFiniteOptimWrapper']()
 try:guard.backward(torch.tensor(float('nan')))
 except FloatingPointError:pass
 else:raise AssertionError('NaN allowed')
 param=guard.optimizer.param_groups[0]['params'][0];param.grad=torch.tensor([float('inf')])
 try:guard.step()
 except FloatingPointError:pass
 else:raise AssertionError('Inf allowed')
 assert guard.base_calls==0 and float(param)==1
 sources=['scripts/direct_query_execution.py','scripts/run-direct-query-screen.py','scripts/test-direct-query-execution.py','scripts/direct_query_feature_head.py','scripts/ap_screen_runtime.py','scripts/run-valid-region-ap-screen.py','scripts/valid_region_head.py','reports/direct-query-screen-preparation-20261101/candidate.py','reports/direct-query-screen-preparation-20261101/preparation.json','reports/direct-query-startup-20261101/acceptance.json','reports/direct-query-startup-20261101/candidate.json','reports/valid-region-ap-screen-run/comparison.json','reports/feature-readout-export-20261101/result.json']
 report=dict(passed=True,model_constructed=False,optimizer_updates=0,training_authorized=False,server_import_checked=False,simulated_iteration_checks=5907,epoch_limit_rejected=True,deadline_rejected=True,existing_output_rejected=True,checkpoint_negative_checks=negative,stage_cases=runs,nonfinite_loss_and_gradient_rejected=True,finite_guard_test='Actual class AST with no-update stub base; real MMEngine runtime checked separately',no_execute_rejected=True,sources={p:sha(ROOT/p) for p in sources})
 (out/'acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))
if __name__=='__main__':main()
