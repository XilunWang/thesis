"""Pure control-flow contracts, importable without ML dependencies."""
import json,time,traceback
EPOCHS=3
ITERATIONS=5907
OPTIMIZER_STATES=347
BUDGET_SECONDS=4*3600

def check_next_update(epoch,iteration,now,deadline):
 if epoch>=EPOCHS or iteration>=ITERATIONS:raise RuntimeError('three-epoch budget exhausted')
 if now>=deadline:raise TimeoutError('execution deadline reached')

def check_output_paths(paths):
 for path in paths:
  if path.exists():raise FileExistsError('refuse existing output; no implicit restart/resume: '+str(path))

def validate_final(saved,expected):
 if saved.get('diagnostic_only') or saved.get('partial_stop'):raise ValueError('diagnostic/partial checkpoint cannot be evaluated')
 if saved.get('direct_query_contract')!=expected:raise ValueError('checkpoint contract mismatch')
 if (saved['meta']['epoch'],saved['meta']['iter'])!=(EPOCHS,ITERATIONS):raise ValueError('not the fixed final epoch')
 states=saved['optimizer']['state']
 if len(states)!=OPTIMIZER_STATES or any(int(v['step'])!=ITERATIONS for v in states.values()):raise ValueError('incomplete optimizer steps')
 return True

def sequence(train,evaluate,compare,persist):
 result=dict(completed=False,stages=[])
 try:
  checkpoint=train();result['stages'].append('candidate_training_complete');persist(result)
  evaluation=evaluate(checkpoint);result['stages'].append('candidate_evaluation_complete');result['evaluation']=evaluation;persist(result)
  result['comparison']=compare();result['stages'].append('comparison_complete');result['completed']=True
  return result
 except BaseException:
  result['error']=traceback.format_exc();raise
 finally:persist(result)
