"""One authorized detached 300-step run. No restart or resume."""
import hashlib,json,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/coverage-priority-pilot-execution-20261101'
RUN=ROOT/'work_dirs/position_coverage_priority_pilot'
PLAN=ROOT/'docs/coverage-priority-pilot-plan-20261101.json'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
 if sys.argv[1:]==['--worker']:
  status=dict(started_at=now(),command=['scripts/run-coverage-priority-pilot.py','--execute'])
  (OUT/'process-result.json').write_text(json.dumps(status,indent=2))
  try:status['exit_code']=subprocess.call([sys.executable,'-u','-B',*status['command']],cwd=ROOT)
  except BaseException as e:status.update(exit_code=-1,error=repr(e));raise
  finally:
   status['finished_at']=now()
   if (RUN/'result.json').exists():shutil.copy2(RUN/'result.json',OUT/'result.json')
   (OUT/'process-result.json').write_text(json.dumps(status,indent=2))
 else:
  if sys.argv[1:]!=['--launch-authorized']:raise SystemExit('Explicit authorization flag required; no default action')
  subprocess.run([sys.executable,'-B','scripts/run-coverage-priority-pilot.py'],cwd=ROOT,check=True)
  proof=json.loads((ROOT/'reports/coverage-priority-entry-20261101/result.json').read_text())
  assert proof['passed'] and proof['training_updates']==0 and proof['plan_sha256']==sha(PLAN)
  for name,h in proof['codes'].items():
   if (ROOT/'scripts'/name).exists():assert sha(ROOT/'scripts'/name)==h,name
  assert not RUN.exists(),'Run already exists; never launch twice'
  assert shutil.disk_usage(ROOT).free>5*1024**3
  OUT.mkdir(exist_ok=False)
  launch=dict(authorized=True,maximum_steps=300,plan_sha256=sha(PLAN),launched_at=now(),authorization='User said start after frozen coverage-priority300-step plan; prepare checks then run and stop.',additional_training_allowed=False,initial_evaluation_before_updates=True)
  (OUT/'launch.json').write_text(json.dumps(launch,indent=2))
  with (OUT/'console.log').open('wb') as log:
   process=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  launch['pid']=process.pid;(OUT/'launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps(launch),flush=True)
