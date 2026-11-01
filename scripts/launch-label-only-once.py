"""Detached one-shot launcher for the explicitly authorized frozen 300-step run."""
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/label-only-pilot-execution-20261101'
RUN=ROOT/'work_dirs/position_label_only_pilot'


def now():return datetime.now(timezone.utc).isoformat()


if __name__=='__main__':
    if sys.argv[1:]==['--worker']:
        status=dict(started_at=now(),command=['scripts/run-label-only-pilot.py','--execute'])
        (OUT/'process-result.json').write_text(json.dumps(status,indent=2))
        try:
            status['exit_code']=subprocess.call([sys.executable,'-u','-B',*status['command']],cwd=ROOT)
        except BaseException as e:
            status.update(exit_code=-1,error=repr(e));raise
        finally:
            status['finished_at']=now()
            if (RUN/'result.json').exists():
                shutil.copy2(RUN/'result.json',OUT/'result.json')
            (OUT/'process-result.json').write_text(json.dumps(status,indent=2))
    else:
        if sys.argv[1:]!=['--launch-authorized']:
            raise SystemExit('Requires explicit --launch-authorized; no default action')
        subprocess.run([sys.executable,'-B','scripts/run-label-only-pilot.py'],cwd=ROOT,check=True)
        plan=ROOT/'docs/label-only-pilot-plan-20261101.json'
        digest=hashlib.sha256(plan.read_bytes()).hexdigest()
        for file in ['result.json','restore-process.json']:
            proof=json.loads((ROOT/'reports/label-only-pilot-entry-20261101'/file).read_text())
            assert proof['passed'] and proof['training_updates']==0 and proof['plan_sha256']==digest
        assert not RUN.exists(),'Candidate run already exists; never launch twice'
        assert shutil.disk_usage(ROOT).free>5*1024**3
        OUT.mkdir(exist_ok=False)
        with (OUT/'console.log').open('wb') as log:
            process=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        launch=dict(pid=process.pid,launched_at=now(),maximum_steps=300,plan_sha256=digest,
                    authorization='User explicitly said start after the 300-step recommendation; original model frozen, evaluate then stop.',
                    initial_evaluation_before_updates=True,additional_training_allowed=False)
        (OUT/'launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps(launch),flush=True)
