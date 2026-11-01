"""Detach the explicitly authorized single pilot and record its actual exit code."""
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/position-grid4-execution-20261101'


def now():return datetime.now(timezone.utc).isoformat()


if __name__=='__main__':
    if '--worker' in sys.argv:
        status=dict(started_at=now(),command=['scripts/run-position-grid4-pilot.py','--execute'])
        try:
            status['exit_code']=subprocess.call([sys.executable,'-u','-B',*status['command']],cwd=ROOT)
        except BaseException as e:
            status.update(exit_code=-1,error=repr(e));raise
        finally:
            status['finished_at']=now();(OUT/'process-result.json').write_text(json.dumps(status,indent=2))
    else:
        proof=json.loads((ROOT/'reports/position-grid4-entry-20261101/result.json').read_text())
        assert proof['passed'] and proof['training_updates']==0
        for n,h in proof['codes'].items():assert hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest()==h,n
        assert not (ROOT/'work_dirs/position_grid4_pilot').exists(),'Pilot already exists; do not relaunch'
        assert shutil.disk_usage(ROOT).free>5*1024**3
        OUT.mkdir(exist_ok=False)
        with (OUT/'console.log').open('wb') as log:
            p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker'],cwd=ROOT,
                               stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        launched=dict(pid=p.pid,launched_at=now(),maximum_steps=300,original_model_frozen=True,
                      authorization='User explicitly requested one 300-step head-only learning run, evaluation, then stop')
        (OUT/'launch.json').write_text(json.dumps(launched,indent=2));print(json.dumps(launched),flush=True)
