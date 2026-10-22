"""One authorized resume followed by final evaluation; never retries training."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/e1-resume-35epochs'
WORK=ROOT/'work_dirs/uiis_e1_bll_r50_36e'

def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'launch.json').open('x') as f:
            json.dump(dict(pid=os.getpid(),started_at=stamp(),authorized_epochs=[2,36],automatic_retry=False),f,indent=2)
        result=dict(completed=False,started_at=stamp(),training_returncode=None,evaluation_returncode=None)
        started=time.time()
        try:
            pre=json.loads((OUT/'preflight.json').read_text())
            assert pre['passed'] and pre['protocol_unchanged']
            checkpoint=WORK/'epoch_1.pth'
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==pre['checkpoint_sha256']
            assert Path((WORK/'last_checkpoint').read_text().strip())==checkpoint
            command=[sys.executable,'-u','-B',str(ROOT/'scripts/train-uiis-bll.py'),'--stage','resume','--work-dir',str(WORK),'--execute-formal']
            with (OUT/'training-console.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                save('training-process.json',dict(pid=child.pid,command=command,started_at=stamp()))
                result['training_returncode']=child.wait()
            assert result['training_returncode']==0,'Training failed; evaluation not started'
            stage=json.loads((WORK/'result-resume.json').read_text())
            assert stage['completed_epochs']==36 and stage['completed_iterations']==70884 and stage['resume_verified']
            result.update(training_completed_at=stamp(),stage=stage)
            save('process-result.json',result)
            with (OUT/'evaluation-console.log').open('x') as log:
                result['evaluation_returncode']=subprocess.call([sys.executable,'-u','-B',str(ROOT/'scripts/evaluate-e1-final.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            assert result['evaluation_returncode']==0,'Evaluation failed; do not retrain'
            metrics=json.loads((ROOT/'reports/uiis-e1-bll-36e-evaluation/metrics.json').read_text())
            assert metrics['completed'] and metrics['evaluated_images']==691
            baseline=json.loads((ROOT/'reports/uiis-baseline-36e-evaluation/metrics.json').read_text())
            comparison={k:dict(e0=baseline['metrics'][k]*100,e1=v*100,difference=(v-baseline['metrics'][k])*100) for k,v in metrics['metrics'].items() if k.startswith('coco/segm_mAP') and k in baseline['metrics']}
            save('comparison.json',dict(metrics=comparison,limitation='Single seed; framework-rounded COCO values. Early epoch1 test inspection disclosed. Boundary evaluation not included.'))
            result.update(completed=True,metrics=metrics,comparison=comparison)
        except BaseException:
            result['error']=traceback.format_exc()
            raise
        finally:
            result.update(finished_at=stamp(),elapsed_seconds=time.time()-started)
            save('process-result.json',result)
        print('E1_RESUME_AND_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':
    main()
