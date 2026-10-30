"""One authorized continuation, fixed epoch36 evaluation, then three-arm diagnostics."""
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
OUT=ROOT/'reports/e1p-resume-35epochs'
WORK=ROOT/'work_dirs/uiis_e1p_pixel_bll_r50_36e'
EVAL=ROOT/'reports/uiis-e1p-pixel-bll-36e-evaluation'
COMPARE=ROOT/'reports/e1p-final-comparison'


def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')


def main():
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'launch.json').open('x') as f:
            json.dump(dict(pid=os.getpid(),started_at=stamp(),authorized_epochs=[2,36],automatic_retry=False),f,indent=2)
        result=dict(completed=False,started_at=stamp(),stage='prelaunch')
        started=time.time()
        expected=json.loads((OUT/'expected-sources.json').read_text())
        def verify_sources():
            for f,h in expected['sources'].items():assert sha(ROOT/f)==h,f
        def execute(stage,command):
            verify_sources()
            result.update(stage=stage);save('process-result.json',result)
            with (OUT/(stage+'-console.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                save(stage+'-process.json',dict(pid=child.pid,command=command,started_at=stamp()))
                code=child.wait()
            result[stage+'_returncode']=code
            assert code==0,stage+' failed; no automatic retry or retraining'
        try:
            pre=json.loads((OUT/'preflight.json').read_text())
            assert pre['passed'] and pre['protocol_unchanged']
            assert json.loads((OUT/'restore-check.json').read_text())['passed']
            assert sha(WORK/'epoch_1.pth')==pre['checkpoint_sha256']
            assert not EVAL.exists() and not COMPARE.exists()
            reference_files=[ROOT/'reports'/folder/name for folder in ['uiis-baseline-36e-evaluation','uiis-e1-bll-36e-evaluation'] for name in ['metrics.json','predictions.segm.json']]
            reference_hashes={str(p):sha(p) for p in reference_files}
            execute('training',[sys.executable,'-u','-B',str(ROOT/'scripts/resume-uiis-pixel-bll.py'),'--execute-remaining'])
            stage=json.loads((WORK/'result-resume.json').read_text())
            assert stage['completed_epochs']==36 and stage['completed_iterations']==70884 and stage['resume_verified']
            result.update(training_completed_at=stamp(),training_summary=stage)
            execute('evaluation',[sys.executable,'-u','-B',str(ROOT/'scripts/evaluate-e1p-final.py')])
            metrics=json.loads((EVAL/'metrics.json').read_text())
            assert metrics['completed'] and metrics['evaluated_images']==691
            execute('comparison',[sys.executable,'-u','-B',str(ROOT/'scripts/compare-pixel-bll-offline.py'),
                '--candidate',str(EVAL/'predictions.segm.json'),'--output',str(COMPARE)])
            summary=json.loads((COMPARE/'summary.json').read_text())
            panels=json.loads((COMPARE/'paired-panels.json').read_text())
            assert set(summary['coco'])=={'E0','E1','E1-P'}
            assert set(panels['panels'])=={'E0/E1-P','E1/E1-P','E0/E1/E1-P'}
            for f,h in reference_hashes.items():assert sha(f)==h
            table={name:{label:summary['coco'][label]['stats'][i]*100 for label in ['E0','E1','E1-P']}
                for i,name in enumerate(['Mask AP','AP50','AP75','AP_small','AP_medium','AP_large'])}
            save('comparison.json',dict(unrounded_coco_percent=table,paired_boundary=panels['panels'],
                note='Single seed; conditional Boundary IoU is not Boundary AP; report misses and coverage.'))
            result.update(completed=True,stage='complete',metrics=metrics,reference_predictions_unchanged=True,
                final_checkpoint_sha256=sha(WORK/'epoch_36.pth'))
        except BaseException:
            result['error']=traceback.format_exc();raise
        finally:
            result.update(last_updated_at=stamp(),elapsed_seconds=time.time()-started)
            save('process-result.json',result)
        print('E1P_TRAIN_EVALUATE_COMPARE_COMPLETE',flush=True)


if __name__=='__main__':main()
