"""Verify BLL Runner with COCO init on four images/five mini-epochs only."""
import copy
import fcntl
import json
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REPORT=ROOT/'reports/bll-training-entry'
WORK=ROOT/'work_dirs/e1_entry_verification_coco'


def main():
    import torch
    from mmengine.config import Config
    if REPORT.exists() or WORK.exists(): raise FileExistsError('Existing verification output')
    REPORT.mkdir(parents=True)
    result=dict(passed=False,formal_training_started=False)
    deadline=time.monotonic()+600
    command=[sys.executable,'-B',str(ROOT/'scripts/train-uiis-bll.py'),'--work-dir',str(WORK),'--verification']
    def execute(name,args,check=True):
        with (REPORT/(name+'.log')).open('w') as f:
            p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.monotonic()))
        if check and p.returncode: raise RuntimeError(f'{name} failed; inspect log')
        return p.returncode,(REPORT/(name+'.log')).read_text()
    try:
        base=runpy.run_path(str(ROOT/'scripts/prepare-uiis-baseline.py'))['build_config']()
        e1=runpy.run_path(str(ROOT/'scripts/prepare-uiis-bll.py'))['build_config']()
        restored=copy.deepcopy(e1)
        restored.work_dir=base.work_dir
        restored._cfg_dict.pop('custom_imports')
        restored.model.panoptic_head.type=base.model.panoptic_head.type
        del restored.model.panoptic_head['bll_loss_weight'],restored.model.panoptic_head['bll_boundary_width']
        assert restored.to_dict()==base.to_dict()
        # Also compare against actual saved E0 run config (runtime fields excluded).
        actual=Config.fromfile(str(ROOT/'work_dirs/uiis_baseline_r50_36e/config-first.py'))
        keys=['model','train_dataloader','train_cfg','optim_wrapper','param_scheduler','randomness','env_cfg','load_from']
        differences=[key for key in keys if base[key]!=actual[key]]
        assert not differences,differences
        result['config_diff_whitelist']=['head.type','head.bll_loss_weight','head.bll_boundary_width','custom_imports','work_dir']
        result['actual_E0_config_fields_equal']=keys
        with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            code,log=execute('lock-rejection',command+['--stage','first'],False)
            assert code!=0 and 'TRAINING_ALREADY_RUNNING' in log
        no_mode=[x for x in command if x!='--verification']+['--stage','first']
        code,log=execute('formal-mode-required',no_mode,False)
        assert code!=0 and 'Choose exactly one' in log
        execute('first',command+['--stage','first'])
        first=json.loads((WORK/'result-first.json').read_text())
        assert first['completed_epochs']==1 and first['completed_iterations']==2
        assert first['load_check']['compatible_values_exact'] and first['load_check']['class_initialization_retained']
        execute('resume',command+['--stage','resume'])
        resumed=json.loads((WORK/'result-resume.json').read_text())
        assert resumed['completed_epochs']==5 and resumed['completed_iterations']==10 and resumed['resume_verified']
        assert resumed['load_check']['mismatched_keys']==[] and resumed['load_check']['compatible_values_exact']
        assert sorted(p.name for p in WORK.glob('epoch_*.pth'))==['epoch_3.pth','epoch_4.pth','epoch_5.pth']
        assert sorted(p.name for p in (WORK/'milestones').glob('*.pth'))==['epoch_1.pth','epoch_3.pth','epoch_5.pth']
        assert abs(max(resumed['lrs'])-1e-6)<1e-12
        saved=torch.load(WORK/'epoch_5.pth',map_location='cpu')
        assert saved['meta']['epoch']==5 and saved['meta']['iter']==10 and saved['optimizer']['state']
        code,log=execute('overwrite-rejection',command+['--stage','first'],False)
        assert code!=0 and 'Refusing to overwrite' in log
        code,log=execute('budget-rejection',command+['--stage','resume'],False)
        assert code!=0 and 'already reached budget' in log
        assert not (ROOT/'work_dirs/uiis_e1_bll_r50_36e').exists()
        result.update(passed=True,first=first,resumed=resumed,
            checks=['COCO compatible values exact','only classifier shape mismatch and init preserved',
                    'model/optimizer/scheduler/epoch/iteration exact restore','five mini epochs and LR decay',
                    'checkpoint retention and archives','lock/no-mode/overwrite/completed-budget rejection'],
            limitation='10 actual optimizer steps on four train images; no formal E1, no mid-epoch multiworker/RNG trajectory equivalence')
    except Exception:
        result['error']=traceback.format_exc(); raise
    finally:
        (REPORT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BLL_ENTRY_CHECK_PASSED',flush=True)


if __name__=='__main__': main()
