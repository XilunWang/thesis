"""Explicitly authorized E1-P epoch 1 -> 36 continuation; frozen first entry intact."""
import argparse
import fcntl
import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
entry = runpy.run_path(str(ROOT / 'scripts/train-uiis-pixel-bll.py'))
torch, Runner = entry['torch'], entry['Runner']
WORK = ROOT / 'work_dirs/uiis_e1p_pixel_bll_r50_36e'
OUT = ROOT / 'reports/e1p-resume-35epochs'
FIRST_HASH = 'ada5705de4ad9a7a57cb35e4a1862e81f7cf84fc013830a05d52dcbcdb159353'


class InspectionComplete(Exception):
    pass


class InspectGuard(entry['BLLGuard']):
    def before_train(self, runner):
        super().before_train(runner)
        assert self.resume_verified and runner.epoch == 1 and runner.iter == 1969
        assert runner.train_dataloader.num_workers == 2
        assert len(runner.train_dataloader.dataset) == 3937 and len(runner.train_dataloader) == 1969
        (OUT / 'restore-check.json').write_text(json.dumps(dict(passed=True,epoch=runner.epoch,
            iteration=runner.iter,model_exact=self.load_check,optimizer_scheduler_exact=True,
            optimizer_updates=0,workers=2,train_images=3937,
            limitation='No mid-epoch worker RNG or uninterrupted trajectory equivalence claim.'),indent=2))
        raise InspectionComplete()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--execute-remaining', action='store_true')
    args = p.parse_args()
    with (ROOT / 'work_dirs/uiis-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected=json.loads((OUT/'expected-sources.json').read_text())
        for f,h in expected['sources'].items(): assert entry['sha'](ROOT/f)==h,f
        cfg = entry['build_config']()
        protocol = json.loads((WORK/'run-protocol.json').read_text())
        assert not protocol['verification'] and protocol['config'] == cfg.pretty_text
        for f,h in protocol['source_sha256'].items(): assert entry['sha'](ROOT/f)==h,f
        checkpoint=WORK/'epoch_1.pth'
        assert entry['sha'](checkpoint)==FIRST_HASH
        assert Path((WORK/'last_checkpoint').read_text().strip())==checkpoint
        assert not (WORK/'result-resume.json').exists()
        assert cfg.train_cfg.max_epochs==36 and cfg.val_cfg is None
        if args.execute_remaining:
            assert json.loads((OUT/'preflight.json').read_text())['passed']
            assert json.loads((OUT/'restore-check.json').read_text())['passed']
            assert not (WORK/'config-resume.py').exists(), 'Existing attempt; do not automatically retry'
        else:
            assert not (OUT/'restore-check.json').exists()
            check_work=ROOT/'work_dirs/e1p_resume_restore_check'
            check_work.mkdir(exist_ok=False)
            cfg.work_dir=str(check_work)
        cfg.resume=True;cfg.load_from=str(checkpoint)
        if args.execute_remaining:cfg.dump(str(WORK/'config-resume.py'))
        entry['register_all_modules']()
        runner=Runner.from_cfg(cfg)
        guard=(entry['BLLGuard'] if args.execute_remaining else InspectGuard)(None,[12,24,36],checkpoint)
        runner.register_hook(guard)
        try:
            runner.train()
        except InspectionComplete:
            assert args.check_only
            print('E1P_RESTORE_CHECK_PASS_NO_UPDATE',flush=True)
            return
        assert args.execute_remaining and runner.epoch==36 and runner.iter==70884 and guard.resume_verified
        result=dict(stage='resume',verification=False,completed_epochs=runner.epoch,completed_iterations=runner.iter,
            resume_verified=True,load_check=guard.load_check,
            lrs=sorted(set(g['lr'] for g in runner.optim_wrapper.optimizer.param_groups)))
        (WORK/'result-resume.json').write_text(json.dumps(result,indent=2))
        print('E1P_36_EPOCHS_COMPLETE',flush=True)


if __name__=='__main__':main()
