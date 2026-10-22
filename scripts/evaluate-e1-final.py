"""Evaluate the frozen E1 epoch-36 checkpoint; never train or select checkpoints."""
import fcntl
import hashlib
import json
import math
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

ROOT = Path(__file__).resolve().parents[1]


class CountImages(Hook):
    def __init__(self):
        self.ids = []

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.ids.extend(int(sample.img_id) for sample in outputs)


def main():
    work = ROOT / 'work_dirs/uiis_e1_bll_r50_36e'
    output = ROOT / 'reports/uiis-e1-bll-36e-evaluation'
    output.mkdir(parents=True, exist_ok=False)
    with (ROOT / 'work_dirs/uiis-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'metrics.json').exists():
            raise RuntimeError('Completed evaluation exists; refusing to overwrite')
        checkpoint = work / 'epoch_36.pth'
        saved = torch.load(checkpoint, map_location='cpu')
        assert saved['meta']['epoch'] == 36 and saved['meta']['iter'] == 70884
        completion=json.loads((work/'result-resume.json').read_text())
        assert completion['completed_epochs']==36 and completion['completed_iterations']==70884
        assert completion['resume_verified'] and not completion['verification']
        protocol=json.loads((work/'run-protocol.json').read_text())
        assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in protocol['source_sha256'].items())
        assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
        assert len(saved['optimizer']['state'])==345
        assert all(int(v['step'])==70884 for v in saved['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for state in saved['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v))
        initial_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        expected_state = saved['state_dict']
        del saved
        annotations = ROOT / 'data/UIIS/annotations/val.json'
        assert hashlib.sha256(annotations.read_bytes()).hexdigest() == '5ed740baa1aa857058285d5947aa480774902af596e8b7effab50e631960e4a2'
        test = json.loads(annotations.read_text())
        expected_ids = {int(item['id']) for item in test['images']}
        assert len(test['images']) == len(expected_ids) == 691
        cfg = Config.fromfile(str(work / 'config-resume.py'))
        baseline = Config.fromfile(str(ROOT / 'work_dirs/uiis_baseline_r50_36e/config-resume.py'))
        for key in ('test_dataloader', 'test_evaluator', 'test_cfg'):
            assert cfg[key] == baseline[key], key
        assert cfg.model.test_cfg == baseline.model.test_cfg
        assert cfg.model.data_preprocessor == baseline.model.data_preprocessor
        cfg.load_from = str(checkpoint)
        cfg.resume = False
        cfg.work_dir = str(output / 'runner')
        cfg.test_evaluator.classwise = True
        cfg.test_evaluator.outfile_prefix = str(output / 'predictions')
        cfg.dump(str(output / 'evaluation-config.py'))
        register_all_modules()
        runner = Runner.from_cfg(cfg)
        class CheckLoaded(Hook):
            def before_test_epoch(self, runner):
                actual = runner.model.state_dict()
                assert actual.keys() == expected_state.keys()
                assert all(torch.equal(actual[k].detach().cpu(), v) for k, v in expected_state.items())
                print('E1_CHECKPOINT_EXACT_LOAD_OK', len(actual), flush=True)
        runner.register_hook(CheckLoaded())
        counter = CountImages()
        runner.register_hook(counter)
        start = time.time()
        metrics = runner.test()
        assert len(counter.ids) == 691 and set(counter.ids) == expected_ids
        metrics = {key: float(value) for key, value in metrics.items()}
        assert metrics and all(math.isfinite(value) for value in metrics.values())
        result = dict(completed=True, evaluated_images=len(counter.ids), epoch=36,
                      iterations=70884, checkpoint=str(checkpoint),
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      annotations_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),
                      seconds=time.time() - start, metrics=metrics,
                      protocol='Fixed final epoch36, official 691-image test split and duplicate retained. Same-budget E0 comparison; epoch1 test inspection disclosed; no checkpoint selection or tuning.')
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==initial_hash
        (output / 'metrics.json').write_text(json.dumps(result, indent=2))
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == result['checkpoint_sha256']
        print('UIIS_EVALUATION_OK', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
