"""Evaluate the frozen E1 epoch-one checkpoint; never train or select checkpoints."""
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
    output = ROOT / 'reports/uiis-e1-bll-epoch1-evaluation'
    output.mkdir(parents=True, exist_ok=False)
    with (ROOT / 'work_dirs/uiis-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'metrics.json').exists():
            raise RuntimeError('Completed evaluation exists; refusing to overwrite')
        checkpoint = work / 'epoch_1.pth'
        saved = torch.load(checkpoint, map_location='cpu')
        assert saved['meta']['epoch'] == 1 and saved['meta']['iter'] == 1969
        expected_state = saved['state_dict']
        del saved
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == '4a40dc4a90b1e8e3e90e1ab538c0c74ae3081e186630e27603a6d84887e6e356'
        annotations = ROOT / 'data/UIIS/annotations/val.json'
        assert hashlib.sha256(annotations.read_bytes()).hexdigest() == '5ed740baa1aa857058285d5947aa480774902af596e8b7effab50e631960e4a2'
        test = json.loads(annotations.read_text())
        expected_ids = {int(item['id']) for item in test['images']}
        assert len(test['images']) == len(expected_ids) == 691
        cfg = Config.fromfile(str(work / 'config-first.py'))
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
        result = dict(completed=True, evaluated_images=len(counter.ids), epoch=1,
                      iterations=1969, checkpoint=str(checkpoint),
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      annotations_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),
                      seconds=time.time() - start, metrics=metrics,
                      protocol='User-authorized exploratory epoch-one evaluation; official 691-image test split and duplicate retained. No training, tuning or checkpoint selection. Not a matched-budget comparison with E0 epoch36.')
        (output / 'metrics.json').write_text(json.dumps(result, indent=2))
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == result['checkpoint_sha256']
        print('UIIS_EVALUATION_OK', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
