"""Native Runner save/resume in fresh process; no train() or optimizer step."""
import argparse
import gc
import hashlib
import json
import random
import runpy
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
W=R/'work_dirs/pixel_bll_save_restore_verified'
O=R/'reports/bll-pixel-runner-checkpoint-verified-20261031'
SOURCE=R/'work_dirs/uiis_e1_bll_r50_36e/epoch_36.pth'
EXPECTED='6d89f04575ec2a28b441c3f681e38627cf2273638de3ee6394bedb03279e8176'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fingerprint(obj):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            y=x.detach().cpu().contiguous();h.update(str((y.dtype,tuple(y.shape))).encode());h.update(y.numpy().tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(obj);return h.hexdigest()


def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def snapshot(runner):
    return dict(model=fingerprint(runner.model.state_dict()),optimizer=fingerprint(runner.optim_wrapper.state_dict()),
                schedulers=fingerprint([s.state_dict() for s in runner.param_schedulers]),
                epoch=runner.epoch,iteration=runner.iter,rng=fingerprint(rng()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');args=parser.parse_args()
    from mmengine.runner import Runner
    from mmengine.dataset import pseudo_collate
    from mmdet.utils import register_all_modules
    import scripts.pixel_bll_head  # In-memory Config does not execute custom_imports.
    register_all_modules();torch.set_num_threads(2)
    if not args.child:W.mkdir(exist_ok=False);O.mkdir(exist_ok=False)
    assert sha(SOURCE)==EXPECTED
    cfg=runpy.run_path(str(R/'scripts/prepare-uiis-bll.py'))['build_config']()
    cfg.model.panoptic_head.type='PixelBLLMask2FormerHead'
    cfg.custom_imports=dict(imports=['scripts.pixel_bll_head'],allow_failed_imports=False)
    cfg.load_from=None;cfg.resume=False
    cfg.work_dir=str(W/('child' if args.child else 'parent'))
    cfg.train_dataloader.num_workers=0;cfg.train_dataloader.persistent_workers=False
    runner=Runner.from_cfg(cfg)
    target=W/'isolated_pixel_probe.pth'
    # Seed nonempty AdamW moments from existing E1 only for serialization testing.
    # No formal initialization/resume authorization is implied.
    runner.resume(str(target if args.child else SOURCE))
    assert runner.epoch==36 and runner.iter==70884
    assert type(runner.model.panoptic_head).__name__=='PixelBLLMask2FormerHead'
    assert runner.model.panoptic_head.bll_loss_weight==1
    assert len(runner.optim_wrapper.optimizer.state)==345
    if not args.child:
        source=torch.load(SOURCE,map_location='cpu')
        assert fingerprint(source['state_dict'])==fingerprint(runner.model.state_dict())
        assert fingerprint(source['optimizer'])==fingerprint(runner.optim_wrapper.state_dict())
        del source
        saved_rng=rng();before=snapshot(runner)
        runner.save_checkpoint(str(W),target.name,meta=dict(epoch=runner.epoch,iter=runner.iter,
             diagnostic_only=True,candidate='PixelBLL-v1',training_updates=0))
        torch.save(saved_rng,W/'rng-sidecar.pth')
        assert snapshot(runner)==before
        (W/'expected.json').write_text(json.dumps(before,indent=2))
        digest=sha(target)
        del runner;gc.collect();torch.cuda.empty_cache()
        subprocess.run([sys.executable,'-B',__file__,'--child'],check=True)
        child=json.loads((O/'child.json').read_text());assert child['passed']
        result=dict(passed=True,child=child,checkpoint_sha256=digest,source_unchanged=sha(SOURCE)==EXPECTED,
            native_runner_save_resume=True,optimizer_updates=0,
            scope='Existing nonempty E1 optimizer state used only as serialization fixture. No optimizer.step, train-loop iteration, data-worker cursor, or continuation equivalence check.')
        assert result['source_unchanged']
        (O/'result.json').write_text(json.dumps(result,indent=2));print('RUNNER_SAVE_RESTORE_PASS',flush=True)
    else:
        saved=torch.load(W/'rng-sidecar.pth',map_location='cpu')
        random.setstate(saved['python']);np.random.set_state(saved['numpy']);torch.set_rng_state(saved['torch']);torch.cuda.set_rng_state_all(saved['cuda'])
        loaded=snapshot(runner);assert loaded==json.loads((W/'expected.json').read_text())
        dataset=runner.train_dataloader.dataset
        index=next(i for i in range(len(dataset)) if dataset.get_data_info(i)['img_id']==22)
        random.seed(20261043);np.random.seed(20261043);torch.manual_seed(20261043)
        item=dataset.prepare_data(index);assert item is not None
        data=runner.model.data_preprocessor(pseudo_collate([item]),training=True)
        runner.model.train();runner.optim_wrapper.zero_grad()
        losses=runner.model(**data,mode='loss');assert 'loss_bll_pixel' in losses
        sum(losses.values()).backward()
        assert all(torch.isfinite(p.grad).all() for p in runner.model.parameters() if p.grad is not None)
        after=snapshot(runner)
        for k in ['model','optimizer','schedulers','epoch','iteration']:assert loaded[k]==after[k],k
        result=dict(passed=True,restored_exact=loaded,optimizer_states=345,restored_forward_backward_finite=True,
            loss_pixel=float(losses['loss_bll_pixel'].detach()),state_unchanged_after_backward=True,
            rng_restored_via_explicit_sidecar=True,optimizer_updates=0)
        (O/'child.json').write_text(json.dumps(result,indent=2));print('CHILD_PASS',flush=True)


if __name__=='__main__':main()
