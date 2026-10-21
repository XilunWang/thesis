"""Isolated disk/new-process checkpoint test, not a training run or Runner resume."""
import argparse
import copy
import hashlib
import json
import random
import runpy
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
WORK = ROOT/'work_dirs/bll_checkpoint_verification'
REPORT = ROOT/'reports/bll-checkpoint/validation.json'
E0 = ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth'
E0_HASH = '53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def run(resume,result):
    import numpy as np
    import torch
    from mmengine.optim import build_optim_wrapper
    from mmengine.registry import PARAM_SCHEDULERS
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import MODELS, DATASETS
    from mmdet.utils import register_all_modules
    import scripts.bll_mask2former_head
    register_all_modules()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    assert file_hash(E0)==E0_HASH
    def fingerprint(value):
        h=hashlib.sha256()
        def visit(x):
            if torch.is_tensor(x):
                h.update(str((x.dtype,tuple(x.shape))).encode())
                h.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x,dict):
                for k in sorted(x,key=str):
                    h.update(repr(k).encode()); visit(x[k])
            elif isinstance(x,(tuple,list)):
                h.update(type(x).__name__.encode())
                for v in x: visit(v)
            elif isinstance(x,np.ndarray):
                h.update(str((x.dtype,x.shape)).encode()); h.update(x.tobytes())
            else:
                h.update(repr(x).encode())
        visit(value)
        return h.hexdigest()
    def rng():
        return dict(python=random.getstate(),numpy=np.random.get_state(),
                    torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    def set_rng(r):
        random.setstate(r['python']); np.random.set_state(r['numpy'])
        torch.set_rng_state(r['torch']); torch.cuda.set_rng_state_all(r['cuda'])
    cfg=runpy.run_path(str(ROOT/'scripts/prepare-uiis-baseline.py'))['build_config']()
    cfg.model.panoptic_head.update(type='BLLMask2FormerHead',bll_loss_weight=1.,bll_boundary_width=3)
    model=MODELS.build(cfg.model).cuda().train()
    wrapper=build_optim_wrapper(model,copy.deepcopy(cfg.optim_wrapper))
    schedulers=[PARAM_SCHEDULERS.build(copy.deepcopy(s),default_args=dict(optimizer=wrapper)) for s in cfg.param_scheduler]
    def snapshot():
        return dict(model=fingerprint(model.state_dict()),optimizer=fingerprint(wrapper.state_dict()),
                    schedulers=fingerprint([s.state_dict() for s in schedulers]),
                    inner_count=wrapper._inner_count,rng=fingerprint(rng()))
    def real_backward():
        dataset=DATASETS.build(cfg.train_dataloader.dataset)
        items=[]
        for i in [0,1]:
            random.seed(20261021+i); np.random.seed(20261021+i); torch.manual_seed(20261021+i)
            items.append(dataset.prepare_data(i))
        torch.manual_seed(20261100)
        data=model.data_preprocessor(pseudo_collate(items),training=True)
        wrapper.zero_grad()
        losses=model(**data,mode='loss')
        assert all(torch.isfinite(v).all() for v in losses.values())
        wrapper.backward(sum(losses.values()))
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        return {k:float(v.detach()) for k,v in losses.items()}
    def controlled_step(names):
        wrapper.zero_grad()
        for n,p in model.named_parameters():
            if n in names:
                p.grad=torch.full_like(p,1e-4)
        wrapper.step()  # Same clipping and optimizer as configured, fixed gradients.
        wrapper.zero_grad()
    checkpoint=WORK/'checkpoint.pth'
    if not resume:
        model.load_state_dict(torch.load(E0,map_location='cpu')['state_dict'],strict=True)
        before=fingerprint(model.state_dict())
        losses=real_backward()
        names=[n for n,p in model.named_parameters() if p.grad is not None]
        wrapper.step(); wrapper.zero_grad()
        for scheduler in schedulers: scheduler.step()
        assert fingerprint(model.state_dict())!=before
        assert wrapper.optimizer.state
        saved=dict(state_dict=model.state_dict(),optim_wrapper=wrapper.state_dict(),
                   schedulers=[s.state_dict() for s in schedulers],rng=rng(),inner_count=wrapper._inner_count,
                   names=names,meta=dict(real_updates=1,virtual_scheduler_ticks=1,bll_weight=1.,width=3,
                                        e0_sha256=E0_HASH))
        before_save=snapshot()
        torch.save(saved,checkpoint)
        (WORK/'before.json').write_text(json.dumps(before_save,indent=2))
        controlled_step(names)
        after=snapshot()
        (WORK/'expected-next.json').write_text(json.dumps(after,indent=2))
        result.update(first_real_losses=losses,gradient_parameters=len(names),
                      optimizer_states=len(wrapper.optimizer.state),checkpoint_sha256=file_hash(checkpoint),
                      before_save=before_save,expected_controlled_step=after)
        del saved,model,wrapper,schedulers
        torch.cuda.empty_cache()
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--resume-child'],check=True)
        child=json.loads((WORK/'child-result.json').read_text())
        assert child['passed']
        result.update(passed=True,child=child,e0_unchanged=file_hash(E0)==E0_HASH)
        assert result['e0_unchanged']
    else:
        saved=torch.load(checkpoint,map_location='cpu')
        assert saved['meta']['e0_sha256']==E0_HASH and saved['meta']['bll_weight']==1.
        model.load_state_dict(saved['state_dict'],strict=True)
        wrapper.load_state_dict(saved['optim_wrapper'])
        wrapper._inner_count=saved['inner_count']
        for scheduler,state in zip(schedulers,saved['schedulers']): scheduler.load_state_dict(state)
        set_rng(saved['rng'])
        loaded=snapshot()
        assert loaded==json.loads((WORK/'before.json').read_text())
        controlled_step(saved['names'])
        continued=snapshot()
        assert continued==json.loads((WORK/'expected-next.json').read_text())
        param_before=fingerprint(dict(model.named_parameters()))
        finite_losses=real_backward()  # Fresh real forward/backward after restore; no step.
        assert fingerprint(dict(model.named_parameters()))==param_before
        result.update(passed=True,loaded_exact=True,controlled_next_step_exact=True,
                      restored_real_losses=finite_losses,e0_unchanged=file_hash(E0)==E0_HASH)
        assert result['e0_unchanged']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-child',action='store_true')
    args=parser.parse_args()
    if not args.resume_child:
        if WORK.exists() or REPORT.exists(): raise FileExistsError('Refusing to overwrite checkpoint verification')
        WORK.mkdir(parents=True); REPORT.parent.mkdir(parents=True,exist_ok=True)
    output=WORK/'child-result.json' if args.resume_child else REPORT
    result=dict(passed=False,scope='one real update + fixed-gradient continuation on isolated copies; not Runner resume')
    try: run(args.resume_child,result)
    except Exception:
        result['error']=traceback.format_exc(); raise
    finally:
        result['script_sha256']=file_hash(__file__)
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('CHECKPOINT_CHECK',result['passed'],flush=True)


if __name__=='__main__': main()
