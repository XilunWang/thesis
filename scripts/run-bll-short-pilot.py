"""Paired 16-image/20-step technical pilot; four fresh worker processes, no AP."""
import argparse
import copy
import hashlib
import json
import os
import random
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
WORK=ROOT/'work_dirs/bll_short_pilot'
REPORT=ROOT/'reports/bll-short-pilot'
E0=ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth'
E0_HASH='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def worker(arm,resume):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from mmengine.dataset import pseudo_collate
    from mmengine.optim import build_optim_wrapper
    from mmdet.registry import MODELS,DATASETS
    from mmdet.utils import register_all_modules
    import scripts.bll_mask2former_head as adapter
    register_all_modules()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    manifest=json.loads((WORK/'manifest.json').read_text())
    manifest_hash=sha(WORK/'manifest.json')
    def budget():
        if time.time()>manifest['deadline_unix']: raise TimeoutError('Pilot total 600-second budget exceeded')
    budget()
    assert sha(E0)==E0_HASH
    assert sha(ROOT/'data/UIIS/annotations/train.json')==manifest['annotation_sha256']
    for f,h in manifest['sources'].items(): assert sha(ROOT/f)==h
    cfg=runpy.run_path(str(ROOT/'scripts/prepare-uiis-baseline.py'))['build_config']()
    if arm=='B': cfg.model.panoptic_head.update(type='BLLMask2FormerHead',bll_loss_weight=1.,bll_boundary_width=3)
    cfg_digest=hashlib.sha256(cfg.pretty_text.encode()).hexdigest()
    out=WORK/arm
    out.mkdir(exist_ok=True)
    model=MODELS.build(cfg.model).cuda().train()
    wrapper=build_optim_wrapper(model,copy.deepcopy(cfg.optim_wrapper))
    scheduler=torch.optim.lr_scheduler.LambdaLR(wrapper.optimizer,lr_lambda=lambda _:1.)
    dataset=DATASETS.build(cfg.train_dataloader.dataset)
    dataset.full_init()
    indices=sorted(range(len(dataset)),key=lambda i:dataset.get_data_info(i)['img_id'])[:16]
    info=[dataset.get_data_info(i) for i in indices]
    selected=[dict(id=d['img_id'],filename=Path(d['img_path']).name,sha256=sha(d['img_path'])) for d in info]
    assert selected==manifest['images']
    def seed(n):
        random.seed(n); np.random.seed(n); torch.manual_seed(n)
    def rng():
        return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    def restore_rng(r):
        random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch']); torch.cuda.set_rng_state_all(r['cuda'])
    def fingerprint(x):
        h=hashlib.sha256()
        def visit(v):
            if torch.is_tensor(v):
                h.update(str((v.dtype,tuple(v.shape))).encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(v,np.ndarray):
                h.update(str((v.dtype,v.shape)).encode()); h.update(v.tobytes())
            elif isinstance(v,dict):
                for k in sorted(v,key=str): h.update(repr(k).encode()); visit(v[k])
            elif isinstance(v,(tuple,list)):
                for a in v: visit(a)
            else: h.update(repr(v).encode())
        visit(x); return h.hexdigest()
    def state(completed,empty):
        return dict(model=model.state_dict(),optimizer=wrapper.state_dict(),scheduler=scheduler.state_dict(),
                    inner_count=wrapper._inner_count,rng=rng(),completed=completed,cursor=(2*completed)%16,
                    empty_streak=empty,manifest_hash=manifest_hash,config_hash=cfg_digest,arm=arm)
    start=0; empty_streak=0
    if resume:
        saved=torch.load(out/'step_10.pth',map_location='cpu')
        assert saved['completed']==10 and saved['cursor']==4 and saved['arm']==arm
        assert saved['manifest_hash']==manifest_hash and saved['config_hash']==cfg_digest
        model.load_state_dict(saved['model'],strict=True)
        wrapper.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        wrapper._inner_count=saved['inner_count']; restore_rng(saved['rng'])
        start=10; empty_streak=saved['empty_streak']
        assert fingerprint(state(start,empty_streak))==json.loads((out/'step_10_state.json').read_text())['fingerprint']
        write_json(out/'resume-check.json',dict(passed=True,completed=10,next_step=11,cursor=4,pid=os.getpid()))
        del saved
    else:
        assert not (out/'steps.jsonl').exists()
        model.load_state_dict(torch.load(E0,map_location='cpu')['state_dict'],strict=True)
        cfg.dump(str(out/'config.py'))
    stop=20 if resume else 10
    match_counts=[]; helper_diag=[]; grid=[]
    target_method=model.panoptic_head._get_targets_single
    def observe(*a,**kw):
        value=target_method(*a,**kw); match_counts.append(int(value[6].pos_inds.numel())); return value
    model.panoptic_head._get_targets_single=observe
    helper=adapter.query_boundary_loss
    def observe_helper(*a,**kw):
        loss,diag=helper(*a,**kw); helper_diag.append(diag); return loss,diag
    adapter.query_boundary_loss=observe_helper
    def head_hook(module,args,out): grid.append(tuple(out[1][-1].shape[-2:]))
    hook=model.panoptic_head.register_forward_hook(head_hook)
    try:
        for step in range(start,stop):
            budget(); packed=[]; ids=[]
            for slot in range(2):
                position=(2*step+slot)%16; seed(20262000+2*step+slot)
                item=dataset.prepare_data(indices[position])
                assert item is not None and item['data_samples'].img_id==info[position]['img_id']
                packed.append(item); ids.append(info[position]['img_id'])
            seed(20263000+step)
            data=model.data_preprocessor(pseudo_collate(packed),training=True)
            # Fingerprint tensors/GT explicitly; DataSample repr is not a content hash.
            input_hash=fingerprint(dict(inputs=data['inputs'],samples=[dict(meta=s.metainfo,
                labels=s.gt_instances.labels,masks=s.gt_instances.masks.masks) for s in data['data_samples']]))
            match_counts.clear(); helper_diag.clear(); grid.clear()
            wrapper.zero_grad(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); began=time.perf_counter()
            losses=model(**data,mode='loss')
            assert all(torch.isfinite(v).all() for v in losses.values())
            gt_counts=[]; vanished=[]
            for s in data['data_samples']:
                gt=s.gt_instances.masks.to_tensor(dtype=torch.float32,device='cuda')
                vh,vw=s.img_shape[:2]; hp,wp=gt.shape[-2:]; h,w=grid[-1]
                assert hp%h==0 and wp%w==0
                assert not gt[:,vh:,:].any() and not gt[:,:,vw:].any()
                gt_counts.append(len(gt))
                small=F.interpolate(gt[:,None],(h,w),mode='nearest')[:,0] if len(gt) else gt.new_empty((0,h,w))
                lost=((gt.flatten(1).sum(1)>0)&(small[:,:vh//(hp//h),:vw//(wp//w)].flatten(1).sum(1)==0)).nonzero().flatten().tolist()
                vanished.append(lost)
            selected=helper_diag[-1]['selected_pixels'] if helper_diag else None
            if arm=='B' and sum(gt_counts)>0: empty_streak=empty_streak+1 if selected==0 else 0
            else: empty_streak=0
            row=dict(step=step+1,ids=ids,input_sha256=input_hash,gt_counts=gt_counts,matched=sum(match_counts[-2:]),
                     vanished_gt_indices=vanished,selected_pixels=selected,losses={k:float(v.detach()) for k,v in losses.items()},
                     lr=sorted(set(g['lr'] for g in wrapper.optimizer.param_groups)))
            reason=None
            if arm=='B' and any(vanished): reason='nonempty GT vanished on BLL grid'
            if empty_streak>=3: reason='three nonempty-GT steps without BLL selected pixels'
            if reason:
                row['stopped_before_update']=reason
                with (out/'steps.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                raise RuntimeError(reason)
            wrapper.backward(sum(losses.values()))
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            row['preclip_gradient_norm']=float(torch.stack([g.norm() for g in grads]).norm())
            wrapper.step(); wrapper.zero_grad(); scheduler.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            torch.cuda.synchronize()
            row.update(seconds=time.perf_counter()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            with (out/'steps.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            print(arm,'STEP',step+1,'BLL',row['losses'].get('loss_bll'),flush=True)
        budget()
        payload=state(stop,empty_streak)
        checkpoint=out/f'step_{stop}.pth'
        assert not checkpoint.exists()
        digest=fingerprint(payload)
        torch.save(payload,checkpoint)
        write_json(out/f'step_{stop}_state.json',dict(fingerprint=digest,checkpoint_sha256=sha(checkpoint),pid=os.getpid()))
        assert sha(E0)==E0_HASH
    finally:
        hook.remove(); adapter.query_boundary_loss=helper


def orchestrate():
    import fcntl
    REPORT.mkdir(parents=True,exist_ok=True)
    if (REPORT/'result.json').exists() or WORK.exists(): raise FileExistsError('Existing pilot; refusing overwrite/restart')
    result=dict(passed=False,scope='technical 16-image paired pilot; not AP/E1')
    with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        WORK.mkdir()
        try:
            started=time.time()
            annotations=json.loads((ROOT/'data/UIIS/annotations/train.json').read_text())
            images=sorted(annotations['images'],key=lambda d:d['id'])[:16]
            manifest=dict(start_unix=started,deadline_unix=started+600,e0_sha256=E0_HASH,
                annotation_sha256=sha(ROOT/'data/UIIS/annotations/train.json'),
                images=[dict(id=i['id'],filename=Path(i['file_name']).name,
                    sha256=sha(ROOT/'data/UIIS/train'/i['file_name'])) for i in images],
                sources={f:sha(ROOT/f) for f in ['scripts/run-bll-short-pilot.py','scripts/prepare-uiis-baseline.py',
                    'scripts/bll_mask2former_head.py','scripts/query_bll.py','scripts/bll_core.py','docs/bll-short-pilot-protocol.md']},
                steps_per_arm=20,batch_size=2,seed_data='20262000+2*step+slot',seed_model='20263000+step',
                scheduler='LambdaLR constant 1, stepped per update; no epoch decay',num_workers=0)
            write_json(WORK/'manifest.json',manifest)
            for arm in ['A','B']:
                for resume in [False,True]:
                    remaining=manifest['deadline_unix']-time.time()
                    if remaining<=0: raise TimeoutError('Total pilot budget exceeded')
                    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',arm]
                    if resume: command.append('--resume')
                    log=REPORT/(f'{arm}-'+('resume.log' if resume else 'first.log'))
                    with log.open('w') as f:
                        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=remaining,check=True)
            rows={arm:[json.loads(line) for line in (WORK/arm/'steps.jsonl').read_text().splitlines()] for arm in ['A','B']}
            for arm,seq in rows.items():
                assert [r['step'] for r in seq]==list(range(1,21))
                assert json.loads((WORK/arm/'resume-check.json').read_text())['passed']
                assert all('stopped_before_update' not in r for r in seq)
            assert [r['input_sha256'] for r in rows['A']]==[r['input_sha256'] for r in rows['B']]
            assert sha(E0)==E0_HASH
            result.update(passed=True,elapsed_seconds=time.time()-started,paired_inputs_exact=True,e0_unchanged=True,
                          arms={arm:dict(steps=len(seq),first_losses=seq[0]['losses'],last_losses=seq[-1]['losses'],
                              max_gradient_norm=max(r['preclip_gradient_norm'] for r in seq),
                              max_memory_bytes=max(r['peak_allocated_bytes'] for r in seq)) for arm,seq in rows.items()})
        except Exception:
            result['error']=traceback.format_exc(); raise
        finally:
            write_json(REPORT/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',choices=['A','B']); p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.worker: worker(a.worker,a.resume)
    else: orchestrate()
