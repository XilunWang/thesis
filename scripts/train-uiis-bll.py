"""BLL E1 Runner entry. Verification is isolated; formal launch requires explicit flag."""
import argparse
import fcntl
import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
import scripts.bll_mask2former_head as adapter

base=runpy.run_path(str(ROOT/'scripts/train-uiis.py'))
TrainingGuard=base['TrainingGuard']
build_config=runpy.run_path(str(ROOT/'scripts/prepare-uiis-bll.py'))['build_config']
PRETRAIN_HASH='41b088b62838c986004f94bab2870395b669ec6bd11b6590369b838b0de06879'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


class BLLGuard(TrainingGuard):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.load_check={}
        self.empty_streak=0

    def after_load_checkpoint(self,runner,checkpoint):
        current=runner.model.state_dict()
        incoming={k.removeprefix('module.'):v for k,v in checkpoint['state_dict'].items()}
        assert set(current)==set(incoming), 'Unexpected pretrained/checkpoint parameter keys'
        mismatch=[k for k in current if current[k].shape!=incoming[k].shape]
        expected=[] if self.resume_checkpoint else ['panoptic_head.cls_embed.weight','panoptic_head.cls_embed.bias']
        assert set(mismatch)==set(expected),mismatch
        self.initial_class={k:current[k].detach().cpu().clone() for k in mismatch}
        self.incoming=incoming
        self.load_check=dict(mismatched_keys=mismatch,compatible_tensors=len(current)-len(mismatch))

    def before_train(self,runner):
        current=runner.model.state_dict()
        for k,v in self.incoming.items():
            if k in self.initial_class:
                assert torch.equal(current[k].cpu(),self.initial_class[k])
                assert current[k].shape[0]==8 and v.shape[0]==81
            else:
                assert torch.equal(current[k].cpu(),v.cpu()),k
        self.load_check.update(compatible_values_exact=True,class_initialization_retained=not bool(self.resume_checkpoint))
        del self.incoming,self.initial_class
        super().before_train(runner)
        if self.resume_checkpoint:
            self.empty_streak=torch.load(self.resume_checkpoint,map_location='cpu').get('bll_guard',{}).get('empty_streak',0)
        self.original_helper=adapter.query_boundary_loss
        def checked(previous,final,gts,matches,shapes,**kw):
            lost=0
            for gt,(vh,vw) in zip(gts,shapes):
                if not len(gt): continue
                h,w=final.shape[-2:]; hp,wp=gt.shape[-2:]
                assert hp%h==0 and wp%w==0
                small=F.interpolate(gt[:,None].float(),(h,w),mode='nearest')[:,0]
                lost+=int(((gt.flatten(1).sum(1)>0)&(small[:,:vh//(hp//h),:vw//(wp//w)].flatten(1).sum(1)==0)).sum())
            if lost: raise RuntimeError(f'BLL_GT_VANISHED: {lost}; stop before update and inspect')
            loss,diag=self.original_helper(previous,final,gts,matches,shapes,**kw)
            self.empty_streak=self.empty_streak+1 if sum(len(g) for g in gts)>0 and diag['selected_pixels']==0 else 0
            if self.empty_streak>=3: raise RuntimeError('BLL_EMPTY_REGION_STREAK')
            self.last_diag=dict(matched=diag['matched'],selected_pixels=diag['selected_pixels'],vanished=lost)
            return loss,diag
        adapter.query_boundary_loss=checked

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        super().after_train_iter(runner,batch_idx,data_batch,outputs)
        if not all(torch.isfinite(p).all() for p in runner.model.parameters()):
            raise RuntimeError('NONFINITE_MODEL_PARAMETER')
        if runner.iter%50==0:
            runner.logger.info('BLL_DIAGNOSTIC %s',self.last_diag)

    def before_save_checkpoint(self,runner,checkpoint):
        checkpoint['bll_guard']=dict(empty_streak=self.empty_streak)

    def after_train(self,runner):
        adapter.query_boundary_loss=self.original_helper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['first','resume'],required=True)
    p.add_argument('--work-dir',required=True)
    p.add_argument('--verification',action='store_true')
    p.add_argument('--execute-formal',action='store_true')
    args=p.parse_args()
    if args.verification==args.execute_formal:
        p.error('Choose exactly one: --verification or explicit --execute-formal')
    work=Path(args.work_dir).resolve()
    if work.parent!=ROOT/'work_dirs': raise ValueError('work-dir must be a direct project work_dirs child')
    if args.verification and not work.name.startswith('e1_entry_verification_'): raise ValueError('Verification directory prefix required')
    if args.execute_formal and work.name!='uiis_e1_bll_r50_36e': raise ValueError('Formal E1 directory required')
    with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('TRAINING_ALREADY_RUNNING')
        register_all_modules()
        cfg=build_config(); cfg.work_dir=str(work)
        assert sha(cfg.load_from)==PRETRAIN_HASH
        archives=[12,24,36]
        if args.verification:
            cfg.train_dataloader.dataset.indices=4
            cfg.train_cfg.max_epochs=5
            cfg.param_scheduler[0].update(end=5,milestones=[2,4])
            cfg.default_hooks.logger.interval=1
            archives=[1,3,5]
        source_files=['scripts/train-uiis-bll.py','scripts/prepare-uiis-bll.py','scripts/train-uiis.py',
                      'scripts/prepare-uiis-baseline.py','scripts/bll_mask2former_head.py','scripts/query_bll.py','scripts/bll_core.py']
        protocol=dict(verification=args.verification,config=cfg.pretty_text,pretrained_sha256=PRETRAIN_HASH,
            train_sha256=sha(ROOT/'data/UIIS/annotations/train.json'),
            val_sha256=sha(ROOT/'data/UIIS/annotations/val.json'),source_sha256={f:sha(ROOT/f) for f in source_files})
        marker=work/'run-protocol.json'; checkpoint=None
        if args.stage=='first':
            if work.exists() and any(work.iterdir()): raise ValueError('Refusing to overwrite existing run')
            work.mkdir(parents=True,exist_ok=True); marker.write_text(json.dumps(protocol,indent=2))
        else:
            if json.loads(marker.read_text())!=protocol: raise ValueError('Run protocol differs')
            checkpoint=Path((work/'last_checkpoint').read_text().strip()).resolve()
            if checkpoint.parent!=work or not checkpoint.is_file(): raise ValueError('Invalid checkpoint')
            meta=torch.load(checkpoint,map_location='cpu')['meta']
            if meta['epoch']>=cfg.train_cfg.max_epochs: raise ValueError('Run already reached budget')
            cfg.resume=True; cfg.load_from=str(checkpoint)
        cfg.dump(str(work/f'config-{args.stage}.py'))
        runner=Runner.from_cfg(cfg)
        guard=BLLGuard(1 if args.stage=='first' else None,archives,checkpoint)
        runner.register_hook(guard)
        runner.train()
        assert runner.epoch==(1 if args.stage=='first' else cfg.train_cfg.max_epochs)
        result=dict(stage=args.stage,verification=args.verification,completed_epochs=runner.epoch,
            completed_iterations=runner.iter,resume_verified=guard.resume_verified,load_check=guard.load_check,
            lrs=sorted(set(g['lr'] for g in runner.optim_wrapper.optimizer.param_groups)))
        (work/f'result-{args.stage}.json').write_text(json.dumps(result,indent=2))
        print('BLL_RUNNER_STAGE_OK',json.dumps(result),flush=True)


if __name__=='__main__': main()
