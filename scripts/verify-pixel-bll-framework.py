"""One actual model forward, shared RNG/loss checks, backward only; no optimizer."""
import fcntl
import hashlib
import json
import random
import runpy
import sys
import types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))


def main():
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import MODELS,DATASETS
    from mmdet.models.dense_heads import Mask2FormerHead
    from mmdet.utils import register_all_modules
    import scripts.pixel_bll_head as module
    from scripts.pixel_bll import build_bll_targets
    from scripts.bll_core import boundary
    out=R/'reports/bll-pixel-framework-20261031';out.mkdir(exist_ok=False)
    with (R/'work_dirs/uiis-training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        register_all_modules();torch.set_num_threads(2)
        random.seed(20261043);np.random.seed(20261043);torch.manual_seed(20261043)
        cfg=runpy.run_path(str(R/'scripts/prepare-uiis-bll.py'))['build_config']()
        cfg.model.panoptic_head.type='PixelBLLMask2FormerHead'
        dataset=DATASETS.build(cfg.train_dataloader.dataset);dataset.full_init()
        index=next(i for i in range(len(dataset)) if dataset.get_data_info(i)['img_id']==22)
        item=dataset.prepare_data(index);assert item is not None
        checkpoint=R/'work_dirs/uiis_e1_bll_r50_36e/epoch_36.pth'
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest=='6d89f04575ec2a28b441c3f681e38627cf2273638de3ee6394bedb03279e8176'
        state=torch.load(checkpoint,map_location='cpu')['state_dict']
        model=MODELS.build(cfg.model).cuda().train();model.load_state_dict(state,strict=True)
        assert set(model.state_dict())==set(state)
        head=model.panoptic_head;original_method=head.loss_by_feat;helper=module.pixel_boundary_loss
        report={}
        def tap(previous,final,gts,pairs,shapes,**kwargs):
            loss,diag=helper(previous,final,gts,pairs,shapes,**kwargs)
            qi,gi=pairs[0];gt=gts[0][gi].float();hp,wp=gt.shape[-2:];h,w=final.shape[-2:];vh,vw=shapes[0]
            with torch.no_grad():
                target,valid=build_bll_targets(gt,(h,w),(vh,vw))
                region=(boundary((previous[0,qi].sigmoid()>=.5)&valid)|boundary(target))&valid
                selected=region.repeat_interleave(hp//h,1).repeat_interleave(wp//w,2)
                valid_pixels=torch.zeros(hp,wp,dtype=torch.bool,device=final.device);valid_pixels[:vh,:vw]=True
                selected &= valid_pixels
            up=F.interpolate(final[0,qi,None],(hp,wp),mode='bilinear',align_corners=False)[:,0]
            direct=F.binary_cross_entropy_with_logits(up,gt,reduction='none')[selected].mean()
            torch.testing.assert_close(loss,direct,rtol=2e-6,atol=1e-7)
            ga,=torch.autograd.grad(loss,final,retain_graph=True)
            gb,=torch.autograd.grad(direct,final,retain_graph=True)
            torch.testing.assert_close(ga,gb,rtol=2e-5,atol=1e-8)
            report.update(helper_matches_direct=True,diag=diag,areas=gt.sum((1,2)).tolist(),
                selected_foreground=(gt.bool()&selected).sum((1,2)).tolist(),loss_pixel=float(loss.detach()))
            return loss,diag
        def wrapped(self,*args,**kwargs):
            cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
            parent=Mask2FormerHead.loss_by_feat(self,*args,**kwargs)
            torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);self.bll_loss_weight=0
            disabled=original_method(*args,**kwargs)
            assert set(parent)==set(disabled)
            for k in parent:assert torch.equal(parent[k],disabled[k]),k
            torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);self.bll_loss_weight=1
            enabled=original_method(*args,**kwargs)
            assert set(enabled)==set(parent)|{'loss_bll_pixel'}
            for k in parent:assert torch.equal(parent[k],enabled[k]),k
            report.update(disabled_losses_exact=True,enabled_original_losses_exact=True)
            return enabled
        module.pixel_boundary_loss=tap;head.loss_by_feat=types.MethodType(wrapped,head)
        try:
            data=model.data_preprocessor(pseudo_collate([item]),training=True)
            torch.cuda.reset_peak_memory_stats()
            losses=model(**data,mode='loss');sum(losses.values()).backward()
        finally:
            module.pixel_boundary_loss=helper;head.loss_by_feat=original_method
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
        report.update(passed=True,image_id=22,seed=20261043,model_state_unchanged=True,
            checkpoint_sha256=digest,gradient_tensors=len(grads),all_parameter_gradients_finite=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_created=False,training_updates=0,
            scope='One actual training augmentation, shared forward and RNG. No repeated-backward bitwise claim, no Runner/checkpoint-resume/AP validation.')
        (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
