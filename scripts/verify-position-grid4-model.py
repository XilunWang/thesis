"""Three fixed pilot images, frozen E0, no optimizer or parameter update."""
import fcntl
import hashlib
import json
import random
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS,MODELS
from mmdet.utils import register_all_modules
from pycocotools import mask as mu
from scripts.identity_data_pipeline import with_identity
from scripts.position_grid4 import Grid4PositionHead,make_targets,position_loss,select
OUT=ROOT/'reports/position-grid4-model-20261101'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main(result):
    torch.set_num_threads(2);random.seed(20261101);np.random.seed(20261101);torch.manual_seed(20261101);torch.cuda.manual_seed_all(20261101)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    register_all_modules()
    plan=json.loads((ROOT/'reports/position-learning-plan-20261101/manifest.json').read_text())
    chosen=[('learn',2809,20263821)]
    for group in ['dense','empty']:
        item=next(r for r in plan['splits']['holdout'] if r['stratum']==group)
        chosen.append(('holdout',item['image_id'],item['augmentation_seeds'][0]))
    result['selection']=chosen
    cfg=Config.fromfile(str(ROOT/'reports/uiis-baseline-36e-evaluation/evaluation-config.py'))
    ck=ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth'
    assert sha(ck)=='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
    initial=torch.load(ck,map_location='cpu')['state_dict']
    model=MODELS.build(cfg.model);model.load_state_dict(initial,strict=True);model.requires_grad_(False);model.cuda().eval()
    head=Grid4PositionHead().cuda();state={k:v.detach().clone() for k,v in head.state_dict().items()}
    traincfg=Config.fromfile(str(ROOT/'work_dirs/uiis_baseline_r50_36e/config-resume.py'))
    ds=DATASETS.build(with_identity(traincfg.train_dataloader.dataset));ds.full_init()
    index={ds.get_data_info(i)['img_id']:i for i in range(len(ds))}
    result['runs']=[]
    for split,iid,seed in chosen:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        item=ds.prepare_data(index[iid]);assert item is not None
        data=model.data_preprocessor(pseudo_collate([item]),training=True)
        sample=data['data_samples'][0];gt=sample.gt_instances
        p=ROOT/'reports/position-learning-preflight-20261101'/f'{split}-{iid}-{seed}.json';cached=json.loads(p.read_text())
        assert gt.annotation_ids.tolist()==cached['annotation_ids'] and gt.original_areas.tolist()==cached['original_areas']
        masks=gt.masks.masks
        expected=mu.decode(cached['gt_rles']).transpose(2,0,1) if cached['gt_rles'] else np.zeros_like(masks)
        assert np.array_equal(masks,expected) and list(sample.img_shape)==cached['valid_shape']
        t,v,a=make_targets(masks,gt.annotation_ids.tolist(),gt.original_areas.tolist(),sample.img_shape)
        with torch.no_grad():
            features=model.extract_feat(data['inputs'])
            before=model.panoptic_head(features,data['data_samples'])
            before=[[x.cpu().clone() for x in part] for part in before]
        head.zero_grad(set_to_none=True);logits=head(features[1],features[2]);logits.retain_grad()
        assert logits.shape==t[None].shape
        loss=position_loss(logits,t[None].cuda(),v[None].cuda());loss.backward()
        gradients={k:float(p.grad.norm()) for k,p in head.named_parameters()}
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        assert all(g>0 for g in gradients.values()) and not logits.grad[~v[None].cuda()].any()
        assert all(p.grad is None for p in model.parameters())
        assert all(torch.equal(state[k],val) for k,val in head.state_dict().items())
        cells,xy=select(logits[0].detach().cpu().numpy(),sample.img_shape,100)
        assert len(cells)==min(100,int(v.sum())) and len(set(cells.tolist()))==len(cells)
        assert np.all((xy>=0)&(xy<1))
        with torch.no_grad():after=model.panoptic_head(model.extract_feat(data['inputs']),data['data_samples'])
        assert all(torch.equal(x,y.cpu()) for p,q in zip(before,after) for x,y in zip(p,q))
        assert all(torch.equal(x.cpu(),initial[k]) for k,x in model.state_dict().items())
        row=dict(image_id=iid,seed=seed,positives=len(a),valid_cells=int(v.sum()),loss=float(loss.detach()),
                 logits_shape=list(logits.shape),selected=len(cells),branch_gradient_norms=gradients,
                 original_stage_logits_exact=True,base_gradients_absent=True,all_parameters_buffers_unchanged=True,
                 padding_gradients_zero=True,source_cache_sha256=sha(p))
        if iid==2809:row['collision_targets']={str(gid):a[gid] for gid in [16538,16539]};assert len(a)==54 and a[16538]!=a[16539]
        result['runs'].append(row);print('MODEL_ACCEPTED',iid,len(a),flush=True)
        del features,before,after,logits,loss,data
    result.update(passed=True,parameters=sum(p.numel() for p in head.parameters()),checkpoint_sha256=sha(ck),
                  code_sha256={n:sha(ROOT/'scripts'/n) for n in ['position_grid4.py','isolated_position_head.py','identity_data_pipeline.py','verify-position-grid4-model.py']})


if __name__=='__main__':
    OUT.mkdir(exist_ok=False);result=dict(passed=False,training_updates=0)
    try:
        with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(result)
    except BaseException:result['error']=traceback.format_exc();raise
    finally:(OUT/'result.json').write_text(json.dumps(result,indent=2))
