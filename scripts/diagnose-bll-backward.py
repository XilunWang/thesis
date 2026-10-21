"""Diagnose original E0 CUDA backward repeatability; no BLL or optimizer."""
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import traceback
import warnings

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[1]


def diagnose(result):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import MODELS, DATASETS
    from mmdet.utils import register_all_modules
    register_all_modules()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    def seed(n):
        random.seed(n)
        np.random.seed(n)
        torch.manual_seed(n)
    cfg = runpy.run_path(str(ROOT/'scripts/prepare-uiis-baseline.py'))['build_config']()
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    items = []
    for i in [0,1]:
        seed(20261021+i)
        items.append(dataset.prepare_data(i))
    model = MODELS.build(cfg.model).cuda().train()
    state = torch.load(ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth',map_location='cpu')['state_dict']
    model.load_state_dict(state,strict=True)
    data = model.data_preprocessor(pseudo_collate(items),training=True)
    refs = {}
    def backbone_hook(module,args,out):
        refs.update({f'backbone_feature_{i}':t for i,t in enumerate(out)})
    def head_hook(module,args,out):
        refs['final_masks'] = out[1][-1]
    hooks = [model.backbone.register_forward_hook(backbone_hook),
             model.panoptic_head.register_forward_hook(head_hook)]
    result.update(scope='Original E0 only, fixed real batch IDs 0/1, same graph repeated backward',
                  torch=torch.__version__, no_bll_imported=True)
    try:
        seed(20261100)
        torch.use_deterministic_algorithms(True)
        try:
            losses = model(**data,mode='loss')
            sum(losses.values()).backward()
            result['strict_determinism'] = 'passed'
        except RuntimeError as error:
            result['strict_determinism'] = str(error)
        finally:
            model.zero_grad(set_to_none=True)
            refs.clear()
            if 'losses' in locals():
                del losses
        seed(20261100)
        torch.use_deterministic_algorithms(True,warn_only=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            losses = model(**data,mode='loss')
            refs['backbone_conv1_weight'] = model.backbone.conv1.weight
            names = list(refs)
            targets = list(refs.values())
            total = sum(losses.values())
            a = torch.autograd.grad(total,targets,retain_graph=True)
            b = torch.autograd.grad(total,targets,retain_graph=True)
            result['same_graph'] = {n:dict(equal=torch.equal(x,y),
                max_abs_difference=float((x-y).abs().max()),
                max_abs_gradient=float(x.abs().max()),
                relative_l2=float((x-y).double().norm()/x.double().norm().clamp_min(1e-30)))
                for n,x,y in zip(names,a,b)}
            result['warnings'] = sorted(set(str(w.message) for w in caught))
        source = refs['final_masks'].detach()[:,:2].reshape(-1,1,256,256).contiguous()
        # Isolate the grid_sample operation used by MMDetection point_sample.
        # Input values are actual final logits; coordinates/cotangent fixed synthetically.
        seed(9)
        grid = torch.rand(source.shape[0],12544,1,2,device='cuda')*2-1
        cotangent = torch.randn(source.shape[0],1,12544,1,device='cuda')
        torch.use_deterministic_algorithms(False)
        result['isolated_grid_sample'] = {}
        for device in ['cuda','cpu']:
            x = source.to(device).detach().requires_grad_()
            y = F.grid_sample(x,grid.to(device),align_corners=False)
            gradients = [torch.autograd.grad(y,x,cotangent.to(device),retain_graph=True)[0] for _ in range(3)]
            result['isolated_grid_sample'][device] = dict(
                repeated_equal=all(torch.equal(gradients[0],g) for g in gradients[1:]),
                maximum_difference=max(float((gradients[0]-g).abs().max()) for g in gradients[1:]))
        result['completed'] = True
    finally:
        for h in hooks:
            h.remove()


def main():
    output = ROOT/'reports/bll-backward-diagnosis/diagnosis.json'
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    result = dict(completed=False)
    try:
        diagnose(result)
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
