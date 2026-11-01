"""Replay fixed coral views with saved step0/300 heads. Inference only."""
import fcntl
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.position_grid4 import select
OUT=ROOT/'reports/position-grid4-coral-scores-20261101'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(result):
    spec=importlib.util.spec_from_file_location('pilot_entry',ROOT/'scripts/run-position-grid4-pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    selection=json.loads((OUT/'selection.json').read_text())
    assert len(selection['views'])==18 and len({r['image_id'] for r in selection['views']})==10
    pilot=module.Pilot();pilot.head.requires_grad_(False).eval()
    # Refuse any accidental optimizer update in this export process.
    def forbidden(*a,**k):raise AssertionError('No optimization permitted during export')
    pilot.optimizer.step=forbidden
    checkpoints={};hashes={};saved={}
    for step in [0,300]:
        p=ROOT/'work_dirs/position_grid4_pilot'/f'step{step}.pth';hashes[str(step)]=sha(p)
        ck=torch.load(p,map_location='cpu')
        assert ck['step']==step and ck['kind']=='grid4_pilot' and ck['manifest_hash']==module.digest(module.PLAN)
        assert ck['code_hashes']==pilot.codes and ck['view_hash']==pilot.options['view_hash']
        checkpoints[step]=ck['head']
        evaluation=json.loads((ROOT/'work_dirs/position_grid4_pilot'/f'eval-{step}.json').read_text())
        saved[step]={(r['image_id'],r['augmentation_seed']):r['selected_cells'] for r in evaluation['records'] if r['source']=='head' and r['k']==200 and r['split']=='holdout'}
    result.update(checkpoint_sha256=hashes,selection_sha256=sha(OUT/'selection.json'),runs=[])
    for chosen in selection['views']:
        iid,seed=chosen['image_id'],chosen['seed']
        row=next(r for r in pilot.views if r['split']=='holdout' and r['image_id']==iid and r['seed']==seed)
        data,target,valid=pilot.data(row)
        with torch.no_grad():features=pilot.model.extract_feat(data['inputs'])
        payload={'valid':valid.numpy()}
        for step in [0,300]:
            pilot.head.load_state_dict(checkpoints[step],strict=True)
            with torch.no_grad():logits=pilot.head(features[1],features[2])[0].cpu().numpy()
            assert np.isfinite(logits).all() and logits.shape==valid.shape
            cells,_=select(logits,row['valid_shape'],200)
            assert cells.tolist()==saved[step][(iid,seed)],(step,iid,seed,'top200 mismatch')
            assert all(torch.equal(v.cpu(),checkpoints[step][k]) for k,v in pilot.head.state_dict().items())
            payload[f'logits_{step}']=logits
        filename=f'image-{iid}-seed-{seed}.npz'
        np.savez_compressed(OUT/filename,**payload)
        result['runs'].append(dict(image_id=iid,seed=seed,file=filename,sha256=sha(OUT/filename),
                                  valid_shape=row['valid_shape'],valid_cells=int(valid.sum()),top200_exact_both=True))
        print('EXPORTED',iid,seed,flush=True)
        del data,features,payload,logits
    pilot.protected();assert all(p.grad is None for p in pilot.head.parameters())
    for step in [0,300]:assert sha(ROOT/'work_dirs/position_grid4_pilot'/f'step{step}.pth')==hashes[str(step)]
    result.update(passed=True,original_model_unchanged=True,head_parameters_unchanged=True,
                  code_sha256={**pilot.codes,Path(__file__).name:sha(Path(__file__))})


if __name__=='__main__':
    assert not (OUT/'result.json').exists()
    result=dict(passed=False,training_updates=0,scope='10 fixed holdout coral images / 18 frozen augmented views, step0 and step300')
    try:
        with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(result)
    except BaseException:result['error']=traceback.format_exc();raise
    finally:(OUT/'result.json').write_text(json.dumps(result,indent=2))
