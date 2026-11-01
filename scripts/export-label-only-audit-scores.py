"""Four saved heads, fixed twelve views. No backward or optimizer updates."""
import fcntl,hashlib,importlib.util,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.position_grid4 import select
OUT=ROOT/'reports/label-only-audit-scores-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(result):
    request=json.loads((OUT/'request.json').read_text())
    spec=importlib.util.spec_from_file_location('base',ROOT/'scripts/run-position-grid4-pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pilot=module.Pilot();pilot.head.requires_grad_(False).eval()
    def forbidden(*a,**k):raise AssertionError('Updates forbidden')
    pilot.optimizer.step=forbidden
    states={};saved={};sources={}
    for name,s in request['stages'].items():
        folder=ROOT/'work_dirs'/s['directory'];step=s['step']
        for f,h in s['sha256'].items():assert sha(folder/f)==h,(name,f)
        ck=torch.load(folder/f'step{step}.pth',map_location='cpu')
        assert ck['step']==step and ck['view_hash']==pilot.options['view_hash']
        assert ck['kind']==('grid4_pilot' if name in ['initial','old300'] else 'label_only_pilot')
        states[name]=ck['head'];sources[name]=s['sha256']
        ev=json.loads((folder/f'eval-{step}.json').read_text())
        saved[name]={(r['image_id'],r['augmentation_seed']):r['selected_cells'] for r in ev['records'] if r['source']=='head' and r['k']==200 and r['split']=='holdout'}
    result.update(request_sha256=sha(OUT/'request.json'),sources=sources,runs=[])
    for c in request['cases']:
        key=(c['image_id'],c['seed'])
        row=next(r for r in pilot.views if r['split']=='holdout' and (r['image_id'],r['seed'])==key)
        data,_,valid=pilot.data(row)
        with torch.no_grad():features=pilot.model.extract_feat(data['inputs'])
        payload={'valid':valid.numpy()}
        for name,state in states.items():
            pilot.head.load_state_dict(state,strict=True)
            with torch.no_grad():z=pilot.head(features[1],features[2])[0].cpu().numpy()
            assert z.shape==valid.shape and np.isfinite(z).all()
            cells,_=select(z,row['valid_shape'],200)
            assert cells.tolist()==saved[name][key],(c['case_id'],name,'top200 mismatch')
            assert all(torch.equal(v.cpu(),state[k]) for k,v in pilot.head.state_dict().items())
            payload[name]=z
        file=f"case-{c['case_id']:02d}.npz";np.savez_compressed(OUT/file,**payload)
        result['runs'].append(dict(case_id=c['case_id'],image_id=key[0],seed=key[1],file=file,sha256=sha(OUT/file),input_sha256=row['input_sha256'],masks_sha256=row['masks_sha256'],top200_exact_all=True))
        print('EXPORTED',c['case_id'],flush=True)
    pilot.protected();assert all(p.grad is None for p in pilot.head.parameters())
    for s in request['stages'].values():
        for f,h in s['sha256'].items():assert sha(ROOT/'work_dirs'/s['directory']/f)==h
    result.update(passed=True,base_unchanged=True,heads_unchanged=True,script_sha256=sha(Path(__file__)))
if __name__=='__main__':
    assert not (OUT/'result.json').exists()
    result=dict(passed=False,training_updates=0)
    try:
        with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(result)
    except BaseException:result['error']=traceback.format_exc();raise
    finally:(OUT/'result.json').write_text(json.dumps(result,indent=2))
