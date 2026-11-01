"""Fixed 2809/16538 failure decomposition, no training or coefficient changes."""
import fcntl
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from scripts.instance_representative import objective,WEIGHTS
from scripts.isolated_position_head import position_loss

OUT=ROOT/'reports/representative-conflict-decomposition-20261101'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main(result):
    check=load('integration','check-representative-model.py')
    previous=json.loads((check.OUT/'gpu-result.json').read_text())
    plan=json.loads(check.PLAN.read_text())
    assert sha(check.PLAN)==previous['plan_sha256'] and plan['weights']==WEIGHTS
    for n,h in plan['code_sha256'].items():assert sha(ROOT/'scripts'/n)==h
    pilot_module=load('pilot','run-position-grid4-pilot.py');pilot=pilot_module.Pilot()
    def forbidden(*a,**k):raise AssertionError('No parameter updates permitted')
    pilot.optimizer.step=forbidden
    original=pilot.original_output()
    result.update(runs=[],source_result_sha256=sha(check.OUT/'gpu-result.json'),
                  frozen_plan_sha256=sha(check.PLAN),script_sha256=sha(Path(__file__)))
    for stage in [0,300]:
        path=ROOT/'work_dirs/position_grid4_pilot'/f'step{stage}.pth'
        assert sha(path)==previous['checkpoint_sha256'][str(stage)]
        ck=torch.load(path,map_location='cpu');assert ck['code_hashes']==pilot.codes
        pilot.head.load_state_dict(ck['head'],strict=True);pilot.head.eval().requires_grad_(True)
        saved={k:v.detach().clone() for k,v in pilot.head.state_dict().items()}
        row=next(r for r in pilot.views if (r['image_id'],r['seed'])==(2809,20263821))
        data,target,valid=pilot.data(row)
        with torch.no_grad():features=pilot.model.extract_feat(data['inputs'])
        logits=pilot.head(features[1],features[2]);z=logits.reshape(-1)
        gt=data['data_samples'][0].gt_instances
        item=dict(masks=gt.masks.masks,ids=gt.annotation_ids.tolist(),areas=gt.original_areas.tolist())
        target=target[None].cuda();valid=valid[None].cuda()
        total,details=objective(logits,target,valid,[item],enabled=True)
        layout=details['layouts'][0];reps=layout['representatives'];groups=layout['groups']
        terms={'new_'+k:WEIGHTS[k]*v for k,v in details['terms'][0].items()}
        terms['new_total']=total
        p=z[valid.reshape(-1)];t=target.reshape(-1)[valid.reshape(-1)]
        terms['legacy_positive']=0.5*F.softplus(-p[t==1]).mean()
        terms['legacy_negative']=0.5*F.softplus(p[t==0]).mean()
        terms['legacy_total']=position_loss(logits,target,valid)
        assert torch.allclose(terms['legacy_total'],terms['legacy_positive']+terms['legacy_negative'],atol=1e-7,rtol=1e-6)
        zero=z[:0].sum();nonempty=sum(bool(c) for c in groups.values())
        for name,ids in [('self',[16538]),('neighbor',[16539]),('others',[g for g in reps if g not in [16538,16539]])]:
            terms['positive_'+name]=WEIGHTS['positive']*sum((F.softplus(-z[reps[g]]) for g in ids),zero)/len(reps)
            terms['duplicate_'+name]=WEIGHTS['duplicate']*sum((F.softplus(z[groups[g]]-z[reps[g]]).mean() for g in ids if groups[g]),zero)/nonempty if nonempty else zero
        params=list(pilot.head.parameters())
        gradients={name:check.flatgrad(value,params) for name,value in terms.items()}
        residuals={}
        for total_name,parts in [('new_total',['new_positive','new_duplicate','new_outside']),
                                 ('legacy_total',['legacy_positive','legacy_negative']),
                                 ('new_positive',['positive_self','positive_neighbor','positive_others']),
                                 ('new_duplicate',['duplicate_self','duplicate_neighbor','duplicate_others'])]:
            residual=float(torch.linalg.vector_norm(gradients[total_name]-sum(gradients[k] for k in parts)))/max(float(torch.linalg.vector_norm(gradients[total_name])),1e-12)
            assert residual<1e-5,(total_name,residual)
            residuals[total_name]=residual
        anchors={}
        for gid in [16538,16539]:
            x,y=row['assignments'][str(gid)]
            anchors[f'{gid}_legacy_seed']=int(y)*logits.shape[-1]+int(x)
            anchors[f'{gid}_new_representative']=reps[gid]
        cell_results={}
        for cell in sorted(set(anchors.values())):
            jac=check.flatgrad(z[cell],params)
            cell_results[str(cell)]={name:dict(**check.interaction(gradient,jac),
                direct_logit_gradient=float(torch.autograd.grad(terms[name],logits,retain_graph=True)[0].reshape(-1)[cell]))
                for name,gradient in gradients.items()}
        prior=next(r for r in previous['runs'] if r['stage']==stage and r['image_id']==2809)['pairs'][0]
        for gid,key in [(16538,'total_to_a'),(16539,'total_to_b')]:
            actual=cell_results[str(reps[gid])]['new_total']['normalized_direction']
            assert abs(actual-prior[key]['normalized_direction'])<1e-6
        assert not pilot.optimizer.state and all(p.grad is None for p in pilot.model.parameters())
        pilot.protected();assert all(torch.equal(saved[k],v) for k,v in pilot.head.state_dict().items())
        assert all(p.grad is None for p in pilot.head.parameters())  # autograd.grad only
        assert sha(path)==previous['checkpoint_sha256'][str(stage)]
        run=dict(stage=stage,image_id=2809,seed=20263821,targets=len(reps),
            anchors=anchors,losses={k:float(v.detach()) for k,v in terms.items()},
            parameter_gradient_norms={k:float(torch.linalg.vector_norm(v)) for k,v in gradients.items()},
            relative_sum_residuals=residuals,cells=cell_results,
            tiny_duplicate_counts={str(g):len(groups[g]) for g in [16538,16539]},
            new_positive_cells_changed=len(set(reps.values())-set(map(int,torch.nonzero(target.reshape(-1),as_tuple=False).reshape(-1).cpu().tolist()))))
        result['runs'].append(run);print('DECOMPOSED',stage,flush=True)
    pilot.protected();assert original==pilot.original_output()
    result.update(passed=True,parameters_unchanged=True,original_output_exact=True,optimizer_state_empty=True,
        limitation='Weighted first-order SGD directional derivatives on common cells, not actual AdamW updates or causal training improvements; original loss is old position-head loss, not segmentation BLL.')


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists()
    result=dict(passed=False,training_updates=0)
    try:
        with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(result)
    except BaseException:result['error']=traceback.format_exc();raise
    finally:(OUT/'result.json').write_text(json.dumps(result,indent=2))
