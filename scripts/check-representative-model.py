"""Real frozen E0 + existing heads, backward only. No optimizer step path."""
import fcntl
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.instance_representative import objective,instance_term,WEIGHTS,VERSION
from scripts.isolated_position_head import position_loss

OUT=ROOT/'reports/representative-integration-20261101'
PLAN=ROOT/'docs/representative-integration-plan-20261101.json'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def flatgrad(value,params):
    gradients=torch.autograd.grad(value,params,retain_graph=True,allow_unused=False)
    assert all(torch.isfinite(g).all() for g in gradients)
    return torch.cat([g.reshape(-1) for g in gradients]).detach().double()


def interaction(gradient,jacobian):
    dot=float(torch.dot(gradient,jacobian))
    scale=float(torch.linalg.vector_norm(gradient)*torch.linalg.vector_norm(jacobian))
    return dict(sgd_directional_score_change=-dot,normalized_direction=-dot/scale if scale else 0.)


def main(result):
    plan=json.loads(PLAN.read_text());assert plan['weights']==WEIGHTS and plan['version']==VERSION
    for n,h in plan['code_sha256'].items():assert sha(ROOT/'scripts'/n)==h
    assert plan['training_updates_allowed']==0
    spec=importlib.util.spec_from_file_location('pilot',ROOT/'scripts/run-position-grid4-pilot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pilot=module.Pilot()
    def forbidden(*a,**k):raise AssertionError('Parameter update prohibited')
    pilot.optimizer.step=forbidden
    assert not pilot.optimizer.state
    before_original=pilot.original_output()
    result.update(plan_sha256=sha(PLAN),runs=[],checkpoint_sha256={})
    proof=json.loads((ROOT/'reports/position-grid4-coral-scores-20261101/result.json').read_text())
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for stage in plan['head_checkpoints']:
        p=ROOT/'work_dirs/position_grid4_pilot'/f'step{stage}.pth'
        assert sha(p)==proof['checkpoint_sha256'][str(stage)]
        result['checkpoint_sha256'][str(stage)]=sha(p)
        checkpoint=torch.load(p,map_location='cpu')
        assert checkpoint['step']==stage and checkpoint['kind']=='grid4_pilot'
        assert checkpoint['code_hashes']==pilot.codes and checkpoint['manifest_hash']==module.digest(module.PLAN)
        assert checkpoint['view_hash']==pilot.options['view_hash']
        pilot.head.load_state_dict(checkpoint['head'],strict=True);pilot.head.eval().requires_grad_(True)
        saved={k:v.detach().clone() for k,v in pilot.head.state_dict().items()}
        params=list(pilot.head.parameters())
        for selection in plan['views']:
            row=next(r for r in pilot.views if (r['image_id'],r['seed'])==(selection['image_id'],selection['seed']))
            data,target,valid=pilot.data(row)
            with torch.no_grad():features=pilot.model.extract_feat(data['inputs'])
            c3=features[1].detach().requires_grad_();c4=features[2].detach().requires_grad_()
            logits=pilot.head(c3,c4)
            original_logits=logits.detach().clone()
            gt=data['data_samples'][0].gt_instances
            items=[dict(masks=gt.masks.masks,ids=gt.annotation_ids.tolist(),areas=gt.original_areas.tolist())]
            target=target[None].cuda();valid=valid[None].cuda()
            disabled,_=objective(logits,target,valid,enabled=False)
            legacy=position_loss(logits,target,valid)
            assert torch.equal(disabled,legacy)
            ga=torch.autograd.grad(disabled,logits,retain_graph=True)[0]
            gb=torch.autograd.grad(legacy,logits,retain_graph=True)[0]
            assert torch.equal(ga,gb)
            value,details=objective(logits,target,valid,items,enabled=True)
            layout=details['layouts'][0];reps=layout['representatives'];groups=layout['groups']
            grad=torch.autograd.grad(value,logits,retain_graph=True)[0]
            assert torch.isfinite(grad).all() and not grad[~valid].any()
            flat=grad.reshape(-1)
            assert all(float(flat[c])<0 for c in reps.values())
            assert all(float(flat[c])>0 for cells in groups.values() for c in cells)
            gtotal=flatgrad(value,params)
            assert float(torch.linalg.vector_norm(gtotal))>0
            pairs=[]
            for a,b in selection['pairs']:
                assert a in reps and b in reps and reps[a]!=reps[b]
                la=instance_term(logits[0],layout,a);lb=instance_term(logits[0],layout,b)
                da=torch.autograd.grad(la,logits,retain_graph=True)[0].reshape(-1)
                db=torch.autograd.grad(lb,logits,retain_graph=True)[0].reshape(-1)
                assert float(da[reps[b]])==0 and float(db[reps[a]])==0
                g_a=flatgrad(la,params);g_b=flatgrad(lb,params)
                jac_a=flatgrad(logits.reshape(-1)[reps[a]],params)
                jac_b=flatgrad(logits.reshape(-1)[reps[b]],params)
                pairs.append(dict(ids=[a,b],cells=[reps[a],reps[b]],direct_cross_logit_gradients=[0.,0.],
                    a_to_b=interaction(g_a,jac_b),b_to_a=interaction(g_b,jac_a),
                    total_to_a=interaction(gtotal,jac_a),total_to_b=interaction(gtotal,jac_b),
                    parameter_gradient_cosine=float(torch.dot(g_a,g_b)/(torch.linalg.vector_norm(g_a)*torch.linalg.vector_norm(g_b)))))
            pilot.head.zero_grad(set_to_none=True);value.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert c3.grad is None and c4.grad is None
            pilot.protected()
            assert all(torch.equal(saved[k],v) for k,v in pilot.head.state_dict().items())
            with torch.no_grad():after_logits=pilot.head(c3,c4)
            assert torch.equal(original_logits,after_logits)
            assert not pilot.optimizer.state
            run=dict(image_id=row['image_id'],seed=row['seed'],stage=stage,
                targets=len(reps),loss=float(value.detach()),
                parts={k:float(v.detach()) for k,v in details['terms'][0].items()},
                gradient_norm=float(torch.linalg.vector_norm(gtotal)),
                head_output_unchanged=True,head_state_unchanged=True,backbone_isolated=True,
                default_off_exact=True,logit_signs_passed=True,padding_gradients_zero=True,pairs=pairs)
            result['runs'].append(run)
            print('CHECKED',stage,row['image_id'],row['seed'],'PAIRS',len(pairs),flush=True)
            (OUT/'progress.json').write_text(json.dumps(result,indent=2))
            del data,features,c3,c4,logits,value,details,after_logits,gtotal,grad,original_logits
    pilot.protected();assert before_original==pilot.original_output()
    for stage,h in result['checkpoint_sha256'].items():assert sha(ROOT/'work_dirs/position_grid4_pilot'/f'step{stage}.pth')==h
    pairs=[p for r in result['runs'] for p in r['pairs']]
    tol=plan['normalized_direction_tolerance']
    cross=[p[k]['normalized_direction'] for p in pairs for k in ['a_to_b','b_to_a']]
    total=[p[k]['normalized_direction'] for p in pairs for k in ['total_to_a','total_to_b']]
    result.update(engineering_passed=True,original_output_exact=True,original_model_unchanged=True,
        parameters_unchanged=True,optimizer_state_empty=True,checked_pairs=len(pairs),
        cross_parameter_suppressions=sum(x < -tol for x in cross),
        total_direction_suppressions=sum(x < -tol for x in total),
        neighbor_parameter_non_suppression_passed=all(x>=-tol for x in cross+total),
        elapsed_seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
    result['passed']=result['engineering_passed'] and result['neighbor_parameter_non_suppression_passed']


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'gpu-result.json').exists(), 'Preserve prior result'
    result=dict(passed=False,training_updates=0)
    try:
        with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(result)
    except BaseException:result['error']=traceback.format_exc();raise
    finally:(OUT/'gpu-result.json').write_text(json.dumps(result,indent=2))
