"""Bounded real-model check of label-only ablation, never optimizer updates."""
import fcntl
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.representative_label_only import objective,VERSION
from scripts.isolated_position_head import position_loss

OUT=ROOT/'reports/representative-label-model-20261101'
PLAN=ROOT/'docs/representative-label-model-plan-20261101.json'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main(result):
    plan=json.loads(PLAN.read_text());assert plan['version']==VERSION and plan['training_updates_allowed']==0
    for n,h in plan['code_sha256'].items():assert sha(ROOT/'scripts'/n)==h
    helper=load('gradient_helpers','check-representative-model.py')
    pilot_module=load('pilot','run-position-grid4-pilot.py');pilot=pilot_module.Pilot()
    def forbidden(*a,**k):raise AssertionError('Parameter updates prohibited')
    pilot.optimizer.step=forbidden
    original=pilot.original_output()
    proof=json.loads((ROOT/'reports/representative-integration-20261101/gpu-result.json').read_text())
    result.update(plan_sha256=sha(PLAN),runs=[],checkpoint_sha256=proof['checkpoint_sha256'])
    for stage in plan['head_checkpoints']:
        path=ROOT/'work_dirs/position_grid4_pilot'/f'step{stage}.pth'
        assert sha(path)==proof['checkpoint_sha256'][str(stage)]
        ck=torch.load(path,map_location='cpu')
        assert ck['code_hashes']==pilot.codes and ck['manifest_hash']==pilot.options['manifest_hash'] and ck['view_hash']==pilot.options['view_hash']
        pilot.head.load_state_dict(ck['head'],strict=True);pilot.head.eval().requires_grad_(True)
        saved={k:v.detach().clone() for k,v in pilot.head.state_dict().items()}
        params=list(pilot.head.parameters())
        for case in plan['views']:
            row=next(r for r in pilot.views if (r['image_id'],r['seed'])==(case['image_id'],case['seed']))
            data,target,valid=pilot.data(row)
            with torch.no_grad():features=pilot.model.extract_feat(data['inputs'])
            c3=features[1].detach().requires_grad_();c4=features[2].detach().requires_grad_()
            logits=pilot.head(c3,c4);snapshot=logits.detach().clone()
            gt=data['data_samples'][0].gt_instances
            items=[dict(masks=gt.masks.masks,ids=gt.annotation_ids.tolist(),areas=gt.original_areas.tolist())]
            target=target[None].cuda();valid=valid[None].cuda()
            old=position_loss(logits,target,valid)
            disabled,_=objective(logits,target,valid,enabled=False)
            assert torch.equal(old,disabled)
            oldgrad=torch.autograd.grad(old,logits,retain_graph=True)[0]
            assert torch.equal(oldgrad,torch.autograd.grad(disabled,logits,retain_graph=True)[0])
            new,details=objective(logits,target,valid,items,enabled=True)
            labels=details['labels'];layout=details['layouts'][0];reps=layout['representatives']
            gradient=torch.autograd.grad(new,logits,retain_graph=True)[0]
            unchanged=(target==labels)&valid
            assert torch.equal(gradient[unchanged],oldgrad[unchanged])
            assert torch.isfinite(new) and torch.isfinite(gradient).all()
            assert (gradient[(labels==1)&valid]<0).all() and (gradient[(labels==0)&valid]>0).all()
            assert not gradient[~valid].any()
            g_old=helper.flatgrad(old,params);g_new=helper.flatgrad(new,params)
            assert float(torch.linalg.vector_norm(g_new))>0
            # Common spatial anchors, retaining both old seeds and new representatives.
            anchors={}
            for gid in case['anchor_ids']:
                x,y=row['assignments'][str(gid)]
                anchors[str(gid)]=dict(legacy=int(y)*logits.shape[-1]+int(x),candidate=reps[gid])
            comparisons={}
            for cell in sorted({c for d in anchors.values() for c in d.values()}):
                jac=helper.flatgrad(logits.reshape(-1)[cell],params)
                comparisons[str(cell)]=dict(legacy=helper.interaction(g_old,jac),candidate=helper.interaction(g_new,jac),
                    legacy_positive=bool(target.reshape(-1)[cell]),candidate_positive=bool(labels.reshape(-1)[cell]))
            pairs=[]
            for a,b in case['pairs']:
                assert reps[a]!=reps[b]
                assert float(gradient.reshape(-1)[reps[a]])<0 and float(gradient.reshape(-1)[reps[b]])<0
                pairs.append(dict(ids=[a,b],candidate_cells=[reps[a],reps[b]],both_positive=True))
            pilot.head.zero_grad(set_to_none=True);new.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert c3.grad is None and c4.grad is None
            pilot.protected();assert all(torch.equal(saved[k],v) for k,v in pilot.head.state_dict().items())
            with torch.no_grad():after=pilot.head(c3,c4)
            assert torch.equal(snapshot,after) and not pilot.optimizer.state
            run=dict(stage=stage,image_id=row['image_id'],seed=row['seed'],targets=len(reps),
                positive_cells_replaced=int(((labels==1)&(target==0)).sum()),
                losses=dict(legacy=float(old.detach()),candidate=float(new.detach())),
                gradient_norms=dict(legacy=float(torch.linalg.vector_norm(g_old)),candidate=float(torch.linalg.vector_norm(g_new))),
                gradient_cosine=float(torch.dot(g_old,g_new)/(torch.linalg.vector_norm(g_old)*torch.linalg.vector_norm(g_new))),
                default_off_exact=True,unchanged_label_gradients_exact=True,logit_signs_passed=True,
                parameters_and_head_outputs_unchanged=True,features_isolated=True,anchors=anchors,common_cells=comparisons,pairs=pairs)
            result['runs'].append(run);print('CHECKED',stage,row['image_id'],flush=True)
    pilot.protected();assert original==pilot.original_output()
    for stage,h in proof['checkpoint_sha256'].items():assert sha(ROOT/'work_dirs/position_grid4_pilot'/f'step{stage}.pth')==h
    result.update(passed=True,engineering_passed=True,parameters_unchanged=True,original_output_exact=True,
        optimizer_state_empty=True,shared_parameter_directions_are_diagnostic_only=True,
        limitation='No training or AP benefit. Common-cell first-order SGD comparisons, not AdamW updates. Smaller label-dependent loss is not improvement evidence.')


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists()
    result=dict(passed=False,training_updates=0)
    try:
        with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main(result)
    except BaseException:result['error']=traceback.format_exc();raise
    finally:(OUT/'result.json').write_text(json.dumps(result,indent=2))
