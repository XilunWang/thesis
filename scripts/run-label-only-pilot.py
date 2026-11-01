"""Default local metadata check. GPU preflight never updates; execute is opt-in."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if sys.platform=='win32':sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT/'.local/bll-libs')]
PLAN=ROOT/'docs/label-only-pilot-plan-20261101.json'
RUN=ROOT/'work_dirs/position_label_only_pilot'
CHECK=ROOT/'reports/label-only-pilot-entry-20261101'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def control_file(name):
    server=ROOT/'work_dirs/position_grid4_pilot'/name
    return server if server.exists() else ROOT/'reports/position-grid4-execution-20261101'/name


def verify():
    plan=json.loads(PLAN.read_text())
    assert plan['steps']==300 and plan['batch_size']==2 and plan['minimum_coral_hits']==16
    for name,h in plan['code_sha256'].items():assert sha(ROOT/'scripts'/name)==h,name
    for name,h in plan['source_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in plan['control_sha256'].items():assert sha(control_file(name))==h,name
    result=json.loads(control_file('result.json').read_text())
    assert result['completed'] and result['step']==300 and result['original_preserved']
    entry=json.loads((ROOT/'reports/representative-label-model-20261101/result.json').read_text())
    assert entry['engineering_passed'] and entry['parameters_unchanged'] and entry['training_updates']==0
    return plan


def main(args):
    plan=verify()
    if not args.preflight and not args.execute:
        entry=json.loads((CHECK/'result.json').read_text()) if (CHECK/'result.json').exists() else {}
        print(json.dumps(dict(passed=True,mode='metadata_only',training_updates=0,plan_sha256=sha(PLAN),
            gpu_preflight_completed=entry.get('passed',False) and entry.get('plan_sha256')==sha(PLAN),control_reusable=True)))
        return
    import fcntl
    import torch
    from scripts.position_pilot_runtime import load_checkpoint,save_checkpoint,view_at
    from scripts.representative_label_only import objective
    from scripts.label_only_pilot_metrics import compare
    spec=importlib.util.spec_from_file_location('original_pilot',ROOT/'scripts/run-position-grid4-pilot.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

    class Pilot(base.Pilot):
        def backward(self,rows):
            pieces=[self.data(r) for r in rows]
            inputs=torch.cat([d['inputs'] for d,t,v in pieces])
            with torch.no_grad():features=self.model.extract_feat(inputs)
            targets=torch.stack([t for d,t,v in pieces]).cuda()
            valid=torch.stack([v for d,t,v in pieces]).cuda()
            instances=[]
            for data,t,v in pieces:
                gt=data['data_samples'][0].gt_instances
                instances.append(dict(masks=gt.masks.masks,ids=gt.annotation_ids.tolist(),areas=gt.original_areas.tolist()))
            self.optimizer.zero_grad(set_to_none=True)
            logits=self.head(features[1],features[2])
            loss,_=objective(logits,targets,valid,instances,enabled=True)
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.head.parameters())
            assert all(p.grad is None for p in self.model.parameters())
            norm=torch.nn.utils.clip_grad_norm_(self.head.parameters(),self.plan['grad_clip_norm'],error_if_nonfinite=True)
            return float(loss.detach()),float(norm)

    with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pilot=Pilot()
        # Restore the ACTUAL original step0, including empty Adam and RNG state.
        state=load_checkpoint(control_file('step0.pth'),pilot.head,pilot.optimizer,**pilot.options)
        assert state['step']==0 and state['cursor']==0 and not pilot.optimizer.state
        pilot.codes={**pilot.codes,**plan['code_sha256'],'candidate_plan':sha(PLAN)}
        pilot.options=dict(manifest_hash=sha(PLAN),code_hashes=pilot.codes,
                           view_hash=pilot.options['view_hash'],kind='label_only_pilot')
        if args.preflight:
            CHECK.mkdir(exist_ok=False)
            def forbidden(*a,**k):raise AssertionError('Updates forbidden in preflight')
            pilot.optimizer.step=forbidden
            before={k:v.clone() for k,v in pilot.head.state_dict().items()};original=pilot.original_output()
            selected=[next(r for r in pilot.views if (r['image_id'],r['seed'])==key)
                      for key in [(2809,20263821),(2971,20263983)]]
            loss,norm=pilot.backward(selected);pilot.protected()
            assert all(torch.equal(v,before[k]) for k,v in pilot.head.state_dict().items())
            save_checkpoint(CHECK/'step0.pth',pilot.head,pilot.optimizer,**pilot.options,step=0,cursor=0,elapsed_seconds=0)
            load_checkpoint(CHECK/'step0.pth',pilot.head,pilot.optimizer,**pilot.options)
            assert not pilot.optimizer.state and all(torch.equal(v,before[k]) for k,v in pilot.head.state_dict().items())
            assert original==pilot.original_output()
            base.write_json(CHECK/'result.json',dict(passed=True,training_updates=0,plan_sha256=sha(PLAN),
                codes=pilot.codes,batch2_loss=loss,gradient_norm=norm,same_process_restore_exact=True))
            print('PREFLIGHT_PASSED_NO_UPDATES');return
        proof=json.loads((CHECK/'result.json').read_text())
        assert proof['passed'] and proof['training_updates']==0 and proof['plan_sha256']==sha(PLAN) and proof['codes']==pilot.codes
        if args.resume:
            state=load_checkpoint(RUN/'latest.pth',pilot.head,pilot.optimizer,**pilot.options)
            step=state['step'];elapsed=state['elapsed_seconds']
            original=json.loads((RUN/'original-output.json').read_text())
        else:
            RUN.mkdir(exist_ok=False);step=0;elapsed=0.;original=pilot.original_output()
            base.write_json(RUN/'original-output.json',original)
            save_checkpoint(RUN/'latest.pth',pilot.head,pilot.optimizer,**pilot.options,step=0,cursor=0,elapsed_seconds=0)
        initial=json.loads(control_file('eval-0.json').read_text())
        while True:
            pilot.protected();assert original==pilot.original_output()
            if step in [0,100,300]:
                save_checkpoint(RUN/f'step{step}.pth',pilot.head,pilot.optimizer,**pilot.options,step=step,cursor=step*2,elapsed_seconds=elapsed)
                evaluation=pilot.eval(step,RUN,random_controls=step==0)
                if step==0:
                    assert evaluation['records']==initial['records'],'Initial predictions or metrics differ: no updates allowed'
            if step==300 or elapsed>=plan['max_training_seconds']:break
            stop=100 if step<100 else 300
            start=time.monotonic();base_elapsed=elapsed
            while step<stop and elapsed<plan['max_training_seconds']:
                rows=[view_at(pilot.learn,step*2+j,pilot.plan['order_seed']) for j in [0,1]]
                pilot.head.train();loss,norm=pilot.backward(rows)
                pilot.optimizer.step()
                assert all(torch.isfinite(p).all() for p in pilot.head.parameters())
                step+=1;elapsed=base_elapsed+time.monotonic()-start
                save_checkpoint(RUN/'latest.pth',pilot.head,pilot.optimizer,**pilot.options,step=step,cursor=step*2,elapsed_seconds=elapsed)
                with (RUN/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=loss,gradient_norm=norm,elapsed_seconds=elapsed))+'\n')
                if step%25==0:print('STEP',step,'LOSS',loss,flush=True)
        pilot.protected();assert original==pilot.original_output()
        result=dict(completed=step==300,step=step,training_updates=step,original_preserved=True,elapsed_training_seconds=elapsed)
        if step==300:
            final=json.loads((RUN/'eval-300.json').read_text())
            control=json.loads(control_file('eval-300.json').read_text())
            views=json.loads((ROOT/'reports/position-learning-plan-grid4-20261101/views.json').read_text())
            result.update(compare(ROOT,final,control,initial,plan,views))
            result['passed']=result['position_gates_passed'] and result['original_preserved']
        else:result.update(passed=False,status='TIME_BUDGET_INCOMPLETE_NOT_METHOD_FAILURE')
        base.write_json(RUN/'result.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.resume and not args.execute:parser.error('--resume requires --execute')
    try:main(args)
    except BaseException:
        if args.preflight or args.execute:
            import traceback
            failures=ROOT/'reports/label-only-pilot-failures-20261101';failures.mkdir(exist_ok=True)
            (failures/f'failure-{time.time_ns()}.txt').write_text(traceback.format_exc())
        raise
