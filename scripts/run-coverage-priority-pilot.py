"""Default metadata-only; preflight no updates; execute requires launch proof."""
import argparse,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if sys.platform=='win32':sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT/'.local/bll-libs')]
PLAN=ROOT/'docs/coverage-priority-pilot-plan-20261101.json'
CHECK=ROOT/'reports/coverage-priority-entry-20261101'
RUN=ROOT/'work_dirs/position_coverage_priority_pilot'
NEW_CODE=['coverage_priority.py','coverage_priority_objective.py','coverage_pilot_metrics.py','run-coverage-priority-pilot.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resolve(name):
    name=name.replace(chr(92),'/')
    for folder,work in [('position-grid4-execution-20261101','position_grid4_pilot'),('label-only-pilot-execution-20261101','position_label_only_pilot')]:
        prefix='reports/'+folder+'/'
        if name.startswith(prefix):
            server=ROOT/'work_dirs'/work/name[len(prefix):]
            if server.exists():return server
    return ROOT/name

def verify():
    plan=json.loads(PLAN.read_text());assert plan['steps']==300 and plan['batch_size']==2
    for name,h in {**plan['old_code_sha256'],**plan['candidate_code_sha256']}.items():assert sha(ROOT/'scripts'/name)==h,name
    for name,h in plan['source_sha256'].items():assert sha(resolve(name))==h,name
    assert sha(resolve(plan['initialization']['path']))==plan['initialization']['sha256']
    import numpy as np
    views=json.loads((ROOT/'reports/position-learning-plan-grid4-20261101/views.json').read_text())
    learn=[r for r in views['runs'] if r['split']=='learn'];order=[]
    for i in range(600):
        cycle,off=divmod(i,len(learn));r=learn[int(np.random.default_rng(plan['settings']['order_seed']+cycle).permutation(len(learn))[off])];order.append([r['image_id'],r['seed']])
    assert hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()==plan['training_order_sha256']
    return plan

def main(args):
    plan=verify()
    if not args.preflight and not args.execute:
        print(json.dumps(dict(passed=True,mode='metadata_only',training_updates=0,plan_sha256=sha(PLAN))));return
    import fcntl,numpy as np,torch
    from scripts.coverage_priority_objective import objective,settings_hash
    from scripts.position_pilot_runtime import save_checkpoint,load_checkpoint,view_at
    from scripts.position_grid4 import select
    from scripts.coverage_pilot_metrics import compare
    spec=importlib.util.spec_from_file_location('base_pilot',ROOT/'scripts/run-position-grid4-pilot.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    class Pilot(base.Pilot):
        def backward(self,rows):
            pieces=[self.data(r) for r in rows]
            with torch.no_grad():features=self.model.extract_feat(torch.cat([d['inputs'] for d,t,v in pieces]))
            target=torch.stack([t for d,t,v in pieces]).cuda();valid=torch.stack([v for d,t,v in pieces]).cuda();items=[]
            for data,t,v in pieces:
                gt=data['data_samples'][0].gt_instances;items.append(dict(masks=gt.masks.masks,ids=gt.annotation_ids.tolist(),areas=gt.original_areas.tolist()))
            self.optimizer.zero_grad(set_to_none=True);z=self.head(features[1],features[2]);loss,info=objective(z,target,valid,items,coverage_priority=True)
            assert torch.isfinite(loss)
            traces=[]
            for i,(row,item,layout) in enumerate(zip(rows,items,info['layouts'])):
                reps=layout['representatives'];gh,gw=valid.shape[-2:];records=[]
                reconstructed=np.zeros((gh,gw),np.float32)
                for gid,cell in sorted(reps.items()):
                    mask=item['masks'][item['ids'].index(gid)];count=int(mask.reshape(gh,4,gw,4).sum((1,3)).flat[cell]);assert count>0
                    reconstructed.flat[cell]=1;records.append(dict(gt_id=gid,cell=cell,foreground_pixels=count))
                assert np.array_equal(reconstructed,info['labels'][i].cpu().numpy())
                traces.append(dict(image_id=row['image_id'],seed=row['seed'],split=row['split'],input_sha256=row['input_sha256'],masks_sha256=row['masks_sha256'],label_sha256=base.array_hash(reconstructed),representatives=records))
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.head.parameters());assert all(p.grad is None for p in self.model.parameters())
            norm=torch.nn.utils.clip_grad_norm_(self.head.parameters(),self.plan['grad_clip_norm'],error_if_nonfinite=True)
            return float(loss.detach()),float(norm),traces
        def eval(self,step,destination,rows=None,random_controls=False):
            chosen=self.views if rows is None else rows;index=0;exports=[];scores=destination/f'full-scores-{step}';scores.mkdir(exist_ok=False)
            ck=destination/f'step{step}.pth';checkpoint_hash=sha(ck)
            def capture(module,inputs,output):
                nonlocal index
                row=chosen[index];index+=1
                if row['split']!='holdout':return
                z=output[0].detach().cpu().numpy();assert z.dtype==np.float32 and np.isfinite(z).all()
                keep=np.zeros(z.shape,bool);vh,vw=row['valid_shape'][:2];keep[:(vh+3)//4,:(vw+3)//4]=True
                file=f"image-{row['image_id']}-seed-{row['seed']}.npz";np.savez_compressed(scores/file,logits=z,valid=keep)
                cells,_=select(z,row['valid_shape'],200)
                exports.append(dict(image_id=row['image_id'],seed=row['seed'],file=file,sha256=sha(scores/file),input_sha256=row['input_sha256'],masks_sha256=row['masks_sha256'],selected_cells=cells.tolist()))
            hook=self.head.register_forward_hook(capture)
            try:result=super().eval(step,destination,rows=rows,random_controls=random_controls)
            finally:hook.remove()
            assert index==len(chosen)
            saved={(r['image_id'],r['augmentation_seed']):r['selected_cells'] for r in result['records'] if r['split']=='holdout' and r['source']=='head' and r['k']==200}
            assert len(exports)==len(saved)
            for r in exports:assert r.pop('selected_cells')==saved[(r['image_id'],r['seed'])]
            base.write_json(scores/'index.json',dict(step=step,checkpoint_sha256=checkpoint_hash,top200_exact=True,runs=exports));print('EVALUATED',step,'VIEWS',len(chosen),'FULL_SCORES',len(exports),flush=True)
            return result
    with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pilot=Pilot()
        for key in ['optimizer','order_seed','initialization_seed','grad_clip_norm','diagnostic_k','random_control_seeds']:assert pilot.plan[key]==plan['settings'][key],key
        initial=load_checkpoint(resolve(plan['initialization']['path']),pilot.head,pilot.optimizer,**pilot.options)
        assert initial['step']==0 and initial['cursor']==0 and not pilot.optimizer.state
        pilot.codes={**pilot.codes,**plan['old_code_sha256'],**{n:sha(ROOT/'scripts'/n) for n in NEW_CODE},'candidate_plan':sha(PLAN),'objective_settings':settings_hash(True)}
        pilot.options=dict(manifest_hash=sha(PLAN),code_hashes=pilot.codes,view_hash=pilot.options['view_hash'],kind='coverage_priority_pilot')
        if args.preflight:
            CHECK.mkdir(exist_ok=False)
            def forbidden(*a,**kw):raise AssertionError('No updates in preflight')
            pilot.optimizer.step=forbidden;before={k:v.clone() for k,v in pilot.head.state_dict().items()};original=pilot.original_output()
            rows=[next(r for r in pilot.views if (r['image_id'],r['seed'])==key) for key in [(2971,21263983),(3151,20264163)]]
            loss,norm,traces=pilot.backward(rows);pilot.protected();assert all(torch.equal(v,before[k]) for k,v in pilot.head.state_dict().items());assert not pilot.optimizer.state
            base.write_json(CHECK/'label-records.json',traces)
            save_checkpoint(CHECK/'step0.pth',pilot.head,pilot.optimizer,**pilot.options,step=0,cursor=0,elapsed_seconds=0)
            ev=pilot.eval(0,CHECK,rows=rows,random_controls=True);old=json.loads(resolve('reports/position-grid4-execution-20261101/eval-0.json').read_text())
            keys={(r['image_id'],r['seed']) for r in rows};assert ev['records']==[r for r in old['records'] if (r['image_id'],r['augmentation_seed']) in keys]
            load_checkpoint(CHECK/'step0.pth',pilot.head,pilot.optimizer,**pilot.options);assert all(torch.equal(v,before[k]) for k,v in pilot.head.state_dict().items());assert original==pilot.original_output()
            base.write_json(CHECK/'result.json',dict(passed=True,training_updates=0,plan_sha256=sha(PLAN),codes=pilot.codes,common_initial_exact=True,optimizer_empty=True,recording_passed=True,step0_subset_records_exact=True,loss=loss,gradient_norm=norm));print('PREFLIGHT_PASSED_NO_UPDATES',flush=True);return
        proof=json.loads((CHECK/'result.json').read_text());assert proof['passed'] and proof['codes']==pilot.codes and proof['plan_sha256']==sha(PLAN)
        launch=json.loads((ROOT/'reports/coverage-priority-pilot-execution-20261101/launch.json').read_text());assert launch['maximum_steps']==300 and launch['plan_sha256']==sha(PLAN) and launch['authorized']
        RUN.mkdir(exist_ok=False);original=pilot.original_output();base.write_json(RUN/'original-output.json',original)
        step=0;elapsed=0.;initial_eval=json.loads(resolve('reports/position-grid4-execution-20261101/eval-0.json').read_text())
        while True:
            pilot.protected();assert original==pilot.original_output()
            save_checkpoint(RUN/f'step{step}.pth',pilot.head,pilot.optimizer,**pilot.options,step=step,cursor=step*2,elapsed_seconds=elapsed)
            save_checkpoint(RUN/'latest.pth',pilot.head,pilot.optimizer,**pilot.options,step=step,cursor=step*2,elapsed_seconds=elapsed)
            ev=pilot.eval(step,RUN,random_controls=step==0)
            if step==0:
                assert ev['records']==initial_eval['records'],'Full initial equality failed; no updates allowed'
                base.write_json(RUN/'initial-check.json',dict(passed=True,training_updates=0,all_records_exact=True));print('FULL_STEP0_EXACT_UPDATES_ALLOWED',flush=True)
            if step==300 or elapsed>=plan['max_training_seconds']:break
            stop=100 if step<100 else 300;start=time.monotonic();previous_elapsed=elapsed
            while step<stop and elapsed<plan['max_training_seconds']:
                rows=[view_at(pilot.learn,step*2+j,pilot.plan['order_seed']) for j in [0,1]];pilot.head.train();loss,norm,traces=pilot.backward(rows)
                with (RUN/'training-labels.jsonl').open('a') as f:f.write(json.dumps(dict(step_before=step,attempted_update=step+1,views=traces))+'\n')
                pilot.optimizer.step();assert all(torch.isfinite(p).all() for p in pilot.head.parameters());step+=1;elapsed=previous_elapsed+time.monotonic()-start
                save_checkpoint(RUN/'latest.pth',pilot.head,pilot.optimizer,**pilot.options,step=step,cursor=step*2,elapsed_seconds=elapsed)
                with (RUN/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=loss,gradient_norm=norm,elapsed_seconds=elapsed))+'\n')
                if step%25==0:print('STEP',step,'LOSS',loss,flush=True)
        pilot.protected();assert original==pilot.original_output();result=dict(completed=step==300,step=step,training_updates=step,original_preserved=True,elapsed_training_seconds=elapsed)
        if step==300:
            adaptive=json.loads(resolve('reports/label-only-pilot-execution-20261101/eval-300.json').read_text());geometric=json.loads(resolve('reports/position-grid4-execution-20261101/eval-300.json').read_text());views=json.loads((ROOT/'reports/position-learning-plan-grid4-20261101/views.json').read_text())
            result.update(compare(ROOT,ev,adaptive,geometric,initial_eval,plan,views));result['passed']=result['position_gates_passed']
        else:result.update(passed=False,status='TIME_BUDGET_INCOMPLETE_NOT_METHOD_FAILURE')
        base.write_json(RUN/'result.json',result);print('FINISHED',step,'PASSED',result['passed'],flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group();mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true');args=parser.parse_args()
    try:main(args)
    except BaseException:
        if args.preflight or args.execute:
            p=ROOT/'reports/coverage-priority-failures-20261101';p.mkdir(exist_ok=True);(p/f'failure-{time.time_ns()}.txt').write_text(traceback.format_exc())
        raise
