"""Explicitly gated 300-step pilot. Default --check never calls optimizer.step."""
import argparse
import fcntl
import hashlib
import json
import random
import time
import traceback
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from pycocotools import mask as mu
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS,MODELS
from mmdet.utils import register_all_modules
from scripts.position_grid4 import Grid4PositionHead,make_targets,position_loss,select
from scripts.identity_data_pipeline import with_identity
from scripts.position_pilot_runtime import digest,rng_state,restore_rng,view_at,save_checkpoint,load_checkpoint
from scripts.position_grid4_metrics import measure,aggregate,random_mean,decide
PLAN=ROOT/'reports/position-learning-plan-grid4-20261101/manifest.json'
CHECK=ROOT/'reports/position-grid4-entry-20261101'
RUN=ROOT/'work_dirs/position_grid4_pilot'
CODE=['run-position-grid4-pilot.py','position_grid4_metrics.py','position_grid4.py','isolated_position_head.py','identity_data_pipeline.py','position_pilot_runtime.py']


def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def array_hash(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()
def write_json(p,value):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(p)


class Pilot:
    def __init__(self):
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        assert digest(PLAN)=='efbdf5f3afe9bb8a9dc5be5449dfdbf1d1f1ad0d4786aa37f83ea1d95ee5fd04'
        self.plan=json.loads(PLAN.read_text());self.codes={n:digest(ROOT/'scripts'/n) for n in CODE}
        assert self.plan['representation']=='grid4-v1' and self.plan['steps']==300 and self.plan['batch_size']==2
        for n,h in self.plan['code_sha256'].items():assert digest(ROOT/'scripts'/n)==h
        vp=PLAN.parent/'views.json';assert digest(vp)==self.plan['views_sha256']
        views=json.loads(vp.read_text());self.views=views['runs']
        self.strata={r['image_id']:r['stratum'] for split in self.plan['splits'].values() for r in split}
        self.neighbors={(r['image_id'],r['seed']):r['pairs'] for r in views['frozen_neighbor_observations']}
        self.learn=[r for r in self.views if r['split']=='learn'];assert len(self.learn)==256
        assert len(self.views)==384
        assert not {r['image_id'] for r in self.learn}&{r['image_id'] for r in self.views if r['split']=='holdout'}
        assert digest(ROOT/'data/UIIS/annotations/train.json')==self.plan['source_sha256']
        self.options=dict(manifest_hash=digest(PLAN),code_hashes=self.codes,view_hash=digest(vp),kind='grid4_pilot')
        register_all_modules()
        cfg=Config.fromfile(str(ROOT/'reports/uiis-baseline-36e-evaluation/evaluation-config.py'))
        cp=ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth'
        assert digest(cp)=='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
        self.initial=torch.load(cp,map_location='cpu')['state_dict']
        self.model=MODELS.build(cfg.model);self.model.load_state_dict(self.initial,strict=True)
        self.model.requires_grad_(False);self.model.cuda().eval()
        config=ROOT/'work_dirs/uiis_baseline_r50_36e/config-resume.py'
        preflight=json.loads((ROOT/'reports/position-learning-preflight-20261101/result.json').read_text())
        assert digest(config)==preflight['config_sha256']
        traincfg=Config.fromfile(str(config));self.ds=DATASETS.build(with_identity(traincfg.train_dataloader.dataset));self.ds.full_init()
        self.index={self.ds.get_data_info(i)['img_id']:i for i in range(len(self.ds))}
        seed(self.plan['initialization_seed']);self.head=Grid4PositionHead().cuda()
        p=self.plan['optimizer'];self.optimizer=torch.optim.AdamW(self.head.parameters(),lr=p['lr'],betas=tuple(p['betas']),eps=p['eps'],weight_decay=p['weight_decay'])
        assert {id(p) for g in self.optimizer.param_groups for p in g['params']}=={id(p) for p in self.head.parameters()}

    def data(self,row):
        seed(row['seed']);item=self.ds.prepare_data(self.index[row['image_id']]);assert item is not None
        data=self.model.data_preprocessor(pseudo_collate([item]),training=True);gt=data['data_samples'][0].gt_instances
        masks=gt.masks.masks
        assert array_hash(data['inputs'].cpu().numpy())==row['input_sha256']
        assert array_hash(masks)==row['masks_sha256']
        target,valid,assignment=make_targets(masks,gt.annotation_ids.tolist(),gt.original_areas.tolist(),row['valid_shape'])
        assert {str(k):v for k,v in assignment.items()}==row['assignments']
        assert array_hash(target.numpy())==row['target_sha256'] and array_hash(valid.numpy())==row['valid_sha256']
        return data,target,valid

    def protected(self):
        assert all(p.grad is None for p in self.model.parameters())
        assert all(torch.equal(v.cpu(),self.initial[k]) for k,v in self.model.state_dict().items())

    def original_output(self):
        state=rng_state();row=next(r for r in self.views if r['image_id']==2809 and r['seed']==20263821)
        data,_,_=self.data(row)
        with torch.no_grad():parts=self.model.panoptic_head(self.model.extract_feat(data['inputs']),data['data_samples'])
        result=[[array_hash(t.cpu().numpy()) for t in part] for part in parts];restore_rng(state);return result

    def eval(self,step,destination,rows=None,random_controls=False):
        path=destination/f'eval-{step}.json'
        if path.exists():
            old=json.loads(path.read_text());assert old['codes']==self.codes and old['manifest_hash']==digest(PLAN)
            return old
        state=rng_state();self.head.eval();rows=self.views if rows is None else rows
        records=[]
        with torch.no_grad():
            for row in rows:
                data,_,valid=self.data(row);features=self.model.extract_feat(data['inputs'])
                logits=self.head(features[1],features[2])[0].cpu().numpy()
                gt=data['data_samples'][0].gt_instances
                base=dict(image_id=row['image_id'],augmentation_seed=row['seed'],split=row['split'],stratum=self.strata[row['image_id']])
                pairs=self.neighbors.get((row['image_id'],row['seed']),[])
                sources={'head':None}
                if random_controls:
                    cells=np.flatnonzero(valid.numpy())
                    for s in self.plan['random_control_seeds']:
                        rng=np.random.default_rng(np.random.SeedSequence([s,row['image_id'],row['seed']]))
                        sources[f'random_{s}']=rng.permutation(cells)[:200]
                for name,order in sources.items():
                    for k in self.plan['diagnostic_k']:
                        selected,coordinates=select(logits,row['valid_shape'],k) if order is None else (order[:k],None)
                        metric=measure(selected,row['assignments'],gt.masks.masks,gt.annotation_ids.tolist(),gt.original_areas.tolist(),gt.labels.tolist(),pairs)
                        records.append(dict(**base,source=name,k=k,selected_cells=selected.tolist(),**metric))
        result=dict(step=step,codes=self.codes,manifest_hash=digest(PLAN),records=records,summaries={})
        for split in ['learn','holdout']:
            for name in sorted({r['source'] for r in records}):
                for k in self.plan['diagnostic_k']:
                    subset=[r for r in records if r['split']==split and r['source']==name and r['k']==k]
                    if subset:result['summaries'][f'{split}/{name}/{k}']=aggregate(subset)
                    for stratum in ['dense','sparse_coral','sparse_other','empty']:
                        group=[r for r in subset if r['stratum']==stratum]
                        if group:result['summaries'][f'{split}/{stratum}/{name}/{k}']=aggregate(group)
        result['random_mean_std']={}
        if random_controls:
            for split in ['learn','holdout']:
                for k in self.plan['diagnostic_k']:
                    summaries=[result['summaries'].get(f'{split}/random_{s}/{k}') for s in self.plan['random_control_seeds']]
                    if not all(summaries):continue
                    result['random_mean_std'][f'{split}/{k}']={}
                    for metric in ['occupied_recall','seed_recall','utilization','pair_recall','image_mean','no_touch','redundant']:
                        vals=[s[metric] for s in summaries if s[metric] is not None]
                        result['random_mean_std'][f'{split}/{k}'][metric]=dict(mean=float(np.mean(vals)),std=float(np.std(vals))) if vals else None
        write_json(path,result);restore_rng(state);return result

    def backward(self,rows):
        pieces=[self.data(r) for r in rows]
        inputs=torch.cat([d['inputs'] for d,t,v in pieces])
        with torch.no_grad():features=self.model.extract_feat(inputs)
        target=torch.stack([t for d,t,v in pieces]).cuda();valid=torch.stack([v for d,t,v in pieces]).cuda()
        self.optimizer.zero_grad(set_to_none=True)
        logits=self.head(features[1],features[2]);loss=position_loss(logits,target,valid)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.head.parameters())
        assert all(p.grad is None for p in self.model.parameters())
        norm=torch.nn.utils.clip_grad_norm_(self.head.parameters(),self.plan['grad_clip_norm'],error_if_nonfinite=True)
        return float(loss.detach()),float(norm)


def check(pilot):
    CHECK.mkdir(exist_ok=False)
    def forbidden(*a,**kw):raise AssertionError('Parameter update forbidden in check mode')
    pilot.optimizer.step=forbidden
    before={k:v.clone() for k,v in pilot.head.state_dict().items()};original=pilot.original_output()
    # Exercise actual batch=2 at the collision image and a dense held-out view, no update.
    chosen=[next(r for r in pilot.views if r['image_id']==i and r['seed']==s) for i,s in [(2809,20263821),(2971,20263983),(1911,20262923)]]
    loss,norm=pilot.backward(chosen[:2]);pilot.protected()
    assert all(torch.equal(before[k],v) for k,v in pilot.head.state_dict().items())
    assert not pilot.optimizer.state and original==pilot.original_output()
    pilot.eval(0,CHECK,chosen,random_controls=True)
    order=[view_at(pilot.learn,i,pilot.plan['order_seed']) for i in range(600)]
    assert all(r['split']=='learn' for r in order)
    save_checkpoint(CHECK/'step0.pth',pilot.head,pilot.optimizer,**pilot.options,step=0,cursor=0,elapsed_seconds=0)
    load_checkpoint(CHECK/'step0.pth',pilot.head,pilot.optimizer,**pilot.options)
    assert all(torch.equal(before[k],v) for k,v in pilot.head.state_dict().items())
    resume=json.loads((ROOT/'reports/position-grid4-resume-20261101/result.json').read_text());assert resume['passed'] and resume['training_updates']==0
    for n,h in resume['source_sha256'].items():assert digest(ROOT/'scripts'/n)==h
    result=dict(passed=True,training_updates=0,codes=pilot.codes,manifest_hash=digest(PLAN),view_hash=pilot.options['view_hash'],
                batch2_backward_loss=loss,gradient_norm_before_clip=norm,parameters_unchanged=True,original_output_exact=True,
                evaluated_views=3,training_order_entries=600,resume_cpu_proof=digest(ROOT/'reports/position-grid4-resume-20261101/result.json'))
    write_json(CHECK/'result.json',result);print('ENTRY_CHECK_PASSED_NO_UPDATES',flush=True)


def train(pilot,resume):
    proof=json.loads((CHECK/'result.json').read_text())
    assert proof['passed'] and proof['codes']==pilot.codes and proof['manifest_hash']==digest(PLAN)
    if resume:
        assert RUN.exists()
        state=load_checkpoint(RUN/'latest.pth',pilot.head,pilot.optimizer,**pilot.options)
        step=state['step'];elapsed=state['elapsed_seconds']
        original=json.loads((RUN/'original-output.json').read_text())
    else:
        RUN.mkdir(exist_ok=False);step=0;elapsed=0.;original=pilot.original_output();write_json(RUN/'original-output.json',original)
        save_checkpoint(RUN/'latest.pth',pilot.head,pilot.optimizer,**pilot.options,step=0,cursor=0,elapsed_seconds=0)
        save_checkpoint(RUN/'step0.pth',pilot.head,pilot.optimizer,**pilot.options,step=0,cursor=0,elapsed_seconds=0)
    # Evaluate a checkpoint before leaving its step; interruption resumes missing evaluation.
    while True:
        pilot.protected();assert pilot.original_output()==original
        if step in [0,100,300]:
            save_checkpoint(RUN/f'step{step}.pth',pilot.head,pilot.optimizer,**pilot.options,step=step,cursor=step*2,elapsed_seconds=elapsed)
            pilot.eval(step,RUN,random_controls=step==0)
        if step==300 or elapsed>=1800:break
        stop=min(100 if step<100 else 300,300)
        start=time.monotonic();base_elapsed=elapsed
        while step<stop and elapsed<1800:
            rows=[view_at(pilot.learn,step*2+j,pilot.plan['order_seed']) for j in [0,1]]
            pilot.head.train();loss,norm=pilot.backward(rows)
            pilot.optimizer.step()
            assert all(torch.isfinite(p).all() for p in pilot.head.parameters())
            step+=1;elapsed=base_elapsed+time.monotonic()-start
            save_checkpoint(RUN/'latest.pth',pilot.head,pilot.optimizer,**pilot.options,step=step,cursor=step*2,elapsed_seconds=elapsed)
            with (RUN/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=loss,gradient_norm=norm,elapsed_seconds=elapsed))+'\n')
            if step%25==0:print('STEP',step,'LOSS',loss,flush=True)
    pilot.protected();assert pilot.original_output()==original
    result=dict(completed=step==300,step=step,elapsed_training_seconds=elapsed,original_preserved=True,
                note='Evaluation time excluded from the 30-minute training-loop cap; no automatic extension')
    if step==300:
        initial=json.loads((RUN/'eval-0.json').read_text());final=json.loads((RUN/'eval-300.json').read_text())
        def subset(d,name):return [r for r in d['records'] if r['split']=='holdout' and r['k']==100 and r['source']==name]
        random_groups=[subset(initial,f'random_{s}') for s in pilot.plan['random_control_seeds']]
        result.update(decide(subset(final,'head'),subset(initial,'head'),random_mean(random_groups),pilot.plan))
        result['random_recall_mean']=float(np.mean([aggregate(g)['occupied_recall'] for g in random_groups]))
        result['random_recall_std']=float(np.std([aggregate(g)['occupied_recall'] for g in random_groups]))
        result['passed']=result['position_gates_passed'] and result['original_preserved']
    else:result.update(passed=False,status='TIME_BUDGET_INCOMPLETE_NOT_METHOD_FAILURE')
    write_json(RUN/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.resume and not args.execute:parser.error('--resume requires explicit --execute')
    with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            pilot=Pilot()
            if args.execute:train(pilot,args.resume)
            else:check(pilot)
        except BaseException:
            destination=RUN if args.execute else CHECK
            destination.mkdir(exist_ok=True)
            (destination/f'failure-{time.time_ns()}.txt').write_text(traceback.format_exc())
            raise
