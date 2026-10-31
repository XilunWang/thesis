"""Same-forward gradient decomposition. No optimizer and no parameter updates."""
import argparse, fcntl, hashlib, json, math, random, sys
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.pixel_bll_head as adapter
from scripts.query_bll import query_boundary_loss
IDS=[1334,345,3541,8,113,308,0,1,2,3,4,5]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def group(n):
    for k in ['backbone','pixel_decoder','transformer_decoder','query','mask_embed','cls_embed']:
        if k in n:return k
    return 'other'

def stats(a,b,indices,clip):
    aa=bb=ab=0.
    for i in indices:
        x=a[i].double();y=b[i].double();aa+=float((x*x).sum());bb+=float((y*y).sum());ab+=float((x*y).sum())
    na=math.sqrt(aa);nb=math.sqrt(bb);nt=math.sqrt(max(0,aa+bb+2*ab))
    ca=min(1.,clip/(na+1e-6));ct=min(1.,clip/(nt+1e-6))
    return dict(original_norm=na,bll_norm=nb,bll_to_original=nb/na if na else None,
                cosine=ab/(na*nb) if na*nb else None,total_norm=nt,
                total_vs_original_cosine=(aa+ab)/(na*nt) if na*nt else None,
                original_clip_factor=ca,combined_clip_factor=ct,original_contribution_scale=ct/ca)

def main():
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,choices=[1,36],required=True);args=p.parse_args()
    out=ROOT/'reports/bll-gradient-closeout-20261031'/f'epoch-{args.epoch}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'result.json').exists()
    with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        register_all_modules();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        work=ROOT/'work_dirs/uiis_e1p_pixel_bll_r50_36e';ck=work/f'epoch_{args.epoch}.pth'
        if args.epoch==1:ck=ROOT/'checkpoints/diagnostic-e1p-epoch1.pth'
        digest=sha(ck);assert digest=={1:'ada5705de4ad9a7a57cb35e4a1862e81f7cf84fc013830a05d52dcbcdb159353',36:'ad1a2cb9c886045aed6902716b3e73f866464a02b0debcaac3fe328fa4815d2e'}[args.epoch]
        cfg=Config.fromfile(str(work/'config-resume.py'));clip=cfg.optim_wrapper.clip_grad.max_norm;assert clip==.01
        dataset=DATASETS.build(cfg.train_dataloader.dataset);dataset.full_init();items=[];coverage=[]
        for iid in IDS:
            seed=20261031+iid;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            idx=next(i for i in range(len(dataset)) if dataset.get_data_info(i)['img_id']==iid)
            item=dataset.prepare_data(idx);assert item is not None and item['data_samples'].img_id==iid
            gt=item['data_samples'].gt_instances
            coverage.append(dict(image_id=iid,labels=gt.labels.tolist(),areas=gt.masks.areas.tolist(),input_sha256=hashlib.sha256(item['inputs'].numpy().tobytes()).hexdigest()))
            items.append(item)
        model=MODELS.build(cfg.model).cuda();state=torch.load(ck,map_location='cpu')['state_dict'];model.load_state_dict(state,strict=True);model.train()
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for n,p in named]
        groups={k:[i for i,(n,p) in enumerate(named) if group(n)==k] for k in sorted({group(n) for n,p in named})}
        result=dict(epoch=args.epoch,checkpoint_sha256=digest,coverage=coverage,parameter_groups={k:len(v) for k,v in groups.items()},batches=[],optimizer_created=False)
        production=adapter.pixel_boundary_loss;live={}
        def tap(*a,**kw):
            value,diag=production(*a,**kw);old,od=query_boundary_loss(*a,**kw)
            live.update(old=old,diag=diag,old_diag=od);return value,diag
        adapter.pixel_boundary_loss=tap
        try:
            for batch in range(6):
                seed=20261101+batch;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                data=model.data_preprocessor(pseudo_collate(items[batch*2:batch*2+2]),training=True)
                losses=model(**data,mode='loss');new=losses['loss_bll_pixel'];original=sum(v for k,v in losses.items() if k!='loss_bll_pixel')
                def grad(loss,retain=True):
                    gg=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
                    assert all(g is None or torch.isfinite(g).all() for g in gg)
                    return [torch.zeros_like(p,device='cpu') if g is None else g.detach().cpu() for p,g in zip(params,gg)]
                go=grad(original);repeat=grad(original);old=grad(live['old']);gn=grad(new,False)
                repeat_norm=math.sqrt(sum(float((a.double()-b.double()).square().sum()) for a,b in zip(go,repeat)))
                base_norm=math.sqrt(sum(float(a.double().square().sum()) for a in go))
                rr=dict(image_ids=IDS[batch*2:batch*2+2],loss_original=float(original.detach()),loss_old=float(live['old'].detach()),loss_new=float(new.detach()),repeat_relative_l2=repeat_norm/max(base_norm,1e-12),old={},new={})
                for label,g in [('old',old),('new',gn)]:
                    rr[label]['all']=stats(go,g,range(len(params)),clip)
                    rr[label]['groups']={k:{key:v for key,v in stats(go,g,ix,clip).items() if key not in ['original_clip_factor','combined_clip_factor','original_contribution_scale']} for k,ix in groups.items()}
                result['batches'].append(rr)
                (out/'progress.json').write_text(json.dumps(result,indent=2));print(args.epoch,batch,'GRADIENT_CHECKED',flush=True)
                del go,repeat,old,gn,losses,original,new,data;live.clear()
        finally:adapter.pixel_boundary_loss=production
        assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())
        assert all(p.grad is None for p in params);assert sha(ck)==digest
        result.update(passed=True,model_state_unchanged=True,source_sha256=sha(Path(__file__)),scope='Same graph; total gradient formed by vector addition; clipping diagnostic is not AdamW step or causal AP attribution.')
        (out/'result.json').write_text(json.dumps(result,indent=2));print('GRADIENT_CLOSEOUT_PASSED',args.epoch,flush=True)

if __name__=='__main__':main()
