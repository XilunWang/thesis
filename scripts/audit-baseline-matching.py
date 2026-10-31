"""Read-only E0 matching audit on frozen real train augmentations."""
import fcntl,hashlib,inspect,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS,MODELS
from mmdet.utils import register_all_modules
from pycocotools import mask as mu
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def main():
 out=ROOT/'reports/baseline-matching-audit-20261031';selpath=out/'selection.json';sel=json.loads(selpath.read_text());assert not (out/'result.json').exists()
 with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);register_all_modules();torch.set_num_threads(2)
  cfg=Config.fromfile(str(ROOT/'work_dirs/uiis_baseline_r50_36e/config-resume.py'));ck=ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth';ann=ROOT/'data/UIIS/annotations/train.json'
  hashes={str(p.relative_to(ROOT)):sha(p) for p in [ck,ann,selpath]};assert sha(ck)=='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
  cache={r['id']:r for r in [json.loads(l) for l in (ROOT/'reports/uiis-pixel-bll-data-20261031/images.jsonl').read_text().splitlines()]}
  ds=DATASETS.build(cfg.train_dataloader.dataset);ds.full_init();idx={ds.get_data_info(i)['img_id']:i for i in range(len(ds))}
  model=MODELS.build(cfg.model).cuda();state=torch.load(ck,map_location='cpu')['state_dict'];model.load_state_dict(state,strict=True);model.train()
  assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
  head=model.panoptic_head;assign=head.assigner.assign
  (out/'matching-source.txt').write_text(inspect.getsource(type(head)._get_targets_single)+'\n'+inspect.getsource(type(head.assigner).assign))
  results=[];capture={}
  def traced(pred_instances,gt_instances,img_meta=None,**kwargs):
   ar=assign(pred_instances=pred_instances,gt_instances=gt_instances,img_meta=img_meta,**kwargs)
   costs=[cost(pred_instances=pred_instances,gt_instances=gt_instances,img_meta=img_meta) for cost in head.assigner.match_costs];total=torch.stack(costs).sum(0)
   qr,gc=linear_sum_assignment(total.cpu().numpy());expected=np.zeros(len(pred_instances),dtype=int);expected[qr]=gc+1;assert np.array_equal(expected,ar.gt_inds.cpu().numpy())
   capture.clear();capture.update(assignment=ar.gt_inds.cpu().tolist(),costs=[x.cpu().tolist() for x in costs],cost_names=[type(x).__name__ for x in head.assigner.match_costs],sampled_gt_positive=(gt_instances.masks>0).sum(1).cpu().tolist(),sampled_gt_atleast_half=(gt_instances.masks>=.5).sum(1).cpu().tolist(),sampled_gt_mass=gt_instances.masks.sum(1).cpu().tolist())
   return ar
  with torch.no_grad():
   for iid in sel['image_ids']:
    seed(sel['augmentation_seed_base']+iid);item=ds.prepare_data(idx[iid]);assert item is not None
    data=model.data_preprocessor(pseudo_collate([item]),training=True);sample=data['data_samples'][0];meta=sample.metainfo
    target=head.preprocess_gt([sample.gt_instances],[None])[0];g=target.masks;labels=target.labels;n=len(labels);assert 0<n<=100
    assert labels.cpu().tolist()==cache[iid]['labels'];assert g.flatten(1).sum(1).cpu().tolist()==cache[iid]['gt_pixels']
    allcls,allmasks=head(model.extract_feat(data['inputs']),data['data_samples']);cls=allcls[-1][0];masks=allmasks[-1][0]
    up=F.interpolate(masks[:,None],size=g.shape[-2:],mode='bilinear',align_corners=False)[:,0];binary=up>0;probs=cls.softmax(-1);quality=(up.sigmoid()*binary).flatten(1).sum(1)/(binary.flatten(1).sum(1)+1e-6)
    bm=binary.cpu().numpy();gm=g.cpu().numpy();pr=[];gr=[]
    for arr,container in [(bm,pr),(gm,gr)]:
     for m in arr:
      r=mu.encode(np.asfortranarray(m.astype('uint8')));r['counts']=r['counts'].decode('ascii');container.append(r)
    ious=mu.iou(pr,gr,[0]*n);runs=[]
    for offset in sel['matching_seed_offsets']:
     sd=sel['matching_seed_base']+iid+offset;seed(sd);head.assigner.assign=traced
     first=head._get_targets_single(cls,masks,target,meta);run=json.loads(json.dumps(capture));head.assigner.assign=assign
     seed(sd);second=head._get_targets_single(cls,masks,target,meta)
     assert all(torch.equal(first[k],second[k]) for k in range(6));assert torch.equal(first[6].pos_assigned_gt_inds,second[6].pos_assigned_gt_inds)
     assert sorted(v for v in run['assignment'] if v>0)==list(range(1,n+1))
     for q,v in enumerate(run['assignment']):assert int(first[0][q])==(int(labels[v-1]) if v else head.num_classes)
     run.update(seed=sd,offset=offset,original_target_parity=True);runs.append(run)
    record=dict(image_id=iid,labels=labels.cpu().tolist(),gt_pixels=g.flatten(1).sum(1).cpu().tolist(),shape=list(g.shape[-2:]),ious=ious.tolist(),class_probabilities=probs.cpu().tolist(),mask_quality=quality.cpu().tolist(),pred_rles=pr,gt_rles=gr,runs=runs,num_points=head.num_points)
    (out/f'image-{iid}.json').write_text(json.dumps(record));results.append(dict(image_id=iid,instances=n,small=sum(0<a<1024 for a in record['gt_pixels'])))
    print(iid,'MATCHING_3_SEEDS_VERIFIED',flush=True)
  assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items()) and all(p.grad is None for p in model.parameters())
  assert all(sha(ROOT/p)==v for p,v in hashes.items())
  result=dict(passed=True,training_updates=0,optimizer_created=False,model_state_unchanged=True,model_forward_mode='train, frozen original BN',input_hashes=hashes,source_sha256=sha(Path(__file__)),images=results)
  (out/'result.json').write_text(json.dumps(result,indent=2));print('MATCHING_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()
