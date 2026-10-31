"""Audit all E0 queries before class top-k; no updates or prediction replacement."""
import fcntl,hashlib,inspect,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS,MODELS
from mmdet.utils import register_all_modules
from pycocotools.coco import COCO
from pycocotools import mask as mu
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 out=ROOT/'reports/baseline-query-audit-20261031';out.mkdir(exist_ok=True)
 assert not (out/'result.json').exists()
 selpath=ROOT/'reports/baseline-scale-pilot-20261031/selection.json'
 selection=json.loads(selpath.read_text());(out/'selection.json').write_text(json.dumps(selection,indent=2))
 with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  register_all_modules();torch.set_num_threads(2)
  cfg=Config.fromfile(str(ROOT/'reports/uiis-baseline-36e-evaluation/evaluation-config.py'))
  ck=ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth'
  predpath=ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json';ann=ROOT/'data/UIIS/annotations/val.json'
  hashes={str(p.relative_to(ROOT)):sha(p) for p in [ck,predpath,ann,selpath]}
  assert sha(ck)=='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
  ds=DATASETS.build(cfg.test_dataloader.dataset);ds.full_init()
  idx={ds.get_data_info(i)['img_id']:i for i in range(len(ds))}
  model=MODELS.build(cfg.model).cuda().eval();state=torch.load(ck,map_location='cpu')['state_dict'];model.load_state_dict(state,strict=True)
  fusion=model.panoptic_fusion_head
  (out/'postprocess-source.txt').write_text(inspect.getsource(type(fusion).predict)+'\n'+inspect.getsource(type(fusion).instance_postprocess))
  saved=json.loads(predpath.read_text());gt=COCO(str(ann));records=[];checks=[]
  assert fusion.num_classes==7 and fusion.num_things_classes==7
  with torch.inference_mode():
   for iid in selection['image_ids']:
    data=model.data_preprocessor(pseudo_collate([ds.prepare_data(idx[iid])]),training=False)
    cls,padded=model.panoptic_head.predict(model.extract_feat(data['inputs']),data['data_samples'])
    actual=fusion.predict(cls,padded,data['data_samples'],rescale=True)[0]['ins_results']
    meta=data['data_samples'][0].metainfo;h,w=meta['img_shape'][:2];oh,ow=meta['ori_shape'][:2]
    logits=F.interpolate(padded[0,:,:h,:w,None].permute(0,3,1,2),size=(oh,ow),mode='bilinear',align_corners=False)[:,0]
    probs=cls[0].softmax(-1);assert probs.shape==(100,8)
    binary=logits>0;quality=(logits.sigmoid()*binary).flatten(1).sum(1)/(binary.flatten(1).sum(1)+1e-6)
    cp,top=probs[:,:-1].flatten().topk(100,sorted=False);qs=top//7;cats=top%7
    scores=cp*quality[qs]
    assert torch.equal(actual.labels,cats) and torch.equal(actual.masks,binary[qs]) and torch.equal(actual.scores,scores)
    refs=[r for r in saved if r['image_id']==iid];assert len(refs)==100
    assert [r['category_id'] for r in refs]==(cats+1).tolist()
    bm=binary.cpu().numpy();err=max(abs(r['score']-float(scores[j])) for j,r in enumerate(refs))
    xor=sum(int(np.count_nonzero(mu.decode(r['segmentation']).astype(bool)!=bm[int(qs[j])])) for j,r in enumerate(refs))
    assert xor==0 and err<1e-4,(iid,xor,err)
    rles=[]
    for mask in bm:
     r=mu.encode(np.asfortranarray(mask.astype('uint8')));r['counts']=r['counts'].decode('ascii');rles.append(r)
    anns=[a for a in gt.imgToAnns[iid] if not a.get('iscrowd',0)]
    ious=mu.iou(rles,[gt.annToRLE(a) for a in anns],[0]*len(anns))
    chosen=set(top.cpu().tolist());pp=probs.cpu().numpy();qq=quality.cpu().numpy()
    for j,a in enumerate(anns):
     c=a['category_id']-1;queries=[]
     for q in range(100):
      queries.append(dict(query=q,iou=float(ious[q,j]),correct_class_probability=float(pp[q,c]),background_probability=float(pp[q,-1]),top_class=int(pp[q].argmax()),mask_quality=float(qq[q]),correct_class_final_score=float(pp[q,c]*qq[q]),correct_class_selected=q*7+c in chosen,any_class_selected=any(q*7+k in chosen for k in range(7))))
     records.append(dict(gt_id=a['id'],image_id=iid,category_id=a['category_id'],area=a['area'],queries=queries))
    (out/f'image-{iid}.json').write_text(json.dumps(dict(image_id=iid,queries=[dict(query=q,segmentation=rles[q],class_probabilities=pp[q].tolist(),mask_quality=float(qq[q])) for q in range(100)],selected_query_class_flat_indices=top.cpu().tolist())))
    checks.append(dict(image_id=iid,score_max_abs_error=err,mask_changed_pixels=xor,unique_selected_queries=len(set(qs.cpu().tolist()))))
    print(iid,'ALL_100_QUERIES_CHECKED',flush=True)
  assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items())
  assert all(p.grad is None for p in model.parameters())
  assert all(sha(ROOT/p)==v for p,v in hashes.items())
  result=dict(passed=True,training=False,optimizer_created=False,model_state_unchanged=True,input_hashes=hashes,source_sha256=sha(Path(__file__)),checks=checks,records=records)
  (out/'result.json').write_text(json.dumps(result));print('QUERY_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()
