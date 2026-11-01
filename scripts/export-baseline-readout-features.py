"""Export frozen E0 features for the preselected three-image readout diagnostic."""
import fcntl,hashlib,inspect,json,time
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
 out=ROOT/'reports/feature-readout-export-20261101';selpath=out/'selection.json';sel=json.loads(selpath.read_text());assert not (out/'result.json').exists()
 assert sel['image_ids']==[2,19,42] and len(sel['cases'])==14
 started=time.monotonic()
 with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);register_all_modules();torch.set_num_threads(2)
  cfg=Config.fromfile(str(ROOT/'reports/uiis-baseline-36e-evaluation/evaluation-config.py'))
  ck=ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth';predpath=ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json';ann=ROOT/'data/UIIS/annotations/val.json'
  hashes={str(p.relative_to(ROOT)):sha(p) for p in [ck,predpath,ann,selpath]};assert sha(ck)=='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
  ds=DATASETS.build(cfg.test_dataloader.dataset);ds.full_init();idx={ds.get_data_info(i)['img_id']:i for i in range(len(ds))}
  model=MODELS.build(cfg.model).cuda().eval();state=torch.load(ck,map_location='cpu')['state_dict'];model.load_state_dict(state,strict=True)
  head=model.panoptic_head;fusion=model.panoptic_fusion_head
  capture={}
  def pixel_hook(module,args,output): capture['mask_features']=output[0].detach()
  def embed_hook(module,args,output): capture.setdefault('embeddings',[]).append(output.detach())
  hooks=[head.pixel_decoder.register_forward_hook(pixel_hook),head.mask_embed.register_forward_hook(embed_hook)]
  (out/'head-source.txt').write_text(inspect.getsource(type(head).forward)+'\n'+inspect.getsource(type(head).predict))
  saved=json.loads(predpath.read_text());gt=COCO(str(ann));checks=[]
  with torch.inference_mode():
   for iid in sel['image_ids']:
    assert time.monotonic()-started<1200,'20 minute export budget exceeded'
    capture.clear()
    data=model.data_preprocessor(pseudo_collate([ds.prepare_data(idx[iid])]),training=False)
    features=model.extract_feat(data['inputs'])
    classes,masks=head(features,data['data_samples']);assert len(classes)==len(masks)==10
    assert len(features)==4 and len(capture['embeddings'])==10
    reconstruction_errors=[]
    for emb,raw in zip(capture['embeddings'],masks):
     reconstructed=torch.einsum('bqc,bchw->bqhw',emb,capture['mask_features'])
     reconstruction_errors.append(float((reconstructed-raw).abs().max()))
     assert torch.equal(reconstructed,raw),'Embedding reconstruction differs'
    arrays={f'C{i+2}':v.detach().cpu().numpy() for i,v in enumerate(features)}
    arrays.update(mask_features=capture['mask_features'].cpu().numpy(),embeddings=torch.stack(capture['embeddings']).cpu().numpy(),raw_masks=torch.stack(masks).cpu().numpy())
    assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in arrays.values())
    np.savez(out/f'features-{iid}.npz',**arrays)
    meta=data['data_samples'][0].metainfo;h,w=meta['img_shape'][:2];oh,ow=meta['ori_shape'][:2];bh,bw=meta['batch_input_shape'][:2]
    gids=[r['gt_id'] for r in sel['cases'] if r['image_id']==iid];anns=[gt.anns[g] for g in gids];layers=[]
    for layer,(cls,raw) in enumerate(zip(classes,masks)):
     padded=F.interpolate(raw,size=(bh,bw),mode='bilinear',align_corners=False)
     logits=F.interpolate(padded[0,:,:h,:w,None].permute(0,3,1,2),size=(oh,ow),mode='bilinear',align_corners=False)[:,0]
     probs=cls[0].softmax(-1);assert probs.shape==(100,8)
     binary=logits>0;quality=(logits.sigmoid()*binary).flatten(1).sum(1)/(binary.flatten(1).sum(1)+1e-6)
     cp,top=probs[:,:-1].flatten().topk(100,sorted=False);qs=top//7;cats=top%7;scores=cp*quality[qs]
     bm=binary.cpu().numpy();rles=[]
     for mask in bm:
      r=mu.encode(np.asfortranarray(mask.astype('uint8')));r['counts']=r['counts'].decode('ascii');rles.append(r)
     if layer==9:
      actual=fusion.predict(cls,padded,data['data_samples'],rescale=True)[0]['ins_results']
      assert torch.equal(actual.labels,cats) and torch.equal(actual.masks,binary[qs]) and torch.equal(actual.scores,scores)
      refs=[r for r in saved if r['image_id']==iid];assert len(refs)==100 and [r['category_id'] for r in refs]==(cats+1).tolist()
      err=max(abs(r['score']-float(scores[j])) for j,r in enumerate(refs));xor=sum(int(np.count_nonzero(mu.decode(r['segmentation']).astype(bool)!=bm[int(qs[j])])) for j,r in enumerate(refs))
      assert xor==0 and err<1e-4;(checks.append(dict(image_id=iid,max_score_error=err,mask_changed_pixels=xor)))
     ious=mu.iou(rles,[gt.annToRLE(a) for a in anns],[0]*len(anns))
     layers.append(dict(stage=layer,class_probabilities=probs.cpu().tolist(),mask_quality=quality.cpu().tolist(),selected_query_class_flat_indices=top.cpu().tolist(),rles=rles,ious=ious.tolist()))
    (out/f'image-{iid}.json').write_text(json.dumps(dict(image_id=iid,gt_ids=gids,layers=layers,metainfo={k:list(meta[k]) for k in ['img_shape','ori_shape','batch_input_shape','scale_factor']},feature_sha256=sha(out/f'features-{iid}.npz'),feature_shapes={k:list(v.shape) for k,v in arrays.items()},reconstruction_errors=reconstruction_errors)))
    print(iid,'ALL_10_STAGES_CHECKED',flush=True)
  for hook in hooks: hook.remove()
  assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items()) and all(p.grad is None for p in model.parameters())
  assert all(sha(ROOT/p)==v for p,v in hashes.items())
  result=dict(passed=True,training=False,optimizer_created=False,model_state_unchanged=True,input_hashes=hashes,source_sha256=sha(Path(__file__)),checks=checks,stages=10,queries_per_stage=100,elapsed_seconds=time.monotonic()-started)
  (out/'result.json').write_text(json.dumps(result,indent=2));print('FEATURE_EXPORT_PASSED',flush=True)
if __name__=='__main__':main()
