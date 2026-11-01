"""R50 direct-query integration acceptance. Forward/backward only, no optimizer."""
import copy,fcntl,gc,hashlib,json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS,MODELS
from mmdet.utils import register_all_modules
from mmdet.models.dense_heads import Mask2FormerHead
from pycocotools import mask as mu
from scripts.direct_query_feature_head import DirectQueryFeatureHead
OUT=ROOT/'reports/direct-query-integration-20261101'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

def mode(head,name):
 head.direct_query_features=name=='on'
 head.forward=(Mask2FormerHead.forward if name=='parent' else DirectQueryFeatureHead.forward).__get__(head,type(head))

def prediction(model,data):
 cls,masks=model.panoptic_head(model.extract_feat(data['inputs']),data['data_samples'])
 masks=F.interpolate(masks[-1],size=data['data_samples'][0].metainfo['batch_input_shape'],mode='bilinear',align_corners=False)
 return model.panoptic_fusion_head.predict(cls[-1],masks,data['data_samples'],rescale=True)[0]['ins_results']

def cpu_pred(x):return {k:getattr(x,k).detach().cpu().clone() for k in ['labels','scores','masks']}

def unchanged(model,state):
 assert all(torch.equal(v.detach().cpu(),state[k]) for k,v in model.state_dict().items())

def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists()
 with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);register_all_modules();torch.set_num_threads(2)
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;seed(20261101)
  cfg=Config.fromfile(str(ROOT/'reports/uiis-baseline-36e-evaluation/evaluation-config.py'))
  ck=ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth';assert sha(ck)=='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
  initial=torch.load(ck,map_location='cpu')['state_dict']
  cfg.model.panoptic_head.type='DirectQueryFeatureHead'
  model=MODELS.build(cfg.model);head=model.panoptic_head;assert head.direct_query_features is False
  head.init_weights();assert torch.count_nonzero(head.direct_c3.weight)==torch.count_nonzero(head.direct_c4.weight)==0
  missing=model.load_state_dict(initial,strict=False)
  extra=['panoptic_head.direct_c3.weight','panoptic_head.direct_c4.weight']
  assert sorted(missing.missing_keys)==extra and not missing.unexpected_keys
  state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};assert all(torch.equal(state[k],v) for k,v in initial.items())
  assert sum(state[k].numel() for k in extra)==393216
  for name in ['predict','loss','loss_by_feat','_get_targets_single']:
   assert getattr(DirectQueryFeatureHead,name) is getattr(Mask2FormerHead,name)
  model.cuda().eval();head=model.panoptic_head
  # A nonzero synthetic branch checks scale routing and channel-only normalization.
  with torch.no_grad():
   features=[torch.randn(1,c,h,w,device='cuda') for c,h,w in [(256,16,24),(512,8,12),(1024,4,6),(2048,2,3)]]
   for projection in [head.direct_c3,head.direct_c4]:projection.weight[0,0,0,0]=1
   for level,hw in enumerate([(2,3),(4,6),(8,12)]):
    memory=torch.randn(1,256,*hw,device='cuda');actual=head.add_direct_features(memory,features,level);expected=memory.clone()
    if level:
     raw=features[2 if level==1 else 1];norm=F.layer_norm(raw.permute(0,2,3,1),(raw.shape[1],),eps=1e-5);expected[:,0]+=norm[:,:,:,0]
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-5)
   try:head.add_direct_features(torch.zeros(1,256,7,12,device='cuda'),features,2)
   except ValueError:pass
   else:raise AssertionError('Misaligned shape accepted')
   head._zero_direct_weights()
  del features,actual,expected,memory,raw,norm;unchanged(model,state)
  ds=DATASETS.build(cfg.test_dataloader.dataset);ds.full_init();indices={ds.get_data_info(i)['img_id']:i for i in range(len(ds))}
  saved=json.loads((ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json').read_text());parity=[]
  with torch.inference_mode():
   for iid in [2,19,42]:
    data=model.data_preprocessor(pseudo_collate([ds.prepare_data(indices[iid])]),training=False)
    preds={}
    for name in ['parent','off','on']:
     mode(head,name);preds[name]=cpu_pred(prediction(model,data))
    assert all(torch.equal(preds['parent'][k],preds[name][k]) for name in ['off','on'] for k in preds['parent'])
    refs=[r for r in saved if r['image_id']==iid];assert len(refs)==100
    assert preds['parent']['labels'].add(1).tolist()==[r['category_id'] for r in refs]
    score_error=max(abs(float(preds['parent']['scores'][j])-r['score']) for j,r in enumerate(refs))
    xor=sum(int(np.count_nonzero(preds['parent']['masks'][j].numpy()!=mu.decode(r['segmentation']).astype(bool))) for j,r in enumerate(refs))
    assert xor==0 and score_error<1e-4
    parity.append(dict(image_id=iid,parent_off_on_exact=True,mask_changed_pixels=xor,max_score_error=score_error))
    print('OUTPUT_PARITY',iid,flush=True)
  del data,preds
  traincfg=Config.fromfile(str(ROOT/'work_dirs/uiis_baseline_r50_36e/config-resume.py'))
  trainset=DATASETS.build(traincfg.train_dataloader.dataset);trainset.full_init();ti={trainset.get_data_info(i)['img_id']:i for i in range(len(trainset))}
  cache={x['id']:x for x in [json.loads(line) for line in (ROOT/'reports/uiis-pixel-bll-data-20261031/images.jsonl').read_text().splitlines()]}
  gradients=[]
  for iid in [345,620]:
   model.train();model.zero_grad(set_to_none=True);seed(20261012+iid)
   data=model.data_preprocessor(pseudo_collate([trainset.prepare_data(ti[iid])]),training=True)
   assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
   sample=data['data_samples'][0];gt=head.preprocess_gt([sample.gt_instances],[None])[0]
   assert gt.labels.cpu().tolist()==cache[iid]['labels'] and gt.masks.flatten(1).sum(1).cpu().tolist()==cache[iid]['gt_pixels']
   reference=None;rng_ref=None
   for name in ['parent','off','on']:
    mode(head,name);seed(20261101+iid)
    with torch.set_grad_enabled(name=='on'):
     loss=model.loss(data['inputs'],data['data_samples'])
    values={k:v.detach().cpu().clone() for k,v in loss.items()};rng=(torch.get_rng_state(),torch.cuda.get_rng_state())
    if reference is None:reference=values;rng_ref=rng
    else:
     assert reference.keys()==values.keys() and all(torch.equal(reference[k],values[k]) for k in reference)
     assert all(torch.equal(a,b) for a,b in zip(rng_ref,rng))
    if name=='on':sum(loss.values()).backward()
    del loss
   grad={k:p.grad for k,p in model.named_parameters() if p.grad is not None}
   assert grad and all(torch.isfinite(g).all() for g in grad.values())
   branch={k:{'norm':float(grad[k].norm()),'abs_max':float(grad[k].abs().max()),'nonzero':int(torch.count_nonzero(grad[k]))} for k in extra}
   assert all(v['norm']>0 and v['nonzero']>0 for v in branch.values());unchanged(model,state)
   gradients.append(dict(image_id=iid,valid_shape=list(sample.metainfo['img_shape']),padded_shape=list(data['inputs'].shape[-2:]),gt_count=len(gt.labels),loss_terms=len(reference),loss_parent_off_on_exact=True,rng_exact=True,finite_gradient_tensors=len(grad),branch_gradients=branch))
   print('BACKWARD_OK',iid,branch,flush=True);del grad,data,gt,sample
  model.zero_grad(set_to_none=True);model.eval();gc.collect();torch.cuda.empty_cache()
  # ABBA order; same resident, preprocessed input; complete backbone/head/postprocess.
  performance=[]
  with torch.inference_mode():
   for iid in [2,19,42]:
    data=model.data_preprocessor(pseudo_collate([ds.prepare_data(indices[iid])]),training=False);blocks=[]
    for name in ['parent','on','on','parent']:
     mode(head,name)
     for mod in [head.direct_c3,head.direct_c4]:mod.to('cpu' if name=='parent' else 'cuda')
     gc.collect();torch.cuda.empty_cache()
     for _ in range(5):prediction(model,data)
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
     for _ in range(20):
      torch.cuda.synchronize();t=time.perf_counter();prediction(model,data);torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
     blocks.append(dict(mode=name,milliseconds=times,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    base=[t for b in blocks if b['mode']=='parent' for t in b['milliseconds']];candidate=[t for b in blocks if b['mode']=='on' for t in b['milliseconds']]
    base_peak=max(b['peak_allocated_bytes'] for b in blocks if b['mode']=='parent');on_peak=max(b['peak_allocated_bytes'] for b in blocks if b['mode']=='on')
    speed=float(np.median(candidate)/np.median(base)-1);memory=on_peak/base_peak-1
    performance.append(dict(image_id=iid,original_median_ms=float(np.median(base)),candidate_median_ms=float(np.median(candidate)),time_increase=speed,original_peak_bytes=base_peak,candidate_peak_bytes=on_peak,memory_increase=memory,passed=speed<=.10 and memory<=.15,blocks=blocks))
    print('PERFORMANCE',iid,speed,memory,flush=True)
   head.direct_c3.cuda();head.direct_c4.cuda();mode(head,'on')
   data=model.data_preprocessor(pseudo_collate([ds.prepare_data(indices[2])]),training=False);before=cpu_pred(prediction(model,data));del data
  unchanged(model,state);assert all(p.grad is None for p in model.parameters())
  cfg.model.panoptic_head.direct_query_features=True;cfg.dump(str(OUT/'acceptance-model-config.py'))
  checkpoint=OUT/'untrained-acceptance-state.pth';torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'meta':{'training_updates':0,'direct_query_features':True}},checkpoint)
  del head,model;gc.collect();torch.cuda.empty_cache()
  restored=MODELS.build(cfg.model);restored.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True);restored.cuda().eval()
  with torch.inference_mode():
   data=restored.data_preprocessor(pseudo_collate([ds.prepare_data(indices[2])]),training=False);after=cpu_pred(prediction(restored,data))
  assert all(torch.equal(before[k],after[k]) for k in before);unchanged(restored,state)
  assert sha(ck)=='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
  result=dict(completed=True,passed=all(x['passed'] for x in performance),functional_passed=True,performance_passed=all(x['passed'] for x in performance),training_updates=0,optimizer_created=False,model_state_unchanged=True,new_parameters=393216,default_off=True,zero_after_init_weights=True,synthetic_routing_passed=True,output_parity=parity,gradients=gradients,performance=performance,save_reload_exact=True,checkpoint_sha256=sha(checkpoint),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/direct_query_feature_head.py']})
  (OUT/'result.json').write_text(json.dumps(result,indent=2));print('ACCEPTANCE_COMPLETE',result['passed'],flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  OUT.mkdir(exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc());raise
