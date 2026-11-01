"""CPU-only identity/parity checks on a frozen six-image selection."""
import copy,hashlib,json,random,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS,MODELS
from mmdet.utils import register_all_modules
from pycocotools import mask as mu
from scripts.identity_data_pipeline import with_identity,IdentityFilterAnnotations,IdentityPackDetInputs
OUT=ROOT/'reports/identity-augmentation-check-20261101'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def arrayhash(x):
 if torch.is_tensor(x):x=x.detach().cpu().numpy()
 a=np.ascontiguousarray(x);return hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()
def run(ds,index,n,reference,source):
 seed(n);data=copy.deepcopy(ds.get_data_info(index));trace=[];classes=None;original_ids=np.array([x['annotation_id'] for x in source['instances']],dtype=np.int64)
 for transform in ds.pipeline.transforms:
  name=type(transform).__name__.replace('Identity','')
  if name=='PackDetInputs':
   if reference:data['gt_bboxes_labels']=classes[data['gt_bboxes_labels']]
   data=transform(data);break
  data=transform(data)
  if data is None:trace.append(dict(stage=name,filtered=True));break
  if name=='LoadAnnotations' and reference:
   classes=data['gt_bboxes_labels'].copy();data['gt_bboxes_labels']=np.arange(len(classes),dtype=np.int64)
  if 'gt_masks' in data:
   ids=original_ids[data['gt_bboxes_labels']] if reference else data['gt_instances_ids']
   labels=classes[data['gt_bboxes_labels']] if reference else data['gt_bboxes_labels']
   trace.append(dict(stage=name,ids=ids.tolist(),labels=labels.tolist(),areas=data['gt_masks'].areas.tolist(),image=arrayhash(data['img']),masks=arrayhash(data['gt_masks'].masks),boxes=arrayhash(data['gt_bboxes'].tensor),ignore=data['gt_ignore_flags'].tolist()))
 return data,trace
def synthetic():
 from mmdet.structures.bbox import HorizontalBoxes
 from mmdet.structures.mask import BitmapMasks
 masks=np.zeros((3,8,8),dtype=np.uint8);masks[0,1:3,1:3]=1;masks[2,4:6,4:6]=1
 d=dict(gt_bboxes=HorizontalBoxes(torch.tensor([[1.,1.,3.,3.],[2.,2.,4.,4.],[4.,4.,6.,6.]])),gt_bboxes_labels=np.array([0,0,1]),gt_ignore_flags=np.array([False,False,True]),gt_masks=BitmapMasks(masks,8,8),gt_instances_ids=np.array([100,101,102]),original_area_by_id={100:4.25,101:7.5,102:8.75})
 filtered=IdentityFilterAnnotations(by_mask=True)(d);assert filtered['gt_instances_ids'].tolist()==[100,102]
 packed=IdentityPackDetInputs(meta_keys=())(filtered)['data_samples'];assert packed.gt_instances.annotation_ids.tolist()==[100] and packed.ignored_instances.annotation_ids.tolist()==[102]
 assert packed.gt_instances.original_areas.tolist()==[4.25] and packed.ignored_instances.original_areas.tolist()==[8.75]
 return True
def main():
 assert not (OUT/'result.json').exists();result=dict(passed=False,model_built=False,optimizer_built=False,training_updates=0,runs=[])
 try:
  register_all_modules();torch.set_num_threads(2);result['synthetic_filter_and_ignore_split']=synthetic()
  selection=json.loads((OUT/'selection.json').read_text());ann=ROOT/'data/UIIS/annotations/train.json';assert sha(ann)==selection['annotation_sha256']
  raw={a['id']:a for a in json.loads(ann.read_text())['annotations']}
  cfgp=ROOT/'work_dirs/uiis_baseline_r50_36e/config-resume.py';cfg=Config.fromfile(str(cfgp))
  base=DATASETS.build(cfg.train_dataloader.dataset);new=DATASETS.build(with_identity(cfg.train_dataloader.dataset));base.full_init();new.full_init()
  lookup={base.get_data_info(i)['img_id']:i for i in range(len(base))};assert len(base)==len(new)==3937
  prep=MODELS.build(cfg.model.data_preprocessor).cpu()
  for iid in selection['image_ids']:
   index=lookup[iid];source=new.get_data_info(index);assert source['img_id']==iid
   for offset in selection['seed_offsets']:
    n=selection['seed_base']+iid+offset
    a,ta=run(base,index,n,True,source);b,tb=run(new,index,n,False,source)
    assert ta==tb,(iid,n,'stage mismatch');assert (a is None)==(b is None)
    row=dict(image_id=iid,seed=n,stages=tb,pipeline_filtered=b is None,parity_exact=True)
    if b is not None:
     seed(n);ordinary=base.prepare_data(index)
     assert torch.equal(ordinary['inputs'],a['inputs']) and torch.equal(a['inputs'],b['inputs'])
     for field in ['gt_instances','ignored_instances']:
      ref=getattr(ordinary['data_samples'],field);aug=getattr(b['data_samples'],field)
      assert torch.equal(ref.labels,aug.labels) and torch.equal(ref.bboxes.tensor,aug.bboxes.tensor) and np.array_equal(ref.masks.masks,aug.masks.masks)
      for gid,area,label in zip(aug.annotation_ids.tolist(),aug.original_areas.tolist(),aug.labels.tolist()):
       assert raw[gid]['image_id']==iid and float(raw[gid]['area'])==area and new.cat2label[raw[gid]['category_id']]==label
     with torch.no_grad():
      pa=prep(pseudo_collate([ordinary]),training=True);pb=prep(pseudo_collate([b]),training=True)
     assert torch.equal(pa['inputs'],pb['inputs'])
     oldgt=pa['data_samples'][0].gt_instances;g=pb['data_samples'][0].gt_instances
     assert np.array_equal(oldgt.masks.masks,g.masks.masks) and torch.equal(oldgt.labels,g.labels)
     row.update(annotation_ids=g.annotation_ids.tolist(),original_areas=g.original_areas.tolist(),augmented_areas=g.masks.areas.tolist(),labels=g.labels.tolist(),valid_shape=list(pb['data_samples'][0].img_shape),padded_shape=list(g.masks.masks.shape[-2:]))
     masks=[]
     for m in g.masks.masks:
      r=mu.encode(np.asfortranarray(m.astype(np.uint8)));r['counts']=r['counts'].decode();masks.append(r)
     row['gt_rles']=masks
     row['original_small_retained']=sum(0<x<=1024 for x in row['original_areas'])
     row['original_small_now_large']=sum(0<x<=1024 and y>1024 for x,y in zip(row['original_areas'],row['augmented_areas']))
     row['original_large_now_small']=sum(x>1024 and 0<y<=1024 for x,y in zip(row['original_areas'],row['augmented_areas']))
    p=OUT/f'image-{iid}-seed-{n}.json';p.write_text(json.dumps(row,indent=2));result['runs'].append({k:v for k,v in row.items() if k not in ['stages','gt_rles']})
    print(iid,n,'FILTERED' if b is None else len(row['annotation_ids']),'EXACT',flush=True)
  result.update(passed=True,inputs={str(p.relative_to(ROOT)):sha(p) for p in [ann,cfgp,OUT/'selection.json',ROOT/'scripts/identity_data_pipeline.py',Path(__file__)]},scope='Six fixed train images, two fixed seeds; data pipeline/preprocessor CPU only; no model weights loaded.')
 except BaseException:result['error']=traceback.format_exc();raise
 finally:(OUT/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
