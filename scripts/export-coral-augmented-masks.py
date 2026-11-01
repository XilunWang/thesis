"""Export five frozen augmentations on CPU using data pipeline only."""
import json,hashlib,random,traceback
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS,MODELS
from mmdet.utils import register_all_modules
from pycocotools import mask as mu
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/coral-augmented-export-20261101'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False);result=dict(passed=False,model_built=False,optimizer_built=False,training_updates=0,device='cpu',images=[])
 try:
  register_all_modules();torch.set_num_threads(2)
  cfgp=ROOT/'work_dirs/uiis_baseline_r50_36e/config-resume.py';cfg=Config.fromfile(str(cfgp))
  selp=ROOT/'reports/coral-neighbor-coverage-20261101/selection.json';sel=json.loads(selp.read_text());ids=[x['image_id'] for x in sel['images']];assert ids==[12,309,326,337,381,414]
  refp=ROOT/'reports/uiis-pixel-bll-data-20261031/images.jsonl';refs={r['id']:r for r in map(json.loads,refp.read_text().splitlines())}
  ann=ROOT/'data/UIIS/annotations/train.json';assert sha(ann)=='9cc586e559f54533a938e34a89017d47784c3646bc43ab2cd0d88e140ef5b49d'
  ds=DATASETS.build(cfg.train_dataloader.dataset);ds.full_init();lookup={ds.get_data_info(i)['img_id']:i for i in range(len(ds))};prep=MODELS.build(cfg.model.data_preprocessor).cpu()
  for chosen in sel['images']:
   iid=chosen['image_id']
   if iid==12:continue
   ref=refs[iid];assert chosen['seed']==ref['seed']==20261012+iid
   info=ds.get_data_info(lookup[iid]);assert sha(info['img_path'])==ref['image_sha256']
   random.seed(ref['seed']);np.random.seed(ref['seed']);torch.manual_seed(ref['seed'])
   item=ds.prepare_data(lookup[iid]);assert item is not None
   with torch.no_grad():data=prep(pseudo_collate([item]),training=True)
   sample=data['data_samples'][0];gt=sample.gt_instances;masks=gt.masks.to_ndarray().astype('uint8');labels=gt.labels.tolist();areas=masks.sum((1,2)).tolist();valid=list(sample.img_shape[:2])
   assert labels==ref['labels'] and areas==ref['gt_pixels'] and valid==ref['valid_shape'] and list(masks.shape[-2:])==ref['padded_shape']
   rles=[]
   for mask in masks:
    rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode('ascii');rles.append(rle)
   record=dict(image_id=iid,seed=ref['seed'],labels=labels,gt_pixels=areas,shape=list(masks.shape[-2:]),valid_shape=valid,gt_rles=rles,reference_exact=True,image_sha256=ref['image_sha256'])
   p=OUT/f'image-{iid}.json';p.write_text(json.dumps(record));result['images'].append(dict(image_id=iid,sha256=sha(p),instances=len(labels)));print(iid,'EXACT_AUGMENTATION_EXPORTED',flush=True)
  result.update(passed=True,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ann,refp,cfgp,selp]},source_sha256=sha(__file__))
 except BaseException:result['error']=traceback.format_exc();raise
 finally:(OUT/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
