"""Trace fixed train augmentations and prove parity with original pipeline. CPU only."""
import copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules
ROOT=Path(__file__).resolve().parents[1]
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(d):return {int(i):int(a) for i,a in zip(d['gt_bboxes_labels'],d['gt_masks'].areas)}
def main():
    out=ROOT/'reports/baseline-augmentation-audit-20261031';assert not (out/'result.json').exists()
    sel=json.loads((out/'selection.json').read_text());register_all_modules();torch.set_num_threads(2)
    cfg=Config.fromfile(str(ROOT/'work_dirs/uiis_baseline_r50_36e/config-resume.py'))
    ds=DATASETS.build(cfg.train_dataloader.dataset);ds.full_init();indices={ds.get_data_info(i)['img_id']:i for i in range(len(ds))}
    ann=ROOT/'data/UIIS/annotations/train.json';digest=sha(ann);records=[];runs=[]
    for iid in sel['image_ids']:
        idx=indices[iid]
        for offset in sel['seed_offsets']:
            n=sel['seed_base']+iid+offset;seed(n);d=copy.deepcopy(ds.get_data_info(idx));stages={};classes=None;ignore=None;meta={}
            for transform in ds.pipeline.transforms:
                name=type(transform).__name__
                if name=='PackDetInputs':
                    stages['final']=snapshot(d)
                    d['gt_bboxes_labels']=classes[d['gt_bboxes_labels']]
                d=transform(d)
                if name=='LoadAnnotations':
                    classes=d['gt_bboxes_labels'].copy();ignore=d['gt_ignore_flags'].copy()
                    d['gt_bboxes_labels']=np.arange(len(classes),dtype=np.int64);stages['source']=snapshot(d)
                if name in ['RandomResize','RandomCrop','FilterAnnotations']:
                    stages[name]={} if d is None else snapshot(d)
                    if d is not None:meta[name]={k:list(d[k]) for k in ['img_shape','scale_factor'] if k in d}
                if d is None:break
            stages.setdefault('final',{})
            seed(n);reference=ds.prepare_data(idx);assert (reference is None)==(d is None)
            if d is not None:
                assert torch.equal(reference['inputs'],d['inputs'])
                for field in ['gt_instances','ignored_instances']:
                    a=getattr(reference['data_samples'],field);b=getattr(d['data_samples'],field)
                    assert torch.equal(a.labels,b.labels) and np.array_equal(a.masks.masks,b.masks.masks)
            assert classes is not None
            for j,area in stages['source'].items():
                if ignore[j]:continue
                records.append(dict(image_id=iid,seed=n,instance_index=j,category_id=int(classes[j])+1,
                    source_area=area,resized_area=stages.get('RandomResize',{}).get(j,0),
                    cropped_area=stages.get('RandomCrop',{}).get(j,0),final_area=stages['final'].get(j,0),
                    retained=j in stages['final']))
            runs.append(dict(image_id=iid,seed=n,parity_exact=True,pipeline_filtered=d is None,transforms=meta))
        if len(runs)%15==0:print('AUGMENTATIONS_CHECKED',len(runs),flush=True)
        (out/'progress.json').write_text(json.dumps(dict(runs=len(runs),instances=len(records))))
    assert sha(ann)==digest
    result=dict(passed=True,images=len(sel['image_ids']),augmentations=len(runs),records=records,runs=runs,
                annotation_sha256=digest,source_sha256=sha(Path(__file__)),model_created=False,optimizer_created=False,pipeline=cfg.train_dataloader.dataset.pipeline)
    (out/'result.json').write_text(json.dumps(result,indent=2));print('AUGMENTATION_AUDIT_PASSED',len(runs),flush=True)
if __name__=='__main__':main()
