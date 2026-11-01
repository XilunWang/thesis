"""Opt-in identity metadata for the existing MMDetection 3.3 UIIS pipeline."""
import copy
import numpy as np
from mmdet.datasets import CocoDataset
from mmdet.datasets.transforms import LoadAnnotations,FilterAnnotations,PackDetInputs
from mmdet.registry import DATASETS,TRANSFORMS

@DATASETS.register_module()
class IdentityCocoDataset(CocoDataset):
 def parse_data_info(self,raw):
  info=super().parse_data_info(raw);expanded=[]
  # Let the installed original parser decide inclusion, including ignore/crowd.
  for annotation in raw['raw_ann_info']:
   one=dict(raw,raw_ann_info=[annotation])
   parsed=super().parse_data_info(one)['instances']
   for instance in parsed:
    instance=dict(instance,annotation_id=int(annotation['id']),original_area=float(annotation['area']))
    expanded.append(instance)
  assert [{k:v for k,v in x.items() if k not in ('annotation_id','original_area')} for x in expanded]==info['instances']
  info['instances']=expanded;return info

@TRANSFORMS.register_module()
class IdentityLoadAnnotations(LoadAnnotations):
 def transform(self,results):
  results=super().transform(results)
  ids=np.asarray([r['annotation_id'] for r in results['instances']],dtype=np.int64)
  assert len(ids)==len(results['gt_bboxes_labels']) and len(set(ids.tolist()))==len(ids)
  results['gt_instances_ids']=ids
  results['original_area_by_id']={r['annotation_id']:r['original_area'] for r in results['instances']}
  return results

@TRANSFORMS.register_module()
class IdentityFilterAnnotations(FilterAnnotations):
 def transform(self,results):
  ids=results['gt_instances_ids'].copy();boxes=results['gt_bboxes']
  assert len(ids)==len(boxes)
  keep=np.ones(len(ids),dtype=bool)
  if self.by_box:
   keep&=((boxes.widths>self.min_gt_bbox_wh[0])&(boxes.heights>self.min_gt_bbox_wh[1])).numpy()
  if self.by_mask:keep&=results['gt_masks'].areas>=self.min_gt_mask_area
  output=super().transform(results)
  if output is not None:
   output['gt_instances_ids']=ids[keep]
   assert len(output['gt_instances_ids'])==len(output['gt_bboxes_labels'])
  return output

@TRANSFORMS.register_module()
class IdentityPackDetInputs(PackDetInputs):
 mapping_table={**PackDetInputs.mapping_table,'gt_instances_ids':'annotation_ids','gt_original_areas':'original_areas'}
 def transform(self,results):
  ids=results['gt_instances_ids'];assert len(ids)==len(results['gt_bboxes_labels'])
  results['gt_original_areas']=np.asarray([results['original_area_by_id'][int(i)] for i in ids],dtype=np.float64)
  return super().transform(results)

def with_identity(dataset_config):
 cfg=copy.deepcopy(dataset_config)
 assert cfg.type=='CocoDataset','Only the audited plain CocoDataset is supported'
 expected=['LoadImageFromFile','LoadAnnotations','RandomFlip','RandomResize','RandomCrop','FilterAnnotations','PackDetInputs']
 assert [x['type'] for x in cfg.pipeline]==expected,'Unreviewed transformation sequence'
 cfg.type='IdentityCocoDataset'
 names={'LoadAnnotations':'IdentityLoadAnnotations','FilterAnnotations':'IdentityFilterAnnotations','PackDetInputs':'IdentityPackDetInputs'}
 for stage in cfg.pipeline:stage['type']=names.get(stage['type'],stage['type'])
 return cfg
