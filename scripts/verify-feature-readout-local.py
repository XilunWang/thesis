"""Local replay of learned-feature readouts and cached baseline identities."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'));sys.path.insert(1,str(ROOT/'.local/bll-libs'))
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 src=ROOT/'reports/feature-readout-export-20261101';out=ROOT/'reports/feature-readout-probe-20261101'
 result=json.loads((out/'result.json').read_text());export=json.loads((src/'result.json').read_text());assert result['completed'] and export['passed']
 assert sha(src/'result.json')==result['export_result_sha256'];assert sha(ROOT/'scripts/probe-frozen-feature-readout.py')==result['source_sha256'];assert sha(ROOT/'scripts/export-baseline-readout-features.py')==export['source_sha256']
 old={x['gt_id']:x for x in json.loads((ROOT/'reports/baseline-decoder-audit-20261031/instances.json').read_text())}
 coco=COCO(str(ROOT/'data/UIIS/annotations/val.json'));checks=[]
 for iid in [2,19,42]:
  record=json.loads((src/f'image-{iid}.json').read_text());p=src/f'features-{iid}.npz';assert sha(p)==record['feature_sha256'];meta=record['metainfo'];bh,bw=meta['batch_input_shape'];h,w=meta['img_shape'];oh,ow=meta['ori_shape']
  with np.load(p,allow_pickle=False) as z, np.load(out/f'vectors-{iid}.npz',allow_pickle=False) as vectors:
   union=np.logical_or.reduce([coco.annToMask(a).astype(bool) for a in coco.loadAnns(coco.getAnnIds(imgIds=[iid]))])
   for source in ['mask_features','C2','C3','C4','C5']:
    rows=[r for r in result['rows'] if r['image_id']==iid and r['source']==source]
    coeff=torch.from_numpy(np.stack([vectors[f"{source}_gt{r['gt_id']}"] for r in rows]));feat=torch.from_numpy(z[source])
    logits=torch.einsum('kc,bchw->bkhw',coeff,feat)
    logits=F.interpolate(logits,size=(bh,bw),mode='bilinear',align_corners=False)[:,:,:h,:w]
    masks=F.interpolate(logits,size=(oh,ow),mode='bilinear',align_corners=False)[0].numpy()>0
    for row,pred in zip(rows,masks):
     gid=row['gt_id'];gt=coco.annToMask(coco.anns[gid]).astype(bool);tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum());neighbor=int((pred&union&~gt).sum())
     assert (tp,fp,fn,neighbor)==tuple(row['metrics'][k] for k in ['tp','fp','fn','neighbor_fp']), (gid,source)
     assert row['final_best_iou']==old[gid]['stages'][-1]['best_iou']
     assert row['all_stage_best_iou']==max(s['best_iou'] for s in old[gid]['stages'])
     checks.append({'gt_id':gid,'source':source,'reproduced':True})
 verified=dict(passed=True,readouts_checked=len(checks),feature_sha256={str(i):sha(src/f'features-{i}.npz') for i in [2,19,42]},checks=checks,result_sha256=sha(out/'result.json'))
 (out/'local-verification.json').write_text(json.dumps(verified,indent=2));print('LOCAL_REPLAY_PASSED',len(checks))
if __name__=='__main__': main()
