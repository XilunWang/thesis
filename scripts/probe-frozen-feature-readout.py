"""CPU-only, fixed GT-assisted ridge readout. Never loads or updates the model."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import solve
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mu
ROOT=Path(__file__).resolve().parents[1]
SEED=20261101

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def restore(x,meta):
 bh,bw=meta['batch_input_shape']; h,w=meta['img_shape']; oh,ow=meta['ori_shape']
 return F.interpolate(F.interpolate(x,size=(bh,bw),mode='bilinear',align_corners=False)[:,:,:h,:w],size=(oh,ow),mode='bilinear',align_corners=False)

def metrics(pred,gt,neighbor,selected):
 tp=int((pred&gt).sum()); fp=int((pred&~gt).sum()); fn=int((~pred&gt).sum())
 unsampled=(~gt).copy().reshape(-1);unsampled[selected]=False; unsampled=unsampled.reshape(gt.shape)
 return dict(iou=tp/max(tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),tp=tp,fp=fp,fn=fn,neighbor_fp=int((pred&neighbor&~gt).sum()),unselected_negative_fp=int((pred&unsampled).sum()),unselected_negative_pixels=int(unsampled.sum()))

def sampling(gt,neighbor,ann):
 h,w=gt.shape; x,y,bw,bh=ann['bbox']; near=np.zeros_like(gt)
 near[max(0,int(np.floor(y))-32):min(h,int(np.ceil(y+bh))+32),max(0,int(np.floor(x))-32):min(w,int(np.ceil(x+bw))+32)]=True
 pools=[np.flatnonzero(neighbor&~gt),np.flatnonzero(near&~neighbor&~gt),np.flatnonzero(~near&~neighbor&~gt)]
 quota=[0,0,0];total=min(4096,sum(map(len,pools)))
 for _ in range(total):
  candidates=[j for j in range(3) if quota[j]<len(pools[j])]
  j=min(candidates,key=lambda j:(quota[j],j));quota[j]+=1
 rng=np.random.default_rng(np.random.SeedSequence([SEED,int(ann['id'])]))
 neg=np.concatenate([rng.choice(p,size=n,replace=False) for p,n in zip(pools,quota)])
 pos=np.flatnonzero(gt); assert len(pos)>0 and len(neg)>0
 ids=np.concatenate([pos,neg]); y=np.r_[np.ones(len(pos)),-np.ones(len(neg))]
 weights=np.r_[np.full(len(pos),.5/len(pos)),np.full(len(neg),.5/len(neg))]
 return ids,y,weights,neg,quota

def ridge(X,y,weight):
 a=X.astype(np.float64); gram=a.T@(a*weight[:,None]); rhs=a.T@(weight*y)
 gram.flat[::len(gram)+1]+=1e-3
 v=solve(gram,rhs,assume_a='pos',check_finite=True)
 residual=float(np.linalg.norm(gram@v-rhs)/max(np.linalg.norm(rhs),1e-12));assert residual<1e-7
 return v,residual

def selfcheck():
 X=np.eye(2); v,r=ridge(X,np.array([1.,-1.]),np.array([.5,.5])); assert np.allclose(v,[.5/.501,-.5/.501])
 # A linear readout commutes with both bilinear restores, including a partial padding crop.
 gen=torch.Generator().manual_seed(1); a=torch.randn(1,4,3,5,generator=gen); v=torch.randn(4,generator=gen);meta={'batch_input_shape':[12,20],'img_shape':[11,17],'ori_shape':[7,13]}
 lhs=restore(torch.einsum('c,bchw->bhw',v,a)[:,None],meta);rhs=torch.einsum('c,bchw->bhw',v,restore(a,meta))[:,None]
 assert torch.allclose(lhs,rhs,atol=1e-6,rtol=1e-5)
 return dict(ridge_analytic=True,linear_restore_max_error=float((lhs-rhs).abs().max()))

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 src=ROOT/'reports/feature-readout-export-20261101';out=ROOT/'reports/feature-readout-probe-20261101';out.mkdir(exist_ok=True)
 assert not (out/'result.json').exists();export=json.loads((src/'result.json').read_text());assert export['passed']
 sel=json.loads((src/'selection.json').read_text());assert sel['image_ids']==[2,19,42]
 coco=COCO(str(ROOT/'data/UIIS/annotations/val.json'));assert sha(ROOT/'data/UIIS/annotations/val.json')==export['input_hashes']['data/UIIS/annotations/val.json']
 checks=selfcheck(); rows=[];witnesses=[]
 for iid in sel['image_ids']:
  meta_record=json.loads((src/f'image-{iid}.json').read_text());meta=meta_record['metainfo'];fp=src/f'features-{iid}.npz';assert sha(fp)==meta_record['feature_sha256']
  cases=[x for x in sel['cases'] if x['image_id']==iid];allanns=coco.loadAnns(coco.getAnnIds(imgIds=[iid])); masks={a['id']:coco.annToMask(a).astype(bool) for a in allanns}
  union=np.logical_or.reduce(list(masks.values()));samples=[]
  for c in cases:
   gt=masks[c['gt_id']];neighbor=union&~gt; ids,y,weights,neg,quota=sampling(gt,neighbor,coco.anns[c['gt_id']]);samples.append((ids,y,weights,neg,quota,gt,neighbor))
  unique=np.unique(np.concatenate([s[0] for s in samples])); positions=[np.searchsorted(unique,s[0]) for s in samples]
  z=np.load(fp,allow_pickle=False);maskfeat=torch.from_numpy(z['mask_features']);embed=z['embeddings']
  # Known-query witness reproduces each selected target's best cached all-stage geometry.
  for c,sample in zip(cases,samples):
   gindex=meta_record['gt_ids'].index(c['gt_id']);best=max(((float(layer['ious'][q][gindex]),layer['stage'],q) for layer in meta_record['layers'] for q in range(100)))
   old_iou,st,q=best;v=torch.from_numpy(embed[st,0,q]);logit=torch.einsum('c,bchw->bhw',v,maskfeat)[:,None];p=restore(logit,meta)[0,0].numpy()>0
   cached=mu.decode(meta_record['layers'][st]['rles'][q]).astype(bool);assert np.array_equal(p,cached)
   measured=metrics(p,sample[5],sample[6],sample[3]);assert abs(measured['iou']-old_iou)<1e-10
   witnesses.append(dict(image_id=iid,gt_id=c['gt_id'],stage=st,query=q,iou=old_iou,changed_pixels=0))
  del maskfeat,embed
  vectors={}
  for source in ['mask_features','C2','C3','C4','C5','RGB','random']:
   assert time.monotonic()-start<1800,'30 minute probe budget exceeded'
   if source=='RGB':
    arr=np.asarray(Image.open(ROOT/'data/UIIS/val'/coco.imgs[iid]['file_name']).convert('RGB'),dtype=np.float32).transpose(2,0,1)[None]/255.
   elif source=='random':
    shape=meta_record['feature_shapes']['mask_features'];arr=np.random.default_rng(np.random.SeedSequence([SEED,iid])).standard_normal(shape).astype(np.float32)
   else:arr=z[source]
   feature=torch.from_numpy(arr);channels=arr.shape[1]
   if source=='RGB': valid=torch.ones((1,1,*arr.shape[-2:]))
   else:
    bh,bw=meta['batch_input_shape'];h,w=meta['img_shape'];valid=torch.zeros(1,1,bh,bw);valid[:,:,:h,:w]=1
    valid=F.interpolate(valid,size=arr.shape[-2:],mode='area')
   rms=torch.sqrt((feature.square()*valid).sum((0,2,3))/valid.sum()).numpy();rms=np.maximum(rms,1e-8)
   design=np.empty((len(unique),channels),dtype=np.float32)
   for ch in range(0,channels,32):
    chunk=feature[:,ch:ch+32]; rendered=chunk if source=='RGB' else restore(chunk,meta)
    design[:,ch:ch+32]=rendered[0].flatten(1)[:,unique].T.numpy()/rms[ch:ch+32]
   fitted=[]
   for c,s,p in zip(cases,samples,positions):
    v,res=ridge(design[p],s[1],s[2]);fitted.append((v/rms).astype(np.float32));vectors[f"{source}_gt{c['gt_id']}"]=fitted[-1]
   coefficients=torch.from_numpy(np.stack(fitted));native=torch.einsum('kc,bchw->bkhw',coefficients,feature)
   full=native if source=='RGB' else restore(native,meta)
   for j,(c,s) in enumerate(zip(cases,samples)):
    pred=full[0,j].numpy()>0;met=metrics(pred,s[5],s[6],s[3]);row=dict(c,source=source,metrics=met,negative_stratum_counts=s[4],positive_fit_pixels=int(s[5].sum()),success=met['iou']>=.5,strong_gain=met['iou']-c['final_best_iou']>=.1)
    rows.append(row);print(iid,c['gt_id'],source,round(met['iou'],4),flush=True)
   (out/'partial.json').write_text(json.dumps(dict(rows=rows,witnesses=witnesses),indent=2))
   del arr,feature,design,full,native
  np.savez(out/f'vectors-{iid}.npz',**vectors);z.close()
 summary={s:{'successes':sum(r['success'] for r in rows if r['source']==s),'mean_iou':float(np.mean([r['metrics']['iou'] for r in rows if r['source']==s]))} for s in sorted({r['source'] for r in rows})}
 calibration=[r for r in rows if r['source']=='mask_features' and r['group']!='never_good'];fail=[r['gt_id'] for r in calibration if not r['success']]
 result=dict(completed=True,model_training=False,gt_assisted_fit=True,model_weights_loaded=False,elapsed_seconds=time.monotonic()-start,checks=checks,witnesses=witnesses,summary=summary,calibration_failures=fail,negative_feature_claims_allowed=False,rows=rows,source_sha256=sha(Path(__file__)),export_result_sha256=sha(src/'result.json'))
 (out/'result.json').write_text(json.dumps(result,indent=2));print('PROBE_COMPLETED',summary,'calibration_failures',fail,flush=True)
if __name__=='__main__':main()
