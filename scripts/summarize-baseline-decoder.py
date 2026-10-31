"""Verify and summarize stage-wise candidates without GT-selected deployment output."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.local/analysis-libs'))
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu

def main():
 out=ROOT/'reports/baseline-decoder-audit-20261031';result=json.loads((out/'result.json').read_text());assert result['passed'];sel=json.loads((out/'selection.json').read_text())
 for line in (out/'remote-hashes.txt').read_text(encoding='utf-8-sig').splitlines():
  parts=line.split()
  if len(parts)==2 and len(parts[0])==64:assert hashlib.sha256((ROOT/parts[1]).read_bytes()).hexdigest()==parts[0]
 gt=COCO(str(ROOT/'data/UIIS/annotations/val.json'));prior={r['gt_id']:r for r in json.loads((ROOT/'reports/baseline-query-audit-20261031/instances.json').read_text())};rows=[]
 for iid in sel['image_ids']:
  data=json.loads((out/f'image-{iid}.json').read_text());gids=data['gt_ids'];anns=[gt.anns[g] for g in gids]
  for layer in data['layers']:
   ious=mu.iou(layer['rles'],[gt.annToRLE(a) for a in anns],[0]*len(anns));assert np.allclose(ious,layer['ious'],rtol=0,atol=1e-12)
  for j,gid in enumerate(gids):
   a=gt.anns[gid];c=a['category_id']-1;group=next(r['group'] for r in sel['cases'] if r['gt_id']==gid);stages=[]
   for layer in data['layers']:
    io=np.array(layer['ious'])[:,j];probs=np.array(layer['class_probabilities']);quality=np.array(layer['mask_quality']);chosen=set(layer['selected_query_class_flat_indices']);q=int(io.argmax());good=io>=.5
    goodsel=[k for k in range(100) if good[k] and k*7+c in chosen];eligible=[k for k in goodsel if probs[k,c]*quality[k]>=.5]
    goodscore=max((probs[k,c]*quality[k] for k in range(100) if good[k]),default=None)
    stages.append(dict(stage=layer['stage'],best_query=q,best_iou=float(io[q]),best_geometry_class_probability=float(probs[q,c]),best_geometry_background_probability=float(probs[q,7]),good_correct_selected=len(goodsel),eligible_above05=len(eligible),best_good_correct_score=goodscore))
   assert abs(stages[-1]['best_iou']-prior[gid]['best_raw_iou'])<1e-12
   bestearly=max(stages[:-1],key=lambda x:x['best_iou']);q=bestearly['best_query'];g=gt.annToMask(a).astype(bool);ga=int(g.sum());track=[]
   others=[(b['id'],gt.annToMask(b).astype(bool)) for b in gt.imgToAnns[iid] if b['id']!=gid and not b.get('iscrowd',0)]
   for layer in data['layers']:
    pm=mu.decode(layer['rles'][q]).astype(bool);inter=int((pm&g).sum());pa=int(pm.sum());fp=pa-inter;fn=ga-inter
    neighbors=[bid for bid,bm in others if bm.sum() and (pm&bm).sum()/bm.sum()>=.5]
    track.append(dict(stage=layer['stage'],query=q,iou=float(layer['ious'][q][j]),coverage=inter/ga,purity=inter/pa if pa else 0,pred_pixels=pa,fp=fp,fn=fn,neighbors=neighbors,correct_probability=layer['class_probabilities'][q][c],background_probability=layer['class_probabilities'][q][7]))
   rows.append(dict(gt_id=gid,image_id=iid,category_id=a['category_id'],group=group,stages=stages,best_early_stage=bestearly['stage'],best_early_query=q,best_early_iou=bestearly['best_iou'],early_query_track=track))
 groups={}
 for name in ['failure','low_score_control']:
  rs=[r for r in rows if r['group']==name]
  groups[name]=dict(count=len(rs),ever_early_geometry05=sum(r['best_early_iou']>=.5 for r in rs),ever_early_geometry075=sum(r['best_early_iou']>=.75 for r in rs),ever_updated_early_geometry05=sum(max(s['best_iou'] for s in r['stages'][1:9])>=.5 for r in rs),ever_early_eligible05=sum(any(s['eligible_above05'] for s in r['stages'][:-1]) for r in rs),
   per_stage=[dict(stage=s,geometry05=sum(r['stages'][s]['best_iou']>=.5 for r in rs),geometry075=sum(r['stages'][s]['best_iou']>=.75 for r in rs),eligible05=sum(bool(r['stages'][s]['eligible_above05']) for r in rs)) for s in range(10)])
 lost=[r for r in rows if r['group']=='failure' and r['best_early_iou']>=.5]
 tracks=dict(count=len(lost),same_query_zero_final_overlap=sum(r['early_query_track'][-1]['coverage']==0 for r in lost),same_query_nonzero_expanded=sum(r['early_query_track'][-1]['coverage']>0 and r['early_query_track'][-1]['pred_pixels']>r['early_query_track'][r['best_early_stage']]['pred_pixels'] for r in lost),same_query_added_other_gt=sum(bool(set(r['early_query_track'][-1]['neighbors'])-set(r['early_query_track'][r['best_early_stage']]['neighbors'])) for r in lost))
 assert hashlib.sha256((ROOT/'scripts/audit-baseline-decoder.py').read_bytes()).hexdigest()==result['source_sha256']
 for path,digest in result['input_hashes'].items():
  local=ROOT/path
  if local.exists():assert hashlib.sha256(local.read_bytes()).hexdigest()==digest
 summary=dict(lost_early_candidate_tracks=tracks,passed=True,images=len(sel['image_ids']),cases=len(rows),groups=groups,final_prediction_max_score_error=max(c['max_score_error'] for c in result['checks']),download_hashes_verified=True,all_stage_rle_ious_verified=True)
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'instances.json').write_text(json.dumps(rows,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
