"""Fixed-case inference only. Verify stock postprocess and decompose final scores."""
import argparse
import fcntl
import hashlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules
from pycocotools.coco import COCO
from pycocotools import mask as mu
import scripts.bll_mask2former_head
import scripts.pixel_bll_head

CASES = [(253,1098),(673,3724),(689,None),(652,None),(636,3557),(382,1845)]
MODELS_INFO = {
    'E0': ('uiis-baseline-36e-evaluation','uiis_baseline_r50_36e','53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'),
    'E1': ('uiis-e1-bll-36e-evaluation','uiis_e1_bll_r50_36e','6d89f04575ec2a28b441c3f681e38627cf2273638de3ee6394bedb03279e8176'),
    'E1-P': ('uiis-e1p-pixel-bll-36e-evaluation','uiis_e1p_pixel_bll_r50_36e','ad1a2cb9c886045aed6902716b3e73f866464a02b0debcaac3fe328fa4815d2e')}


def sha(p):
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def avg(x): return float(x.mean()) if x.numel() else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS_INFO,required=True)
    args=parser.parse_args();label=args.model
    out=ROOT/'reports/e1p-score-components-20261031'/'matched'/label;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'result.json').exists(), 'Preserve completed result'
    with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        register_all_modules();torch.set_num_threads(2);torch.manual_seed(20261031)
        folder,work,digest=MODELS_INFO[label];ck=ROOT/'work_dirs'/work/'epoch_36.pth'
        assert sha(ck)==digest
        cfg=Config.fromfile(str(ROOT/'reports'/folder/'evaluation-config.py'))
        baseline=Config.fromfile(str(ROOT/'reports/uiis-baseline-36e-evaluation/evaluation-config.py'))
        assert cfg.test_dataloader==baseline.test_dataloader
        assert cfg.model.test_cfg==baseline.model.test_cfg
        assert cfg.model.data_preprocessor==baseline.model.data_preprocessor
        dataset=DATASETS.build(cfg.test_dataloader.dataset);dataset.full_init()
        model=MODELS.build(cfg.model).cuda().eval()
        state=torch.load(ck,map_location='cpu')['state_dict'];model.load_state_dict(state,strict=True)
        assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items())
        predpath=ROOT/'reports'/folder/'predictions.segm.json';predhash=sha(predpath)
        preds=json.loads(predpath.read_text());gt=COCO(str(ROOT/'data/UIIS/annotations/val.json'))
        fusion=model.panoptic_fusion_head
        (out/'postprocess-source.txt').write_text(inspect.getsource(type(fusion).predict)+'\n'+inspect.getsource(type(fusion).instance_postprocess))
        result=dict(model=label,checkpoint_sha256=digest,predictions_sha256=predhash,
                    optimizer_created=False,training=False,config_matches_baseline=True,cases=[])
        with torch.inference_mode():
            for iid,gid in CASES:
                index=next(i for i in range(len(dataset)) if dataset.get_data_info(i)['img_id']==iid)
                item=dataset.prepare_data(index);data=model.data_preprocessor(pseudo_collate([item]),training=False)
                feats=model.extract_feat(data['inputs'])
                cls,padded=model.panoptic_head.predict(feats,data['data_samples'])
                actual=fusion.predict(cls,padded,data['data_samples'],rescale=True)[0]['ins_results']
                meta=data['data_samples'][0].metainfo;h,w=meta['img_shape'][:2];oh,ow=meta['ori_shape'][:2]
                # Independently reproduce padding removal, original-resolution logits and top-k.
                cropped=padded[0,:,:h,:w]
                logits=F.interpolate(cropped[:,None],size=(oh,ow),mode='bilinear',align_corners=False)[:,0]
                probs=cls[0].softmax(-1)[:,:-1]
                class_score,top=probs.flatten().topk(fusion.test_cfg.get('max_per_image',100),sorted=False)
                cats=top%fusion.num_classes;queries=top//fusion.num_classes
                keep=cats<fusion.num_things_classes
                class_score,cats,queries=class_score[keep],cats[keep],queries[keep]
                picked=logits[queries];binary=picked>0
                mask_score=(picked.sigmoid()*binary).flatten(1).sum(1)/(binary.flatten(1).sum(1)+1e-6)
                calculated=class_score*mask_score
                assert torch.equal(actual.labels,cats) and torch.equal(actual.masks,binary)
                assert torch.equal(actual.scores,calculated)
                saved=[p for p in preds if p['image_id']==iid]
                assert len(saved)==len(actual.scores)
                assert [p['category_id'] for p in saved]==(cats+1).tolist()
                error=max(abs(p['score']-float(actual.scores[j])) for j,p in enumerate(saved))
                xor=[int(np.count_nonzero(mu.decode(p['segmentation']).astype(bool)!=binary[j].cpu().numpy())) for j,p in enumerate(saved)]
                # Report exact agreement separately; do not hide numerical differences.
                reproduction=dict(score_max_abs_error=error,mask_changed_pixels=sum(xor),
                                  masks_exact=sum(x==0 for x in xor),predictions=len(saved))
                assert error<1e-4, reproduction
                if gid:
                    gm=torch.as_tensor(gt.annToMask(gt.anns[gid]),device='cuda',dtype=torch.bool)
                    cid=gt.anns[gid]['category_id']
                    eligible=[j for j,p in enumerate(saved) if p['category_id']==cid]
                    saved_gt=gt.annToRLE(gt.anns[gid])
                    overlaps=mu.iou([saved[j]['segmentation'] for j in eligible],[saved_gt],[0])[:,0]
                    # Fixed same-class mask with highest GT IoU, not selecting an improved run.
                    paired=json.loads((ROOT/'reports/e1p-final-comparison/instances.json').read_text())
                    record=next(r for r in paired if r['gt_id']==gid)[label]
                    assert record['matched']
                    sorted_ids=sorted(range(len(saved)),key=lambda n:-saved[n]['score'])
                    j=sorted_ids[record['pred_index']]
                    assert abs(saved[j]['score']-record['score'])<1e-10
                    assert saved[j]['category_id']==cid
                else:
                    j=max((j for j,p in enumerate(saved) if p['category_id']==6),key=lambda j:saved[j]['score'])
                    gm=None
                q=int(queries[j]);p=picked[j].sigmoid();pm=binary[j]
                row=dict(image_id=iid,gt_id=gid,file_name=gt.imgs[iid]['file_name'],query_index=q,
                         selection='frozen score0.5 matched prediction' if gid else 'highest saved robot score',
                         class_probability=float(class_score[j]),mask_probability=float(mask_score[j]),
                         final_score=float(calculated[j]),saved_score=saved[j]['score'],
                         class_probabilities=cls[0,q].softmax(-1).cpu().tolist(),
                         predicted_pixels=int(pm.sum()),reproduction=reproduction,
                         postprocess_formula_exact=True,original_shape=[oh,ow],input_shape=[h,w])
                if gm is not None:
                    inter=int((pm&gm).sum());union=int((pm|gm).sum())
                    row.update(mask_iou=inter/union,gt_pixels=int(gm.sum()),
                               under_pixels=int((gm&~pm).sum()),over_pixels=int((~gm&pm).sum()),
                               mean_gt_probability=avg(p[gm]),mean_missed_probability=avg(p[gm&~pm]),
                               missed_confident_background=int((gm&(p<.1)).sum()),
                               false_positive_confident_foreground=int((~gm&(p>.9)).sum()))
                    # Check if geometry failure is already present before final image rescale.
                    low=cropped[q]>0
                    gsmall=F.interpolate(gm[None,None].float(),size=(h,w),mode='nearest')[0,0].bool()
                    row['pre_final_rescale_mask_iou']=float((low&gsmall).sum()/(low|gsmall).sum())
                np.savez_compressed(out/f'image-{iid}.npz',selected_logits=picked[j].cpu().numpy(),
                                    class_logits=cls[0,q].cpu().numpy(),query_index=np.array(q))
                result['cases'].append(row);print(label,iid,'CASE_CHECKED',flush=True)
        assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        assert sha(ck)==digest and sha(predpath)==predhash
        result.update(passed=True,model_state_unchanged=True,checkpoint_unchanged=True,
                      prediction_files_unchanged=True,source_sha256=sha(Path(__file__)))
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print(label,'SCORE_COMPONENTS_PASSED',flush=True)


if __name__=='__main__': main()
