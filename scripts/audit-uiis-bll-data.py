"""All official training images, one seeded augmentation each; no model or optimizer."""
import hashlib
import argparse
import json
import random
import runpy
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/uiis-bll-occupancy-audit'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def progress(*values):
    # SSH stdout can disappear while the audit continues; logging is not data.
    try:
        print(*values, flush=True)
    except BrokenPipeError:
        pass


def audit(result, image_ids=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scripts.query_bll import build_bll_targets, TARGET_RULE
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import MODELS,DATASETS
    from mmdet.utils import register_all_modules
    register_all_modules()
    torch.set_num_threads(2)
    cfg=runpy.run_path(str(ROOT/'scripts/prepare-uiis-bll.py'))['build_config']()
    assert cfg.model.panoptic_head.strides[0]==4
    dataset=DATASETS.build(cfg.train_dataloader.dataset); dataset.full_init()
    ann=ROOT/'data/UIIS/annotations/train.json'
    annotation_hash=sha(ann)
    raw=json.loads(ann.read_text())
    order=sorted(range(len(dataset)),key=lambda i:dataset.get_data_info(i)['img_id'])
    assert len(order)==len(raw['images'])==3937
    assert [dataset.get_data_info(i)['img_id'] for i in order]==sorted(d['id'] for d in raw['images'])
    if image_ids is not None:
        assert len(image_ids)==len(set(image_ids))
        order=[i for i in order if dataset.get_data_info(i)['img_id'] in image_ids]
        assert len(order)==len(image_ids)
    preprocessor=MODELS.build(cfg.model.data_preprocessor).cpu()
    e0=ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth'
    e0_hash=sha(e0)
    previous=ROOT/'reports/uiis-bll-data-audit/images.jsonl'
    reference={r['id']:r for r in map(json.loads,previous.read_text().splitlines())}
    rows=[]; pending=[]; began=time.monotonic()
    result.update(expected_images=len(order),official_images=3937,seed_rule='20261012 + image_id',device='cpu',torch=torch.__version__,
                  annotation_sha256=annotation_hash,model_built=False,optimizer_built=False,
                  target_rule=TARGET_RULE,reference_sha256=sha(previous),grid_stride=4,pipeline=cfg.train_pipeline,preprocessor=cfg.model.data_preprocessor.to_dict())
    def record(row):
        rows.append(row)
        with (OUT/'images.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        if len(rows)%200==0: progress('AUDITED',len(rows),'vanished_images',sum(bool(r.get('vanished')) for r in rows))
    def process(batch):
        try:
            with torch.no_grad():
                data=preprocessor(pseudo_collate([x[0] for x in batch]),training=True)
                assert data['inputs'].dtype==torch.float32 and torch.isfinite(data['inputs']).all()
                assert data['inputs'].shape[-2:]==(1024,1024)
                for s,(_,row) in zip(data['data_samples'],batch):
                    try:
                        gt=s.gt_instances
                        masks=gt.masks.to_tensor(dtype=torch.float32,device='cpu')
                        h,w=s.img_shape[:2]; hp,wp=masks.shape[-2:]
                        assert (hp,wp)==(1024,1024) and 0<h<=hp and 0<w<=wp
                        assert len(gt.labels)==len(masks) and ((gt.labels>=0)&(gt.labels<7)).all()
                        assert ((masks==0)|(masks==1)).all()
                        assert not masks[:,h:,:].any() and not masks[:,:,w:].any()
                        assert torch.isfinite(gt.bboxes).all()
                        before=masks.flatten(1).sum(1)
                        small=F.interpolate(masks[:,None],(256,256),mode='nearest')[:,0] if len(masks) else masks.new_empty((0,256,256))
                        nearest=small.flatten(1).sum(1)
                        after=small[:,:h//4,:w//4].flatten(1).sum(1)
                        old_after=after
                        target,valid=build_bll_targets(masks,(256,256),(h,w))
                        after=target.flatten(1).sum(1)
                        lost=((before>0)&(after==0)).nonzero().flatten().tolist()
                        ref=reference[row['id']]
                        assert ref['processing_ok'] and ref['image_sha256']==row['image_sha256']
                        assert before.tolist()==ref['gt_pixels'] and nearest.tolist()==ref['nearest_pixels']
                        assert old_after.tolist()==ref['valid_grid_pixels']
                        # Every valid source foreground pixel must lie in an occupied block.
                        expanded=target.repeat_interleave(4,1).repeat_interleave(4,2)
                        assert not ((masks>0)&(expanded==0)).any()
                        expanded_pixels=expanded[:,:h,:w].flatten(1).sum(1)
                        row.update(old_vanished=len(ref['vanished']),paired_reference_exact=True,
                            occupied_expanded_valid_pixels=expanded_pixels.tolist(),
                            old_valid_grid_pixels=old_after.tolist())
                        row.update(processing_ok=True,valid_shape=[h,w],padded_shape=[hp,wp],valid_grid=[(h+3)//4,(w+3)//4],
                            augmented_instances=len(masks),ignored_instances=len(s.ignored_instances),
                            labels=gt.labels.tolist(),gt_pixels=before.tolist(),nearest_pixels=nearest.tolist(),
                            valid_grid_pixels=after.tolist(),empty_augmented_masks=int((before==0).sum()),
                            vanished=[dict(index=j,label=int(gt.labels[j]),gt_pixels=float(before[j]),
                                bbox=gt.bboxes[j].tolist(),nearest_pixels=float(nearest[j]),
                                reason='nearest_downsample' if nearest[j]==0 else 'partial_cell_exclusion') for j in lost],
                            has_padding=(h<hp or w<wp),partial_cells=(h%4!=0 or w%4!=0),
                            transform={k:v for k,v in s.metainfo.items() if k in ['ori_shape','img_shape','scale_factor','flip','flip_direction']})
                    except Exception:
                        row.update(processing_ok=False,error=traceback.format_exc())
                    record(row)
        except Exception:
            error=traceback.format_exc()
            for _,row in batch:
                if not any(r['id']==row['id'] for r in rows):
                    row.update(processing_ok=False,error=error); record(row)
    for index in order:
        info=dataset.get_data_info(index); img_id=info['img_id']; seed=20261012+img_id
        row=dict(id=img_id,index=index,filename=Path(info['img_path']).name,seed=seed,
                 source_instances=len(info['instances']),source_ignored=sum(x['ignore_flag'] for x in info['instances']))
        try:
            row['image_sha256']=sha(info['img_path'])
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            item=dataset.prepare_data(index)  # No silent refetch/replacement.
            if item is None:
                ref=reference[img_id]
                assert not ref['processing_ok'] and img_id==976 and ref['image_sha256']==row['image_sha256']
                row.update(processing_ok=True,pipeline_filtered=True,paired_reference_exact=True)
                record(row); continue
            assert item['data_samples'].img_id==img_id
            pending.append((item,row))
            if len(pending)==2: process(pending); pending=[]
        except Exception:
            row.update(processing_ok=False,error=traceback.format_exc()); record(row)
    if pending: process(pending)
    assert len(rows)==len(order) and len({r['id'] for r in rows})==len(order)
    good=[r for r in rows if r['processing_ok'] and not r.get('pipeline_filtered')]
    filtered=[r for r in rows if r.get('pipeline_filtered')]
    cases=[r for r in good if r['vanished']]
    result.update(completed=True,checked=len(rows),processing_errors=len(rows)-len(good)-len(filtered),
        grid_checked=len(good),pipeline_filtered=len(filtered),
        paired_reference_exact=all(r.get("paired_reference_exact",False) for r in rows),
        previous_vanished_instances=sum(r["old_vanished"] for r in good),
        occupied_expanded_valid_pixels=sum(sum(r["occupied_expanded_valid_pixels"]) for r in good),
        original_valid_foreground_pixels=sum(sum(r["gt_pixels"]) for r in good),
        images_with_padding=sum(r['has_padding'] for r in good),images_with_partial_cells=sum(r['partial_cells'] for r in good),
        empty_images_after_augmentation=sum(r['augmented_instances']==0 for r in good),
        augmented_instances=sum(r['augmented_instances'] for r in good),
        empty_augmented_masks=sum(r['empty_augmented_masks'] for r in good),
        vanished_images=len(cases),vanished_instances=sum(len(r['vanished']) for r in cases),
        vanished_by_reason={reason:sum(v['reason']==reason for r in cases for v in r['vanished'])
                           for reason in ['nearest_downsample','partial_cell_exclusion']},
        annotation_unchanged=sha(ann)==annotation_hash,e0_unchanged=sha(e0)==e0_hash,
        elapsed_seconds=time.monotonic()-began,
        limitations=['one fixed augmentation per image, not all random augmentations or exact Runner epoch order',
                     'no forward, prediction boundary union, Hungarian matching or AP evaluated'])
    result['passed']=result['processing_errors']==result['empty_augmented_masks']==result['vanished_instances']==0 and result['annotation_unchanged'] and result['e0_unchanged'] and result['paired_reference_exact']
    (OUT/'findings.json').write_text(json.dumps(dict(vanished=cases,processing_errors=[r for r in rows if not r['processing_ok']]),indent=2)+'\n')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--ids',type=int,nargs='+',help='Explicit IDs for recorded infrastructure failures only')
    args=parser.parse_args()
    OUT=args.output
    if OUT.exists(): raise FileExistsError('Refusing to overwrite audit')
    OUT.mkdir(parents=True)
    result=dict(completed=False,passed=False)
    try: audit(result,args.ids)
    except Exception:
        result['error']=traceback.format_exc(); raise
    finally:
        result['sources']={f:sha(ROOT/f) for f in ['scripts/audit-uiis-bll-data.py','scripts/prepare-uiis-bll.py',
                                                'scripts/prepare-uiis-baseline.py','scripts/query_bll.py']}
        (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    progress(json.dumps({k:result[k] for k in ['completed','passed','checked','processing_errors','vanished_images','vanished_instances','elapsed_seconds']}))
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__': main()
