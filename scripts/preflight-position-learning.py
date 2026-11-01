"""CPU preflight of the frozen pilot manifest. No model training or optimizer step."""
import fcntl
import hashlib
import json
import random
import sys
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules
from pycocotools import mask as mu
from scripts.identity_data_pipeline import with_identity
from scripts.isolated_position_head import make_targets
OUT = ROOT/'reports/position-learning-preflight-20261101'
PLAN = ROOT/'reports/position-learning-plan-20261101/manifest.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def array_sha(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode()+a.tobytes()).hexdigest()


def main(result):
    assert sha(PLAN) == '841257d619d35c50399ee44c08927194a81f35025c88e488a336c42d08b8f852'
    plan = json.loads(PLAN.read_text())
    ann = ROOT/'data/UIIS/annotations/train.json'
    assert sha(ann) == plan['source_sha256']
    for name, value in plan['code_sha256'].items():
        assert sha(ROOT/'scripts'/name) == value
    raw = {a['id']: a for a in json.loads(ann.read_text())['annotations']}
    register_all_modules(); torch.set_num_threads(2)
    cfgpath = ROOT/'work_dirs/uiis_baseline_r50_36e/config-resume.py'
    cfg = Config.fromfile(str(cfgpath))
    ds = DATASETS.build(with_identity(cfg.train_dataloader.dataset)); ds.full_init()
    assert len(ds) == 3937
    index = {ds.get_data_info(i)['img_id']: i for i in range(len(ds))}
    prep = MODELS.build(cfg.model.data_preprocessor).cpu()
    result.update(manifest_sha256=sha(PLAN), config_sha256=sha(cfgpath), script_sha256=sha(Path(__file__)), runs=[])
    retained = dict(learn=set(), holdout=set()); classes = {1:set(), 2:set()}
    pairs = set(); pair_images = set(); errors = []
    for split in ['learn', 'holdout']:
        for item in plan['splits'][split]:
            iid = item['image_id']
            for seed in item['augmentation_seeds']:
                row = dict(split=split, image_id=iid, seed=seed)
                try:
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    value = ds.prepare_data(index[iid])
                    row['filtered'] = value is None
                    if value is not None:
                        with torch.no_grad():
                            data = prep(pseudo_collate([value]), training=True)
                        sample = data['data_samples'][0]; g = sample.gt_instances
                        masks = g.masks.masks
                        ids = g.annotation_ids.tolist(); areas = g.original_areas.tolist()
                        for gid, area, label in zip(ids, areas, g.labels.tolist()):
                            a = raw[gid]
                            assert a['image_id'] == iid and a['area'] == area
                            assert ds.cat2label[a['category_id']] == label
                            assert not a.get('iscrowd', 0) and not a.get('ignore', 0)
                        rles = []
                        for mask in masks:
                            r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
                            r['counts'] = r['counts'].decode(); rles.append(r)
                        row.update(annotation_ids=ids, original_areas=areas, labels=g.labels.tolist(),
                                   valid_shape=list(sample.img_shape), padded_shape=list(masks.shape[-2:]),
                                   input_sha256=array_sha(data['inputs'].numpy()), masks_sha256=array_sha(masks), gt_rles=rles)
                        target, valid, assigned = make_targets(masks, ids, areas, sample.img_shape)
                        row.update(positives=len(assigned), assignments=assigned, target_sha256=array_sha(target.numpy()),
                                   valid_sha256=array_sha(valid.numpy()), valid_cells=int(valid.sum()))
                        assert int(target.sum()) == sum(0 < a <= 1024 for a in areas)
                        retained[split].add(iid)
                        if split == 'holdout':
                            for gid in assigned:
                                cat = raw[gid]['category_id']
                                if cat in classes:
                                    classes[cat].add(gid)
                            keys = sorted(assigned)
                            for j, a in enumerate(keys):
                                for b in keys[j+1:]:
                                    if max(abs(assigned[a][q]-assigned[b][q]) for q in [0,1]) <= 1:
                                        pairs.add((a,b)); pair_images.add(iid)
                    row['checked'] = True
                except Exception:
                    row.update(checked=False, error=traceback.format_exc())
                    errors.append(dict(split=split, image_id=iid, seed=seed, error=row['error']))
                (OUT/f'{split}-{iid}-{seed}.json').write_text(json.dumps(row, indent=2))
                result['runs'].append({k:v for k,v in row.items() if k not in ['gt_rles', 'assignments']})
            if len(result['runs']) % 32 == 0:
                print('CHECKED', len(result['runs']), 'ERRORS', len(errors), flush=True)
    gates = dict(no_errors=not errors, learn_images=len(retained['learn']) >= 80,
                 holdout_images=len(retained['holdout']) >= 48, fish=len(classes[1]) >= 50,
                 coral=len(classes[2]) >= 10, neighbor_pairs=len(pairs) >= 10, neighbor_images=len(pair_images) >= 3)
    result.update(passed=all(gates.values()), gates=gates, errors=errors,
                  summary=dict(views=len(result['runs']), filtered=sum(x.get('filtered', False) for x in result['runs']),
                               retained_images={k:len(v) for k,v in retained.items()},
                               holdout_distinct_fish=len(classes[1]), holdout_distinct_coral=len(classes[2]),
                               distinct_neighbor_pairs=len(pairs), neighbor_images=sorted(pair_images)),
                  neighbor_pairs=sorted(pairs))
    print(json.dumps(result['summary']), flush=True)
    print('GATES', gates, flush=True)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    result = dict(passed=False, training_updates=0, scope='CPU data only; no E0 or learned-head loading')
    try:
        with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main(result)
    except BaseException:
        result['fatal_error'] = traceback.format_exc()
        raise
    finally:
        (OUT/'result.json').write_text(json.dumps(result, indent=2))
