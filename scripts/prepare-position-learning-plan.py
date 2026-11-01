"""Freeze a train-internal position-head pilot manifest; never run training."""
import collections
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/position-learning-plan-20261101'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'data/UIIS/annotations/train.json'
    assert sha(source) == '9cc586e559f54533a938e34a89017d47784c3646bc43ab2cd0d88e140ef5b49d'
    data = json.loads(source.read_text()); assert len(data['images']) == 3937
    small = collections.defaultdict(list)
    for a in data['annotations']:
        if not a.get('ignore', 0) and not a.get('iscrowd', 0) and 0 < a['area'] <= 1024:
            small[a['image_id']].append(a)
    # Existing engineering cases must not become a cherry-picked holdout.
    excluded = {2968, 2355, 2224, 2167, 3400, 976}
    groups = collections.defaultdict(list)
    for im in data['images']:
        if im['id'] in excluded:
            continue
        a = small[im['id']]
        group = ('empty' if not a else 'dense' if len(a) > 10 else
                 'sparse_coral' if any(x['category_id'] == 2 for x in a) else 'sparse_other')
        groups[group].append(im)
    quotas = dict(dense=(32, 16), sparse_coral=(16, 8), sparse_other=(48, 24), empty=(32, 16))
    splits = dict(learn=[], holdout=[])
    for group, (ntrain, ntest) in quotas.items():
        ordered = sorted(groups[group], key=lambda im: (hashlib.sha256(f"position-pilot-20261101:{im['id']}".encode()).hexdigest(), im['id']))
        assert len(ordered) >= ntrain + ntest
        for split, images in [('holdout', ordered[:ntest]), ('learn', ordered[ntest:ntest+ntrain])]:
            for im in images:
                aid = im['id']; a = small[aid]
                splits[split].append(dict(image_id=aid, file_name=im['file_name'], stratum=group,
                    original_small=len(a), small_by_category=dict(collections.Counter(x['category_id'] for x in a)),
                    augmentation_seeds=[20261012+aid, 21261012+aid]))
    assert len(splits['learn']) == 128 and len(splits['holdout']) == 64
    assert not {x['image_id'] for x in splits['learn']} & {x['image_id'] for x in splits['holdout']}
    summary = {}
    for split, rows in splits.items():
        c = collections.Counter()
        for r in rows:
            c.update(r['small_by_category'])
        summary[split] = dict(images=len(rows), raw_small=sum(c.values()), small_by_category=dict(c),
                              strata=dict(collections.Counter(r['stratum'] for r in rows)))
    protocol = dict(schema=1, stage='PLAN_ONLY_NOT_AUTHORIZED_TO_TRAIN', steps=300, batch_size=2,
        initialization_seed=20261101, order_seed=20261102, optimizer=dict(type='AdamW', lr=0.0003, weight_decay=0.0001, betas=[0.9, 0.999], eps=1e-8),
        grad_clip_norm=1.0, precision='fp32', schedule='constant', checkpoints=[0, 100, 300],
        final_checkpoint_only=True, primary_k=100, diagnostic_k=[20, 50, 100, 200],
        random_control_seeds=list(range(20261201, 20261211)), bootstrap_seed=20261301, bootstrap_repeats=1000,
        max_training_wall_minutes=30, source_sha256=sha(source),
        code_sha256={name: sha(ROOT/'scripts'/name) for name in ['isolated_position_head.py', 'identity_data_pipeline.py', 'audit-position-selection-protocol.py']},
        excluded_engineering_ids=sorted(excluded), quotas=quotas, summary=summary, splits=splits,
        scope='Stratified diagnostic subset of official train; holdout is unseen by new head only. E0 trained on official train. No official split modification.')
    OUT.mkdir(exist_ok=False)
    (OUT/'manifest.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps(summary, indent=2))
    print('manifest_sha256', sha(OUT/'manifest.json'))


if __name__ == '__main__':
    main()
