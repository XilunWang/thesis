"""Version the accepted grid4 representation without changing the pilot split."""
import copy
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    oldpath=ROOT/'reports/position-learning-plan-20261101/manifest.json'
    assert sha(oldpath)=='841257d619d35c50399ee44c08927194a81f35025c88e488a336c42d08b8f852'
    original=json.loads(oldpath.read_text())
    cp=ROOT/'reports/position-grid4-cache-20261101/result.json'
    gp=ROOT/'reports/position-grid4-model-20261101/result.json'
    cache=json.loads(cp.read_text());gpu=json.loads(gp.read_text())
    for proof in [cache,gpu]:
        assert proof['passed'] and proof['training_updates']==0
        for name,h in proof['code_sha256'].items():assert sha(ROOT/'scripts'/name)==h
    assert cache['views']==384 and cache['targets']==2350 and len(gpu['runs'])==3
    for split in ['learn','holdout']:
        expected={(r['image_id'],s) for r in original['splits'][split] for s in r['augmentation_seeds']}
        actual={(r['image_id'],r['seed']) for r in cache['runs'] if r['split']==split}
        assert actual==expected
    out=ROOT/'reports/position-learning-plan-grid4-20261101';out.mkdir(exist_ok=False)
    views=dict(representation='grid4-v1',runs=cache['runs'],frozen_neighbor_observations=cache['frozen_neighbor_observations'])
    (out/'views.json').write_text(json.dumps(views,indent=2))
    updated=copy.deepcopy(original)
    updated.update(schema=2,stage='GRID4_ACCEPTED_NO_TRAINING_AUTHORIZATION',representation='grid4-v1',stride=4,
                   parent_manifest_sha256=sha(oldpath),views_sha256=sha(out/'views.json'),
                   acceptance_sha256=dict(cache=sha(cp),model=sha(gp)),
                   pending=['training_and_evaluation_entry','grid4_checkpoint_roundtrip'],
                   code_sha256={n:sha(ROOT/'scripts'/n) for n in ['position_grid4.py','isolated_position_head.py','identity_data_pipeline.py','position_pilot_runtime.py']},
                   neighbor_policy='Frozen original stride8 neighbor observations by image/seed/GT IDs; 46 distinct pairs on 8 images')
    assert updated['splits']==original['splits']
    (out/'manifest.json').write_text(json.dumps(updated,indent=2))
    print(json.dumps(dict(manifest_sha256=sha(out/'manifest.json'),views_sha256=sha(out/'views.json'),
                          images=192,views=384,targets=2350,training_updates=0),indent=2))


if __name__=='__main__':main()
