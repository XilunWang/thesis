"""Exact selected CPU data augmentation replay; no model or prediction."""
import hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import cv2
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS,MODELS
from mmdet.utils import register_all_modules
from scripts.identity_data_pipeline import with_identity
OUT=ROOT/'reports/label-only-visual-audit-20261101'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def array_sha(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()

def main():
    assert not (OUT/'export.json').exists()
    selection=json.loads((OUT/'selection.json').read_text())
    register_all_modules();torch.set_num_threads(2)
    cp=ROOT/'work_dirs/uiis_baseline_r50_36e/config-resume.py'
    reference=json.loads((ROOT/'reports/position-learning-preflight-20261101/result.json').read_text())
    assert sha(cp)==reference['config_sha256']
    cfg=Config.fromfile(str(cp));dc=with_identity(cfg.train_dataloader.dataset)
    dc.pipeline[-1]['meta_keys']=('img_id','img_path','ori_shape','img_shape','scale_factor','flip','flip_direction','homography_matrix')
    ds=DATASETS.build(dc);ds.full_init();lookup={ds.get_data_info(i)['img_id']:i for i in range(len(ds))}
    prep=MODELS.build(cfg.model.data_preprocessor).cpu()
    runs=[]
    for case in selection['cases']:
        iid,s=case['image_id'],case['seed'];ref=json.loads((ROOT/'reports/position-learning-preflight-20261101'/f'holdout-{iid}-{s}.json').read_text())
        random.seed(s);np.random.seed(s);torch.manual_seed(s)
        item=ds.prepare_data(lookup[iid]);assert item is not None
        # Packed inputs retain original BGR uint8 values before normalization.
        bgr=item['inputs'].permute(1,2,0).cpu().numpy().copy()
        meta=item['data_samples'].metainfo
        with torch.no_grad():data=prep(pseudo_collate([item]),training=True)
        sample=data['data_samples'][0]
        assert array_sha(data['inputs'].numpy())==ref['input_sha256']
        assert array_sha(sample.gt_instances.masks.masks)==ref['masks_sha256']
        assert list(bgr.shape[:2])==ref['valid_shape'][:2]
        H=np.asarray(meta['homography_matrix'],dtype=float);assert H.shape==(3,3)
        name=f"case-{case['case_id']:02d}-aug.png";assert cv2.imwrite(str(OUT/name),bgr)
        info=ds.get_data_info(lookup[iid]);rawpath=Path(info['img_path'])
        runs.append(dict(case_id=case['case_id'],image_id=iid,seed=s,file=name,sha256=sha(OUT/name),
            original_image_sha256=sha(rawpath),homography=H.tolist(),original_shape=list(meta['ori_shape']),
            valid_shape=list(bgr.shape[:2]),input_exact=True,masks_exact=True))
        print('EXPORTED',case['case_id'],flush=True)
    (OUT/'export.json').write_text(json.dumps(dict(passed=True,model_built=False,training_updates=0,device='cpu',selection_sha256=sha(OUT/'selection.json'),script_sha256=sha(Path(__file__)),runs=runs),indent=2))

if __name__=='__main__':main()
