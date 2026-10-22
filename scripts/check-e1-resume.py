"""Read-only model/data preflight for the explicitly authorized E1 resume from epoch one."""
import copy
import fcntl
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/e1-resume-35epochs'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    from mmengine.config import Config
    import torch,mmcv,mmdet,mmengine
    output=OUT/'preflight.json'
    if output.exists():raise FileExistsError(output)
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(passed=False,training_started=False)
    try:
        expected=json.loads((ROOT/'reports/e1-first-epoch/expected-sources.json').read_text())
        assert all(sha(ROOT/f)==h for f,h in expected['sources'].items())
        work=ROOT/'work_dirs/uiis_e1_bll_r50_36e'
        checkpoint=work/'epoch_1.pth'
        assert sorted(p.name for p in work.glob('epoch_*.pth'))==['epoch_1.pth']
        assert not (work/'result-resume.json').exists()
        assert Path((work/'last_checkpoint').read_text().strip())==checkpoint
        assert sha(checkpoint)=='4a40dc4a90b1e8e3e90e1ab538c0c74ae3081e186630e27603a6d84887e6e356'
        torch.set_num_threads(2)
        saved=torch.load(checkpoint,map_location='cpu')
        assert saved['meta']['epoch']==1 and saved['meta']['iter']==1969
        assert len(saved['state_dict'])==610 and all(torch.isfinite(v).all() for v in saved['state_dict'].values())
        assert len(saved['optimizer']['state'])==345
        assert all(int(v['step'])==1969 for v in saved['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for state in saved['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v))
        assert sorted(set(g['lr'] for g in saved['optimizer']['param_groups']))==[1e-5,1e-4]
        assert saved['param_schedulers'] and saved['bll_guard']['empty_streak']<3
        del saved
        assert json.loads((ROOT/'reports/e1-first-epoch/inspection.json').read_text())['passed']
        with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip()
        assert int(gpu.split(',')[-1])<100,gpu
        free=shutil.disk_usage(ROOT).free
        assert free>10*1024**3
        base=runpy.run_path(str(ROOT/'scripts/prepare-uiis-baseline.py'))['build_config']()
        e1=runpy.run_path(str(ROOT/'scripts/prepare-uiis-bll.py'))['build_config']()
        restored=copy.deepcopy(e1);restored.work_dir=base.work_dir
        restored._cfg_dict.pop('custom_imports')
        restored.model.panoptic_head.type=base.model.panoptic_head.type
        del restored.model.panoptic_head['bll_loss_weight'],restored.model.panoptic_head['bll_boundary_width']
        assert restored.to_dict()==base.to_dict()
        actual=Config.fromfile(str(ROOT/'work_dirs/uiis_baseline_r50_36e/config-first.py'))
        keys=['model','train_dataloader','train_cfg','optim_wrapper','param_scheduler','randomness','env_cfg','load_from']
        assert all(base[k]==actual[k] for k in keys)
        assert e1.model.panoptic_head.bll_loss_weight==1 and e1.model.panoptic_head.bll_boundary_width==3
        assert e1.train_cfg.max_epochs==36 and e1.randomness.seed==20261012
        assert e1.val_cfg is None and not e1.resume
        pretrained=sha(Path(e1.load_from))
        assert pretrained=='41b088b62838c986004f94bab2870395b669ec6bd11b6590369b838b0de06879'
        e0=sha(ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth')
        assert e0=='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
        data=ROOT/'data/UIIS';manifest=json.loads((data/'download-manifest.json').read_text())
        assert manifest['revision']=='44ca5db46599f14faf8e57643014664435823738'
        for item in manifest['files']:
            path=data/item['path'];assert path.stat().st_size==item['size'],item['path']
            if 'lfs' in item:assert sha(path)==item['lfs']['oid'],item['path']
            else:
                content=path.read_bytes()
                assert hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()==item['oid'],item['path']
        train=json.loads((data/'annotations/train.json').read_text())
        val=json.loads((data/'annotations/val.json').read_text())
        assert len(train['images'])==3937 and len(val['images'])==691
        assert sha(data/'train/XL_266.jpg')==sha(data/'val/XL_679.jpg')
        audit=json.loads((ROOT/'reports/uiis-bll-occupancy-verified/summary.json').read_text())
        full=json.loads((ROOT/'reports/bll-occupancy-fix/full-model.json').read_text())
        assert audit['passed'] and audit['previous_183_preserved']==183 and full['passed']
        for f,h in full['source_sha256'].items():assert sha(ROOT/f)==h,f
        protocol=json.loads((work/'run-protocol.json').read_text())
        assert protocol['config']==e1.pretty_text and not protocol['verification']
        assert protocol['pretrained_sha256']==pretrained
        assert protocol['train_sha256']==sha(data/'annotations/train.json')
        assert protocol['val_sha256']==sha(data/'annotations/val.json')
        assert all(sha(ROOT/f)==h for f,h in protocol['source_sha256'].items())
        e1.dump(str(OUT/'preflight-config.py'))
        result.update(passed=True,checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),gpu=gpu,free_bytes=free,
            environment=dict(torch=torch.__version__,mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmengine=mmengine.__version__),
            files_checked=len(manifest['files']),train_images=3937,test_images=691,official_duplicate_retained=True,
            train_sha256=sha(data/'annotations/train.json'),val_sha256=sha(data/'annotations/val.json'),
            pretrained_sha256=pretrained,e0_sha256=e0,expected_sources=expected,config_equal_to_actual_E0_fields=keys,
            allowed_changes=['head.type','head.bll_loss_weight','head.bll_boundary_width','custom_imports','work_dir'],
            first_epoch_iterations=1969,formal_total_iterations=70884,resume_epoch=1,resume_iteration=1969,stop_after_epoch=36,checkpoint_sha256=sha(checkpoint),protocol_unchanged=True,
            audit_summary_sha256=sha(ROOT/'reports/uiis-bll-occupancy-verified/summary.json'),
            full_model_report_sha256=sha(ROOT/'reports/bll-occupancy-fix/full-model.json'))
    except Exception:
        result['error']=traceback.format_exc();raise
    finally:output.write_text(json.dumps(result,indent=2)+'\n')
    print('E1_RESUME_PREFLIGHT_PASSED',flush=True)


if __name__=='__main__':main()
