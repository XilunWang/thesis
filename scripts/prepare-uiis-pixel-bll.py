"""Independent E1 config builder; no execution on import."""
import copy
import runpy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build_config():
    cfg=runpy.run_path(str(ROOT/'scripts/prepare-uiis-baseline.py'))['build_config']()
    original=copy.deepcopy(cfg.model.panoptic_head)
    cfg.model.panoptic_head.update(type='PixelBLLMask2FormerHead',bll_loss_weight=1.,bll_boundary_width=3)
    cfg.custom_imports=dict(imports=['scripts.pixel_bll_head'],allow_failed_imports=False)
    cfg.work_dir=str(ROOT/'work_dirs/uiis_e1p_pixel_bll_r50_36e')
    restored=copy.deepcopy(cfg.model.panoptic_head)
    restored.type='Mask2FormerHead'
    del restored['bll_loss_weight'],restored['bll_boundary_width']
    assert restored==original
    return cfg
