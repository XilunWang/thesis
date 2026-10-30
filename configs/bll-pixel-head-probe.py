"""Independent head overlay only; no training runner or execution entry."""
custom_imports = dict(imports=['scripts.pixel_bll_head'], allow_failed_imports=False)
panoptic_head = dict(type='PixelBLLMask2FormerHead', bll_loss_weight=1.0, bll_boundary_width=3)
