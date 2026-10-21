"""Head-only verification overlay; not a training configuration or E1 protocol.

Merge with the existing E0 model.panoptic_head config in the verification script.
Weight 1 is solely a numerical probe. The E0 training entry never loads this file.
"""
custom_imports = dict(imports=['scripts.bll_mask2former_head'], allow_failed_imports=False)
panoptic_head = dict(type='BLLMask2FormerHead', bll_loss_weight=1.0,
                     bll_boundary_width=3)
