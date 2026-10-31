"""Head overlay only: disabled by default, not a training runner entry."""
custom_imports = dict(imports=['scripts.valid_region_head'], allow_failed_imports=False)
panoptic_head = dict(type='ValidRegionMask2FormerHead', match_valid_region=False)
