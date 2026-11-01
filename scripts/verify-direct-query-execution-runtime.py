"""Import and negative guard checks only; no model or successful optimizer step."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from mmengine.config import Config
from mmdet.utils import register_all_modules
from scripts.ap_screen_runtime import ScreenFiniteOptimWrapper,sha
import scripts.direct_query_feature_head
register_all_modules()
for filename in ['run-direct-query-screen.py','run-valid-region-ap-screen.py']:
 spec=importlib.util.spec_from_file_location(filename.replace('-','_'),ROOT/'scripts'/filename)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
seal=json.loads((ROOT/'reports/direct-query-execution-20261101/acceptance.json').read_text())
for name,digest in seal['sources'].items():assert sha(ROOT/name)==digest,name
cfg=Config.fromfile(str(ROOT/'reports/direct-query-screen-preparation-20261101/candidate.py'))
assert cfg.model.panoptic_head.type=='DirectQueryFeatureHead' and cfg.train_cfg.max_epochs==3
param=torch.nn.Parameter(torch.tensor([1.]))
optimizer=torch.optim.AdamW([param],lr=1e-5)
wrapper=ScreenFiniteOptimWrapper(optimizer=optimizer)
for action in ['loss','gradient']:
 try:
  if action=='loss':wrapper.backward(torch.tensor(float('nan')))
  else:
   param.grad=torch.tensor([float('inf')]);wrapper.step()
 except FloatingPointError:pass
 else:raise AssertionError('nonfinite '+action+' accepted')
assert float(param)==1 and not optimizer.state
for path in ['work_dirs/direct_query_ap_screen_candidate','reports/direct-query-ap-screen-run','reports/direct-query-ap-screen-candidate']:
 assert not (ROOT/path).exists(),path
report=dict(passed=True,imports_passed=True,source_hashes_verified=True,real_mmengine_nonfinite_loss_and_gradient_rejected=True,model_constructed=False,optimizer_updates=0,training_outputs_absent=True,torch_version=torch.__version__)
(ROOT/'reports/direct-query-execution-20261101/runtime.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
