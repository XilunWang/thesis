"""Fresh-process restoration of preflight step0, with no optimizer updates."""
import hashlib
import json
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.position_grid4 import Grid4PositionHead
from scripts.position_pilot_runtime import load_checkpoint


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    out=ROOT/'reports/label-only-pilot-entry-20261101'
    assert not (out/'restore-process.json').exists()
    proof=json.loads((out/'result.json').read_text())
    pp=ROOT/'docs/label-only-pilot-plan-20261101.json';plan=json.loads(pp.read_text())
    assert proof['passed'] and proof['training_updates']==0 and proof['plan_sha256']==sha(pp)
    for n,h in plan['code_sha256'].items():assert sha(ROOT/'scripts'/n)==h
    path=out/'step0.pth';checkpoint_hash=sha(path)
    original=torch.load(ROOT/'work_dirs/position_grid4_pilot/step0.pth',map_location='cpu')
    expected=torch.load(path,map_location='cpu')
    assert all(torch.equal(v,original['head'][k]) for k,v in expected['head'].items())
    torch.manual_seed(987654);head=Grid4PositionHead()
    assert any(not torch.equal(v,expected['head'][k]) for k,v in head.state_dict().items())
    optimizer=torch.optim.AdamW(head.parameters(),lr=0.123,weight_decay=0.987)
    def forbidden(*a,**k):raise AssertionError('Parameter updates prohibited')
    optimizer.step=forbidden
    options=dict(manifest_hash=sha(pp),code_hashes=proof['codes'],view_hash=original['view_hash'],kind='label_only_pilot')
    # Actual Pilot constructs its CUDA model BEFORE restoring the checkpoint.
    # Match that order; do not leave initial CUDA seeding queued until after restore.
    torch.cuda.init()
    assert torch.cuda.is_initialized()
    restored=load_checkpoint(path,head,optimizer,**options)
    assert restored['step']==0 and restored['cursor']==0 and restored['elapsed_seconds']==0
    assert all(torch.equal(v,expected['head'][k]) for k,v in head.state_dict().items())
    assert not optimizer.state and optimizer.state_dict()['param_groups']==expected['optimizer']['param_groups']
    assert optimizer.param_groups[0]['lr']==plan['settings']['optimizer']['lr']
    assert random.getstate()==expected['rng']['python']
    actual=np.random.get_state();wanted=expected['rng']['numpy']
    assert actual[0]==wanted[0] and np.array_equal(actual[1],wanted[1]) and actual[2:]==wanted[2:]
    assert torch.equal(torch.get_rng_state(),expected['rng']['torch'])
    cuda=torch.cuda.get_rng_state_all()
    assert len(cuda)==len(expected['rng']['cuda']) and all(torch.equal(x,y) for x,y in zip(cuda,expected['rng']['cuda']))
    rejected=False
    try:load_checkpoint(ROOT/'work_dirs/position_grid4_pilot/step0.pth',head,optimizer,**options)
    except ValueError:rejected=True
    assert rejected and all(torch.equal(v,expected['head'][k]) for k,v in head.state_dict().items())
    assert sha(path)==checkpoint_hash and not (ROOT/'work_dirs/position_label_only_pilot').exists()
    result=dict(passed=True,training_updates=0,plan_sha256=sha(pp),checkpoint_sha256=checkpoint_hash,
        fresh_initial_parameters_differed=True,head_restored_exact=True,original_step0_head_exact=True,
        optimizer_groups_restored_from_different_initial_values=True,optimizer_state_empty=True,
        cuda_initialized_before_restore_as_in_actual_entry=True,
        rng_python_numpy_torch_cuda_exact=True,step_cursor_elapsed_zero=True,
        old_checkpoint_kind_rejected=True,training_directory_absent=True,
        limitation='Real preflight step0 only. Does not establish nonzero Adam moment restoration or updated training trajectory.',
        script_sha256=sha(Path(__file__)))
    (out/'restore-process.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
