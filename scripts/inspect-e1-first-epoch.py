"""Inspect the stopped E1 epoch-1 checkpoint and logs, without training/resume."""
import ast
import hashlib
import json
import math
import re
import traceback
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/e1-first-epoch'
WORK=ROOT/'work_dirs/uiis_e1_bll_r50_36e'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    output=OUT/'inspection.json'
    if output.exists():raise FileExistsError(output)
    result=dict(passed=False,training_resumed=False)
    try:
        process=json.loads((OUT/'process-result.json').read_text())
        first=json.loads((WORK/'result-first.json').read_text())
        pre=json.loads((OUT/'preflight.json').read_text())
        protocol=json.loads((WORK/'run-protocol.json').read_text())
        assert process['returncode']==0 and first['stage']=='first' and not first['verification']
        assert first['completed_epochs']==1 and first['completed_iterations']==1969
        assert first['load_check']['compatible_values_exact'] and first['load_check']['class_initialization_retained']
        assert first['load_check']['compatible_tensors']==608
        assert set(first['load_check']['mismatched_keys'])=={'panoptic_head.cls_embed.weight','panoptic_head.cls_embed.bias'}
        assert sorted(p.name for p in WORK.glob('epoch_*.pth'))==['epoch_1.pth']
        assert not (WORK/'result-resume.json').exists()
        checkpoint=WORK/'epoch_1.pth'
        assert Path((WORK/'last_checkpoint').read_text().strip())==checkpoint
        saved=torch.load(checkpoint,map_location='cpu')
        assert saved['meta']['epoch']==1 and saved['meta']['iter']==1969
        assert len(saved['state_dict'])==610
        assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
        states=saved['optimizer']['state'];assert len(states)==345
        steps=set()
        for state in states.values():
            steps.add(int(state['step']))
            assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
        assert steps=={1969}
        lrs=sorted(set(g['lr'] for g in saved['optimizer']['param_groups']))
        assert lrs==[.00001,.0001] and lrs==first['lrs']
        assert saved['param_schedulers'] and 'bll_guard' in saved
        assert saved['bll_guard']['empty_streak']<3
        text=(OUT/'console.log').read_text(errors='replace')
        assert 'FIRST_EPOCH_GATE: checkpoint saved; stopping for inspection' in text
        assert 'BLL_RUNNER_STAGE_OK' in text and 'Epoch(train)  [2]' not in text
        for marker in ['Traceback (most recent call last)','CUDA out of memory','BLL_GT_VANISHED','BLL_EMPTY_REGION_STREAK','NONFINITE_MODEL_PARAMETER']:
            assert marker not in text,marker
        diagnostics=[ast.literal_eval(l.split('BLL_DIAGNOSTIC ',1)[1]) for l in text.splitlines() if 'BLL_DIAGNOSTIC ' in l]
        assert len(diagnostics)==40 and all(d['vanished']==0 for d in diagnostics)
        metric_files=list(WORK.glob('*/vis_data/scalars.json'));assert len(metric_files)==1
        rows=[json.loads(l) for l in metric_files[0].read_text().splitlines() if l.strip()]
        losses=[r for r in rows if 'loss_bll' in r];assert losses
        assert all(math.isfinite(v) for r in losses for k,v in r.items() if isinstance(v,(float,int)) and ('loss' in k or k=='grad_norm'))
        assert all(r.get('epoch',1)==1 for r in losses)
        assert all(sha(ROOT/f)==h for f,h in protocol['source_sha256'].items())
        assert all(protocol['source_sha256'][f]==pre['expected_sources']['sources'][f] for f in protocol['source_sha256'])
        assert sha(ROOT/'data/UIIS/annotations/train.json')==pre['train_sha256']==protocol['train_sha256']
        assert sha(ROOT/'data/UIIS/annotations/val.json')==pre['val_sha256']==protocol['val_sha256']
        assert sha(ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth')==pre['e0_sha256']
        ranges={k:dict(first=losses[0][k],last=losses[-1][k],minimum=min(r[k] for r in losses),maximum=max(r[k] for r in losses)) for k in ['loss','loss_bll']}
        memories=[int(x) for x in re.findall(r'memory:\s*(\d+)',text)]
        result.update(passed=True,process=process,first=first,checkpoint_sha256=sha(checkpoint),
            checkpoint_bytes=checkpoint.stat().st_size,checkpoint_meta=dict(epoch=1,iter=1969),
            state_tensors=610,optimizer_states=345,optimizer_step_values=sorted(steps),lrs=lrs,
            scheduler=saved['param_schedulers'],bll_guard=saved['bll_guard'],
            all_model_optimizer_tensors_finite=True,diagnostic_records=len(diagnostics),
            logged_selected_pixels_range=[min(d['selected_pixels'] for d in diagnostics),max(d['selected_pixels'] for d in diagnostics)],
            logged_vanished=0,logged_loss_rows=len(losses),logged_loss_ranges=ranges,
            max_logged_memory_mib=max(memories),data_and_e0_unchanged=True,core_sources_unchanged=True,
            checkpoint_file=str(checkpoint),limitations=[
                'Loss rows are logger window averages, not all per-step values or AP.',
                'Runtime guards check every step; diagnostics are sampled every 50 steps.',
                'Read-only checkpoint integrity inspection, not a new Runner resume or training trajectory test.',
                'One training epoch cannot establish accuracy improvement; no test-set evaluation.'])
    except Exception:
        result['error']=traceback.format_exc();raise
    finally:
        result['inspection_script_sha256']=sha(Path(__file__))
        output.write_text(json.dumps(result,indent=2,default=str)+'\n')
    print('E1_FIRST_EPOCH_INSPECTION_PASSED',flush=True)


if __name__=='__main__':main()
