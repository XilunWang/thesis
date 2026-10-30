"""Run authorized E1-P epoch one only, then inspect; never resume or evaluate."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/e1p-first-epoch'


def main():
    # No automatic retry after any partial or failed run.
    path = OUT / 'console.log'
    if path.exists() or (OUT / 'process-result.json').exists():
        raise FileExistsError('Existing first-epoch attempt; inspect before any action')
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/check-e1p-first-epoch.py')], check=True)
    result = dict(returncode=None, stop_after_epoch=1, evaluation=False)
    began = time.time()
    try:
        with path.open('x') as log:
            p = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/train-uiis-pixel-bll.py'),
                '--stage', 'first', '--execute-formal', '--work-dir',
                str(ROOT / 'work_dirs/uiis_e1p_pixel_bll_r50_36e')],
                stdout=log, stderr=subprocess.STDOUT, timeout=10800)
        result['returncode'] = p.returncode
        if p.returncode:
            raise RuntimeError('First epoch failed; inspect console.log; no automatic resume')
    finally:
        result['elapsed_seconds'] = time.time() - began
        (OUT / 'process-result.json').write_text(json.dumps(result, indent=2))
    subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/inspect-e1p-first-epoch.py')], check=True)
    print('E1P_FIRST_EPOCH_FINISHED_AND_STOPPED', flush=True)


if __name__ == '__main__':
    main()
