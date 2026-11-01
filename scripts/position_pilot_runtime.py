"""Checkpoint/order helpers for a bounded position pilot. No optimization loop."""
import hashlib
import os
import random
from pathlib import Path
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def view_at(views, cursor, order_seed):
    if not views or cursor < 0:
        raise ValueError('Empty view list or negative cursor')
    cycle, offset = divmod(cursor, len(views))
    order = np.random.default_rng(order_seed+cycle).permutation(len(views))
    return views[int(order[offset])]


def save_checkpoint(path, head, optimizer, *, manifest_hash, code_hashes, view_hash,
                    step, cursor, elapsed_seconds, kind='pilot'):
    if not 0 <= step <= 300 or cursor != step*2 or elapsed_seconds < 0:
        raise ValueError('Invalid step/cursor/time')
    path = Path(path)
    payload = dict(schema=1, kind=kind, manifest_hash=manifest_hash, code_hashes=code_hashes,
                   view_hash=view_hash, step=step, cursor=cursor, elapsed_seconds=elapsed_seconds,
                   head=head.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state())
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temp); os.replace(temp, path)


def load_checkpoint(path, head, optimizer, *, manifest_hash, code_hashes, view_hash, kind='pilot'):
    data = torch.load(path, map_location='cpu')
    expected = dict(schema=1, kind=kind, manifest_hash=manifest_hash, code_hashes=code_hashes, view_hash=view_hash)
    if any(data.get(k) != v for k,v in expected.items()):
        raise ValueError('Checkpoint kind, plan, code or view manifest mismatch')
    if not 0 <= data['step'] <= 300 or data['cursor'] != data['step']*2 or data['elapsed_seconds'] < 0:
        raise ValueError('Invalid continuation cursor')
    head.load_state_dict(data['head'], strict=True); optimizer.load_state_dict(data['optimizer'])
    restore_rng(data['rng'])
    return data
