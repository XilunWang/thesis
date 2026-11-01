"""Default-off, label-only ablation; unchanged legacy BCE and inference."""
import torch
from scripts.instance_representative import make_layout
from scripts.isolated_position_head import position_loss

VERSION = 'representative-label-only-v1'


def objective(logits, targets, valid, instances=None, *, enabled=False):
    if not enabled:
        return position_loss(logits, targets, valid), None
    if instances is None or len(instances)!=len(logits):
        raise ValueError('Per-image masks, IDs and original areas required')
    if logits.shape!=targets.shape or logits.shape!=valid.shape:
        raise ValueError('Mismatched shapes')
    labels=torch.zeros_like(targets);layouts=[]
    for index,(z,keep,item) in enumerate(zip(logits,valid,instances)):
        layout=make_layout(z.detach().cpu().numpy(),item['masks'],item['ids'],item['areas'],keep.cpu().numpy())
        cells=list(layout['representatives'].values())
        if cells: labels[index].reshape(-1)[cells]=1
        if int(labels[index].sum())!=int(targets[index].sum()):
            raise ValueError('Representative count differs from legacy target count')
        layouts.append(layout)
    return position_loss(logits,labels,valid),dict(version=VERSION,labels=labels,layouts=layouts)
