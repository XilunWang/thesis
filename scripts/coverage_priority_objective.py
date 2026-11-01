"""Default off preserves the previous adaptive representative objective."""
import hashlib,json
import torch
from scripts.representative_label_only import objective as previous_objective
from scripts.isolated_position_head import position_loss
from scripts.coverage_priority import representatives,VERSION

def settings(enabled=False):return dict(schema=1,coverage_priority=bool(enabled),rule=VERSION,loss='unchanged-balanced-bce')
def settings_hash(enabled=False):return hashlib.sha256(json.dumps(settings(enabled),sort_keys=True).encode()).hexdigest()
def objective(logits,targets,valid,instances,*,coverage_priority=False):
    if not coverage_priority:
        return previous_objective(logits,targets,valid,instances,enabled=True)
    if logits.shape!=targets.shape or logits.shape!=valid.shape or len(instances)!=len(logits):
        raise ValueError('Mismatched batch or tensor shape')
    labels=torch.zeros_like(targets);layouts=[]
    for i,(z,keep,item) in enumerate(zip(logits,valid,instances)):
        reps=representatives(z.detach().cpu().numpy(),item['masks'],item['ids'],item['areas'],keep.cpu().numpy())
        if reps:labels[i].reshape(-1)[list(reps.values())]=1
        if int(labels[i].sum())!=int(targets[i].sum()):raise ValueError('Changed positive cardinality')
        layouts.append(dict(representatives=reps))
    return position_loss(logits,labels,valid),dict(settings=settings(True),labels=labels,layouts=layouts)
