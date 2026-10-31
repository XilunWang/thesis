"""Fixed E0, two predefined test scales, no optimization."""
import copy,fcntl,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmdet.registry import DATASETS,MODELS
from mmdet.utils import register_all_modules
from pycocotools import mask as mu
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    out=ROOT/'reports/baseline-scale-pilot-20261031';sel=json.loads((out/'selection.json').read_text())
    assert not (out/'run-result.json').exists()
    with (ROOT/'work_dirs/uiis-training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        register_all_modules();torch.set_num_threads(2)
        cfg=Config.fromfile(str(ROOT/'reports/uiis-baseline-36e-evaluation/evaluation-config.py'))
        ck=ROOT/'work_dirs/uiis_baseline_r50_36e/epoch_36.pth';digest=sha(ck)
        assert digest=='53f9382f8db706be2d91da38b765b6bd5b8f26c33a2434f9a26e795fbbb8376f'
        source=ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json';sourcehash=sha(source)
        old=json.loads(source.read_text());datasets={};indices={}
        for name,scale in sel['scales'].items():
            dc=copy.deepcopy(cfg.test_dataloader.dataset)
            transforms=[t for t in dc.pipeline if t['type']=='Resize'];assert len(transforms)==1
            assert tuple(transforms[0]['scale'])==(1333,800);transforms[0]['scale']=tuple(scale)
            datasets[name]=DATASETS.build(dc);datasets[name].full_init()
            indices[name]={datasets[name].get_data_info(i)['img_id']:i for i in range(len(datasets[name]))}
        model=MODELS.build(cfg.model).cuda().eval();state=torch.load(ck,map_location='cpu')['state_dict'];model.load_state_dict(state,strict=True)
        result={'checkpoint_sha256':digest,'selection_sha256':sha(out/'selection.json'),'source_sha256':sha(Path(__file__)),'measurements':[],'reproduction':[]}
        exported={name:[] for name in datasets}
        with torch.inference_mode():
            for name,ds in datasets.items():
                for _ in range(2):
                    warm=model.test_step(pseudo_collate([ds.prepare_data(indices[name][sel['image_ids'][0]])]));del warm
                torch.cuda.synchronize()
            for n,iid in enumerate(sel['image_ids']):
                order=['original','larger'] if n%2==0 else ['larger','original']
                for name in order:
                    ds=datasets[name];item=ds.prepare_data(indices[name][iid]);data=pseudo_collate([item])
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    output=model.test_step(data)[0];torch.cuda.synchronize();seconds=time.perf_counter()-start
                    mem=torch.cuda.max_memory_allocated();ins=output.pred_instances
                    scores=ins.scores.cpu().numpy();cats=ins.labels.cpu().numpy()+1;masks=ins.masks.cpu().numpy()
                    imagepred=[]
                    for j,mask in enumerate(masks):
                        rle=mu.encode(np.asfortranarray(mask.astype('uint8')));rle['counts']=rle['counts'].decode('ascii')
                        imagepred.append(dict(image_id=iid,category_id=int(cats[j]),score=float(scores[j]),segmentation=rle))
                    exported[name].extend(imagepred)
                    if name=='original':
                        refs=[p for p in old if p['image_id']==iid];assert len(refs)==len(imagepred)
                        assert [p['category_id'] for p in refs]==cats.tolist()
                        err=max(abs(p['score']-float(scores[j])) for j,p in enumerate(refs))
                        xor=sum(int(np.count_nonzero(mu.decode(p['segmentation']).astype(bool)!=masks[j])) for j,p in enumerate(refs))
                        result['reproduction'].append(dict(image_id=iid,max_score_error=err,mask_changed_pixels=xor))
                        assert err<1e-4 and xor==0,(iid,err,xor)
                    result['measurements'].append(dict(image_id=iid,scale=name,input_shape=list(item['inputs'].shape),seconds=seconds,peak_allocated_bytes=mem))
                    del output,ins,masks,data,item
                print(iid,'BOTH_SCALES_COMPLETE',flush=True)
                (out/'progress.json').write_text(json.dumps(result,indent=2))
        assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        assert sha(ck)==digest and sha(source)==sourcehash
        for name,preds in exported.items():(out/f'{name}-predictions.json').write_text(json.dumps(preds))
        result.update(passed=True,model_state_unchanged=True,optimizer_created=False,prediction_source_unchanged=True)
        (out/'run-result.json').write_text(json.dumps(result,indent=2));print('SCALE_PILOT_COMPLETE',flush=True)
if __name__=='__main__':main()
