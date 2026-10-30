"""GT distance-field construction: representability example, never inference."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

R=Path(__file__).resolve().parents[1]


def metrics(pred,gt,valid):
    tp=int((pred&gt&valid).sum());fn=int((~pred&gt&valid).sum());fp=int((pred&~gt&valid).sum())
    return dict(fn=fn,fp=fp,pixel_errors=fn+fp,iou=tp/max(tp+fn+fp,1))


def main():
    out=R/'reports/bll-scale-oracle-20261031';out.mkdir(exist_ok=False)
    folders=['bll-real-gradient-20261030'+('' if i==0 else f'-id{i}') for i in range(4)]
    folders += [f'bll-real-gradient-20261030-small-rank{i}' for i in range(3)]
    rows=[]
    for folder in folders:
        path=R/'reports'/folder;meta=json.loads((path/'metadata.json').read_text())
        assert hashlib.sha256((path/'trace.npz').read_bytes()).hexdigest()==meta['trace_sha256']
        a=np.load(path/'trace.npz');vh,vw=map(int,a['valid_shape'])
        for j,gi in enumerate(a['gi']):
            gt=a['gt'][gi].astype(bool);hp,wp=gt.shape
            valid=np.zeros_like(gt);valid[:vh,:vw]=True
            padded=np.pad(gt.astype(np.uint8),1)
            inside=cv2.distanceTransform(padded,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
            outside=cv2.distanceTransform(1-padded,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
            sdf=inside-outside
            variants={}
            for stride in [4,2,1]:
                ys=(np.arange(hp//stride,dtype=np.float32)+.5)*stride-.5
                xs=(np.arange(wp//stride,dtype=np.float32)+.5)*stride-.5
                xx,yy=np.meshgrid(xs,ys)
                coarse=cv2.remap(sdf,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                up=cv2.resize(coarse,(wp,hp),interpolation=cv2.INTER_LINEAR)
                variants[str(stride)]=metrics(up>=0,gt,valid)
                if meta['image_id']==7 and int(a['qi'][j])==38:
                    variants[str(stride)]['known_two_values']=[float(up[y,x]) for y,x in [(622,1007),(623,1007)]]
                if stride==1:assert variants['1']['pixel_errors']==0
            current=cv2.resize(a['final'][0,a['qi'][j]],(wp,hp),interpolation=cv2.INTER_LINEAR)
            rows.append(dict(image_id=meta['image_id'],query=int(a['qi'][j]),area=int(gt.sum()),
                size='small' if int(gt.sum())<1024 else 'non_small',current=metrics(current>=0,gt,valid),oracle=variants))
    result=dict(passed=True,rows=rows,scope='GT-derived signed Euclidean distance at native grid centres, bilinear reconstruction, threshold0. Constructive examples only; not optimal fits or deployable predictions, no model training. All valid pixels, not selected-band-only.')
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps([r for r in rows if r['size']=='small'],indent=2))


if __name__=='__main__':main()
