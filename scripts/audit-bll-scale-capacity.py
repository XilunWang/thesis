"""Necessary bilinear sign constraints on fixed real GT. No fitting/training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parents[1]


def scan(mask,stride):
    # Output sample centres between adjacent native grid centres, align_corners=False.
    start=stride//2
    runs=(mask.shape[1]-start)//stride
    blocks=mask[:,start:start+runs*stride].reshape(mask.shape[0],runs,stride)
    switches=np.count_nonzero(np.diff(blocks.astype(np.int8),axis=-1),axis=-1)
    bad=switches>1
    patterns=[]
    for cut in range(stride+1):
        a=np.arange(stride)>=cut;patterns.extend([a,~a])
    # Minimum Hamming error to any at-most-one-transition sign sequence.
    minimum=np.full(blocks.shape[:2],stride,dtype=np.int64)
    for p in patterns:minimum=np.minimum(minimum,np.count_nonzero(blocks!=p,axis=-1))
    return dict(violating_segments=int(bad.sum()),pixel_error_lower_bound=int(minimum.sum()),
                examples=[dict(line=int(y),start=int(start+b*stride),labels=blocks[y,b].astype(int).tolist()) for y,b in np.argwhere(bad)[:3]])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/bll-scale-capacity-20261031');args=parser.parse_args()
    out=R/args.output;out.mkdir(exist_ok=False)
    # A linear function must satisfy this necessary condition; test varied slopes/offsets.
    for slope in [-3,-.2,0,.2,3]:
        for offset in [-2,0,2]:
            assert scan((slope*np.arange(18)+offset>=0)[None,:],4)['violating_segments']==0
    assert scan(np.array([[0,0,0,1,1,0,0,0]],bool),4)['pixel_error_lower_bound']==1
    folders=['bll-real-gradient-20261030'+('' if i==0 else f'-id{i}') for i in range(4)]
    folders += [f'bll-real-gradient-20261030-small-rank{i}' for i in range(3)]
    rows=[]
    for folder in folders:
        path=R/'reports'/folder;meta=json.loads((path/'metadata.json').read_text())
        assert hashlib.sha256((path/'trace.npz').read_bytes()).hexdigest()==meta['trace_sha256']
        a=np.load(path/'trace.npz');vh,vw=map(int,a['valid_shape'])
        for j,gi in enumerate(a['gi']):
            mask=a['gt'][gi,:vh,:vw].astype(bool);area=int(mask.sum());scales={}
            for stride in [4,2]:
                horizontal=scan(mask,stride);vertical=scan(mask.T,stride)
                scales[str(stride)]=dict(horizontal=horizontal,vertical=vertical,
                    # Cannot add axis lower bounds because they may concern same pixels.
                    total_pixel_error_lower_bound=max(horizontal['pixel_error_lower_bound'],vertical['pixel_error_lower_bound']))
            row=dict(image_id=meta['image_id'],query=int(a['qi'][j]),area=area,
                     size='small' if area<1024 else 'non_small',scales=scales)
            if meta['image_id']==7 and int(a['qi'][j])==38:
                row['known_two_horizontal_runs']=[dict(y=y,x_start=1006,labels=mask[y,1006:1010].astype(int).tolist()) for y in [622,623]]
            rows.append(row)
    totals={}
    for size in ['small','non_small']:
        rr=[r for r in rows if r['size']==size]
        totals[size]=dict(instances=len(rr),stride4_proven_nonexact=sum(r['scales']['4']['total_pixel_error_lower_bound']>0 for r in rr),
            stride4_sum_lower_bound=sum(r['scales']['4']['total_pixel_error_lower_bound'] for r in rr),
            stride2_axis_violations=sum(r['scales']['2']['total_pixel_error_lower_bound'] for r in rr))
    result=dict(passed=True,rows=rows,totals=totals,
        scope='Necessary sign condition only. Exact binary GT at augmented pixel centres. Stride2 no violation is NOT proof of full representability. Lower bound is not achieved error or AP loss. No model/optimizer/logit fitting.')
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(totals=totals,small=[r for r in rows if r['size']=='small']),indent=2))


if __name__=='__main__':main()
