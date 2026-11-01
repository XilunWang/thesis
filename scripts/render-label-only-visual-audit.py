"""Evidence panels from original photographs and exact replayed augmentation."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'.local/analysis-libs'),str(ROOT)]
import numpy as np
import cv2
from pycocotools import mask as mu
OUT=ROOT/'reports/label-only-visual-audit-20261101'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fit(image,size=350):
    h,w=image.shape[:2];scale=min(size/w,size/h)
    a=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_NEAREST)
    result=np.full((size,size,3),245,dtype=np.uint8);y=(size-a.shape[0])//2;x=(size-a.shape[1])//2
    result[y:y+a.shape[0],x:x+a.shape[1]]=a;return result
def crop(image,x,y,r):
    h,w=image.shape[:2];x=int(round(x));y=int(round(y));r=int(r)
    out=np.full((2*r,2*r,3),245,dtype=np.uint8)
    x0,x1=max(0,x-r),min(w,x+r);y0,y1=max(0,y-r),min(h,y+r)
    out[y0-y+r:y1-y+r,x0-x+r:x1-x+r]=image[y0:y1,x0:x1];return out
def mask_from_annotation(a,h,w):
    s=a['segmentation']
    if isinstance(s,list):return mu.decode(mu.merge(mu.frPyObjects(s,h,w)))
    if isinstance(s.get('counts'),list):s=mu.frPyObjects(s,h,w)
    return mu.decode(s)

def main():
    selection=json.loads((OUT/'selection.json').read_text());proof=json.loads((OUT/'export.json').read_text())
    assert proof['passed'] and proof['selection_sha256']==sha(OUT/'selection.json')
    export={r['case_id']:r for r in proof['runs']}
    annotations=json.loads((ROOT/'data/UIIS/annotations/train.json').read_text())
    originals={x['id']:x for x in annotations['images']};byimage={}
    for a in annotations['annotations']:byimage.setdefault(a['image_id'],[]).append(a)
    panels=[];records=[]
    for case in selection['cases']:
        e=export[case['case_id']];rawpath=ROOT/'data/UIIS/train'/case['original_filename']
        assert sha(rawpath)==e['original_image_sha256'] and sha(OUT/e['file'])==e['sha256']
        raw=cv2.imread(str(rawpath));aug=cv2.imread(str(OUT/e['file']))
        ref=json.loads((ROOT/'reports/position-learning-preflight-20261101'/f"holdout-{case['image_id']}-{case['seed']}.json").read_text())
        masks=mu.decode(ref['gt_rles']).transpose(2,0,1)
        gw=ref['padded_shape'][1]//4;cx=case['cell']%gw*4;cy=case['cell']//gw*4
        x=(cx+min(cx+4,aug.shape[1]))/2;y=(cy+min(cy+4,aug.shape[0]))/2
        assert not masks[:,cy:cy+4,cx:cx+4].any()
        H=np.asarray(e['homography']);inv=np.linalg.inv(H);q=inv@np.array([x,y,1]);ox,oy=q[:2]/q[2]
        assert -1<=ox<raw.shape[1]+1 and -1<=oy<raw.shape[0]+1
        rawmask=np.zeros(raw.shape[:2],dtype=np.uint8);original_hits=[]
        for a in byimage[case['image_id']]:
            m=mask_from_annotation(a,*raw.shape[:2]);rawmask|=m
            ix=min(max(int(ox),0),raw.shape[1]-1);iy=min(max(int(oy),0),raw.shape[0]-1)
            if m[iy,ix]:original_hits.append(a['id'])
        context=raw.copy();contours,_=cv2.findContours(rawmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(context,contours,-1,(0,230,0),1)
        cv2.circle(context,(round(ox),round(oy)),max(4,round(max(raw.shape[:2])/100)),(0,0,255),2)
        scale=float(np.sqrt(abs(np.linalg.det(H[:2,:2]))));radius=max(12,round(48/scale))
        detail=aug.copy();vh,vw=aug.shape[:2]
        contours,_=cv2.findContours(masks.any(axis=0).astype(np.uint8)[:vh,:vw],cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(detail,contours,-1,(0,230,0),1)
        cv2.rectangle(detail,(cx,cy),(cx+3,cy+3),(0,0,255),1)
        images=[context,crop(raw,ox,oy,radius),crop(aug,x,y,48),crop(detail,x,y,48)]
        panel=np.full((420,1440,3),255,dtype=np.uint8)
        title=f"CASE {case['case_id']:02d} | image {case['image_id']} | rank {case['new_rank']} | distance {case['nearest_GT_pixel_distance']:.1f}px | new-only"
        cv2.putText(panel,title,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.64,(0,0,0),1,cv2.LINE_AA)
        for j,(im,label) in enumerate(zip(images,['Original context: GT green / point red','Original raw detail (point at center)','Augmented raw detail (point at center)','Augmented GT + red candidate cell'])):
            panel[58:408,j*360+5:j*360+355]=fit(im)
            cv2.putText(panel,label,(j*360+5,47),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA)
        name=f"case-{case['case_id']:02d}-panel.png";cv2.imwrite(str(OUT/name),panel);panels.append(panel)
        records.append(dict(case_id=case['case_id'],panel=name,original_xy=[float(ox),float(oy)],augmented_xy=[x,y],original_center_GT_ids=original_hits,original_crop_radius=radius))
    for start in range(0,len(panels),3):cv2.imwrite(str(OUT/f'contact-{start//3+1}.png'),np.concatenate(panels[start:start+3],axis=0))
    (OUT/'render.json').write_text(json.dumps(dict(passed=True,script_sha256=sha(Path(__file__)),runs=records),indent=2))
    print('Rendered 12 cases; original GT hits',[(r['case_id'],r['original_center_GT_ids']) for r in records])

if __name__=='__main__':main()
