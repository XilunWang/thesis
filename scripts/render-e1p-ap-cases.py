"""Render fixed diagnostic cases from saved masks; no model inference."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.local/analysis-libs'))
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools import mask as mu


def main():
    out = ROOT/'reports/e1p-ap-drop-20261031'
    gt = COCO(str(ROOT/'data/UIIS/annotations/val.json'))
    rows = {r['gt_id']:r for r in json.loads((ROOT/'reports/e1p-final-comparison/instances.json').read_text(encoding='utf-8'))}
    groups = {}
    for m, folder in zip(['E0','E1','E1-P'], ['uiis-baseline-36e-evaluation','uiis-e1-bll-36e-evaluation','uiis-e1p-pixel-bll-36e-evaluation']):
        groups[m] = {}
        for p in json.loads((ROOT/'reports'/folder/'predictions.segm.json').read_text(encoding='utf-8')):
            groups[m].setdefault(p['image_id'], []).append(p)
        for pp in groups[m].values(): pp.sort(key=lambda p:-p['score'])
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    canvas = Image.new('RGB',(1440,1240),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((15,8),'Saved prediction cases: green = GT contour; red = prediction contour',font=font,fill='black')
    cases = [1098,3557,1845,None]
    metadata = []
    for row, gid in enumerate(cases):
        iid = gt.anns[gid]['image_id'] if gid else 689
        im = gt.imgs[iid]; path=ROOT/'data/UIIS/val'/im['file_name']
        base=np.array(Image.open(path).convert('RGB'))
        selected = [gt.anns[gid]] if gid else gt.imgToAnns[iid]
        gm=np.zeros(base.shape[:2],dtype='uint8')
        for a in selected: gm |= gt.annToMask(a).astype('uint8')
        desc = f"GT {gid} / {gt.cats[gt.anns[gid]['category_id']]['name']}" if gid else 'Image 689: robot false positive (no robot GT)'
        detail={'gt_id':gid,'image_id':iid,'file_name':im['file_name'],'models':{}}
        for col, m in enumerate(['GT','E0','E1','E1-P']):
            arr=base.copy(); text=m
            pm=None
            if m!='GT':
                pp=groups[m][iid]
                if gid:
                    d=rows[gid][m]; ix=d['pred_index'] if d['matched'] else d['best_same_index']; p=pp[ix] if ix is not None else None
                else: p=max([p for p in pp if p['category_id']==6],key=lambda p:p['score'],default=None)
                if p:
                    pm=mu.decode(p['segmentation']).astype('uint8')
                    inter=(pm.astype(bool)&gm.astype(bool)).sum(); union=(pm.astype(bool)|gm.astype(bool)).sum()
                    iou=float(inter/union) if gid and union else None
                    text += f" score {p['score']:.3f}" + (f" / IoU {iou:.3f}" if gid else '')
                    detail['models'][m]={'score':p['score'],'iou':iou}
                    cs,_=cv2.findContours(pm,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(arr,cs,-1,(255,50,50),2)
            cs,_=cv2.findContours(gm,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(arr,cs,-1,(30,255,60),1)
            tile=Image.fromarray(arr);tile.thumbnail((350,238))
            x=col*360+5;y=45+row*295
            if col==0: draw.text((x,y),desc,font=font,fill='black')
            draw.text((x,y+24),text,font=font,fill='black');canvas.paste(tile,(x,y+48))
        metadata.append(detail)
    canvas.save(out/'cases.png')
    (out/'cases.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
