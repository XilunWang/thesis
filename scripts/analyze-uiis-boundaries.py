"""Full UIIS fixed-threshold paired boundary diagnosis; not Boundary AP."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summarize(pairs, counts):
    result = dict(counts)
    result['pair_count'] = len(pairs)
    result['match_rate'] = counts['matched']/counts['gt'] if counts['gt'] else None
    for key in ['mask_iou', 'boundary_iou', 'combined_iou', 'mask_minus_boundary']:
        values = [p[key] for p in pairs if p[key] is not None]
        result[key] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                           median=float(np.median(values)) if values else None,
                           q25=float(np.quantile(values, .25)) if values else None,
                           q75=float(np.quantile(values, .75)) if values else None)
    result['boundary_iou_below_05'] = sum(p['boundary_iou'] is not None and p['boundary_iou'] < .5 for p in pairs)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/uiis-boundary-analysis')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing results are protected; use a new output directory')
    started = time.monotonic()
    core = load('boundary_core', ROOT/'scripts/validate-uiis-boundary.py')
    match = load('error_core', ROOT/'scripts/analyze-uiis-errors.py')
    match.self_test()
    reference_path = ROOT/'reports/uiis-boundary-pilot/reference-comparison.json'
    reference = json.loads(reference_path.read_text(encoding='utf8'))
    assert reference['passed'] and reference['differing_pixels'] == 0
    assert reference['local_script_sha256'] == core.sha(ROOT/'scripts/validate-uiis-boundary.py')
    annpath = ROOT/'data/UIIS/annotations/val.json'
    predpath = ROOT/'reports/uiis-baseline-36e-evaluation/predictions.segm.json'
    diagpath = ROOT/'reports/uiis-error-analysis/diagnostics.json'
    pilotpath = ROOT/'reports/uiis-boundary-pilot/validation.json'
    input_hashes = {str(p.relative_to(ROOT)): core.sha(p) for p in [annpath, predpath, diagpath, pilotpath, reference_path]}
    diag = json.loads(diagpath.read_text(encoding='utf8'))
    pilot = json.loads(pilotpath.read_text(encoding='utf8'))['fixed_six']
    assert core.sha(annpath) == diag['annotation_sha256'] == reference['annotation_sha256']
    assert core.sha(predpath) == diag['prediction_sha256'] == reference['prediction_sha256']
    coco = COCO(str(annpath))
    assert len(coco.imgs) == 691
    assert not any(a.get('ignore', 0) and not a.get('iscrowd', 0) for a in coco.anns.values())
    grouped = defaultdict(list)
    for idx, p in enumerate(json.loads(predpath.read_text())):
        assert p['image_id'] in coco.imgs
        if p['score'] >= .5:
            grouped[p['image_id']].append((idx, p))
    oldrows = {r['image_id']: r for r in diag['per_image']}
    pilotrows = {r['image_id']: r for r in pilot['images']}
    total = Counter()
    classes = defaultdict(Counter)
    sizes = defaultdict(Counter)
    pairs, rows = [], []
    for number, (iid, im) in enumerate(coco.imgs.items(), 1):
        gt = [a for a in coco.imgToAnns[iid] if not a.get('iscrowd', 0)]
        crowd = [a for a in coco.imgToAnns[iid] if a.get('iscrowd', 0)]
        indexed = sorted(grouped[iid], key=lambda x: -x[1]['score'])
        pp = [x[1] for x in indexed]
        gr = [coco.annToRLE(a) for a in gt]
        for r in gr:
            assert tuple(r['size']) == (im['height'], im['width']) and mu.area(r) > 0
        for p in pp:
            assert tuple(p['segmentation']['size']) == (im['height'], im['width'])
        matrix = mu.iou([p['segmentation'] for p in pp], gr, [0]*len(gt)) if pp and gt else np.zeros((len(pp), len(gt)))
        labels, matches, used = match.match(matrix, [p['category_id'] for p in pp], [a['category_id'] for a in gt])
        ciou = mu.iou([p['segmentation'] for p in pp], [coco.annToRLE(a) for a in crowd], [1]*len(crowd)) if pp and crowd else np.zeros((len(pp), len(crowd)))
        for k, label in enumerate(labels):
            if label != 'matched' and any(a['category_id'] == pp[k]['category_id'] and ciou[k, j] >= .5 for j, a in enumerate(crowd)):
                labels[k] = 'ignored_crowd'
        counts = Counter(labels)
        counts.update(gt=len(gt), crowd_gt=len(crowd), predictions=len(pp), missed=len(gt)-len(used))
        for key in set(counts) | set(oldrows[iid]['thresholds']['0.5']):
            assert counts[key] == oldrows[iid]['thresholds']['0.5'].get(key, 0), (iid, key)
        counts['unmatched_nonignored_predictions'] = len(pp)-len(used)-counts['ignored_crowd']
        total.update(counts)
        for j, a in enumerate(gt):
            size = 'small' if a['area'] < 32**2 else 'medium' if a['area'] < 96**2 else 'large'
            for bucket in [classes[coco.cats[a['category_id']]['name']], sizes[size]]:
                bucket['gt'] += 1
                bucket['matched' if j in used else 'missed'] += 1
        for p, label in zip(pp, labels):
            bucket = classes[coco.cats[p['category_id']]['name']]
            bucket['predictions'] += 1
            bucket['ignored_crowd'] += label == 'ignored_crowd'
            bucket['unmatched_nonignored_predictions'] += label not in ('matched', 'ignored_crowd')
        local_pairs = []
        for k, j in enumerate(matches):
            if j is None:
                continue
            a = gt[j]
            result = core.scores(mu.decode(pp[k]['segmentation']), mu.decode(gr[j]))
            assert abs(result['mask_iou']-float(matrix[k, j])) < 1e-12
            assert result['boundary_iou'] is not None
            size = 'small' if a['area'] < 32**2 else 'medium' if a['area'] < 96**2 else 'large'
            pair = dict(image_id=iid, file_name=im['file_name'], gt_id=a['id'], prediction_rank=k+1,
                        prediction_json_index=indexed[k][0], category=coco.cats[a['category_id']]['name'],
                        category_id=a['category_id'], gt_area=a['area'], size=size, score=pp[k]['score'],
                        boundary_width=max(1, round(.02*np.hypot(im['height'], im['width']))),
                        mask_minus_boundary=result['mask_iou']-result['boundary_iou'], **result)
            pairs.append(pair)
            local_pairs.append(pair)
        if iid in pilotrows:
            assert len(local_pairs) == len(pilotrows[iid]['pairs'])
            for new, old in zip(local_pairs, pilotrows[iid]['pairs']):
                assert all(new[k] == v for k, v in old.items())
        rows.append(dict(image_id=iid, file_name=im['file_name'], counts=dict(counts)))
        if number % 50 == 0 or number == 691:
            print(f'BOUNDARY_PROGRESS {number}/691 pairs={len(pairs)} seconds={time.monotonic()-started:.1f}', flush=True)
    assert total['matched'] == len(pairs) == 2509
    assert total['matched']+total['missed'] == 3748
    assert total['matched']+total['ignored_crowd']+total['unmatched_nonignored_predictions'] == 5022
    assert total['crowd_gt'] == 36
    assert all(core.sha(ROOT/name) == h for name, h in input_hashes.items())
    result = dict(completed=True, images=len(rows), score_threshold=.5, mask_match_threshold=.5, dilation_ratio=.02,
                  protocol='docs/boundary-evaluation-protocol.md; conditional same-class matched pairs, not AP',
                  seconds=time.monotonic()-started, input_hashes=input_hashes,
                  source_hashes={name:core.sha(ROOT/name) for name in ['scripts/analyze-uiis-boundaries.py','scripts/analyze-uiis-errors.py','scripts/validate-uiis-boundary.py']},
                  base_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
                  environment=dict(python=platform.python_version(),numpy=np.__version__),
                  overall=summarize(pairs,total),
                  by_class={name:summarize([p for p in pairs if p['category']==name],counts) for name,counts in classes.items()},
                  by_size={name:summarize([p for p in pairs if p['size']==name],counts) for name,counts in sizes.items()},
                  per_image=rows, pairs=pairs)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    lines=['# UIIS 全量离线边界诊断','',
           '完整覆盖691张官方测试图；原始划分和重复图片保留。只使用已有预测，无训练或推理。', '',
           '固定score≥0.5、同类Mask IoU≥0.5一对一配对，边界带宽为图像对角线2%。下表IoU均值仅针对已匹配实例，不是Boundary AP。', '',
           f'普通GT {total["gt"]}，匹配{len(pairs)}，漏检{total["missed"]}；预测{total["predictions"]}，crowd忽略{total["ignored_crowd"]}，其余未匹配预测{total["unmatched_nonignored_predictions"]}。']
    for title, groups in [('总体',{'all':result['overall']}),('类别',result['by_class']),('尺寸',result['by_size'])]:
        lines+=['',f'## {title}','','|组|GT|匹配|漏检|匹配率|Mask IoU均值|Boundary IoU均值|Boundary中位数|Boundary<0.5数|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name, v in groups.items():
            fmt=lambda x:'—' if x is None else f'{x:.4f}'
            lines.append(f'|{name}|{v["gt"]}|{v.get("matched",0)}|{v.get("missed",0)}|{fmt(v["match_rate"])}|{fmt(v["mask_iou"]["mean"])}|{fmt(v["boundary_iou"]["mean"])}|{fmt(v["boundary_iou"]["median"])}|{v["boundary_iou_below_05"]}|')
    lines+=['','## 解释限制','','条件均值不惩罚漏检和误报，不能代表整体性能。尺寸组按GT面积<32²、<96²、其余分组；小物体可能全部落入边界带，不能把小组高分解释为边界一定更好。两种IoU差不是可恢复的AP损失。类别样本量不同，少样本结果需谨慎。测试集已用于诊断，未来设计选择存在测试反馈影响。当前结果不证明DSGAT/MFRM/BLL有效。', '', '逐图计数与已有阈值诊断全部一致；全部配对Mask IoU与pycocotools一致；固定六图配对结果不变；输入文件运行前后哈希不变。完整逐实例结果及哈希见diagnostics.json。']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(result['overall'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
