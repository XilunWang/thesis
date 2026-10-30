"""Validate saved real-image traces and render actual supervision, no model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def boundary(mask):
    p = np.pad(mask.astype(float), ((0, 0), (3, 3), (3, 3)))
    sums = np.zeros_like(mask, dtype=float)
    for y in range(7):
        for x in range(7):
            sums += p[:, y:y + mask.shape[1], x:x + mask.shape[2]]
    return np.abs(49 * mask - sums) > 7.35


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/bll-real-gradient-summary-20261030')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(exist_ok=False)
    rows = []
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    for image_id in range(4):
        folder = ROOT / ('reports/bll-real-gradient-20261030' + ('' if image_id == 0 else f'-id{image_id}'))
        meta = json.loads((folder / 'metadata.json').read_text())
        assert hashlib.sha256((folder / 'trace.npz').read_bytes()).hexdigest() == meta['trace_sha256']
        a = np.load(folder / 'trace.npz')
        gt = a['gt'][a['gi']]; target = a['target']; valid = a['valid']; region = a['region']
        n, hp, wp = gt.shape; h, w = target.shape[-2:]; vh, vw = a['valid_shape']
        expected = gt.reshape(n, h, hp // h, w, wp // w).max(axis=(2, 4))
        np.testing.assert_array_equal(target, expected)
        prev = (a['previous'][0, a['qi']] >= 0) & valid
        np.testing.assert_array_equal(region, (boundary(prev) | boundary(target)) & valid)
        z = a['final'][0, a['qi']].astype(float)
        denominator = max(int(region.sum()), 1)
        loss = (np.logaddexp(0, z) - z * target)[region].sum() / denominator
        assert np.isclose(loss, meta['losses']['loss_bll'], rtol=2e-6, atol=1e-7)
        gb = a['gradient_bll'][0, a['qi']]
        expected_grad = (1 / (1 + np.exp(-np.clip(z, -700, 700))) - target) * region / denominator
        np.testing.assert_allclose(gb, expected_grad, rtol=5e-5, atol=1e-8)
        assert np.count_nonzero(np.delete(a['gradient_bll'][0], a['qi'], axis=0)) == 0
        ga = a['gradient_original'][0, a['qi']]
        full = np.zeros((h, w), bool); full[:vh // (hp // h), :vw // (wp // w)] = True
        minority = region & full & (a['coverage'] > 0) & (a['coverage'] < .5)
        conflicts = minority & (ga > 0) & (gb < 0)
        assert int(conflicts.sum()) == meta['comparisons']['minority_full_cells']['original_pushes_down_bll_up']
        row = dict(image_id=image_id, filename=meta['filename'], matched=n,
                   minority=int(minority.sum()), conflict=int(conflicts.sum()),
                   comparisons=meta['comparisons'], independent_checks=True)
        rows.append(row)
        # Fixed first matched instance, not chosen by severity. Panel masks use nearest
        # expansion ONLY to show where grid supervision lies, not model inference.
        rgb = a['input_bgr'].transpose(1, 2, 0)[:vh, :vw, ::-1].astype(np.uint8)
        base = Image.fromarray(rgb)
        panels = [('增强后的真实训练图', None, None), ('人工标注（绿色）', gt[0, :vh, :vw], (0, 200, 80)),
                  ('BLL占有目标展开（橙色）', target[0], (255, 140, 0)),
                  ('实际BLL监督区域（黄色）', region[0], (255, 220, 0)),
                  ('不足半格却标前景（紫色）', minority[0], (210, 0, 210)),
                  ('原损失少涂 / BLL多涂（红色）', conflicts[0], (255, 20, 20))]
        canvas = Image.new('RGB', (1200, 900), 'white'); draw = ImageDraw.Draw(canvas)
        for j, (label, mask, color) in enumerate(panels):
            panel = base.copy()
            if mask is not None:
                if mask.shape != (vh, vw):
                    mask = np.repeat(np.repeat(mask, hp // h, 0), wp // w, 1)[:vh, :vw]
                arr = np.asarray(panel).copy(); selected = mask.astype(bool)
                arr[selected] = (.35 * arr[selected] + .65 * np.array(color)).astype(np.uint8)
                panel = Image.fromarray(arr)
            x, y = (j % 3) * 400, (j // 3) * 450
            draw.text((x + 8, y + 8), label, font=font, fill='black')
            tile = ImageOps.contain(panel, (390, 400))
            canvas.paste(tile, (x + (400 - tile.width) // 2, y + 45))
        canvas.save(out / f'image-{image_id}.png')
    result = dict(passed=True, images=4, matched=sum(r['matched'] for r in rows),
                  minority=sum(r['minority'] for r in rows), conflict=sum(r['conflict'] for r in rows), rows=rows,
                  limitation='First image exploratory, then next three IDs checked; one fixed augmentation each. Not total-loss/full-model gradient or causal ablation.')
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
