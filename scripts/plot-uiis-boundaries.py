"""Render saved conditional boundary diagnostics as a static PNG (Pillow only)."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root / 'reports/uiis-boundary-analysis/diagnostics.json')
    parser.add_argument('--output', type=Path, default=root / 'reports/uiis-boundary-analysis/visualization.png')
    parser.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts/msyh.ttc'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    raw = args.input.read_bytes()
    data = json.loads(raw)
    assert data['completed']
    overall = data['overall']
    assert overall['matched'] + overall['missed'] == overall['gt']
    assert overall['matched'] + overall['unmatched_nonignored_predictions'] + overall['ignored_crowd'] == overall['predictions']
    image = Image.new('RGB', (2400, 2600), '#f5f7fa')
    draw = ImageDraw.Draw(image)
    fonts = {size: ImageFont.truetype(str(args.font), size) for size in (24, 28, 30, 34, 38, 48, 64)}
    ink, muted, blue, teal, orange = '#172d42', '#52677b', '#357bbd', '#008879', '#d18a39'

    def text(x, y, value, size=30, color=ink):
        draw.text((x, y), str(value), font=fonts[size], fill=color)

    def bar(x, y, width, value, color, height=20):
        draw.rectangle((x, y, x + width, y + height), fill='#e8edf2')
        draw.rectangle((x, y, x + width * value, y + height), fill=color)

    text(100, 65, f"{data['images']} 张图：找到多少，轮廓多准", 64)
    text(100, 160, f"UIIS · Mask2Former-R50 · score ≥ {data['score_threshold']} · 同类 Mask IoU ≥ {data['mask_match_threshold']}", 34, muted)
    text(100, 215, f"边界带宽：图像对角线的 {data['dilation_ratio']:.0%}　｜　基于保存的全量离线诊断，不重新训练或推理", 30, muted)
    draw.rounded_rectangle((80, 295, 2320, 750), 24, fill='white')
    text(120, 325, '01  全部目标与预测的去向', 38)
    for y, label, denominator, segments in [
        (410, '普通标注（GT）', overall['gt'], [(overall['matched'], teal), (overall['missed'], orange)]),
        (555, '保留预测', overall['predictions'], [(overall['matched'], teal), (overall['unmatched_nonignored_predictions'], orange), (overall['ignored_crowd'], '#a7b4c1')]),
    ]:
        text(120, y, f'{label}：{denominator:,}', 34)
        x = 610
        for count, color in segments:
            next_x = x + 1550 * count / denominator
            draw.rectangle((x, y + 4, next_x, y + 40), fill=color)
            x = next_x
    text(610, 465, f"匹配 {overall['matched']:,}　｜　未匹配标注 {overall['missed']:,}", 30)
    text(610, 610, f"匹配 {overall['matched']:,}　｜　未匹配且未忽略 {overall['unmatched_nonignored_predictions']:,}　｜　crowd 忽略 {overall['ignored_crowd']}", 28)
    text(120, 685, f"另有 {overall['crowd_gt']} 个 crowd 标注；不计入普通 GT 匹配率。", 28, muted)

    names = {'small': '小目标', 'medium': '中目标', 'large': '大目标', 'fish': '鱼', 'reefs': '礁石', 'sea-floor': '海底', 'aquatic plants': '水生植物', 'human divers': '潜水员', 'robots': '机器人', 'wrecks/ruins': '沉船 / 遗迹'}

    def panel(top, title, groups, keys):
        bottom = top + 180 + len(keys) * 100
        draw.rounded_rectangle((80, top, 2320, bottom), 24, fill='white')
        text(120, top + 25, title, 38)
        text(450, top + 90, '匹配率（匹配数 / 普通 GT）', 28, muted)
        text(1170, top + 90, '匹配实例的 IoU 均值（0–1）', 28, muted)
        text(1920, top + 90, 'Boundary < 0.5', 28, muted)
        for index, key in enumerate(keys):
            group = groups[key]
            y = top + 150 + index * 100
            text(120, y + 5, names[key], 34)
            bar(450, y + 12, 360, group['match_rate'], teal, 28)
            text(835, y, f"{group['match_rate']:.1%}", 30)
            text(835, y + 40, f"{group['matched']}/{group['gt']}", 24, muted)
            for offset, metric, color in [(0, 'mask_iou', blue), (39, 'boundary_iou', teal)]:
                mean = group[metric]['mean']
                bar(1170, y + offset + 5, 490, mean, color)
                text(1690, y + offset - 4, f'{mean:.4f}', 28, color)
            text(1950, y + 5, f"{group['boundary_iou_below_05']} / {group['matched']}", 30)
        return bottom

    panel(785, '02  按目标尺寸查看', data['by_size'], ['small', 'medium', 'large'])
    panel(1300, '03  按类别查看', data['by_class'], ['fish', 'reefs', 'aquatic plants', 'sea-floor', 'human divers', 'robots', 'wrecks/ruins'])
    draw.rectangle((120, 2220, 150, 2245), fill=blue)
    text(170, 2210, 'Mask IoU：区域重合', 28)
    draw.rectangle((630, 2220, 660, 2245), fill=teal)
    text(680, 2210, 'Boundary IoU：边界重合', 28)
    text(120, 2270, f"全部 {overall['matched']:,} 对均值：Mask {overall['mask_iou']['mean']:.4f}　｜　Boundary {overall['boundary_iou']['mean']:.4f}　｜　Boundary < 0.5：{overall['boundary_iou_below_05']} 对", 30)
    text(120, 2335, '读图提醒：IoU 均值仅统计已匹配实例，不含漏检 / 误报惩罚，不是 AP 或整体准确率。', 30, muted)
    text(120, 2380, '小目标边界带可能覆盖整个掩码；不能因其分数高就认为轮廓更好。少样本类别需谨慎解读。', 30, muted)
    text(120, 2435, '来源：reports/uiis-boundary-analysis/diagnostics.json　｜　静态数据图，并非聊天可视化截图', 28, muted)
    text(120, 2480, 'SHA256: ' + hashlib.sha256(raw).hexdigest(), 24, muted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, dpi=(240, 240))
    print(json.dumps({'output': str(args.output), 'size': image.size, 'source_sha256': hashlib.sha256(raw).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
