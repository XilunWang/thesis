"""Render a proposal only. No model imports or training operations."""
from pathlib import Path
from html import escape
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
OUT.mkdir(exist_ok=True)
W, H = 1800, 1540
canvas = Image.new('RGB', (W, H), '#f7f9fc')
draw = ImageDraw.Draw(canvas)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', f'<rect width="{W}" height="{H}" fill="#f7f9fc"/>']


def text(x, y, s, size=25, color='#203047', bold=False):
    font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc', size)
    draw.text((x, y), s, font=font, fill=color, anchor='mm')
    svg.append(f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central" font-family="Microsoft YaHei,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(s)}</text>')


def box(x, y, w, h, title, lines, color='#ffffff', stroke='#8b99aa'):
    draw.rounded_rectangle((x, y, x+w, y+h), radius=12, fill=color, outline=stroke, width=2)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" stroke="{stroke}" stroke-width="2"/>')
    text(x+w/2, y+31, title, 27, bold=True)
    for i, s in enumerate(lines):
        assert draw.textbbox((0, 0), s, font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23))[2] < w-18, s
        text(x+w/2, y+69+i*31, s, 23)


def arrow(points, color='#66788b', dashed=False):
    for a, b in zip(points, points[1:]):
        if dashed:
            distance = math.dist(a, b)
            for t in range(0, int(distance), 16):
                u, v = t/distance, min(t+8, distance)/distance
                draw.line([(a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1])), (a[0]+v*(b[0]-a[0]), a[1]+v*(b[1]-a[1]))], fill=color, width=3)
        else:
            draw.line([a,b], fill=color, width=3)
    a,b=points[-2:]
    angle=math.atan2(b[1]-a[1],b[0]-a[0])
    tri=[b,(b[0]-15*math.cos(angle-.45),b[1]-15*math.sin(angle-.45)),(b[0]-15*math.cos(angle+.45),b[1]-15*math.sin(angle+.45))]
    draw.polygon(tri,fill=color)
    svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="3"'+(' stroke-dasharray="8 8"' if dashed else '')+'/>')
    svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in tri)}" fill="{color}"/>')


text(900, 49, '面向 Mask2Former 的适配设计', 40, bold=True)
text(900, 99, '2026-10-30 · 设计稿，尚未实现或验证收益 · 本轮不改模型、不训练', 24)
text(900, 145, '白色：原有机制    蓝色：先验证 D    绿色：后续 R    橙色虚线：仅训练的 B', 23)
box(50,200,470,140,'水下图像 → ResNet-50',['C2：stride 4，浅层空间信息','C3 / C4 / C5：高层特征'])
box(660,200,480,140,'原像素解码器',['6 层多尺度注意力 + 浅层融合','输出共享 mask 特征 F 和 memory'])
box(1280,200,470,140,'原 Transformer 解码器',['9 层更新 100 个实例 query','每层按原逻辑预测 mask'])
arrow([(520,270),(660,270)])
arrow([(1140,270),(1280,270)])
box(50,400,470,140,'浅层特征 C2',['不新增检测框或 RoI','与 F 处于相同 stride 4 网格'])
box(660,400,480,170,'D：门控细节残差补充',['局部变化 H + 语义门控 G','F_D = F + 零初始化出口的残差','关闭时直接使用原 F'],'#e7f0ff','#3975b5')
box(1280,400,470,140,'最终实例信息 q',['q：每个实例的特征向量','保留原分类分支'])
arrow([(285,340),(285,400)])
arrow([(900,340),(900,400)])
arrow([(520,470),(660,470)])
arrow([(1515,340),(1515,400)])
box(660,640,480,135,'原点积掩码头 → 初始掩码 M0',['mask_embed(q) 与 F_D 结合','D 用于各层 mask，影响后续注意力'])
box(1280,640,470,135,'原分类输出',['不新增分类器或评分模块','联合训练仍可能影响类别判断'])
arrow([(900,570),(900,640)])
arrow([(1280,505),(1210,505),(1210,695),(1140,695)])
arrow([(1515,540),(1515,640)])
box(660,850,480,170,'R：query 条件空间细化',['输入 C2、F 或 F_D、q、M0','共享轻量卷积，按 query 修正','M_ref = M0 + 实例空间残差'],'#e7f5ed','#38875b')
arrow([(900,775),(900,850)])
arrow([(1210,695),(1210,905),(1140,905)])
arrow([(660,535),(600,535),(600,880),(660,880)])
text(584,796,'F',22)
arrow([(285,540),(285,935),(660,935)])
text(434,902,'C2 空间细节',22)
box(1280,850,470,170,'R 的设计边界',['同一 query 保持实例对应','一次细化，仍是 stride 4','无 RoI；不宣称新增真实分辨率'])
box(660,1090,480,120,'细化后的实例掩码 M_ref',['原区域损失监督；推理使用此输出'],'#e7f5ed','#38875b')
arrow([(900,1020),(900,1090)])
box(1280,1090,470,120,'沿用原实例后处理',['原类别概率 × 细化 mask 置信度'])
arrow([(1140,1150),(1280,1150)])
arrow([(1750,707),(1774,707),(1774,1050),(1515,1050),(1515,1090)])
box(50,1090,470,120,'增强后的原始实例标注',['不使用旧 E1 的占有式缩小目标'])
box(660,1290,480,145,'B：边缘点监督（候选）',['原 GT 与 M0 决定检查区域','从原 GT 取点，监督 M_ref'],'#fff0db','#b87821')
arrow([(285,1210),(285,1362),(660,1362)],'#b87821',True)
arrow([(900,1210),(900,1290)],'#b87821',True)
text(900,1490,'实施顺序：D 单独验证 → R 单独验证 → 按证据组合 → 再验证 B；每步可停止',26,bold=True)
svg.append('</svg>')
canvas.save(OUT/'mask2former-adaptation-20261030.png')
(OUT/'mask2former-adaptation-20261030.svg').write_text('\n'.join(svg),encoding='utf-8')
print('DESIGN_DIAGRAM_RENDERED')
