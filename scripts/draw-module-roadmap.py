"""Render the approved experiment roadmap as editable SVG and preview PNG."""
from pathlib import Path
from html import escape
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/figures'
W,H=1600,1400
image=Image.new('RGB',(W,H),'#f7f9fc')
draw=ImageDraw.Draw(image)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
     '<rect width="1600" height="1400" fill="#f7f9fc"/>']


def text(x,y,value,size=25,color='#233348',bold=False):
    font=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc',size)
    draw.text((x,y),value,font=font,fill=color,anchor='mm')
    svg.append(f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="central" font-family="Microsoft YaHei, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(value)}</text>')


def box(x,y,w,h,title,lines,fill='#ffffff',stroke='#9caabb',titlecolor='#233348'):
    draw.rounded_rectangle((x,y,x+w,y+h),radius=16,fill=fill,outline=stroke,width=2)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    text(x+w/2,y+31,title,28,titlecolor,True)
    for n,line in enumerate(lines):text(x+w/2,y+70+n*34,line,23)


def line(points,arrow=False):
    draw.line(points,fill='#60748c',width=3)
    svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+'" fill="none" stroke="#60748c" stroke-width="3"/>')
    if arrow:
        x,y=points[-1];p=[(x-8,y-12),(x+8,y-12),(x,y)]
        draw.polygon(p,fill='#60748c')
        svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in p)+'" fill="#60748c"/>')


text(800,55,'先单模块适配，再组合验证',46,bold=True)
text(800,105,'实验路线图 · 箭头表示决策顺序，不表示模型数据流或权重继承',24)
box(380,145,840,100,'E0  原版 Mask2Former-R50',['基线已完成：Mask AP 32.6 · 保留既有结果，不重复训练'],'#e9f1ff','#6490c9')
line([(800,245),(800,275)])
line([(285,275),(1315,275)])
for x in [285,800,1315]:line([(x,275),(x,310)],True)
box(55,310,460,190,'E1  原版 + BLL',[
    '只检验边界监督的作用',
    '相邻 decoder 层，同 query 对应',
    '工程验证完成；正式训练未开始'], '#eaf7ef','#409263','#24613c')
box(570,310,460,190,'E2  原版 + MFRM',[
    '只检验掩码细化的作用',
    '不加 BLL / DSGAT',
    '尚未实现；接口与监督待定'])
box(1085,310,460,190,'E3  原版 + DSGAT',[
    '只检验特征增强的作用',
    '不加 BLL / MFRM',
    '尚未实现；插入位置待定'])
text(800,531,'适配优先级：E1 → E2 → E3；三项各自与 E0 比较，不继承前项训练结果',23)
for x in [285,800,1315]:line([(x,550),(x,570)])
line([(285,570),(1315,570)])
line([(800,570),(800,605)],True)
box(310,605,980,100,'组合前决策：结果是否支持明确的互补假设？',[
    '先完成单模块评估；证据不足则分析或修订，不自动叠加'], '#fff4df','#d6a244','#885e15')
line([(800,705),(800,730)])
line([(425,730),(1175,730)])
for x in [425,1175]:line([(x,730),(x,765)],True)
box(95,765,660,155,'E4a  MFRM + BLL（条件候选）',[
    '与 E2 比较：细化结果是否受益于 BLL？',
    '重新确定监督位置、尺度、匹配和损失规则'])
box(845,765,660,155,'E4b  DSGAT + 有依据的组合',[
    '与对应不含 DSGAT 的配置比较',
    '具体组合待登记；不默认三模块全部加入'])
for x in [425,1175]:line([(x,920),(x,945)])
line([(425,945),(1175,945)])
line([(800,945),(800,980)],True)
box(270,980,1060,105,'E5  去除模块消融    →    E6  配对重复验证',[
    '形成候选后再规划、另定预算与授权；复用已有等价对照'])
box(90,1125,1420,165,'BLL 监督位置：独立实验与组合实验要区分',[
    '当前 E1：前一 decoder 层 + GT 形成选区 → 监督最后一层掩码',
    '候选 E4a：细化前掩码 + GT 形成选区 → 监督 MFRM 细化后掩码',
    'E4a 尚未实现；不默认同时叠加两处 BLL，也不新增独立边界预测头'], '#eef4f8','#9caabb')
text(800,1330,'统一规则：官方 3937 / 691 划分及重复图保留 · 同源预训练、预算和评价 · 不挑 best',23)
text(800,1372,'本次仅确认路线和更新文档，不启动训练。',25,'#9b4624',True)
svg.append('</svg>')
OUT.mkdir(exist_ok=True)
(OUT/'module-adaptation-roadmap.svg').write_text('\n'.join(svg),encoding='utf-8')
image.save(OUT/'module-adaptation-roadmap.png')
print('ROADMAP_RENDERED')
