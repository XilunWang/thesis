import zipfile,xml.etree.ElementTree as E,posixpath
from pathlib import Path
B=Path('presentations/.build/related-work');P='http://schemas.openxmlformats.org/presentationml/2006/main';A='http://schemas.openxmlformats.org/drawingml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';E.register_namespace('p',P);E.register_namespace('a',A);E.register_namespace('r',R)
with zipfile.ZipFile(B/'original.pptx') as z,zipfile.ZipFile(B/'authored.pptx') as n:
 data={x:z.read(x) for x in z.namelist()};changed=[]
 for i in [7,8]:
  sn=f'ppt/slides/slide{i}.xml';root=E.fromstring(data[sn]);tree=root.find(f'.//{{{P}}}spTree')
  for el in list(tree):
   off=el.find(f'.//{{{A}}}xfrm/{{{A}}}off')
   if off is not None:
    x,y=int(off.get('x')),int(off.get('y'))
    if x>=180*9525 and (y>=100*9525 or el.tag==f'{{{P}}}sp'):tree.remove(el)
  origids=[int(t.get('id')) for t in root.iter(f'{{{P}}}cNvPr')];idx=max(origids)+1
  src=E.fromstring(n.read(sn)).find(f'.//{{{P}}}spTree')
  for el in src:
   props=el.find(f'.//{{{P}}}cNvPr')
   if props is not None and props.get('name','').startswith('research-'):
    props.set('id',str(idx));idx+=1;tree.append(el)
  data[sn]=E.tostring(root,encoding='utf-8',xml_declaration=True);changed.append(sn)
  rel=f'ppt/slides/_rels/slide{i}.xml.rels'
  def notespath(zz):
   rr=E.fromstring(zz.read(rel));t=next(t for t in rr if t.get('Type').endswith('/notesSlide')).get('Target');return posixpath.normpath(posixpath.join('ppt/slides',t)).lstrip('/')
  orig_note=notespath(z);data[orig_note]=n.read(notespath(n));changed.append(orig_note)
 with zipfile.ZipFile(B/'candidate.pptx','w',zipfile.ZIP_DEFLATED) as out:
  for k,v in data.items():out.writestr(k,v)
 assert all(data[k]==z.read(k) for k in z.namelist() if k not in changed)
 print('ONLY_CHANGED',changed)
