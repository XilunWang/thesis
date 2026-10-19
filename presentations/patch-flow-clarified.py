import zipfile,xml.etree.ElementTree as E,posixpath
from pathlib import Path
B=Path('presentations/.build/flow-clarified');P='http://schemas.openxmlformats.org/presentationml/2006/main';A='http://schemas.openxmlformats.org/drawingml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
for key,val in [('p',P),('a',A),('r',R)]:E.register_namespace(key,val)
with zipfile.ZipFile(B/'original.pptx') as z,zipfile.ZipFile(B/'authored.pptx') as n:
 data={x:z.read(x) for x in z.namelist()};sn='ppt/slides/slide13.xml';root=E.fromstring(data[sn]);tree=root.find(f'.//{{{P}}}spTree')
 for el in list(tree):
  off=el.find(f'.//{{{A}}}xfrm/{{{A}}}off')
  if off is not None and int(off.get('x'))>=180*9525 and (int(off.get('y'))>=100*9525 or el.tag==f'{{{P}}}sp'):tree.remove(el)
 idx=max(int(t.get('id')) for t in root.iter(f'{{{P}}}cNvPr'))+1
 src=E.fromstring(n.read(sn)).find(f'.//{{{P}}}spTree');added=[];ids={}
 for el in src:
  props=el.find(f'.//{{{P}}}cNvPr')
  if props is not None and props.get('name','').startswith('flow-'):
   ids[props.get('id')]=str(idx);props.set('id',str(idx));idx+=1;added.append(el)
 for el in added:
  for tag in ['stCxn','endCxn']:
   for conn in el.iter(f'{{{A}}}{tag}'):
    if conn.get('id') in ids:conn.set('id',ids[conn.get('id')])
  tree.append(el)
 data[sn]=E.tostring(root,encoding='utf-8',xml_declaration=True)
 def note(zz):
  rr=E.fromstring(zz.read('ppt/slides/_rels/slide13.xml.rels'));t=next(t for t in rr if t.get('Type').endswith('/notesSlide')).get('Target');return posixpath.normpath(posixpath.join('ppt/slides',t)).lstrip('/')
 nt=note(z);data[nt]=n.read(note(n))
 with zipfile.ZipFile(B/'candidate.pptx','w',zipfile.ZIP_DEFLATED) as out:
  for k,v in data.items():out.writestr(k,v)
 assert all(data[k]==z.read(k) for k in z.namelist() if k not in [sn,nt])
 print('ONLY_CHANGED',sn,nt,'NATIVE_OBJECTS',len(added))
