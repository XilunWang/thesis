import zipfile,xml.etree.ElementTree as E,posixpath
from pathlib import Path
src=Path(r'C:\Users\Lenovo\Desktop\左怀琪-专业学位硕士开题PPT.pptx');b=Path('presentations/.build/ablation-table')
P='http://schemas.openxmlformats.org/presentationml/2006/main';A='http://schemas.openxmlformats.org/drawingml/2006/main';R='http://schemas.openxmlformats.org/package/2006/relationships';CT='http://schemas.openxmlformats.org/package/2006/content-types';O='http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
E.register_namespace('p',P);E.register_namespace('a',A);E.register_namespace('r',O.rstrip('/'));E.register_namespace('p14','http://schemas.microsoft.com/office/powerpoint/2010/main');E.register_namespace('a14','http://schemas.microsoft.com/office/drawing/2010/main');E.register_namespace('a16','http://schemas.microsoft.com/office/drawing/2014/main');E.register_namespace('mc','http://schemas.openxmlformats.org/markup-compatibility/2006')
with zipfile.ZipFile(src) as z,zipfile.ZipFile(b/'authored.pptx') as n:
 data={x:z.read(x) for x in z.namelist()};r=E.fromstring(data['ppt/slides/slide10.xml']);tree=r.find(f'.//{{{P}}}spTree');new=E.fromstring(n.read('ppt/slides/slide10.xml')).find(f'.//{{{P}}}spTree')
 ids=[int(x.get('id')) for x in r.iter(f'{{{P}}}cNvPr')];k=max(ids)+1
 for item in list(new)[-4:]:
  for tag in item.iter(f'{{{P}}}cNvPr'):tag.set('id',str(k));k+=1
  tree.append(item)
 data['ppt/slides/slide10.xml']=E.tostring(r,encoding='utf-8',xml_declaration=True)
 relname='ppt/slides/_rels/slide10.xml.rels';rels=E.fromstring(data[relname]);newrels=E.fromstring(n.read(relname));nr=next(x for x in newrels if x.get('Type').endswith('/notesSlide'));nt=posixpath.normpath(posixpath.join('ppt/slides',nr.get('Target')))
 data['ppt/notesSlides/notesSlide99.xml']=n.read(nt.lstrip('/'))
 E.SubElement(rels,f'{{{R}}}Relationship',Id='rIdNotes99',Type=O+'notesSlide',Target='../notesSlides/notesSlide99.xml');data[relname]=E.tostring(rels,encoding='utf-8',xml_declaration=True)
 rr=E.Element(f'{{{R}}}Relationships');E.SubElement(rr,f'{{{R}}}Relationship',Id='rId1',Type=O+'notesMaster',Target='../notesMasters/notesMaster1.xml');E.SubElement(rr,f'{{{R}}}Relationship',Id='rId2',Type=O+'slide',Target='../slides/slide10.xml');data['ppt/notesSlides/_rels/notesSlide99.xml.rels']=E.tostring(rr,encoding='utf-8',xml_declaration=True)
 E.register_namespace('',CT);ct=E.fromstring(data['[Content_Types].xml']);E.SubElement(ct,f'{{{CT}}}Override',PartName='/ppt/notesSlides/notesSlide99.xml',ContentType='application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml');data['[Content_Types].xml']=E.tostring(ct,encoding='utf-8',xml_declaration=True)
 with zipfile.ZipFile(b/'candidate.pptx','w',zipfile.ZIP_DEFLATED) as out:
  for key,val in data.items():out.writestr(key,val)
 assert all(data[x]==z.read(x) for x in z.namelist() if x not in ['ppt/slides/slide10.xml',relname,'[Content_Types].xml'])
 print('OTHER_ORIGINAL_PARTS_UNCHANGED')
