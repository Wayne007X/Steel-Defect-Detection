from __future__ import annotations
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .core import require,load,write,sha,digest,utc,ProtocolError


def ssim(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64)
    blur=lambda x:cv2.GaussianBlur(x,(11,11),1.5)[5:-5,5:-5]
    ma,mb=blur(a),blur(b)
    va,vb,cov=blur(a*a)-ma*ma,blur(b*b)-mb*mb,blur(a*b)-ma*mb
    return float((((2*ma*mb+6.5025)*(2*cov+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))).mean())


def phash(image):
    v=cv2.dct(cv2.resize(image,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].ravel()[1:]
    return v>np.median(v)


def audit(p):
    out=p.file('audit/manifest.json')
    if out.exists():
        rows=p.source_integrity();summary=load(p.file('audit/summary.json'))
        summary.update(object_distribution(p,rows))
        write(p.file('audit/summary.json'),summary)
        return summary
    src=Path(p.config['source']);names=p.config['names'];rows=[];images=[];hashes=[]
    sources=[]
    for split in ['train','validation']:
        paths=[x for x in (src/split/'images').rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}]
        corresponding={(src/split/'labels'/x.relative_to(src/split/'images')).with_suffix('.txt') for x in paths}
        require(corresponding==set((src/split/'labels').rglob('*.txt')),f'Orphan or missing label files in {split}')
        sources.extend((x,split) for x in paths)
    def key(item):
        image,_=item
        require(image.parent.name in names,f'Unknown image class: {image}')
        return names.index(image.parent.name),int(image.stem.rsplit('_',1)[1])
    sources.sort(key=key)
    seen=set()
    for image,split in sources:
        c=names.index(image.parent.name);identity=f'{image.parent.name}/{image.name}'
        require(identity not in seen,f'Duplicate image identity: {identity}');seen.add(identity)
        label=(src/split/'labels'/image.relative_to(src/split/'images')).with_suffix('.txt')
        xml=src/split/'annotations'/(image.stem+'.xml')
        require(label.exists() and xml.exists(),f'Missing label/XML for {image}')
        gray=cv2.imread(str(image),cv2.IMREAD_GRAYSCALE)
        require(gray is not None and gray.shape==(200,200),f'Invalid source image {image}')
        labels=[]
        for line in label.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            r=list(map(float,line.split()))
            require(len(r)==5 and np.isfinite(r).all() and r[0]==int(r[0]) and 0<=r[0]<6,f'Invalid class/label {label}')
            _,x,y,w,h=r
            require(w>0 and h>0 and min(x-w/2,y-h/2)>=-1e-5 and max(x+w/2,y+h/2)<=1+1e-5,f'Invalid box {label}')
            labels.append(r)
        require(labels,f'Unexpected empty source label: {label}')
        tree=ET.parse(xml).getroot();w=int(tree.findtext('size/width'));h=int(tree.findtext('size/height'))
        require((h,w)==gray.shape,f'XML dimensions differ: {xml}')
        expected=[]
        for obj in tree.findall('object'):
            box=obj.find('bndbox');x1,y1,x2,y2=[float(box.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
            expected.append([names.index(obj.findtext('name').strip()),(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h])
        require(len(expected)==len(labels) and np.allclose(sorted(expected),sorted(labels),atol=1e-6,rtol=0),f'XML/YOLO mismatch {xml}')
        image_rel='dataset/images/'+identity;label_rel='dataset/labels/'+str(Path(identity).with_suffix('.txt')).replace('\\','/')
        for source,relative in [(image,image_rel),(label,label_rel)]:
            target=p.file(relative);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():require(sha(source)==sha(target),f'Canonical copy changed: {target}')
            else:shutil.copy2(source,target)
        with Image.open(image) as im:
            rgb=np.asarray(im.convert('RGB'))
        pixel_sha=hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
        rows.append(dict(id=identity,class_id=c,image=image_rel,label=label_rel,source_image=str(image),source_label=str(label),
                         source_xml=str(xml),xml_sha256=sha(xml),old_split=split,image_sha256=sha(image),label_sha256=sha(label),
                         pixel_sha256=pixel_sha,boxes=len(labels)))
        images.append(gray);hashes.append(phash(gray))
    require(len(rows)==1800 and all(sum(r['class_id']==c for r in rows)==300 for c in range(6)),'Expected 1800 images, 300/class')
    hashes=np.array(hashes);pairs=[]
    for i in range(len(rows)):
        distances=np.count_nonzero(hashes[i+1:]!=hashes[i],axis=1)
        for j in np.flatnonzero(distances<=p.config['candidate_hamming'])+i+1:
            pairs.append(dict(a=rows[i]['id'],b=rows[j]['id'],hamming=int(distances[j-i-1]),
                              ssim=ssim(images[i],images[j]),exact_pixels=rows[i]['pixel_sha256']==rows[j]['pixel_sha256']))
    # Exact duplicates are mandatory even if a future hash implementation changes.
    groups=defaultdict(list)
    for r in rows:groups[r['pixel_sha256']].append(r['id'])
    require(all(any({x,y}=={v['a'],v['b']} for v in pairs) for g in groups.values() for x in g for y in g if x!=y),'Exact pair missing from review')
    pairs.sort(key=lambda r:(r['hamming'],r['a'],r['b']))
    for i,pair in enumerate(pairs,1):pair['pair_id']=f'P{i:03}'
    write(out,rows,exclusive=True);write(p.file('audit/candidates.json'),pairs,exclusive=True)
    summary=dict(images=len(rows),boxes=sum(r['boxes'] for r in rows),classes=dict(Counter(names[r['class_id']] for r in rows)),
                 unique_pixel_images=len(groups),exact_duplicate_groups=[g for g in groups.values() if len(g)>1],
                 candidate_pairs=len(pairs),candidate_hamming=p.config['candidate_hamming'],xml_checked=len(rows),
                 source=p.config['source'],disclosure=p.config['disclosure'])
    write(p.file('audit/summary.json'),summary,exclusive=True)
    template=dict(reviewer='',reviewed_at='',note='same/different/uncertain; exact duplicates must remain same',
                  candidates_sha256=sha(p.file('audit/candidates.json')),
                  decisions=[dict(pair_id=r['pair_id'],decision='same' if r['exact_pixels'] else None,reason='exact pixels' if r['exact_pixels'] else '') for r in pairs])
    write(p.file('review/decisions_template.json'),template,exclusive=True)
    render_review(p,rows,pairs)
    summary.update(object_distribution(p,rows));write(p.file('audit/summary.json'),summary)
    return summary


def object_distribution(p,rows):
    counts=Counter();effective=Counter();mixed=[];duplicates=[]
    for r in rows:
        values=np.asarray([list(map(float,line.split())) for line in p.file(r['label']).read_text(encoding='utf-8-sig').splitlines() if line.strip()],dtype=np.float32)
        classes=[int(v) for v in values[:,0]];unique=np.unique(values,axis=0)
        counts.update(p.config['names'][c] for c in classes)
        effective.update(p.config['names'][int(c)] for c in unique[:,0])
        if len(unique)<len(values):duplicates.append(dict(id=r['id'],raw=len(values),effective=len(unique),removed_in_memory=len(values)-len(unique)))
        if any(c!=r['class_id'] for c in classes):mixed.append(r['id'])
    return dict(object_classes=dict(counts),effective_object_classes=dict(effective),duplicate_label_rows=duplicates,multi_class_images=mixed,
                stratification='Primary image category from source identity; object labels may include multiple classes and are preserved.')


def render_review(p,rows,pairs):
    lookup={r['id']:r for r in rows};panels=[]
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    font=lambda n:ImageFont.truetype(str(font_path),n) if font_path.exists() else ImageFont.load_default()
    lines=['# 相似性分组人工复核材料','','这些是原图相似候选，不是模型预测或已确认的钢板来源组。',
           '请复制 decisions_template.json 为 decisions.json，填写 reviewer、reviewed_at 与每对的 decision/reason。',
           'decision 只能为 same（同源或近重复）、different（不同）、uncertain（无法确定，保守同组）。完全重复对固定 same。',
           '请不要改动图片、标签或协议。SSIM 与 pHash 只辅助检查，不代表已经证明同钢板。','']
    for pair in pairs:
        canvas=Image.new('RGB',(1200,485),'white');draw=ImageDraw.Draw(canvas)
        title=f"{pair['pair_id']}  pHash={pair['hamming']}  SSIM={pair['ssim']:.5f}  exact={pair['exact_pixels']}"
        draw.text((12,8),title,fill='black',font=font(22))
        a=Image.open(p.file(lookup[pair['a']]['image'])).convert('RGB')
        b=Image.open(p.file(lookup[pair['b']]['image'])).convert('RGB')
        diff=Image.fromarray(cv2.absdiff(np.array(a),np.array(b)))
        for i,(im,label) in enumerate([(a,pair['a']),(b,pair['b']),(diff,'Absolute pixel difference')]):
            draw.text((i*400+8,45),label,fill='black',font=font(17))
            canvas.paste(im.resize((390,390),Image.Resampling.NEAREST),(i*400+5,75))
        path=p.file('review/pairs/'+pair['pair_id']+'.jpg');path.parent.mkdir(parents=True,exist_ok=True)
        canvas.save(path,quality=94);panels.append(canvas)
        lines.extend([f"## {pair['pair_id']}：{pair['a']} / {pair['b']}",f"pHash={pair['hamming']}；SSIM={pair['ssim']:.5f}；完全重复={pair['exact_pixels']}",
                      f"![{pair['pair_id']}](pairs/{pair['pair_id']}.jpg)",''])
    for start in range(0,len(panels),10):
        sheet=Image.new('RGB',(1200,1215),'#eeeeee')
        for j,panel in enumerate(panels[start:start+10]):sheet.paste(panel.resize((600,243),Image.Resampling.LANCZOS),((j%2)*600,(j//2)*243))
        sheet.save(p.file(f'review/contact_{start//10+1:02}.jpg'),quality=94)
    p.file('review/人工分组复核.md').write_text('\n'.join(lines),encoding='utf-8')


def import_review(p,path):
    require(not p.file('freezes/split.json').exists(),'Split already frozen')
    value=load(path);pairs=load(p.file('audit/candidates.json'))
    require(value.get('reviewer','').strip() and value.get('reviewed_at','').strip(),'Human reviewer and review date are required')
    require(value['candidates_sha256']==sha(p.file('audit/candidates.json')),'Review belongs to different candidate list')
    decisions=value['decisions'];require(len(decisions)==len(pairs),'All candidate pairs must be reviewed')
    mapping={r['pair_id']:r for r in decisions};require(len(mapping)==len(pairs),'Duplicate decision IDs')
    for r in pairs:
        require(r['pair_id'] in mapping,f"Missing decision {r['pair_id']}")
        d=mapping[r['pair_id']]
        require(d['decision'] in ['same','different','uncertain'] and d.get('reason','').strip(),f"Unreviewed pair {r['pair_id']}")
        require(not r['exact_pixels'] or d['decision']=='same','Exact pixel duplicates cannot be split')
    # Reject contradictory triangles before persisting a human approval, so a
    # draft can be corrected without an already-imported inconsistent record.
    make_groups(load(p.file('audit/manifest.json')),pairs,value)
    write(p.file('review/approved_decisions.json'),value,exclusive=True)
    return value


def make_groups(rows,pairs,review):
    parents={r['id']:r['id'] for r in rows}
    def find(x):
        while parents[x]!=x:parents[x]=parents[parents[x]];x=parents[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    decisions={r['pair_id']:r for r in review['decisions']}
    for pair in pairs:
        if pair['exact_pixels'] or decisions[pair['pair_id']]['decision'] in ['same','uncertain']:union(pair['a'],pair['b'])
    for pair in pairs:
        require(decisions[pair['pair_id']]['decision']!='different' or find(pair['a'])!=find(pair['b']),
                f"Contradictory transitive group decisions: {pair['pair_id']}")
    groups=defaultdict(list)
    for row in rows:groups[find(row['id'])].append(row['id'])
    return [{'group_id':digest(sorted(members))[:20],'members':sorted(members)} for _,members in sorted(groups.items())]


def propose_split(p):
    require(not p.file('freezes/split.json').exists(),'Split already frozen')
    if p.file('split/proposal.json').exists():
        old=load(p.file('split/proposal.json'));p.check_files(old)
        require(old['code']==p.code_hashes(),'Code changed since proposal; create a new protocol version')
        return old
    require(p.file('review/approved_decisions.json').exists(),'BLOCKED: all human grouping decisions are required')
    rows=p.source_integrity();pairs=load(p.file('audit/candidates.json'));review=load(p.file('review/approved_decisions.json'))
    split_implementation=p.file('tools/split_deps/sklearn/model_selection/_split.py')
    require(split_implementation.exists(),'Install the fixed isolated splitter dependencies first')
    groups=make_groups(rows,pairs,review);by_id={m:g['group_id'] for g in groups for m in g['members']}
    payload=dict(y=[r['class_id'] for r in rows],groups=[by_id[r['id']] for r in rows],splitter=p.config['splitter'])
    result=subprocess.run([sys.executable,str(p.file('neu_protocol/split_worker.py')),str(p.file('tools/split_deps'))],
                          input=json.dumps(payload),capture_output=True,text=True,encoding='utf-8')
    require(result.returncode==0,'Split worker failed (install isolated split dependencies first): '+result.stderr[-2000:])
    folds=json.loads(result.stdout)['folds']
    lists={s:[] for s in ['train','val','test']}
    for row,fold in zip(rows,folds):lists['test' if fold==0 else 'val' if fold==1 else 'train'].append(row)
    verify_partition(rows,groups,lists)
    write(p.file('split/groups.json'),groups,exclusive=True)
    write(p.file('split/assignment.json'),{s:[r['id'] for r in v] for s,v in lists.items()},exclusive=True)
    hashes={}
    for split,items in lists.items():
        path=p.file('dataset/'+split+'.txt')
        with path.open('x',encoding='utf-8',newline='\n') as f:f.write(''.join('./images/'+r['id']+'\n' for r in items))
        hashes[split]=dict(sha256=sha(path),md5=hashlib.md5(path.read_bytes()).hexdigest())
    import yaml
    p.file('dataset/development.yaml').write_text(yaml.safe_dump(dict(path=str(p.file('dataset')),train='train.txt',val='val.txt',nc=6,names=p.config['names']),sort_keys=False),encoding='utf-8')
    summary=dict(counts={s:len(v) for s,v in lists.items()},classes={s:dict(Counter(p.config['names'][r['class_id']] for r in v)) for s,v in lists.items()},
                 objects={s:object_distribution(p,v)['object_classes'] for s,v in lists.items()},
                 effective_objects={s:object_distribution(p,v)['effective_object_classes'] for s,v in lists.items()},
                 groups=len(groups),non_singleton_groups=[g for g in groups if len(g['members'])>1],hashes=hashes,
                 reviewer=review['reviewer'],splitter=p.config['splitter'])
    for split,counts in summary['objects'].items():require(set(counts)==set(p.config['names']),f'Object class missing in {split}')
    files=p.fingerprint([p.file('protocol.json'),p.file('audit/manifest.json'),p.file('audit/candidates.json'),p.file('audit/summary.json'),p.file('review/approved_decisions.json'),
                         split_implementation,p.file('requirements-split.txt'),
                         p.file('split/groups.json'),p.file('split/assignment.json'),p.file('dataset/development.yaml'),
                         *[p.file('dataset/'+s+'.txt') for s in lists]])
    proposal=dict(summary=summary,files=files,code=p.code_hashes());proposal['id']=digest(proposal)
    write(p.file('split/proposal.json'),proposal,exclusive=True)
    lines=['# 划分审计：等待人工确认','',f"Proposal SHA256: `{proposal['id']}`",'',
           'Train/Val/Test 只按分组和类别生成，未运行模型，也未按分数调整划分。','',json.dumps(summary,ensure_ascii=False,indent=2),
           '', '非单例组已完整列出，请复核链式合并后成员；确认后使用 approve-split 命令及上述 SHA256。']
    p.file('split/划分审计.md').write_text('\n'.join(lines),encoding='utf-8')
    return proposal


def verify_partition(rows,groups,lists):
    expected={r['id'] for r in rows};seen=set();group_lookup={m:g['group_id'] for g in groups for m in g['members']};used={}
    for split,items in lists.items():
        ids={r['id'] for r in items};require(not seen&ids,f'Image overlap {split}');seen|=ids
        require({r['class_id'] for r in items}==set(range(6)),f'Missing class in {split}')
        for row in items:
            g=group_lookup[row['id']];require(g not in used or used[g]==split,'Group crosses splits');used[g]=split
    require(seen==expected and sum(len(v) for v in lists.values())==len(rows),'Missing/duplicate split rows')


def approve_split(p,proposal_id,reviewer):
    require(reviewer.strip(),'Reviewer required')
    r=load(p.file('split/proposal.json'));require(r['id']==proposal_id,'Specify the exact reviewed proposal SHA256')
    require(r['id']==digest({k:v for k,v in r.items() if k!='id'}),'Invalid proposal digest')
    p.check_files(r);require(r['code']==p.code_hashes(),'Code changed since proposal')
    p.source_integrity()
    require(p.file('models/official_provenance.json').exists(),'Official pretrained weights must be verified before freeze')
    from .training import environment_record
    write(p.file('state/environment.json'),environment_record())
    weights=load(p.file('models/official_provenance.json'))
    require(sha(p.file('models/yolov8n_coco.pt'))==weights['sha256'],'Official weight hash mismatch')
    record=dict(proposal=proposal_id,reviewer=reviewer,approved_at=utc(),files={**r['files'],**p.fingerprint([
        p.file('models/yolov8n_coco.pt'),p.file('models/official_provenance.json'),p.file('state/environment.json')])},code=p.code_hashes())
    record['id']=digest(record);write(p.file('freezes/split.json'),record,exclusive=True)
    return record
