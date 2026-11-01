"""Native validation and real inference share FP32 square preprocessing/NMS."""
from __future__ import annotations
import gc
import time
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ap_per_class
from .core import require,write,load,sha,digest,Budget
from .fusion_reference import PrecisionDetector,fuse


class CaptureValidator(DetectionValidator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.records=[]

    def update_metrics(self,preds,batch):
        require(batch['img'].dtype==torch.float32 and tuple(batch['img'].shape[-2:])==(640,640),'Validation dtype/shape changed')
        for i,pred in enumerate(preds):
            target=self._prepare_batch(i,batch)
            self.records.append(dict(path=batch['im_file'][i],input_shape=tuple(batch['img'].shape[-2:]),
                boxes=pred['bboxes'].cpu().numpy(),conf=pred['conf'].cpu().numpy(),cls=pred['cls'].cpu().numpy(),
                gt_boxes=target['bboxes'].cpu().numpy(),gt_cls=target['cls'].cpu().numpy()))
        return super().update_metrics(preds,batch)

    def get_stats(self):
        np.save(self.save_dir/'records.npy',np.array(self.records,dtype=object),allow_pickle=True)
        return super().get_stats()


def subset(r,keep):
    return {**r,**{k:r[k][keep] for k in ['boxes','conf','cls']}}


def postprocess(r,profile):
    kept=[]
    for c,threshold in enumerate(profile):
        ids=np.flatnonzero(r['cls']==c)
        if threshold<.7 and len(ids):
            local=batched_nms(torch.from_numpy(r['boxes'][ids]),torch.from_numpy(r['conf'][ids]),
                              torch.from_numpy(r['cls'][ids]).long(),threshold).numpy()
            ids=ids[local]
        kept.extend(ids.tolist())
    ids=np.asarray(kept,dtype=int)
    ids=ids[np.argsort(-r['conf'][ids],kind='stable')]
    return subset(r,ids)


def combine(parts,profile):
    result=[postprocess(r,profile) for r in parts]
    if len(result)==1:return result[0]
    require(len(result)==2 and result[0]['path']==result[1]['path'],'Prediction streams are not aligned')
    require(np.allclose(result[0]['gt_boxes'],result[1]['gt_boxes'],atol=1e-4),'Target coordinates differ')
    return fuse(*result)


def statistics(records):
    # Matching needs only these attributes. Avoid the validator constructor,
    # which would create a default runs/detect directory in the caller's cwd.
    matcher=DetectionValidator.__new__(DetectionValidator)
    matcher.iouv=torch.linspace(.5,.95,10);matcher.niou=10;parts=[]
    for r in records:
        require(np.all(r['conf'][:-1]>=r['conf'][1:]),'Predictions must be descending confidence')
        tp=matcher._process_batch(dict(bboxes=torch.from_numpy(r['boxes']),cls=torch.from_numpy(r['cls'])),
                                  dict(bboxes=torch.from_numpy(r['gt_boxes']),cls=torch.from_numpy(r['gt_cls'])))['tp']
        parts.append((tp,r['conf'],r['cls'],r['gt_cls']))
    return [np.concatenate([v[i] for v in parts]) for i in range(4)]


def measure(records,names,thresholds):
    tp,conf,classes,targets=statistics(records)
    result=ap_per_class(tp,conf,classes,targets,names=dict(enumerate(names)))
    ap,ids=result[5:7]
    per_class=np.zeros((6,10));per_class[ids]=ap
    points=[]
    for threshold in thresholds:
        keep=conf>=threshold;t=int(tp[keep,0].sum());fp=int(keep.sum())-t;fn=len(targets)-t
        precision=t/max(t+fp,1);recall=t/max(t+fn,1)
        rows=[]
        for c in range(6):
            mask=keep&(classes==c);ct=int(tp[mask,0].sum());cf=int(mask.sum())-ct;cn=int((targets==c).sum())-ct
            rows.append(dict(class_id=c,name=names[c],TP=ct,FP=cf,FN=cn,
                             precision=ct/max(ct+cf,1),recall=ct/max(ct+cn,1),F1=2*ct/max(2*ct+cf+cn,1)))
        points.append(dict(threshold=threshold,TP=t,FP=fp,FN=fn,precision=precision,recall=recall,
                           F1=2*t/max(2*t+fp+fn,1),per_class=rows))
    return dict(AP50=float(per_class[:,0].mean()),AP95=float(per_class.mean()),
                class_AP50=per_class[:,0].tolist(),class_AP95=per_class.mean(1).tolist(),points=points)


def native_capture(p,weight,split,tag,tta=False):
    require(split in ['val','test'],'Unsupported split')
    if split=='test':p.test_guard()
    else:p.split_freeze(full=False)
    cache=p.file(f'cache/{split}/{tag}');meta=cache/'complete.json'
    key=dict(weight_sha256=sha(weight),split=split,split_sha256=sha(p.file('dataset/'+split+'.txt')),tta=tta,
             code=p.code_hashes(),FP32=True,rect=False,batch=1)
    if meta.exists():
        info=load(meta);require(info['key']==key and sha(cache/'records.npy')==info['records_sha256'],'Cached predictions changed')
        return np.load(cache/'records.npy',allow_pickle=True).tolist()
    data=p.file('dataset/development.yaml')
    if split=='test':
        import yaml
        data=p.file('state/released_test.yaml')
        data.write_text(yaml.safe_dump(dict(path=str(p.file('dataset')),train='train.txt',val='val.txt',test='test.txt',nc=6,names=p.config['names']),sort_keys=False),encoding='utf-8')
    budget=Budget(p)
    with budget.job(f'capture:{split}:{tag}'):
        model=YOLO(str(weight))
        require(model.names==dict(enumerate(p.config['names'])),'Checkpoint is not the expected six-class model')
        model.add_callback('on_val_batch_end',lambda _:budget.check())
        metric=model.val(validator=CaptureValidator,data=str(data),split=split,imgsz=640,batch=1,device=0,workers=0,
                         conf=.001,iou=.7,max_det=300,rect=False,quantize=None,augment=tta,plots=False,
                         save_json=False,save_txt=False,project=str(cache.parent),name=cache.name,exist_ok=True,verbose=False)
        records=np.load(cache/'records.npy',allow_pickle=True).tolist()
        required=load(p.file('split/assignment.json'))[split]
        actual=[str(Path(r['path']).relative_to(p.file('dataset/images'))).replace('\\','/') for r in records]
        require(set(actual)==set(required) and len(actual)==len(required),'Validator used the wrong split')
        from collections import Counter
        expected_objects=load(p.file('split/proposal.json'))['summary']['effective_objects'][split]
        actual_objects=dict(Counter(p.config['names'][int(c)] for r in records for c in r['gt_cls']))
        require(actual_objects==expected_objects,'Native GT counts differ from the frozen duplicate-row loading rule')
        # Sorting aligns multiple models even if loader file sorting changes.
        records.sort(key=lambda r:r['path'])
        np.save(cache/'records.npy',np.array(records,dtype=object),allow_pickle=True)
        calculated=measure(records,p.config['names'],[.25])
        require(abs(calculated['AP50']-float(metric.box.map50))<1e-7 and abs(calculated['AP95']-float(metric.box.map))<1e-7,
                'Cached metric/native metric parity failed')
        write(meta,dict(key=key,records_sha256=sha(cache/'records.npy'),native_AP50=float(metric.box.map50),
                        native_AP95=float(metric.box.map),images=len(records)))
        del model;gc.collect();torch.cuda.empty_cache()
        return records


class RuntimeDetector(PrecisionDetector):
    """Single/double backend with the exact protocol class-specific NMS."""
    def __init__(self,weights,device='cuda:0',augment=False,profile=None):
        super().__init__(weights,device=device,augment=augment,class_nms=profile)

    @torch.inference_mode()
    def capture(self,path):
        _,r=self.records(path,iou=.7)
        return r


def actual_capture(p,weights,targets,tta,profile,device,split,budget=None):
    if split=='test':p.test_guard()
    require(split=='val' or split=='test','Invalid runtime split')
    detector=RuntimeDetector(weights,device=device,augment=tta,profile=profile)
    require(detector.names==dict(enumerate(p.config['names'])),'Unexpected class order')
    detector.capture(targets[0]['path']);latencies=[];records=[]
    for target in targets:
        if budget is not None:budget.check()
        if device!='cpu':torch.cuda.synchronize()
        started=time.perf_counter();r=detector.capture(target['path'])
        if device!='cpu':torch.cuda.synchronize()
        latencies.append(1000*(time.perf_counter()-started))
        records.append({**r,'gt_boxes':target['gt_boxes'],'gt_cls':target['gt_cls']})
    del detector;gc.collect();torch.cuda.empty_cache()
    return records,dict(median_ms=float(np.median(latencies)),p95_ms=float(np.percentile(latencies,95)),device=device,batch=1,
                       images=len(records),excludes='model load, warm-up, drawing, serialization and network')
