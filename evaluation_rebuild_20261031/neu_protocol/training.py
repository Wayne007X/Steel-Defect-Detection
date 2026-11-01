from __future__ import annotations
from copy import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import shutil
import time
import urllib.request
from .core import require,write,load,sha,digest,utc,Budget,configure_runtime,ProtocolError


def environment_record():
    versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','ultralytics','numpy','opencv-python','onnxruntime']}
    base=Path(importlib.util.find_spec('ultralytics').origin).parent
    files={name:sha(base/name) for name in ['engine/trainer.py','engine/validator.py','models/yolo/detect/train.py',
             'models/yolo/detect/val.py','utils/metrics.py','utils/nms.py','data/augment.py','data/build.py','data/utils.py','nn/tasks.py']}
    return dict(versions=versions,ultralytics_implementation=files)


def check_environment(p):
    r=environment_record()
    for name,version in p.config['environment'].items():require(r['versions'][name]==version,f'Pinned {name} version differs')
    frozen=p.file('state/environment.json')
    if frozen.exists():require(r==load(frozen),'Training environment changed after freeze')
    return r


def prepare_weights(p):
    configure_runtime(p);check_environment(p)
    destination=p.file('models/yolov8n_coco.pt');record=p.file('models/official_provenance.json')
    if record.exists():
        r=load(record);require(sha(destination)==r['sha256'],'Official checkpoint changed');return r
    require(not p.file('freezes/split.json').exists(),'Cannot acquire new initial weights after freeze')
    destination.parent.mkdir(parents=True,exist_ok=True);download=destination.with_suffix('.download')
    url=p.config['weights_url']
    with urllib.request.urlopen(url,timeout=60) as response,download.open('wb') as f:
        shutil.copyfileobj(response,f)
    require(1_000_000<download.stat().st_size<30_000_000,'Unexpected official asset size')
    import torch
    ckpt=torch.load(download,map_location='cpu',weights_only=False)
    model=ckpt.get('ema') or ckpt.get('model');args=ckpt.get('train_args',{})
    require(model is not None and len(model.names)==80 and str(args.get('data','')).endswith('coco.yaml'),
            'Official asset is not an 80-class COCO initialization')
    download.replace(destination)
    local=Path('E:/yolo_project/bt_demo/yolov8n.pt')
    r=dict(url=url,sha256=sha(destination),classes=80,data=args['data'],downloaded_at=utc(),
           existing_local_matches=local.exists() and sha(local)==sha(destination))
    write(record,r,exclusive=True);return r


class StrictEarlyStopping:
    def __init__(self,patience=20):
        self.best_fitness=-math.inf;self.best_epoch=0;self.patience=patience or math.inf;self.possible_stop=False

    def __call__(self,epoch,fitness):
        if fitness is None:return False
        if fitness>self.best_fitness:self.best_fitness=float(fitness);self.best_epoch=epoch
        delta=epoch-self.best_epoch;self.possible_stop=delta>=self.patience-1
        return delta>=self.patience


def trainer_type(p,budget):
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer,DetectionValidator
    from ultralytics.data import build_yolo_dataset
    from ultralytics.utils.torch_utils import unwrap_model

    class AP50Validator(DetectionValidator):
        def preprocess(self,batch):
            result=super().preprocess(batch)
            require(result['img'].dtype==torch.float32 and tuple(result['img'].shape[-2:])==(640,640),
                    'Training validation must be FP32 square 640')
            return result

        def get_stats(self):
            r=super().get_stats();r['fitness']=float(r['metrics/mAP50(B)']);return r

    class AP50Trainer(DetectionTrainer):
        _new_ap50_best=False
        _budget_interrupted=False

        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.add_callback('on_train_start',lambda t:require(bool(t.amp) and t.device.type=='cuda','Pinned CUDA/AMP training was changed by the framework'))
            self.add_callback('on_train_batch_start',lambda t:require(t.batch_size==p.config['common_train']['batch'],'Automatic batch change is prohibited; inspect OOM instead'))
            self.add_callback('on_train_batch_end',self._budget_tick)
            self.add_callback('on_val_batch_end',lambda _:budget.check())

        def _setup_train(self,*args,**kwargs):
            super()._setup_train(*args,**kwargs)
            old=self.stopper;self.stopper=StrictEarlyStopping(self.args.patience)
            if self.resume:self.stopper.__dict__.update(old.__dict__)

        def _budget_tick(self,trainer):
            if budget.remaining(training=True)<=0:
                self._budget_interrupted=True;self.stop=True
            budget.check()

        def build_dataset(self,img_path,mode='train',batch=None):
            stride=max(int(unwrap_model(self.model).stride.max()),32)
            return build_yolo_dataset(self.args,img_path,batch,self.data,mode=mode,rect=False,stride=stride)

        def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
            return super().get_dataloader(dataset_path,1 if mode=='val' else batch_size,rank,mode)

        def get_validator(self):
            args=copy(self.args);args.rect=False;args.quantize=None;args.batch=1
            return AP50Validator(self.test_loader,save_dir=self.save_dir,args=args,_callbacks=self.callbacks)

        def validate(self):
            old_best=self.best_fitness;amp=self.amp
            try:
                self.amp=False
                metrics,fitness=super().validate()
            finally:self.amp=amp
            require(fitness is not None and math.isfinite(fitness),'Nonfinite Val AP50')
            self._new_ap50_best=old_best is None or fitness>old_best
            if self._new_ap50_best:
                write(self.save_dir/'ap50_best.json',dict(epoch=self.epoch+1,AP50=float(fitness),
                      AP95=float(metrics['metrics/mAP50-95(B)']),precision='float32',rect=False,batch=1))
            return metrics,fitness

        def save_model(self):
            # BaseTrainer writes best.pt even on equal fitness. Redirect only
            # that equal-score write to last.pt; resume state remains truthful.
            original=self.best
            if not self._new_ap50_best:self.best=self.last
            try:return super().save_model()
            finally:self.best=original

        def final_eval(self):
            # Keep resumable optimizer/scaler state when budget, not early stop,
            # ended a task. Never mark such a task completed.
            if self._budget_interrupted:return
            self.validator.args.quantize=None;self.validator.args.rect=False
            return super().final_eval()
    return AP50Trainer


def recipe(p,family,seed):
    c=p.config;fine=family in ['C','D']
    args={**c['common_train'],**c['standard_aug'],**(c['finetune'] if fine else c['base_train'])}
    if family=='C':args.update(c['texture_aug'])
    args.update(seed=seed,mosaic=1.0 if family=='A' else 0.0,close_mosaic=10 if family=='A' else 0,
                data=str(p.file('dataset/development.yaml')),project=str(p.file('runs')),name=f'{family}_seed{seed}',
                exist_ok=False,resume=False)
    return args


def train_all(p,resume=False):
    freeze=p.split_freeze();configure_runtime(p);check_environment(p)
    import torch
    from ultralytics import YOLO
    budget=Budget(p)
    for family in p.config['families']:
        for seed in p.config['seeds']:
            name=f'{family}_seed{seed}';run=p.file('runs/'+name);done=run/'completion.json'
            if done.exists():
                r=load(done);require(r['split_freeze']==freeze['id'] and sha(p.file(r['best']))==r['best_sha256'],f'Changed completed job {name}')
                continue
            initial=p.file(f'runs/B_seed{seed}/weights/best.pt') if family in ['C','D'] else p.file('models/yolov8n_coco.pt')
            if family in ['C','D']:
                parent=load(p.file(f'runs/B_seed{seed}/completion.json'))
                require(sha(initial)==parent['best_sha256'],'B initialization checkpoint changed')
            init_hash=sha(initial);job_args=recipe(p,family,seed)
            if run.exists():require(resume and (run/'weights/last.pt').exists(),f'Incomplete job {name}; inspect then use train --resume')
            with budget.job('train:'+name,training=True):
                Trainer=trainer_type(p,budget)
                if run.exists():
                    before=load(run/'initialization.json')
                    require(before['initialization_sha256']==init_hash and before['split_freeze']==freeze['id'],'Resume provenance differs')
                    model=YOLO(str(run/'weights/last.pt'));model.train(trainer=Trainer,resume=True)
                else:
                    model=YOLO(str(initial))
                    def provenance(trainer):
                        write(trainer.save_dir/'initialization.json',dict(family=family,seed=seed,initialization=str(initial.relative_to(p.root)),
                             initialization_sha256=init_hash,split_freeze=freeze['id'],args=job_args),exclusive=True)
                    model.add_callback('on_pretrain_routine_start',provenance)
                    model.train(trainer=Trainer,**job_args)
                trainer=model.trainer
                require(not trainer._budget_interrupted,'Training budget reached; task checkpoint saved but task is incomplete')
                best=run/'weights/best.pt';require(best.exists(),'No validated best checkpoint')
                info=load(run/'initialization.json')
                info.update(best=str(best.relative_to(p.root)).replace('\\','/'),best_sha256=sha(best),
                            best_epoch=load(run/'ap50_best.json'),finished_at=utc())
                write(done,info,exclusive=True)
                del model,trainer;gc.collect();torch.cuda.empty_cache()
    return p.completed_jobs()
