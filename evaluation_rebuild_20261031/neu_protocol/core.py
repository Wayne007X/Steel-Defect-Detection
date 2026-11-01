from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time


class ProtocolError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    else:
        temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(text,encoding='utf-8',newline='\n');temp.replace(path)


class Protocol:
    def __init__(self,root):
        self.root=Path(root).resolve();self.config=load(self.root/'protocol.json')

    def file(self,relative):
        return self.root/relative

    def fingerprint(self,paths):
        return {str(p.relative_to(self.root)).replace('\\','/'):sha(p) for p in sorted(paths) if p.is_file()}

    def code_hashes(self):
        return self.fingerprint([self.file('protocol.json'),self.file('run_protocol.py'),
                                 self.file('requirements-split.txt'),
                                 *self.file('neu_protocol').glob('*.py')])

    def source_integrity(self):
        rows=load(self.file('audit/manifest.json'))
        for row in rows:
            for key,hash_key in [('source_image','image_sha256'),('source_label','label_sha256'),('source_xml','xml_sha256')]:
                require(sha(row[key])==row[hash_key],f'Source changed: {row[key]}')
            for key,hash_key in [('image','image_sha256'),('label','label_sha256')]:
                require(sha(self.file(row[key]))==row[hash_key],f'Canonical data changed: {row[key]}')
        return rows

    def check_files(self,record):
        for path,expected in record['files'].items():
            require(self.file(path).is_file() and sha(self.file(path))==expected,f'Frozen file missing or changed: {path}')

    def split_freeze(self,full=True):
        path=self.file('freezes/split.json')
        require(path.exists(),'BLOCKED: human review and explicit split approval are required before training.')
        record=load(path);self.check_files(record)
        require(record['code']==self.code_hashes(),'Code/config changed after split freeze; create a new protocol version.')
        if full:self.source_integrity()
        require(record['id']==digest({k:v for k,v in record.items() if k!='id'}),'Invalid split freeze digest')
        return record

    def completed_jobs(self):
        freeze=self.split_freeze()
        jobs={}
        for family in self.config['families']:
            for seed in self.config['seeds']:
                name=f'{family}_seed{seed}';path=self.file(f'runs/{name}/completion.json')
                require(path.exists(),f'Incomplete required job: {name}')
                r=load(path);require(r['split_freeze']==freeze['id'],f'Wrong split provenance: {name}')
                require(r['family']==family and r['seed']==seed,f'Wrong job identity: {name}')
                require(r['best']==f'runs/{name}/weights/best.pt',f'Checkpoint outside managed job: {name}')
                from .training import recipe
                require(r['args']==recipe(self,family,seed),f'Training recipe changed: {name}')
                require(sha(self.file(r['best']))==r['best_sha256'],f'Checkpoint changed: {name}')
                jobs[name]=r
        generic=load(self.file('models/official_provenance.json'))
        for name,r in jobs.items():
            expected=(jobs[f"B_seed{r['seed']}"]['best_sha256'] if r['family'] in ['C','D'] else generic['sha256'])
            require(r['initialization_sha256']==expected,f'Invalid initialization: {name}')
            initial=(f"runs/B_seed{r['seed']}/weights/best.pt" if r['family'] in ['C','D'] else 'models/yolov8n_coco.pt')
            require(str(r['initialization']).replace('\\','/')==initial,f'Invalid initialization path: {name}')
        return jobs

    def final_freeze(self):
        self.completed_jobs()
        path=self.file('freezes/final.json');require(path.exists(),'BLOCKED: final Val selection/export freeze is missing.')
        r=load(path);self.check_files(r)
        require(r['split_freeze']==self.split_freeze(full=False)['id'],'Final freeze uses a different split')
        require(r['code']==self.code_hashes(),'Code changed after final freeze')
        require(r['id']==digest({k:v for k,v in r.items() if k!='id'}),'Invalid final freeze digest')
        return r

    def authorize_test(self,freeze_id,reviewer):
        r=self.final_freeze()
        require(reviewer.strip() and freeze_id==r['id'],'Test authorization must name reviewer and exact final freeze SHA256.')
        write(self.file('freezes/test_authorization.json'),dict(final_freeze=freeze_id,reviewer=reviewer,at=utc()),exclusive=True)

    def test_guard(self):
        r=self.final_freeze();path=self.file('freezes/test_authorization.json')
        require(path.exists(),'BLOCKED: the user must explicitly approve the final freeze before Test release.')
        require(load(path)['final_freeze']==r['id'],'Test approval refers to a different freeze')
        return r


class Budget:
    """Serial compute accounting. Incomplete jobs charge their recorded wall time conservatively."""
    def __init__(self,protocol):
        self.p=protocol;self.path=protocol.file('state/budget.json');self.active=None
        self._data=None;self._token=None;self._heartbeat_errors=[]

    def used(self):
        data=self._data if self._data is not None else load(self.path) if self.path.exists() else {'jobs':[]}
        return sum(j.get('seconds',0) for j in data['jobs'] if self._token is None or j.get('token')!=self._token)

    def remaining(self,training=False):
        limit=self.p.config['training_seconds' if training else 'budget_seconds']
        active=time.monotonic()-self.active if self.active is not None else 0
        return limit-self.used()-active

    def check(self,training=False):
        require(not self._heartbeat_errors,'Compute heartbeat could not be persisted; stop rather than lose budget accounting.')
        require(self.remaining(training)>0,'Compute budget exhausted; no new work or Test release is permitted.')

    @contextmanager
    def job(self,name,training=False):
        self.check(training)
        # OS lock is released even when a process terminates. A journal entry is
            # written before compute; interrupted work is charged from its last
            # heartbeat, not the hours/days spent waiting after a machine failure.
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.lock').open('a+b') as lock:
            lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
            if os.name=='nt':
                import msvcrt
                try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
                except OSError as e:raise ProtocolError('Another compute command is running') from e
            else:
                import fcntl
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            data=load(self.path) if self.path.exists() else {'jobs':[]}
            now=time.time()
            for row in data['jobs']:
                if row['status']=='running':
                    row.update(status='interrupted',seconds=row.get('seconds',0)+1.0)
            write(self.path,data);self.check(training)
            row=dict(name=name,status='running',started=utc(),started_unix=now,seconds=0,
                     token=digest([name,now,os.getpid()]))
            data['jobs'].append(row);write(self.path,data);self.active=time.monotonic()
            self._data=data;self._token=row['token'];finished=threading.Event();heartbeat_errors=[];self._heartbeat_errors=heartbeat_errors
            def heartbeat():
                while not finished.wait(1.0):
                    row['seconds']=time.monotonic()-self.active
                    try:write(self.path,data)
                    except OSError as e:heartbeat_errors.append(str(e))
            thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
            try:
                yield self
                row['status']='completed'
            except BaseException:
                row['status']='interrupted';raise
            finally:
                finished.set();thread.join()
                row.update(seconds=time.monotonic()-self.active,ended=utc())
                if heartbeat_errors:row['heartbeat_errors']=heartbeat_errors
                self.active=None;self._data=None;self._token=None
                write(self.path,data)
                if os.name=='nt':msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


def configure_runtime(p):
    os.environ['YOLO_CONFIG_DIR']=str(p.file('.ultralytics'))
    p.file('.ultralytics').mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(p.file('.matplotlib'))
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['WANDB_MODE']='disabled'
    import ultralytics.utils as utils
    # Keep every framework-generated artifact in this protocol, and disable
    # external experiment trackers without editing the user's global settings.
    target=p.file('.ultralytics/Ultralytics/settings.json');target.parent.mkdir(parents=True,exist_ok=True)
    utils.SETTINGS.file=target;utils.SETTINGS.file_path=target
    settings=dict(datasets_dir=str(p.file('dataset')),weights_dir=str(p.file('runtime_weights')),
                  runs_dir=str(p.file('runtime_runs')),api_key='',openai_api_key='')
    settings.update({k:False for k in ['sync','wandb','mlflow','clearml','comet','dvc','raytune','tensorboard'] if k in utils.SETTINGS})
    utils.SETTINGS.update(settings)
    utils.WEIGHTS_DIR=p.file('runtime_weights')
