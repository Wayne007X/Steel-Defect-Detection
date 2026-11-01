import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from neu_protocol.core import Protocol,ProtocolError,write,load,sha,digest,Budget,configure_runtime
from neu_protocol.data import make_groups,verify_partition,import_review,phash,ssim,propose_split
from neu_protocol.training import StrictEarlyStopping,recipe,trainer_type
from neu_protocol.evaluation import configurations,choose


class Fixture(unittest.TestCase):
    def setUp(self):
        base=ROOT/'.test_work';base.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='fixture_',dir=base);self.root=Path(self.temp.name).resolve()
        assert self.root.is_relative_to(base.resolve())
        write(self.root/'protocol.json',load(ROOT/'protocol.json'));self.p=Protocol(self.root)
    def tearDown(self):self.temp.cleanup()


class Guards(Fixture):
    def test_train_requires_explicit_human_split_approval(self):
        with self.assertRaises(ProtocolError):self.p.split_freeze()
        self.assertFalse(self.p.file('runs').exists())
    def test_test_cannot_release_before_all_jobs(self):
        with self.assertRaises(ProtocolError):self.p.authorize_test('any','human')
    def test_test_requires_exact_final_digest_and_approval(self):
        with patch.object(self.p,'completed_jobs',return_value={}),patch.object(self.p,'split_freeze',return_value={'id':'split'}):
            frozen=dict(files={},code=self.p.code_hashes(),split_freeze='split');frozen['id']=digest(frozen)
            write(self.p.file('freezes/final.json'),frozen)
            with self.assertRaises(ProtocolError):self.p.test_guard()
            with self.assertRaises(ProtocolError):self.p.authorize_test('wrong','human')
            self.p.authorize_test(frozen['id'],'human')
            self.assertEqual(self.p.test_guard()['id'],frozen['id'])
    def test_frozen_file_tampering_is_detected(self):
        path=self.p.file('a');path.write_bytes(b'original');record=dict(files={'a':sha(path)})
        self.p.check_files(record);path.write_bytes(b'edited')
        with self.assertRaises(ProtocolError):self.p.check_files(record)
    def test_code_change_invalidates_split_freeze(self):
        record=dict(files={},code=self.p.code_hashes());record['id']=digest(record)
        write(self.p.file('freezes/split.json'),record)
        with patch.object(self.p,'source_integrity',return_value=[]):self.p.split_freeze()
        self.p.file('protocol.json').write_text('{}')
        with self.assertRaises(ProtocolError):self.p.split_freeze()
    def test_pending_human_decisions_rejected(self):
        candidates=[dict(pair_id='P001',exact_pixels=False)]
        write(self.p.file('audit/candidates.json'),candidates)
        review=dict(reviewer='human',reviewed_at='2026-10-31',candidates_sha256=sha(self.p.file('audit/candidates.json')),
                    decisions=[dict(pair_id='P001',decision=None,reason='')])
        write(self.p.file('draft.json'),review)
        with self.assertRaises(ProtocolError):import_review(self.p,self.p.file('draft.json'))
        self.assertFalse(self.p.file('review/approved_decisions.json').exists())
    def test_contradictory_review_is_rejected_before_approval_is_written(self):
        rows=[dict(id=x) for x in ['a','b','c']]
        pairs=[dict(pair_id=str(i),a=a,b=b,exact_pixels=False) for i,(a,b) in enumerate([('a','b'),('b','c'),('a','c')])]
        write(self.p.file('audit/manifest.json'),rows);write(self.p.file('audit/candidates.json'),pairs)
        value=dict(reviewer='synthetic fixture',reviewed_at='2026-10-31',candidates_sha256=sha(self.p.file('audit/candidates.json')),
                   decisions=[dict(pair_id=str(i),decision='same' if i<2 else 'different',reason='fixture') for i in range(3)])
        write(self.p.file('draft.json'),value)
        with self.assertRaises(ProtocolError):import_review(self.p,self.p.file('draft.json'))
        self.assertFalse(self.p.file('review/approved_decisions.json').exists())
    def test_budget_is_cumulative_and_blocks_new_jobs(self):
        b=Budget(self.p)
        with patch('neu_protocol.core.time.monotonic',side_effect=[1000.,1000.1,1000.5]):
            with b.job('fixture'):
                self.assertGreater(b.remaining(),0)
        self.assertGreater(b.used(),0)
        write(b.path,dict(jobs=[dict(status='completed',seconds=28800)]))
        with self.assertRaises(ProtocolError):b.check()
        with self.assertRaises(ProtocolError):
            with b.job('never runs'):self.fail('budget bypass')
    def test_incomplete_job_blocks_validation(self):
        with patch.object(self.p,'split_freeze',return_value={'id':'frozen'}):
            with self.assertRaises(ProtocolError):self.p.completed_jobs()
    def test_c_d_must_share_the_new_B_checkpoint(self):
        self.p.file('models').mkdir();generic=self.p.file('models/yolov8n_coco.pt');generic.write_bytes(b'official-fixture')
        write(self.p.file('models/official_provenance.json'),{'sha256':sha(generic)})
        for family in ['A','B','C','D']:
            for seed in self.p.config['seeds']:
                name=f'{family}_seed{seed}';best=self.p.file(f'runs/{name}/weights/best.pt');best.parent.mkdir(parents=True);best.write_bytes(name.encode())
                initial=f'runs/B_seed{seed}/weights/best.pt' if family in ['C','D'] else 'models/yolov8n_coco.pt'
                write(self.p.file(f'runs/{name}/completion.json'),dict(family=family,seed=seed,best=f'runs/{name}/weights/best.pt',
                      best_sha256=sha(best),split_freeze='frozen',args=recipe(self.p,family,seed),initialization=initial,
                      initialization_sha256=sha(self.p.file(initial))))
        with patch.object(self.p,'split_freeze',return_value={'id':'frozen'}):
            self.assertEqual(len(self.p.completed_jobs()),12)
            changed=load(self.p.file('runs/C_seed42/completion.json'));changed['initialization_sha256']='old-neu-weight'
            write(self.p.file('runs/C_seed42/completion.json'),changed)
            with self.assertRaises(ProtocolError):self.p.completed_jobs()
    def test_source_xml_and_canonical_label_tampering_are_detected(self):
        row={}
        for name in ['image','label','xml']:
            src=self.p.file('source/'+name);src.parent.mkdir(exist_ok=True);src.write_bytes(name.encode())
            row['source_'+name]=str(src);row[('image' if name=='image' else name)+'_sha256']=sha(src)
            if name!='xml':
                canonical=self.p.file('dataset/'+name);canonical.parent.mkdir(exist_ok=True);canonical.write_bytes(src.read_bytes());row[name]='dataset/'+name
        write(self.p.file('audit/manifest.json'),[row]);self.p.source_integrity()
        self.p.file('dataset/label').write_bytes(b'changed')
        with self.assertRaises(ProtocolError):self.p.source_integrity()
    def test_interrupted_budget_does_not_charge_days_of_human_wait(self):
        b=Budget(self.p)
        write(b.path,dict(jobs=[dict(name='crash',status='running',seconds=10.,started_unix=0.)]))
        with b.job('resumed'):
            self.assertLess(b.used(),12.)


class Grouping(Fixture):
    def test_draft_split_lists_are_hashed_and_development_yaml_has_no_test(self):
        for relative in ['tools/split_deps/sklearn/model_selection/_split.py','requirements-split.txt']:
            target=self.p.file(relative);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
        rows=[]
        for c,name in enumerate(self.p.config['names']):
            for i in range(10):
                label=self.p.file(f'dataset/labels/{name}/{name}_{i}.txt');label.parent.mkdir(parents=True,exist_ok=True)
                label.write_text(f'{c} 0.5 0.5 0.2 0.2\n',encoding='utf-8')
                rows.append(dict(id=f'{name}/{name}_{i}.jpg',class_id=c,label=str(label.relative_to(self.root))))
        write(self.p.file('audit/manifest.json'),rows);write(self.p.file('audit/candidates.json'),[]);write(self.p.file('audit/summary.json'),{})
        write(self.p.file('review/approved_decisions.json'),dict(reviewer='synthetic fixture',decisions=[]))
        real_run=subprocess.run
        def worker(command,**kwargs):
            return real_run([sys.executable,str(ROOT/'neu_protocol/split_worker.py'),str(ROOT/'tools/split_deps')],**kwargs)
        with patch.object(self.p,'source_integrity',return_value=rows),patch('neu_protocol.data.subprocess.run',side_effect=worker):
            draft=propose_split(self.p);again=propose_split(self.p)
        self.assertEqual(draft['id'],again['id'])
        self.assertEqual(draft['summary']['counts'],dict(train=36,val=12,test=12))
        import yaml
        self.assertNotIn('test',yaml.safe_load(self.p.file('dataset/development.yaml').read_text(encoding='utf-8')))
        for split in ['train','val','test']:
            path=self.p.file('dataset/'+split+'.txt')
            self.assertEqual(draft['summary']['hashes'][split]['sha256'],sha(path))
        self.assertFalse(self.p.file('freezes/split.json').exists())
    def test_uncertain_edges_are_conservatively_grouped(self):
        rows=[dict(id=x) for x in ['a','b','c']]
        pairs=[dict(pair_id='x',a='a',b='b',exact_pixels=False)]
        review=dict(decisions=[dict(pair_id='x',decision='uncertain')])
        groups=make_groups(rows,pairs,review)
        self.assertIn(['a','b'],[r['members'] for r in groups])
    def test_transitive_contradiction_is_not_silently_ignored(self):
        rows=[dict(id=x) for x in ['a','b','c']]
        pairs=[dict(pair_id=str(i),a=a,b=b,exact_pixels=False) for i,(a,b) in enumerate([('a','b'),('b','c'),('a','c')])]
        review=dict(decisions=[dict(pair_id=str(i),decision='same' if i<2 else 'different') for i in range(3)])
        with self.assertRaises(ProtocolError):make_groups(rows,pairs,review)
    def test_group_overlap_is_rejected(self):
        rows=[dict(id=f'{split}{c}',class_id=c) for split in ['t','v','x'] for c in range(6)]
        groups=[dict(group_id='one',members=[r['id'] for r in rows])]
        lists={s:[r for r in rows if r['id'].startswith(letter)] for s,letter in [('train','t'),('val','v'),('test','x')]}
        with self.assertRaises(ProtocolError):verify_partition(rows,groups,lists)
    def test_similarity_identity(self):
        image=np.random.default_rng(42).integers(0,256,(200,200),dtype=np.uint8)
        self.assertEqual(len(phash(image)),63);self.assertEqual(ssim(image,image),1.)
    def test_pinned_group_split_is_reproducible(self):
        payload=dict(y=[c for c in range(6) for _ in range(10)],groups=[f'{c}_{i}' for c in range(6) for i in range(10)],splitter=self.p.config['splitter'])
        payload['groups'][1]=payload['groups'][0]
        command=[sys.executable,str(ROOT/'neu_protocol/split_worker.py'),str(ROOT/'tools/split_deps')]
        a=subprocess.run(command,input=json.dumps(payload),text=True,capture_output=True,check=True)
        b=subprocess.run(command,input=json.dumps(payload),text=True,capture_output=True,check=True)
        self.assertEqual(a.stdout,b.stdout);folds=json.loads(a.stdout)['folds'];self.assertEqual(folds[0],folds[1])


class TrainingContract(Fixture):
    def test_zero_and_tied_ap50_do_not_reset_patience(self):
        stop=StrictEarlyStopping(2)
        self.assertFalse(stop(1,0.));self.assertFalse(stop(2,0.));self.assertTrue(stop(3,0.));self.assertEqual(stop.best_epoch,1)
        stop=StrictEarlyStopping(2)
        self.assertFalse(stop(1,.5));self.assertFalse(stop(2,.6));self.assertFalse(stop(3,.6));self.assertTrue(stop(4,.59));self.assertEqual(stop.best_epoch,2)
    def test_finetune_control_changes_only_profile_not_budget(self):
        c=recipe(self.p,'C',42);d=recipe(self.p,'D',42)
        differences={key for key in c if c[key]!=d[key]}
        self.assertEqual(differences,{'name','hsv_h','hsv_s','hsv_v','scale','translate'})
        self.assertEqual(c['epochs'],40);self.assertEqual(c['warmup_bias_lr'],0)
    def test_mosaic_ablation_is_one_factor(self):
        a=recipe(self.p,'A',42);b=recipe(self.p,'B',42)
        self.assertEqual({k for k in a if a[k]!=b[k]},{'name','mosaic','close_mosaic'})
    def test_best_checkpoint_keeps_earlier_tied_epoch(self):
        configure_runtime(self.p)
        from ultralytics.models.yolo.detect import DetectionTrainer
        cls=trainer_type(self.p,Budget(self.p));obj=cls.__new__(cls)
        obj.best=self.root/'best.pt';obj.last=self.root/'last.pt';obj.best.write_bytes(b'earlier')
        obj._new_ap50_best=False;obj.best_fitness=.5;obj.fitness=.5
        def underlying(x):
            x.last.write_bytes(b'new last')
            if x.best_fitness==x.fitness:x.best.write_bytes(b'later tie')
            return True
        with patch.object(DetectionTrainer,'save_model',underlying):obj.save_model()
        self.assertEqual(obj.best.read_bytes(),b'earlier');self.assertEqual(obj.last.read_bytes(),b'later tie')
        obj._new_ap50_best=True
        with patch.object(DetectionTrainer,'save_model',underlying):obj.save_model()
        self.assertEqual(obj.best.read_bytes(),b'later tie')
    def test_validator_fitness_is_ap50_and_validation_disables_amp_temporarily(self):
        configure_runtime(self.p)
        from ultralytics.models.yolo.detect import DetectionTrainer,DetectionValidator
        from ultralytics.cfg import get_cfg
        cls=trainer_type(self.p,Budget(self.p));obj=cls.__new__(cls)
        obj.args=get_cfg(overrides={'plots':False});obj.test_loader=None;obj.save_dir=self.root;obj.callbacks=None
        validator=obj.get_validator()
        with patch.object(DetectionValidator,'get_stats',return_value={'metrics/mAP50(B)':.5,'metrics/mAP50-95(B)':.9,'fitness':.9}):
            self.assertEqual(validator.get_stats()['fitness'],.5)
        obj.amp=True;obj.best_fitness=None;obj.epoch=0
        def underlying(x):
            self.assertIs(x.amp,False);x.best_fitness=.5
            return {'metrics/mAP50-95(B)':.9},.5
        with patch.object(DetectionTrainer,'validate',underlying):obj.validate()
        self.assertIs(obj.amp,True);self.assertTrue(obj._new_ap50_best)
        self.assertEqual(load(self.root/'ap50_best.json')['AP50'],.5)
    def test_validation_is_square_and_fp32(self):
        configure_runtime(self.p)
        import torch
        from ultralytics.cfg import get_cfg
        cls=trainer_type(self.p,Budget(self.p));obj=cls.__new__(cls)
        obj.args=get_cfg(overrides={'plots':False});obj.test_loader=None;obj.save_dir=self.root;obj.callbacks=None
        validator=obj.get_validator();validator.device=torch.device('cpu');validator.args.quantize=None
        result=validator.preprocess({'img':torch.zeros((1,3,640,640),dtype=torch.uint8)})
        self.assertEqual(result['img'].dtype,torch.float32)
        with self.assertRaises(ProtocolError):validator.preprocess({'img':torch.zeros((1,3,320,320),dtype=torch.uint8)})
        validator.args.quantize=16
        with self.assertRaises(ProtocolError):validator.preprocess({'img':torch.zeros((1,3,640,640),dtype=torch.uint8)})
    def test_actual_val_dataset_disables_library_rectangular_padding(self):
        configure_runtime(self.p)
        import torch,cv2
        from ultralytics.cfg import get_cfg
        image=self.p.file('fixture_data/images/crazing/a.jpg');image.parent.mkdir(parents=True)
        label=self.p.file('fixture_data/labels/crazing/a.txt');label.parent.mkdir(parents=True)
        cv2.imwrite(str(image),np.zeros((200,200,3),np.uint8));label.write_text('0 0.5 0.5 0.2 0.2\n')
        cls=trainer_type(self.p,Budget(self.p));obj=cls.__new__(cls)
        obj.args=get_cfg(overrides={'imgsz':640,'rect':False,'plots':False})
        obj.model=torch.nn.Identity();obj.model.stride=torch.tensor([32])
        obj.data=dict(nc=6,names=dict(enumerate(self.p.config['names'])),channels=3)
        dataset=obj.build_dataset(str(image.parent),mode='val',batch=1)
        self.assertFalse(dataset.rect);self.assertEqual(tuple(dataset[0]['img'].shape),(3,640,640))


class Selection(Fixture):
    def rows(self):
        result=[]
        for c in configurations(self.p):
            for seed in self.p.config['seeds']:
                ap=.6 if c['id']=='B_t0_native' else .59
                if c['id']=='AC_t1_adaptive':ap=.7
                if c['id']=='A_t0_native':ap=.99  # rejected below for weak-class collapse
                per=[ap]*6
                if c['id']=='A_t0_native':per[0]=.1
                points=[dict(threshold=t,precision=.8,recall=.8,F1=.8 if t==.25 else .7) for t in self.p.config['evaluation']['thresholds']]
                result.append(dict(config_id=c['id'],seed=seed,AP50=ap,AP95=.4,class_AP50=per,points=points))
        return result
    def test_primary_selection_uses_guard_and_all_three_seeds(self):
        r=choose(self.p,self.rows());self.assertEqual(r['profiles']['research']['config']['id'],'AC_t1_adaptive')
        self.assertTrue(r['promoted']);self.assertEqual(r['profiles']['balanced']['config']['cost'],1)
        with self.assertRaises(ProtocolError):choose(self.p,self.rows()[:-1])
    def test_failed_candidates_fall_back_to_B(self):
        rows=self.rows()
        for row in rows:
            if row['config_id']=='AC_t1_adaptive':row.update(AP50=.601,class_AP50=[.601]*6)
        r=choose(self.p,rows);self.assertFalse(r['promoted']);self.assertEqual(r['profiles']['research']['config']['id'],'B_t0_native')
    def test_no_test_data_is_needed_for_selection(self):
        self.assertFalse(self.p.file('dataset/test.txt').exists());r=choose(self.p,self.rows())
        self.assertGreaterEqual(len(r['test_rows']),8)
        for row in r['test_rows']:self.assertIn(row['threshold'],[.25,*self.p.config['evaluation']['thresholds']])


class PredictionRules(Fixture):
    def test_workpoint_counts_equal_prefix_rematching(self):
        configure_runtime(self.p)
        from neu_protocol.prediction import measure,subset,statistics,postprocess
        r=dict(path='synthetic',input_shape=(640,640),boxes=np.array([[0,0,100,100],[5,5,95,95],[200,200,300,300]],np.float32),
               conf=np.array([.9,.5,.1],np.float32),cls=np.array([0,0,0],np.float32),
               gt_boxes=np.array([[0,0,100,100],[200,200,300,300]],np.float32),gt_cls=np.array([0,0],np.float32))
        from ultralytics.models.yolo.detect import DetectionValidator
        with patch.object(DetectionValidator,'__init__',side_effect=AssertionError('must not create default runs directories')):
            report=measure([r],self.p.config['names'],[.25])
        tp,_,_,targets=statistics([subset(r,r['conf']>=.25)])
        self.assertEqual(report['points'][0]['TP'],int(tp[:,0].sum()));self.assertEqual(report['points'][0]['FN'],len(targets)-int(tp[:,0].sum()))
        suppressed=postprocess(r,[.3]*6);self.assertEqual(len(suppressed['conf']),2)


if __name__=='__main__':unittest.main()
