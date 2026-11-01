"""Read-only repository evidence verification; never train, infer, or authorize Test."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def check_manifest(root,manifest):
    require(manifest['id']==digest({k:v for k,v in manifest.items() if k!='id'}),'Evidence manifest digest changed')
    require(bool(manifest['files']),'Evidence manifest is empty')
    for relative,expected in manifest['files'].items():
        path=(root/relative).resolve()
        require(path.is_relative_to(root),'Manifest path escapes the repository: '+relative)
        require(path.is_file(),'Missing repository evidence: '+relative)
        require(sha(path)==expected,'Changed repository evidence: '+relative)


def near(actual,expected,label):
    require(math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-10),'Incorrect statistic: '+label)


def verify_statistics(root):
    exp=root/'evaluation_rebuild_20261031'
    final=load(exp/'freezes/final.json');selection=load(exp/'val/selection.json')
    for name,value in [('final freeze',final),('Val selection',selection)]:
        require(value['id']==digest({k:v for k,v in value.items() if k!='id'}),name+' digest changed')
    require(final['selection']==selection['id'],'Final freeze refers to another Val selection')
    require(final['selected_profiles']==selection['profiles'],'Selected profiles differ from Val')
    require(sha(exp/'val/grid.json')==selection['grid_sha256'],'Val grid hash changed')
    test=load(exp/'test/final_results.json')
    require(test['freeze_id']==final['id'],'Test uses another final freeze')
    require(test['selected_profiles']==final['selected_profiles'],'Test changed the selected profiles')
    require(len(test['results'])==len(final['test_rows'])==11,'Expected 11 preregistered Test rows')
    counts=load(exp/'split/proposal.json')['summary']['effective_objects']['test']
    expected_gt=sum(counts.values());require(expected_gt==858,'Unexpected effective GT count')
    for row,expected in zip(test['results'],final['test_rows']):
        for key in ('label','config','threshold'):require(row[key]==expected[key],'Test configuration changed: '+row['label'])
        require([r['seed'] for r in row['seeds']]==[42,7,123],'Unexpected/reordered Test seeds')
        for seed in row['seeds']:
            for ap,classes in [('AP50','class_AP50'),('AP95','class_AP95')]:
                require(len(seed[classes])==6,'Expected six AP classes')
                near(seed[ap],statistics.mean(seed[classes]),ap)
            require(len(seed['points'])==1,'Unexpected Test workpoint count')
            point=seed['points'][0]
            require(point['threshold']==row['threshold'],'Changed workpoint threshold')
            require(point['TP']+point['FN']==expected_gt,'Incorrect total GT count')
            require([r['name'] for r in point['per_class']]==test['class_names'],'Class order changed')
            for metric in ('TP','FP','FN'):require(sum(r[metric] for r in point['per_class'])==point[metric],'Class counts do not sum')
            for p in [point,*point['per_class']]:
                near(p['precision'],p['TP']/max(p['TP']+p['FP'],1),'precision')
                near(p['recall'],p['TP']/max(p['TP']+p['FN'],1),'recall')
                near(p['F1'],2*p['TP']/max(2*p['TP']+p['FP']+p['FN'],1),'F1')
            for p in point['per_class']:require(p['TP']+p['FN']==counts[p['name']],'Incorrect per-class GT count')
        for metric in ('AP50','AP95','precision','recall','F1'):
            values=[r[metric] if metric in ('AP50','AP95') else r['points'][0][metric] for r in row['seeds']]
            near(row['mean'][metric],statistics.mean(values),metric+' mean')
            near(row['std'][metric],statistics.stdev(values),metric+' sample std')
        for c,summary in enumerate(row['per_class_summary']):
            require(summary['name']==test['class_names'][c],'Class summary order changed')
            for metric in ('class_AP50','class_AP95'):
                values=[seed[metric][c] for seed in row['seeds']]
                near(summary[metric+'_mean'],statistics.mean(values),'class AP mean')
                near(summary[metric+'_std'],statistics.stdev(values),'class AP sample std')
    supplement=load(exp/'reports/supplement_v1/supplement_freeze.json')
    require(supplement['id']==digest({k:v for k,v in supplement.items() if k!='id'}),'Supplement digest changed')
    require(supplement['original_final_freeze']==final['id'],'Supplement refers to another freeze')
    require(supplement['original_final_json_sha256']==sha(exp/'freezes/final.json'),'Supplement final JSON hash mismatch')
    for rel,expected in supplement['files'].items():require(sha(exp/rel)==expected,'Supplement report changed')
    return dict(final_freeze=final['id'],test_rows=11,seed_results=33,sample_std_verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=['repository','full'],default='repository')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args();root=args.root.resolve()
    try:
        manifest=load(root/'docs/evidence/repository_evidence_manifest.json')
        check_manifest(root,manifest)
        checked=verify_statistics(root)
        if args.scope=='full':
            sys.path.insert(0,str(root/'evaluation_rebuild_20261031'))
            from neu_protocol.core import Protocol
            from neu_protocol.training import check_environment
            p=Protocol(root/'evaluation_rebuild_20261031');p.final_freeze();check_environment(p)
            for record in ('test/archive_manifest.json','post_test_delivery/delivery_manifest.json'):
                p.check_files(load(p.file(record)))
        print(json.dumps(dict(status='passed',scope=args.scope,repository_files=len(manifest['files']),**checked,
                             models_and_source_data_verified=args.scope=='full',
                             note='Repository scope verifies recorded evidence and arithmetic, not model accuracy by new inference. No training, inference, authorization, or network request performed.'),indent=2))
        return 0
    except (ValueError,KeyError,FileNotFoundError,ModuleNotFoundError) as error:
        print('VERIFICATION FAILED: '+str(error),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
