import argparse
import json
import os
from pathlib import Path
import sys
from .core import Protocol,ProtocolError,load,sha,Budget,configure_runtime


def main(default_root):
    parser=argparse.ArgumentParser(description='NEU-DET rebuilt internal evaluation; Test is closed by default.')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['status','audit','prepare-weights','propose-split','select-val','verify-export','freeze-final','test']:
        commands.add_parser(name)
    review=commands.add_parser('import-review');review.add_argument('--file',type=Path,required=True)
    approve=commands.add_parser('approve-split');approve.add_argument('--proposal-sha',required=True);approve.add_argument('--reviewer',required=True)
    train=commands.add_parser('train');train.add_argument('--resume',action='store_true')
    release=commands.add_parser('approve-test');release.add_argument('--freeze-sha',required=True);release.add_argument('--reviewer',required=True)
    args=parser.parse_args();p=Protocol(default_root)
    if hasattr(args,'file'):args.file=args.file.resolve()
    os.chdir(p.root)
    try:
        if args.command=='status':
            completed=sorted(str(x.parent.name) for x in p.file('runs').glob('*/completion.json'))
            result=dict(protocol=p.config['version'],audit_complete=p.file('audit/manifest.json').exists(),
                        human_review_imported=p.file('review/approved_decisions.json').exists(),
                        split_proposed=p.file('split/proposal.json').exists(),split_frozen=p.file('freezes/split.json').exists(),
                        completed_training_jobs=completed,required_training_jobs=12,compute_seconds_used=Budget(p).used(),
                        final_frozen=p.file('freezes/final.json').exists(),test_authorized=p.file('freezes/test_authorization.json').exists(),
                        test_complete=p.file('test/final_results.json').exists())
        elif args.command in ['audit','import-review','propose-split','approve-split']:
            from . import data
            if args.command=='audit':result=data.audit(p)
            elif args.command=='import-review':result=data.import_review(p,args.file)
            elif args.command=='propose-split':result=data.propose_split(p)
            else:result=data.approve_split(p,args.proposal_sha,args.reviewer)
        elif args.command=='prepare-weights':
            from .training import prepare_weights
            result=prepare_weights(p)
        elif args.command=='train':
            from .training import train_all
            result=train_all(p,args.resume)
        elif args.command=='approve-test':
            p.authorize_test(args.freeze_sha,args.reviewer);result={'test_authorized':True,'freeze':args.freeze_sha}
        else:
            # Guards precede imports and any model/dataset loading.
            if args.command=='test':p.test_guard()
            else:p.completed_jobs()
            configure_runtime(p)
            from . import evaluation
            function={'select-val':evaluation.select_val,'verify-export':evaluation.export_verify,
                      'freeze-final':evaluation.freeze_final,'test':evaluation.evaluate_test}[args.command]
            result=function(p)
        print(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False))
    except (ProtocolError,FileNotFoundError,FileExistsError) as e:
        print('PROTOCOL BLOCKED: '+str(e),file=sys.stderr);raise SystemExit(2)
