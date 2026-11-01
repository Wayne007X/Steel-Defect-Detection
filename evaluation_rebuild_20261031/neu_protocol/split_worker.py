"""Dependency-isolated splitter: never imported into a training process."""
import json
import sys
sys.path.insert(0,sys.argv[1])
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
if __name__=='__main__':
    payload=json.load(sys.stdin);cfg=payload['splitter']
    assert sklearn.__version__==cfg['version'],(sklearn.__version__,cfg['version'])
    splitter=StratifiedGroupKFold(n_splits=cfg['folds'],shuffle=cfg['shuffle'],random_state=cfg['seed'])
    folds=[None]*len(payload['y'])
    for fold,(_,test) in enumerate(splitter.split(payload['y'],payload['y'],payload['groups'])):
        for i in test:folds[int(i)]=fold
    print(json.dumps(dict(folds=folds,sklearn=sklearn.__version__)))
