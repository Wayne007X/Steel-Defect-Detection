"""Hash static delivery inputs; runtime outputs and approvals are not release inputs."""
import argparse
import json
from pathlib import Path
from neu_protocol.core import require,load,write,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--create',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;manifest=root/'PACKAGE_MANIFEST_SHA256.json'
    if args.create:
        require(not manifest.exists(),'Static delivery manifest already exists')
        excluded={'.git','.test_work','__pycache__','.ultralytics','.matplotlib','reports','state','freezes','runs','cache','exports','test'}
        files={}
        for path in sorted(root.rglob('*')):
            if not path.is_file() or excluded&set(path.relative_to(root).parts) or path.suffix=='.log' or path==manifest:continue
            files[path.relative_to(root).as_posix()]=dict(sha256=sha(path),bytes=path.stat().st_size)
        write(manifest,dict(scope='Initial static code/data/dependency/review inputs only; not a split/final experiment freeze.',files=files),exclusive=True)
    data=load(manifest)
    for relative,entry in data['files'].items():
        path=(root/relative).resolve();require(path.is_relative_to(root),'Unsafe manifest path')
        require(path.is_file() and sha(path)==entry['sha256'],f'Delivery integrity failure: {relative}')
    print(json.dumps(dict(verified_files=len(data['files']),total_bytes=sum(v['bytes'] for v in data['files'].values())),indent=2))


if __name__=='__main__':main()
