"""Stop only the processes recorded for this isolated web delivery."""
import json
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parent
manifest=ROOT/'running_services.json'
records=json.loads(manifest.read_text(encoding='utf-8'))['processes']
for record in records:
    try:
        process=psutil.Process(record['pid'])
        if abs(process.create_time()-record['create_time'])>.01:
            raise RuntimeError('PID reused; refusing to stop it')
        if process.cmdline()!=record['cmdline'] or Path(process.cwd()).resolve()!=ROOT.resolve():
            raise RuntimeError('Process identity changed; refusing to stop it')
        process.terminate()
        try:process.wait(timeout=10)
        except psutil.TimeoutExpired:process.kill();process.wait(timeout=5)
        print('Stopped recorded process',record['pid'])
    except psutil.NoSuchProcess:
        print('Already stopped',record['pid'])
