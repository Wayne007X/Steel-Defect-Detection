"""Supervise an existing training process; never authorize or run Test."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--training-created', type=float, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    state = root / 'reports' / 'continuation_status.json'
    python = Path(r'E:\yolo_workspace\Scripts\python.exe')
    cli = root / 'run_protocol.py'

    def record(stage, **details):
        value = dict(stage=stage, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     supervisor_pid=os.getpid(), training_pid=args.training_pid,
                     test_will_run=False, **details)
        tmp = state.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(state)

    try:
        process = psutil.Process(args.training_pid)
        if abs(process.create_time() - args.training_created) > 0.01:
            raise RuntimeError('Training process identity changed; refusing to attach')
        command = process.cmdline()
        if str(cli).lower() not in [item.lower() for item in command] or 'train' not in command:
            raise RuntimeError('Specified process is not this protocol training command')
        while True:
            record('waiting_for_training', completed_tasks=len(list((root / 'runs').glob('*/completion.json'))))
            try:
                process.wait(timeout=30)
                break
            except psutil.TimeoutExpired:
                pass

        result = subprocess.run([str(python), str(cli), 'status'], cwd=root, capture_output=True, text=True, check=True)
        status = json.loads(result.stdout)
        if not status['split_frozen'] or len(status['completed_training_jobs']) != 12:
            record('stopped_training_incomplete', protocol_status=status,
                   reason='Inspect training_console.log; no automatic resume or Test execution')
            return 2
        for stage in ('select-val', 'verify-export', 'freeze-final'):
            record('running_' + stage)
            with (root / 'reports' / (stage + '_console.log')).open('ab', buffering=0) as log:
                result = subprocess.run([str(python), '-u', str(cli), stage], cwd=root,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                record('stopped_' + stage, returncode=result.returncode,
                       reason='Inspect stage log; no automatic retry, protocol change, or Test execution')
                return result.returncode
        record('awaiting_human_final_freeze_confirmation',
               final_report=str(root / 'freezes' / '最终冻结报告.md'))
        return 0
    except Exception:
        record('supervisor_failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    sys.exit(main())
