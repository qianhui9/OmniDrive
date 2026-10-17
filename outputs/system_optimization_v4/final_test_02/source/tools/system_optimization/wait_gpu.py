"""Observe only: never stop another user's GPU processes."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def wait(output, minimum_mib=19500, interval=10):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    last = None
    while True:
        proc = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True)
        row = dict(epoch_s=time.time(), raw=proc.stdout.strip(), error=proc.stderr.strip(),
                   required_free_mib=minimum_mib)
        try:
            free, used, util = map(int, proc.stdout.splitlines()[0].split(','))
            row.update(free_mib=free, used_mib=used, utilization_percent=util, ready=free>=minimum_mib)
        except (ValueError, IndexError):
            row['ready'] = False
        with path.open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        if row['ready'] or last is None or abs(row.get('free_mib', 0)-last) > 256:
            print(json.dumps(row), flush=True)
        if row['ready']:
            return
        last = row.get('free_mib', 0)
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/system_optimization/logs/gpu_availability.jsonl')
    parser.add_argument('--minimum-mib', type=int, default=19500)
    args = parser.parse_args()
    wait(args.output, args.minimum_mib)
