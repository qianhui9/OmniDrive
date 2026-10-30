"""Read-only hardware/progress sampler; never signals or changes any process."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();output=a.root/'gpu_telemetry.jsonl'
    while True:
        row=dict(UTC=datetime.now(timezone.utc).isoformat())
        try:
            result=subprocess.run(['nvidia-smi',
                '--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu',
                '--format=csv,noheader'],text=True,capture_output=True,timeout=10)
            row.update(returncode=result.returncode,hardware=result.stdout.strip(),error=result.stderr.strip())
        except Exception as e:row['error']=repr(e)
        progress=a.root/'main/progress.json'
        row['progress']=json.loads(progress.read_text()) if progress.exists() else None
        with output.open('a') as f:f.write(json.dumps(row)+'\n')
        if (a.root/'main/run_status.json').exists() or (a.root/'main_failure.json').exists():break
        time.sleep(30)


if __name__=='__main__':main()
