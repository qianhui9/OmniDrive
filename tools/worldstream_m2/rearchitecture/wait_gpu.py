"""Poll once per minute without touching or terminating any external process."""
import csv
from datetime import datetime,timezone
import io
import json
import subprocess
import time
from .common import ROOT,write


def main():
    path=ROOT/'gpu_availability.jsonl'
    while True:
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
        rows=list(csv.reader(io.StringIO(result.stdout.strip()))) if result.stdout.strip() else []
        available=result.returncode==0 and not rows
        record=dict(time_utc=datetime.now(timezone.utc).isoformat(),available=available,processes=rows,error=result.stderr.strip())
        with path.open('a') as handle:handle.write(json.dumps(record)+'\n')
        print(record,flush=True)
        if available:
            write(ROOT/'gpu_ready.json',record);return
        time.sleep(60)

if __name__=='__main__':main()
