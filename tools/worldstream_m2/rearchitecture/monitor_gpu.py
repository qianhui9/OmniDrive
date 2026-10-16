"""Read-only one-minute audit while an authorized experiment process is alive."""
import argparse
import csv
from datetime import datetime,timezone
import io
import json
import os
import subprocess
import time
from .common import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);args=p.parse_args()
    while True:
        try:os.kill(args.pid,0)
        except ProcessLookupError:return
        r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
        rows=list(csv.reader(io.StringIO(r.stdout.strip()))) if r.stdout.strip() else []
        def belongs(pid):
            while pid>1:
                if pid==args.pid:return True
                try:pid=int(open(f'/proc/{pid}/stat').read().rsplit(')',1)[1].split()[1])
                except (OSError,ValueError,IndexError):return False
            return False
        other=[x for x in rows if not belongs(int(x[0].strip()))]
        record=dict(time_utc=datetime.now(timezone.utc).isoformat(),experiment_pid=args.pid,
            processes=rows,competing_processes=other,query_ok=r.returncode==0)
        with (ROOT/'gpu_execution_audit.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(dict(time_utc=record['time_utc'],competing=len(other))),flush=True)
        time.sleep(60)

if __name__=='__main__':main()
