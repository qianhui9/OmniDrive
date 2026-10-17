"""Observe shared GPU conditions while authorized stages execute."""
import json,subprocess,time
from pathlib import Path


def main():
    root=Path('outputs/system_optimization_v2')
    while True:
        states={s:json.loads((root/(s+'_status.json')).read_text()) if (root/(s+'_status.json')).exists() else None for s in ('smoke','formal')}
        proc=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        row=dict(epoch_s=time.time(),raw=proc.stdout.strip(),error=proc.stderr.strip(),stage_status=states)
        with (root/'logs/gpu_during_execution.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
        if any(v and v['status']=='failed' for v in states.values()):return
        if states['formal'] and states['formal']['status']=='completed':return
        time.sleep(10)


if __name__=='__main__':main()
