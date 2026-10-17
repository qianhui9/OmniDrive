"""Observe shared GPU conditions while authorized stages execute."""
import json,subprocess,time
from pathlib import Path


def main():
    root=Path('outputs/system_optimization_v2')
    while True:
        stages=('smoke','formal','cost_refresh')
        states={s:json.loads((root/(s+'_status.json')).read_text()) if (root/(s+'_status.json')).exists() else None for s in stages}
        proc=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        row=dict(epoch_s=time.time(),raw=proc.stdout.strip(),error=proc.stderr.strip(),stage_status=states)
        with (root/'logs/gpu_during_execution.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
        if any(v and v['status']=='failed' for v in states.values()):return
        target='cost_refresh' if (root/'cost_refresh_protocol.json').exists() else 'formal'
        if states[target] and states[target]['status']=='completed':return
        time.sleep(10)


if __name__=='__main__':main()
