"""Read-only shared GPU and process-environment monitoring."""
import json,subprocess,time
from pathlib import Path
from .audit import write


def main():
    root=Path('outputs/system_optimization_v3')
    while True:
        state=json.loads((root/'gpu_status.json').read_text()) if (root/'gpu_status.json').exists() else {}
        env=json.loads((root/'environment_before.json').read_text()) if (root/'environment_before.json').exists() else {}
        pid=env.get('pid');threads=[]
        if pid:
            for path in (Path('/proc')/str(pid)/'task').glob('*/status'):
                try:
                    d=dict(s.split(':',1) for s in path.read_text().splitlines() if ':' in s)
                    threads.append(dict(tid=int(path.parent.name),name=d['Name'].strip(),affinity=d['Cpus_allowed_list'].strip()))
                except FileNotFoundError:pass
        gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw','--format=csv,noheader'],capture_output=True,text=True)
        processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
        with (root/'logs/environment_monitor.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch_s=time.time(),gpu=gpu.stdout.strip(),gpu_error=gpu.stderr.strip(),processes=processes.stdout.strip(),threads=threads,status=state))+'\n')
        if state.get('status') in ('completed','failed'):break
        time.sleep(10)


if __name__=='__main__':main()
