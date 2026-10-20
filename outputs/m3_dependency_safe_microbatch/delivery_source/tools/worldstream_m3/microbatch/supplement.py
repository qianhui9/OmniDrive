"""Serialize the actual first-divergence trajectory probe after CUDA regression."""
import argparse
import hashlib
from pathlib import Path
import time
import torch
from .common import ROOT,pin,load,read,write_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT));args=p.parse_args();root=Path(args.output)
    deadline=time.monotonic()+3600
    while not (root/'regression_preflight/status.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('CUDA regression has not finished')
        time.sleep(15)
    if read(root/'regression_preflight/status.json')['returncode']!=0:raise RuntimeError('Fix regression before supplementary probe')
    hashes=pin(root/'trajectory_validation');model,dataset=load(root/'trajectory_validation')
    if (root/'final_source_confirmation.json').exists():
        (root/'final_source_confirmation_before_trajectory.json').write_bytes((root/'final_source_confirmation.json').read_bytes())
    from .final_source import run
    run(model,root)
    # Preserve the originally loaded late-stage source and manifest before the
    # supplementary source replaces the current reporting pointer.
    late=read(root/'fallback_execution_sources.json')
    write_json(root/'fallback_execution_sources_before_trajectory.json',late)
    for name in ('final_source.py','trajectory.py'):
        path=Path('tools/worldstream_m3/microbatch')/name
        old=root/'fallback_execution_source'/name
        if old.exists():(root/'fallback_execution_source'/(name+'.before_trajectory')).write_bytes(old.read_bytes())
        old.write_bytes(path.read_bytes());late[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(root/'fallback_execution_sources.json',late)
    write_json(root/'trajectory_validation/status.json',dict(completed=True,
        sources_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())))
    # Model ownership ends when this process exits. Finalizer is read-only w.r.t.
    # GPU and may run here without queueing another GPU job.
    from .finalize import main as finalize
    import sys
    sys.argv=['finalize','--output',str(root)];finalize()


if __name__=='__main__':main()
