"""Wait for the owned model job, then run all CUDA regressions on idle GPU."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from tools.worldstream_m2.source_state_refinement_cuda_batch import wait_idle
from .common import ROOT,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT));args=p.parse_args();root=Path(args.output)
    dest=root/'regression_preflight';dest.mkdir(exist_ok=False)
    limit=time.monotonic()+4*3600
    while not (root/'validation/status.json').exists():
        if 'Traceback (most recent call last)' in (root/'service.log').read_text():
            raise RuntimeError('Service failed; inspect its retained log')
        if time.monotonic()>limit:raise TimeoutError('Model validation not completed')
        time.sleep(15)
    wait_idle(dest,None)
    files=[*Path('worldstream').rglob('*.py'),*Path('tools/worldstream_m3/microbatch').glob('*.py'),
        *(p for d in ('m1','m2','m3') for p in Path('tests/worldstream_'+d).rglob('*.py'))]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write_json(dest/'source_manifest.json',hashes)
    cmd=[sys.executable,'-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','-q',
        '--junitxml='+str(root/'regression_cuda.xml')]
    with (root/'regression.log').open('x') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    write_json(dest/'status.json',dict(returncode=result.returncode,
        sources_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())))
    if result.returncode:raise SystemExit(result.returncode)
    subprocess.run([sys.executable,'-m','tools.worldstream_m3.microbatch.finalize','--output',str(root)],check=True)


if __name__=='__main__':main()
