from pathlib import Path
import hashlib
import json
import subprocess
import torch
from tools.worldstream_m1.io_utils import write_json
from tools.worldstream_m3.sgvd.probe import measure,append,probability_tools
from tools.worldstream_m3.decode_optimization.profile import load,SAMPLING,ForwardTimer

ROOT=Path('outputs/m3_dependency_safe_microbatch')
OLD=Path('outputs/m3_state_guided_verified_decode')


def pin(dest):
    dest.mkdir(parents=True,exist_ok=False)
    files=[*Path('worldstream').rglob('*.py'),*Path('tools/worldstream_m3/microbatch').glob('*.py'),
        *Path('projects/mmdet3d_plugin').rglob('*.py'),*Path('tests/worldstream_m3').glob('*.py')]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    write_json(dest/'sources.json',hashes)
    for f in files:
        out=dest/'source'/f;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(f.read_bytes())
    write_json(dest/'baseline.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        worktree=subprocess.check_output(['git','status','--short'],text=True)))
    return hashes


def read(path):return json.loads(Path(path).read_text())


def stats(values):
    import numpy as np
    return dict(n=len(values),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95))) if values else None


def metrics(a,b):
    a=a.detach().float().cpu();b=b.detach().float().cpu()
    return dict(equal=torch.equal(a,b),max_abs=float((a-b).abs().max()),
                different_elements=int((a!=b).sum()),elements=a.numel())
