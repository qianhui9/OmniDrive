import argparse
import hashlib
import os
from pathlib import Path
import torch
from .common import ROOT,OLD,pin,load,write_json
from . import attribution,feasibility


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT));args=p.parse_args();root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False);hashes=pin(root/'probe')
    model,dataset=load(root/'probe')
    snapshots=torch.load(OLD/'phase1/carrier_snapshots.pt',map_location='cpu')
    import transformers,peft
    write_json(root/'environment.json',dict(python=os.sys.executable,torch=torch.__version__,transformers=transformers.__version__,
        peft=peft.__version__,gpu=torch.cuda.get_device_name(),affinity=sorted(os.sched_getaffinity(0)),
        checkpoint='ckpts/iter_10548.pth',sampling='original do_sample=True temperature=.1 top_p=.75 default top_k num_beams=1 max_new_tokens=320 use_cache=True'))
    attribution.run(model,snapshots,root)
    feasibility.run(model,snapshots,root)
    write_json(root/'probe/status.json',dict(completed=True,sources_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())))


if __name__=='__main__':main()
