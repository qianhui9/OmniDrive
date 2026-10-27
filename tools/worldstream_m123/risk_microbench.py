"""Synchronized risk-only overhead on real checkpoint H2; no serving hooks."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from tools.worldstream_m2.common import backbone
from tools.worldstream_m2.global_residual_benchmark import read,wait_gpu
from tools.worldstream_m2.global_residual_analysis import write
from tools.worldstream_m123.benchmark.metrics import distribution


@torch.no_grad()
def run(root):
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    from worldstream.m2.dependency_planner import block0_input
    from worldstream.m2.anchor_risk import feature_statistics,risk_score
    from worldstream.m2 import M2Config
    wait_gpu();cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    cfg.data.test.test_mode=True
    cfg.data.test.pipeline=[p for p in cfg.data.test.pipeline if p['type'] in ('LoadMultiViewImageFromFiles','ResizeCropFlipRotImage','ResizeMultiview3D','NormalizeMultiviewImage','PadMultiViewImage')]
    dataset=build_dataset(cfg.data.test);tokens=read(root/'development_sequence.json')['tokens'][:2]
    by_token={v['token']:v for v in dataset.data_infos};dataset.data_infos=[by_token[t] for t in tokens]
    model=backbone();h2=[]
    for i in range(2):
        image=torch.from_numpy(np.stack(dataset[i]['img']).transpose(0,3,1,2).copy()).cuda()
        x=block0_input(model,image)
        for block in model.blocks[:2]:x=block(x)
        h2.append(x)
    previous,current=h2;config=M2Config(**read(root/'selection.json')['config'])
    def old():
        a=current.float();b=previous.float();dims=tuple(range(1,a.ndim))
        return float(((a-b).square().mean(dims)/b.square().mean(dims).clamp_min(1e-12)).sqrt().max().cpu())
    def new():return risk_score(feature_statistics(current,previous,.1),.5,1,config)[0]
    records=[]
    for repeat in range(35):
        for label,call in ([('old_rms',old),('unified_risk',new)] if repeat%2==0 else [('unified_risk',new),('old_rms',old)]):
            torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            tick=perf_counter();a.record();value=call();b.record();torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
            if repeat>=5:records.append(dict(repeat=repeat,kind=label,wall_ms=wall,cuda_span_ms=a.elapsed_time(b),score=value))
    write(root/'risk_microbenchmark.json',dict(tokens=tokens,shape=list(current.shape),dtype=str(current.dtype),records=records,
        summary={label:{k:distribution(r[k] for r in records if r['kind']==label) for k in ('wall_ms','cuda_span_ms')} for label in ('old_rms','unified_risk')},
        scope='Same real dense-prefix H2 tensors, strict checkpoint backbone loading, pre-synchronized before every call, 5 warmup + 30 alternating repeats. Isolates statistic cost; excludes prefix work, M1, suffix and serving latency.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args()
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    with gpu_job_lock():run(Path(a.root))
