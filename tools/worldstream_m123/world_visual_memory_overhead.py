"""Small real-regional-tensor memory transaction costs, without the LLM."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import torch

from tools.worldstream_m123.benchmark.common import write
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m2.global_residual_benchmark import wait_gpu
from tools.worldstream_m123.world_visual_memory_semantics import context, regions, replay
from tools.worldstream_m123.world_visual_memory_analysis import distribution
from worldstream.m3.visual_memory import VisualMemoryManager, VisualMemoryProjection


@torch.no_grad()
def run(root):
    # Entire untouched natural sequence, including all no-memory frames.
    natural=root/'natural_audit'
    life={str(k):replay(natural,k) for k in (64,128,256)}
    write(natural/'memory_lifecycle.json',dict(events=life,scope='All24 original natural development VAL frames, no positive-event selection'))
    nrows=[json.loads(l) for l in (natural/'bridge_capture/frames.jsonl').open()]
    nfeatures=torch.load(natural/'bridge_capture/region_features.pt',weights_only=False,map_location='cpu')
    indexed={r['frame_id']:regions([x for x in nfeatures if x['source_frame']==r['frame_id']]) for r in nrows}
    qualification=json.loads((root/'experiment_plan.json').read_text())['qualification_sha256']
    update_cost=[]
    for rep in range(11):
        manager=VisualMemoryManager(64)
        for row in nrows:
            start=perf_counter()
            memory,event=manager.update(context(row),indexed[row['frame_id']],row['qualified_ids'],qualification)
            elapsed=(perf_counter()-start)*1000
            assert memory is None
            if rep>=2:update_cost.append(dict(repeat=rep-2,scene_id=row['scene_id'],frame_id=row['frame_id'],wall_ms=elapsed,event=event))
    write(root/'natural_memory_update_cost.json',dict(records=update_cost,
        wall_ms=distribution([r['wall_ms'] for r in update_cost]),
        scope='Actual CPU memory update of natural no-memory path, current real qualified IDs and regions. No GT input. No GPU work in this path.'))
    chosen=json.loads((root/'controlled_cost/summary.json').read_text())
    frames=[json.loads(l) for l in (root/'bridge_capture/frames.jsonl').open()]
    captures=torch.load(root/'bridge_capture/region_features.pt',weights_only=False,map_location='cpu')
    arow=next(r for r in frames if r['frame_id']==chosen['source_frame'])
    brow=next(r for r in frames if r['frame_id']==chosen['target_frame'])
    first=regions([x for x in captures if x['source_frame']==arow['frame_id']])
    second=regions([x for x in captures if x['source_frame']==brow['frame_id']])
    rows=[];probes=[]
    with gpu_job_lock():
        wait_gpu();torch.manual_seed(20261028)
        first=[replace(r,features=r.features.cuda()) for r in first]
        second=[replace(r,features=r.features.cuda()) for r in second]
        projection=VisualMemoryProjection().cuda().eval().requires_grad_(False)
        def timed(fn):
            torch.cuda.synchronize();a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            tick=perf_counter();a.record();value=fn();b.record();b.synchronize()
            return value,dict(wall_ms=(perf_counter()-tick)*1000,cuda_stream_ms=a.elapsed_time(b))
        for k in (64,128,256):
            for rep in range(11):
                manager=VisualMemoryManager(k)
                (memory,_),cold=timed(lambda:manager.update(context(arow),first,arow['qualified_ids'],qualification))
                assert memory is not None
                (_,event),warm=timed(lambda:manager.update(context(brow),second,brow['qualified_ids'],qualification))
                assert event=='keep'
                projection._key=None
                _,project=timed(lambda:projection(memory))
                if rep>=2:rows.append(dict(K=k,repeat=rep-2,create=cold,keep=warm,projection=project))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                manager=VisualMemoryManager(k)
                manager.update(context(arow),first,arow['qualified_ids'],qualification)
                torch.cuda.synchronize()
            gpu=[e for e in prof.events() if str(e.device_type).endswith('CUDA')]
            probes.append(dict(K=k,GPU_activity_ms=sum(e.time_range.elapsed_us() for e in gpu)/1000,
                events=len(gpu),scope='Separate actual memory creation activity probe, includes finite check/hash D2H/allocation. Projection/bridge excluded.'))
    write(root/'memory_transaction_cost.json',dict(records=rows,gpu_activity_probes=probes,
        source_frame=chosen['source_frame'],target_frame=chosen['target_frame'],
        scope='Real captured source/target regional tensors, CUDA lifecycle and projection; extraction and M1 read measured separately in live captures.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/m3_world_visual_memory_run01'))
    run(p.parse_args().root)
