"""External, profiling-only observers; no inference or refresh decisions."""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import cProfile
import json
from pathlib import Path
import pstats
import time

import torch


class CriticalPathProbe:
    def __init__(self):
        self.enabled=False
        self.rows=[]
        self.originals=[]

    def wrap(self,obj,name,label):
        original=getattr(obj,name)
        @wraps(original)
        def measured(*args,**kwargs):
            if not self.enabled:return original(*args,**kwargs)
            wall=time.perf_counter();cpu=time.process_time();thread=time.thread_time()
            with torch.autograd.profiler.record_function('critical/'+label):
                result=original(*args,**kwargs)
            self.rows.append(dict(stage=label,host_ms=(time.perf_counter()-wall)*1000,
                process_cpu_ms=(time.process_time()-cpu)*1000,thread_cpu_ms=(time.thread_time()-thread)*1000))
            return result
        setattr(obj,name,measured);self.originals.append((obj,name,original))

    def install(self,model):
        import worldstream.m1.omnidrive_adapter as adapter
        import worldstream.m1.state_manager as manager
        import worldstream.m1.geometry as geometry
        import worldstream.m1.temporal_visibility as temporal
        import worldstream.m2.runtime as vision
        import worldstream.m2.window_executor as window
        from worldstream.m1.state_store import StateStore
        from worldstream.m2.state_cache import StateCache
        for name in ('frame_from_model','outputs_from_model','numpy_copy','quaternion_transform'):
            self.wrap(adapter,name,'M1/'+name)
        for name in ('associate','frame_digest','project_anchor','predict_geometry'):
            self.wrap(manager,name,'M1/'+name)
        for name in ('precheck_with_observation','_precheck','commit','_commit','_history_digest','export_m2_inputs'):
            self.wrap(manager.WorldStateManager,name,'M1/'+name)
        for name in ('fork','source_digest'):
            self.wrap(StateStore,name,'M1/store_'+name)
        for name in ('estimate_temporal_evidence','gray_image'):
            if hasattr(temporal,name):self.wrap(temporal,name,'M1/'+name)
        for name in ('transform_points',):self.wrap(adapter,name,'M1/'+name)
        for name in ('computation_admission','block0_input'):
            self.wrap(vision,name,'M2/'+name)
        for name in ('_plan_refresh','_execute','_update_global','forward_from_m1'):
            self.wrap(vision.IncrementalVisionRuntime,name,'M2/'+name)
        for name in ('commit','mismatch'):self.wrap(StateCache,name,'M2/cache_'+name)
        for name in ('_grid','execute'):self.wrap(window.WindowExecutor,name,'M2/window_'+name)
        self.wrap(window,'partition','M2/partition')

    def clear(self):self.rows=[]

    def summary(self):
        grouped=defaultdict(list)
        for r in self.rows:grouped[r['stage']].append(r)
        return {k:dict(calls=len(rs),**{s:sum(r[s] for r in rs) for s in ('host_ms','process_cpu_ms','thread_cpu_ms')}) for k,rs in grouped.items()}


@contextmanager
def profile_frame(enabled,dest):
    if not enabled:
        yield
        return
    cpu=cProfile.Profile();cpu.enable()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,profile_memory=False,with_stack=False) as prof:
        yield
    cpu.disable()
    stats=pstats.Stats(cpu)
    functions=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_ms=v[2]*1000,cumulative_ms=v[3]*1000)
        for k,v in stats.stats.items() if ('worldstream/m1' in k[0] or 'worldstream/m2' in k[0])]
    functions.sort(key=lambda r:r['cumulative_ms'],reverse=True)
    operators=[dict(name=e.key,calls=e.count,cpu_total_ms=e.cpu_time_total/1000,
        self_cpu_ms=e.self_cpu_time_total/1000,cuda_total_ms=e.cuda_time_total/1000,
        self_cuda_ms=e.self_cuda_time_total/1000) for e in prof.key_averages()]
    trace=Path('/tmp')/('omnidrive-critical-'+str(__import__('os').getpid())+'.json')
    prof.export_chrome_trace(str(trace));events=json.loads(trace.read_text())['traceEvents'];trace.unlink()
    kernels=[e for e in events if e.get('cat')=='kernel' and 'dur' in e]
    copies=[e for e in events if 'memcpy' in e.get('cat','').lower() and 'dur' in e]
    runtime=[e for e in events if e.get('cat')=='cuda_runtime' and 'dur' in e]
    def aggregate(es):
        groups=defaultdict(list)
        for e in es:groups[e['name']].append(e['dur']/1000)
        return sorted([dict(name=k,calls=len(v),total_ms=sum(v)) for k,v in groups.items()],key=lambda x:x['total_ms'],reverse=True)
    # Attribute GPU events through the CPU launch's external correlation ID and
    # nested record_function intervals. This separates visual work from LLM work.
    ranges=[e for e in events if e.get('name','').startswith('critical/M2/') and 'dur' in e]
    launches={e['args']['correlation']:e for e in runtime if 'correlation' in e.get('args',{})}
    regional=defaultdict(list)
    for e in kernels+copies:
        launch=launches.get(e.get('args',{}).get('correlation'))
        if launch:
            hits=[r for r in ranges if r['ts']<=launch['ts']<=r['ts']+r['dur']]
            for r in hits:regional[r['name']].append(e)
    result=dict(heavy_profiler=True,timing_not_formal=True,kernels=aggregate(kernels),copies=aggregate(copies),
        cuda_runtime=aggregate(runtime),kernel_count=len(kernels),kernel_compute_ms=sum(e['dur'] for e in kernels)/1000,
        copy_ms=sum(e['dur'] for e in copies)/1000,
        visual_ranges={k:dict(kernel_count=sum(e.get('cat')=='kernel' for e in es),
            kernel_ms=sum(e['dur'] for e in es if e.get('cat')=='kernel')/1000,
            copy_ms=sum(e['dur'] for e in es if e.get('cat')!='kernel')/1000,
            kernels=aggregate([e for e in es if e.get('cat')=='kernel'])) for k,es in regional.items()},
        operators=operators,cprofile=functions[:120],raw_trace_retained=False,
        timing_caveat='CUDA range span includes launches/gaps; sum of kernel durations is work, not wall span. Nested ranges overlap.')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
