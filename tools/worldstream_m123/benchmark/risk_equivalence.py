"""Same real H2/M1/cache comparison of pre-pruning and pruned anchor updates.

Offline only: extra reference suffix work makes this entire run non-timing
evidence. The normal Runtime still publishes exactly once per actual frame.
"""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import torch
from tools.worldstream_m123.latency_controls import before_package
from tools.worldstream_m123.benchmark.metrics import distribution
from tools.worldstream_m2.global_residual_analysis import write


class AnchorEquivalence:
    def __init__(self,runtime,observer,reference,output):
        reference=Path(reference);self.runtime=runtime;self.output=Path(output)
        old=before_package(reference.parent,'m2')
        import sys
        old_global=sys.modules[old.__name__+'.global_residual'].update_global
        old_risk=sys.modules[old.__name__+'.anchor_risk']
        old_config=old.M2Config(**asdict(runtime.config))
        self.current=[];self.micro=[];self.profiled=False
        original=runtime._update_global
        def update(x,context,signature):
            original_h2=x.detach().clone()
            new=original(x,context,signature)
            count=len(observer.events)
            prior=old_global(original_h2,runtime.model.blocks,runtime.prefix,runtime.cache,context,signature,old_config)
            del observer.events[count:]
            fields=('global_feature_drift','global_world_change','global_risk_score','global_anchor_refresh_reason',
                'global_suffix_executed_blocks','global_suffix_skipped_blocks','global_anchor_age',
                'global_anchor_age_after','global_anchor_publication_version','global_anchor_frame_id')
            checks={key:new[3][key]==prior[3][key] for key in fields}
            checks.update(final_features_bitwise_equal=torch.equal(new[0],prior[0]),
                residual_bitwise_equal=torch.equal(new[1].residual,prior[1].residual),
                anchor_h2_bitwise_equal=torch.equal(new[1].h2,prior[1].h2),
                cache_version_equal=new[1].publication_version==prior[1].publication_version==runtime.cache.version+1)
            self.current.append(dict(frame_id=context['frame_id'],scene_id=context['scene_id'],checks=checks,
                before={k:prior[3][k] for k in fields},after={k:new[3][k] for k in fields}))
            if not all(checks.values()):
                write(self.output/'equivalence_failure.json',self.current[-1]);raise ValueError('Anchor pruning changed real decisions or outputs')
            if runtime.cache.global_anchor is not None:
                self.measure(original_h2,runtime.cache.global_anchor.h2,old_risk,runtime.config)
            return new
        runtime._update_global=update

    def measure(self,current,previous,old,config):
        from worldstream.m2.anchor_risk import feature_statistics,risk_score
        if len(self.micro)>=160:return
        def before():return old.risk_score(old.feature_statistics(current,previous,.1),.5,1,config)[0]
        def after():return risk_score(feature_statistics(current,previous,.1,include_local=False),.5,1,config)[0]
        for repeat in range(4):
            for name,call in ([('before',before),('after',after)] if repeat%2==0 else [('after',after),('before',before)]):
                torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start=perf_counter();a.record();score=call();b.record();torch.cuda.synchronize()
                if repeat:self.micro.append(dict(kind=name,wall_ms=(perf_counter()-start)*1000,cuda_span_ms=a.elapsed_time(b),score=score))
        if not self.profiled:
            self.profiled=True;self.kernels={}
            for name,call in [('before',before),('after',after)]:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:call();torch.cuda.synchronize()
                self.kernels[name]=[dict(name=e.key,calls=e.count,self_cuda_time_us=e.self_cuda_time_total) for e in prof.key_averages()]
                prof.export_chrome_trace(str(self.output/('risk_'+name+'_trace.json')))

    def clear(self):self.current.clear()

    def collect(self):
        if len(self.current)!=1:raise ValueError('Expected one anchor update per frame')
        row=self.current[0]
        if self.runtime.cache.version!=row['after']['global_anchor_publication_version']:raise ValueError('Atomic cache version did not publish')
        return row

    def finish(self):
        write(self.output/'risk_pruning_microbenchmark.json',dict(records=self.micro,
            summary={label:{key:distribution(r[key] for r in self.micro if r['kind']==label) for key in ('wall_ms','cuda_span_ms')} for label in ('before','after')},
            profiler=self.kernels,scope='Same real checkpoint H2 and prior anchor, pre-synchronized alternating controls, warmup excluded. Extra reference suffixes and profiling run only in this diagnostic job.'))
