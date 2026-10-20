"""Matched one/many-token microbench and complete original sampled requests."""
import hashlib
import json
import os
from time import perf_counter
from functools import wraps
import torch
from worldstream.m3.native_decode import StaticDecodeGraphs
from worldstream.m3.visual_state import base_model,expand_input,execution_signature
from tools.worldstream_m1.io_utils import write_json
from .profile import ROOT,SAMPLING,pin,load,ForwardTimer


class SynchronousTimer(ForwardTimer):
    """Separate barrier-instrumented full-generation attribution, not service E2E."""
    def __enter__(self):
        self.first_decode=None
        if hasattr(self,'call'):
            self.base.forward=self.call
            return self
        @wraps(self.native)
        def call(*args,**kw):
            stage='prefill' if kw.get('inputs_embeds') is not None or kw.get('past_key_values') is None else 'decode'
            x=kw.get('inputs_embeds');x=x if x is not None else kw.get('input_ids')
            torch.cuda.synchronize()
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            tick=perf_counter()
            if stage=='decode' and self.first_decode is None:self.first_decode=tick
            a.record();out=self.native(*args,**kw);b.record();b.synchronize()
            self.rows.append(dict(stage=stage,wall_ms=(perf_counter()-tick)*1000,gpu_interval_ms=a.elapsed_time(b),input_tokens=int(x.shape[1])))
            return out
        self.call=call;self.base.forward=call;return self
    def collect(self):return list(self.rows)


@torch.no_grad()
def main():
    dest=ROOT/'micro';hashes=pin(dest);model,_=load(dest);lm=model.lm_head;base=base_model(lm)
    snapshots=torch.load(ROOT/'service/carrier_snapshots.pt',map_location='cpu')
    snap=snapshots[0];vision=snap['carriers'].cuda();ids=snap['ids'][0].cuda()
    x=expand_input(lm,ids,vision,scope=('micro',),signature=execution_signature(lm))
    seed=base(inputs_embeds=x.embeddings,position_ids=x.position_ids,attention_mask=x.attention_mask,use_cache=True,return_dict=True)
    kv=seed.past_key_values;token=seed.logits[:,-1].argmax(-1)[:,None];del seed
    graphs=StaticDecodeGraphs();rows=[];cold=[]
    for mode in ('native','graph'):
        from contextlib import nullcontext
        ctx=graphs.activate(lm) if mode=='graph' else nullcontext({})
        with ctx as diag:
            for length in (1,8):
                for rep in range(24):
                    state=kv;t=token
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize();a.record();tick=perf_counter()
                    for step in range(length):
                        pos=x.length+step
                        out=base(input_ids=t,past_key_values=state,attention_mask=torch.ones((1,pos+1),dtype=torch.long,device='cuda'),position_ids=torch.tensor([[pos]],device='cuda'),use_cache=True,return_dict=True)
                        state=out.past_key_values;t=out.logits[:,-1].argmax(-1)[:,None]
                    enqueue=(perf_counter()-tick)*1000;b.record();b.synchronize()
                    row=dict(mode=mode,tokens=length,repeat=rep,warmup=rep<4,wall_ms=(perf_counter()-tick)*1000,
                        cpu_submission_ms=enqueue,gpu_interval_ms=a.elapsed_time(b),last_token=int(t),peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved())
                    rows.append(row)
                    del out,state
        cold.append(dict(mode=mode,diagnostics=diag))
    write_json(dest/'microbench.json',dict(rows=rows,diagnostics=cold,scope='Fixed initial real prefill; one/eight autoregressive forward steps; not task output budget'))
    del kv;graphs.clear();torch.cuda.empty_cache()
    full=[];refs={};timer=SynchronousTimer(base)
    for si,snap in enumerate(snapshots):
        vision=snap['carriers'].cuda()
        for qi,raw in enumerate(snap['ids']):
            ids=raw.cuda()
            for rep in range(3):
                for mode in ('native','graph'):
                    torch.manual_seed(123+qi);torch.cuda.manual_seed_all(123+qi)
                    timer.rows.clear()
                    with timer:
                        ctx=graphs.activate(lm) if mode=='graph' else nullcontext({})
                        torch.cuda.synchronize();tick=perf_counter()
                        with ctx as diag:answer=lm.generate(inputs=ids[None],images=vision,**SAMPLING)
                        torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
                        decode_span=(perf_counter()-timer.first_decode)*1000 if timer.first_decode else 0.
                        invocations=timer.collect()
                    key=(si,qi,rep)
                    if mode=='native':refs[key]=(answer.clone(),torch.cuda.get_rng_state())
                    equal,rng=(True,True) if mode=='native' else (torch.equal(refs[key][0],answer),torch.equal(refs[key][1],torch.cuda.get_rng_state()))
                    row=dict(snapshot=si,scene=snap['scene'],frame=snap['frame'],question_index=qi,repeat=rep,warmup=rep==0,mode=mode,
                        wall_ms=wall,decode_span_wall_ms=decode_span,invocations=invocations,ids=answer.cpu().tolist(),token_equal=equal,rng_equal=rng,diagnostics=diag,
                        peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved())
                    full.append(row)
                    with (dest/'full_generation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(si,qi,rep,mode,round(wall,2),equal,flush=True)
    # Optional trace: torch 1.13's profiler stalled after collecting graphed
    # kernels on this host. Keep complete measurements independent of that tool.
    trace_requested=os.environ.get('OMNIDRIVE_TRACE_OPTIMIZED_DECODE')=='1'
    write_json(dest/'status.json',dict(completed=True,full_generations=len(full),
        measurement_completed=True,trace_requested=trace_requested,trace_completed=False,
        sources_unchanged=all(hashlib.sha256(__import__('pathlib').Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())))
    if trace_requested:
        snap=snapshots[0];x=expand_input(lm,snap['ids'][0].cuda(),snap['carriers'].cuda(),scope=('trace',),signature=execution_signature(lm))
        seed=base(inputs_embeds=x.embeddings,attention_mask=x.attention_mask,position_ids=x.position_ids,use_cache=True,return_dict=True)
        with graphs.activate(lm):
            kw=dict(input_ids=seed.logits[:,-1:].argmax(-1),past_key_values=seed.past_key_values,attention_mask=torch.ones((1,x.length+1),device='cuda',dtype=torch.long),position_ids=torch.tensor([[x.length]],device='cuda'),use_cache=True,return_dict=True)
            base(**kw)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
                for _ in range(6):base(**kw)
                torch.cuda.synchronize()
            p.export_chrome_trace(str(dest/'optimized_decode_trace.json'))
        status=json.loads((dest/'status.json').read_text());status['trace_completed']=True
        write_json(dest/'status.json',status)


if __name__=='__main__':main()
