"""Fresh decode-only attribution, separate from generation and E2E timers."""
import hashlib
import inspect
import json
import os
from pathlib import Path
from time import perf_counter
import torch
from tools.worldstream_m1.io_utils import write_json
from worldstream.m3.visual_state import base_model, expand_input, execution_signature

ROOT = Path('outputs/m3_decode_optimization')
SAMPLING = dict(do_sample=True, temperature=.1, top_p=.75, num_beams=1,
                max_new_tokens=320, use_cache=True)


def pin(dest):
    dest.mkdir(parents=True, exist_ok=False)
    files = [*Path('worldstream').rglob('*.py'), *Path('tools/worldstream_m3/decode_optimization').glob('*.py'),
             *Path('projects/mmdet3d_plugin').rglob('*.py'), Path('configs/worldstream_m1_engineering_frozen.yaml')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write_json(dest/'sources.json', hashes)
    for p in files:
        target = dest/'source'/p; target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes(p.read_bytes())
    return hashes


def load(dest):
    from tools.worldstream_m2.source_state_refinement_cuda_batch import wait_idle
    selected = wait_idle(dest, None);os.environ['CUDA_VISIBLE_DEVICES'] = str(selected['index'])
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32 = False
    import cv2
    cv2.setNumThreads(1)
    from tools.worldstream_m2.rearchitecture.integration import load as original_load
    model, dataset = original_load()
    # PEFT 0.10's generic generate wrapper + Transformers 4.31 sampling leaves
    # attention_mask=None and crashes. Call the original concrete generator;
    # all injected live LoRA modules/adapters remain in this same base model.
    model.lm_head.generate = base_model(model.lm_head).generate
    return model, dataset


class ForwardTimer:
    """Deferred CUDA events, no per-forward forced synchronize in native loop.

    CPU interval is submission wall (may include native synchronizations).
    CUDA interval includes gaps caused by host submission, not kernel-only time.
    """
    def __init__(self, base):self.base=base;self.native=base.forward;self.rows=[]
    def __enter__(self):
        def forward(*args, **kw):
            x=kw.get('inputs_embeds');x=x if x is not None else kw.get('input_ids')
            if x is None and args:x=args[0]
            stage='prefill' if kw.get('past_key_values') is None or kw.get('inputs_embeds') is not None else 'decode'
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();tick=perf_counter()
            out=self.native(*args,**kw)
            cpu=(perf_counter()-tick)*1000;b.record()
            self.rows.append((stage,a,b,cpu,int(x.shape[1])))
            return out
        self.base.forward=forward;return self
    def collect(self):
        torch.cuda.synchronize()
        return [dict(stage=s,gpu_interval_ms=a.elapsed_time(b),cpu_submission_ms=c,input_tokens=n) for s,a,b,c,n in self.rows]
    def __exit__(self,*exc):self.base.forward=self.native


@torch.no_grad()
def main():
    dest=ROOT/'profiling';hashes=pin(dest)
    model,dataset=load(dest);lm=model.lm_head;base=base_model(lm)
    # Historical carriers are real current-frame original model activations;
    # all timings/logits below are newly executed, not historical measurements.
    snaps=torch.load('outputs/m2_freeze_m3_task_serving/service/carrier_snapshots.pt',map_location='cpu')
    snap=next(s for s in snaps if s['population']=='natural_planning')
    ids=snap['ids'][0].cuda();vision=snap['carriers'].cuda()
    x=expand_input(lm,ids,vision,scope=('profile',),signature=execution_signature(lm))
    adapters=[dict(name=n,type=str(type(m)),active=list(m.active_adapters),merged=m.merged,
                   weight_dtype=str(m.base_layer.weight.dtype),A_dtype=str(next(iter(m.lora_A.values())).weight.dtype),
                   shape=list(m.base_layer.weight.shape)) for n,m in base.named_modules() if hasattr(m,'lora_A')]
    import transformers,peft
    write_json(dest/'environment.json',dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
        gpu=torch.cuda.get_device_name(),affinity=sorted(os.sched_getaffinity(0)),adapters=adapters,sampling=SAMPLING,
        input_tokens=ids.numel(),visual_tokens=int(x.visual_mask.sum()),effective_tokens=x.length,
        checkpoint='ckpts/iter_10548.pth',attention=str(type(base.model.layers[0].self_attn))))
    full=[]
    for rep in range(4):
        torch.manual_seed(123);torch.cuda.manual_seed_all(123)
        with ForwardTimer(base) as timer:
            torch.cuda.synchronize();tick=perf_counter()
            out=lm.generate(inputs=ids[None],images=vision,**SAMPLING)
            torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
            full.append(dict(repeat=rep,warmup=rep==0,wall_ms=wall,ids=out.cpu().tolist(),steps=timer.collect()))
    write_json(dest/'native_generation.json',full)
    seed=base(inputs_embeds=x.embeddings,attention_mask=x.attention_mask,position_ids=x.position_ids,use_cache=True,return_dict=True)
    kv=seed.past_key_values;token=seed.logits[:,-1:].argmax(-1);del seed
    def step(past=kv):
        return base(input_ids=token,past_key_values=past,attention_mask=torch.ones((1,x.length+1),device='cuda',dtype=torch.long),
                    position_ids=torch.tensor([[x.length]],device='cuda'),use_cache=True,return_dict=True)
    micro=[]
    for rep in range(24):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();a.record();tick=perf_counter()
        out=step();enqueue=(perf_counter()-tick)*1000;b.record();b.synchronize()
        micro.append(dict(repeat=rep,warmup=rep<4,wall_ms=(perf_counter()-tick)*1000,gpu_interval_ms=a.elapsed_time(b),cpu_submission_ms=enqueue))
        del out
    write_json(dest/'single_token.json',micro)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as p:
        for i in range(6):
            with torch.profiler.record_function('isolated_decode_token'):
                out=step();del out
        torch.cuda.synchronize()
    p.export_chrome_trace(str(dest/'decode_only_trace.json'))
    ops=[dict(name=e.key,calls=e.count,self_cpu_ms=e.self_cpu_time_total/1000,self_cuda_ms=e.self_cuda_time_total/1000) for e in p.key_averages()]
    kernels=[dict(name=e.name,duration_us=e.time_range.elapsed_us()) for e in p.events() if e.device_type==torch.autograd.DeviceType.CUDA]
    write_json(dest/'operators.json',dict(operators=ops,kernels=kernels,note='Operator CUDA attribution overlaps leaf kernel totals; never add them together',decode_calls=6))
    write_json(dest/'status.json',dict(completed=True,sources_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
        peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved()))


if __name__=='__main__':main()
