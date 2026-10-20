"""Bounded fallback analysis; no Attention or original KV interface is changed."""
import json
from pathlib import Path
import torch
from tools.worldstream_m1.io_utils import write_json
from worldstream.m3.prefix_kv_cache import PrefixKVCache, kv_bytes
from worldstream.m3.batched_executor import BatchedExecutor
from worldstream.m3.visual_state import expand_input, base_model
from .probe import measure, SAMPLING


@torch.no_grad()
def run(model, snapshots, root):
    dest=root/'alternative';dest.mkdir(parents=True,exist_ok=False)
    base=base_model(model.lm_head);snap=snapshots[0]
    ids=torch.tensor(snap['prompt_ids'],device='cuda')
    x=expand_input(model.lm_head,ids,snap['carriers'].cuda())
    seed=base(inputs_embeds=x.embeddings,attention_mask=x.attention_mask,position_ids=x.position_ids,
              use_cache=True,return_dict=True)
    kv=seed.past_key_values;token=seed.logits[:,-1:].argmax(-1)
    def step():
        return base(input_ids=token,past_key_values=kv,attention_mask=x.attention_mask.new_ones((1,x.length+1)),
                    position_ids=x.position_ids.new_tensor([[x.length]]),use_cache=True,return_dict=True)
    rows=[]
    for rep in range(12):
        out,timing=measure(step);rows.append(dict(repeat=rep,warmup=rep<2,**timing));del out
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as profile:
        for _ in range(4):
            out=step();del out
        torch.cuda.synchronize()
    operators=[dict(name=e.key,calls=e.count,self_cpu_ms=e.self_cpu_time_total/1000,
                    self_cuda_ms=e.self_cuda_time_total/1000) for e in profile.key_averages()]
    profile.export_chrome_trace(str(dest/'native_decode_trace.json'))
    write_json(dest/'operators.json',operators)
    # The following measures storage operations on the REAL legacy cache. It
    # does not substitute for running Attention with a different layout.
    buffers=tuple(tuple(t.new_empty((*t.shape[:-2],x.length+320,t.shape[-1])) for t in layer) for layer in kv)
    for layer,dst in zip(kv,buffers):
        for t,b in zip(layer,dst):b[...,:x.length,:].copy_(t)
    def concatenate():
        return tuple(tuple(torch.cat((t,t[...,-1:,:]),dim=2) for t in layer) for layer in kv)
    def append_only():
        for layer,dst in zip(kv,buffers):
            for t,b in zip(layer,dst):b[...,x.length:x.length+1,:].copy_(t[...,-1:,:])
        return tuple(tuple(b[...,:x.length+1,:] for b in layer) for layer in buffers)
    storage=[]
    for rep in range(16):
        cats,cat_time=measure(concatenate);appended,append_time=measure(append_only)
        equal=all(torch.equal(a,b) for la,lb in zip(cats,appended) for a,b in zip(la,lb))
        storage.append(dict(repeat=rep,warmup=rep<3,cat=cat_time,append=append_time,equal=equal,
            append_contiguous=all(t.is_contiguous() for layer in appended for t in layer)))
        del cats,appended
    del buffers
    batch= BatchedExecutor(PrefixKVCache(512*1024**2))
    result,batch_diag=batch.try_generate(base,[x,x],**SAMPLING)
    assert result is None
    requests=[json.loads(s) for s in (root/'phase1/requests.jsonl').read_text().splitlines()]
    frames={}
    for r in requests:
        frames.setdefault((r['population'],r['scene'],r['frame']),[]).append(r)
    import numpy as np
    cat_cuda=sum(r['self_cuda_ms'] for r in operators if r['name']=='aten::cat')/4
    token_gpu=float(np.median([r['gpu_ms'] for r in rows if not r['warmup']]))
    artifact=dict(implemented=False,production_enabled=False,
        single_token=rows,storage_operation_microbench=storage,operators_path=str(dest/'operators.json'),
        measured_cat_self_cuda_ms_per_decode=cat_cuda,measured_token_gpu_interval_ms=token_gpu,
        optimistic_decode_speedup_upper_bound=token_gpu/max(1e-9,token_gpu-cat_cuda),
        caveat='aten::cat includes all concatenations, so this overestimates removable KV cost; '
               'operator GPU attribution is not added to leaf kernels. Storage copy is not an Attention benchmark.',
        actual_kv_bytes=kv_bytes(kv),preallocated_bytes=kv_bytes(kv)*(x.length+320)//x.length,
        cats_per_decode=sum(r['calls'] for r in operators if r['name']=='aten::cat')/4,
        compatibility='Transformers 4.31 LlamaAttention performs out-of-place torch.cat inside forward. '
            'A tuple adapter cannot eliminate those calls without replacing Attention or globally intercepting torch.cat. '
            'A buffer view also changes K/V strides. Prefix cache owns immutable clones; temporal heads have separate state.',
        append_adapter_decision='no_go_without_attention_rewrite_and_new_numerical_certificate',
        microbatch=dict(active_native_generations=1,frame_question_counts=[len(rs) for rs in frames.values()],
            available_same_frame_tasks_max=max(map(len,frames.values())),
            generated_lengths=[len(r['generated_token_ids']) for r in requests],
            installed_executor_fallback=batch_diag,
            decision='no_go_original_sampling_contract',
            reason='Same-frame questions exist in expanded native_multi, but original decoder submits serially. '
                   'Installed batch executor rejects do_sample=True before execution to preserve request RNG and EOS. '
                   'No synthetic concurrent workload or greedy substitution was benchmarked.'))
    write_json(root/'alternative_kv_feasibility.json',artifact)
    return artifact
