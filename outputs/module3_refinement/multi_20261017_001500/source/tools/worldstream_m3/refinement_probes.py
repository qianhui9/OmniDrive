"""Controlled real-model probes, kept separate from continuous E2E timings."""
from functools import wraps
from pathlib import Path
from time import perf_counter
import torch
import torch.nn.functional as F
from worldstream.m3 import expand_input,M3Runtime
from worldstream.m3.numerics import precision_probe
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.prefill_executor import PrefillExecutor
from worldstream.m3.visual_state import base_model,execution_signature
from .experiment import write,Timer,stats

ROOT=Path('outputs/module3/clean_multi_20261016_220113/carriers')


def timed(call):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();tick=perf_counter();a.record();result=call();b.record();torch.cuda.synchronize()
    return result,dict(wall_ms=(perf_counter()-tick)*1000,gpu_ms=a.elapsed_time(b))


@torch.no_grad()
def run(model,dest):
    dest=Path(dest);dest.mkdir(exist_ok=False);base=base_model(model.lm_head)
    snap=torch.load(ROOT/'e12ae352f31d4268a7f71c4d27aa0886.pt',map_location='cpu')
    xs=[expand_input(model.lm_head,snap['ids'][i].cuda(),snap['carriers'].cuda()) for i in snap['selected_indices']]
    captured={}
    hook=base.model.layers[0].input_layernorm.register_forward_hook(lambda m,a,o:captured.update(norm=o.detach().clone()))
    base(inputs_embeds=xs[0].embeddings,position_ids=xs[0].position_ids,attention_mask=xs[0].attention_mask,use_cache=True)
    hook.remove();norm=captured['norm'];sweep=[]
    for projection in ('q_proj','k_proj','v_proj'):
        module=getattr(base.model.layers[0].self_attn,projection).base_layer
        for fp32 in (False,True):
            data=norm.float() if fp32 else norm;weight=module.weight.float() if fp32 else module.weight
            reference=F.linear(data[:,:570].contiguous(),weight)
            for length in (11,35,559,570,577,584,640):
                x=data[:,:length].contiguous() if length<=data.shape[1] else F.pad(data,(0,0,0,length-data.shape[1]))
                y,latency=timed(lambda:F.linear(x,weight));n=min(559,length)
                sweep.append(dict(projection=projection,precision=str(data.dtype),length=length,
                    common_rows=n,input_rows_bitwise_equal=torch.equal(x[:,:n],data[:,:n]),
                    max_abs=float((y[:,:n]-reference[:,:n]).abs().max()),
                    rounded_fp16_max_abs=float((y[:,:n].half()-reference[:,:n].half()).abs().max()),**latency))
    write(dest/'gemm_sweep.json',sweep)
    del captured,norm,data,weight,reference,x,y
    cache=PrefixKVCache(512*1024**2);executor=PrefillExecutor(cache)
    executor.generate(base,xs[0],max_new_tokens=1)
    saved=tuple(tuple(t.clone() for t in layer) for layer in cache.entry.kv)
    overhead={}
    from .refinement import legacy_runtime
    import sys
    legacy_runtime();legacy_expand=sys.modules['m3_frozen_legacy.visual_state'].expand_input
    legacy_match=sys.modules['m3_frozen_legacy.prefix_matcher'].exact_lcp
    vision_probe=snap['carriers'].cuda();ids_probe=snap['ids'][1].cuda()
    functions=dict(clone=lambda:cache.branch(559),borrow=lambda:cache.branch(559,read_only=True),
        match=lambda:cache.match(xs[1]),legacy_match=lambda:legacy_match(cache.entry.expanded,xs[1]),
        preparation=lambda:expand_input(model.lm_head,ids_probe,vision_probe),
        legacy_preparation=lambda:legacy_expand(model.lm_head,ids_probe,vision_probe),
        signature=lambda:execution_signature(model.lm_head))
    for name,call in functions.items():
        times=[]
        for repeat in range(12):
            value,latency=timed(call)
            if repeat>=2:times.append(latency)
            del value
        overhead[name]=dict(wall_ms=stats([x['wall_ms'] for x in times]),gpu_ms=stats([x['gpu_ms'] for x in times]))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        executor.generate(base,xs[1],cache.match(xs[1]),borrow_prefix=True,max_new_tokens=3)
    prof.export_chrome_trace(str(dest/'runtime_borrow_trace.json'))
    overhead['cache_immutable_after_borrow']=all(torch.equal(a,b) for x,y in zip(saved,cache.entry.kv) for a,b in zip(x,y))
    overhead['clone_bytes_avoided_per_branch']=sum(t[...,:559,:].numel()*t.element_size() for layer in saved for t in layer)
    write(dest/'overhead.json',overhead)
    cache.reset();del saved;torch.cuda.empty_cache()
    # The same actual original questions with altered precision on the FULL path.
    # This separates changed arithmetic from prefix scheduling. Not a speed table.
    full=[]
    for path in [ROOT/'e12ae352f31d4268a7f71c4d27aa0886.pt',ROOT/'def4504377694f7dbb99241a60d49455.pt']:
        if not path.exists():continue
        s=torch.load(path,map_location='cpu');ids=[s['ids'][i].cuda() for i in s['selected_indices']]
        vision=s['carriers'].cuda()
        reference=[]
        for policy in ('native','deterministic','no_fp16_reduction','fp32_lora'):
            with precision_probe(base,policy):
                for qi,ids_ in enumerate(ids):
                    result,latency=timed(lambda:model.lm_head.generate(inputs=ids_[None],images=vision,do_sample=False,max_new_tokens=320))
                    if policy=='native':reference.append(result.cpu())
                    full.append(dict(policy=policy,token=path.stem,question=s['questions'][s['selected_indices'][qi]],
                        equal=torch.equal(reference[qi],result.cpu()),ids=result.cpu().tolist(),**latency))
        print('FULL_PRECISION_PROBE',path.stem,flush=True)
    write(dest/'full_precision_sequences.json',full)
    ids=[snap['ids'][i].cuda() for i in snap['selected_indices'][:2]];vision=snap['carriers'].cuda()
    generation=dict(do_sample=False,max_new_tokens=320,return_dict_in_generate=True,output_scores=True)
    references=[model.lm_head.generate(inputs=x[None],images=vision,**generation) for x in ids]
    score_rows=[]
    for policy in ('native','deterministic','no_fp16_reduction','fp32_lora'):
        with precision_probe(base,policy):
            runtime=M3Runtime(dict(mode='exact',offline_experiment=True,borrow_prefix=True))
            actual,diag=runtime.generate(model.lm_head,ids,vision,context=dict(scene_id=snap['scene'],frame_id='failed_replay',camera_ids=['all']),**generation)
            for qi,(a,b) in enumerate(zip(references,actual)):
                n=min(a.sequences.shape[1],b.sequences.shape[1])
                first=next((i for i in range(1,n) if a.sequences[0,i]!=b.sequences[0,i]),None)
                deltas=[float((x-y).abs().max()) for x,y in zip(a.scores,b.scores)]
                valid=deltas if first is None else deltas[:first]
                score_rows.append(dict(policy=policy,question=snap['questions'][snap['selected_indices'][qi]],
                    equal=torch.equal(a.sequences,b.sequences),first_divergent_generated_token=first,
                    prefill_logits_error=valid[0],same_history_decode_logits_error=max(valid[1:],default=0),
                    same_history_steps=len(valid),step_max_abs_before_or_at_first_divergence=valid,
                    reference_ids=a.sequences.cpu().tolist(),actual_ids=b.sequences.cpu().tolist()))
            runtime.reset()
    write(dest/'same_history_logits.json',score_rows)
    del references,actual
    # Real batch trace + immutable prefix and fixed-RNG sampling fallback proof.
    ids=[snap['ids'][i].cuda() for i in snap['selected_indices']];vision=snap['carriers'].cuda()
    runtime=M3Runtime(dict(mode='exact',offline_experiment=True,batched_decode=True))
    context=dict(scene_id=snap['scene'],frame_id='controlled_batch',camera_ids=['all'])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        answers,diagnostics=runtime.generate(model.lm_head,ids,vision,context=context,do_sample=False,max_new_tokens=320)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(dest/'batched_decode_trace.json'))
    write(dest/'batch_proof.json',dict(diagnostics=diagnostics,ids=[x.cpu().tolist() for x in answers]))
    runtime.reset()
    # Full original max length and per-request RNG ordering are preserved by fallback.
    sample_ids=ids[:2]
    torch.manual_seed(29);reference=[model.lm_head.generate(inputs=x[None],images=vision,do_sample=True,temperature=.1,top_p=.75,max_new_tokens=320) for x in sample_ids]
    torch.manual_seed(29);answers,diag=runtime.generate(model.lm_head,sample_ids,vision,context=context,
        do_sample=True,temperature=.1,top_p=.75,max_new_tokens=320)
    write(dest/'sampling_proof.json',dict(token_equal=[torch.equal(a,b) for a,b in zip(reference,answers)],diagnostics=diag))
    runtime.reset()
