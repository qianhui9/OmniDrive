"""Supplementary original-model prefix witnesses and native decode trace."""
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import torch
from worldstream.m3 import M3Runtime,expand_input
from worldstream.m3.visual_state import base_model,execution_signature
from worldstream.m3.prefix_matcher import exact_lcp
from worldstream.m3.prefix_kv_cache import PrefixKVCache,kv_bytes
from worldstream.m3.prefill_executor import PrefillExecutor
from tools.worldstream_m1.io_utils import write_json


@torch.no_grad()
def run_checks(model,snapshots,dest):
    dest=Path(dest);dest.mkdir(exist_ok=False)
    from .task_serving_supplement import run_supplement
    run_supplement(model,snapshots,dest/'supplement')
    lm=model.lm_head;base=base_model(lm);device=base.get_input_embeddings().weight.device
    snap=next(s for s in snapshots if s['population']=='natural_planning');ids=snap['ids'][0].to(device);vision=snap['carriers'].to(device)
    signature=execution_signature(lm);x=expand_input(lm,ids,vision,scope=(snap['scene'],snap['frame']),signature=signature,source=(snap['scene'],snap['frame'],0,0))
    cache=PrefixKVCache(512*1024**2);executor=PrefillExecutor(cache);kw=dict(do_sample=False,max_new_tokens=320,use_cache=True)
    reference=lm.generate(inputs=ids[None],images=vision,**kw)
    independent,first=executor.generate(base,x,profile=True,**kw)
    original=tuple(tuple(t.clone() for t in layer) for layer in cache.entry.kv)
    torch.cuda.synchronize();tick=perf_counter();repeated,second=executor.generate(base,x,x.length,profile=True,**kw);torch.cuda.synchronize();repeat_wall=(perf_counter()-tick)*1000
    changed=vision.clone();changed[0,0,0]+=1
    y=expand_input(lm,ids,changed,scope=x.scope,signature=signature,source=('changed',))
    boundaries=dict(identical=exact_lcp(x,x),changed_vision=exact_lcp(x,y),scene_change=exact_lcp(x,replace(x,scope=('new_scene',))),model_version_change=exact_lcp(x,replace(x,signature=('changed_weights',))))
    for name in ('position_ids','attention_mask','token_ids','visual_mask'):
        z=x.prefix(x.length,clone=True);getattr(z,name)[0,3]=not bool(getattr(z,name)[0,3]) if name=='visual_mask' else 0 if name=='attention_mask' else int(getattr(z,name)[0,3])+1;boundaries[name+'_changed']=exact_lcp(x,z)
    near_ids=ids.clone();near_ids[-2]=ids[-2]+1
    near=expand_input(lm,near_ids,vision,scope=x.scope,signature=signature);boundaries['different_question_token']=exact_lcp(x,near)
    result=dict(population='controlled_identical_original_request_on_real_carriers',input_length=x.length,visual_tokens=int(x.visual_mask.sum()),boundaries=boundaries,
        native_independent_token_ids_equal=torch.equal(reference,independent),native_repeated_token_ids_equal=torch.equal(reference,repeated),
        cache_immutable=all(torch.equal(a,b) for old,new in zip(original,cache.entry.kv) for a,b in zip(old,new)),
        first=first,repeated=second,repeated_wall_ms=repeat_wall,kv_bytes=kv_bytes(cache.entry.kv),computed_source=list(cache.entry.computed_source),
        kv_shape=list(cache.entry.kv[0][0].shape),kv_dtype=str(cache.entry.kv[0][0].dtype),all_original_visual_tokens_retained=True,
        semantic_matching_used=False,production_auto_enabled=False)
    sampling=dict(do_sample=True,temperature=.1,top_p=.75,max_new_tokens=320,use_cache=True)
    torch.manual_seed(21);torch.cuda.manual_seed_all(21)
    sampled_full=base.generate(inputs=ids[None],images=vision,**sampling)
    torch.manual_seed(21);torch.cuda.manual_seed_all(21)
    sampled_cached,sampling_diag=executor.generate(base,x,x.length,**sampling)
    result['original_sampling_identical_request']=dict(temperature=.1,top_p=.75,token_ids_equal=torch.equal(sampled_full,sampled_cached),cached_diagnostics=sampling_diag,
        note='Actual original base.generate, bypasses runner greedy wrapper; active embedded LoRA unchanged')
    del sampled_full,sampled_cached
    del original,cache,executor,independent,repeated,reference
    torch.cuda.empty_cache()
    # Actual original decode, unmodified masks/weights; short output only to
    # bound trace size. This does not estimate full-answer speed by token ratio.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as p:
        lm.generate(inputs=ids[None],images=vision,do_sample=False,max_new_tokens=5,use_cache=True);torch.cuda.synchronize()
    p.export_chrome_trace(str(dest/'native_decode_trace.json'))
    events=[]
    for e in p.key_averages():
        events.append(dict(name=e.key,calls=e.count,self_cpu_ms=e.self_cpu_time_total/1000,self_cuda_ms=e.self_cuda_time_total/1000))
    # A true shared prefix can still change floating-point results when the
    # full/suffix GEMM shapes differ. Preserve logits/KV errors, never relax the
    # frozen greedy whole-output target after observing service failures.
    multi=next(s for s in snapshots if s['population']=='native_multi')
    v=multi['carriers'].to(device)
    a=expand_input(lm,multi['ids'][0].to(device),v,scope=('numerical_probe',),signature=signature,source=('q0',))
    b=expand_input(lm,multi['ids'][1].to(device),v,scope=a.scope,signature=signature,source=('q1',))
    c=PrefixKVCache(512*1024**2)
    first=base(inputs_embeds=a.embeddings,attention_mask=a.attention_mask,position_ids=a.position_ids,use_cache=True,return_dict=True)
    c.put(a,first.past_key_values,first.logits);del first
    n=c.match(b)
    full=base(inputs_embeds=b.embeddings,attention_mask=b.attention_mask,position_ids=b.position_ids,use_cache=True,return_dict=True)
    suffix=base(inputs_embeds=b.embeddings[:,n:],position_ids=b.position_ids[:,n:],attention_mask=b.attention_mask,past_key_values=c.branch(n),use_cache=True,return_dict=True)
    delta=(full.logits[:,-1]-suffix.logits[:,-1]).float()
    result['native_related_question_numerics']=dict(exact_input_prefix=n,full_length=b.length,suffix_length=b.length-n,
        final_prefill_logit_max_abs=float(delta.abs().max()),last_argmax_equal=torch.equal(full.logits[:,-1].argmax(-1),suffix.logits[:,-1].argmax(-1)),
        shared_KV_max_abs=max(float((s[...,:n,:]-t[...,:n,:]).abs().max()) for layer,other in zip(c.entry.kv,full.past_key_values) for s,t in zip(layer,other)),
        frozen_target_relaxed=False,task_quality_certificate=False)
    del full,suffix,c
    result['trace']=dict(max_new_tokens=5,scope='Actual native one prefill plus four decode steps, diagnostic only; profiler overhead excluded from service comparison',top_cuda=sorted(events,key=lambda r:r['self_cuda_ms'],reverse=True)[:30],top_cpu=sorted(events,key=lambda r:r['self_cpu_ms'],reverse=True)[:30])
    write_json(dest/'results.json',result)
