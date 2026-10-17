"""Additional real-model cache-capture/immutability and residual-path evidence."""
from pathlib import Path
import torch
from worldstream.m3 import M3Runtime,expand_input
from worldstream.m3.visual_state import base_model
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.prefix_matcher import exact_lcp
from .experiment import write,stats
from .numerical_refinement import LayerTrace,compare
from .refinement_probes import timed


class ResidualTrace(LayerTrace):
    def __enter__(self):
        super().__enter__()
        for name,module in self.model.named_modules():
            if name.endswith('post_attention_layernorm'):
                def before(m,a,name=name):self.save(name+'.residual_input',a[0])
                self.handles.append(module.register_forward_pre_hook(before))
        return self


@torch.no_grad()
def run(model,dest):
    dest=Path(dest);dest.mkdir(exist_ok=False);lm=model.lm_head;base=base_model(lm)
    path=Path('outputs/module3/clean_multi_20261016_220113/carriers/e12ae352f31d4268a7f71c4d27aa0886.pt')
    snap=torch.load(path,map_location='cpu');vision=snap['carriers'].cuda();ids=[snap['ids'][i].cuda() for i in snap['selected_indices']]
    xs=[expand_input(lm,x,vision) for x in ids];cache=PrefixKVCache(512*1024**2)
    original=base(inputs_embeds=xs[0].embeddings,position_ids=xs[0].position_ids,attention_mask=xs[0].attention_mask,use_cache=True)
    captures=[]
    for rep in range(12):
        result,latency=timed(lambda:cache.put(xs[0],original.past_key_values,original.logits))
        assert result
        if rep>=2:captures.append(latency)
    capture=dict(wall_ms=stats([r['wall_ms'] for r in captures]),gpu_interval_ms=stats([r['gpu_ms'] for r in captures]),
        cache_bytes=cache.nbytes,samples=10,warmup=2,scope='Owned persistent full-prompt KV+expanded input+last logits capture; warm CUDA allocator, cold cache slot')
    write(dest/'capture_cost.json',capture);cache.reset();del original
    evidence=[]
    common=min(exact_lcp(xs[0],x) for x in xs[1:]);legacy=exact_lcp(xs[0],xs[1])
    for name,seed_x,prefix in [('original_full_seed',xs[0],legacy),('independent_common_seed',xs[0].prefix(common),common)]:
        target=xs[1]
        positions=sorted(set([0,34,35,100,300,prefix-1]+list(range(prefix,target.length))))
        with ResidualTrace(base,positions) as source_trace:
            seed=base(inputs_embeds=seed_x.embeddings,position_ids=seed_x.position_ids,
                attention_mask=seed_x.attention_mask,use_cache=True)
        kv=tuple(tuple(t[...,:prefix,:] for t in layer) for layer in seed.past_key_values)
        preserved=tuple(tuple(t.clone() for t in layer) for layer in kv)
        with ResidualTrace(base,positions) as full_trace:
            full=base(inputs_embeds=target.embeddings,position_ids=target.position_ids,attention_mask=target.attention_mask,use_cache=True)
        with ResidualTrace(base,positions,offset=prefix) as suffix_trace:
            suffix=base(inputs_embeds=target.embeddings[:,prefix:],position_ids=target.position_ids[:,prefix:],
                attention_mask=target.attention_mask,past_key_values=kv,use_cache=True)
        unchanged=all(torch.equal(a,b) for x,y in zip(kv,preserved) for a,b in zip(x,y));assert unchanged
        values=dict(strategy=name,seed_tokens=seed_x.length,prefix=prefix,target_tokens=target.length,
            input_prefix_bitwise_equal=torch.equal(seed_x.embeddings[:,:prefix],target.embeddings[:,:prefix]),
            prefix_operators=compare(source_trace.data,full_trace.data,range(prefix)),
            suffix_operators=compare(full_trace.data,suffix_trace.data,range(prefix,target.length)),
            prefix_kv=[dict(layer=i,key_max_abs=float((a[0]-b[0][...,:prefix,:]).abs().max()),
                value_max_abs=float((a[1]-b[1][...,:prefix,:]).abs().max())) for i,(a,b) in enumerate(zip(kv,full.past_key_values))],
            prefill_logits_error=float((full.logits[:,-1]-suffix.logits[:,-1]).abs().max()),prefix_unchanged_after_suffix=unchanged)
        evidence.append(values)
        torch.save(dict(source=source_trace.data,full=full_trace.data,suffix=suffix_trace.data),dest/(name+'_residual_intermediates.pt'))
        del full,suffix,seed,kv,preserved
    write(dest/'numerical_paths.json',evidence)
    # Check the full greedy batch against an independently recomputed same-shape
    # seed, including every element in all 64 K/V tensors.
    runtime=M3Runtime(dict(mode='exact',offline_experiment=True,batched_decode=True))
    outputs,diag=runtime.generate(lm,ids,vision,context=dict(scene_id=snap['scene'],frame_id='final_batch_proof',camera_ids=['all']),
        do_sample=False,max_new_tokens=320)
    prefix=diag['batch']['shared_prefix_tokens'];x=xs[0].prefix(prefix)
    expected=base(inputs_embeds=x.embeddings,position_ids=x.position_ids,attention_mask=x.attention_mask,use_cache=True)
    unchanged=all(torch.equal(a,b) for layer,ref in zip(runtime.cache.entry.kv,expected.past_key_values) for a,b in zip(layer,ref))
    assert unchanged
    write(dest/'batch_cache_immutable.json',dict(verified_all_elements=True,layers=len(expected.past_key_values),
        prefix_tokens=prefix,token_counts=[x.shape[1]-1 for x in outputs],max_new_tokens=320,diagnostics=diag))
    runtime.reset()
