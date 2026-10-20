"""Replay every fresh sampled-prefix failure with query RNG and divergent KV."""
from functools import wraps
import json
from pathlib import Path
import torch
from tools.worldstream_m1.io_utils import write_json
from worldstream.m3.visual_state import base_model,expand_input,execution_signature
from worldstream.m3.prefill_executor import PrefillExecutor
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from .profile import SAMPLING


def difference(a,b):
    a=a[0];b=b[0]
    return next((i for i,(u,v) in enumerate(zip(a,b)) if u!=v),min(len(a),len(b)) if len(a)!=len(b) else None)


@torch.no_grad()
def run(model,snapshots,root,dest):
    from .prefix import kv_error
    rows=[json.loads(s) for s in (root/'service/runs.jsonl').read_text().splitlines()]
    refs={(r['scene'],r['ordinal']):r for r in rows if r['mode']=='original' and r['population']=='native_multi' and r['repeat']==1}
    cached={(r['scene'],r['ordinal']):r for r in rows if r['mode']=='legacy_prefix' and r['population']=='native_multi' and r['repeat']==1}
    lm=model.lm_head;base=base_model(lm);native_forward=base.forward;all_results=[]
    for snap in snapshots:
        key=snap['scene'],snap['ordinal'];reference=refs[key];prefix=cached[key]
        failed={i:difference(a,b) for i,(a,b) in enumerate(zip(reference['generated_ids'],prefix['generated_ids'])) if a!=b}
        if not failed:continue
        vision=snap['carriers'].cuda();signature=execution_signature(lm)
        expanded=[expand_input(lm,t.cuda(),vision,scope=(snap['scene'],snap['frame']),signature=signature,source=(snap['scene'],snap['frame'],i)) for i,t in enumerate(snap['ids'])]
        witnesses={};rngs={};native_answers={};frame=[]
        for path in ('native','prefix'):
            cache=PrefixKVCache(512*1024**2)
            torch.manual_seed(123+snap['ordinal']);torch.cuda.manual_seed_all(123+snap['ordinal'])
            for qi,x in enumerate(expanded):
                before=torch.cuda.get_rng_state().clone();index=[0]
                target=failed.get(qi)
                @wraps(native_forward)
                def observe(*args,**kw):
                    output=native_forward(*args,**kw)
                    if target is not None and index[0]==target-1:
                        value=dict(logits=output.logits[:,-1].float().cpu(),kv=tuple(tuple(t.cpu() for t in layer) for layer in output.past_key_values))
                        if path=='native':witnesses[qi]=value
                        else:
                            original=witnesses[qi]
                            frame.append(dict(question_index=qi,first_different_sequence_index=target,
                                logit_max_abs=float((original['logits']-value['logits']).abs().max()),kv=kv_error(original['kv'],value['kv']),
                                native_top5=torch.topk(original['logits'],5).indices.tolist(),prefix_top5=torch.topk(value['logits'],5).indices.tolist()))
                            torch.save(dict(native_logits=original['logits'],cached_logits=value['logits']),dest/f'sampling_logits_{len(all_results)}_{qi}.pt')
                            del witnesses[qi]
                    index[0]+=1
                    return output
                base.forward=observe
                try:
                    n=cache.match(x) if path=='prefix' else 0
                    fields={name:torch.equal(getattr(cache.entry.expanded,name)[:,:n],getattr(x,name)[:,:n]) for name in ('token_ids','embeddings','visual_mask','position_ids','attention_mask')} if n else {}
                    if path=='native':answer=lm.generate(inputs=snap['ids'][qi].cuda()[None],images=vision,**SAMPLING)
                    else:answer,_=PrefillExecutor(cache).generate(base,x,n,**SAMPLING)
                finally:base.forward=native_forward
                actual=answer.cpu().tolist()
                after=torch.cuda.get_rng_state().clone()
                if path=='native':
                    rngs[qi]=(before,after);native_answers[qi]=actual
                elif qi in failed:
                    item=next(r for r in frame if r['question_index']==qi)
                    equal_before=torch.equal(before,rngs[qi][0])
                    item.update(native_query_start_rng_equal=equal_before,after_query_rng_equal=torch.equal(after,rngs[qi][1]),
                        full_input_prefix=n,field_equal=fields,same_scope=cache.entry.expanded.scope==x.scope,same_signature=cache.entry.expanded.signature==x.signature,
                        cache_source=list(cache.entry.computed_source),cache_lineage=cache.entry.prefix_lineage,
                        fresh_native_matches_service=native_answers[qi]==reference['generated_ids'][qi],fresh_prefix_matches_service=actual==prefix['generated_ids'][qi],
                        diagnosis='upstream_answer_changed_rng_state' if not equal_before else 'equal_rng_numerically_different_prefill_or_decode' if item['logit_max_abs']>0 else 'unresolved_equal_logits_equal_rng',
                        original_generation=SAMPLING,whole_quality_relaxed=False)
            del cache
        all_results.append(dict(scene=snap['scene'],frame=snap['frame'],ordinal=snap['ordinal'],queries=frame))
        write_json(dest/'sampling_failures.json',all_results)
        del expanded;torch.cuda.empty_cache()
    write_json(dest/'sampling_failures.json',all_results)
    return all_results
