"""Separate first-divergence diagnosis; no decode optimization or timing claims."""
from dataclasses import replace
import hashlib
import json
import torch
from tools.worldstream_m1.io_utils import write_json
from worldstream.m3.visual_state import base_model,expand_input,execution_signature
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.prefill_executor import PrefillExecutor
from worldstream.m3.prefix_matcher import exact_lcp
from .profile import ROOT,SAMPLING,pin,load


def kv_error(a,b,n=None):
    rows=[]
    for i,(x,y) in enumerate(zip(a,b)):
        rows.append(dict(layer=i,max_abs=max(float((u[...,:n,:]-v[...,:n,:]).float().abs().max()) for u,v in zip(x,y))))
    return dict(max_abs=max(r['max_abs'] for r in rows),layers=rows)


def first_difference(a,b):
    a=a.flatten().tolist();b=b.flatten().tolist()
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)) if len(a)!=len(b) else None)


@torch.no_grad()
def main():
    dest=ROOT/'prefix';hashes=pin(dest);model,_=load(dest);lm=model.lm_head;base=base_model(lm)
    snapshots=torch.load(ROOT/'service/carrier_snapshots.pt',map_location='cpu')
    old=[json.loads(s) for s in open('outputs/m2_freeze_m3_task_serving/service/runs.jsonl')]
    reference={(r['scene'],r['ordinal']):r for r in old if r['population']=='native_multi' and r['mode']=='original'}
    failures=[r for r in old if r['population']=='native_multi' and r['mode']=='exact_prefix' and r['generated_ids']!=reference[(r['scene'],r['ordinal'])]['generated_ids']]
    rows=[]
    for failure in failures:
        snap=next(s for s in snapshots if s['scene']==failure['scene'] and s['ordinal']==failure['ordinal'])
        # Reconstruct the EXACT historical question IDs via the original tokenizer
        # pipeline: current service prepared the same fixed native question list.
        assert snap['questions']==failure['questions'],(snap['questions'],failure['questions'])
        vision=snap['carriers'].cuda();sig=execution_signature(lm)
        expanded=[expand_input(lm,t.cuda(),vision,scope=(snap['scene'],snap['frame']),signature=sig,source=(snap['scene'],snap['frame'],i)) for i,t in enumerate(snap['ids'])]
        a=expanded[0];seed=base(inputs_embeds=a.embeddings,position_ids=a.position_ids,attention_mask=a.attention_mask,use_cache=True,return_dict=True)
        cache=PrefixKVCache(512*1024**2);cache.put(a,seed.past_key_values,seed.logits);del seed
        for qi,(orig,cached) in enumerate(zip(reference[(failure['scene'],failure['ordinal'])]['generated_ids'],failure['generated_ids'])):
            if orig==cached:continue
            x=expanded[qi];n=cache.match(x)
            kwargs=dict(do_sample=False,max_new_tokens=320,use_cache=True)
            native=lm.generate(inputs=snap['ids'][qi].cuda()[None],images=vision,**kwargs)
            repeated=lm.generate(inputs=snap['ids'][qi].cuda()[None],images=vision,**kwargs)
            reused,_=PrefillExecutor(cache).generate(base,x,n,capture=False,**kwargs)
            reuse_repeat,_=PrefillExecutor(cache).generate(base,x,n,capture=False,**kwargs)
            full=base(inputs_embeds=x.embeddings,position_ids=x.position_ids,attention_mask=x.attention_mask,use_cache=True,return_dict=True)
            suffix=base(inputs_embeds=x.embeddings[:,n:],position_ids=x.position_ids[:,n:],attention_mask=x.attention_mask,past_key_values=cache.branch(n),use_cache=True,return_dict=True)
            initial_error=kv_error(full.past_key_values,suffix.past_key_values)
            shared_error=kv_error(full.past_key_values,cache.branch(n),n)
            teacher=[]
            for step in range(320):
                u=full.logits[:,-1].float();v=suffix.logits[:,-1].float();t=u.argmax(-1);s=v.argmax(-1)
                r=dict(step=step,max_abs_logit_error=float((u-v).abs().max()),native_top5=torch.topk(u,5).indices.cpu().tolist(),cached_top5=torch.topk(v,5).indices.cpu().tolist(),
                       native_top5_logits=torch.topk(u,5).values.cpu().tolist(),cached_at_native_top5=v.gather(-1,torch.topk(u,5).indices).cpu().tolist())
                teacher.append(r)
                if not torch.equal(t,s) or step>=native.numel()-2:
                    if not torch.equal(t,s):
                        r['kv_error']=kv_error(full.past_key_values,suffix.past_key_values)
                        torch.save(dict(native_logits=u.cpu(),cached_logits=v.cpu(),token_step=step),dest/f'logits_{len(rows)}.pt')
                    break
                length=x.length+step+1
                mask=torch.ones((1,length),device='cuda',dtype=x.attention_mask.dtype)
                pos=torch.tensor([[length-1]],device='cuda')
                fkv,skv=full.past_key_values,suffix.past_key_values;del full,suffix
                full=base(input_ids=t[:,None],attention_mask=mask,position_ids=pos,past_key_values=fkv,use_cache=True,return_dict=True)
                suffix=base(input_ids=t[:,None],attention_mask=mask,position_ids=pos,past_key_values=skv,use_cache=True,return_dict=True)
                del fkv,skv
            # Complete identical full-prefill hit under ORIGINAL sampling.
            identical_cache=PrefixKVCache(512*1024**2);executor=PrefillExecutor(identical_cache)
            torch.manual_seed(123);torch.cuda.manual_seed_all(123)
            full_sample=lm.generate(inputs=snap['ids'][qi].cuda()[None],images=vision,**SAMPLING)
            torch.manual_seed(123);torch.cuda.manual_seed_all(123)
            independent,_=executor.generate(base,x,0,**SAMPLING)
            torch.manual_seed(123);torch.cuda.manual_seed_all(123)
            identical,diag=executor.generate(base,x,x.length,**SAMPLING)
            record=dict(scene=snap['scene'],frame=snap['frame'],ordinal=snap['ordinal'],question_index=qi,question=snap['questions'][qi],
                historical_failures_covered=['exact_prefix','direct_prefix_fallback'],historical_first_difference=first_difference(torch.tensor(orig),torch.tensor(cached)),
                native_reproducible=torch.equal(native,repeated),cached_reproducible=torch.equal(reused,reuse_repeat),
                fresh_first_difference=first_difference(native,reused),historical_native_reproduced=native.cpu().tolist()==orig,
                historical_cached_reproduced=reused.cpu().tolist()==cached,generated_native=native.cpu().tolist(),generated_cached=reused.cpu().tolist(),
                first_teacher_forced_divergent_step=teacher[-1]['step'] if teacher[-1]['native_top5'][0][0]!=teacher[-1]['cached_top5'][0][0] else None,
                teacher_forced=teacher,initial_full_vs_suffix_kv=initial_error,shared_prefix_kv=shared_error,
                exact_input_prefix=n,input_tokens=x.length,seed_tokens=a.length,suffix_tokens=x.length-n,
                prefix_field_equal={name:torch.equal(getattr(a,name)[:,:n],getattr(x,name)[:,:n]) for name in ('embeddings','token_ids','position_ids','attention_mask','visual_mask')},
                same_scope=a.scope==x.scope,same_signature=a.signature==x.signature,kv_shape=list(cache.entry.kv[0][0].shape),kv_dtype=str(cache.entry.kv[0][0].dtype),
                source=cache.entry.computed_source,lineage=cache.entry.prefix_lineage,cache_after_branch_input_equal=cache.match(a)==a.length,
                sampling=False,diagnosis='See per-step evidence; no sampling RNG consumed in greedy diagnosis',
                original_sampling_identical_full_prefill_equal=torch.equal(full_sample,independent) and torch.equal(full_sample,identical),identical_hit=diag)
            rows.append(record);write_json(dest/'failures.json',rows)
            del full,suffix,identical_cache,executor
        del cache,expanded;torch.cuda.empty_cache()
    write_json(dest/'status.json',dict(completed=True,unique_failed_frames=len(failures),diagnosed_queries=len(rows),
        sources_unchanged=all(hashlib.sha256(__import__('pathlib').Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())))


if __name__=='__main__':main()
