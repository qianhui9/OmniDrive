"""Observe original native operations; never substitute higher precision math."""
from contextlib import contextmanager
import copy
from functools import wraps
import torch
from transformers import LogitsProcessorList
from worldstream.m3.visual_state import base_model,expand_input
from .common import OLD,read,metrics,probability_tools,write_json,SAMPLING


@contextmanager
def capture_layers(base,mode,n):
    import transformers.models.llama.modeling_llama as llama
    captures={};active=[None];step=[0];hooks=[];originals=[]
    def save(name,tensor,attention=False,clip=False):
        t=tensor.detach()
        if attention:
            for j in range(t.shape[-2]):
                pos=j if mode=='parallel' else step[0]
                val=t[...,j:j+1,:n+pos+1] if clip else t[...,j:j+1,:]
                captures[(name,pos)]=val.clone()
        else:
            for j in range(t.shape[1]):
                pos=j if mode=='parallel' else step[0]
                captures[(name,pos)]=t[:,j:j+1].clone()
    # The outer layer output is after the residual and MLP, in actual native order.
    for i,layer in enumerate(base.model.layers):
        modules={'layer_input':layer.input_layernorm,'post_attention_norm':layer.post_attention_layernorm,
            'mlp':layer.mlp,'residual_mlp':layer,**{f'{p}_proj':getattr(layer.self_attn,p+'_proj') for p in ('q','k','v','o')},
            **{f'mlp_{p}':getattr(layer.mlp,p+'_proj') for p in ('gate','up','down')}}
        for label,module in modules.items():
            def hook(m,args,out,i=i,label=label):save(f'{i:02d}/{label}',out[0] if isinstance(out,tuple) else out)
            hooks.append(module.register_forward_hook(hook))
        original=layer.self_attn.forward
        def wrap(native,index):
            @wraps(native)
            def forward(*a,**kw):
                active[0]=index
                try:return native(*a,**kw)
                finally:active[0]=None
            return forward
        originals.append((layer.self_attn,original));layer.self_attn.forward=wrap(original,i)
    rope=llama.apply_rotary_pos_emb;matmul=torch.matmul;softmax=torch.nn.functional.softmax
    def rotary(*a,**kw):
        q,k=rope(*a,**kw)
        if active[0] is not None:
            save(f'{active[0]:02d}/rope_q',q,True);save(f'{active[0]:02d}/rope_k',k,True)
        return q,k
    def mm(a,b,*args,**kw):
        out=matmul(a,b,*args,**kw)
        if active[0] is not None and out.ndim==4:
            is_score=b.shape[-2]==base.config.hidden_size//base.config.num_attention_heads
            save(f'{active[0]:02d}/attention_score_matmul' if is_score else f'{active[0]:02d}/attention_output',out,True,is_score)
        return out
    def soft(*a,**kw):
        out=softmax(*a,**kw)
        if active[0] is not None:save(f'{active[0]:02d}/attention_softmax',out,True,True)
        return out
    llama.apply_rotary_pos_emb=rotary;torch.matmul=mm;torch.nn.functional.softmax=soft
    try:yield captures,step
    finally:
        llama.apply_rotary_pos_emb=rope;torch.matmul=matmul;torch.nn.functional.softmax=softmax
        for h in hooks:h.remove()
        for module,native in originals:module.forward=native


@torch.no_grad()
def audit_protocol(model,snapshot,root):
    base=base_model(model.lm_head);config,offline=probability_tools(base)
    native_sample=base.sample;rows=[];protocol={}
    @wraps(native_sample)
    def sample(input_ids,*args,**kwargs):
        processor=kwargs['logits_processor'];warper=kwargs['logits_warper']
        protocol.update(initial_history=input_ids.cpu().tolist(),
            processors=[dict(type=type(x).__name__,options=vars(x)) for x in processor],
            warpers=[dict(type=type(x).__name__,options=vars(x)) for x in warper])
        class Observe(LogitsProcessorList):
            def __call__(self,ids,scores):
                processed=warper(ids,scores)
                # Check the exact same raw logits before the native processors.
                expected=offline(ids,raw[0]);actual=torch.softmax(processed,dim=-1)
                rows.append(dict(history=ids.cpu().tolist(),history_length=ids.shape[1],
                    target_equal=torch.equal(expected,actual),**metrics(expected,actual)))
                return processed
        class Processor(LogitsProcessorList):
            def __call__(self,ids,scores):
                raw[0]=scores.clone();return processor(ids,scores)
        raw=[None];kwargs['logits_processor']=Processor();kwargs['logits_warper']=Observe()
        return native_sample(input_ids,*args,**kwargs)
    base.sample=sample
    try:
        torch.manual_seed(123);torch.cuda.manual_seed_all(123)
        answer=model.lm_head.generate(inputs=torch.tensor(snapshot['prompt_ids'],device='cuda'),
            images=snapshot['carriers'].cuda(),**SAMPLING)
    finally:base.sample=native_sample
    result=dict(captured=protocol,actual_steps=len(rows),all_processed_probabilities_equal=bool(rows) and all(r['target_equal'] for r in rows),
        rows=rows,actual_native_ids=answer.cpu().tolist(),offline_config=config.to_dict(),
        prompt_length_semantics='Original Llava.generate passes full embeddings to HF, which starts token history at BOS length=1; '
            'expanded visual prompt length is used for model positions, not processor token history.',
        processing_order='native logits_processor -> native logits_warper (temperature, default top_k, top_p) -> dtype-native softmax')
    write_json(root/'attribution/processing_protocol.json',result)
    return result


@torch.no_grad()
def run(model,snapshots,root):
    (root/'attribution').mkdir(exist_ok=False)
    failures=[r for r in read(OLD/'kv_state_correctness.json')['rows'] if any(not c['target_probabilities_equal'] for c in r['comparisons'])]
    lookup={s['key']:s for s in snapshots};base=base_model(model.lm_head);config,probs=probability_tools(base)
    protocol=audit_protocol(model,lookup[failures[0]['key']],root);results=[]
    for case in failures:
        s=lookup[case['key']];offset=case['offset'];k=case['k']
        x=expand_input(model.lm_head,torch.tensor(s['prompt_ids'],device='cuda'),s['carriers'].cuda())
        seed=base(inputs_embeds=x.embeddings,attention_mask=x.attention_mask,position_ids=x.position_ids,use_cache=True,return_dict=True)
        def forward(tokens,kv,pos):
            return base(input_ids=tokens,past_key_values=kv,attention_mask=torch.ones((1,pos+tokens.shape[1]),dtype=torch.long,device='cuda'),
                position_ids=torch.arange(pos,pos+tokens.shape[1],device='cuda')[None],use_cache=True,return_dict=True)
        for token in s['native_tokens'][:offset]:seed=forward(torch.tensor([[token]],device='cuda'),seed.past_key_values,seed.past_key_values[0][0].shape[-2])
        n=x.length+offset;draft=torch.tensor([s['history'].token_ids[offset:offset+k]],device='cuda')
        versions=[t._version for layer in seed.past_key_values for t in layer]
        unobserved=forward(draft,seed.past_key_values,n)
        with capture_layers(base,'parallel',n) as (parallel,_):observed=forward(draft,seed.past_key_values,n)
        observation_equal=torch.equal(observed.logits,unobserved.logits)
        serial_logits=[];kv=seed.past_key_values
        with capture_layers(base,'serial',n) as (serial,step):
            for j in range(k):
                step[0]=j;out=forward(draft[:,j:j+1],kv,n+j);kv=out.past_key_values;serial_logits.append(out.logits)
        serial_logits=torch.cat(serial_logits,1);details=[]
        for (name,j),value in parallel.items():
            m=metrics(value,serial[(name,j)]);details.append(dict(stage=name,position=j+1,**m))
        ordered=['layer_input','q_proj','k_proj','v_proj','rope_q','rope_k','attention_score_matmul',
                 'attention_softmax','attention_output','o_proj','post_attention_norm','mlp_gate','mlp_up','mlp_down','mlp','residual_mlp']
        divergences=sorted([d for d in details if not d['equal']],key=lambda d:(int(d['stage'].split('/')[0]),ordered.index(d['stage'].split('/')[1]),d['position']))
        positions=[]
        for j in range(k):
            history=torch.tensor([[config.bos_token_id]+s['native_tokens'][:offset]+list(s['history'].token_ids[offset:offset+j+1])],device='cuda')
            a=probs(history,observed.logits[:,j]);b=probs(history,serial_logits[:,j])
            positions.append(dict(position=j+1,logits=metrics(observed.logits[:,j],serial_logits[:,j]),
                target_equal=torch.equal(a,b),target_total_variation=float((a.double()-b.double()).abs().sum()/2),
                support_equal=torch.equal(a>0,b>0)))
        first=divergences[0] if divergences else None
        if first:
            name=first['stage'];j=first['position']-1
            torch.save(dict(parallel=parallel[(name,j)].cpu(),serial=serial[(name,j)].cpu(),
                parallel_logits=observed.logits.cpu(),serial_logits=serial_logits.cpu(),draft=draft.cpu(),
                key=s['key'],offset=offset,k=k,first=first),root/'attribution'/f'witness_{len(results)}.pt')
        result=dict(key=s['key'],offset=offset,k=k,failed_positions_before=[c['position'] for c in case['comparisons'] if not c['target_probabilities_equal']],
            first_divergence=first,layer_stages=details,targets=positions,observer_preserves_logits=observation_equal,
            positions=list(range(n,n+k)),serial_positions=list(range(n,n+k)),
            full_causal_attention=True,source_kv_unmodified=versions==[t._version for layer in seed.past_key_values for t in layer],
            source_kv_length=n,mask_semantics='all prior positions visible; causal block masks only uncommitted future positions')
        results.append(result);print('attribution',len(results),s['key'],offset,k,'first',first,flush=True)
        del parallel,serial,seed,kv,out,observed,unobserved,serial_logits,x
    artifact=dict(cases=results,known_failing_target_positions=sum(len(r['failed_positions_before']) for r in results),
        protocol_consistent=protocol['all_processed_probabilities_equal'],
        root_cause='native_fp16_execution_shape_rounding' if protocol['all_processed_probabilities_equal'] and all(r['observer_preserves_logits'] and r['source_kv_unmodified'] for r in results) else 'requires_investigation',
        earliest_operations=sorted({r['first_divergence']['stage'] for r in results if r['first_divergence']}),
        sgvd_frozen=True,weights_or_precision_changed=False,
        scope='Only previously recorded probability failures. Hooks observe original QKV, native RoPE, '
              'actual matmul/softmax, projections, residual/MLP and final logits. No substitute forward math.')
    write_json(root/'sgvd_numerical_root_cause.json',artifact)
    (root/'sgvd_final_freeze_decision.md').write_text('# SGVD final freeze\n\n'+artifact['root_cause']+
        '\n\nOriginal FP16 arithmetic and sampling settings are unchanged. SGVD remains disabled. '
        'A mathematical acceptance law does not certify logits produced by a different GEMM/Attention shape. '
        'No larger draft search or SGVD generation runtime is introduced.\n')
    return artifact
