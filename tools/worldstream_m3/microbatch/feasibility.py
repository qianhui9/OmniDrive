"""Actual independent nuScenes requests, native full prefill, sampled decode costs."""
from collections import defaultdict
from dataclasses import asdict
import gc
import hashlib
import json
import torch
from worldstream.m3.visual_state import base_model,expand_input
from worldstream.m3.question_dependencies import analyze_questions,ready_groups
from worldstream.m3.microbatch_kv import pack_independent_kv
from .common import OLD,read,write_json,append,measure,probability_tools,metrics,stats


@torch.no_grad()
def census(model,snapshots,root):
    byframe=defaultdict(list)
    for s in snapshots:byframe[(s['scene'],s['frame'])].append(s)
    results=[]
    for (scene,frame),ss in byframe.items():
        ids=[torch.tensor(s['prompt_ids']) for s in ss];qs=[s['question'] for s in ss]
        same=all(torch.equal(ss[0]['carriers'],s['carriers']) for s in ss)
        dep=analyze_questions(model.tokenizer,ids,qs,inputs_ready=True,shared_carriers=same)
        results.append(dict(scene=scene,frame=frame,questions=len(ss),same_complete_carriers=same,
            rows=[dict(asdict(d),question=q,prompt_tokens=int(i.numel()),native_generated_tokens=len(s['native_tokens']),
                full_prompt_sha256=hashlib.sha256(i.numpy().tobytes()).hexdigest()) for d,q,i,s in zip(dep,qs,ids,ss)],
            ready_groups={str(k):ready_groups(dep,k) for k in (1,2,4,8)},
            dependencies=[(source,d.request_id) for d in dep for source in d.dependencies],
            semantics='Independent two-message human/empty assistant sources; shared completed heads; '
                'generation reads carriers, never writes temporal heads; outputs are appended in original question order'))
    artifact=dict(population='exploratory_reused_population',frames=results,
        independent_requests=sum(r['independent'] for f in results for r in f['rows']),
        total_requests=sum(f['questions'] for f in results),
        independent_decode_tokens=sum(r['native_generated_tokens'] for f in results for r in f['rows'] if r['independent']),
        population_is_synthetic=False,
        source_contracts=['LoadAnnoatationVQATest.preprocess_vqa builds each source independently',
            'petr3d.simple_test_pts computes det/map memory and shared carriers before any question',
            'native_prompt_matches binds exact single USER turn and empty ASSISTANT',
            'No prior generated answer is added to later input_ids; unbound/answer-reference prompts are barriers'],
        scheduling='Ready groups cannot cross unready/dependent/ordered barriers; all results returned in original order')
    write_json(root/'native_question_dependency_census.json',artifact)
    return byframe,artifact


@torch.no_grad()
def run(model,snapshots,root):
    dest=root/'feasibility';dest.mkdir(exist_ok=False)
    groups,population=census(model,snapshots,root)
    first=next(ss for ss in groups.values() if len(ss)>=8)
    base=base_model(model.lm_head);config,probs=probability_tools(base)
    states=[];seeds=[];prefill=[];offset=16
    for qi,s in enumerate(first[:8]):
        x=expand_input(model.lm_head,torch.tensor(s['prompt_ids'],device='cuda'),s['carriers'].cuda())
        out,timing=measure(lambda:base(inputs_embeds=x.embeddings,attention_mask=x.attention_mask,
            position_ids=x.position_ids,use_cache=True,return_dict=True))
        for j,t in enumerate(s['native_tokens'][:offset]):
            n=out.past_key_values[0][0].shape[-2]
            out=base(input_ids=torch.tensor([[t]],device='cuda'),past_key_values=out.past_key_values,
                attention_mask=torch.ones((1,n+1),device='cuda',dtype=torch.long),
                position_ids=torch.tensor([[n]],device='cuda'),use_cache=True,return_dict=True)
        states.append(tuple(tuple(t.cpu() for t in layer) for layer in out.past_key_values))
        seeds.append(s['native_tokens'][offset]);prefill.append(dict(request_id=qi,input_tokens=x.length,**timing))
        del out,x
    torch.cuda.empty_cache();rows=[];parity=[];components=[]
    for batch in (1,2,4,8):
        try:
            torch.cuda.reset_peak_memory_stats()
            (kv,mask,pos),packing=measure(lambda:pack_independent_kv(states[:batch],device='cuda'))
            input_ids=torch.tensor(seeds[:batch],device='cuda')[:,None]
            mask=torch.cat((mask,mask.new_ones((batch,1))),1)
            kw=dict(input_ids=input_ids,past_key_values=kv,attention_mask=mask,position_ids=pos,use_cache=True,return_dict=True)
            generators=[torch.Generator(device='cuda').manual_seed(1000+j) for j in range(batch)]
            def sample_step():
                out=base(**kw)
                sampled=[]
                for j in range(batch):
                    ids=torch.tensor([[config.bos_token_id]+first[j]['native_tokens'][:offset+1]],device='cuda')
                    p=probs(ids,out.logits[j:j+1,-1])
                    sampled.append(torch.multinomial(p,1,generator=generators[j]))
                return out,sampled
            for rep in range(14):
                (out,sampled),timing=measure(sample_step)
                rows.append(dict(batch=batch,repeat=rep,warmup=rep<3,**timing,
                    tokens_per_second=batch*1000/timing['wall_ms'],kv_pack=packing,
                    kv_bytes=sum(t.numel()*t.element_size() for layer in kv for t in layer),
                    padding_tokens=int((mask==0).sum()),position_ids=pos.cpu().tolist(),
                    peak_gpu_allocated=torch.cuda.max_memory_allocated(),sampled=[int(t) for t in sampled]))
                del out,sampled
            # Original serial comparison at the EXACT same prompt and legal token history.
            block=base(**kw);batch_logits=block.logits[:,-1].cpu();del block
            del kv,kw;torch.cuda.empty_cache()
            for j in range(batch):
                state=tuple(tuple(t.cuda() for t in layer) for layer in states[j]);n=state[0][0].shape[-2]
                serial=base(input_ids=input_ids[j:j+1],past_key_values=state,
                    attention_mask=torch.ones((1,n+1),device='cuda',dtype=torch.long),
                    position_ids=torch.tensor([[n]],device='cuda'),use_cache=True,return_dict=True)
                history=torch.tensor([[config.bos_token_id]+first[j]['native_tokens'][:offset+1]],device='cuda')
                a=probs(history,batch_logits[j:j+1].cuda());b=probs(history,serial.logits[:,-1])
                row=dict(batch=batch,request_id=j,key=first[j]['key'],logits=metrics(batch_logits[j:j+1],serial.logits[:,-1]),
                    target_equal=torch.equal(a,b),target_total_variation=float((a.double()-b.double()).abs().sum()/2),
                    support_equal=torch.equal(a>0,b>0))
                parity.append(row)
                if not row['target_equal']:
                    torch.save(dict(batch_logits=batch_logits[j],serial_logits=serial.logits[0,-1].cpu(),
                        batch_probability=a.cpu(),serial_probability=b.cpu(),history=history.cpu(),source=row),
                        dest/f'batch{batch}_request{j}_target_difference.pt')
                del serial,state
            # Separate module GPU events. No adding inclusive CPU/GPU layers to kernel totals.
            kv,mask,pos=pack_independent_kv(states[:batch],device='cuda');mask=torch.cat((mask,mask.new_ones((batch,1))),1)
            events=[];hooks=[];starts={}
            for name,module in base.named_modules():
                if (name.endswith(('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'))
                        or name=='lm_head'):
                    def before(m,args,name=name):
                        e=torch.cuda.Event(enable_timing=True);e.record();starts[name]=e
                    def after(m,args,out,name=name):
                        e=torch.cuda.Event(enable_timing=True);e.record();events.append((name,starts[name],e))
                    hooks.append(module.register_forward_pre_hook(before));hooks.append(module.register_forward_hook(after))
            for _ in range(3):
                out=base(input_ids=input_ids,past_key_values=kv,attention_mask=mask,position_ids=pos,use_cache=True,return_dict=True);del out
            torch.cuda.synchronize()
            for h in hooks:h.remove()
            components.append(dict(batch=batch,events=[dict(module=name,gpu_interval_ms=a.elapsed_time(b)) for name,a,b in events]))
            del kv,input_ids,mask,pos;torch.cuda.empty_cache()
            print('feasibility batch',batch,stats([r['wall_ms'] for r in rows if r['batch']==batch and not r['warmup']]),flush=True)
        except torch.cuda.OutOfMemoryError as error:
            rows.append(dict(batch=batch,fallback='oom',error=str(error)))
            # Do not turn an OOM arm into an apparent missing denominator.
            for name in ('kv','kw','out','block','state','serial'):
                if name in locals():locals().pop(name,None)
            gc.collect();torch.cuda.empty_cache()
    single=next(r for r in rows if r['batch']==1 and not r.get('warmup',True) and 'wall_ms' in r)
    baseline=stats([r['wall_ms'] for r in rows if r['batch']==1 and not r.get('warmup',True)])['p50']
    summary=[]
    for batch in (1,2,4,8):
        rr=[r for r in rows if r['batch']==batch and not r.get('warmup',True) and 'wall_ms' in r]
        if rr:summary.append(dict(batch=batch,wall_ms=stats([r['wall_ms'] for r in rr]),gpu_ms=stats([r['gpu_ms'] for r in rr]),
            aggregate_tokens_s=stats([r['tokens_per_second'] for r in rr]),
            throughput_speedup=baseline*batch/stats([r['wall_ms'] for r in rr])['p50'],
            max_gpu_memory=max(r['peak_gpu_allocated'] for r in rr)))
    artifact=dict(rows=rows,summary=summary,prefill=prefill,module_components=components,
        opportunity=any(r['batch'] in (2,4) and r['throughput_speedup']>1.1 for r in summary),
        task_independence=population['independent_requests']>1,
        workload='Different real same-frame questions, first 8 in native order; no copied prompts. '
            'Single-step microbench at native-prefix offset=16, full original prefill and native processor/warper plus per-request sampling. '
            'This is not full generation or complete-service speedup; gather/pack is separately charged in the runtime experiment.',
        production_qualified=False)
    write_json(root/'microbatch_gpu_feasibility.json',artifact)
    write_json(root/'microbatch_numerical_parity.json',dict(rows=parity,
        all_targets_equal=all(r['target_equal'] for r in parity),
        max_target_total_variation=max(r['target_total_variation'] for r in parity),
        exact_sampling_claim=False))
    return artifact
