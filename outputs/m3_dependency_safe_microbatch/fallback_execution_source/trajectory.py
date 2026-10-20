"""Compare first divergence at identical actually sampled legal prefixes."""
from functools import wraps
import hashlib
import torch
from worldstream.m3.visual_state import base_model
from worldstream.m3.sampling_microbatch import request_seed
from .common import SAMPLING,probability_tools,write_json,read,metrics


@torch.no_grad()
def run(model,selected,context,batch_tokens,diag,records,root):
    base=base_model(model.lm_head);config,probs=probability_tools(base)
    byrequest={i:[] for i in range(len(selected))};decode_index=0;prefill_index=0
    for record in records:
        if record['stage']=='prefill':
            byrequest[prefill_index].append(record['logits'][0]);prefill_index+=1
        else:
            step=diag['steps'][decode_index];decode_index+=1
            for row,index in enumerate(step['request_ids']):byrequest[index].append(record['logits'][row])
    results=[];native_generate=model.lm_head.generate;native_forward=base.forward
    for i,s in enumerate(selected):
        native_logits=[]
        @wraps(native_forward)
        def observe(*a,**kw):
            out=native_forward(*a,**kw);native_logits.append(out.logits[:,-1].detach().cpu());return out
        base.forward=observe;seed=request_seed(123,context,i)
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                answer=native_generate(inputs=torch.tensor(s['prompt_ids'],device='cuda'),images=s['carriers'].cuda(),**SAMPLING)
                native_rng=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
        finally:base.forward=native_forward
        native=answer.cpu().tolist()[0];batched=batch_tokens[i].cpu().tolist()[0]
        first=next((j for j,(a,b) in enumerate(zip(native[1:],batched[1:])) if a!=b),min(len(native),len(batched))-1 if len(native)!=len(batched) else None)
        comparisons=[]
        length=min(len(native_logits),len(byrequest[i]),first+1 if first is not None else len(native_logits))
        for position in range(length):
            history=torch.tensor([native[:position+1]],device='cuda')
            a=probs(history,byrequest[i][position][None].cuda());b=probs(history,native_logits[position].cuda())
            tv=float((a.double()-b.double()).abs().sum()/2)
            comparisons.append(dict(position=position,target_equal=torch.equal(a,b),total_variation=tv,
                logits=metrics(byrequest[i][position][None],native_logits[position]),support_equal=torch.equal(a>0,b>0)))
            if not torch.equal(a,b):
                torch.save(dict(batch_logits=byrequest[i][position],native_logits=native_logits[position],
                    batch_probability=a.cpu(),native_probability=b.cpu(),history=history.cpu(),request_id=i,
                    source=s['key'],position=position),root/'feasibility'/f'actual_path_request{i}_position{position}.pt')
        results.append(dict(request_id=i,key=s['key'],seed=seed,first_token_difference=first,
            native_ids=native,batch_ids=batched,native_rng_sha256=native_rng,
            batch_rng_sha256=diag['requests'][i]['final_rng_sha256'],
            same_number_of_draws=len(native)==len(batched),same_legal_prefix_comparisons=comparisons))
    write_json(root/'actual_trajectory_sampling_parity.json',dict(rows=results,
        protocol='Original native generate with identical per-request seed versus actual fixed-4 sampled trajectory. '
            'Compare only identical emitted prefixes up to and including first divergence; never compare different conditional histories.',
        selected='First four actual archived questions in native order, not selected for numerical error',
        max_target_total_variation=max((c['total_variation'] for r in results for c in r['same_legal_prefix_comparisons']),default=0.)))
    artifact=read(root/'microbatch_numerical_parity.json')
    artifact['actual_trajectory_source']='actual_trajectory_sampling_parity.json'
    artifact['actual_trajectory_max_target_total_variation']=max((c['total_variation'] for r in results for c in r['same_legal_prefix_comparisons']),default=0.)
    artifact['max_target_total_variation']=max(artifact['max_target_total_variation'],artifact['actual_trajectory_max_target_total_variation'])
    artifact['all_targets_equal']=artifact['all_targets_equal'] and all(c['target_equal'] for r in results for c in r['same_legal_prefix_comparisons'])
    write_json(root/'microbatch_numerical_parity.json',artifact)
