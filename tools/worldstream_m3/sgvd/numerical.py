"""Real-model serial/parallel target and legal KV-prefix diagnostics."""
import torch
from worldstream.m3.visual_state import base_model,expand_input
from worldstream.m3.prefix_kv_cache import clone_kv
from worldstream.m3.verified_sampling import verify_deterministic_proposal
from tools.worldstream_m1.io_utils import write_json
from .probe import probability_tools,measure


@torch.no_grad()
def run(model,snapshots,root):
    dest=root/'numerical';dest.mkdir(parents=True,exist_ok=False)
    base=base_model(model.lm_head);config,probs=probability_tools(base);rows=[];saved=False;sampling_micro=[]
    chosen=[(s,offset) for s in [s for s in snapshots if s['history'] is not None][:4]
            for offset in (0,16,32,64) if offset+16<=min(len(s['history'].token_ids),len(s['native_tokens']))]
    for si,(s,offset) in enumerate(chosen):
        x=expand_input(model.lm_head,torch.tensor(s['prompt_ids'],device='cuda'),s['carriers'].cuda())
        seed=base(inputs_embeds=x.embeddings,attention_mask=x.attention_mask,position_ids=x.position_ids,
                  use_cache=True,return_dict=True)
        historical=list(s['history'].token_ids)
        draft=torch.tensor([historical[offset:offset+16]],device='cuda')
        def forward(t,kv,pos):
            return base(input_ids=t,past_key_values=kv,attention_mask=x.attention_mask.new_ones((1,pos+t.shape[1])),
                position_ids=torch.arange(pos,pos+t.shape[1],device='cuda')[None],use_cache=True,return_dict=True)
        # Reach the current prefix with original serial execution, not an
        # additional differently shaped prefill whose KV could already differ.
        for h in range(offset):
            seed=forward(draft.new_tensor([[s['native_tokens'][h]]]),seed.past_key_values,x.length+h)
        n=x.length+offset
        prefix=draft.new_tensor([[config.bos_token_id]+s['native_tokens'][:offset]])
        versions=tuple(t._version for layer in seed.past_key_values for t in layer)
        if si==0:
            for repeat in range(12):
                target,timing=measure(lambda:probs(prefix,seed.logits[:,-1]))
                (sample,accepted),sampling=measure(lambda:verify_deterministic_proposal(target[0],draft[0,0]))
                sampling_micro.append(dict(repeat=repeat,warmup=repeat<2,processing=timing,
                    acceptance_and_rejection_correction=sampling,accepted=bool(accepted),token=int(sample),
                    historical_candidate=int(draft[0,0])))
        for k in (1,2,4,8,16):
            if draft.shape[1]<k:continue
            block=forward(draft[:,:k],seed.past_key_values,n)
            serial=seed.past_key_values;comparisons=[];legal=[]
            for j in range(k):
                step=forward(draft[:,j:j+1],serial,n+j);serial=step.past_key_values
                ids=torch.cat((prefix,draft[:,:j+1]),1)
                bp=probs(ids,block.logits[:,j]);sp=probs(ids,step.logits[:,-1])
                comparisons.append(dict(position=j+1,logits_equal=torch.equal(block.logits[:,j],step.logits[:,-1]),
                    max_abs_logits=float((block.logits[:,j]-step.logits[:,-1]).abs().max()),
                    target_probabilities_equal=torch.equal(bp,sp),
                    target_total_variation=float((bp.double()-sp.double()).abs().sum()/2),
                    support_equal=torch.equal(bp>0,sp>0)))
                if not saved and not torch.equal(bp,sp):
                    torch.save(dict(parallel_logits=block.logits[:,j].cpu(),serial_logits=step.logits[:,-1].cpu(),
                        parallel_probabilities=bp.cpu(),serial_probabilities=sp.cpu(),ids=ids.cpu(),
                        key=s['key'],offset=offset,k=k,position=j+1,
                        conditional_history=draft[:,:j+1].cpu()),dest/'first_target_distribution_difference.pt')
                    saved=True
                del step
            # Reject at each position: retain exactly the accepted prefix, then
            # feed a legal alternative. Compare with the same serial legal path.
            # Every rejection position is tested at the initial prompt; later
            # real native prefixes test the first/middle/final boundary.
            for j in (range(k) if offset==0 else sorted({0,k//2,k-1})):
                cropped,timing=measure(lambda:clone_kv(block.past_key_values,n+j))
                assert all(t.shape[-2]==n+j for layer in cropped for t in layer)
                t=draft.new_tensor([[(int(draft[0,j])+1)%base.config.vocab_size]])
                corrected=forward(t,cropped,n+j)
                serialkv=seed.past_key_values
                for h in range(j):
                    reference=forward(draft[:,h:h+1],serialkv,n+h);serialkv=reference.past_key_values
                ref=forward(t,serialkv,n+j)
                legal.append(dict(reject_position=j,retained_tokens=n+j,
                    expected_retained_tokens=n+j,correction_avoids_rejected_token=True,
                    kv_copy=timing,corrected_kv_length=corrected.past_key_values[0][0].shape[-2],
                    continuation_logits_equal=torch.equal(corrected.logits,ref.logits),
                    continuation_logits_max_abs=float((corrected.logits-ref.logits).abs().max())))
                del corrected,ref,cropped,serialkv
            rows.append(dict(key=s['key'],offset=offset,k=k,comparisons=comparisons,rejections=legal,
                prompt_tokens=x.length,
                final_kv_layer_max_abs=[max(float((a-b).abs().max()) for a,b in zip(la,lb))
                                        for la,lb in zip(block.past_key_values,serial)],
                seed_kv_unmodified=versions==tuple(t._version for layer in seed.past_key_values for t in layer)))
            if si==0 and k==16:
                torch.save(dict(parallel_logits=block.logits.cpu(),
                    draft=draft.cpu(),prompt_source=s['key']),dest/'parallel_witness.pt')
            del block,serial
        del seed,x
    result=dict(rows=rows,sampling_microbench=sampling_micro,
        all_legal_lengths=all(c['retained_tokens']==c['expected_retained_tokens'] for r in rows for c in r['rejections']),
        all_seeds_unmodified=all(r['seed_kv_unmodified'] for r in rows),
        parallel_native_target_bitwise_equal=all(c['target_probabilities_equal'] for r in rows for c in r['comparisons']),
        max_target_total_variation=max((c['target_total_variation'] for r in rows for c in r['comparisons']),default=None),
        note='Numerical parallel/serial evidence is separate from categorical sampling proof. '
             'No tolerance is used to issue an exact-target certificate. These are offline hypothetical histories only.')
    write_json(root/'kv_state_correctness.json',result)
    return result
