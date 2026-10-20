"""Additional matched native-prefix positions; no teacher-forced target KV."""
from collections import defaultdict
import torch
from worldstream.m3.visual_state import base_model,expand_input
from worldstream.m3.microbatch_kv import pack_independent_kv
from .common import OLD,read,write_json,probability_tools,metrics


@torch.no_grad()
def run(model,root):
    snapshots=torch.load(OLD/'phase1/carrier_snapshots.pt',map_location='cpu')
    groups=defaultdict(list)
    for s in snapshots:groups[s['scene'],s['frame']].append(s)
    # Only 2 real frames, fixed first 4 different native questions each. These
    # requests were not selected by favorable speed or numerical error.
    specs=list(groups.values())[:2];base=base_model(model.lm_head);config,probs=probability_tools(base)
    results=[];saved=False
    for frame,ss in enumerate(specs):
        ss=ss[:4];states=[None]*len(ss);reached=[0]*len(ss)
        for offset in (1,2,4,8,16,32,64,96,128):
            valid=[i for i,s in enumerate(ss) if len(s['native_tokens'])>offset]
            records={}
            for i in valid:
                s=ss[i]
                if states[i] is None:
                    x=expand_input(model.lm_head,torch.tensor(s['prompt_ids'],device='cuda'),s['carriers'].cuda())
                    output=base(inputs_embeds=x.embeddings,attention_mask=x.attention_mask,position_ids=x.position_ids,
                        use_cache=True,return_dict=True)
                    kv=output.past_key_values;del output,x
                else:kv=tuple(tuple(t.cuda() for t in layer) for layer in states[i])
                def forward(token,kv):
                    n=kv[0][0].shape[-2]
                    return base(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=kv,
                        attention_mask=torch.ones((1,n+1),device='cuda',dtype=torch.long),
                        position_ids=torch.tensor([[n]],device='cuda'),use_cache=True,return_dict=True)
                for position in range(reached[i],offset-1):
                    output=forward(s['native_tokens'][position],kv);kv=output.past_key_values;del output
                before=tuple(tuple(t.cpu() for t in layer) for layer in kv)
                output=forward(s['native_tokens'][offset-1],kv)
                records[i]=dict(kv=before,logits=output.logits[:,-1].cpu())
                states[i]=tuple(tuple(t.cpu() for t in layer) for layer in output.past_key_values);reached[i]=offset
                del kv,output
            for batch in (2,4):
                for start in range(0,len(valid),batch):
                    ids=valid[start:start+batch]
                    if len(ids)<2:continue
                    kv,mask,pos=pack_independent_kv([records[i]['kv'] for i in ids],device='cuda')
                    tokens=torch.tensor([[ss[i]['native_tokens'][offset-1]] for i in ids],device='cuda')
                    out=base(input_ids=tokens,past_key_values=kv,
                        attention_mask=torch.cat((mask,mask.new_ones((len(ids),1))),1),position_ids=pos,use_cache=True,return_dict=True)
                    for row,i in enumerate(ids):
                        history=torch.tensor([[config.bos_token_id]+ss[i]['native_tokens'][:offset]],device='cuda')
                        p=probs(history,out.logits[row:row+1,-1]);q=probs(history,records[i]['logits'].cuda())
                        result=dict(frame=frame,key=ss[i]['key'],offset=offset,batch=len(ids),
                            target_equal=torch.equal(p,q),logits=metrics(out.logits[row:row+1,-1],records[i]['logits']),
                            total_variation=float((p.double()-q.double()).abs().sum()/2),support_equal=torch.equal(p>0,q>0))
                        results.append(result)
                        if not saved and not result['target_equal']:
                            torch.save(dict(batch_logits=out.logits[row:row+1,-1].cpu(),native_logits=records[i]['logits'],
                                batch_probability=p.cpu(),native_probability=q.cpu(),history=history.cpu(),source=result),
                                root/'feasibility/first_matched_prefix_probability_difference.pt');saved=True
                    del kv,out
            print('matched_prefix',frame,offset,'comparisons',len(results),flush=True)
    old=read(root/'microbatch_numerical_parity.json')
    old.update(matched_prefix_rows=results,
        all_targets_equal=old['all_targets_equal'] and all(r['target_equal'] for r in results),
        max_target_total_variation=max([old['max_target_total_variation']]+[r['total_variation'] for r in results]),
        matched_prefix_protocol='Original full visual prefill, all history tokens reached with single-token native serial '
            'forward. Same owned per-request KV fed to native single-token vs padded batch 2/4. Original dtype/processors/warpers.',
        production_qualification='Finite point checks cannot establish general native FP16 exact sampling even if every measured probability is equal.')
    write_json(root/'microbatch_numerical_parity.json',old)
    return old
