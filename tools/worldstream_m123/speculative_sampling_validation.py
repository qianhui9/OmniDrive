"""Seeded actual-Llama joint distribution diagnostic; no GPU timing claims."""
import argparse
import copy
import math
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from worldstream.m3 import expand_input
from worldstream.m3.prefill_executor import PrefillExecutor, deterministic_draft_sample
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from tools.worldstream_m123.benchmark.common import write


@torch.no_grad()
def validate(root, draws=3000):
    torch.manual_seed(20261028)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=8,hidden_size=16,intermediate_size=24,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
        bos_token_id=1,eos_token_id=None,pad_token_id=0)).eval()
    ex=expand_input(model,torch.tensor([[1,3,4]]),None)
    generation=dict(max_new_tokens=2,do_sample=True,temperature=.1,top_p=.75,top_k=50,eos_token_id=None)
    cfg=copy.deepcopy(model.generation_config);cfg.update(**generation);cfg.max_length=3
    processors=model._get_logits_processor(cfg,1,None,None,[])
    warpers=model._get_logits_warper(cfg)
    def probability(raw, ids):
        return torch.softmax(warpers(ids,processors(ids,raw)),-1)[0]
    output=model(inputs_embeds=ex.embeddings,attention_mask=ex.attention_mask,use_cache=True)
    p1=probability(output.logits[:,-1],torch.tensor([[1]]))
    joint=torch.zeros(8,8)
    for d in range(8):
        second=model(input_ids=torch.tensor([[d]]),past_key_values=output.past_key_values,
            attention_mask=torch.ones(1,ex.length+1,dtype=torch.long),position_ids=torch.tensor([[ex.length]]))
        joint[d]=p1[d]*probability(second.logits[:,-1],torch.tensor([[1,d]]))
    counts={k:torch.zeros(8,8) for k in ('native','speculative')}
    executor=PrefillExecutor(PrefixKVCache(0))
    for mode in counts:
        torch.manual_seed(8001 if mode=='native' else 8002)
        for _ in range(draws):
            if mode=='native':
                tokens=model.generate(inputs_embeds=ex.embeddings,attention_mask=ex.attention_mask,**generation)
            else:
                tokens,_=executor.generate(model,ex,capture=False,draft_tokens=[3,4],
                    experimental_sampling=True,draft_length=4,**generation)
            counts[mode][tokens[0,1],tokens[0,2]]+=1
    bound=math.sqrt(math.log(2*64/.001)/(2*draws))
    diagnostics={}
    for mode,c in counts.items():
        error=(c/draws-joint).abs()
        diagnostics[mode]=dict(draws=draws,counts=c.int().tolist(),max_abs_probability_error=float(error.max()),
            total_variation=float(error.sum()/2),simultaneous_hoeffding_bound_99_9_percent=bound,
            zero_probability_violations=int(c[joint==0].sum()),
            passed=bool(error.max()<bound and c[joint==0].sum()==0))
    boundaries=[]
    for pd in (0.,.01,.25,.9,1.):
        p=torch.tensor([[pd,(1-pd)*.3,(1-pd)*.7]],dtype=torch.float64)
        count=torch.zeros(3);accepted=0
        g=torch.Generator().manual_seed(410)
        for _ in range(12000):
            d,ok=deterministic_draft_sample(p,0,g);count[d]+=1;accepted+=ok
        error=float((count/12000-p[0]).abs().max())
        boundaries.append(dict(p_d=pd,draws=12000,observed=(count/12000).tolist(),accepted=accepted,
            max_abs_probability_error=error,passed=error<.018 and abs(accepted/12000-pd)<.018))
    passed=all(d['passed'] for d in diagnostics.values()) and all(d['passed'] for d in boundaries)
    write(root/'sampling_distribution_validation.json',dict(status='PASS' if passed else 'FAIL',
        scope='Actual small CPU Llama exhaustive two-token joint law and deterministic proposal transition boundaries. Separate from production-model sampling quality/deployment qualification.',
        torch_version=torch.__version__,seed=20261028,generation=generation,
        expected_joint=joint.tolist(),diagnostics=diagnostics,boundaries=boundaries,
        production_sampling_deployment='NOT_QUALIFIED_BY_THIS_TEST'))
    print('SAMPLING',passed,diagnostics,flush=True)
    if not passed:raise RuntimeError('Sampling distribution diagnostic failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/m3_world_guided_decode_run01'))
    parser.add_argument('--draws',type=int,default=3000)
    args=parser.parse_args();validate(args.root,args.draws)
