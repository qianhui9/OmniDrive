"""Phase one only: original complete model, lawful history, offline target probes.

No generation loop is replaced. Teacher forcing and cost results are collected
after the current native answer has completed and cannot affect its history.
"""
import argparse
import copy
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from time import perf_counter
import numpy as np
import torch
from transformers import LogitsProcessorList
from worldstream.m3.historical_draft import DraftContract, HistoricalDraftCache, digest
from worldstream.m3.visual_state import base_model, expand_input, execution_signature
from tools.worldstream_m3.decode_optimization.profile import load, SAMPLING, ForwardTimer
from tools.worldstream_m1.io_utils import write_json
from tools.worldstream_m2.rearchitecture.common import arrays, memory
from tools.worldstream_m2.refine_integration import planning_metric

LENGTHS = (1, 2, 4, 8, 16)


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, allow_nan=False, default=str) + '\n')


def measure(fn):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); tick = perf_counter()
    result = fn()
    enqueue = (perf_counter() - tick) * 1000
    b.record(); b.synchronize()
    return result, dict(wall_ms=(perf_counter()-tick)*1000,
                       gpu_ms=a.elapsed_time(b), cpu_submission_ms=enqueue)


def probability_tools(base):
    config = copy.deepcopy(base.generation_config)
    config.update(**SAMPLING)
    config.max_length = 1 + config.max_new_tokens
    processor = base._get_logits_processor(config, 1, None, None, LogitsProcessorList())
    warper = base._get_logits_warper(config)
    def probs(ids, logits):
        # Match native HF's logits dtype and ordering, including default top_k.
        scores = processor(ids, logits.clone())
        return torch.softmax(warper(ids, scores), dim=-1)
    return config, probs


def category(question):
    q = question.lower()
    if q == 'please provide the planning trajectory for the ego car without reasons.':
        return 'planning'
    if 'what would happen' in q or 'safe' in q or 'risk' in q:
        return 'global_safety'
    if any(s in q for s in ('driving conditions', 'panoramic', 'overall', 'driving scenario', 'weather')):
        return 'scene_description'
    from worldstream.m3.task_conditioner import task_type
    return task_type(question)


@torch.no_grad()
def analyze(base, expanded, native, history, config, probs, dest, key):
    """All outputs of this function are offline; never a draft selector."""
    tokens = torch.tensor(history.token_ids, device='cuda', dtype=torch.long)[None]
    n = expanded.length
    seed = base(inputs_embeds=expanded.embeddings, attention_mask=expanded.attention_mask,
                position_ids=expanded.position_ids, use_cache=True, return_dict=True)
    def forward(t, kv, pos):
        length = t.shape[1]
        return base(input_ids=t, past_key_values=kv,
                    attention_mask=torch.ones((1, pos+length), device='cuda', dtype=torch.long),
                    position_ids=torch.arange(pos, pos+length, device='cuda')[None],
                    use_cache=True, return_dict=True)
    # Whole historical answer likelihood, conditional on that hypothetical path.
    forced = forward(tokens, seed.past_key_values, n)
    logits = torch.cat((seed.logits[:, -1:], forced.logits[:, :-1]), dim=1)
    prefix = torch.tensor([[config.bos_token_id]], device='cuda')
    p = []
    for i in range(tokens.shape[1]):
        p.append(float(probs(torch.cat((prefix, tokens[:, :i]), 1), logits[:, i])[0, tokens[0, i]]))
    raw = dict(token_ids=history.token_ids, conditional_probabilities=p,
               conditioning='current full visual prompt + hypothetical accepted historical prefix')
    write_json(dest/'teacher_forced'/f'{key}.json', raw)
    del forced, logits
    # Pair k-parallel and k-serial on the SAME proposed tokens and real KV.
    costs=[]
    for k in LENGTHS:
        if tokens.shape[1] < k:
            continue
        for rep in range(5):
            output, parallel = measure(lambda: forward(tokens[:, :k], seed.past_key_values, n))
            del output
            def serial():
                kv=seed.past_key_values
                for j in range(k):
                    out=forward(tokens[:, j:j+1], kv, n+j);kv=out.past_key_values
                return out
            output, sequential = measure(serial)
            # Simulate legal retained KV ownership and rejection processing.
            start=perf_counter()
            owned=tuple(tuple(t[..., :n+min(k, 1), :].clone() for t in layer)
                        for layer in output.past_key_values)
            torch.cuda.synchronize();kv_ms=(perf_counter()-start)*1000
            del owned, output
            costs.append(dict(k=k, repeat=rep, warmup=rep==0, parallel=parallel,
                              serial=sequential, owned_branch_copy_ms=kv_ms,
                              peak_gpu_allocated=torch.cuda.max_memory_allocated()))
    # Probe EVERY target-prefix position, using the previous completed answer at
    # the same output offset. This is an offline cost upper bound, not online
    # acceptance or an oracle length policy. Shared max-k logits only for analysis.
    target=torch.tensor(native, device='cuda', dtype=torch.long)[None]
    teacher=forward(target, seed.past_key_values, n)
    native_logits=torch.cat((seed.logits[:, -1:], teacher.logits[:, :-1]), 1)
    shadow=[]
    for offset in range(min(len(history.token_ids), len(native))):
        k=min(16, len(history.token_ids)-offset, len(native)-offset)
        kv=tuple(tuple(t[..., :n+offset, :] for t in layer) for layer in teacher.past_key_values)
        out=forward(tokens[:, offset:offset+k], kv, n+offset)
        scores=torch.cat((native_logits[:, offset:offset+1], out.logits[:, :-1]), 1)
        pp=[]
        tick=perf_counter()
        for j in range(k):
            ids=torch.cat((prefix, target[:, :offset], tokens[:, offset:offset+j]),1)
            pp.append(float(probs(ids, scores[:, j])[0, tokens[0, offset+j]]))
        shadow.append(dict(offset=offset, probabilities=pp,
                           processing_wall_ms=(perf_counter()-tick)*1000))
        del out, kv, scores
    initial=[]
    for k in LENGTHS:
        if len(p)<k:continue
        survival=np.cumprod(p[:k]);reject=np.asarray([1]+survival[:-1].tolist())*(1-np.asarray(p[:k]))
        initial.append(dict(k=k,expected_accepted=float(sum(survival)),all_accept_probability=float(survival[-1]),
                            first_reject_probabilities=reject.tolist()))
    lcp=next((i for i,(a,b) in enumerate(zip(history.token_ids,native)) if a!=b),min(len(history.token_ids),len(native)))
    del teacher,native_logits,seed
    return dict(key=key,lcp=lcp,initial=initial,shadow=shadow,costs=costs,
                historical_frame=history.frame,historical_timestamp=history.timestamp)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/m3_state_guided_verified_decode')
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    (root/'phase1').mkdir();(root/'phase1/teacher_forced').mkdir()
    source=[*Path('worldstream').rglob('*.py'),*Path('tools/worldstream_m3/sgvd').glob('*.py'),
            *Path('projects/mmdet3d_plugin').rglob('*.py')]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source}
    for f in source:
        out=root/'phase1/source'/f;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(f.read_bytes())
    write_json(root/'phase1/baseline.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],text=True),sources=hashes))
    model,dataset=load(root/'phase1');lm=model.lm_head;base=base_model(lm)
    config,probs=probability_tools(base)
    signature=digest(str(execution_signature(lm)))
    tokenizer_signature=digest(dict(vocab=model.tokenizer.get_vocab(),special=model.tokenizer.special_tokens_map))
    protocol=json.loads(Path('outputs/module1_final_validation_refinement/replay/protocol.json').read_text())
    specs=[]
    for group in protocol['scenes'][:2]:
        seq=sorted([i for i,v in enumerate(dataset.data_infos) if v['scene_token']==group['scene_id']],
                   key=lambda i:dataset.data_infos[i]['timestamp'])
        start=next(j for j,i in enumerate(seq) if dataset.data_infos[i]['token']==group['samples'][0])
        specs.append((group['scene_id'],[seq[start+j] for j in (0,1,2,4)]))
    write_json(root/'phase1/protocol.json',dict(specs=specs,population='exploratory_reused_population',
        workloads=['planning','native_multi','controlled_same_question'],sampling=SAMPLING,
        default_generation=config.to_dict(),draft_lengths=LENGTHS,
        selection='fixed scenes and offsets before observing answers; no heldout qualification',
        native_multi='all planning+conv questions from native loader; no answer truncation',
        shadow='offline full target-prefix probes; no acceleration credited'))
    import transformers,peft
    write_json(root/'phase1/environment.json',dict(torch=torch.__version__,transformers=transformers.__version__,
        peft=peft.__version__,gpu=torch.cuda.get_device_name(),affinity=sorted(os.sched_getaffinity(0)),
        model_signature=signature,tokenizer_signature=tokenizer_signature,sampling=SAMPLING,
        checkpoint='ckpts/iter_10548.pth',python=os.sys.executable))
    from mmcv.parallel import collate,scatter,DataContainer
    current={};requests=[];observed=[];native=lm.generate
    @wraps(native)
    def generate(*a,**kw):
        index=len(observed)
        ids=kw['inputs'];vision=kw['images']
        with ForwardTimer(base) as timer:
            out,timing=measure(lambda:native(*a,**kw))
            steps=timer.collect()
        observed.append(dict(prompt_ids=ids.cpu().tolist(),token_ids=out.cpu().tolist(),
                             timing=timing,steps=steps,carriers=vision.detach().cpu()))
        return out
    lm.generate=generate
    original_pts=model.simple_test_pts
    def pts(img_metas,**data):
        if current['population']=='controlled_same_question':
            # The same ORIGINAL description template, no answer or GT supplied.
            q=list(img_metas[0]['vlm_labels'].data)
            take=[0,1]
            img_metas[0]['vlm_labels']=DataContainer([q[i] for i in take],cpu_only=True)
            data['input_ids'][0]=[data['input_ids'][0][i] for i in take]
        return original_pts(img_metas,**data)
    model.simple_test_pts=pts
    snapshots=[]
    for population in ('planning','native_multi','controlled_same_question'):
        current['population']=population
        for t in dataset.pipeline.transforms:
            if hasattr(t,'load_type'):t.load_type=['planning'] if population=='planning' else ['planning','conv']
        for scene,seq in specs:
            cache=HistoricalDraftCache();model.test_flag=False;model.m1_manager=None;model.m2_runtime=None;model.m3_runtime=None
            model.save_path=str(root/'phase1/language'/population/scene)+'/'
            for ordinal,index in enumerate(seq):
                info=dataset.data_infos[index];frame=info['token'];timestamp=info['timestamp']/1e6
                # Controlled population pins only the original question RNG.
                random.seed(0 if population=='controlled_same_question' else 1000+index)
                sample=dataset[index]
                random.seed(123+ordinal);np.random.seed(123+ordinal);torch.manual_seed(123+ordinal);torch.cuda.manual_seed_all(123+ordinal)
                batch=scatter(collate([copy.deepcopy(sample)],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                cameras=tuple(batch['img_metas'][0][0]['filename'])
                cameras=tuple(str(Path(f).parent.name) for f in cameras)
                observed.clear();torch.cuda.reset_peak_memory_stats()
                output,timing=measure(lambda:model(return_loss=False,rescale=True,**batch))
                text=output[0]['text_out'];planning=planning_metric(text,info)
                for qi,(qa,raw) in enumerate(zip(text,observed)):
                    key=f'{population}_{scene}_{ordinal}_{qi}'
                    tokens=raw['token_ids'][0][1:]  # native HF inputs_embeds BOS convention
                    eos=config.eos_token_id; eos=[eos] if isinstance(eos,int) else eos or []
                    stop='eos' if tokens and tokens[-1] in eos else 'max_new_tokens' if len(tokens)==320 else 'unknown'
                    contract=DraftContract(scene,cameras,category(qa['Q']),qa['Q'],digest(raw['prompt_ids']),
                                           signature,tokenizer_signature,digest(config.to_dict()))
                    tick=perf_counter();valid=cache.begin_frame(contract,frame,timestamp);history=cache.lookup(contract) if valid else None
                    lookup_ms=(perf_counter()-tick)*1000
                    parsed=planning.get('parsed',False) if contract.task=='planning' else bool(qa['A'] and qa['A'][0].strip())
                    row=dict(key=key,population=population,scene=scene,frame=frame,timestamp=timestamp,
                        ordinal=ordinal,question_index=qi,task=contract.task,question=qa['Q'],prompt_ids=raw['prompt_ids'],
                        generated_token_ids=tokens,bos_sequence=raw['token_ids'],stop_reason=stop,parsed=parsed,text=qa['A'],
                        historical_draft_available=history is not None,history_lookup_ms=lookup_ms,
                        generation=raw['timing'],steps=raw['steps'],full_model=timing,planning=planning if qi==0 else None,
                        contract=vars(contract),cache_bytes=cache.nbytes,peak_gpu_allocated=torch.cuda.max_memory_allocated(),
                        current_prompt_source=dict(cameras=batch['img_metas'][0][0]['filename'],checkpoint='ckpts/iter_10548.pth',
                            carriers_sha256=hashlib.sha256(raw['carriers'].numpy().tobytes()).hexdigest(),
                            full_visual_prompt=True),concurrency=1)
                    append(root/'phase1/requests.jsonl',row)
                    if history is not None:
                        expanded=expand_input(lm,torch.tensor(raw['prompt_ids'],device='cuda'),raw['carriers'].cuda())
                        analysis=analyze(base,expanded,tokens,history,config,probs,root/'phase1',key)
                        analysis.update(task=contract.task,population=population,scene=scene,frame=frame,
                            gap_s=timestamp-history.timestamp,lookup_ms=lookup_ms)
                        append(root/'phase1/probes.jsonl',analysis)
                        del expanded
                    committed=cache.commit(contract,frame,timestamp,tokens,stop,completed=True,parsed=parsed,legally_submitted=True)
                    append(root/'phase1/commits.jsonl',dict(key=key,committed=committed,cache_bytes=cache.nbytes))
                    if population=='native_multi':
                        snapshots.append(dict(key=key,scene=scene,frame=frame,timestamp=timestamp,ordinal=ordinal,
                            question=qa['Q'],task=contract.task,prompt_ids=raw['prompt_ids'],carriers=raw['carriers'],
                            native_tokens=tokens,history=history))
                rawdir=root/'phase1/raw_outputs';rawdir.mkdir(exist_ok=True)
                torch.save(dict(output=arrays(output),memory=memory(model),planning=planning),rawdir/f'{population}_{scene}_{ordinal}.pt')
                print(population,scene[:8],ordinal,'queries',len(text),'wall_ms',round(timing['wall_ms'],2),flush=True)
                requests.append(dict(population=population,scene=scene,frame=frame,requests=len(text),timing=timing))
                torch.save(snapshots,root/'phase1/carrier_snapshots.pt')
    write_json(root/'phase1/status.json',dict(completed=True,frames=len(requests),
        sources_unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())))


if __name__=='__main__':
    main()
