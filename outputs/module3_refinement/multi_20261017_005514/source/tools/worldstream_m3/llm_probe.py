"""Paired real-carrier LLM probes, counterexamples and profiler evidence."""
import copy
from functools import wraps
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch
from worldstream.m3 import M3Runtime, CarrierLayout, expand_input
from worldstream.m3.visual_state import base_model, execution_signature
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.prefill_executor import PrefillExecutor
from .experiment import write, stats


def quality(answer, reference):
    # Deterministic diagnostics against unchanged GT; no claim that F1 measures safety.
    from collections import Counter
    norm = lambda s: re.findall(r'\w+', s.lower())
    a, b = norm(answer), norm(reference)
    overlap = sum((Counter(a) & Counter(b)).values())
    f1 = 2 * overlap / (len(a) + len(b)) if a or b else 1.
    number = lambda s: [float(x) for x in re.findall(r'(?<!\w)[+-]?\d+(?:\.\d+)?', s)]
    x, y = number(answer), number(reference)
    return dict(exact=answer.strip().lower() == reference.strip().lower(), token_f1=f1,
        numeric_count_match=len(x) == len(y), numeric_mae=float(np.mean(np.abs(np.array(x)-y))) if x and len(x) == len(y) else None,
        answer_numbers=x, gt_numbers=y)


@torch.no_grad()
def run(lm_head, tokenizer, snapshots, dest, args):
    write(dest/'llm_probe_source.json', dict(path=__file__, sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='This module is loaded after E2E, and its actual source is recorded separately.'))
    base = base_model(lm_head)
    layout = CarrierLayout(256, 1, 256)
    device = base.get_input_embeddings().weight.device
    full_events=[]
    native_forward=base.forward
    @wraps(native_forward)
    def full_forward(*a,**kw):
        label='prefill' if kw.get('inputs_embeds') is not None or kw.get('past_key_values') is None else 'decode'
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        result=native_forward(*a,**kw)
        end.record();full_events.append((label,start,end))
        return result
    base.forward=full_forward
    # Existing model wrappers continue to measure actual transformer calls.
    rows = []
    for fi, (token, snap) in enumerate(snapshots.items()):
        if fi % args.frames:
            continue  # one real frame per scene, separately from E2E continuity runs
        ids = [snap['ids'][i].to(device) for i in snap['selected_indices']]
        questions = [snap['questions'][i] for i in snap['selected_indices']]
        vision = snap['carriers'].to(device)
        for mode, selector, ratio in [('full', 'full', 1.), ('exact', 'full', 1.),
                ('compact', 'task', .5), ('compact', 'topk', .75),
                ('compact', 'topk', .5), ('compact', 'random', .5), ('combined', 'task', .5)]:
            runtime = M3Runtime(dict(mode=mode, selector=selector, keep_ratio=ratio,
                offline_experiment=True, cross_frame=False))
            context = dict(scene_id=snap['scene'], frame_id=token, timestamp=float(fi),
                camera_ids=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT'], profile=True)
            state = SimpleNamespace(scene_id=snap['scene'], last_timestamp=float(fi))
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            full_events.clear()
            torch.cuda.synchronize(); tick = perf_counter()
            outputs, diagnostics = runtime.generate(lm_head, ids, vision, world_state=state,
                context=context, questions=questions, layout=layout, do_sample=False, max_new_tokens=320, use_cache=True)
            torch.cuda.synchronize(); elapsed = (perf_counter()-tick)*1000
            decoded = [tokenizer.batch_decode(x, skip_special_tokens=True)[0] for x in outputs]
            record = dict(token=token, scene=snap['scene'], mode=mode, selector=selector, ratio=ratio,
                wall_ms=elapsed, questions=questions, token_ids=[x.cpu().tolist() for x in outputs], answers=decoded,
                diagnostics=diagnostics,
                full_forward_gpu_ms={stage:sum(a.elapsed_time(b) for label,a,b in full_events if label==stage) for stage in ('prefill','decode')},
                full_forward_calls={stage:sum(label==stage for label,a,b in full_events) for stage in ('prefill','decode')},
                quality=[quality(a, snap['ground_truth'][q]) if q in snap['ground_truth'] else None
                    for a,q in zip(decoded,questions)], ground_truth=[snap['ground_truth'].get(q) for q in questions])
            rows.append(record)
            with (dest/'llm_probes.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            print('LLM_PROBE', fi, mode, selector, ratio, round(elapsed,2), flush=True)
    correctness = correctness_probe(lm_head, tokenizer, next(iter(snapshots.values())), dest)
    write(dest/'kv_correctness.json', correctness)
    prefix_ablation(lm_head, snapshots, dest)
    extended_quality(lm_head, tokenizer, snapshots, dest)
    diagnose_exact_failures(lm_head,snapshots,dest)
    write(dest/'llm_summary.json', dict(rows=len(rows)))


@torch.no_grad()
def correctness_probe(lm_head, tokenizer, snap, dest):
    base = base_model(lm_head); device = base.get_input_embeddings().weight.device
    vision = snap['carriers'].to(device)
    ids = [snap['ids'][i].to(device) for i in snap['selected_indices'][:2]]
    results = {}
    for sampling in (False, True):
        kwargs = dict(do_sample=sampling, max_new_tokens=320, return_dict_in_generate=True, output_scores=True)
        if sampling: kwargs.update(temperature=.1, top_p=.75)
        torch.manual_seed(21); torch.cuda.manual_seed_all(21)
        reference = [lm_head.generate(inputs=x[None], images=vision, **kwargs) for x in ids]
        torch.manual_seed(21); torch.cuda.manual_seed_all(21)
        runtime = M3Runtime(dict(mode='exact', offline_experiment=True))
        actual, diagnostics = runtime.generate(lm_head, ids, vision,
            context=dict(scene_id='correctness', frame_id='1', camera_ids=['all']), **kwargs)
        results['sampling' if sampling else 'greedy'] = dict(
            token_ids_equal=[torch.equal(a.sequences,b.sequences) for a,b in zip(reference,actual)],
            finite_logit_max_abs=[max((float((s[torch.isfinite(s) & torch.isfinite(t)] - t[torch.isfinite(s) & torch.isfinite(t)]).abs().max())
                for s,t in zip(a.scores,b.scores)), default=0.) for a,b in zip(reference,actual)], diagnostics=diagnostics)
    expanded = expand_input(lm_head, ids[0], vision, scope=('scene',), signature=('weights',), source=('frame0',))
    cache = PrefixKVCache(512*1024**2); executor = PrefillExecutor(cache)
    executor.generate(base, expanded, max_new_tokens=1, do_sample=False)
    snapshot = tuple(tuple(t.clone() for t in layer) for layer in cache.entry.kv)
    from worldstream.m3.prefix_matcher import exact_lcp
    changed = vision.clone(); changed[0, 0, 0] += 1
    different = expand_input(lm_head, ids[0], changed, scope=('scene',), signature=('weights',))
    results['visual_change_prefix'] = exact_lcp(expanded, different)
    results['identical_prefix'] = cache.match(expanded)
    results['scope_change_prefix'] = cache.match(expand_input(lm_head, ids[0], vision, scope=('new_scene',), signature=('weights',)))
    results['weight_version_change_prefix'] = cache.match(expand_input(lm_head, ids[0], vision, scope=('scene',), signature=('new_lora',)))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        _, trace_diag = executor.generate(base, expanded, expanded.length, max_new_tokens=3, do_sample=False)
        torch.cuda.synchronize()
    (dest/'traces').mkdir(exist_ok=True)
    prof.export_chrome_trace(str(dest/'traces/identical_prefix.json'))
    results['identical_request_profiler'] = trace_diag
    results['cache_immutable'] = all(torch.equal(a,b) for x,y in zip(snapshot,cache.entry.kv) for a,b in zip(x,y))
    tiny = PrefixKVCache(1)
    _, results['budget_fallback'] = PrefillExecutor(tiny).generate(base,expanded,max_new_tokens=1)
    results['memory_bytes'] = cache.nbytes
    results['same_prompt_input_tokens'] = expanded.length
    return results


@torch.no_grad()
def prefix_ablation(lm_head, snapshots, dest):
    """Same real questions: no cache / text-only / visual / cross-frame.

    One generated token isolates prefill logits/cost without confusing changes
    in output length with faster decode. E2E above always uses max_new_tokens=320.
    """
    from dataclasses import replace
    from worldstream.m3.prefix_matcher import exact_lcp
    base = base_model(lm_head); device = base.get_input_embeddings().weight.device
    rows = []
    for index, (token, snap) in enumerate(snapshots.items()):
        vision = snap['carriers'].to(device)
        inputs = [expand_input(lm_head, snap['ids'][i].to(device), vision,
            scope=(snap['scene'],), signature=('actual_loaded_weights',), source=(snap['scene'],token,index,i))
            for i in snap['selected_indices']]
        for mode in ('independent', 'text_prefix', 'same_frame_visual'):
            for rep in range(4):
                cache = PrefixKVCache(512*1024**2); executor = PrefillExecutor(cache)
                records = []
                torch.cuda.synchronize(); tick = perf_counter()
                for x in inputs:
                    matched = cache.match(x)
                    prefix = matched
                    if mode == 'independent': prefix = 0
                    if mode == 'text_prefix':
                        first_image = x.visual_mask[0].nonzero()[0].item() if x.visual_mask.any() else x.length
                        prefix = min(matched, first_image)
                    output, diag = executor.generate(base, x, prefix, capture=mode != 'independent',
                        max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True)
                    diag['last_logits'] = output.scores[0].cpu()
                    records.append(diag)
                torch.cuda.synchronize(); elapsed = (perf_counter()-tick)*1000
                # Store logits separately for exact numerical comparisons.
                logits = [r.pop('last_logits') for r in records]
                (dest/'logits').mkdir(exist_ok=True)
                torch.save(logits, dest/'logits'/f'{token}_{mode}_{rep}.pt')
                row = dict(token=token, mode=mode, repeat=rep, warmup=rep==0, wall_ms=elapsed, requests=records)
                rows.append(row)
        # Cross-frame comparison uses actual changed visual embeddings.
        if index > 0:
            previous_token, previous = list(snapshots.items())[index-1]
            if previous['scene'] == snap['scene']:
                old = expand_input(lm_head, previous['ids'][0].to(device), previous['carriers'].to(device),
                    scope=(snap['scene'],), signature=('actual_loaded_weights',), source=(snap['scene'],previous_token,index-1,0))
                cache = PrefixKVCache(512*1024**2); executor = PrefillExecutor(cache)
                executor.generate(base, old, max_new_tokens=1)
                matching = cache.match(inputs[0])
                _, diag = executor.generate(base, inputs[0], matching, max_new_tokens=1, profile=True)
                rows.append(dict(token=token, previous_token=previous_token, mode='cross_frame', exact_lcp=matching, requests=[diag]))
    write(dest/'prefix_ablation.json', rows)


@torch.no_grad()
def extended_quality(lm_head, tokenizer, snapshots, dest):
    """Additional original questions, selected only by requested semantic domains."""
    patterns = dict(quantity=r'\b(how many|number of|count)\b', distance=r'\b(distance|far|near|close)\b',
        relative_direction=r'\b(left|right|front|behind|direction|located|position)\b',
        lane_map=r'\b(lane|road|intersection|turn)\b',
        motion_interaction=r'\b(moving|motion|interaction|speed|overtak|pedestrian)\w*',
        safety_global=r'\b(safe|safety|risk|influence|affect|impact|conditions|scenario)\b')
    base = base_model(lm_head); device = base.get_input_embeddings().weight.device
    domain_coverage = {key: [] for key in patterns}
    chosen = []
    for token, snap in snapshots.items():
        for domain, pattern in patterns.items():
            matches = [i for i,q in enumerate(snap['questions']) if re.search(pattern,q.lower())]
            domain_coverage[domain].extend(dict(token=token,index=i,question=snap['questions'][i]) for i in matches)
            # One additional original question per domain, globally; no GT filtering.
            if matches and not any(domain in item[3] for item in chosen):
                i = matches[0]
                previous = next((x for x in chosen if x[0] == token and x[2] == i), None)
                if previous: previous[3].append(domain)
                else: chosen.append([token,snap,i,[domain]])
    rows = []
    for token, snap, i, domains in chosen:
        ids = [snap['ids'][i].to(device)]; vision = snap['carriers'].to(device)
        context = dict(scene_id=snap['scene'],frame_id=token,camera_ids=['actual_six_cameras'],timestamp=1.,profile=True)
        state = SimpleNamespace(scene_id=snap['scene'],last_timestamp=1.)
        for mode in ('full','compact'):
            runtime = M3Runtime(dict(mode=mode,selector='task',keep_ratio=.5,offline_experiment=True))
            answer, diag = runtime.generate(lm_head,ids,vision,context=context,world_state=state,
                layout=CarrierLayout(256,1,256),questions=[snap['questions'][i]],do_sample=False,max_new_tokens=320)
            text = tokenizer.batch_decode(answer[0],skip_special_tokens=True)[0]
            gt = snap['ground_truth'].get(snap['questions'][i])
            rows.append(dict(token=token,index=i,domains=domains,mode=mode,question=snap['questions'][i],
                ground_truth=gt,answer=text,token_ids=answer[0].cpu().tolist(),diagnostics=diag,
                quality=quality(text,gt) if gt is not None else None))
    write(dest/'extended_quality.json',dict(domain_coverage=domain_coverage,rows=rows,
        missing_domains=[key for key,value in domain_coverage.items() if not value]))


@torch.no_grad()
def diagnose_exact_failures(lm_head,snapshots,dest):
    """Preserve FP16 batch-shape effects rather than relaxing frozen equality."""
    frames=[json.loads(s) for s in (dest/'frames.jsonl').read_text().splitlines()]
    failures={}
    for row in frames:
        if row['mode']!='exact':continue
        baseline=next(x for x in frames if x['mode']=='baseline' and x['index']==row['index'] and x['repeat']==row['repeat'])
        for qi,(a,b) in enumerate(zip(row['generated_ids'],baseline['generated_ids'])):
            if a!=b:failures[(row['token'],qi)]=row
    base=base_model(lm_head);device=base.get_input_embeddings().weight.device
    results=[]
    for (token,qi),row in failures.items():
        snap=snapshots[token];vision=snap['carriers'].to(device)
        ids=[snap['ids'][i].to(device) for i in snap['selected_indices']]
        x0=expand_input(lm_head,ids[0],vision,scope=('probe',),signature=('weights',))
        x=expand_input(lm_head,ids[qi],vision,scope=('probe',),signature=('weights',))
        cache=PrefixKVCache(512*1024**2);executor=PrefillExecutor(cache)
        executor.generate(base,x0,max_new_tokens=1,do_sample=False)
        prefix=cache.match(x)
        full_prefill=base(inputs_embeds=x.embeddings,attention_mask=x.attention_mask,
            position_ids=x.position_ids,use_cache=True,return_dict=True)
        differences=[dict(layer=i,key_max_abs=float((old[0][...,:prefix,:]-fresh[0][...,:prefix,:]).abs().max()),
                         value_max_abs=float((old[1][...,:prefix,:]-fresh[1][...,:prefix,:]).abs().max()))
                     for i,(old,fresh) in enumerate(zip(cache.entry.kv,full_prefill.past_key_values))]
        del full_prefill
        kw=dict(do_sample=False,max_new_tokens=320,return_dict_in_generate=True,output_scores=True)
        reference=lm_head.generate(inputs=ids[qi][None],images=vision,**kw)
        actual,diag=executor.generate(base,x,prefix,**kw)
        count=min(len(reference.scores),len(actual.scores))
        divergence=next((i for i in range(count) if reference.sequences[0,i+1]!=actual.sequences[0,i+1]),None)
        def logit_details(step):
            a,b=reference.scores[step],actual.scores[step]
            va,ia=a.topk(5);vb,ib=b.topk(5)
            return dict(max_abs=float((a-b).abs().max()),reference_top_ids=ia.tolist(),reference_top_values=va.tolist(),
                cached_top_ids=ib.tolist(),cached_top_values=vb.tolist())
        results.append(dict(token=token,question= snap['questions'][snap['selected_indices'][qi]],
            prefix_tokens=prefix,first_divergent_output_step=divergence,
            prefill_logits=logit_details(0),divergence_logits=logit_details(divergence) if divergence is not None else None,
            prefix_kv_layer_differences=differences,diagnostics=diag,
            reference_tokens=reference.sequences.cpu().tolist(),actual_tokens=actual.sequences.cpu().tolist(),
            conclusion='Exact input dependency is certified, but FP16 computation under different prefill shapes is not bitwise identical; frozen token equality remains failed.'))
    write(dest/'exact_failure_diagnosis.json',results)
