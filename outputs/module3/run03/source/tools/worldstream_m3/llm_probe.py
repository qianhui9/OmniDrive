"""Paired real-carrier LLM probes, counterexamples and profiler evidence."""
import copy
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
    base = base_model(lm_head)
    layout = CarrierLayout(256, 1, 256)
    device = base.get_input_embeddings().weight.device
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
            torch.cuda.synchronize(); tick = perf_counter()
            outputs, diagnostics = runtime.generate(lm_head, ids, vision, world_state=state,
                context=context, questions=questions, layout=layout, do_sample=False, max_new_tokens=320, use_cache=True)
            torch.cuda.synchronize(); elapsed = (perf_counter()-tick)*1000
            decoded = [tokenizer.batch_decode(x, skip_special_tokens=True)[0] for x in outputs]
            record = dict(token=token, scene=snap['scene'], mode=mode, selector=selector, ratio=ratio,
                wall_ms=elapsed, questions=questions, token_ids=[x.cpu().tolist() for x in outputs], answers=decoded,
                diagnostics=diagnostics, quality=[quality(a, snap['ground_truth'][q]) if q in snap['ground_truth'] else None
                    for a,q in zip(decoded,questions)], ground_truth=[snap['ground_truth'].get(q) for q in questions])
            rows.append(record)
            with (dest/'llm_probes.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            print('LLM_PROBE', fi, mode, selector, ratio, round(elapsed,2), flush=True)
    correctness = correctness_probe(lm_head, tokenizer, next(iter(snapshots.values())), dest)
    write(dest/'kv_correctness.json', correctness)
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
