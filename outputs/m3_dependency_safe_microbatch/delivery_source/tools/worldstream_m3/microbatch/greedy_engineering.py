"""Existing adaptive greedy executor on all archived real frame questions.

This supplementary engineering control is not a Sampling benchmark or E2E run.
The complete service's original all-question greedy batch admission (including
its memory-budget fallback) is reported separately without hiding rejected work.
"""
from collections import defaultdict
import torch
from worldstream.m3.batched_executor import AdaptiveBatchExecutor
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.visual_state import base_model,expand_input
from .common import OLD,SAMPLING,measure,write_json


@torch.no_grad()
def run(model,root):
    ss=torch.load(OLD/'phase1/carrier_snapshots.pt',map_location='cpu');groups=defaultdict(list)
    for s in ss:groups[s['scene'],s['frame']].append(s)
    rows=[];base=base_model(model.lm_head)
    generation={**SAMPLING,'do_sample':False};generation.pop('temperature');generation.pop('top_p')
    for (scene,frame),snapshots in list(groups.items())[:2]:
        xs=[expand_input(model.lm_head,torch.tensor(s['prompt_ids'],device='cuda'),s['carriers'].cuda(),
            scope=(scene,frame),signature=('explicit_greedy_control',)) for s in snapshots]
        def serial():
            return [model.lm_head.generate(inputs=torch.tensor(s['prompt_ids'],device='cuda'),images=s['carriers'].cuda(),**generation) for s in snapshots]
        native,native_time=measure(serial)
        cache=PrefixKVCache(512*1024**2)
        executor=AdaptiveBatchExecutor(cache,6144*1024**2,4)
        (answer,diag),timing=measure(lambda:executor.try_generate(base,xs,**generation))
        rows.append(dict(scene=scene,frame=frame,requests=len(xs),native=native_time,existing_adaptive_greedy=timing,
            admitted=answer is not None,diagnostics=diag,
            native_ids=[a.cpu().tolist() for a in native],
            greedy_batch_ids=[a.cpu().tolist() for a in answer] if answer is not None else None,
            exact_tokens_equal=all(torch.equal(a,b) for a,b in zip(native,answer)) if answer is not None else None,
            measured_generation_speedup=native_time['wall_ms']/timing['wall_ms'] if answer is not None else None))
        del native,answer,xs,cache,executor;torch.cuda.empty_cache()
    write_json(root/'greedy_engineering_control.json',dict(rows=rows,
        do_sample=False,not_sampling=True,not_end_to_end=True,
        workload='All different real questions from two archived native_multi frames, original output order restored',
        full_service_all_question_batch_fallback='service/runs.jsonl retains batch_workspace_budget rejection; '
            'this length-grouped existing executor is an explicitly separate supplementary engineering control.'))
