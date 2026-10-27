"""Real captured-input audit and alternating GPU planner microbenchmark.

Extra probes exclude this run from service latency evidence. No synthetic
embeddings, changed question order, or policy writes are used.
"""
import argparse
from collections import Counter
from functools import wraps
from pathlib import Path
import pstats
import sys
from time import perf_counter
import torch

from tools.worldstream_m123.benchmark.common import write
from tools.worldstream_m123.benchmark.worker import effective_config, load_model, append
from tools.worldstream_m123.benchmark.metrics import distribution
from tools.worldstream_m123.latency_controls import load_capture, before_package
from tools.worldstream_m2.global_residual_benchmark import read, rows
from worldstream.m3 import M3Runtime
from worldstream.m3.prefix_matcher import prefix_statistics


@torch.no_grad()
def run(root):
    import projects.mmdet3d_plugin
    dest=root/'paired_diagnostics';dest.mkdir(exist_ok=False)
    model=load_model(effective_config('B3','language',dest,'wsdr_kv'),'ckpts/iter_10548.pth',dest)
    old=before_package(root,'m3')
    old_planner=sys.modules[old.__name__+'.reuse_planner'].ReusePlanner
    old_scheduler=sys.modules[old.__name__+'.scheduler'].Scheduler
    old_config=sys.modules[old.__name__+'.config'].M3Config
    runtime=M3Runtime(dict(reuse_policy=str(root/'frozen_m3_policy.json')))
    previous=old_planner(old_scheduler(old_config(reuse_policy=str(root/'frozen_m3_policy.json'))))
    old_statistics=sys.modules[old.__name__+'.prefix_matcher'].prefix_statistics
    traces={(t['context']['frame_id'],t['question']):t for t in rows(root/'development_diagnostic/decision_traces.jsonl')}
    micro=[];checks=[];prefixes=[];profiled=set()
    native_plan=runtime.planner.plan
    old_choose=previous.scheduler.choose;new_choose=runtime.scheduler.choose
    @wraps(native_plan)
    def plan(question,expanded,cache,history,context,m2_diagnostics=None,overhead_ms=0.):
        trace=traces[(context['frame_id'],question)];overhead=trace.get('overhead_ms',0.)
        previous.scheduler.choose=lambda cs,ctx,ms:old_choose(cs,ctx,overhead)
        runtime.scheduler.choose=lambda cs,ctx,ms:new_choose(cs,ctx,overhead)
        old_decision,old_details=previous.plan(question,expanded,cache,history,context,m2_diagnostics,overhead_ms)
        new_decision,new_details=native_plan(question,expanded,cache,history,context,m2_diagnostics,overhead_ms)
        fields=('reuse_length','reuse_type','cache_source_frame','cache_age_frames','approx_reuse_depth',
            'estimated_quality_risk','root_frame_index','decision_reason')
        check={k:getattr(old_decision,k)==getattr(new_decision,k) for k in fields}
        checks.append(dict(frame_id=context['frame_id'],question=question,checks=check))
        if not all(check.values()):raise ValueError('Pruning changed a fixed-overhead decision')
        entry=cache.entry
        if entry is not None and entry.expanded.source[:2]!=expanded.source[:2]:
            a,b=entry.expanded,expanded
            exact,legal,blocks=old_statistics(a,b,32)
            group_stats={}
            for group,start,stop in b.visual_group_spans:
                aa=a.embeddings[0,start:stop];bb=b.embeddings[0,start:stop]
                equal=(aa==bb).all(-1)
                delta=(aa.float()-bb.float()).square().sum(-1)
                scale=torch.maximum(aa.float().square().sum(-1),bb.float().square().sum(-1)).clamp_min(1e-12)
                group_stats[group]=dict(tokens=stop-start,strictly_equal_tokens=int(equal.sum()),
                    relative_rms=float((delta.sum()/scale.sum()).sqrt()),max_token_change=float((delta/scale).sqrt().max()))
            prefixes.append(dict(scene_id=context['scene_id'],frame_id=context['frame_id'],question=question,
                source=list(entry.computed_source),root_frame_index=entry.root_frame_index,
                exact_prefix=exact,structural_prefix=legal,groups=group_stats,blocks=blocks,
                world_signals={k:v for k,v in new_details.items() if k.startswith(('m1_','m2_'))},
                approximate_statistics_rejection=new_details.get('approximate_statistics_rejection'),
                endpoints={str(size):[x['end'] for x in prefix_statistics(a,b,size)[2]] for size in (16,32,64)}))
            # At most six distinct real adjacent frame pairs; alternate order.
            pair=(entry.computed_source[1],context['frame_id'])
            if pair not in profiled and len(profiled)<6:
                profiled.add(pair)
                for repeat in range(7):
                    for label in (('before','after') if repeat%2==0 else ('after','before')):
                        torch.cuda.synchronize();start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                        tick=perf_counter();start.record()
                        if label=='before':previous.plan(question,expanded,cache,history,context,m2_diagnostics,overhead_ms)
                        else:native_plan(question,expanded,cache,history,context,m2_diagnostics,overhead_ms)
                        end.record();torch.cuda.synchronize()
                        if repeat:micro.append(dict(pair=pair,repeat=repeat,kind=label,wall_ms=(perf_counter()-tick)*1000,cuda_ms=start.elapsed_time(end),
                            profile=new_details['task_profile'],before_blocks=len(old_details.get('block_changes',[])),after_blocks=len(new_details.get('block_changes',[]))))
        return new_decision,new_details
    runtime.planner.plan=plan
    for token in read(root/'development_sequence.json')['tokens']:
        frame=load_capture(root/'development_diagnostic/m3_inputs'/(token+'.pt'))
        # Restore the originally captured generator state at each request.
        for qi,(ids,question,state) in enumerate(zip(frame['ids'],frame['questions'],frame['rng'])):
            torch.set_rng_state(state[0]);torch.cuda.set_rng_state(state[1])
            output,diag=runtime.generate(model.lm_head,[ids.cuda()],frame['carriers'].cuda(),
                context=dict(frame['context'],profile=True),questions=[question],layout=frame['layout'],
                verified_state=frame['verified_state'],tokenizer=model.tokenizer,m2_diagnostics=frame['m2'],**frame['generation'])
            append(dest/'executions.jsonl',dict(frame_id=token,question=question,diagnostics=diag['requests'][0],token_ids=output[0].cpu().tolist()))
        print('AUDIT',token,flush=True)
    for name,data in [('prefix_structure.json',prefixes),('planner_equivalence.json',dict(all_passed=True,requests=len(checks),checks=checks)),
            ('planner_microbenchmark.json',dict(records=micro,pairs=len(profiled),warmup_excluded=True,
                summary={kind:{key:distribution(x[key] for x in micro if x['kind']==kind) for key in ('wall_ms','cuda_ms')} for kind in ('before','after')},
                scope='Actual checkpoint embeddings/cache, same frozen cost overhead for decision equivalence; extra probes excluded from E2E'))]:write(root/name,data)
    f=root/'development_diagnostic/profile_output/cpu.prof'
    if f.exists():
        stats=pstats.Stats(str(f)).stats
        write(root/'cpu_profile_summary.json',sorted([dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_ms=v[2]*1000,cumulative_ms=v[3]*1000) for k,v in stats.items()],key=lambda x:-x['self_ms'])[:150])


@torch.no_grad()
def prefill_control(root):
    """Limited development-only zero/trial equal-token GPU cost evidence."""
    import projects.mmdet3d_plugin
    from worldstream.m3.visual_state import base_model
    from tools.worldstream_m123.benchmark.fixed_tokens import forced_generation
    from tools.worldstream_m123.benchmark.m3_recovery import ZeroReuseControl
    dest=root/'development_fixed_control';dest.mkdir(exist_ok=False)
    model=load_model(effective_config('B3','language',dest,'native_full'),'ckpts/iter_10548.pth',dest)
    base=base_model(model.lm_head);original=base.forward;events=[];decode_inputs=[];observed_prefills=[]
    @wraps(original)
    def forward(*args,**kwargs):
        kind='prefill' if kwargs.get('inputs_embeds') is not None else 'decode'
        if kind=='decode':decode_inputs.append(kwargs['input_ids'].detach())
        else:
            kv=kwargs.get('past_key_values');positions=kwargs['position_ids']
            observed_prefills.append(dict(forward_tokens=kwargs['inputs_embeds'].shape[1],
                past_tokens=0 if kv is None else kv[0][0].shape[-2],
                kv_shapes=[] if kv is None else [[list(t.shape) for t in layer] for layer in kv],
                attention_mask_shape=list(kwargs['attention_mask'].shape),
                first_position=positions[0,0].detach(),last_position=positions[0,-1].detach()))
        start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2));start.record()
        output=original(*args,**kwargs);end.record();events.append((kind,start,end))
        return output
    forward._worldstream_read_only_observer=True
    base.forward=forward
    sequence=read(root/'development_sequence.json')
    calibration_tokens={t for scene,part in sequence['scene_splits'].items() if part=='calibration'
        for t in sequence['continuity'][scene]['sample_tokens']}
    frames=[load_capture(root/'development_diagnostic/m3_inputs'/(t+'.pt')) for t in sequence['tokens'] if t in calibration_tokens]
    threshold=max(read(root/'calibration128/thresholds.json')['values']);records=[]
    for repeat,order in enumerate((('zero','trial'),('trial','zero'),('zero','trial'))):
        for kind in order:
            runtime=M3Runtime();runtime.planner.calibration_threshold=threshold;runtime.planner.calibration_max_visual_tokens=128
            if kind=='zero':ZeroReuseControl(runtime)
            original_plan=runtime.planner.plan;input_overhead=[]
            def observe_plan(*a,_original=original_plan,**kw):
                input_overhead.append(kw.get('overhead_ms',0.))
                return _original(*a,**kw)
            runtime.planner.plan=observe_plan
            for frame in frames:
                for qi,(ids,question,state,generated) in enumerate(zip(frame['ids'],frame['questions'],frame['rng'],frame['generated_ids'])):
                    tokens=generated[0,1:].tolist();processor,overrides=forced_generation(tokens)
                    torch.set_rng_state(state[0]);torch.cuda.set_rng_state(state[1]);events.clear();decode_inputs.clear();observed_prefills.clear();input_overhead.clear()
                    carriers=frame['carriers'].cuda();ids=ids.cuda();torch.cuda.synchronize();tick=perf_counter()
                    outputs,diag=runtime.generate(model.lm_head,[ids],carriers,
                        context=dict(frame['context'],profile=True),questions=[question],layout=frame['layout'],
                        verified_state=frame['verified_state'],tokenizer=model.tokenizer,m2_diagnostics=frame['m2'],
                        **dict(frame['generation'],**overrides))
                    torch.cuda.synchronize();wall=(perf_counter()-tick)*1000;d=diag['requests'][0]
                    actual=torch.cat([v.reshape(-1) for v in decode_inputs]).cpu().tolist() if decode_inputs else []
                    if outputs[0][0,1:].cpu().tolist()!=tokens or actual!=tokens[:-1] or processor.calls!=len(tokens):
                        raise ValueError('Development fixed-token forward history mismatch')
                    if sum(d['forward_prefill_lengths'])!=d['input_tokens']-d['reused_tokens']:
                        raise ValueError('Development prefill forward mismatch')
                    if [v['forward_tokens'] for v in observed_prefills]!=d['forward_prefill_lengths']:
                        raise ValueError('Observed actual model input shapes disagree with executor')
                    observed=[dict(v,first_position=int(v['first_position']),last_position=int(v['last_position'])) for v in observed_prefills]
                    record=dict(repeat=repeat,kind=kind,scene_id=frame['context']['scene_id'],frame_id=frame['context']['frame_id'],
                        question_index=qi,question=question,reference_steps=len(tokens),actual_decode_forward_calls=len(decode_inputs),
                        decode_history_equal=True,m3_wall_ms=wall,observed_prefills=observed,
                        scheduler_overhead_ms=input_overhead[0]+d['planning_ms'],
                        cost_reference='same M3Runtime/PrefillExecutor with ZeroReuseControl',
                        **{label+'_cuda_ms':sum(a.elapsed_time(b) for k,a,b in events if k==label) for label in ('prefill','decode')},diagnostics=d)
                    records.append(record);append(dest/'records.jsonl',record)
            print('DEVELOPMENT FIXED',repeat,kind,flush=True)
    paired=[]
    index={(r['repeat'],r['kind'],r['frame_id'],r['question_index']):r for r in records}
    for r in records:
        d=r['diagnostics']
        if r['kind']!='trial' or not d['reused_tokens'] or d['cache_source_frame']==r['frame_id']:continue
        zero=index[(r['repeat'],'zero',r['frame_id'],r['question_index'])]
        paired.append(dict(repeat=r['repeat'],scene_id=r['scene_id'],frame_id=r['frame_id'],question=r['question'],
            reused_tokens=d['reused_tokens'],visual_tokens=sum(max(0,min(d['reused_tokens'],b)-a) for _,a,b in d['visual_group_spans']),
            forward_lengths=d['forward_prefill_lengths'],source_frame=d['cache_source_frame'],risk=d['estimated_quality_risk'],
            **{key+'_saved':zero[key]-r[key] for key in ('prefill_cuda_ms','decode_cuda_ms','m3_wall_ms')}))
    write(root/'development_fixed_control_summary.json',dict(records=len(records),independent_scenes=2,paired=paired,
        by_length={str(length):{key:distribution(r[key] for r in paired if r['reused_tokens']==length)
            for key in ('prefill_cuda_ms_saved','decode_cuda_ms_saved','m3_wall_ms_saved')} for length in sorted({r['reused_tokens'] for r in paired})},
        scope='Development-only actual checkpoint, original ordered natural questions, common captured free-generated token IDs and identical T-1 actual decode forwards, three alternating rounds. No task quality from forced output and no service/independent acceptance claim. Trial remains rejected on natural semantic/validation evidence.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args()
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    with gpu_job_lock():wait_gpu();run(Path(a.root))
