"""Actual captured-carrier replay and same-state teacher-forced M3 controls.

This diagnostic executes the real checkpoint. Extra logit/legacy-plan probes
exclude this entire run from headline full-system timing.
"""
import argparse
import copy
from functools import wraps
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import sys
import numpy as np
import torch
from tools.worldstream_m123.benchmark.common import write,ground_truth
from tools.worldstream_m123.benchmark.worker import effective_config,load_model,append
from tools.worldstream_m123.benchmark.metrics import distribution
from tools.worldstream_m123.benchmark.quality_calibration import paired_quality
from tools.worldstream_m123.latency_controls import load_capture,before_package
from worldstream.m3 import M3Runtime
from worldstream.m3.visual_state import base_model,expand_input
from worldstream.m3.prefix_matcher import prefix_statistics
from worldstream.m3.scheduler import Scheduler

LIFECYCLE=('_scope','_signature','_timestamp','_state_version','_frame_id','_frame_index','_deployment_key')


def snapshot(runtime):
    return dict(fields={k:getattr(runtime,k) for k in LIFECYCLE},entry=runtime.cache.entry,revision=runtime.cache.revision)


def restore(runtime,value):
    for k,v in value['fields'].items():setattr(runtime,k,v)
    runtime.cache.entry=value['entry'];runtime.cache.revision=value['revision']


def rng(value):
    torch.set_rng_state(value[0]);torch.cuda.set_rng_state(value[1])


@torch.no_grad()
def historical(root,model):
    import json
    old=Path('outputs/m3_quality_bounded_20261026_run01/native_reference')
    qs=[json.loads(x) for x in (old/'per_question_metrics.jsonl').read_text().splitlines()]
    lookup={(x['frame_id'],x['question']):x for x in qs}
    sequence=json.loads((root/'historical_sequence24.json').read_text())
    runtime=M3Runtime(dict(reuse_policy=str(root/'frozen_m3_policy.json')));records=[]
    for token in sequence['tokens']:
        frame=load_capture(old/'m3_inputs'/(token+'.pt'));carriers=frame['carriers'].cuda()
        if frame['questions']!=sequence['questions'][token]:raise ValueError('Historical captured questions differ')
        baselines=[]
        for ids,question,state in zip(frame['ids'],frame['questions'],frame['rng']):
            rng(state)
            output=model.lm_head.generate(inputs=ids.cuda()[None],images=carriers,**frame['generation'])
            if output.cpu().tolist()!=lookup[(token,question)]['token_ids']:raise ValueError('Historical native capture does not reproduce')
            baselines.append(model.tokenizer.batch_decode(output,skip_special_tokens=True)[0])
        rng(frame['rng'][0])
        output,diag=runtime.generate(model.lm_head,[i.cuda() for i in frame['ids']],carriers,
            context=dict(frame['context'],profile=True),questions=frame['questions'],layout=frame['layout'],
            verified_state=frame['verified_state'],tokenizer=model.tokenizer,m2_diagnostics=frame['m2'],**frame['generation'])
        gt=ground_truth(token)
        for i,(question,answer,baseline,d) in enumerate(zip(frame['questions'],output,baselines,diag['requests'])):
            text=model.tokenizer.batch_decode(answer,skip_special_tokens=True)[0]
            row=dict(frame_id=token,scene_id=frame['context']['scene_id'],question_index=i,question=question,
                answer=text,baseline_answer=baseline,quality=paired_quality(text,baseline,gt[question]),diagnostics=d,
                scope='Exact historical native carriers, input IDs, RNG, layout, M1 snapshot, M2 receipt and generation; current M3 only, not full-system E2E')
            records.append(row);append(root/'historical_exact_inputs.jsonl',row)
        print('HISTORICAL EXACT INPUT',len(records),flush=True)
    write(root/'historical_exact_inputs_summary.json',dict(requests=len(records),
        native_tokens_reproduced=True,answers_equal=sum(r['quality']['answer_equal'] for r in records),
        critical_regressions=sum(r['quality']['critical_regressions'] for r in records),
        mean_f1_loss=float(np.mean([r['quality']['f1_loss'] for r in records])),
        reused_tokens=sum(r['diagnostics']['reused_tokens'] for r in records),
        quality_scope='Natural free generation only'))


@torch.no_grad()
def microbenchmark(root,model,frames):
    from torch.profiler import profile,ProfilerActivity
    cases=[];previous=None
    for frame in frames:
        x=expand_input(model.lm_head,frame['ids'][0].cuda(),frame['carriers'].cuda(),
            scope=(frame['context']['scene_id'],),signature=('micro',),carrier_layout=frame['layout'])
        if previous is not None and previous.scope==x.scope:cases.append((previous,x))
        previous=x
        if len(cases)==6:break
    measures=[]
    for case,(a,b) in enumerate(cases):
        full=prefix_statistics(a,b);pruned=prefix_statistics(a,b,include_blocks=False)
        if full[:2]!=pruned[:2]:raise ValueError('Prefix pruning changes exact/legal match')
        for repeat in range(7):
            for label in (('before','after') if repeat%2==0 else ('after','before')):
                torch.cuda.synchronize();start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2));tick=perf_counter();start.record()
                prefix_statistics(a,b,include_blocks=label=='before');end.record();torch.cuda.synchronize()
                if repeat:measures.append(dict(case=case,repeat=repeat,kind=label,wall_ms=(perf_counter()-tick)*1000,cuda_ms=start.elapsed_time(end)))
    if not cases:raise ValueError('No natural adjacent inputs for microbenchmark')
    kernels={}
    for label in ('before','after'):
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            prefix_statistics(*cases[0],include_blocks=label=='before');torch.cuda.synchronize()
        prof.export_chrome_trace(str(root/('prefix_'+label+'_trace.json')))
        kernels[label]={event.key:event.count for event in prof.key_averages() if event.key in ('aten::sqrt','aten::sum','aten::max','aten::stack')}
        import json
        trace=json.loads((root/('prefix_'+label+'_trace.json')).read_text())
        events=[e for e in trace['traceEvents'] if e.get('cat')=='kernel']
        kernels[label].update(cuda_kernel_events=len(events),sampled_active_kernel_total_ms=sum(e.get('dur',0.) for e in events)/1000.)
    write(root/'prefix_pruning_microbenchmark.json',dict(records=measures,real_adjacent_input_pairs=len(cases),
        exact_and_legal_prefix_identical=True,kernel_counts=kernels,
        summary={k:{metric:distribution(r[metric] for r in measures if r['kind']==k) for metric in ('wall_ms','cuda_ms')} for k in ('before','after')},
        scope='Pre-synchronized actual LLM expanded embeddings; warmup discarded; alternating order; isolated statistics only, no E2E claims'))


@torch.no_grad()
def validate_capture_rng(root,model,frames):
    # Before installing any forcing processor, replay six real natural
    # requests with their independently captured per-question CUDA RNG.
    from worldstream.m3.prefill_executor import PrefillExecutor
    from worldstream.m3.prefix_kv_cache import PrefixKVCache
    import hashlib
    replay_executor=PrefillExecutor(PrefixKVCache(0));rng_checks=[]
    for frame in frames[:2]:
        carriers=frame['carriers'].cuda()
        for qi,(ids,state,expected) in enumerate(zip(frame['ids'],frame['rng'],frame['generated_ids'])):
            rng(state)
            expanded=expand_input(model.lm_head,ids.cuda(),carriers,carrier_layout=frame['layout'])
            output,_=replay_executor.generate(base_model(model.lm_head),expanded,capture=False,**frame['generation'])
            if output.cpu().tolist()!=expected.tolist():raise ValueError('Captured per-question RNG does not reproduce natural zero-history IDs')
            rng_checks.append(dict(frame_id=frame['context']['frame_id'],question_index=qi,natural_ids_reproduced=True,
                cuda_rng_sha256=hashlib.sha256(state[1].numpy().tobytes()).hexdigest()))
    write(root/'capture_rng_validation.json',dict(requests=len(rng_checks),all_natural_ids_reproduced=True,forced_selection=False,records=rng_checks))


@torch.no_grad()
def paired(root,model,run_microbenchmark=True):
    import json
    from tools.worldstream_m123.benchmark.observe import Observer
    from tools.worldstream_m123.benchmark.fixed_tokens import FixedTokenBenchmark
    from tools.worldstream_m123.benchmark.m3_recovery import ZeroReuseControl
    from worldstream.m3.task_conditioner import task_profile
    capture=root/'multi_free_r0_zero';sequence=json.loads((root/'multi_sequence24.json').read_text())
    observed_traces=[json.loads(line) for line in (root/'multi_after_diagnostic/decision_traces.jsonl').read_text().splitlines()]
    observed={(t['context']['frame_id'],t['question']):t for t in observed_traces}
    frames=[load_capture(capture/'m3_inputs'/(token+'.pt')) for token in sequence['tokens']]
    validate_capture_rng(root,model,frames)
    # All fixed reference IDs come from actual natural zero-history generation
    # on precisely these carriers. No teacher-forced answer receives a score.
    references={}
    for frame in frames:
        for qi,(question,ids) in enumerate(zip(frame['questions'],frame['generated_ids'])):
            references[frame['context']['frame_id']+'#'+str(qi)]=[dict(question=question,tokens=ids[0,1:].tolist())]
    path=root/'same_input_fixed_reference.json';write(path,dict(source='Actual natural zero-history M3 generation on these exact captured carriers',frames=references))
    runtime=M3Runtime(dict(reuse_policy=str(root/'frozen_m3_policy.json')));model.m3_runtime=runtime
    observer=Observer(model);fixed=FixedTokenBenchmark(model,path);base=base_model(model.lm_head)
    saved_logits=[]
    import tools.worldstream_m123.benchmark.fixed_tokens as fixed_module
    from transformers import LogitsProcessorList
    original_forcing=fixed_module.forced_generation
    class RecordingProcessor(fixed_module.FixedSequenceProcessor):
        def __call__(self,input_ids,scores):
            if len(saved_logits)<4:saved_logits.append(scores.detach().clone())
            return super().__call__(input_ids,scores)
    def forcing(tokens):
        _,overrides=original_forcing(tokens);processor=RecordingProcessor(tokens)
        overrides['logits_processor']=LogitsProcessorList([processor]);return processor,overrides
    fixed_module.forced_generation=forcing
    observed_prefill=[];original_forward=base.forward
    @wraps(original_forward)
    def observe_forward(*args,**kwargs):
        embeds=kwargs.get('inputs_embeds')
        if embeds is not None:observed_prefill.append(int(embeds.shape[1]))
        return original_forward(*args,**kwargs)
    observe_forward._worldstream_read_only_observer=True
    base.forward=observe_forward
    old=before_package(root,'m3');oldplanner_module=sys.modules[old.__name__+'.reuse_planner']
    oldplanner_module.task_profile=task_profile
    legacy_scheduler=Scheduler(runtime.config);legacy=oldplanner_module.ReusePlanner(legacy_scheduler)
    pending={};original_plan=runtime.planner.plan;original_choose=runtime.scheduler.choose
    def choose(candidates,context,overhead_ms):
        # Preserve the ACTUAL original three-question batch's amortized
        # overhead when splitting requests solely for identical-state probes.
        trace=observed[(context['frame_id'],pending['active_question'])]
        held=trace.get('overhead_ms',overhead_ms)
        pending['overhead_ms']=held;pending['standalone_overhead_ms']=overhead_ms
        return original_choose(candidates,context,held)
    runtime.scheduler.choose=choose
    @wraps(original_plan)
    def plan(question,expanded,cache,history,context,m2_diagnostics=None,overhead_ms=0.):
        pending['active_question']=question
        decision,details=original_plan(question,expanded,cache,history,context,m2_diagnostics,overhead_ms)
        pending.update(question=question,expanded=expanded,history=history,context=dict(context),m2=m2_diagnostics,
            decision=decision,details=details,initial_entry=cache.entry,initial_revision=cache.revision)
        return decision,details
    runtime.planner.plan=plan
    records=[];checks=[];ordinal=0
    for frame in frames:
        carriers=frame['carriers'].cuda();ctx=dict(frame['context'],profile=True)
        for qi,(ids,question,state) in enumerate(zip(frame['ids'],frame['questions'],frame['rng'])):
            ids=ids.cuda();initial=snapshot(runtime);outcomes={};reuse_state=None
            orders=(('native','zero','reuse'),('reuse','zero','native'),('zero','native','reuse'))
            for kind in orders[ordinal%3]:
                restore(runtime,initial);runtime.planner.plan=plan
                if kind=='zero':ZeroReuseControl(runtime)
                fixed.prepare(frame['context']['frame_id']+'#'+str(qi));saved_logits.clear();observed_prefill.clear();observer.clear();rng(state)
                torch.cuda.synchronize();tick=perf_counter()
                if kind=='native':
                    output=model.lm_head.generate(inputs=ids[None],images=carriers,**frame['generation']);diag=None
                else:
                    result,diags=runtime.generate(model.lm_head,[ids],carriers,context=ctx,questions=[question],
                        layout=frame['layout'],verified_state=frame['verified_state'],tokenizer=model.tokenizer,
                        m2_diagnostics=frame['m2'],**frame['generation']);output=result[0];diag=diags['requests'][0]
                torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
                fixed.validate_output(0,question,output.cpu().tolist());proof=fixed.collect([{'Q':question}])
                if initial['entry'] is not None and runtime.cache._revision(initial['entry'])!=initial['revision']:
                    raise ValueError('Initial cached storage mutated by a control')
                if diag is not None and observed_prefill!=diag['forward_prefill_lengths']:
                    raise ValueError('Actual model inputs do not match recorded Prefill work')
                outcomes[kind]=dict(wall_ms=wall,fixed_tokens=proof,diagnostics=diag,actual_observed_prefill_forward_lengths=list(observed_prefill),
                    logits=torch.stack(saved_logits).cpu(),token_ids=output.cpu().tolist())
                if kind=='reuse':
                    reuse_state=snapshot(runtime);p=dict(pending)
                    if diag['reused_tokens']!=observed[(ctx['frame_id'],question)]['decision']['reuse_length']:
                        raise ValueError('Same-input probe did not reproduce actual batch reuse')
                    legacy_scheduler.choose=lambda candidates,context,overhead:original_choose(candidates,context,p.get('overhead_ms',overhead))
                    oldcache=SimpleNamespace(entry=p['initial_entry'],revision=p['initial_revision'],validate_storage=lambda:p['initial_entry'])
                    before,_=legacy.plan(question,p['expanded'],oldcache,p['history'],p['context'],p['m2'])
                    keys=('reuse_length','reuse_type','estimated_quality_risk','cache_source_frame','cache_age_frames','approx_reuse_depth','decision_reason','frame_index','root_frame_index','risk_profile')
                    equal={key:getattr(before,key)==getattr(p['decision'],key) for key in keys}
                    if not all(equal.values()):raise ValueError('Legacy/current planner mismatch '+str(equal))
                    checks.append(dict(frame_id=ctx['frame_id'],question_index=qi,checks=equal))
            restore(runtime,reuse_state);runtime.planner.plan=plan
            if sum(outcomes['zero']['actual_observed_prefill_forward_lengths'])-sum(outcomes['reuse']['actual_observed_prefill_forward_lengths'])!=outcomes['reuse']['diagnostics']['reused_tokens']:
                raise ValueError('Actual observed Prefill token reduction differs from KV reuse')
            baseline=outcomes['native']['logits'].float();errors={}
            for kind in ('zero','reuse'):
                values=outcomes[kind].pop('logits').float();delta=values-baseline
                errors[kind]=dict(logit_max_abs=float(delta.abs().max()),logit_relative_rms=float(delta.square().mean().sqrt()/baseline.square().mean().sqrt().clamp_min(1e-12)),
                    sampled_steps=values.shape[0],greedy_argmax_disagreements=int((values.argmax(-1)!=baseline.argmax(-1)).sum()))
            outcomes['native'].pop('logits')
            row=dict(scene_id=ctx['scene_id'],frame_id=ctx['frame_id'],question_index=qi,question=question,
                common_initial_cache_source=list(initial['entry'].computed_source) if initial['entry'] else None,
                same_actual_carriers_ids_rng_history=True,original_batch_scheduler_overhead_held_fixed=True,outcomes=outcomes,compute_errors=errors,
                quality_valid=False,scope='Real full native generation versus same-runtime zero and WSDR; identical cached initial state restored per variant; forced same IDs/steps; diagnostic pre-selection logit probes exclude all timings from headline E2E')
            records.append(row);append(root/'same_input_controls.jsonl',row);ordinal+=1
        print('SAME INPUT CONTROL',ordinal,flush=True)
    fixed_module.forced_generation=original_forcing
    write(root/'planner_equivalence.json',dict(requests=len(checks),all_checks_pass=True,records=checks,
        scope='Same actual input/cache/M1/M2 and conservative task profile; scheduler overhead held to actual observed value to isolate block-statistic omission'))
    if run_microbenchmark:
        single=json.loads((root/'single_sequence48.json').read_text())
        micro_frames=[load_capture(root/'single_free_r0_zero/m3_inputs'/(token+'.pt')) for token in single['tokens'][:8]]
        microbenchmark(root,model,micro_frames)
    write(root/'same_input_control_summary.json',dict(requests=len(records),all_actual_fixed_histories_equal=True,initial_cache_storage_immutable=True,
        planner_equivalence_all_checks_pass=True,headline_performance=False,
        reused_requests=sum(r['outcomes']['reuse']['diagnostics']['reused_tokens']>0 for r in records)))


@torch.no_grad()
def run(root):
    import subprocess,json
    # This single additional traced service run records true KV revisions and
    # the full RMS observer after formal pruning. It is never a timing input.
    journal=root/'control_commands.json'
    entries=json.loads(journal.read_text()) if journal.exists() else []
    for workload,count,questions in (('single',48,1),('multi',24,3)):
        dest=root/(workload+'_after_diagnostic')
        command=['taskset','-c','0-3,6-31',sys.executable,'-u','-m','tools.worldstream_m123.benchmark.worker',
            '--output',str(dest),'--run-root',str(root),'--condition','B3','--workload','language','--limit',str(count),
            '--warmup','1','--seed','0','--questions-per-frame',str(questions),'--sequence-file',str(root/(workload+'_sequence'+str(count)+'.json')),
            '--m3-strategy','wsdr_kv','--m3-policy',str(root/'frozen_m3_policy.json'),'--trace-m3-decisions']
        if not dest.exists():
            with (root/(workload+'_after_diagnostic.log')).open('w') as handle:
                result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
            entries.append(dict(command=command,returncode=result.returncode));write(journal,entries)
            if result.returncode:raise RuntimeError('Post-change trace failed; preserve its log')
        elif json.loads((dest/'status.json').read_text())['status']!='COMPLETED':
            raise ValueError('Preserve incomplete diagnostic run')
    import projects.mmdet3d_plugin
    out=root/'same_input_model';out.mkdir(exist_ok=False)
    cfg=effective_config('B3','language',out,'native_full');model=load_model(cfg,'ckpts/iter_10548.pth',out)
    properties=torch.cuda.get_device_properties(0)
    query=subprocess.run(['taskset','-c','0-3,6-31','nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True)
    write(root/'gpu_runtime_environment.json',dict(device=torch.cuda.get_device_name(0),total_memory=properties.total_memory,capability=torch.cuda.get_device_capability(0),torch=torch.__version__,cuda_runtime=torch.version.cuda,nvidia_smi_returncode=query.returncode,nvidia_smi=query.stdout,scope='Actual authorized CUDA process after strict real-checkpoint load; diagnostic run only'))
    historical(root,model);paired(root,model)


@torch.no_grad()
def rng_only(root):
    import json
    import projects.mmdet3d_plugin
    if (root/'capture_rng_validation.json').exists():return
    out=root/'rng_replay_model';out.mkdir(exist_ok=False)
    cfg=effective_config('B3','language',out,'native_full');model=load_model(cfg,'ckpts/iter_10548.pth',out)
    tokens=json.loads((root/'multi_sequence24.json').read_text())['tokens'][:2]
    frames=[load_capture(root/'multi_free_r0_zero/m3_inputs'/(token+'.pt')) for token in tokens]
    validate_capture_rng(root,model,frames)
    print('Natural per-question RNG replay PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--stage',choices=['all','rng'],default='all');args=p.parse_args()
    if args.stage=='rng':
        from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
        from tools.worldstream_m2.global_residual_benchmark import wait_gpu
        with gpu_job_lock():wait_gpu();rng_only(Path(args.root))
    else:run(Path(args.root))
