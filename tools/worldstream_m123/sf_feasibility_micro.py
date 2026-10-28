"""Controlled exact-prefix hardware bound, never trained/service evidence."""
import argparse
from collections import defaultdict
from dataclasses import replace
from functools import wraps
import json
import os
from pathlib import Path
from time import perf_counter
import torch
from tools.worldstream_m123.benchmark.common import write,sha256,diagnostic_json
from tools.worldstream_m123.latency_controls import load_capture
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m2.global_residual_benchmark import wait_gpu


def distribution(values):
    import numpy as np
    a=np.asarray(list(values),float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),minimum=float(a.min()),maximum=float(a.max()))


@torch.no_grad()
def run(root):
    import projects.mmdet3d_plugin
    from tools.worldstream_m123.benchmark.worker import effective_config,load_model
    from worldstream.m3.visual_state import expand_input,base_model
    from worldstream.m3.world_prefix_composer import SpatiallyFaithfulComposer
    from worldstream.m3.stable_world import StableAnchorManager,digest
    from worldstream.m3.prefix_kv_cache import PrefixKVCache
    from worldstream.m3.prefill_executor import PrefillExecutor
    from worldstream.m3.reuse_planner import ReusePlanner
    from worldstream.m3.scheduler import Scheduler
    from worldstream.m3.config import M3Config
    from tools.worldstream_m123.benchmark.quality_calibration import paired_quality
    dest=root/'controlled_micro';dest.mkdir(exist_ok=False)
    torch.manual_seed(20261028)
    model=load_model(effective_config('B0','language',dest),'ckpts/iter_10548.pth',dest).eval().requires_grad_(False)
    for name,module in model.named_children():
        if name!='lm_head':module.cpu()
    base=base_model(model.lm_head);torch.cuda.empty_cache()
    composer=SpatiallyFaithfulComposer(4096,64,128).cuda().eval().requires_grad_(False)
    # Previously captured development validation only. No final acceptance data.
    prior=Path('outputs/m3_sf_wrcps_optimization_run01');seq=json.loads((prior/'adaptation_sequence.json').read_text())
    managers={};valid=[];other_anchor=None
    for token in seq['tokens']:
        cap=load_capture(prior/'adaptation_capture/m3_inputs'/(token+'.pt'));s=cap['stable_snapshot']
        a,hit=managers.setdefault(s.scene_id,StableAnchorManager()).update(s,cap['stable_context'])
        if a.stable_state_valid:
            if other_anchor is None:other_anchor=a
            if seq['scene_splits'][s.scene_id]=='validation':valid.append((cap,a,hit))
    if len(valid)<2:raise ValueError('Real previously captured development core frames required')
    source,anchor,_=valid[0];target,target_anchor,hit=valid[1]
    assert hit and anchor.anchor_id==target_anchor.anchor_id
    source_carriers=source['carriers'].cuda();target_carriers=target['carriers'].cuda()
    timing=[];semantic=[];forward_events=[];active_probes=[]
    original=base.forward
    @wraps(original)
    def observed(*args,**kwargs):
        a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record()
        result=original(*args,**kwargs);b.record()
        forward_events.append((a,b,kwargs['inputs_embeds'].shape[1] if kwargs.get('inputs_embeds') is not None else 1))
        return result
    observed._worldstream_read_only_observer=True;base.forward=observed
    def timed(fn):
        torch.cuda.synchronize();a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));tick=perf_counter();a.record()
        value=fn();b.record();torch.cuda.synchronize()
        return value,dict(wall_ms=(perf_counter()-tick)*1000,cuda_stream_span_ms=a.elapsed_time(b))
    overhead=[]
    for repeat in range(11):
        composer.reset()
        _,cold=timed(lambda:composer(target_carriers,target_anchor,use_anchor_cache=True))
        composed,warm=timed(lambda:composer(target_carriers,target_anchor,use_anchor_cache=True))
        _,matching=timed(lambda:composer.objects.plan(target_carriers[:,:256]))
        if repeat>=2:overhead.append(dict(repeat=repeat-2,cold=cold,warm=warm,object_slot_matching=matching))
    source_composed=composer(source_carriers,anchor,use_anchor_cache=True)
    target_composed=composer(target_carriers,target_anchor,use_anchor_cache=True)
    assert torch.equal(source_composed[:,:64],target_composed[:,:64])
    same_content_other_anchor=composer.encode_stable(other_anchor)
    assert torch.equal(composer.encode_stable(anchor),same_content_other_anchor)
    stable_audit=dict(one_fact_semantic_bytes='[["object","car"]]',anchor_ID_enters_encoder=False,
        different_real_anchors_same_embedding=True,distinct_observed_semantic_inputs=1,
        conditional_semantic_entropy_bits_on_car_only_valid_cores=0.,stable_tokens=64,
        adapter_trained=False,scope='Encoder structural identity plus untrained forward witness; capacity/semantic novelty audit, not learned utility proof')
    config=M3Config(reuse_strategy='exact_prefix');planner=ReusePlanner(Scheduler(config))
    for qi in range(3):
        source_exp=expand_input(model.lm_head,source['ids'][qi].cuda(),source_composed,carrier_layout=composer.layout,
            scope=('micro',),signature=('controlled_sf_untrained',),source=('micro','source',0,qi),stable_anchor=(anchor.anchor_id,True))
        target_exp=expand_input(model.lm_head,target['ids'][qi].cuda(),target_composed,carrier_layout=composer.layout,
            scope=('micro',),signature=('controlled_sf_untrained',),source=('micro','target',.5,qi),stable_anchor=(anchor.anchor_id,True))
        text_start=target_exp.visual_group_spans[0][1]
        if source_exp.visual_group_spans[0][1]!=text_start:raise ValueError('Different fixed text prefixes')
        forward_events.clear()
        source_output=base(inputs_embeds=source_exp.embeddings,position_ids=source_exp.position_ids,
            attention_mask=source_exp.attention_mask,use_cache=True,return_dict=True)
        source_kv=source_output.past_key_values;source_logits=source_output.logits
        target_output=base(inputs_embeds=target_exp.embeddings,position_ids=target_exp.position_ids,
            attention_mask=target_exp.attention_mask,use_cache=True,return_dict=True)
        cache=PrefixKVCache(512*1024**2);executor=PrefillExecutor(cache)
        for repeat in range(11):
            order=['native','full','text','text_plus64','same_frame_visual']
            if repeat%3==1:order.reverse()
            if repeat%3==2:order=order[1:]+order[:1]
            for kind in order:
                if kind=='same_frame_visual':
                    assert cache.put(target_exp,target_output.past_key_values,target_output.logits)
                else:assert cache.put(source_exp,source_kv,source_logits)
                actual_match=cache.match(target_exp)
                length={'native':0,'full':0,'text':text_start,'text_plus64':text_start+64,'same_frame_visual':text_start+513}[kind]
                assert actual_match>=text_start+64
                forward_events.clear()
                (decision,details),planning=timed(lambda:planner.plan(target['questions'][qi],target_exp,cache,
                    dict(state_valid=True),dict(frame_index=1),overhead_ms=0.))
                if kind=='native':
                    output,wall=timed(lambda:model.lm_head.generate(inputs=target['ids'][qi].cuda()[None],images=target_carriers,
                        max_new_tokens=1,do_sample=False,num_beams=1,use_cache=True))
                    diag=dict(forward_prefill_lengths=[target_exp.length],effective_prefill_tokens=target_exp.length)
                else:
                    output,wall=timed(lambda:executor.generate(base,target_exp,reuse_length=length,capture=True,
                        profile=True,max_new_tokens=1,do_sample=False,num_beams=1,use_cache=True))
                    tokens,diag=output
                model_ms=sum(a.elapsed_time(b) for a,b,_ in forward_events)
                assert sum(n for _,_,n in forward_events)==target_exp.length-length
                if diag['forward_prefill_lengths']!=[target_exp.length-length]:
                    raise ValueError(dict(kind=kind,expected=target_exp.length-length,diagnostics=diag))
                if repeat>=2:
                    row=dict(question_index=qi,question=target['questions'][qi],repeat=repeat-2,kind=kind,
                        input_tokens=target_exp.length,fixed_text_tokens=text_start,actual_matched_prefix=actual_match,
                        controlled_executor_reused_tokens=length,actual_scheduler_admitted_tokens=decision.reuse_length,
                        scheduler_reason=decision.decision_reason,model_prefill_cuda_ms=model_ms,
                        executor_wall_ms=wall['wall_ms'],executor_stream_span_ms=wall['cuda_stream_span_ms'],
                        planner_wall_ms=planning['wall_ms'],planner_stream_span_ms=planning['cuda_stream_span_ms'],
                        effective_forward_tokens=diag['effective_prefill_tokens'],diagnostics=diag,
                        scope='Controlled prefix-only hardware measurement, one generated token/zero decode forwards; not natural serving or trained WRCPS')
                    timing.append(row)
                    with (dest/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if qi==0:
            # Five extra instrumented calls, excluded from all wall estimates.
            # Kineto CUDA events distinguish GPU activity from stream gaps.
            for kind in ('native','full','text','text_plus64','same_frame_visual'):
                if kind=='same_frame_visual':cache.put(target_exp,target_output.past_key_values,target_output.logits)
                else:cache.put(source_exp,source_kv,source_logits)
                length={'native':0,'full':0,'text':text_start,'text_plus64':text_start+64,'same_frame_visual':text_start+513}[kind]
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    if kind=='native':model.lm_head.generate(inputs=target['ids'][qi].cuda()[None],images=target_carriers,
                        max_new_tokens=1,do_sample=False,num_beams=1,use_cache=True)
                    else:executor.generate(base,target_exp,reuse_length=length,capture=True,profile=True,
                        max_new_tokens=1,do_sample=False,num_beams=1,use_cache=True)
                    torch.cuda.synchronize()
                gpu=[e for e in prof.events() if str(e.device_type).endswith('CUDA')]
                active_probes.append(dict(kind=kind,GPU_activity_duration_sum_ms=sum(e.time_range.elapsed_us() for e in gpu)/1000,
                    GPU_activity_events=len(gpu),scope='One extra Kineto probe, includes device kernels/memory activities; not an uninstrumented wall estimate or E2E CI'))
        del source_output,source_kv,source_logits,target_output;cache.reset()
        print('MICRO',qi+1,'/3',flush=True)
    write(dest/'gpu_activity_probes.json',active_probes)
    write(dest/'composer_cost.json',dict(records=overhead,summary={kind:{metric:distribution(r[kind][metric] for r in overhead)
        for metric in ('wall_ms','cuda_stream_span_ms')} for kind in ('cold','warm','object_slot_matching')},
        scope='Actual K64 structure FP32 CUDA forward; untrained fixed random parameters, cost only, not task quality'))
    # Small diagnostic-only counterfactual: fixed dynamic suffix/layout/RNG,
    # change ONLY the Stable64 content. No admission or physical truth claim.
    sample=target;current=target_carriers;actual=composer(current,target_anchor)
    cf_facts=tuple((f[0],f[1],'bus') for f in target_anchor.facts)
    counterfactual=replace(target_anchor,facts=cf_facts,stable_state_digest=digest(cf_facts))
    cf_stable=composer.encode_stable(counterfactual)
    variants={'untrained_car':actual,'zero_stable_counterfactual':torch.cat([torch.zeros_like(actual[:,:64]),actual[:,64:]],1),
        'bus_label_counterfactual':torch.cat([cf_stable,actual[:,64:]],1)}
    source_file=Path('data/nuscenes/conv/train')/(sample['stable_snapshot'].frame_id+'.json')
    gold={p['question']:p['answer'] for p in json.loads(source_file.read_text())}
    for qi,question in enumerate(sample['questions']):
        ids=sample['ids'][qi].cuda();rng=sample['rng'][qi]
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1])
        native=model.lm_head.generate(inputs=ids[None],images=current,**sample['generation'])
        native_answer=model.tokenizer.batch_decode(native,skip_special_tokens=True)[0];reference=gold[question]
        logits={};answers={}
        for label,images in variants.items():
            expanded=expand_input(model.lm_head,ids,images,carrier_layout=composer.layout)
            logits[label]=base(inputs_embeds=expanded.embeddings,attention_mask=expanded.attention_mask,
                position_ids=expanded.position_ids,use_cache=False).logits[:,-1].float()
            torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1])
            tokens=model.lm_head.generate(inputs=ids[None],images=images,**sample['generation'])
            answer=model.tokenizer.batch_decode(tokens,skip_special_tokens=True)[0]
            answers[label]=dict(answer=answer,tokens=tokens.cpu().tolist(),
                descriptive_quality=paired_quality(answer,native_answer,reference))
        diagnostics={label:dict(max_abs_last_prefill_logit_change=float((value-logits['untrained_car']).abs().max()),
            KL_from_car=float(torch.nn.functional.kl_div(value.log_softmax(-1),logits['untrained_car'].softmax(-1),reduction='batchmean')))
            for label,value in logits.items()}
        semantic.append(dict(frame_id=sample['stable_snapshot'].frame_id,question=question,reference=reference,
            native_answer=native_answer,variants=answers,diagnostics=diagnostics,
            same_dynamic_suffix=True,same_513_layout=True,historical_KV=0,adapter_trained=False,
            scope='Diagnostic counterfactual on untrained Composer only; neither natural SF quality nor evidence of useful learned stable semantics'))
        write(dest/'semantic_diagnostics.json',diagnostic_json(semantic))
        print('SEMANTIC_DIAGNOSTIC',qi+1,'/3',flush=True)
    write(dest/'stable_encoder_audit.json',stable_audit);write(dest/'semantic_diagnostics.json',diagnostic_json(semantic))
    write(dest/'gpu_activity_probes.json',active_probes)
    write(dest/'composer_cost.json',dict(records=overhead,summary={kind:{metric:distribution(r[kind][metric] for r in overhead)
        for metric in ('wall_ms','cuda_stream_span_ms')} for kind in ('cold','warm','object_slot_matching')},
        scope='Actual K64 structure FP32 CUDA forward; untrained fixed random parameters, cost only, not task quality'))
    indexed={(r['question_index'],r['repeat'],r['kind']):r for r in timing};paired=[]
    for r in timing:
        if r['kind']!='text_plus64':continue
        full=indexed[(r['question_index'],r['repeat'],'full')];text=indexed[(r['question_index'],r['repeat'],'text')]
        paired.append(dict(question_index=r['question_index'],repeat=r['repeat'],
            full_minus_text64_model_cuda_ms=full['model_prefill_cuda_ms']-r['model_prefill_cuda_ms'],
            text_minus_text64_model_cuda_ms=text['model_prefill_cuda_ms']-r['model_prefill_cuda_ms'],
            full_minus_text64_executor_wall_ms=full['executor_wall_ms']-r['executor_wall_ms'],
            text_minus_text64_executor_wall_ms=text['executor_wall_ms']-r['executor_wall_ms']))
    write(dest/'summary.json',dict(measured_rows=len(timing),paired=paired,
        summary={k:distribution(p[k] for p in paired) for k in paired[0] if k not in ('question_index','repeat')},
        by_kind={kind:{k:distribution(r[k] for r in timing if r['kind']==kind)
            for k in ('model_prefill_cuda_ms','executor_wall_ms','planner_wall_ms')} for kind in ('native','full','text','text_plus64','same_frame_visual')},
        source_frame=source['stable_snapshot'].frame_id,target_frame=target['stable_snapshot'].frame_id,
        actual_core_hit_pair=True,stable_embeddings_equal=True,base_checkpoint_sha256=sha256('ckpts/iter_10548.pth'),
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),formal_crossframe_service_tokens=0,
        scope='Existing real frozen LLM/executor, controlled exact prefix. Untrained SF representation used ONLY for shape/cost; zero decode forwards. Not service acceleration.',
        current_policy='Actual existing scheduler measured separately; no SF policy approval invented. Control executor lengths are explicit legal exact-prefix microbenchmarks.'))
    base.forward=original


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args()
    if {4,5}&set(os.sched_getaffinity(0)):p.error('Use taskset -c 0-3,6-31')
    with gpu_job_lock():wait_gpu();run(Path(a.root))


if __name__=='__main__':main()
