"""Finite real-prefix CUDA costs; no decoding/quality or deployment claims."""
import argparse
from dataclasses import replace
from functools import wraps
import json
from pathlib import Path
from time import perf_counter
import torch

from tools.worldstream_m123.benchmark.common import write, sha256
from tools.worldstream_m123.world_visual_memory_analysis import distribution
from tools.worldstream_m123.latency_controls import load_capture
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m2.global_residual_benchmark import wait_gpu


@torch.no_grad()
def run(root, text_control=False):
    import projects.mmdet3d_plugin  # noqa: F401
    from tools.worldstream_m123.benchmark.worker import effective_config, load_model
    from worldstream.m3.visual_state import expand_input, base_model, execution_signature
    from worldstream.m3.visual_memory import VisualMemoryProjection, VisualMemoryManager
    from tools.worldstream_m123.world_visual_memory_semantics import context, regions
    from worldstream.m3.prefix_kv_cache import PrefixKVCache
    from worldstream.m3.prefill_executor import PrefillExecutor
    from worldstream.m3.reuse_planner import ReusePlanner
    from worldstream.m3.scheduler import Scheduler
    from worldstream.m3.config import M3Config
    assert json.loads((root/'bridge_analysis.json').read_text())['status'] == 'PASS'
    assert json.loads((root/'memory_semantic_probe.json').read_text())['status'] == 'PASS'
    dest = root/('controlled_text_cost' if text_control else 'controlled_cost'); dest.mkdir(exist_ok=False)
    frames = [json.loads(l) for l in (root/'bridge_capture/frames.jsonl').open()]
    captures = torch.load(root/'bridge_capture/region_features.pt', weights_only=False, map_location='cpu')
    events = json.loads((root/'memory_lifecycle.json').read_text())['events']['256']
    target_event = next(e for e in events if e['event'] == 'keep')
    source_frame = target_event['generating_frame']; target_frame = target_event['frame_id']
    arow = next(r for r in frames if r['frame_id'] == source_frame)
    brow = next(r for r in frames if r['frame_id'] == target_frame)
    assert brow['timestamp']-arow['timestamp'] <= 1.
    avalues = [x for x in captures if x['source_frame'] == source_frame]
    bvalues = [x for x in captures if x['source_frame'] == target_frame]
    prior = Path('outputs/m3_sf_wrcps_feasibility_run01/state_audit/m3_inputs')
    source = load_capture(prior/(source_frame+'.pt'))
    target = load_capture(prior/(target_frame+'.pt'))
    qualification = json.loads((root/'experiment_plan.json').read_text())['qualification_sha256']
    timing, overhead, activities = [], [], []
    with gpu_job_lock():
        wait_gpu(); torch.manual_seed(20261028)
        model = load_model(effective_config('B0','language',dest),'ckpts/iter_10548.pth',dest).eval().requires_grad_(False)
        for name, module in model.named_children():
            if name != 'lm_head': module.cpu()
        base = base_model(model.lm_head); torch.cuda.empty_cache()
        projection = VisualMemoryProjection().cuda().eval().requires_grad_(False)
        torch.save(dict(state_dict=projection.state_dict(), trained=False, purpose='Cost scaffold only, never deploy'), dest/'untrained_cost_projection.pt')
        current_a, current_b = source['carriers'].cuda(), target['carriers'].cuda()
        forward_events = []; original = base.forward
        @wraps(original)
        def observed(*args, **kwargs):
            a,b = (torch.cuda.Event(enable_timing=True) for _ in range(2)); a.record()
            result = original(*args, **kwargs); b.record()
            forward_events.append((a,b,kwargs['inputs_embeds'].shape[1] if kwargs.get('inputs_embeds') is not None else 1))
            return result
        observed._worldstream_read_only_observer = True; base.forward = observed
        def timed(fn):
            torch.cuda.synchronize(); a,b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            tick=perf_counter(); a.record(); result=fn(); b.record(); b.synchronize()
            return result, dict(wall_ms=(perf_counter()-tick)*1000, stream_ms=a.elapsed_time(b))
        config=M3Config(reuse_strategy='exact_prefix'); planner=ReusePlanner(Scheduler(config))
        for k in (64,128,256):
            manager = VisualMemoryManager(k)
            memory,event = manager.update(context(arow), regions(avalues), arow['qualified_ids'], qualification)
            assert memory is not None
            same, event = manager.update(context(brow), regions(bvalues), brow['qualified_ids'], qualification)
            assert same is memory and event == 'keep'
            memory.features = memory.features.cuda(); memory.tensor_version = memory.features._version
            memories = projection(memory)
            aimages = projection.compose(current_a, memory); bimages=projection.compose(current_b, memory)
            assert projection(memory) is memories and torch.equal(aimages[:,:k], bimages[:,:k])
            signature, binding_cost = timed(lambda:execution_signature(base, context(arow)['model_sha256']))
            for repeat in range(11):
                projection._key = None
                _, cold = timed(lambda:projection.compose(current_b, memory))
                _, warm = timed(lambda:projection.compose(current_b, memory))
                if repeat >= 2: overhead.append(dict(K=k,repeat=repeat-2,cold=cold,warm=warm,
                    execution_binding_ms=binding_cost['wall_ms']))
            for qi in range(3):
                def expanded(cap, images, frame):
                    ex=expand_input(base,cap['ids'][qi].cuda(),images,scope=('wrvmp_cost',arow['scene_id']),
                        signature=signature,source=(arow['scene_id'],frame),
                        stable_anchor=(str((memory.scope,memory.revision)),True))
                    start=int(ex.visual_mask[0].nonzero()[0])
                    return replace(ex,layout_signature='wrvmp_append_full_native_cost_v1',
                        visual_group_spans=(('stable',start,start+k),('current_native',start+k,start+k+513)))
                aex=expanded(source,aimages,source_frame); bex=expanded(target,bimages,target_frame)
                text=bex.visual_group_spans[0][1]
                assert text == aex.visual_group_spans[0][1]
                computed=base(inputs_embeds=aex.embeddings,position_ids=aex.position_ids,
                    attention_mask=aex.attention_mask,use_cache=True,return_dict=True)
                cache=PrefixKVCache(1024*1024**2); executor=PrefillExecutor(cache)
                for repeat in range(11):
                    order=['native','memory_text_only','memory_exact'] if text_control else ['native','memory_zero','memory_exact']
                    if repeat%3==1:order.reverse()
                    if repeat%3==2:order=order[1:]+order[:1]
                    for kind in order:
                        assert cache.put(aex,computed.past_key_values,computed.logits)
                        match, matching = timed(lambda:cache.match(bex))
                        assert match >= text+k
                        (decision,details), planning = timed(lambda:planner.plan(target['questions'][qi],bex,cache,
                            dict(state_valid=True),dict(frame_index=1),overhead_ms=0.))
                        forward_events.clear()
                        length=text+k if kind=='memory_exact' else text if kind=='memory_text_only' else 0
                        if kind=='native':
                            _, measurement=timed(lambda:model.lm_head.generate(inputs=target['ids'][qi].cuda()[None],
                                images=current_b,max_new_tokens=1,do_sample=False,num_beams=1,use_cache=True))
                            expected=bex.length-k; diag={}
                        else:
                            (_,diag), measurement=timed(lambda:executor.generate(base,bex,reuse_length=length,capture=True,
                                profile=True,max_new_tokens=1,do_sample=False,num_beams=1,use_cache=True))
                            expected=bex.length-length
                        assert sum(n for _,_,n in forward_events)==expected
                        cuda_ms=sum(a.elapsed_time(b) for a,b,_ in forward_events)
                        if repeat>=2:
                            row=dict(K=k,question_index=qi,repeat=repeat-2,round=(repeat-2)//3,
                                kind=kind,input_tokens=bex.length if kind!='native' else bex.length-k,
                                actual_matched_tokens=match,controlled_skipped_tokens=length,
                                actual_scheduler_admitted_tokens=decision.reuse_length,
                                scheduler_reason=decision.decision_reason,model_prefill_cuda_ms=cuda_ms,
                                executor_wall_ms=measurement['wall_ms'],executor_stream_ms=measurement['stream_ms'],
                                matcher_wall_ms=matching['wall_ms'],planner_wall_ms=planning['wall_ms'],
                                effective_forward_tokens=expected,decode_forward_steps=0,diagnostics=diag)
                            timing.append(row)
                            with (dest/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                del computed;cache.reset()
            # Separate actual GPU activity probe, excluded from wall estimates.
            projection._key=None
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                projection.compose(current_b,memory);torch.cuda.synchronize()
            gpu=[e for e in prof.events() if str(e.device_type).endswith('CUDA')]
            activities.append(dict(K=k,projection_composition_gpu_activity_ms=sum(e.time_range.elapsed_us() for e in gpu)/1000,
                gpu_events=len(gpu),scope='Separate cold projection+assembly profiler probe, not extraction GPU or E2E'))
            print('WRVMP_COST',k,flush=True)
        base.forward=original
        write(dest/'overhead.json',overhead);write(dest/'gpu_activity_probes.json',activities)
        write(dest/'summary.json',dict(status='COMPLETED',rows=len(timing),
            by_K={str(k):{kind:{metric:distribution([r[metric] for r in timing if r['K']==k and r['kind']==kind])
                for metric in ('model_prefill_cuda_ms','executor_wall_ms','matcher_wall_ms','planner_wall_ms')}
                for kind in (('native','memory_text_only','memory_exact') if text_control else ('native','memory_zero','memory_exact'))} for k in (64,128,256)},
            source_frame=source_frame,target_frame=target_frame,source_raw_feature_hash=memory.content_sha256,
            source_carriers_sha256=sha256(prior/(source_frame+'.pt')),
            target_carriers_sha256=sha256(prior/(target_frame+'.pt')),
            untrained_projection_sha256=sha256(dest/'untrained_cost_projection.pt'),
            base_weights_sha256=context(arow)['model_sha256'],
            projection_trained=False,LLM_quality='NOT_RUN',formal_service_KV_tokens=0,
            layout='K historical real regional embeddings + complete current513',
            repeats=9,questions=3,alternating_rounds=3,decode_forward_steps=0,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
            scope='Finite prefix/executor cost only. Existing planner admission reported separately; controlled executor length is not formal scheduler success. No LLM answers or quality are evaluated.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/m3_world_visual_memory_run01'))
    p.add_argument('--text-control',action='store_true',help='Separate matched control to isolate K from generic fixed text reuse')
    args=p.parse_args()
    run(args.root,args.text_control)
