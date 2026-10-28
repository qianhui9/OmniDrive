"""Bounded WR-CVMS feasibility. Untrained projection, never task/deployment proof."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from functools import wraps
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from tools.worldstream_m123.benchmark.common import sha256, write
from tools.worldstream_m123.latency_controls import load_capture
from tools.worldstream_m123.world_visual_memory_semantics import context, regions
from worldstream.m3.visual_memory import VisualMemoryManager
from worldstream.m3.world_prefix_composer import CompactVisualComposer

ROOT = Path('outputs/m3_compact_visual_memory_run01')
OLD = Path('outputs/m3_world_visual_memory_run01')


def prepare(root):
    """Reuse frozen sequences, never select scenes by this run's outcome."""
    import shutil
    root.mkdir(parents=True,exist_ok=False)
    for name,source in (('natural',OLD/'natural_audit'),('enriched',OLD)):
        dest=root/(name+'_live');dest.mkdir()
        shutil.copyfile(source/'bridge_sequence.json',dest/'bridge_sequence.json')
        plan=json.loads((source/'experiment_plan.json').read_text())
        plan['compact_memory_tokens']=8
        write(dest/'experiment_plan.json',plan)


def stats(values):
    xs = np.asarray(values, dtype=float)
    return dict(n=len(xs), mean=float(xs.mean()), median=float(np.median(xs)),
                p95=float(np.percentile(xs, 95))) if len(xs) else None


def dataset(root, name):
    live = root/(name+'_live')
    source = live if (live/'bridge_capture/status.json').exists() else OLD/('natural_audit' if name=='natural' else '')
    status = json.loads((source/'bridge_capture/status.json').read_text())
    if status['status'] != 'COMPLETED':
        raise ValueError('Incomplete capture: '+str(source))
    rows = [json.loads(s) for s in (source/'bridge_capture/frames.jsonl').read_text().splitlines()]
    values = torch.load(source/'bridge_capture/region_features.pt', map_location='cpu', weights_only=False)
    indexed = defaultdict(list)
    for x in values:
        indexed[x['source_frame']].append(x)
    qualification = json.loads((source/'experiment_plan.json').read_text())['qualification_sha256']
    return source, rows, indexed, qualification


def carrier_path(source, frame):
    path = source/'bridge_capture/m3_inputs'/(frame+'.pt')
    if path.exists():
        return path
    return Path('outputs/m3_sf_wrcps_feasibility_run01/state_audit/m3_inputs')/(frame+'.pt')


@torch.no_grad()
def replay(root, name, k, base=None):
    from worldstream.m3.visual_state import expand_input
    from worldstream.m3.prefix_matcher import exact_lcp
    source, rows, indexed, qualification = dataset(root, name)
    manager = VisualMemoryManager(k)
    composer = CompactVisualComposer(4096, k).eval().requires_grad_(False)
    if base is not None:
        composer.cuda()
    events, previous_memory, previous_embedding, previous_inputs = [], None, None, {}
    lengths = Counter()
    for row in rows:
        current = regions(indexed[row['frame_id']])
        if base is not None:
            current = [replace(r, features=r.features.cuda()) for r in current]
        memory, event = manager.update(context(row), current, row['qualified_ids'], qualification)
        retained = False
        prefix = {}
        if memory is not None:
            embedding = composer.memory_projection(memory)
            if event == 'keep':
                assert memory is previous_memory and embedding is previous_embedding
                retained = torch.equal(embedding, previous_embedding)
            previous_memory, previous_embedding = memory, embedding
            lengths[(row['scene_id'],memory.revision)] += 1
        else:
            previous_memory = previous_embedding = None
        inputs = {}
        if base is not None:
            cap = load_capture(carrier_path(source,row['frame_id']))
            images = composer(cap['carriers'].cuda(), memory)
            from worldstream.m3.visual_state import CarrierLayout
            layout = composer.layout if memory else CarrierLayout(256,1,256)
            for qi, ids in enumerate(cap['ids'][:3]):
                ex = expand_input(base, ids.cuda(), images, scope=(row['scene_id'],row['state_epoch']),
                    signature=('wr_cvms_cost_projection',k), source=(row['scene_id'],row['frame_id']),
                    carrier_layout=layout, stable_anchor=(repr((memory.scope,memory.revision)),True) if memory else ('invalid',False))
                inputs[qi] = ex
                previous = previous_inputs.get(qi)
                length = exact_lcp(previous,ex) if previous is not None else 0
                start = ex.visual_group_spans[0][1]
                visual = min(k,max(0,length-start)) if memory and event=='keep' else 0
                prefix[str(qi)] = dict(exact_prefix_tokens=length, stable_visual_tokens=visual,
                    dynamic_tokens_matched=max(0,length-start-k) if visual else 0)
            # Three-question deployment owns ONE cache. Compare last previous
            # question to first current; same-frame full visual matches excluded.
            previous = previous_inputs.get('last')
            first = inputs.get(0)
            length = exact_lcp(previous,first) if previous is not None and first is not None else 0
            start = first.visual_group_spans[0][1] if first is not None else 0
            prefix['three_question_boundary'] = dict(exact_prefix_tokens=length,
                stable_visual_tokens=min(k,max(0,length-start)) if memory and event=='keep' else 0)
            if inputs:inputs['last'] = inputs[max(i for i in inputs if isinstance(i,int))]
        previous_inputs = inputs
        events.append(dict(scene_id=row['scene_id'],frame_id=row['frame_id'],event=event,
            memory_valid=memory is not None,revision=memory.revision if memory else None,
            anchor_id=memory.anchor_id if memory else None,
            generating_frame=memory.generating_frame if memory else None,
            age=memory.age if memory else None,content_sha256=memory.content_sha256 if memory else None,
            selected_grid_indices=list(memory.grid_indices) if memory else None,
            exact_held_tensor=retained,actual_prefix_checks=prefix,actual_service_KV_tokens=0))
    valid = sum(e['memory_valid'] for e in events)
    keeps = sum(e['event']=='keep' for e in events)
    opportunities = sum(e['actual_prefix_checks'].get('0',{}).get('stable_visual_tokens',0)==k for e in events)
    three = sum(e['actual_prefix_checks'].get('three_question_boundary',{}).get('stable_visual_tokens',0)==k for e in events)
    result = dict(K=k,frames=len(rows),scenes=len({r['scene_id'] for r in rows}),valid_frames=valid,
        creates=sum(e['event'].startswith('create:') for e in events),continuous_keeps=keeps,
        mean_lifetime_observed_frames=float(np.mean(list(lengths.values()))) if lengths else None,
        max_lifetime_observed_frames=max(lengths.values(),default=0),memory_coverage=valid/len(rows),
        kept_frame_fraction=keeps/len(rows),hold_rate_given_memory=keeps/valid if valid else None,
        actual_prefix_verified=base is not None,one_question_exact_opportunities=opportunities if base else None,
        three_question_crossframe_opportunities=three if base else None,
        one_question_hit_probability=opportunities/len(rows) if base else None,
        events=events,qualification_sha256=qualification,capture=str(source),
        scope='Finite causal playback; enriched sample is availability-selected. Prefix opportunity is not executed KV or scheduler approval.')
    return result


def coverage(root):
    torch.manual_seed(20261028)
    report = {name:{str(k):replay(root,name,k) for k in (8,16)} for name in ('natural','enriched')}
    report['status']='REAL_FEATURE_LIFECYCLE_PREFIX_PENDING'
    write(root/'real_memory_coverage.json',report)
    print(json.dumps({name:{k:{field:v for field,v in r.items() if field not in ('events',)}
        for k,r in report[name].items()} for name in ('natural','enriched')}),flush=True)


@torch.no_grad()
def cost(root):
    import projects.mmdet3d_plugin  # noqa: F401
    from tools.worldstream_m123.benchmark.worker import effective_config, load_model
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    from worldstream.m3.visual_state import base_model, expand_input, execution_signature, CarrierLayout
    from worldstream.m3.prefix_kv_cache import PrefixKVCache
    from worldstream.m3.prefill_executor import PrefillExecutor
    from worldstream.m3.reuse_planner import ReusePlanner
    from worldstream.m3.scheduler import Scheduler
    from worldstream.m3.config import M3Config
    dest=root/'controlled_cost'; dest.mkdir(exist_ok=False)
    with gpu_job_lock():
        wait_gpu(); torch.manual_seed(20261028)
        witness = torch.arange(64,dtype=torch.float32,device='cuda').reshape(8,8)
        assert float((witness@witness.T).sum())==510720.
        write(root/'environment.json',dict(python=os.sys.version,torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
            affinity=sorted(os.sched_getaffinity(0)),witness=510720.,environment_rebuilt=False))
        model=load_model(effective_config('B0','language',dest),'ckpts/iter_10548.pth',dest).eval().requires_grad_(False)
        for name,module in model.named_children():
            if name!='lm_head':module.cpu()
        base=base_model(model.lm_head);torch.cuda.empty_cache()
        # Real frozen LLM, native carriers and question; no eligible identities
        # means no EVA read, no adapter loading and no composer invocation.
        from worldstream.m3.runtime import M3Runtime
        from worldstream.m3.stable_world import StableWorldSnapshot
        natural_source,natural_rows,_,qualification=dataset(root,'natural')
        row=natural_rows[0];cap=load_capture(carrier_path(natural_source,row['frame_id']))
        ctx=dict(cap['context'],camera_ids=row['camera_order'],model_version=context(row)['model_sha256'],
                 weights_sha256=context(row)['model_sha256'],stable_state_version=row['state_version'],profile=True)
        snapshot=StableWorldSnapshot(row['scene_id'],row['frame_id'],row['timestamp'],row['state_epoch'],
            row['state_version'],ctx['state_execution_signature'],(),'{}',qualification_id=qualification)
        policy_path=str(OLD/'experiment_plan.json')
        policy_path=json.loads(Path(policy_path).read_text())['qualification_path']
        native_runtime=M3Runtime(M3Config(reuse_strategy='exact_prefix'))
        compact_runtime=M3Runtime(M3Config(reuse_strategy='exact_prefix',prefix_layout='world_compact_memory',semantic_qualification=policy_path))
        carrier=cap['carriers'].cuda();ids=cap['ids'][0].cuda()
        kwargs=dict(context=ctx,questions=['native captured question'],layout=CarrierLayout(256,1,256),
                    max_new_tokens=8,do_sample=False,num_beams=1,use_cache=True)
        rng_cpu,rng_cuda=torch.get_rng_state(),torch.cuda.get_rng_state()
        aa,_=native_runtime.generate(model.lm_head,[ids],carrier,**kwargs)
        torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_cuda)
        bb,_=compact_runtime.generate(model.lm_head,[ids],carrier,stable_world_state=snapshot,**kwargs)
        assert compact_runtime.composer is None and compact_runtime.last_composed_carriers is carrier
        assert torch.equal(aa[0],bb[0])
        ex_a=expand_input(base,ids,carrier,carrier_layout=CarrierLayout(256,1,256))
        ex_b=expand_input(base,ids,compact_runtime.last_composed_carriers,carrier_layout=CarrierLayout(256,1,256))
        assert all(torch.equal(getattr(ex_a,k),getattr(ex_b,k)) for k in
                   ('embeddings','position_ids','attention_mask','token_ids'))
        write(root/'native_identity_cuda.json',dict(status='PASS',frame_id=row['frame_id'],
            same_storage=True,composer_executed=False,adapter_loaded=False,identical_embeddings_positions_mask=True,
            same_rng=True,identical_generated_tokens=True,native_tokens=aa[0].cpu().tolist(),compact_tokens=bb[0].cpu().tolist()))
        native_runtime.reset();compact_runtime.reset();del native_runtime,compact_runtime,ex_a,ex_b
        coverage_report={name:{str(k):replay(root,name,k,base) for k in (8,16)} for name in ('natural','enriched')}
        coverage_report['status']='REAL_FEATURE_ACTUAL_PREFIX_VERIFIED'
        write(root/'real_memory_coverage.json',coverage_report)
        pairs=[]
        for name in ('natural','enriched'):
            source,rows,indexed,qualification=dataset(root,name)
            scenes=set()
            for e in coverage_report[name]['8']['events']:
                if e['event']=='keep' and e['scene_id'] not in scenes:
                    pairs.append((name,e,source,rows,indexed,qualification));scenes.add(e['scene_id'])
        pairs=pairs[:3]
        if len(pairs)<3:raise ValueError('Need three distinct scene pairs for bounded cost control')
        all_rows,overheads,preservation=[],[],[]
        forward_events=[];original=base.forward
        @wraps(original)
        def observed(*args,**kwargs):
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record()
            value=original(*args,**kwargs);b.record()
            forward_events.append((a,b,kwargs['inputs_embeds'].shape[1] if kwargs.get('inputs_embeds') is not None else 1))
            return value
        observed._worldstream_read_only_observer=True;base.forward=observed
        def timed(fn):
            torch.cuda.synchronize();a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            tick=perf_counter();a.record();value=fn();b.record();b.synchronize()
            return value,dict(wall_ms=(perf_counter()-tick)*1000,cuda_stream_ms=a.elapsed_time(b))
        planner=ReusePlanner(Scheduler(M3Config(reuse_strategy='exact_prefix')))
        for pi,(name,event,source,rows,indexed,qualification) in enumerate(pairs):
            arow=next(r for r in rows if r['frame_id']==event['generating_frame'])
            brow=next(r for r in rows if r['frame_id']==event['frame_id'])
            acap=load_capture(carrier_path(source,arow['frame_id']));bcap=load_capture(carrier_path(source,brow['frame_id']))
            aa=acap['carriers'].cuda();bb=bcap['carriers'].cuda()
            for k in ((8,16) if name=='enriched' and pi==1 else (8,)):
                manager=VisualMemoryManager(k);composer=CompactVisualComposer(4096,k).cuda().eval().requires_grad_(False)
                ars=[replace(r,features=r.features.cuda()) for r in regions(indexed[arow['frame_id']])]
                brs=[replace(r,features=r.features.cuda()) for r in regions(indexed[brow['frame_id']])]
                (memory,_),creation=timed(lambda:manager.update(context(arow),ars,arow['qualified_ids'],qualification))
                if memory is None:continue
                (same,held),holding=timed(lambda:manager.update(context(brow),brs,brow['qualified_ids'],qualification))
                if same is not memory or held!='keep':continue
                torch.save(dict(state_dict=composer.state_dict(),spec=composer.spec,trained=False,
                    purpose='Untrained CUDA cost scaffold; deployment forbidden'),dest/('untrained_composer_'+str(k)+'.pt'))
                for rep in range(11):
                    composer.reset()
                    _,cold=timed(lambda:composer(bb,memory))
                    _,warm=timed(lambda:composer(bb,memory))
                    _,projection=timed(lambda:composer.memory_projection(memory))
                    _,compression=timed(lambda:(composer.objects(bb[:,:256]),composer.maps(bb[:,257:])))
                    if rep>=2:overheads.append(dict(pair=pi,K=k,repeat=rep-2,cold=cold,warm=warm,
                        retained_projection=projection,compression=compression,create=creation,keep=holding))
                aimages=composer(aa,memory);bimages=composer(bb,memory)
                assert torch.equal(aimages[:,:k],bimages[:,:k])
                assert composer(bb,None) is bb
                n=256-k//2
                assert torch.equal(bimages[:,k+n:k+n+1],bb[:,256:257])
                untouched=0
                for merge,native,compressed in ((composer.objects,bb[:,:256],bimages[:,k:k+n]),
                    (composer.maps,bb[:,257:],bimages[:,k+n+1:])):
                    mask=~torch.isin(merge.retained,merge.destination)
                    assert torch.equal(compressed[:,mask],native[:,merge.retained[mask]])
                    untouched+=int(mask.sum())
                preservation.append(dict(pair=pi,K=k,total_visual_tokens=513,untouched_native_slots=untouched+1,
                    merged_destinations=k,removed_slots_with_real_contribution=k,
                    dynamic_tokens=513-k,ego_exact=True,no_memory_same_storage=True,
                    semantics_quality='NOT_RUN',compression_is_lossy=True))
                signature,_=timed(lambda:execution_signature(base,context(arow)['model_sha256']))
                for qi in range(min(3,len(acap['ids']),len(bcap['ids']))):
                    def expand(cap,images,row,layout):
                        return expand_input(base,cap['ids'][qi].cuda(),images,scope=(arow['scene_id'],),
                            signature=signature,source=(arow['scene_id'],row['frame_id']),carrier_layout=layout,
                            stable_anchor=(repr((memory.scope,memory.revision)),True) if layout is composer.layout else ())
                    aex=expand(acap,aimages,arow,composer.layout)
                    bex=expand(bcap,bimages,brow,composer.layout)
                    native=expand(bcap,bb,brow,CarrierLayout(256,1,256))
                    assert native.length==bex.length
                    expansion_cost=[]
                    for er in range(5):
                        _,elapsed=timed(lambda:expand(bcap,bimages,brow,composer.layout))
                        if er:expansion_cost.append(elapsed['wall_ms'])
                    text=bex.visual_group_spans[0][1]
                    computed=base(inputs_embeds=aex.embeddings,position_ids=aex.position_ids,
                        attention_mask=aex.attention_mask,use_cache=True,return_dict=True)
                    cache=PrefixKVCache(1024**3);executor=PrefillExecutor(cache)
                    for rep in range(11):
                        order=['native','compact_zero','compact_exact','compact_text_only']
                        order=order[rep%4:]+order[:rep%4]
                        if rep%2:order.reverse()
                        for kind in order:
                            assert cache.put(aex,computed.past_key_values,computed.logits)
                            match,matching=timed(lambda:cache.match(bex))
                            assert match>=text+k
                            (decision,plan),planning=timed(lambda:planner.plan('captured native question',bex,cache,
                                dict(state_valid=True),dict(frame_index=1),overhead_ms=0.))
                            length=text+k if kind=='compact_exact' else text if kind=='compact_text_only' else 0
                            forward_events.clear()
                            if kind=='native':
                                output,measure=timed(lambda:model.lm_head.generate(inputs=bcap['ids'][qi].cuda()[None],
                                    images=bb,max_new_tokens=1,do_sample=False,num_beams=1,use_cache=True))
                                diag={}
                            else:
                                (output,diag),measure=timed(lambda:executor.generate(base,bex,reuse_length=length,
                                    capture=True,profile=True,max_new_tokens=1,do_sample=False,num_beams=1,use_cache=True))
                            expected=bex.length-length
                            assert sum(n for _,_,n in forward_events)==expected
                            if rep>=2:
                                row=dict(pair=pi,scene_id=arow['scene_id'],sample=name,K=k,question_index=qi,
                                    repeat=rep-2,round=(rep-2)//3,kind=kind,input_tokens=bex.length,
                                    matched_tokens=match,controlled_skipped_tokens=length,
                                    controlled_crossframe_visual_tokens=k if kind=='compact_exact' else 0,
                                    scheduler_approved_tokens=decision.reuse_length,scheduler_reason=decision.decision_reason,
                                    model_prefill_cuda_ms=sum(a.elapsed_time(b) for a,b,_ in forward_events),
                                    executor_wall_ms=measure['wall_ms'],matcher_wall_ms=matching['wall_ms'],
                                    planner_wall_ms=planning['wall_ms'],effective_forward_tokens=expected,
                                    expanded_preparation_ms=float(np.mean(expansion_cost)),
                                    decode_forward_steps=0,diagnostics=diag,cache_bytes=cache.nbytes)
                                all_rows.append(row)
                                with (dest/'records.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    cache.reset();del computed
                print('COMPACT_COST_PAIR',pi,'K',k,flush=True)
        base.forward=original
        write(dest/'overheads.json',overheads)
        write(root/'visual_token_preservation.json',dict(status='STRUCTURAL_PASS',records=preservation,
            no_memory_generation_equivalence='Covered by runtime regression test; real model identity checked separately',
            quality='NOT_RUN'))
        write(root/'compact_prefix_cost.json',dict(status='COMPLETED',rows=len(all_rows),
            groups={str(k):{kind:{m:stats([r[m] for r in all_rows if r['K']==k and r['kind']==kind])
                for m in ('model_prefill_cuda_ms','executor_wall_ms','matcher_wall_ms','planner_wall_ms')}
                for kind in ('native','compact_zero','compact_exact','compact_text_only')} for k in (8,16)},
            projection_trained=False,formal_service_KV_tokens=0,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),three_alternating_rounds=True,
            decode_forward_steps=0,scope='Paired prefill-only structural CUDA control; not answer quality or full service speed.'))


def assert_training_ready(root):
    decision=json.loads((root/'feasibility_decision.json').read_text())
    if decision.get('status')!='GO':
        raise ValueError('Compact training refused: real coverage-weighted cost gate is not GO')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=['prepare','coverage','cost'],required=True)
    args=parser.parse_args()
    if {4,5}&set(os.sched_getaffinity(0)):parser.error('Use taskset -c 0-3,6-31')
    dict(prepare=prepare,coverage=coverage,cost=cost)[args.stage](args.root)


if __name__=='__main__':main()
