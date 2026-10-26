"""Offline real-weight calibration. Never grants production or task approval."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .common import backbone, write
from worldstream.m2.cost_model import cost_key, SCHEMA_VERSION, block_fingerprint
from worldstream.m2.dependency_planner import (DependencyPlanner, block0_input,
    exact_pixel_changes, exact_token_changes, patch_updates)
from worldstream.m2.packing import partition
from worldstream.m2.state_cache import StateCache
from worldstream.m2.window_executor import WindowExecutor, NUMERICAL_POLICY


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(samples):
    return dict(wall_ms=float(np.median([s[0] for s in samples])),
        gpu_span_ms=float(np.median([s[1] for s in samples])),repeats=len(samples),
        wall_p95_ms=float(np.percentile([s[0] for s in samples],95)),
        gpu_span_p95_ms=float(np.percentile([s[1] for s in samples],95)),
        statistic='median',wall_samples_ms=[s[0] for s in samples],
        gpu_span_samples_ms=[s[1] for s in samples])


def paired_measure(incremental, full, repeats, warmup):
    for _ in range(warmup):
        incremental();full()
    torch.cuda.synchronize()
    samples=[[],[]];results=[None,None]
    events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(2)]
    for repeat in range(repeats):
        for index in ((0,1) if repeat%2==0 else (1,0)):
            torch.cuda.synchronize()
            a,b=events[index];tick=perf_counter();a.record()
            results[index]=(incremental,full)[index]()
            b.record();torch.cuda.synchronize()
            samples[index].append(((perf_counter()-tick)*1000,a.elapsed_time(b)))
    return results,summary(samples[0]),summary(samples[1])


def measure_call(call, repeats, warmup):
    for _ in range(warmup):call()
    samples=[];result=None
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize();tick=perf_counter();a.record()
        result=call();b.record();torch.cuda.synchronize()
        samples.append(((perf_counter()-tick)*1000,a.elapsed_time(b)))
    return result,summary(samples)


def difference(actual, expected):
    error=(actual.float()-expected.float()).abs()
    return dict(exact=torch.equal(actual,expected),max_abs=float(error.max()),
        mean_abs=float(error.mean()))


def compare_task_runs(baseline, candidate, output):
    """Actual head/answer agreement on identity-aligned runs; not official AP."""
    import pickle
    from scipy.optimize import linear_sum_assignment
    baseline,candidate=Path(baseline),Path(candidate)
    sequence=[json.loads((p/'sequence.json').read_text())['tokens'] for p in (baseline,candidate)]
    if sequence[0]!=sequence[1]:raise ValueError('Task runs must use identical ordered sample tokens')
    runs=[]
    for path in (baseline,candidate):
        with (path/'predictions.pkl').open('rb') as handle:runs.append(pickle.load(handle))
    if any(len(r)!=len(sequence[0]) for r in runs):raise ValueError('Incomplete task prediction coverage')
    rows=[]
    def array(t):return t.detach().cpu().numpy() if hasattr(t,'detach') else np.asarray(t)
    for token,a,b in zip(sequence[0],*runs):
        det=[]
        for result in (a,b):
            box=result['pts_bbox'];keep=array(box['scores_3d'])>=.3
            det.append((array(box['boxes_3d'].gravity_center)[keep],array(box['labels_3d'])[keep],array(box['scores_3d'])[keep]))
        xyz,labels,scores=det[0];other,other_labels,other_scores=det[1]
        matched=[]
        for label in set(labels.tolist()) | set(other_labels.tolist()):
            ia=np.flatnonzero(labels==label);ib=np.flatnonzero(other_labels==label)
            if not len(ia) or not len(ib):continue
            distances=np.linalg.norm(xyz[ia,None,:]-other[None,ib,:],axis=-1)
            rr,cc=linear_sum_assignment(distances)
            matched.extend((float(distances[r,c]),float(abs(scores[ia[r]]-other_scores[ib[c]]))) for r,c in zip(rr,cc))
        maps=[]
        for result in (a,b):
            lines,confidence=result['lane_results'][0]
            confidence=array(confidence)
            # Native evaluator uses each lane's actual score; preserve its shape.
            keep=(confidence.max(-1) if confidence.ndim>1 else confidence)>=.3
            maps.append(array(lines)[keep])
        lane_dist=[]
        if len(maps[0]) and len(maps[1]):
            cost=np.linalg.norm(maps[0][:,None]-maps[1][None,:],axis=-1).mean(-1)
            reverse=np.linalg.norm(maps[0][:,None]-maps[1][None,:,::-1],axis=-1).mean(-1)
            cost=np.minimum(cost,reverse);rr,cc=linear_sum_assignment(cost)
            lane_dist=cost[rr,cc].tolist()
        qa={r['Q']:r['A'][0] for r in a['text_out']};qb={r['Q']:r['A'][0] for r in b['text_out']}
        common=set(qa)&set(qb)
        rows.append(dict(frame_id=token,native_detections=len(xyz),candidate_detections=len(other),
            matched_detections=len(matched),detection_center_distance_m=[x[0] for x in matched],
            detection_score_difference=[x[1] for x in matched],
            matches_within_2m=sum(x[0]<=2 for x in matched),
            native_lanes=len(maps[0]),candidate_lanes=len(maps[1]),lane_mean_point_distance_m=lane_dist,
            compared_answers=len(common),identical_answers=sum(qa[q]==qb[q] for q in common)))
    centers=[d for r in rows for d in r['detection_center_distance_m']]
    lanes=[d for r in rows for d in r['lane_mean_point_distance_m']]
    result=dict(frames=len(rows),rows=rows,
        detection_mean_matched_center_distance_m=float(np.mean(centers)) if centers else None,
        map_mean_matched_point_distance_m=float(np.mean(lanes)) if lanes else None,
        compared_answers=sum(r['compared_answers'] for r in rows),identical_answers=sum(r['identical_answers'] for r in rows),
        native_metrics=json.loads((baseline/'metrics.json').read_text()),
        candidate_metrics=json.loads((candidate/'metrics.json').read_text()),
        scope='Actual task prediction/answer agreement on a short sequence, NOT official Detection AP, Map AP or independent quality certification',
        production_approvals_added=0,task_quality_certified=False)
    write(output,result)
    return result


def change_metrics(image, previous, current_input, previous_input, projection, plans):
    pixels=exact_pixel_changes(image,previous)
    old_tokens=patch_updates(pixels,projection)
    tokens=exact_token_changes(current_input,previous_input)
    return dict(pixel_changed_ratio=float(pixels.float().mean()),
        pixel_derived_token_update_ratio=float(old_tokens.float().mean()),
        feature_changed_token_ratio=float(tokens.float().mean()),
        token_update_ratio=float(tokens.float().mean()),
        window_update_ratios={p.block:float(p.updated.float().mean()) for p in plans})


@torch.no_grad()
def audit_cache_transition(previous, current, diagnostic, frame_id, ordinary_budget):
    """Offline checks after synchronized timing; never part of measured execution."""
    if not diagnostic['cache_saved']:
        return dict(checked=False, reason='cache_not_published')
    checks={}
    for stage in diagnostic['stages']:
        block=stage['block']
        if block not in current.outputs:continue
        ids=stage['executed_window_ids'];total=stage['total_windows']
        checks[f'block_{block}_counts']=len(set(ids))==stage['executed_windows'] and stage['executed_windows']+stage['reused_windows']==total
        checks[f'block_{block}_plan_consistency']=all(stage.get('selection_consistency',{}).values())
        checks[f'block_{block}_mandatory_executed']=set(stage.get('mandatory_window_ids',[])).issubset(ids)
        checks[f'block_{block}_source_versions']=all(0<s['computation_version']<=current.version for s in current.sources[block])
        keep=torch.ones(total,device=current.outputs[block].device,dtype=torch.bool)
        keep[ids]=False
        checks[f'block_{block}_selected_ages_zero']=bool(current.window_ages[block][~keep].eq(0).all())
        checks[f'block_{block}_selected_source_current']=all(
            current.sources[block][j]['computation_frame_id']==frame_id for j in ids)
        if previous.images is not None and diagnostic['full_update_reason'] is None:
            checks[f'block_{block}_reused_ages_increment']=torch.equal(
                current.window_ages[block][keep],previous.window_ages[block][keep]+1)
            checks[f'block_{block}_reused_outputs_unchanged']=torch.equal(
                current.outputs[block][keep],previous.outputs[block][keep])
            checks[f'block_{block}_reused_sources_unchanged']=all(
                current.sources[block][j]==previous.sources[block][j] for j in range(total) if j not in ids)
            if block==0:
                checks['reused_descriptors_unchanged']=torch.equal(current.descriptors[keep],previous.descriptors[keep])
    refresh=diagnostic['region_refresh']
    if refresh:
        actual=set(diagnostic['stages'][0]['executed_window_ids'])
        checks['mandatory_in_actual_execution']=set(refresh['mandatory_window_ids']).issubset(actual)
        per_camera=diagnostic['stages'][0]['total_windows']//len(current.context['camera_ids'])
        ordinary=refresh['ordinary_window_ids']
        checks['per_camera_ordinary_budget']=all(
            sum(c*per_camera<=j<(c+1)*per_camera for j in ordinary)<=ordinary_budget
            for c in range(len(current.context['camera_ids'])))
    if not all(checks.values()):raise AssertionError('Cache transition audit failed: '+str(checks))
    return dict(checked=True, checks=checks)


@torch.no_grad()
def run_wsgrr(args):
    """Continuous committed-cache evaluation using the production runtime/executor.

    Replay contains actual full-model predictions, never annotation identities.
    Each ablation owns both its M1 transactions and its M2 cache lineage.
    """
    import copy
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    from worldstream.m1 import WorldStateManager, M1Config, Frame, CameraFrame
    from worldstream.m1.omnidrive_adapter import quaternion_transform
    from worldstream.m1.geometry import inverse_transform
    from tools.worldstream_m1.dataset import prediction_outputs
    from tools.worldstream_m1.io_utils import read_prediction_subset
    from worldstream.m2 import IncrementalVisionRuntime
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    source_hashes={str(p):sha256(p) for directory in ('worldstream/m1','worldstream/m2','tools/worldstream_m2')
                   for p in Path(directory).glob('*.py')}
    write(out/'source_hashes.json',source_hashes)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    cfg.data.test.test_mode=True
    cfg.data.test.pipeline=[n for n in cfg.data.test.pipeline if n['type'] in
        ('LoadMultiViewImageFromFiles','ResizeCropFlipRotImage','ResizeMultiview3D','NormalizeMultiviewImage','PadMultiViewImage')]
    dataset=build_dataset(cfg.data.test)
    dataset.data_infos.sort(key=lambda i:(i['scene_token'],i['timestamp']))
    scene=args.scene or dataset.data_infos[0]['scene_token']
    if args.scene or getattr(args,'main_only',False):
        dataset.data_infos=[i for i in dataset.data_infos if i['scene_token']==scene]
    infos=dataset.data_infos[:args.frames]
    if len(infos)<2:raise ValueError('Need a continuous sequence with >=2 frames')
    if getattr(args,'main_only',False):
        from tools.worldstream_m123.benchmark.sequence import validate_sequence
        write(out/'sequence.json',dict(validate_sequence(infos,scene),requested_frames=args.frames,
            actual_frames=len(infos),reduction_reason='selected scene has fewer frames than requested' if len(infos)<args.frames else None))
    predictions=read_prediction_subset(args.m1_predictions,{i['token'] for i in infos})
    missing=[i['token'] for i in infos if i['token'] not in predictions]
    if missing:raise ValueError('Missing actual M1 replay predictions: '+str(missing))
    model=backbone(args.checkpoint)
    specs=([('D3','D',3)] if getattr(args,'main_only',False) else
           [('D3','D',3),('NoM1Region','NoM1Region',3)] if getattr(args,'followup',False) else
           [('A','A',3),('B','B',3),('C','C',3)]+[('D'+str(b),'D',b) for b in args.budgets])
    intervals=getattr(args,'anchor_intervals',None)
    if intervals:specs=[('AR'+str(age),'D',3) for age in intervals]
    runtimes={name:IncrementalVisionRuntime(model,{**dict(cfg.model.m2_config), 'ordinary_budget':b,
        **({'global_anchor_interval':int(name[2:])} if intervals else {}),
        **({'global_drift_threshold':args.global_drift_threshold} if args.global_drift_threshold is not None else {})}) for name,_,b in specs}
    managers={name:WorldStateManager(M1Config.load('configs/worldstream_m1_final.yaml')) for name,_,_ in specs}
    reference=None
    if getattr(args,'reference_runtime',None):
        import importlib.util,sys
        path=Path(args.reference_runtime).resolve()
        spec=importlib.util.spec_from_file_location('worldstream._m2_reference',path/'__init__.py',submodule_search_locations=[str(path)])
        package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
        reference=package.IncrementalVisionRuntime(model,{k:v for k,v in dict(cfg.model.m2_config).items() if not k.startswith('global_')})
        reference_cache_type=type(reference.cache)
    rows=[];native_rows=[];cost_records=[];previous=None;old_info=None
    old_export=None
    if getattr(args,'reference_state_manager',None):
        import ast
        tree=ast.parse(Path(args.reference_state_manager).read_text())
        owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WorldStateManager')
        method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='export_m2_inputs')
        namespace={'copy':copy,'__package__':'worldstream.m1'}
        exec(compile(ast.Module(body=[method],type_ignores=[]),args.reference_state_manager,'exec'),namespace)
        old_export=namespace['export_m2_inputs']
    # The model kernels are warmed before any reported frames.
    for index,info in enumerate(infos):
        sample=dataset[index]
        image=torch.from_numpy(np.stack(sample['img']).transpose(0,3,1,2).copy()).cuda()
        x0=block0_input(model,image)
        if reference is None:
            native_output,dense=measure_call(lambda:model(image)[0],args.repeats,args.warmup)
        else:native_output,dense=None,None
        norm=sample['img_norm_cfg'];cameras={}
        for c,(cid,cam) in enumerate(info['cams'].items()):
            rgb=sample['img'][c]*np.asarray(norm['std'])+np.asarray(norm['mean'])
            if not norm.get('to_rgb',False):rgb=rgb[...,::-1]
            c2w=quaternion_transform(cam['ego2global_rotation'],cam['ego2global_translation']) @ quaternion_transform(cam['sensor2ego_rotation'],cam['sensor2ego_translation'])
            raw_k=np.asarray(cam['cam_intrinsic']);k=sample['intrinsics'][c][:3,:3]
            cameras[cid]=CameraFrame(cid,np.clip(rgb,0,255).astype(np.uint8),k,inverse_transform(c2w),
                cam['timestamp']/1e6,'preprocessed_image',k@np.linalg.inv(raw_k),True,raw_intrinsic=raw_k)
        frame=Frame(info['scene_token'],info['token'],info['timestamp']/1e6,
            quaternion_transform(info['ego2global_rotation'],info['ego2global_translation']),cameras,'wsgrr_offline_real_v1')
        continuous=old_info is not None and old_info['scene_token']==info['scene_token']
        exact_ratio=float(exact_token_changes(x0,previous).float().mean()) if continuous else 1.
        for name,variant,budget in (specs if index%2==0 else list(reversed(specs))):
            rt=runtimes[name];manager=managers[name]
            if not continuous:rt.reset()
            precheck_profile=None
            if getattr(args,'followup_profile',False) and name=='D3' and index==1:
                import cProfile,io,pstats
                precheck_profile=cProfile.Profile();precheck_profile.enable()
            tick=perf_counter();result=manager.precheck(frame);precheck_ms=(perf_counter()-tick)*1000
            if precheck_profile is not None:
                precheck_profile.disable();precheck_profile.dump_stats(str(out/'m1_precheck_cpu.prof'))
                stream=io.StringIO();pstats.Stats(precheck_profile,stream=stream).sort_stats('cumtime').print_stats(60)
                (out/'m1_precheck_cpu.txt').write_text(stream.getvalue())
            context=dict(scene_id=frame.scene_id,frame_id=frame.frame_id,timestamp=frame.timestamp,
                execution_signature=frame.execution_signature,state_epoch=result.diagnostics['state_epoch'],
                coordinate_space='preprocessed_image',camera_ids=list(cameras))
            tick=perf_counter();snapshot=manager.export_m2_inputs(expected_context=context,computation_only=True)
            read_ms=(perf_counter()-tick)*1000
            export_pair=None
            if old_export is not None:
                samples=[[],[]]
                for repeat in range(3):
                    for which in ((0,1) if repeat%2==0 else (1,0)):
                        tick=perf_counter()
                        exported=(manager.export_m2_inputs(context,computation_only=True) if which==0 else
                                  old_export(manager,context,computation_only=True))
                        samples[which].append((perf_counter()-tick)*1000)
                old_snapshot=old_export(manager,context,computation_only=True)
                same_contract=all(getattr(snapshot.computation_state,k)==getattr(old_snapshot.computation_state,k)
                    for k in ('source_records','refresh_contract','region_states','region_events','uncertainty_regions','observation_links','diagnostics'))
                if not same_contract:raise AssertionError('Snapshot optimization changed computation contract')
                export_pair=dict(optimized_wall_ms=samples[0],before_wall_ms=samples[1],same_pending_request=True,
                                 computation_contract_equal=True)
                if index==1 and name==specs[0][0]:
                    import cProfile,io,pstats
                    for label,call in [('optimized',lambda:manager.export_m2_inputs(context,computation_only=True)),
                                       ('before',lambda:old_export(manager,context,computation_only=True))]:
                        profile=cProfile.Profile();profile.runcall(call)
                        profile.dump_stats(str(out/('snapshot_'+label+'.prof')))
                        stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumtime').print_stats(40)
                        (out/('snapshot_'+label+'.txt')).write_text(stream.getvalue())
            if getattr(args,'followup_profile',False) and name=='D3' and index==1:
                import cProfile,io,pstats
                profiler=cProfile.Profile();profiler.enable()
                manager.export_m2_inputs(expected_context=context,computation_only=True)
                profiler.disable();profiler.dump_stats(str(out/'m1_export_cpu.prof'))
                stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(50)
                (out/'m1_export_cpu.txt').write_text(stream.getvalue())
            source_cache=rt.cache
            block_inputs={}
            execute=rt.executor.execute
            if args.profile_block_costs:
                def observe(block,x,cached,ids):
                    block_inputs[id(block)]=(x,cached,ids)
                    return execute(block,x,cached,ids)
                rt.executor.execute=observe
            def candidate():
                # commit replaces owned fields atomically; the source snapshot is
                # immutable, so repeats see identical history without GPU clones.
                rt.cache=copy.copy(source_cache)
                if getattr(args,'snapshot_boundary',False):
                    return rt.forward(image,m1_result=snapshot,execution_context=context)
                if getattr(args,'formal',False):
                    return rt.forward_from_m1(image,manager,context)
                return rt.evaluate_offline(image,variant=variant,m1_result=snapshot,execution_context=context)
            if reference is None:
                measured,timing=measure_call(candidate,args.repeats,args.warmup)
            else:
                (measured,native_output),timing,dense=paired_measure(candidate,lambda:model(image)[0],args.repeats,args.warmup)
            native_rows.append(dict(ordinal=index,frame_id=info['token'],scene_id=info['scene_token'],timing=dense))
            rt.executor.execute=execute
            output,diag=measured
            next_cache=rt.cache
            reference_pair=None
            if reference is not None:
                # Namespace adaptation belongs outside BOTH latency samples.
                # Timed calls only receive shallow views of the same history.
                reference_source=reference_cache_type(source_cache.budget_bytes)
                reference_source.__dict__.update(source_cache.__dict__)
                if source_cache.context is not None:
                    c=reference_source
                    c.signature=reference.signature(context)
                    c.metadata_revision=(repr((c.context,c.sources,c.input_stride,c.output_stride,c.version,c.region_states)),c.signature)
                def prior():
                    reference.cache=copy.copy(reference_source)
                    if getattr(args,'snapshot_boundary',False):
                        return reference.forward(image,m1_result=snapshot,execution_context=context)
                    return reference.forward_from_m1(image,manager,context)
                (optimized,prior_result),opt_cost,prior_cost=paired_measure(candidate,prior,args.repeats,args.warmup)
                before,before_diag=prior_result
                same_ids=all(a['executed_window_ids']==b['executed_window_ids'] and a['planned_window_ids']==b['planned_window_ids']
                    for a,b in zip(diag['stages'][:len(rt.prefix)],before_diag['stages'][:len(rt.prefix)]))
                same_outputs=all(torch.equal(next_cache.outputs[i],reference.cache.outputs[i]) for i in rt.prefix)
                same_ages=all(torch.equal(next_cache.window_ages[i],reference.cache.window_ages[i]) for i in rt.prefix)
                delta_before=difference(output[0],before[0])
                if not (same_ids and same_outputs and same_ages
                    and next_cache.version==reference.cache.version and next_cache.sources==reference.cache.sources):
                    raise AssertionError('Optimization changed window plan/cache/output: '+str(delta_before))
                reference_pair=dict(optimized=opt_cost,before=prior_cost,window_ids_equal=same_ids,
                    packed_outputs_equal=same_outputs,ages_equal=same_ages,features=delta_before,
                    cache_versions_equal=True,provenance_equal=True)
                rt.cache=next_cache
            if getattr(args,'followup_profile',False) and name==specs[0][0] and index==getattr(args,'profile_frame',1):
                from .profiling import profile_call
                rt.cache=source_cache
                # Profiling mutates this shallow copy only; selected history and
                # the measured publication remain exactly the ones above.
                profile_call(rt,candidate,out/'profile_output')
                from .profiling import measure_publication
                measure_publication(rt,candidate,out/'synchronized_actual_publication.json')
                if reference is not None:
                    profile_call(reference,prior,out/'reference_profile_output')
                    measure_publication(reference,prior,out/'reference_synchronized_actual_publication.json')
                rt.cache=next_cache
            delta=difference(output[0],native_output)
            outputs=prediction_outputs(predictions[info['token']],manager.config)
            outputs.execution_receipt=diag
            manager.commit(frame,outputs,result)
            rt.cache=next_cache
            selected=diag['stages'][0]['executed_window_ids']
            selected_mask=torch.zeros(len(diag['output_window_provenance'][0]),device=x0.device,dtype=torch.bool)
            selected_mask[selected]=True
            token_ratio=float(rt.planner.windows_to_tokens(selected_mask,x0.shape[:3],model.blocks[0].window_size).float().mean())
            row=dict(case=name+'_'+str(index),strategy=name,variant=variant,budget=budget,ordinal=index,frame_id=frame.frame_id,scene_id=frame.scene_id,
                timestamp=frame.timestamp,continuous=continuous,exact_token_update_ratio=exact_ratio,
                selected_token_update_ratio=token_ratio,stages=diag['stages'],region_refresh=diag['region_refresh'],
                full_update_reason=diag['full_update_reason'],difference=delta,
                actual_skipped_windows=diag['actual_skipped_windows'],actual_window_reuse=diag['actual_window_reuse'],
                full_recompute=diag['full_recompute'],full_recompute_reason=diag['full_recompute_reason'],
                cache_version=diag['cache_version'],
                cache_mismatch_reason=diag['cache_mismatch_reason'],
                incremental=timing,native=dense,reference_pair=reference_pair,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                M1_precheck_ms=precheck_ms,M1_state_read_ms=read_ms,
                M1_precheck_components_ms=result.diagnostics.get('timings_ms',{}),
                M1_precheck_profiled=precheck_profile is not None,
                region_scoring_host_ms=diag['M2_region_scoring_ms'],window_and_suffix_host_ms=diag['M2_window_and_suffix_ms'],
                cache_publication_host_ms=diag['M2_cache_publication_ms'],cache_saved=diag['cache_saved'],
                m1_regions=len(snapshot.computation_state.region_states),
                region_states=snapshot.computation_state.region_states,
                real_m1_expiry_events=snapshot.computation_state.region_events,
                image_uncertainty_regions=snapshot.computation_state.uncertainty_regions,
                window_ages_after={i:t.cpu().tolist() for i,t in next_cache.window_ages.items()},
                output_window_provenance=diag['output_window_provenance'],
                cache_transition_audit=audit_cache_transition(source_cache,next_cache,diag,frame.frame_id,budget),
                m1_replay_source='historical_full_model_detection_replay',
                downstream_quality='NOT_MEASURED: replay supplies states, not candidate task predictions')
            row['global_state']={k:v for k,v in diag.items() if k.startswith(('global_','approx_global_'))}
            row['cache_bytes']=diag['cache_bytes']
            row['snapshot_export_pair']=export_pair
            rows.append(row)
            if args.profile_block_costs and variant=='D' and continuous:
                publication_cache=StateCache(rt.cache.budget_bytes)
                publication_cache.version=next_cache.version
                def publish():
                    publication_cache.commit(image,next_cache.outputs,context,next_cache.signature,
                        output[0].stride(),next_cache.sources,block0_input=x0,
                        region_states=next_cache.region_states,window_ages=next_cache.window_ages,
                        descriptors=next_cache.descriptors)
                _,row['synchronized_cache_publication']=measure_call(publish,args.repeats,args.warmup)
                for i in rt.prefix:
                    block=model.blocks[i];x,cached,ids=block_inputs[id(block)]
                    (partial,full),a,b=paired_measure(lambda:execute(block,x,cached,ids),lambda:block(x),args.repeats,args.warmup)
                    local_delta=difference(partial[0],full)
                    cost_records.append(dict(key=cost_key(block,x,i,model=model,refresh_policy=rt.config.refresh_signature()),
                        block=i,updated=len(ids),numerical_verified=all(np.isfinite(v) for k,v in local_delta.items() if k!='exact'),
                        numerical_validation_scope='Finite local error only; independent quantitative limits and downstream quality still required',
                        block_incremental=a,block_full=b,
                        difference=local_delta,quality_certified=False,quality_rejection_reason='downstream_quality_unverified',
                        incremental_gpu_ms=a['gpu_span_ms'],full_gpu_ms=b['gpu_span_ms'],
                        incremental_wall_ms=a['wall_ms'],full_wall_ms=b['wall_ms'],
                        pipeline_incremental_wall_ms=timing['wall_ms'],pipeline_full_wall_ms=dense['wall_ms'],
                        pipeline_incremental_gpu_ms=timing['gpu_span_ms'],pipeline_full_gpu_ms=dense['gpu_span_ms'],
                        evidence_cases=[row['case']],numerical_evidence=dict(path='results.json')))
            with (out/'frames.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
            print(index,name,diag['recomputed_windows'],delta,'wall',round(timing['wall_ms'],2),'native',round(dense['wall_ms'],2),flush=True)
        previous=x0;old_info=info
    aggregate={}
    for name,_,budget in specs:
        data=[r for r in rows if r['strategy']==name and r['continuous']]
        stages=[s for r in data for s in r['stages'] if 'total_windows' in s]
        updated=sum(s['executed_windows'] for s in stages);total=sum(s['total_windows'] for s in stages)
        eligible=[r for r in data if r['full_update_reason'] is None and any(
            s.get('selector_planned_windows',s.get('planned_windows',s.get('total_windows',0)))<s.get('total_windows',0)
            for s in r['stages'] if 'total_windows' in s)]
        aggregate[name]=dict(frames=len(data),window_update_ratio=updated/total,reuse_ratio=1-updated/total,
            eligible_partial_frames=len(eligible),eligible_reuse_frames=sum(r['actual_window_reuse'] for r in eligible),
            eligible_reuse_frame_rate=sum(r['actual_window_reuse'] for r in eligible)/len(eligible) if eligible else None,
            reuse_frames=sum(r['actual_window_reuse'] for r in data),
            full_recompute_frames=sum(r['full_recompute'] for r in data),
            exact_token_update_ratio=float(np.mean([r['exact_token_update_ratio'] for r in data])),
            selected_token_update_ratio=float(np.mean([r['selected_token_update_ratio'] for r in data])),
            wall_ms=float(np.median([r['incremental']['wall_ms'] for r in data])),
            gpu_span_ms=float(np.median([r['incremental']['gpu_span_ms'] for r in data])),
            native_wall_ms=float(np.median([r['native']['wall_ms'] for r in data])),
            native_gpu_span_ms=float(np.median([r['native']['gpu_span_ms'] for r in data])),
            mean_abs=float(np.mean([r['difference']['mean_abs'] for r in data])),
            max_abs=max(r['difference']['max_abs'] for r in data),
            event_regions=sum(len(r['region_refresh'].get('event_regions',[])) for r in data))
    result=dict(aggregate=aggregate,frames=len(infos),sample_tokens=[i['token'] for i in infos],
        requested_frames=args.frames,formal_config='projects/configs/OmniDrive/worldstream_m123_auto.py',
        execution_entry='forward(snapshot)' if getattr(args,'snapshot_boundary',False) else 'forward_from_m1' if getattr(args,'formal',False) else 'offline_validation',
        eligibility_definition='Continuous frame with valid M1/cache admission, no full refresh, selector plans fewer than all windows in at least one prefix block',
        timestamp_span_s=(infos[-1]['timestamp']-infos[0]['timestamp'])/1e6,
        checkpoint=args.checkpoint,backbone_fingerprint=block_fingerprint(model),numerical_policy=NUMERICAL_POLICY,
        validation_kind='wsgrr_quantitative_v1',natural=rows,native_measurements=native_rows,
        controlled_microbenchmarks=[],m1_predictions=args.m1_predictions,
        m1_prediction_sha256=sha256(args.m1_predictions),production_approvals_added=0,production_ready=False,
        downstream_quality='PENDING independent Detection, Map, full VLM evaluation',
        coverage=dict(classes=sorted({str(n) for i in infos for n in i.get('gt_names',[])}),
            confirmed_disappearance='UNAVAILABLE: M1 absence is not confirmation',
            traffic_signal_state='UNAVAILABLE: no M1 signal-state producer',
            visibility='M1 RGB support and occlusion hypotheses, not independent ground truth'),
        timing_scope='Embedding, M1 admission, scoring, prefix + full suffix, atomic publication, diagnostics; excludes M1 precheck/read (separate) and downstream tasks',
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            cpu_affinity=sorted(__import__('os').sched_getaffinity(0))))
    write(out/'results.json',result)
    if args.profile_block_costs:
        groups={}
        for r in cost_records:groups.setdefault((r['key'],r['updated']),[]).append(r)
        records=[]
        for group in groups.values():
            record=dict(max(group,key=lambda r:r['incremental_wall_ms']/max(r['full_wall_ms'],1e-9)))
            record.update(measurements=group,evidence_cases=[c for r in group for c in r['evidence_cases']],
                numerical_verified=all(r['numerical_verified'] for r in group),
                numerical_evidence=dict(path='results.json',sha256=sha256(out/'results.json')))
            records.append(record)
        write(out/'window_costs_provisional.json',dict(schema_version=SCHEMA_VERSION,records=records,
            validation_sha256=sha256(out/'results.json'),backbone_fingerprint=block_fingerprint(model),
            formal_quality_policy=dict(approved_counts=[]),production_ready=False))


@torch.no_grad()
def run(args):
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py');cfg.data.test.test_mode=True
    cfg.data.test.pipeline=[node for node in cfg.data.test.pipeline if node['type'] in
        ('LoadMultiViewImageFromFiles','ResizeCropFlipRotImage','ResizeMultiview3D','NormalizeMultiviewImage','PadMultiViewImage')]
    dataset=build_dataset(cfg.data.test);dataset.data_infos.sort(key=lambda i:(i['scene_token'],i['timestamp']))
    model=backbone(args.checkpoint);planner=DependencyPlanner(model);prefix=planner.cacheable_prefix()
    if planner.rejection_reason or not prefix:raise ValueError('No verified cacheable prefix')
    executor=WindowExecutor();records=[]
    def native(image):
        cache={};handles=[]
        for i in prefix:
            handles.append(model.blocks[i].register_forward_hook(lambda block,inputs,output,i=i:
                cache.update({i:partition(output,block.window_size)[0].detach().clone()})))
        try:output=model(image)[0]
        finally:
            for h in handles:h.remove()
        return output,cache
    def execute(x,plans,cache,retain=False):
        inputs={};outputs={};stages=[]
        for i,block in enumerate(model.blocks):
            if i in prefix:
                ids=plans[i].updated.nonzero().flatten();total=plans[i].updated.numel()
                if retain:inputs[i]=x
                x,outputs[i]=executor.execute(block,x,cache[i],ids)
                stages.append(dict(block=i,total_windows=total,updated_windows=len(ids),
                    executed_attention_windows=len(ids),reused_output_windows=total-len(ids),
                    pointwise_window_work=total if len(ids) else 0,
                    pointwise_tokens=len(x)*x.shape[1]*x.shape[2] if len(ids) else 0))
            else:x=block(x)
        return x.permute(0,3,1,2),stages,inputs,outputs
    def pipeline(image,source_input,cache,retain=False):
        x=block0_input(model,image)
        plans=planner.plan(exact_token_changes(x,source_input),stop_after=len(prefix))
        return execute(x,plans,cache,retain)
    def evaluate(image,previous,source_input,cache,name,kind,frame_id):
        x0=block0_input(model,image)
        plans=planner.plan(exact_token_changes(x0,source_input),stop_after=len(prefix))
        actual,stages,inputs,outputs=execute(x0,plans,cache,True)
        reference=model(image)[0];delta=difference(actual,reference)
        _,sparse,dense=paired_measure(lambda:pipeline(image,source_input,cache),
            lambda:model(image)[0],args.repeats,args.warmup)
        # Publication is separately synchronized and timed, never hidden in pure
        # block costs. This excludes M1, loading and downstream tasks explicitly.
        state=StateCache(4*1024**3)
        context=dict(frame_id=frame_id,camera_ids=[str(i) for i in range(len(image))],timestamp=1.)
        def publish():
            state.commit(image,outputs,context,'offline',block0_input=x0)
        _,publication,_=paired_measure(publish,publish,args.repeats,args.warmup)
        row=dict(case=name,kind=kind,frame_id=frame_id,difference=delta,stages=stages,
            actual_reused_windows=sum(s['reused_output_windows'] for s in stages),
            changes=change_metrics(image,previous,x0,source_input,model.patch_embed.proj,plans),
            incremental=sparse,native=dense,cache_publication=publication,cache_bytes=state.bytes,
            scope='Embedding + exact feature comparison + closure + executor + full suffix; publication separately measured; excludes M1/loading/tasks')
        for i in prefix:
            x=inputs[i];ids=plans[i].updated.nonzero().flatten();block=model.blocks[i]
            (selected,full),a,b=paired_measure(lambda:executor.execute(block,x,cache[i],ids),
                lambda:block(x),args.repeats,args.warmup)
            block_delta=difference(selected[0],full)
            records.append(dict(key=cost_key(block,x,i,model=model),block=i,updated=len(ids),
                total_windows=plans[i].updated.numel(),reused_output_windows=plans[i].updated.numel()-len(ids),
                executed_attention_windows=len(ids),pointwise_window_work=plans[i].updated.numel() if len(ids) else 0,
                pointwise_tokens=len(x)*x.shape[1]*x.shape[2] if len(ids) else 0,
                numerical_verified=delta['exact'] and block_delta['exact'],difference=block_delta,
                quality_certified=False,quality_rejection_reason='downstream_quality_unverified',
                incremental_gpu_ms=a['gpu_span_ms'],full_gpu_ms=b['gpu_span_ms'],
                incremental_wall_ms=a['wall_ms'],full_wall_ms=b['wall_ms'],
                incremental_samples=a,full_samples=b,evidence_cases=[name],
                numerical_evidence=dict(path='results.json',case=name,kind=kind)))
        print(name,delta,row['changes'],sparse['wall_ms'],dense['wall_ms'],flush=True)
        return row
    natural=[];previous=None;source_input=None;cache=None;first=None;first_cache=None;first_input=None;old_info=None
    for index in range(min(args.frames,len(dataset))):
        sample=dataset[index];info=dataset.data_infos[index]
        image=torch.from_numpy(np.stack(sample['img']).transpose(0,3,1,2).copy()).cuda()
        reference,next_cache=native(image)
        if previous is not None and info['scene_token']==old_info['scene_token'] and info['timestamp']>old_info['timestamp']:
            natural.append(evaluate(image,previous,source_input,cache,'natural_'+str(index),
                'natural_continuous_visual_operator_validation',info['token']))
        else:
            natural.append(dict(frame_id=info['token'],case='cold_or_scene_boundary',compared=False,
                reason='No ordered history in this scene',actual_reused_windows=0))
        previous=image;source_input=block0_input(model,image);cache=next_cache;old_info=info
        if first is None:first=image.clone();first_cache=next_cache;first_input=source_input.clone()
    cases=[('identical',[]),('one_window',[(0,0,0)]),
        ('padded_edge_window',[(0,first.shape[-2]-1,first.shape[-1]-1)]),
        ('one_window_per_camera',[(c,1,1) for c in range(len(first))])]
    # Sweep real image regions inside each input footprint, across cameras and
    # padding boundaries. Dependency closure determines actual counts per block.
    footprints=planner.input_footprints(*first.shape[-2:])[0]
    total=len(first)*len(footprints)
    counts=args.counts if args.counts is not None else [2,4,8,total//2,total-1,total]
    for count in counts:
        if not 0<=count<=total:raise ValueError('Requested count exceeds actual windows')
        points=[]
        for unit in range(count):
            x0,y0,x1,y1=footprints[unit%len(footprints)]['input_region']
            points.append((unit//len(footprints),(y0+y1-1)//2,(x0+x1-1)//2))
        cases.append(('window_scale_'+str(count),points))
    controlled=[]
    for name,points in cases:
        image=first.clone()
        for camera,y,x in points:image[camera,:,y,x]+=.125
        controlled.append(evaluate(image,first,first_input,first_cache,name,
            'controlled_visual_operator_validation','controlled_'+name))
    result=dict(checkpoint=str(Path(args.checkpoint).resolve()),backbone_fingerprint=block_fingerprint(model),
        numerical_policy=NUMERICAL_POLICY,cacheable_blocks=prefix,natural=natural,
        controlled_microbenchmarks=controlled,warmup=args.warmup,repeats=args.repeats,
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),cpu_affinity=sorted(__import__('os').sched_getaffinity(0))),
        production_approvals_added=0,full_end_to_end_metrics='NOT_MEASURED: no paired full downstream service evaluation',
        downstream_quality='PENDING independent Detection, Map and full VLM assessment',
        pointwise_execution='Native full shapes for any attention updates; zero-update blocks skip all work')
    write(out/'results.json',result)
    # All evidence remains available; select the conservative median-cost case
    # per identity/count, and require exactness in every covered case.
    groups={}
    for record in records:groups.setdefault((record['key'],record['updated']),[]).append(record)
    unique=[]
    for group in groups.values():
        row=dict(max(group,key=lambda r:max(r['incremental_gpu_ms']/max(r['full_gpu_ms'],1e-9),
            r['incremental_wall_ms']/max(r['full_wall_ms'],1e-9))))
        row['numerical_verified']=all(r['numerical_verified'] for r in group)
        row['evidence_cases']=[case for r in group for case in r['evidence_cases']]
        row['measurements']=group
        row['numerical_evidence']['sha256']=sha256(out/'results.json')
        unique.append(row)
    write(out/'window_costs_provisional.json',dict(schema_version=SCHEMA_VERSION,records=unique,
        formal_quality_policy=dict(approved_counts=[]),production_ready=False,
        validation_sha256=sha256(out/'results.json'),backbone_fingerprint=block_fingerprint(model)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint',default='ckpts/iter_10548.pth')
    parser.add_argument('--frames',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--counts',type=int,nargs='+')
    parser.add_argument('--wsgrr',action='store_true',help='Explicit offline A/B/C/D experiment; never promotes costs')
    parser.add_argument('--budgets',type=int,nargs='+',default=[2,3,4,6,9])
    parser.add_argument('--scene')
    parser.add_argument('--m1-predictions',help='Actual full-model results_nusc.json for historical M1 commit replay')
    parser.add_argument('--profile-block-costs',action='store_true',help='Retain provisional per-block costs for independent assessment')
    parser.add_argument('--compare-task-runs',nargs=2,metavar=('BASELINE','CANDIDATE'),help='Compare completed full-model task runs; --output is a JSON file')
    parser.add_argument('--followup-profile',action='store_true',help='Offline torch.profiler/cProfile attribution on frame 1, D3')
    parser.add_argument('--followup',action='store_true',help='Offline paired D3/NoM1Region nominal-budget experiment (same source safety)')
    parser.add_argument('--main-only',action='store_true',help='Only current WSGRR D3 and paired Native timing; no ablations or budget scan')
    parser.add_argument('--snapshot-boundary',action='store_true',help='Measure identical already-exported M1 snapshots, excluding snapshot export from both visual conditions')
    parser.add_argument('--reference-runtime',help='Frozen prior M2 package for paired before/after measurements; never used by serving')
    parser.add_argument('--anchor-intervals',type=int,nargs='+',help='Offline calibration of the same anchor policy')
    parser.add_argument('--global-drift-threshold',type=float,help='Offline calibration parameter')
    parser.add_argument('--reference-state-manager',help='Frozen prior M1 source for same-request export measurement only')
    parser.add_argument('--profile-frame',type=int,default=1,help='Offline profiler frame ordinal; excluded from timings')
    parser.add_argument('--formal',action='store_true',help='Use serving forward_from_m1 with live M1 snapshots; requires --main-only')
    args=parser.parse_args()
    if args.formal and not args.main_only:parser.error('--formal requires --main-only')
    if args.main_only and (not args.wsgrr or args.followup):parser.error('--main-only requires --wsgrr and excludes --followup')
    if args.frames<2 or args.repeats<3 or args.warmup<1:parser.error('Need frames>=2, repeats>=3, warmup>=1')
    if args.compare_task_runs:
        compare_task_runs(*args.compare_task_runs,args.output)
    else:
        if args.wsgrr and not args.m1_predictions:parser.error('--wsgrr requires actual --m1-predictions')
        (run_wsgrr if args.wsgrr else run)(args)
