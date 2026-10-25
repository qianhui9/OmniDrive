"""One isolated condition; native model calls, unchanged sampling and admission."""
import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import pickle
import random
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .common import diagnostic_json, sha256, write
from .common import ground_truth, typed_digest
from worldstream.m3.task_conditioner import task_type
from .metrics import answer_quality, distribution, quality_summary, validate_sequence, finite_metrics
from .preflight import BASELINE, FINAL


def append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(diagnostic_json(value), ensure_ascii=False, allow_nan=False)+'\n')


def cpu_result(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if hasattr(value, 'tensor') and hasattr(value, 'to'):
        return value.to('cpu')
    if isinstance(value, dict):
        return {k: cpu_result(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_result(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_result(v) for v in value)
    return value


def effective_config(condition, workload, dest):
    from mmcv import Config
    cfg = Config.fromfile(BASELINE if condition == 'B0' else FINAL)
    common = Config.fromfile(FINAL)
    cfg.data.test.pipeline = copy.deepcopy(common.data.test.pipeline)
    if condition in ('B1', 'B2'):
        # Existing offline ablations intentionally remove modules; only B3 is formal.
        cfg.model.worldstream_require_m123 = False
        cfg.model.m3_enable = False
        if condition == 'B1':
            cfg.model.m2_enable = False
    for node in cfg.data.test.pipeline:
        if node['type'] == 'LoadAnnoatationVQATest':
            node['load_type'] = ['conv'] if workload == 'language' else ['planning']
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.save_path = str(dest/'answers')+'/'
    cfg.data.test.test_mode = True
    cfg.data.workers_per_gpu = 0
    write(dest/'effective_config.json', dict(config=cfg._cfg_dict.to_dict(),
        note='Temporary evaluation overrides only: output path, shared metadata pipeline, worker count and language workload; official source configs unchanged.'))
    return cfg


def load_model(cfg, checkpoint, dest):
    from mmdet3d.models import build_model
    from mmcv.runner import wrap_fp16_model
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    state = checkpoint_data.get('state_dict', checkpoint_data)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    expected = model.state_dict()
    missing = sorted(set(expected)-set(state))
    unexpected = sorted(set(state)-set(expected))
    derived_buffers = []
    for key in unexpected:
        parent, _, leaf = key.rpartition('.')
        try:
            owner = model.get_submodule(parent)
        except AttributeError:
            continue
        # The deployed transformers version registers regenerated RoPE caches
        # as nonpersistent. Match mmcv's native load behavior only for those
        # actual registered buffers; never waive a learned parameter mismatch.
        if (leaf in ('cos_cached', 'sin_cached') and leaf in owner._buffers
                and leaf in owner._non_persistent_buffers_set):
            derived_buffers.append(key)
    unexpected = sorted(set(unexpected)-set(derived_buffers))
    shapes = {k: dict(expected=list(expected[k].shape), actual=list(state[k].shape))
              for k in set(expected) & set(state) if expected[k].shape != state[k].shape}
    validation = dict(status='PASS' if not (missing or unexpected or shapes) else 'FAIL',
        checkpoint=checkpoint, expected_keys=len(expected), actual_keys=len(state),
        missing_keys=missing, unexpected_keys=unexpected, shape_mismatches=shapes,
        ignored_nonpersistent_rotary_caches=derived_buffers,
        random_initialization_accepted=False,
        metadata=diagnostic_json(checkpoint_data.get('meta', {})))
    # Large training metadata may contain Config objects; retain a readable repr.
    validation['metadata'] = repr(checkpoint_data.get('meta', {}))
    write(dest/'checkpoint_validation.json', validation)
    if validation['status'] != 'PASS':
        raise RuntimeError('Strict checkpoint compatibility failed; see checkpoint_validation.json')
    model.load_state_dict({k: v for k, v in state.items() if k not in derived_buffers}, strict=True)
    del expected, state, checkpoint_data
    gc.collect()
    return model.cuda().eval()


def reset(model):
    for head in (model.pts_bbox_head, model.map_head):
        head.reset_memory()
    for runtime in (model.m1_manager, model.m2_runtime, model.m3_runtime):
        if runtime is not None:
            runtime.reset()


def map_evaluate(dataset, outputs, dest, workers=0):
    """Call the existing OpenLane evaluator, whose values dataset.evaluate drops."""
    from openlanev2.centerline.evaluation import evaluate
    predictions = dict(method='OmniDrive', authors=[], **{'e-mail': '', 'institution / company': '',
                                                       'country / region': 'China'}, results={})
    for info, result in zip(dataset.data_infos, outputs):
        lanes, scores = result['lane_results'][0]
        pred = dict(lane_centerline=[dict(id=i, points=np.asarray(lane, dtype=np.float32),
                                         confidence=float(score)) for i, (lane, score) in enumerate(zip(lanes, scores))],
                    traffic_element=[], topology_lclc=np.zeros((300, 300)), topology_lcte=np.zeros((300, 0)))
        predictions['results'][info['lane_info']] = dict(predictions=pred)
    if workers:
        from tools.worldstream_m2.map_eval import evaluate_map
        value=evaluate_map(dataset.lane_anno_file,predictions,workers)
    else:value = evaluate(ground_truth=dataset.lane_anno_file, predictions=predictions)
    value, unavailable = finite_metrics(value)
    write(dest/'map_metric_unavailability.json', dict(paths=unavailable, reason='Official evaluator returned a nonfinite value'))
    write(dest/'map_metrics.json', value)
    return value


def run(args):
    from mmcv.parallel import collate, scatter
    from mmdet3d.datasets import build_dataset
    from .observe import Observer, IdentityScorer, m1_record
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    write(dest/'status.json', dict(status='RUNNING', args=vars(args)))
    write(dest/'invocation.json', vars(args))
    write(dest/'worker_pid.json', dict(pid=os.getpid()))
    write(dest/'benchmark_source_hashes.json', {str(p): sha256(p) for p in Path('tools/worldstream_m123/benchmark').glob('*.py')})
    hashes = {str(p):sha256(p) for directory in ('worldstream/m1','worldstream/m2','worldstream/m3') for p in Path(directory).glob('*.py')}
    hashes['projects/mmdet3d_plugin/models/detectors/petr3d.py'] = sha256('projects/mmdet3d_plugin/models/detectors/petr3d.py')
    write(dest/'m1_m2_source_hashes.json', {k:v for k,v in hashes.items() if k.startswith(('worldstream/m1/', 'worldstream/m2/'))})
    write(dest/'m1_m2_m3_source_hashes.json', hashes)
    cfg = effective_config(args.condition, args.workload, dest)
    importlib.import_module('projects.mmdet3d_plugin')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    model = load_model(cfg, args.checkpoint, dest)
    manifest = Path(args.run_root)/'preflight.json'
    evidence = json.loads(manifest.read_text()) if manifest.is_file() else {}
    model.worldstream_audit_identity = dict(config=FINAL if args.condition != 'B0' else BASELINE,
        config_sha256=sha256(FINAL if args.condition != 'B0' else BASELINE),
        effective_config_sha256=sha256(dest/'effective_config.json'), checkpoint=args.checkpoint,
        checkpoint_sha256=evidence.get('checkpoint', {}).get('sha256') or sha256(args.checkpoint),
        source_manifest='benchmark_source_hashes.json + m1_m2_m3_source_hashes.json',
        source_manifest_sha256=sha256(dest/'m1_m2_m3_source_hashes.json'))
    model.worldstream_profile = getattr(args, 'execution_audit', False)
    if getattr(args,'offline_wsgrr',None):
        from functools import partial
        from dataclasses import replace
        if args.condition!='B2' or model.m2_runtime is None:
            raise ValueError('Offline WSGRR task evaluation requires B2 (M1 + M2, native VLM)')
        runtime=model.m2_runtime
        runtime.config=replace(runtime.config,ordinary_budget=args.window_budget,cache_capture='always')
        # Explicit offline tool entry only. No deployment configuration can
        # select this path or bypass the production cost/quality checks.
        if args.offline_wsgrr=='NoM1RegionMatched':
            if not args.window_targets:raise ValueError('Matched-compute control requires the D3 per-frame target file')
            target_rows=[json.loads(s) for s in Path(args.window_targets).read_text().splitlines()]
            targets={}
            for row in target_rows:
                diag=row['m2'];stage=diag['stages'][0];cameras=diag['cache_context']['camera_ids']
                per_camera=stage['total_windows']//len(cameras)
                targets[row['frame_id']]=(cameras,[sum(j//per_camera==c for j in stage['executed_window_ids']) for c in range(len(cameras))])
            def matched(images,manager,context,_runtime=runtime,_targets=targets):
                cameras,counts=_targets[context['frame_id']]
                if cameras!=context['camera_ids']:raise ValueError('Matched schedule camera identities differ')
                return _runtime.evaluate_offline_from_m1(images,manager,context,variant='NoM1RegionMatched',window_targets=counts)
            runtime.forward_from_m1=matched
        else:runtime.forward_from_m1=partial(runtime.evaluate_offline_from_m1,variant=args.offline_wsgrr)
        if getattr(args,'force_full_frame',None):
            if args.offline_wsgrr!='D':raise ValueError('Visual counterfactual requires offline D')
            original=runtime.forward_from_m1
            def force_frame(images,manager,context,_runtime=runtime,_original=original):
                if context['frame_id'] not in args.force_full_frame:return _original(images,manager,context)
                snapshot=manager.export_m2_inputs(expected_context=context,computation_only=True)
                snapshot.computation_state.diagnostics['full_refresh_hint']=True
                result,diag=_runtime.evaluate_offline(images,variant='D',m1_result=snapshot,execution_context=context)
                diag['offline_counterfactual']='full_current_visual_prefix; candidate head history retained'
                return result,diag
            runtime.forward_from_m1=force_frame
        write(dest/'offline_experiment.json',dict(variant=args.offline_wsgrr,budget=args.window_budget,
            production_approvals_added=0,quality_certified=False))
    if args.validate_only:
        write(dest/'status.json', dict(status='COMPLETED', checkpoint_validation='PASS', measured_frames=0))
        return
    dataset = build_dataset(cfg.data.test)
    dataset.data_infos = sorted(dataset.data_infos, key=lambda i: (i['scene_token'], i['timestamp']))
    all_infos = dataset.data_infos
    candidates = all_infos
    if getattr(args, 'execution_audit', False):
        scene = getattr(args, 'scene', None) or all_infos[0]['scene_token']
        candidates = [i for i in all_infos if i['scene_token'] == scene]
        from .sequence import validate_sequence as validate_continuous_sequence
        validate_continuous_sequence(candidates[:args.limit], scene)
    selected = candidates if not args.limit else candidates[:args.limit]
    validate_sequence(selected)
    full = len(selected) == len(all_infos)
    write(dest/'sequence.json', dict(validation_frames=len(all_infos), selected_frames=len(selected),
        full_validation=full, tokens=[i['token'] for i in selected],
        requested_frames=args.limit, selection='single scene, consecutive timestamp-sorted frames' if getattr(args, 'execution_audit', False) else 'timestamp-sorted split',
        reduction_reason='selected scene has fewer frames than requested' if args.limit > len(selected) else None,
        scene_ids=sorted({i['scene_token'] for i in selected})))
    observer = Observer(model)
    # Annotation reverse links are only exposed to the offline scorer below.
    scorer = None
    if model.m1_manager is not None and not getattr(args,'perception_only',False):
        from tools.worldstream_m1.dataset import load_oracle
        scorer = IdentityScorer(load_oracle(selected, 'data/nuscenes'))
    frames, questions, outputs = [], [], []
    warmup_count = min(args.warmup, len(selected))
    for ordinal in range(warmup_count):
        sample = dataset[ordinal]
        data = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        if args.workload == 'language':
            select_question(data, ordinal, args.questions_per_frame)
        if getattr(args,'perception_only',False):clear_questions(data)
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        print('WARMUP', ordinal+1, flush=True)
    # Fresh output directory defeats the original baseline's file-existence
    # skip without modifying its native generation implementation.
    model.save_path = str(dest/'measured_answers')+'/'
    reset(model)
    observer.clear()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    previous_scene = None
    for ordinal, info in enumerate(selected):
        seed = (args.seed + int(hashlib.sha256(info['token'].encode()).hexdigest()[:8], 16)) % (2**31)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        observer.clear()
        torch.cuda.synchronize()
        request_tick = perf_counter()
        scene_reset = previous_scene != info['scene_token']
        if scene_reset:
            # Native heads infer resets from time gaps; scene boundaries are
            # explicit in this benchmark so no temporal memory can cross scenes.
            model.pts_bbox_head.reset_memory()
            model.map_head.reset_memory()
            for runtime in (model.m2_runtime, model.m3_runtime):
                if runtime is not None:
                    runtime.reset()
        previous_scene = info['scene_token']
        sample = dataset[ordinal]
        prepared = perf_counter()
        data = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        if args.workload == 'language':
            select_question(data, ordinal, args.questions_per_frame)
        if getattr(args,'perception_only',False):clear_questions(data)
        torch.cuda.synchronize()
        model_tick = perf_counter()
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        a.record()
        with torch.no_grad():
            try:
                result = model(return_loss=False, rescale=True, **data)
            except BaseException as error:
                append(dest/'request_failures.jsonl', dict(scene_id=info['scene_token'],
                    frame_id=info['token'], timestamp=info['timestamp']/1e6,
                    phase='model', error_type=type(error).__name__, message=str(error),
                    request_host_elapsed_ms=(perf_counter()-request_tick)*1000,
                    module_execution=copy.deepcopy(getattr(model, 'worldstream_execution', {}))))
                raise
        b.record()
        torch.cuda.synchronize()
        model_end = perf_counter()
        result = cpu_result(result[0])
        request_end = perf_counter()
        stages = observer.collect()
        row = dict(condition=args.condition, workload=args.workload, repeat=args.repeat,
            measurement_set=args.measurement_set,
            scene_id=info['scene_token'], frame_id=info['token'], timestamp=info['timestamp']/1e6,
            ordinal=ordinal, seed=seed, full_validation=full,
            explicit_scene_memory_reset=scene_reset,
            preprocessing_ms=(prepared-request_tick)*1000, transfer_and_collate_ms=(model_tick-prepared)*1000,
            model_ms=(model_end-model_tick)*1000, output_transfer_ms=(request_end-model_end)*1000,
            request_ms=(request_end-request_tick)*1000, gpu_stream_span_ms=a.elapsed_time(b),
            stages=stages, native_eva_cameras=observer.backbone_cameras,
            observed_window_attention_batches=dict(observer.window_attention_batches),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            m2=copy.deepcopy(getattr(model, 'm2_diagnostics', None)),
            m3=copy.deepcopy(getattr(model, 'm3_diagnostics', None)),
            module_execution=copy.deepcopy(getattr(model, 'worldstream_execution', None)),
            frozen_verified_history_states=len(getattr(getattr(model,'m3_verified_state',None),'records',())),
            output_digest=typed_digest({k: v for k, v in result.items() if k != 'text_out'}),
            timing_scope='Single sequential request: dataset access/preprocess, collate/H2D, model including language and native answer write, CPU result materialization. Audit serialization and counterfactual checks excluded.',
            gpu_scope='CUDA events on default stream; includes host submission gaps, not sum of active kernel durations. Nested stage spans must not be added.')
        row['identity'] = model.worldstream_audit_identity
        if model.m1_manager is not None:
            # Existing CPU timers; read outside request timing without copying
            # dense M1 audit/observation arrays into a 6019-frame result file.
            row['m1_cpu_timings_ms']=dict(model.m1_manager.last_result.diagnostics.get('timings_ms',{}))
            row['m1_commit_timings_ms']={k:v for k,v in model.m1_manager.last_commit_diagnostics.items() if k.endswith('_ms')}
        if observer.commits and not getattr(args,'perception_only',False):
            frame, anchors, audit = m1_record(model, observer.commits[-1])
            row['m1_edges'] = scorer.update(frame, anchors, audit['commit'])
            row['m1_commit'] = audit['commit']
            row['m1_precheck'] = audit['precheck']['diagnostics']
            row['m1_valid_anchors'] = len(audit['precheck'].get('valid_anchor_ids', []))
            row['m1_historical_candidates'] = len(audit['precheck'].get('association_candidate_ids', []))
            row['m1_certifications'] = audit['committed_state']['certifications']
            state = audit['committed_state']
            row['m1_state_ages_s'] = [frame.timestamp-a['source_timestamp'] for a in state['anchors']]
            row['m1_retained_states'] = sum(a['source_frame_id'] != frame.frame_id for a in state['anchors'])
            row['m1_transaction_status'] = state['transaction_status']
            row['m1_physical_phase'] = audit['commit'].get('physical_certification_phase')
            append(dest/'m1_state.jsonl', audit)
        if args.condition in ('B2', 'B3'):
            executed=sum(s.get('executed_windows',0) for s in row['m2'].get('stages',[]))
            if executed!=sum(observer.window_attention_batches.values()):
                raise RuntimeError('M2 window counters disagree with actual attention batches')
        gt = ground_truth(info['token'])
        for qi, answer in enumerate(result['text_out']):
            quality = answer_quality(answer['A'][0], gt.get(answer['Q']))
            native = observer.native_generations[qi] if qi < len(observer.native_generations) else None
            observed = native or (observer.prefill_generations[qi] if qi<len(observer.prefill_generations) else None)
            m3_request=(row['m3'] or {}).get('requests',[])
            detail=m3_request[qi] if qi<len(m3_request) else {}
            qrow = dict(condition=args.condition, workload=args.workload, repeat=args.repeat,
                measurement_set=args.measurement_set,
                scene_id=info['scene_token'], frame_id=info['token'], question_index=qi,
                question=answer['Q'], answer=answer['A'][0], reference=gt.get(answer['Q']), quality=quality,
                task_type=task_type(answer['Q']),
                native_generation_executed=native is not None,
                host_generate_call_ms=observed['host_wall_ms'] if observed else None,
                gpu_generate_stream_span_ms=observed['start'].elapsed_time(observed['end']) if observed else None,
                input_tokens=observed['input_tokens'] if observed else None,
                generated_tokens=observed['generated_tokens'] if observed else None,
                token_ids=observed['output'].detach().cpu().tolist() if observed else None,
                prefill_executor_executed=native is None and observed is not None,
                reused_prefix_length=detail.get('reused_tokens',0),
                historical_context_admitted=detail.get('state_retrieval',{}).get('admitted',False),
                m3_reason=detail.get('prefix_admission') or (row['m3'] or {}).get('reason'))
            questions.append(qrow)
            append(dest/'per_question_metrics.jsonl', qrow)
        for images, reused, diagnostics in ([] if getattr(args,'perception_only',False) else observer.reuses):
            # Counterfactual correctness check on ACTUAL reuse only. Restore
            # sampling RNG and run it outside production latency measurements.
            rng = torch.cuda.get_rng_state()
            observer.events.clear()
            with torch.no_grad():
                native = model.img_backbone(images)[0]
            error = (native.float()-reused.float()).abs()
            row.setdefault('m2_reuse_checks', []).append(dict(
                exact=torch.equal(native, reused), max_abs=float(error.max()),
                mean_abs=float(error.mean()), p95_abs=float(torch.quantile(error.flatten(), .95)),
                atol=1e-5, rtol=1e-5,
                tolerance_pass=torch.allclose(native, reused, atol=1e-5, rtol=1e-5)))
            torch.cuda.set_rng_state(rng)
        frames.append(row)
        append(dest/'per_frame_metrics.jsonl', row)
        if getattr(args, 'execution_audit', False):
            from .audit import validate_execution_row
            problems = validate_execution_row(row, expected_questions=len(result['text_out']))
            if problems:
                raise RuntimeError('Formal execution audit failed: '+', '.join(problems))
        outputs.append(result)
        if (ordinal+1) % 10 == 0 or ordinal == 0:
            write(dest/'progress.json', dict(completed_frames=ordinal+1, target_frames=len(selected),
                questions=len(questions), latest_request_ms=row['request_ms']))
            print('FRAME', args.condition, args.workload, args.repeat, ordinal+1, '/', len(selected),
                  round(row['request_ms'], 2), 'ms', flush=True)
    with (dest/'predictions.pkl').open('wb') as handle:
        pickle.dump(outputs, handle)
    summary = dict(status='COMPLETED', frames=len(frames), questions=len(questions),
        full_validation=full, warmup_frames=warmup_count,
        request_ms=distribution(r['request_ms'] for r in frames),
        model_ms=distribution(r['model_ms'] for r in frames), quality=quality_summary(questions),
        m1=scorer.summary() if scorer else None, official_detection=None, official_map=None,
        official_status='NOT_RUN: subset' if not full else ('PENDING_CPU_EVALUATION' if getattr(args,'perception_only',False) else 'PENDING'))
    if full and not getattr(args,'perception_only',False):
        preflight = json.loads((Path(args.run_root)/'preflight.json').read_text())
        integrity = preflight['dataset']['tables'].get('validation_annotation_integrity', {})
        if integrity.get('status') == 'PASS':
            try:
                dataset.eval_mode = ['det']
                summary['official_detection'] = dataset.evaluate(outputs, metric='bbox', jsonfile_prefix=str(dest/'detection'))
                summary['official_status'] = 'MEASURED'
            except Exception:
                summary['official_status'] = 'FAILED'
                (dest/'detection_evaluation_error.log').write_text(traceback.format_exc())
        else:
            summary['official_status'] = 'NOT_AVAILABLE: sample.anns disagrees with annotation reverse links'
        try:
            summary['official_map'] = map_evaluate(dataset, outputs, dest,workers=8 if getattr(args,'perception_only',False) else 0)
        except Exception:
            (dest/'map_evaluation_error.log').write_text(traceback.format_exc())
    if model.m1_manager is not None and not getattr(args,'perception_only',False):
        from tools.worldstream_m1.scene_metrics import summarize_scene
        from tools.worldstream_m1.scene_metrics import validate_rows
        from .report import read_rows
        scene_rows = {}
        for row in read_rows(dest/'m1_state.jsonl'):
            scene_rows.setdefault(row['precheck']['scene_id'], []).append(row)
        summaries = {}
        for scene, rows in scene_rows.items():
            try:
                summaries[scene] = dict(ledger=validate_rows(rows), metrics=summarize_scene(rows))
            except Exception:
                summaries[scene] = dict(status='FAILED', error=traceback.format_exc())
        write(dest/'m1_scene_integrity.json', summaries)
        summary['m1_scene_integrity'] = {s: 'PASS' if x.get('ledger', {}).get('status') == 'PASS' else 'FAIL' for s, x in summaries.items()}
    summary, unavailable = finite_metrics(summary)
    summary['undefined_metric_paths'] = unavailable
    write(dest/'metrics.json', summary)
    ledger_pass = all(x == 'PASS' for x in summary.get('m1_scene_integrity', {}).values())
    write(dest/'status.json', dict(status='COMPLETED', measured_frames=len(frames),
        full_validation=full, integrity='PASS' if ledger_pass else 'FAIL', official_status=summary['official_status']))
    if full and getattr(args,'perception_only',False):
        # Release this GPU process before CPU-only official evaluation. This
        # never shortens a reported inference timing or substitutes predictions.
        import subprocess,sys
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='')
        command=[sys.executable,'-m','tools.worldstream_m2.task_quality','--run',str(dest),'--output',str(dest/'official_eval')]
        with (dest/'official_eval.log').open('w') as handle:
            child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
        write(dest/'official_eval_child.json',dict(pid=child.pid,command=command,
            scope='CPU-only official evaluator on actual complete predictions; final status is official_eval/status.json'))


def clear_questions(data):
    """Explicit Detection/Map-only evaluation; heads and their state are unchanged."""
    from mmcv.parallel import DataContainer
    data['input_ids'][0][0]=[]
    data['img_metas'][0][0]['vlm_labels']=DataContainer([],stack=False)


def select_question(data, ordinal, limit):
    if not limit:
        return
    meta = data['img_metas'][0][0]
    labels = meta['vlm_labels'].data
    # The dataset inserts one unlabelled random description before the true
    # Conv/VQA questions. Select deterministically from real labelled pairs.
    indices = list(range(1, len(labels))) or list(range(len(labels)))
    start = ordinal % len(indices)
    selected = [indices[(start+i) % len(indices)] for i in range(min(limit, len(indices)))]
    ids = data['input_ids'][0][0]
    data['input_ids'][0][0] = [ids[i] for i in selected]
    from mmcv.parallel import DataContainer
    meta['vlm_labels'] = DataContainer([labels[i] for i in selected], stack=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--condition', choices=['B0', 'B1', 'B2', 'B3'], required=True)
    parser.add_argument('--workload', choices=['planning', 'language'], default='planning')
    parser.add_argument('--checkpoint', default='ckpts/iter_10548.pth')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--questions-per-frame', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--measurement-set', choices=['main', 'attribution', 'efficiency', 'smoke'], default='smoke')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--execution-audit', action='store_true', help='B3 only; one continuous scene, GPU stage profiling, per-frame contract audit')
    parser.add_argument('--scene', help='Scene token for --execution-audit; default first scene')
    parser.add_argument('--offline-wsgrr',choices=['A','B','C','D','NoM1Region','NoM1RegionMatched'],help='Explicit unapproved task experiment, B2 only')
    parser.add_argument('--window-budget',type=int,default=3)
    parser.add_argument('--perception-only',action='store_true',help='Explicit official Detection/Map run with no language questions; no feature counterfactual/audit scorer (done in paired task runs)')
    parser.add_argument('--window-targets',help='D3 per_frame_metrics.jsonl; cardinalities only for the explicit NoM1RegionMatched control')
    parser.add_argument('--force-full-frame',action='append',default=[],help='Explicit offline D counterfactual sample token; never a production option')
    args = parser.parse_args()
    if args.execution_audit and (args.condition != 'B3' or args.workload != 'language'
            or args.limit < 1 or args.offline_wsgrr or args.perception_only):
        parser.error('execution-audit requires B3, language and a positive frame limit; offline/perception-only forbidden')
    if set(os.sched_getaffinity(0)) & {4, 5}:
        parser.error('Launch with taskset -c 0-3,6-31')
    try:
        run(args)
    except BaseException:
        dest = Path(args.output)
        dest.mkdir(parents=True, exist_ok=True)
        (dest/'failure.log').write_text(traceback.format_exc())
        write(dest/'status.json', dict(status='FAILED', error=traceback.format_exc(), integrity='NOT_VALIDATED'))
        raise


if __name__ == '__main__':
    main()
