"""One isolated condition; native model calls, unchanged sampling and admission."""
import argparse
import copy
from contextlib import nullcontext
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


def image_values(value):
    if hasattr(value,'data') and not isinstance(value,(torch.Tensor,np.ndarray)):return image_values(value.data)
    if isinstance(value,(tuple,list)):return [image_values(v) for v in value]
    return value


def effective_config(condition, workload, dest, m3_strategy=None):
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
    if m3_strategy is not None:
        if condition!='B3':raise ValueError('M3 comparison retains formal M1/M2 in B3')
        if m3_strategy=='native_full':
            cfg.model.worldstream_require_m123=False
            cfg.model.m3_enable=False
        else:cfg.model.m3_config.reuse_strategy=m3_strategy
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


def configure_m2_evidence(cfg, args, dest):
    """Explicit benchmark overrides; default deployment configuration is untouched."""
    candidate = getattr(args, 'm2_candidate_cost_table', None)
    formal = getattr(args, 'm2_cost_table', None)
    if candidate and formal: raise ValueError('Select either candidate or formal M2 costs')
    if candidate and (args.condition != 'B2' or args.workload != 'language'
            or getattr(args,'offline_wsgrr',None) != 'D' or getattr(args,'perception_only',False)):
        raise ValueError('M2 candidate evidence requires B2 language and offline D')
    if formal and (args.condition not in ('B2','B3') or getattr(args,'offline_wsgrr',None)):
        raise ValueError('Formal M2 cost override requires B2/B3 without offline execution')
    if getattr(args,'m2_profile_evidence',False) and not candidate:
        raise ValueError('M2 cost profiling requires an explicit candidate table')
    if getattr(args,'m2_task_evidence',False) and (args.workload!='language' or args.condition not in ('B2','B3')):
        raise ValueError('M2 task witnesses require full-model B2/B3 language execution')
    if getattr(args,'m2_acceptance',False) and (args.condition!='B3'
            or args.workload!='language' or args.limit<2 or args.perception_only):
        raise ValueError('M2 acceptance requires B3 language and at least two frames')
    path = candidate or formal
    rules_path = getattr(args,'m2_assessment_rules',None)
    if rules_path:
        if args.condition not in ('B2','B3') or args.workload!='language':
            raise ValueError('M2 assessment rules require B2/B3 language execution')
        from tools.worldstream_m2.assess_candidate import check_limits
        check_limits(json.loads(Path(rules_path).read_text()))
        write(dest/'m2_assessment_rules_binding.json',dict(path=str(Path(rules_path).resolve()),sha256=sha256(rules_path)))
    if path:
        from worldstream.m2.cost_model import CostModel
        from worldstream.m2.config import M2Config
        table = CostModel(path)
        if table.status != 'ready': raise ValueError('Invalid or empty M2 cost table: '+table.status)
        policy = M2Config(**dict(cfg.model.m2_config)).refresh_signature()
        if any(tuple(json.loads(r['key']).get('refresh_policy',())) != policy for r in table.records):
            raise ValueError('M2 candidate refresh policy differs from formal configuration')
        if candidate and args.window_budget!=policy[1]: raise ValueError('Candidate budget must match formal refresh policy')
        write(dest/'m2_cost_binding.json', dict(path=str(Path(path).resolve()), sha256=sha256(path),
            kind='candidate' if candidate else 'formal', records=len(table.records)))
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
    # Offline frozen_worker loads before-code packages in a fresh process. Bind
    # the actual loaded files rather than claiming the checkout's code ran.
    directories=[Path(os.path.relpath(Path(importlib.import_module('worldstream.'+c).__file__).parent)) for c in ('m1','m2','m3')]
    hashes = {str(p):sha256(p) for directory in directories for p in directory.glob('*.py')}
    hashes['projects/mmdet3d_plugin/models/detectors/petr3d.py'] = sha256('projects/mmdet3d_plugin/models/detectors/petr3d.py')
    write(dest/'m1_m2_source_hashes.json', {k:v for k,v in hashes.items() if Path(k).parent in directories[:2]})
    write(dest/'m1_m2_m3_source_hashes.json', hashes)
    cfg = effective_config(args.condition, args.workload, dest,getattr(args,'m3_strategy',None))
    if getattr(args,'dataset_split','val') == 'train':
        if (args.condition!='B3' or not args.capture_m3_inputs or not args.capture_stable_world
                or not args.m3_zero_reuse or getattr(args,'wrcps_adapter',None)):
            raise ValueError('Training split is restricted to native-layout, zero-KV real teacher captures')
        cfg.data.test.ann_file=cfg.data.train.ann_file
        for node in cfg.data.test.pipeline:
            if node['type']=='LoadAnnoatationVQATest':
                node['base_conv_path']='./data/nuscenes/conv/train/'
                node['base_vqa_path']='./data/nuscenes/vqa/train/'
    configure_m2_evidence(cfg,args,dest)
    for key in ('global_anchor_interval','global_drift_threshold'):
        value=getattr(args,'m2_'+key,None)
        if value is not None:
            if args.condition!='B3':raise ValueError('Anchor calibration requires the formal B3 algorithm')
            cfg.model.m2_config[key]=value
    if getattr(args,'m2_risk_config',None):
        if args.condition!='B3':raise ValueError('Risk configuration requires B3')
        risk=json.loads(Path(args.m2_risk_config).read_text())
        allowed={'global_anchor_interval','global_drift_threshold','global_risk_weights','global_risk_scales','global_risk_threshold','global_tail_fraction'}
        if set(risk)-allowed:raise ValueError('Unsupported risk parameter')
        cfg.model.m2_config.update(risk)
    if getattr(args,'m3_policy',None):
        cfg.model.m3_config.reuse_policy=args.m3_policy
    if getattr(args,'wrcps_adapter',None):
        if args.condition!='B3' or args.m3_strategy!='exact_prefix':
            raise ValueError('WRCPS experiments require B3 with exact_prefix')
        cfg.model.m3_config.prefix_adapter=args.wrcps_adapter
    write(dest/'effective_config.json', dict(config=cfg._cfg_dict.to_dict(),
        note='Explicit benchmark workload/output/policy overrides; production source config unchanged.'))
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
    model.worldstream_profile = bool(getattr(args, 'execution_audit', False) or getattr(args,'m3_strategy',None)
        or getattr(args,'m2_task_evidence',False) or getattr(args,'m2_profile_evidence',False))
    model.worldstream_capture_m3_inputs=bool(getattr(args,'capture_m3_inputs',False))
    if getattr(args,'offline_wsgrr',None):
        from functools import partial
        from dataclasses import replace
        if args.condition!='B2' or model.m2_runtime is None:
            raise ValueError('Offline WSGRR task evaluation requires B2 (M1 + M2, native VLM)')
        runtime=model.m2_runtime
        runtime.config=replace(runtime.config,ordinary_budget=args.window_budget)
        # Historical selector controls live only in tools. D uses the same
        # serving algorithm and neither route has M2 cost/quality admission.
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
    m2_evidence = None
    if getattr(args,'m2_profile_evidence',False):
        from tools.worldstream_m2.full_model_evidence import FullModelEvidence
        m2_evidence = FullModelEvidence(model.m2_runtime)
    if args.validate_only:
        write(dest/'status.json', dict(status='COMPLETED', checkpoint_validation='PASS', measured_frames=0))
        return
    dataset = build_dataset(cfg.data.test)
    dataset.data_infos = sorted(dataset.data_infos, key=lambda i: (i['scene_token'], i['timestamp']))
    all_infos = dataset.data_infos
    candidates = all_infos
    specification={}
    if getattr(args,'sequence_file',None):
        specification=json.loads(Path(args.sequence_file).read_text())
        tokens=specification['tokens']
        by_token={i['token']:i for i in all_infos}
        candidates=[by_token[t] for t in tokens]
    elif getattr(args, 'execution_audit', False) or getattr(args,'m3_strategy',None):
        scene = getattr(args, 'scene', None) or all_infos[0]['scene_token']
        candidates = [i for i in all_infos if i['scene_token'] == scene]
        from .sequence import validate_sequence as validate_continuous_sequence
        validate_continuous_sequence(candidates[:args.limit], scene)
    selected = candidates if not args.limit else candidates[:args.limit]
    validate_sequence(selected)
    full = len(selected) == len(all_infos)
    dataset.data_infos=selected
    write(dest/'sequence.json', dict(validation_frames=len(all_infos), selected_frames=len(selected),
        full_validation=full, tokens=[i['token'] for i in selected],
        requested_frames=args.limit, selection='single scene, consecutive timestamp-sorted frames' if getattr(args, 'execution_audit', False) or getattr(args,'m3_strategy',None) else 'timestamp-sorted split',
        reduction_reason='selected scene has fewer frames than requested' if args.limit > len(selected) else None,
        scene_ids=sorted({i['scene_token'] for i in selected})))
    observer = Observer(model)
    fixed=None
    if getattr(args,'fixed_token_reference',None):
        if args.workload!='language' or getattr(args,'shadow_global_labels',False):
            raise ValueError('Fixed tokens require a standalone language performance run')
        from .fixed_tokens import FixedTokenBenchmark
        fixed=FixedTokenBenchmark(model,args.fixed_token_reference)
    if getattr(args,'m3_zero_reuse',False):
        if model.m3_runtime is None:raise ValueError('Zero-reuse control requires the formal M3 runtime')
        from .m3_recovery import ZeroReuseControl
        ZeroReuseControl(model.m3_runtime)
    equivalence=None
    if getattr(args,'risk_equivalence_reference',None):
        if model.m2_runtime is None or fixed is not None:raise ValueError('Risk equivalence requires a separate M2 diagnostic run')
        from .risk_equivalence import AnchorEquivalence
        equivalence=AnchorEquivalence(model.m2_runtime,observer,args.risk_equivalence_reference,dest)
    observer.capture_m1_controls=bool(getattr(args,'capture_m1_controls',False))
    decision_trace = None
    if getattr(args, 'trace_m3_decisions', False):
        from .decision_replay import DecisionTrace
        decision_trace = DecisionTrace(model.m3_runtime)
    observer.capture_inputs=bool(getattr(args,'capture_m3_inputs',False))
    shadow=None
    if getattr(args,'shadow_global_labels',False):
        if model.m2_runtime is None:raise ValueError('Shadow labels need actual M2')
        from .residual_labels import ResidualLabels
        shadow=ResidualLabels(model.m2_runtime,observer)
    if getattr(args,'m3_policy',None):
        # Read-only timers have the same portable backend identity as serving.
        from worldstream.m3.visual_state import deployment_key
        policy=json.loads(Path(args.m3_policy).read_text())
        binding=policy['binding'];canonical=deployment_key(model.lm_head)
        if binding['weights_sha256']!=model.worldstream_audit_identity['checkpoint_sha256']:
            raise ValueError('Policy checkpoint does not match loaded weights')
        if binding['execution_key']!=canonical:
            raise ValueError('Policy does not match the actual model/backend')
        write(dest/'used_m3_policy.json',policy)
    # Annotation reverse links are only exposed to the offline scorer below.
    scorer = None
    if model.m1_manager is not None and not getattr(args,'perception_only',False):
        from tools.worldstream_m1.dataset import load_oracle
        scorer = IdentityScorer(load_oracle(selected, 'data/nuscenes'))
    frames, questions, outputs = [], [], []
    prepared_samples=None
    if getattr(args,'preload_inputs',False):
        prepared_samples=[dataset[i] for i in range(len(selected))]
        write(dest/'prepared_input_hashes.json',{info['token']:typed_digest(image_values(s['img'])) for info,s in zip(selected,prepared_samples)})
    warmup_count = min(args.warmup, len(selected))
    for ordinal in range(warmup_count):
        sample = dataset[ordinal] if prepared_samples is None else copy.deepcopy(prepared_samples[ordinal])
        data = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        if args.workload == 'language':
            names=specification.get('questions',{}).get(selected[ordinal]['token'])
            if names:select_named_questions(data,names)
            else:select_question(data, ordinal, args.questions_per_frame, getattr(args, 'question_offset', 0), getattr(args,'question_selection','ordered'))
        if getattr(args,'perception_only',False):clear_questions(data)
        if fixed is not None:fixed.prepare(selected[ordinal]['token'])
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        torch.cuda.synchronize()
        print('WARMUP', ordinal+1, flush=True)
    # Fresh output directory defeats the original baseline's file-existence
    # skip without modifying its native generation implementation.
    model.save_path = str(dest/'measured_answers')+'/'
    reset(model)
    if shadow is not None:shadow.reset()
    observer.clear()
    if equivalence is not None:equivalence.clear()
    if m2_evidence is not None: m2_evidence.clear()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    previous_scene = None
    for ordinal, info in enumerate(selected):
        if m2_evidence is not None: m2_evidence.clear()
        seed = (args.seed + int(hashlib.sha256(info['token'].encode()).hexdigest()[:8], 16)) % (2**31)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        observer.clear()
        if equivalence is not None:equivalence.clear()
        if fixed is not None:fixed.prepare(info['token'])
        if shadow is not None:shadow.clear()
        if decision_trace is not None:
            decision_trace.clear()
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
        sample = dataset[ordinal] if prepared_samples is None else copy.deepcopy(prepared_samples[ordinal])
        prepared = perf_counter()
        data = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        if args.workload == 'language':
            names=specification.get('questions',{}).get(info['token'])
            if names:select_named_questions(data,names)
            else:select_question(data, ordinal, args.questions_per_frame, getattr(args, 'question_offset', 0), getattr(args,'question_selection','ordered'))
        if getattr(args,'perception_only',False):clear_questions(data)
        torch.cuda.synchronize()
        model_tick = perf_counter()
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        a.record()
        from .runtime_profile import RuntimeProfile
        profile_context=RuntimeProfile(dest/'profile_output') if getattr(args,'runtime_profile',False) and ordinal==args.profile_frame else nullcontext()
        with profile_context, torch.no_grad():
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
        if shadow is not None:
            for label in shadow.rows:append(dest/'residual_labels.jsonl',label)
        stages = observer.collect()
        row = dict(condition=args.condition, workload=args.workload, repeat=args.repeat,
            m3_strategy=getattr(args,'m3_strategy',None),
            measurement_set=args.measurement_set,
            scene_id=info['scene_token'], frame_id=info['token'], timestamp=info['timestamp']/1e6,
            ordinal=ordinal, seed=seed, full_validation=full,
            explicit_scene_memory_reset=scene_reset,
            preprocessing_ms=(prepared-request_tick)*1000, transfer_and_collate_ms=(model_tick-prepared)*1000,
            model_ms=(model_end-model_tick)*1000, output_transfer_ms=(request_end-model_end)*1000,
            request_ms=(request_end-request_tick)*1000, gpu_stream_span_ms=a.elapsed_time(b),
            input_boundary='preloaded_cpu_tensors' if prepared_samples is not None else 'real_dataset_preprocessing',
            shadow_suffix_labels=shadow is not None,
            risk_equivalence_diagnostic=equivalence is not None,
            generation_workload='fixed_reference_tokens' if fixed is not None else 'free_generation',
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
        row['input_tensor_digest']=typed_digest(image_values(sample['img']))
        if fixed is not None:row['fixed_tokens']=fixed.collect(result['text_out'])
        if equivalence is not None:append(dest/'risk_equivalence.jsonl',equivalence.collect())
        row['identity'] = model.worldstream_audit_identity
        if getattr(args, 'capture_stable_world', False):
            from dataclasses import asdict
            from worldstream.m3.stable_world import freeze_stable_world
            manager = model.m1_manager
            snapshot = freeze_stable_world(manager, dict(scene_id=info['scene_token'],
                frame_id=info['token'], timestamp=info['timestamp']/1e6,
                state_epoch=manager.store.generation, state_version=manager.store.commit_count,
                state_execution_signature=manager.store.execution_signature))
            append(dest/'stable_world_snapshots.jsonl', asdict(snapshot))
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
            observed_suffix=sum(v['calls'] for k,v in stages.items() if k.startswith('m2_suffix_block_'))
            expected_suffix=row['m2'].get('global_suffix_executed_blocks')
            if expected_suffix is not None and observed_suffix!=expected_suffix:
                raise RuntimeError('Global suffix diagnostics disagree with actual block calls')
            row['observed_global_suffix_blocks']=observed_suffix
            row['observed_global_suffix_skipped_blocks']=len(model.img_backbone.blocks)-len(model.m2_runtime.prefix)-observed_suffix
        gt = ground_truth(info['token'])
        for qi, answer in enumerate(result['text_out']):
            quality = answer_quality(answer['A'][0], gt.get(answer['Q'])) if fixed is None else None
            native = observer.native_generations[qi] if qi < len(observer.native_generations) else None
            observed = native or (observer.prefill_generations[qi] if qi<len(observer.prefill_generations) else None)
            m3_request=(row['m3'] or {}).get('requests',[])
            detail=m3_request[qi] if qi<len(m3_request) else {}
            qrow = dict(condition=args.condition, workload=args.workload, repeat=args.repeat,
                m3_strategy=getattr(args,'m3_strategy',None),
                measurement_set=args.measurement_set,
                scene_id=info['scene_token'], frame_id=info['token'], question_index=qi,
                question=answer['Q'], answer=answer['A'][0], reference=gt.get(answer['Q']) if fixed is None else None, quality=quality,
                generation_workload=row['generation_workload'],quality_valid_for_task_evaluation=fixed is None,
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
            if fixed is not None:fixed.validate_output(qi,answer['Q'],qrow['token_ids'])
            phase_events=observed.get('phase_events',[]) if observed else []
            native_prefill=[(a,b) for label,a,b,_ in phase_events if label=='prefill']
            qrow.update(prefill_ms=detail.get('prefill_ms') if detail else sum(a.elapsed_time(b) for a,b in native_prefill),
                decode_ms=detail.get('decode_ms') if detail else
                    native_prefill[-1][1].elapsed_time(observed['end']) if native_prefill else None,
                reuse_diagnostics=detail)
            if (getattr(args, 'm3_strategy', None) or getattr(args,'offline_wsgrr',None)
                    or getattr(args,'m2_cost_table',None) or getattr(args,'m2_task_evidence',False)):
                call = observer.last_m3_call
                if native is not None:
                    ids, carriers = native['audit_input_ids'], native['audit_carriers']
                    generation = native['audit_generation']
                else:
                    ids, carriers = call['input_ids_list'][qi], call['vision_carriers']
                    generation = {k:call[k] for k in ('do_sample','temperature','top_p','num_beams','max_new_tokens','use_cache')}
                # Witnesses are materialized after measured request and peak memory.
                qrow.update(input_witness=dict(ids=typed_digest(ids.reshape(-1)),
                    carriers=typed_digest(carriers), generation=generation),
                    actual_reuse_length=detail.get('reused_tokens', 0),
                    effective_prefill_tokens=detail.get('effective_prefill_tokens', qrow['input_tokens']),
                    candidate_reuse_lengths=detail.get('candidate_prefix_tokens', []))
            questions.append(qrow)
            append(dest/'per_question_metrics.jsonl', qrow)
        if decision_trace is not None:
            for trace in decision_trace.rows:
                append(dest/'decision_traces.jsonl', trace)
        if getattr(args,'capture_m1_controls',False):
            folder=dest/'m1_controls';folder.mkdir(exist_ok=True)
            torch.save(observer.m1_control,folder/(info['token']+'.pt'))
        if observer.capture_inputs:
            # Materialize inputs only AFTER measured service time and peak VRAM.
            # Labels/GT are scored separately and never supplied to the planner.
            from worldstream.m3 import CarrierLayout
            calls=[x['captured_input'] for x in observer.native_generations]
            if not calls:
                call=observer.last_m3_call
                if call is None:raise RuntimeError('No executed M3 call to capture')
                calls=[dict(ids=ids,carriers=call['vision_carriers'],cpu_rng=observed['captured_rng'][0],
                    cuda_rng=observed['captured_rng'][1],generation={k:call[k] for k in ('do_sample','temperature','top_p','num_beams','max_new_tokens','use_cache')})
                    for ids,observed in zip(call['input_ids_list'],observer.prefill_generations)]
            if calls[0]['carriers'].shape[1]!=513:raise ValueError('Unexpected real OmniDrive carrier layout')
            state=getattr(model,'m3_verified_state',None)
            if state is None:raise RuntimeError('Capture requires real frozen M1 state')
            from worldstream.m3.stable_world import freeze_stable_world
            manager=model.m1_manager
            stable_context=dict(scene_id=state.scene_id,frame_id=state.frame_id,timestamp=state.timestamp,
                state_epoch=manager.store.generation,state_version=manager.store.commit_count,
                state_execution_signature=manager.store.execution_signature)
            stable_snapshot=freeze_stable_world(manager,stable_context)
            folder=dest/'m3_inputs';folder.mkdir(exist_ok=True)
            torch.save(dict(carriers=calls[0]['carriers'].detach().cpu(),
                ids=[c['ids'].detach().cpu().reshape(-1) for c in calls],
                generated_ids=[g['output'].detach().cpu() for g in (observer.native_generations or observer.prefill_generations)],
                rng=[(c['cpu_rng'],c['cuda_rng']) for c in calls],generation=calls[0]['generation'],
                questions=[a['Q'] for a in result['text_out']],verified_state=state,
                stable_snapshot=stable_snapshot,stable_context=stable_context,
                layout=CarrierLayout(256,1,256),m2=row['m2'],
                context=dict(scene_id=state.scene_id,frame_id=state.frame_id,timestamp=state.timestamp,
                    camera_ids=list(info['cams']),state_epoch=state.state_epoch,state_version=state.state_version,
                    state_execution_signature=state.execution_signature,
                    weights_sha256=model.worldstream_audit_identity['checkpoint_sha256'])),
                folder/(info['token']+'.pt'))
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
        if m2_evidence is not None:
            m2_evidence.collect(info['token'])
            row['m2_evidence_scope']='Extra profiling outside request timing; cumulative GPU peak may include evidence workspace'
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
    if equivalence is not None:equivalence.finish()
    if m2_evidence is not None:
        m2_evidence.finish(dest,model.worldstream_audit_identity['checkpoint_sha256'])
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
    if getattr(args,'m2_acceptance',False):
        from tools.worldstream_m2.production_acceptance import validate_run
        acceptance=validate_run(dest,dest/'m2_acceptance.json')
        if acceptance['status']!='PASS': raise RuntimeError('Formal M2 acceptance failed; see m2_acceptance.json')
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


def select_question(data, ordinal, limit, offset=0, selection='ordered'):
    if not limit:
        return
    meta = data['img_metas'][0][0]
    labels = meta['vlm_labels'].data
    # The dataset inserts one unlabelled random description before the true
    # Conv/VQA questions. Select deterministically from real labelled pairs.
    indices = list(range(1, len(labels))) or list(range(len(labels)))
    if not indices:
        raise ValueError('No language questions available')
    start = (ordinal + offset) % len(indices)
    selected = [indices[(start+i) % len(indices)] for i in range(min(limit, len(indices)))]
    if selection=='balanced':
        from worldstream.m3.task_conditioner import task_profile
        wanted=['object_attributes','map_relations','object_ego_motion','full_visual']
        wanted=wanted[ordinal%4:]+wanted[:ordinal%4]
        if any(task_profile(labels[i])[1]=='object_ego_motion' for i in indices):
            wanted=['object_ego_motion']+[p for p in wanted if p!='object_ego_motion']
        chosen=[]
        for profile in wanted:
            eligible=[i for i in indices if i not in chosen and task_profile(labels[i])[1]==profile]
            if eligible:chosen.append(eligible[ordinal%len(eligible)])
            if len(chosen)==min(limit,len(indices)):break
        selected=(chosen+[i for i in selected if i not in chosen])[:min(limit,len(indices))]
    ids = data['input_ids'][0][0]
    data['input_ids'][0][0] = [ids[i] for i in selected]
    from mmcv.parallel import DataContainer
    meta['vlm_labels'] = DataContainer([labels[i] for i in selected], stack=False)


def select_named_questions(data,names):
    """Replay exactly the existing question/input pairs recorded in a manifest."""
    from mmcv.parallel import DataContainer
    meta=data['img_metas'][0][0];labels=meta['vlm_labels'].data
    if len(set(names))!=len(names) or any(labels.count(q)!=1 for q in names):
        raise ValueError('Explicit question is missing or ambiguous in native dataset')
    selected=[labels.index(q) for q in names];ids=data['input_ids'][0][0]
    data['input_ids'][0][0]=[ids[i] for i in selected]
    meta['vlm_labels']=DataContainer(list(names),stack=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--condition', choices=['B0', 'B1', 'B2', 'B3'], required=True)
    parser.add_argument('--m2-global-anchor-interval',type=int,help='Offline calibration override of the same M2 policy')
    parser.add_argument('--m2-global-drift-threshold',type=float,help='Offline calibration override of the same M2 policy')
    parser.add_argument('--m2-risk-config',help='Offline fixed risk-parameter JSON; same Global-State Update')
    parser.add_argument('--shadow-global-labels',action='store_true',help='Offline exact suffix labels; exclude entire run from latency evidence')
    parser.add_argument('--preload-inputs',action='store_true',help='Prepare identical CPU input tensors before model-only controls')
    parser.add_argument('--m3-zero-reuse',action='store_true',help='Benchmark only: same planner/cache/executor with reuse length fixed to zero')
    parser.add_argument('--fixed-token-reference',help='Benchmark only: common Native reference tokens, real HF forwards, no task-quality scoring')
    parser.add_argument('--risk-equivalence-reference',help='Offline frozen pre-pruning M2 package; extra reference suffixes, no latency claims')
    parser.add_argument('--workload', choices=['planning', 'language'], default='planning')
    parser.add_argument('--checkpoint', default='ckpts/iter_10548.pth')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--questions-per-frame', type=int, default=1)
    parser.add_argument('--question-offset', type=int, default=0,
        help='Deterministic offset into the existing labelled Conv/VQA questions; no answer-based filtering')
    parser.add_argument('--question-selection', choices=['ordered','balanced'], default='ordered')
    parser.add_argument('--sequence-file', help='Explicit ordered real sample tokens for paired multi-scene runs')
    parser.add_argument('--capture-m1-controls',action='store_true',help='Save real committed inputs outside timed requests for identical-state CPU controls')
    parser.add_argument('--capture-stable-world', action='store_true', help='Audit qualified current M1 stable facts after actual six-camera inference')
    parser.add_argument('--dataset-split', choices=['val','train'], default='val', help='train is capture-only adapter development; never acceptance evidence')
    parser.add_argument('--runtime-profile',action='store_true',help='Diagnostic only: sampled cProfile and CUDA trace, not performance evidence')
    parser.add_argument('--profile-frame',type=int,default=1)
    parser.add_argument('--capture-m3-inputs', action='store_true', help='Capture real native inputs outside service timing for offline calibration')
    parser.add_argument('--trace-m3-decisions', action='store_true',
        help='Read-only candidate statistics for offline signal replay; no embeddings or KV copies')
    parser.add_argument('--m3-policy', help='Unified fitted WSDR policy for this experiment')
    parser.add_argument('--wrcps-adapter', help='Trained experimental Composer; requires a newly bound exact policy, never changes production config')
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--measurement-set', choices=['main', 'attribution', 'efficiency', 'smoke'], default='smoke')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--execution-audit', action='store_true', help='B3 only; one continuous scene, GPU stage profiling, per-frame contract audit')
    parser.add_argument('--scene', help='Scene token for --execution-audit; default first scene')
    parser.add_argument('--m3-strategy',choices=['native_full','exact_prefix','wsdr_kv'],
        help='Explicit small M3 comparison with unchanged formal M1/M2 and generation; no approvals')
    parser.add_argument('--offline-wsgrr',choices=['A','B','C','D','NoM1Region','NoM1RegionMatched'],help='Explicit unapproved task experiment, B2 only')
    parser.add_argument('--m2-candidate-cost-table', help='Offline D only: candidate cost identity, never a production approval')
    parser.add_argument('--m2-cost-table', help='Optional profiling metadata only; never affects M2 execution')
    parser.add_argument('--m2-profile-evidence', action='store_true', help='Post-request M2 block/pipeline costs on the actual source snapshot')
    parser.add_argument('--m2-task-evidence', action='store_true', help='Record input-ID and generation witnesses for paired M2 quality evaluation')
    parser.add_argument('--m2-assessment-rules', help='Bind predeclared quality/error limits before model execution')
    parser.add_argument('--m2-acceptance', action='store_true', help='Require actual nonzero reuse through formal B3 admission')
    parser.add_argument('--window-budget',type=int,default=3)
    parser.add_argument('--perception-only',action='store_true',help='Explicit official Detection/Map run with no language questions; no feature counterfactual/audit scorer (done in paired task runs)')
    parser.add_argument('--window-targets',help='D3 per_frame_metrics.jsonl; cardinalities only for the explicit NoM1RegionMatched control')
    parser.add_argument('--force-full-frame',action='append',default=[],help='Explicit offline D counterfactual sample token; never a production option')
    args = parser.parse_args()
    if args.question_offset < 0:parser.error('question-offset must be nonnegative')
    if args.capture_m3_inputs and args.condition!='B3':parser.error('Capture requires B3 with real M1 state')
    if args.trace_m3_decisions and args.m3_strategy!='wsdr_kv':parser.error('Decision traces require wsdr_kv')
    if args.execution_audit and (args.condition != 'B3' or args.workload != 'language'
            or args.limit < 1 or args.offline_wsgrr or args.perception_only):
        parser.error('execution-audit requires B3, language and a positive frame limit; offline/perception-only forbidden')
    if args.m3_strategy and (args.condition!='B3' or args.workload!='language' or args.limit<1
            or args.offline_wsgrr or args.perception_only or args.execution_audit):
        parser.error('M3 comparison requires B3 language, a short positive limit and no other experiment modes')
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
