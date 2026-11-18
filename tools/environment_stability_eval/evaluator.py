"""Invoke the existing model/data pipeline unchanged in isolated paired workers."""
from dataclasses import asdict
import faulthandler
import json
import os
from pathlib import Path
import time

from .common import ROOT, gpu_lease, jsonl, read, require_affinity, write


def verify_manifest(out):
    from tools.worldstream_m123.r0_task_evaluation import sha
    manifest = read(out / 'manifest.json')
    for group in ('source_and_GT_hashes', 'input_hashes'):
        for path, expected in manifest[group].items():
            if sha(path) != expected:
                raise ValueError('Prepared source/GT/input changed: ' + path)
    for name, expected in manifest['frozen_artifact_hashes'].items():
        if sha(out / name) != expected:
            raise ValueError('Prepared GT grouping changed: ' + name)
    return manifest


def validate_runtime(model, method):
    from tools.worldstream_m123.benchmark.common import diagnostic_json
    from worldstream.r0 import model_settings
    runtimes = (model.m1_manager, model.m2_runtime, model.m3_runtime)
    if method == 'native':
        if any(r is not None for r in runtimes) or model.worldstream_require_m123:
            raise RuntimeError('Native must disable all three modules')
        return dict(M1=None, M2=None, M3=None)
    if not all(r is not None for r in runtimes) or not model.worldstream_require_m123:
        raise RuntimeError('WorldStream must load full OGSM/DAVP/QCSE')
    actual = {f'M{i}': diagnostic_json(asdict(r.config)) for i, r in enumerate(runtimes, 1)}
    expected = model_settings()
    for i in range(1, 4):
        if actual[f'M{i}'] != expected[f'm{i}_config']:
            raise RuntimeError('WorldStream differs from frozen R0: M' + str(i))
    return actual


def worker(out, method, lock_fd=None):
    require_affinity()
    directory = out / method
    manifest = verify_manifest(out)
    config = manifest['config']
    from tools.worldstream_ablation.run import stage, wait_gpu
    with (directory / 'native_crash.log').open('a') as crash:
        faulthandler.enable(file=crash, all_threads=True)
        stage(directory, 'FRESH_PROCESS', method=method, cpu_affinity=sorted(os.sched_getaffinity(0)))
        import torch
        with gpu_lease(lock_fd):
            stage(directory, 'WAIT_FOR_IDLE_GPU')
            if os.environ.get('CUDA_VISIBLE_DEVICES', '0') != '0':
                raise RuntimeError('Use physical GPU 0 for both methods')
            wait_gpu(directory)
            # CUDA comes before dataset/pickle/plugin initialization (existing host workaround).
            torch.cuda.set_device(0)
            torch.cuda.init()
            stage(directory, 'CUDA_READY', device=torch.cuda.get_device_name(0))
            import projects.mmdet3d_plugin  # noqa: F401
            from mmcv.parallel import DataContainer, collate, scatter
            from mmdet3d.datasets import build_dataset
            from tools.worldstream_m123.benchmark.worker import effective_config, load_model, reset
            from tools.worldstream_m123.benchmark.common import diagnostic_json, typed_digest
            from tools.worldstream_m123.r0_task_evaluation import sha
            from tools.worldstream_m123.selective_answer_reuse import encode_questions
            from tools.worldstream_m123.frozen_main_benchmark import frame_rng
            from tools.worldstream_m123.r0_task_evaluation import facts_questions
            from tools.worldstream_ablation.instrumentation import Meter
            from worldstream.m3.request_rng import request_seed
            cfg = effective_config('B0' if method == 'native' else 'B3', 'planning', directory)
            cfg.data.test.ann_file = str(ROOT / config['infos'])
            if sha(ROOT / config['checkpoint']) != manifest['checkpoint_sha256']:
                raise ValueError('Checkpoint changed after GT preparation')
            stage(directory, 'LOAD_FROZEN_MODEL')
            model = load_model(cfg, str(ROOT / config['checkpoint']), directory).eval().requires_grad_(False)
            runtime = validate_runtime(model, method)
            model.worldstream_audit_identity = dict(checkpoint_sha256=manifest['checkpoint_sha256'])
            model.worldstream_question_rng_enabled = True
            model.worldstream_profile = True
            write(directory / 'effective_runtime.json', dict(method=method, **runtime,
                driver_environment=manifest.get('driver_environment', {}),
                checkpoint_sha256=manifest['checkpoint_sha256'], device=torch.cuda.get_device_name(0),
                gpu_uuid=torch.cuda.get_device_properties(0).uuid if hasattr(torch.cuda.get_device_properties(0), 'uuid') else None,
                fp16=diagnostic_json(cfg.get('fp16')), batch_size=1,
                generation=dict(do_sample=True, temperature=.1, top_p=.75, num_beams=1, max_new_tokens=320),
                precision_by_dtype=sorted({str(p.dtype) for p in model.parameters()})))
            stage(directory, 'BUILD_EXISTING_DATASET')
            dataset = build_dataset(cfg.data.test)
            original = {i['token']: i for i in dataset.data_infos}
            tokens = list(dict.fromkeys(t for s in [manifest['warmup_scene']] + manifest['scenes'] for t in s['tokens']))
            dataset.data_infos = [original[t] for t in tokens]
            index = {i['token']: k for k, i in enumerate(dataset.data_infos)}
            window_by_frame = {t: w for w in jsonl(out / 'test_windows.jsonl') for t in w['frame_ids']}
            queries = {}
            for row in jsonl(out / 'query_ground_truth.jsonl'):
                queries.setdefault(row['frame_id'], []).append(row['question'])
            meter = Meter(model)

            def execute(token, scene_id, repeat):
                meter.clear()
                context = dict(scene_id=scene_id, frame_id=token, repeat=repeat, experiment_seed=config['seed'])
                model.worldstream_request_rng_context = dict(repeat=repeat, experiment_seed=config['seed'], rng_audit=True)
                with frame_rng(request_seed(context, -1)):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    data = scatter(collate([dataset[index[token]]], samples_per_gpu=1), [0])[0]
                    questions, groups = facts_questions(token, ROOT / config['data_root'], config['other_questions'])
                    ids = encode_questions(model, questions, original[token]['location'])
                    data['input_ids'][0][0] = ids
                    meta = data['img_metas'][0][0]
                    meta['vlm_labels'] = DataContainer(questions, stack=False)
                    with torch.no_grad():
                        outputs = model(return_loss=False, rescale=True, **data)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - tick) * 1000
                answers = outputs[0]['text_out']
                if len(answers) != len(questions) or any(a['Q'] != q for a, q in zip(answers, questions)):
                    raise RuntimeError('Missing, duplicated or reordered model answers')
                trace = model.worldstream_execution
                if not trace.get('completed'):
                    raise RuntimeError('Incomplete model execution')
                if method == 'worldstream':
                    if not (trace['m1']['commit_completed'] and model.m2_diagnostics['execution_completed']
                            and model.m3_diagnostics['execution_completed']):
                        raise RuntimeError('Incomplete OGSM/DAVP/QCSE path')
                    requests = diagnostic_json(model.m3_diagnostics['requests'])
                    if len(requests) != len(questions):
                        raise RuntimeError('Missing QCSE request diagnostics')
                else:
                    requests = [dict(llm_skipped=False, draft_candidate_tokens=0, draft_accepted_tokens=0) for _ in questions]
                fairness = dict(camera_ids=list(meta['cam_infos']), questions=questions,
                    prompt_token_ids=[v.cpu().tolist() for v in ids],
                    request_seeds=[request_seed(context, qi) for qi in range(len(questions))],
                    input_digest=typed_digest({k: data[k] for k in ('img', 'lidar2img', 'intrinsics', 'extrinsics',
                        'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command', 'can_bus')}))
                return dict(method=method, repeat=repeat, scene_id=scene_id, frame_id=token,
                    timestamp=original[token]['timestamp'] / 1e6, questions=questions, groups=groups,
                    answers=[a['A'][0] for a in answers], e2e_ms=elapsed, execution_status='COMPLETED',
                    **meter.collect(), requests=requests, fairness=fairness,
                    visual_diagnostics=diagnostic_json(model.m2_diagnostics) if model.m2_runtime else None,
                    execution=diagnostic_json(trace))

            # Warmup answers live in their own directory, never in formal receipts.
            stage(directory, 'DISJOINT_WARMUP', frames=config['warmup_frames'])
            reset(model)
            warm = manifest['warmup_scene']
            warm_dir = directory / 'warmup_answers'
            warm_dir.mkdir()
            model.save_path = str(warm_dir) + '/'
            for token in warm['tokens']:
                execute(token, warm['scene_id'], -1)
            write(directory / 'warmup.json', dict(frames=len(warm['tokens']),
                  tokens=warm['tokens'], counted_in_results=False))
            stage(directory, 'MEASURED_STREAM')
            calls, segments = 0, []
            with (directory / 'receipts.jsonl').open('x') as handle:
                for repeat in range(config['repeats']):
                    for scene in manifest['scenes']:
                        reset(model)
                        dest = directory / 'answers' / str(repeat) / scene['scene_id']
                        dest.mkdir(parents=True, exist_ok=False)
                        model.save_path = str(dest) + '/'
                        segment = None
                        for fi, token in enumerate(scene['tokens']):
                            window = window_by_frame[token]
                            if segment is None or segment['window_id'] != window['window_id']:
                                if segment:
                                    segment['wall_seconds'] = time.perf_counter() - segment.pop('tick')
                                    segments.append(segment)
                                segment = dict(window_id=window['window_id'], stability_level=window['stability_level'],
                                    scene_id=scene['scene_id'], repeat=repeat, frames=0, tick=time.perf_counter())
                            row = execute(token, scene['scene_id'], repeat)
                            if row['questions'] != queries[token]:
                                raise RuntimeError('Online query workload differs from prepared GT schedule')
                            row.update(frame_index=fi, window_id=window['window_id'], stability_level=window['stability_level'])
                            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                            handle.flush()
                            segment['frames'] += 1
                            calls += 1
                            write(directory / 'progress.json', dict(frames=calls, repeat=repeat,
                                  scene_id=scene['scene_id'], frame_id=token))
                        if segment:
                            segment['wall_seconds'] = time.perf_counter() - segment.pop('tick')
                            segments.append(segment)
            meter.close()
            write(directory / 'processing_segments.json', segments)
            write(directory / 'inference_complete.json', dict(status='COMPLETED', frames=calls,
                receipts_sha256=sha(directory / 'receipts.jsonl'),
                segments_sha256=sha(directory / 'processing_segments.json'), smoke=manifest['smoke']))
            stage(directory, 'COMPLETED', frames=calls)
        faulthandler.disable()
