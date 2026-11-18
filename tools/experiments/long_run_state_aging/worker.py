"""Unchanged frozen model entry and input pipeline, one fresh process per run."""
import os
import time

from .protocol import (ROOT, read, write, require_affinity, lifecycle, check_continuity,
                       append_record, STAGES, stage_index, validate_workload)


def worker(out, directory, method, repeat, lock_fd):
    from .data import verify_manifest
    manifest = verify_manifest(out)
    config = manifest['config']
    require_affinity()
    from tools.environment_stability_eval.common import gpu_lease
    from tools.worldstream_ablation.run import wait_gpu
    with gpu_lease(lock_fd):
        wait_gpu(directory)
        import torch
        # Initialize CUDA before loading dataset/plugin native extensions.
        torch.cuda.set_device(0)
        torch.cuda.init()
        import projects.mmdet3d_plugin  # noqa: F401
        from mmcv.parallel import DataContainer, collate, scatter
        from mmdet3d.datasets import build_dataset
        from tools.worldstream_m123.benchmark.worker import effective_config, load_model, reset
        from tools.worldstream_m123.benchmark.common import diagnostic_json, typed_digest
        from tools.worldstream_m123.selective_answer_reuse import encode_questions
        from tools.worldstream_m123.frozen_main_benchmark import frame_rng
        from tools.worldstream_m123.r0_task_evaluation import facts_questions, sha
        from worldstream.m3.request_rng import request_seed
        from tools.environment_stability_eval.evaluator import validate_runtime
        from .instrumentation import collect
        from .memory import FrameMemory
        memory = FrameMemory(torch.cuda)

        cfg = effective_config('B0' if method == 'native' else 'B3', 'planning', directory)
        cfg.data.test.ann_file = str(ROOT / config['infos'])
        if sha(ROOT / config['checkpoint']) != manifest['checkpoint_sha256']:
            raise ValueError('Checkpoint changed since preparation')
        model = load_model(cfg, str(ROOT / config['checkpoint']), directory).eval().requires_grad_(False)
        runtime = validate_runtime(model, method)
        write(directory / 'runtime_identity.json', dict(pid=os.getpid(), method=method, repeat=repeat,
            checkpoint_sha256=manifest['checkpoint_sha256'], runtime=runtime,
            device=torch.cuda.get_device_name(0),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid) if hasattr(torch.cuda.get_device_properties(0), 'uuid') else None,
            precision_by_dtype=sorted({str(p.dtype) for p in model.parameters()}),
            generation=dict(do_sample=True, temperature=.1, top_p=.75, num_beams=1, max_new_tokens=320)))
        model.worldstream_audit_identity = dict(checkpoint_sha256=manifest['checkpoint_sha256'])
        model.worldstream_question_rng_enabled = True
        model.worldstream_profile = True
        write(directory / 'effective_runtime.json', dict(method=method, repeat=repeat, pid=os.getpid(),
            **runtime, checkpoint_sha256=manifest['checkpoint_sha256'], device=torch.cuda.get_device_name(0),
            fp16=diagnostic_json(cfg.get('fp16')), batch_size=1,
            generation=dict(do_sample=True, temperature=.1, top_p=.75, num_beams=1, max_new_tokens=320),
            precision_by_dtype=sorted({str(p.dtype) for p in model.parameters()}),
            cpu_affinity=sorted(os.sched_getaffinity(0))))
        dataset = build_dataset(cfg.data.test)
        original = {i['token']: i for i in dataset.data_infos}
        tokens = [t for s in manifest['scenes'] for t in s['tokens']]
        dataset.data_infos = [original[t] for t in tokens]
        index = {i['token']: k for k, i in enumerate(dataset.data_infos)}
        from .protocol import jsonl
        queries = {}
        for row in jsonl(out / 'query_ground_truth.jsonl'):
            queries.setdefault(row['frame_id'], []).append(row['question'])
        calls = 0
        with (directory / 'receipts.jsonl').open('x') as handle:
            for scene in manifest['scenes']:
                reset(model)  # Only the real independent sequence boundary.
                previous = None
                dest = directory / 'answers' / scene['scene_id']
                dest.mkdir(parents=True)
                model.save_path = str(dest) + '/'
                frame_times = dict(zip(scene['tokens'], scene['timestamps']))
                for fi, token in enumerate(scene['tokens']):
                    wall_tick = time.perf_counter()
                    if method == 'native':
                        # Native comparator has no cross-frame perception memory.
                        reset(model)
                    before = lifecycle(model)
                    context = dict(scene_id=scene['scene_id'], frame_id=token, repeat=repeat,
                                   experiment_seed=config['seed'])
                    model.worldstream_request_rng_context = dict(repeat=repeat,
                        experiment_seed=config['seed'], rng_audit=True)
                    with frame_rng(request_seed(context, -1)):
                        torch.cuda.synchronize()
                        memory.start()
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
                        memory_metrics = memory.stop()
                    validate_workload(questions, groups)
                    answers = outputs[0]['text_out']
                    if len(answers) != len(questions) or any(a['Q'] != q for a, q in zip(answers, questions)):
                        raise RuntimeError('Missing or reordered answers')
                    if questions != queries[token]:
                        raise RuntimeError('Query schedule differs from frozen GT')
                    trace = model.worldstream_execution
                    if not trace.get('completed'):
                        raise RuntimeError('Model execution incomplete')
                    instrumentation_tick = time.perf_counter()
                    after = lifecycle(model)
                    aging = None
                    if method == 'worldstream':
                        if not (trace['m1']['commit_completed'] and model.m2_diagnostics['execution_completed']
                                and model.m3_diagnostics['execution_completed']):
                            raise RuntimeError('Incomplete OGSM/DAVP/QCSE execution')
                        check_continuity(previous, before, after, token)
                        previous = after
                        aging = collect(model, frame_times[token], frame_times)
                        requests = diagnostic_json(model.m3_diagnostics['requests'])
                    else:
                        requests = [dict(llm_skipped=False, draft_candidate_tokens=0,
                                         draft_accepted_tokens=0) for _ in questions]
                    if len(requests) != len(questions):
                        raise RuntimeError('Missing query diagnostics')
                    instrumentation_ms = (time.perf_counter() - instrumentation_tick) * 1000
                    row = dict(method=method, repeat=repeat, scene_id=scene['scene_id'], frame_id=token,
                        frame_index=fi, timestamp=frame_times[token],
                        stage=STAGES[stage_index(fi, len(scene['tokens']))],
                        # Compatibility labels for reused condition-blind score_rows.
                        window_id=scene['scene_id'], stability_level='Unclassified',
                        questions=questions, groups=groups, answers=[a['A'][0] for a in answers],
                        requests=requests, e2e_ms=elapsed, instrumentation_ms=instrumentation_ms,
                        memory=memory_metrics,
                        state_before=before, state_after=after, aging=aging,
                        execution=diagnostic_json(trace),
                        fairness=dict(camera_ids=list(meta['cam_infos']), questions=questions,
                            prompt_token_ids=[v.cpu().tolist() for v in ids],
                            request_seeds=[request_seed(context, qi) for qi in range(len(questions))],
                            input_digest=typed_digest({k: data[k] for k in ('img', 'lidar2img', 'intrinsics',
                                'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv',
                                'command', 'can_bus')})))
                    # Measured processing wall additionally includes audit overhead;
                    # journal serialization/fsync are outside this auxiliary clock.
                    row['processing_wall_ms'] = (time.perf_counter() - wall_tick) * 1000
                    append_record(handle, row)
                    calls += 1
                    write(directory / 'progress.json', dict(frames=calls, frame_id=token, stage=row['stage']))
                    print('FRAME', calls, method, row['stage'], token, flush=True)
        write(directory / 'complete.json', dict(frames=calls, receipts_sha256=sha(directory / 'receipts.jsonl'),
            system_state_restored=False, execution='Independent fresh process; cold scene start', pid=os.getpid()))
