"""Prepare frozen factual workload; launch fresh-process latency trials."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

from tools.environment_stability_eval.common import ROOT, PYTHON, read, write, require_affinity, csv_write, jsonl
from tools.worldstream_ablation.run import preflight, sha, stage, wait_gpu


def prepare(args):
    config = read(args.config)
    if args.trials is not None:
        config['trials'] = args.trials
    config.setdefault('latency_scope', 'forward_only_v2')
    if config['latency_scope'] != 'forward_only_v2':
        raise ValueError('New measurements require latency_scope=forward_only_v2')
    for key in ('trials', 'warmup_frames', 'smoke_frames', 'smoke_pairs', 'event_batch_frames'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Expected positive integer: ' + key)
    if config['smoke_frames'] < 3 or config['other_questions'] != 2:
        raise ValueError('Keep chronological smoke >=3 frames and official 3+2 query workload')
    frozen = preflight(SimpleNamespace(infos=Path(config['infos']), checkpoint=Path(config['checkpoint']),
        main_manifest=Path(config['main_manifest']), scope='main-scenes', smoke=False, max_scenes=None,
        smoke_frames=config['smoke_frames'], other_questions=config['other_questions'], seed=config['seed'],
        group='all', semantic_threshold=.95))
    reference = read(config['reference_manifest'])
    for key in ('scenes', 'seed', 'other_questions', 'checkpoint_sha256', 'infos'):
        if frozen[key] != reference[key]:
            raise ValueError('Formal factual workload differs from existing experiment: ' + key)
    import pickle
    from tools.worldstream_m123.r0_task_evaluation import selected_scenes
    from tools.environment_stability_eval.data_adapter import validate_frames
    infos = pickle.load(Path(config['infos']).open('rb'))['infos']
    by = {i['token']: i for i in infos}
    ids = {s['scene_id'] for s in frozen['scenes']}
    warm = next(s for s in selected_scenes(infos, 'full-val', ROOT, None)
                if s['scene_id'] not in ids and len(s['tokens']) >= config['warmup_frames'])
    warm = dict(warm, tokens=warm['tokens'][:config['warmup_frames']])
    scenes = [dict(frozen['scenes'][0], tokens=frozen['scenes'][0]['tokens'][:config['smoke_frames']])] if args.smoke else frozen['scenes']
    validate_frames(scenes + [warm], by)
    for token in warm['tokens']:
        for cam in by[token]['cams'].values():
            frozen['data_hashes'][cam['data_path']] = sha(cam['data_path'])
        for folder in ('conv', 'vqa'):
            path = str(Path('data/nuscenes') / folder / 'val' / (token+'.json'))
            frozen['data_hashes'][path] = sha(path)
    for p in Path('tools/component_latency_eval').rglob('*'):
        if p.is_file() and p.suffix in ('.py', '.sh', '.json') and 'outputs' not in p.parts:
            frozen['source_hashes'][str(p)] = sha(p)
    frozen['source_hashes'][str(args.config)] = sha(args.config)
    frozen['source_hashes'][config['reference_manifest']] = sha(config['reference_manifest'])
    return dict(config=config, smoke=args.smoke, scenes=scenes, warmup_scene=warm,
        effective_trials=config['smoke_pairs'] if args.smoke else config['trials'],
        formal_frames=frozen['frames'], measured_frames=sum(len(s['tokens']) for s in scenes),
        checkpoint_sha256=frozen['checkpoint_sha256'], source_hashes=frozen['source_hashes'],
        data_hashes=frozen['data_hashes'], cpu_affinity=sorted(os.sched_getaffinity(0)),
        timing=dict(host='perf_counter_ns elapsed host/enqueue (includes existing implicit waits)',
            device='deferred CUDA events on actual current stream; stream span includes host gaps, not kernel busy time',
            frame_component='Frame Forward',
            scope_version='forward_only_v2',
            frame_forward='Synchronized single model(return_loss=False, rescale=True, **prepared_device_data) call; excludes dataset access, collate/scatter/H2D, question loading/tokenization, and deferred post-QCSE text decode/JSON export',
            query='Synchronized per-query runtime body; excludes shared binding/carriers and later text export',
            mixed='CPU and GPU separate; no unsynchronized mixed wall claimed',
            nested='Inclusive parent totals are not added to children; OGSM/heads use explicit disjoint sums',
            trials='fresh processes, identical seed/workload; reset only at sequence boundaries; warmup excluded',
            limitations='Verifier and native continuation share a token loop, reported merged; post-rejection fallback cannot be timed separately without token-level instrumentation'))


def worker(out, directory, baseline=False):
    manifest = read(out/'experiment_config.json')
    config = manifest['config']
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    stage(directory, 'WAIT_GPU_LEASE', baseline=baseline)
    with gpu_job_lock():
        wait_gpu(directory)
        import torch
        torch.cuda.set_device(0)
        torch.cuda.init()
        stage(directory, 'CUDA_READY', device=torch.cuda.get_device_name(0), baseline=baseline)
        for key in ('source_hashes', 'data_hashes'):
            for path, digest in manifest[key].items():
                if sha(path) != digest:
                    raise RuntimeError('Source/input changed since preparation: ' + path)
        if sha(config['checkpoint']) != manifest['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed')
        import projects.mmdet3d_plugin  # noqa
        from mmcv.parallel import DataContainer, collate, scatter
        from mmdet3d.datasets import build_dataset
        from tools.worldstream_m123.benchmark.worker import effective_config, load_model, reset
        from tools.worldstream_m123.benchmark.common import diagnostic_json, typed_digest
        from tools.worldstream_m123.r0_task_evaluation import facts_questions
        from tools.worldstream_m123.selective_answer_reuse import encode_questions
        from tools.worldstream_m123.frozen_main_benchmark import frame_rng
        from tools.environment_stability_eval.evaluator import validate_runtime
        from worldstream.m3.request_rng import request_seed
        from .timing import Recorder, FIELDS
        from .instrumentation import Instrumentation
        from .forward import DeferredExports, forward_inference
        if manifest.get('timing', {}).get('frame_component') != 'Frame Forward':
            raise RuntimeError('This worker requires a newly prepared forward_only_v2 manifest')
        cfg = effective_config('B3', 'planning', directory)
        cfg.data.test.ann_file = config['infos']
        stage(directory, 'LOAD_MODEL')
        model = load_model(cfg, config['checkpoint'], directory).eval().requires_grad_(False)
        runtime = validate_runtime(model, 'worldstream')
        model.worldstream_audit_identity = dict(checkpoint_sha256=manifest['checkpoint_sha256'])
        model.worldstream_question_rng_enabled = True
        # Existing profile=True inserts a query-end sync and per-forward/token
        # events. Disable it in both control/treatment; our scopes replace it.
        model.worldstream_profile = False
        model.worldstream_request_rng_context = dict(repeat=0, experiment_seed=config['seed'])
        write(directory/'effective_runtime.json', dict(runtime=runtime, batch_size=1, gpu=0,
            device=torch.cuda.get_device_name(0), driver_manifest=os.environ.get('OMNIDRIVE_DRIVER_COMPAT_MANIFEST'),
            fp16=diagnostic_json(cfg.get('fp16')), original_internal_profiling=False,
            latency_scope='forward_only_v2', output_export='deferred_until_after_forward_end_fence',
            precision=sorted({str(p.dtype) for p in model.parameters()})))
        dataset = build_dataset(cfg.data.test)
        original = {i['token']:i for i in dataset.data_infos}
        tokens = list(dict.fromkeys(t for s in [manifest['warmup_scene']] + manifest['scenes'] for t in s['tokens']))
        dataset.data_infos = [original[t] for t in tokens]
        indices = {t:i for i,t in enumerate(tokens)}
        recorder = Recorder(directory.name, cuda=torch.cuda, enabled=not baseline, query_sync=not baseline)
        exports = DeferredExports()
        instrumentation = Instrumentation(model, recorder, exports=exports, components=not baseline)
        write(directory/'instrumentation_audit.json', instrumentation.audit)
        def execute(token, scene_id):
            recorder.frame(scene_id, token)
            context = dict(scene_id=scene_id, frame_id=token, repeat=0, experiment_seed=config['seed'])
            with frame_rng(request_seed(context, -1)):
                # Prepare the complete input outside the timed inference scope.
                data = scatter(collate([dataset[indices[token]]], samples_per_gpu=1), [0])[0]
                questions, groups = facts_questions(token, 'data/nuscenes', config['other_questions'])
                data['input_ids'][0][0] = encode_questions(model, questions, original[token]['location'])
                data['img_metas'][0][0]['vlm_labels'] = DataContainer(questions, stack=False)
                outputs = forward_inference(model, data, recorder, exports)
            answers = outputs[0]['text_out']
            if len(questions) != 5 or [a['Q'] for a in answers] != questions:
                raise RuntimeError('Official 5-query workload incomplete')
            trace = model.worldstream_execution
            if not (trace['completed'] and trace['m1']['commit_completed'] and
                    model.m2_diagnostics['execution_completed'] and model.m3_diagnostics['execution_completed']):
                raise RuntimeError('Incomplete execution')
            if not baseline:
                instrumentation.annotate_frame(model)
            frame_time = next(r['component_wall_ms'] for r in reversed(recorder.rows) if r['component_name']=='Frame Forward')
            return dict(sequence_id=scene_id, frame_id=token, questions=questions, groups=groups,
                answers=[a['A'][0] for a in answers], forward_ms=frame_time, latency_scope='forward_only_v2',
                input_digest=typed_digest({k:data[k] for k in ('img','lidar2img','intrinsics','extrinsics','timestamp','ego_pose','command','can_bus')}),
                requests=diagnostic_json(model.m3_diagnostics['requests']),
                visual_diagnostics=diagnostic_json(model.m2_diagnostics),
                state_version=model.m1_manager.store.commit_count)
        try:
            stage(directory, 'WARMUP', frames=config['warmup_frames'])
            reset(model)
            warm = manifest['warmup_scene']
            dest = directory/'warmup_answers'
            dest.mkdir()
            model.save_path = str(dest)+'/'
            for token in warm['tokens']:
                execute(token, warm['scene_id'])
                recorder.resolve(already_synchronized=True)
                recorder.clear()
            write(directory/'warmup.json', dict(tokens=warm['tokens'], excluded=True, reset_after=True))
            with (directory/'raw_component_latency.csv').open('x', newline='') as raw, (directory/'receipts.jsonl').open('x') as receipts:
                writer = csv.DictWriter(raw, fieldnames=FIELDS)
                writer.writeheader()
                frames = 0
                for scene in manifest['scenes']:
                    reset(model)
                    if model.m1_manager.store.commit_count or model.m2_runtime.cache.context is not None:
                        raise RuntimeError('Sequence reset left committed state')
                    dest = directory/'answers'/scene['scene_id']
                    dest.mkdir(parents=True)
                    model.save_path = str(dest)+'/'
                    for token in scene['tokens']:
                        row = execute(token, scene['scene_id'])
                        receipts.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        receipts.flush()
                        frames += 1
                        if frames % config['event_batch_frames'] == 0:
                            recorder.resolve(already_synchronized=True)
                            writer.writerows(recorder.rows)
                            recorder.clear()
                            raw.flush()
                        stage(directory, 'MEASURED_FRAME', frames=frames, total=manifest['measured_frames'])
                recorder.resolve(already_synchronized=True)
                writer.writerows(recorder.rows)
                recorder.clear()
            write(directory/'completion.json', dict(status='COMPLETED', frames=frames,
                raw_sha256=sha(directory/'raw_component_latency.csv'), receipts_sha256=sha(directory/'receipts.jsonl')))
        finally:
            instrumentation.close()


def overhead(out):
    import numpy as np
    config = read(out/'experiment_config.json')['config']
    latency_key = 'forward_ms' if config.get('latency_scope') == 'forward_only_v2' else 'e2e_ms'
    rows = []
    for trial in sorted(p for p in out.glob('trial_*') if p.is_dir()):
        baseline = out/('baseline_'+trial.name.split('_',1)[1])
        a, b = list(jsonl(baseline/'receipts.jsonl')), list(jsonl(trial/'receipts.jsonl'))
        identity = lambda r:(r['sequence_id'], r['frame_id'], r['questions'], r['answers'], r['input_digest'])
        if [identity(r) for r in a] != [identity(r) for r in b]:
            raise RuntimeError('Instrumentation changed inputs or answers')
        before, after = np.mean([r[latency_key] for r in a]), np.mean([r[latency_key] for r in b])
        rows.append(dict(pair=trial.name, baseline_mean_ms=float(before), instrumented_mean_ms=float(after),
                         relative_shift=float(after/before-1), identical_answers=True))
    shift = float(np.mean([r['relative_shift'] for r in rows]))
    write(out/'instrumentation_overhead.json', dict(pairs=rows, mean_relative_shift=shift,
        latency_field=latency_key,
        status='REVIEW_REQUIRED' if abs(shift) > config['overhead_warning_fraction'] else 'PASS',
        threshold=config['overhead_warning_fraction'],
        interpretation='small-sample startup/thermal variation plus query fences, coarse AST scopes and CUDA events; synchronized prepared-input forward only, output export excluded; no statistical precision claim',
        baseline='same real full R0 without component scopes/query fences; native internal profile=False in both'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('tools/component_latency_eval/config.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--trials', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--baseline', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require_affinity()
    os.chdir(ROOT)
    if args.worker:
        try:
            worker(args.output.resolve(), args.worker.resolve(), args.baseline)
        except BaseException as exc:
            write(args.worker/'exception.json', dict(error=repr(exc), traceback=traceback.format_exc()))
            raise
        return
    if args.report_only:
        from .report import report
        report(args.output)
        return
    manifest = prepare(args)
    out = args.output or Path(manifest['config']['output_root'])/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out/'experiment_config.json', manifest)
    if args.prepare_only:
        print('PREPARED', out, manifest['measured_frames'], 'frames; no GPU inference', flush=True)
        return
    for i in range(manifest['effective_trials']):
        commands = [(out/f'trial_{i+1:02d}', False)]
        if args.smoke:
            commands.insert(0 if i%2==0 else 1, (out/f'baseline_{i+1:02d}', True))
        for directory, baseline in commands:
            directory.mkdir()
            command = ['taskset','-c','0-3,6-31',PYTHON,'-B','-u','-m','tools.component_latency_eval.run',
                       '--output', str(out), '--worker', str(directory)] + (['--baseline'] if baseline else [])
            write(directory/'command.json', command)
            with (directory/'runtime.log').open('x') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                write(directory/'process.json', dict(pid=proc.pid))
                print('WORKER', directory, proc.pid, flush=True)
                code = proc.wait()
            if code:
                raise RuntimeError('Worker failed: '+str(directory))
    if args.smoke:
        overhead(out)
    from .report import report
    report(out)
    print('COMPLETED', out, flush=True)


if __name__ == '__main__':
    main()
