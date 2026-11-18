"""Fresh process per method x workload x repeat. No production code patches."""
import argparse
import csv
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
PYTHON = '/home/qh/miniconda3/envs/OmniDrive/bin/python'
HERE = Path(__file__).resolve().parent


class GPUBusy(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def input_fingerprint(value):
    """Hash actual tensors with at most a 1 MiB host staging chunk."""
    import torch
    if isinstance(value, torch.Tensor):
        digest = hashlib.sha256()
        flat = value.detach().reshape(-1)
        step = max(1, 1024**2 // value.element_size())
        for i in range(0, flat.numel(), step):
            chunk = flat[i:i+step].cpu().contiguous().view(torch.uint8).numpy()
            digest.update(memoryview(chunk))
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest.hexdigest())
    if isinstance(value, (list, tuple)):
        return [input_fingerprint(v) for v in value]
    if isinstance(value, dict):
        return {k: input_fingerprint(v) for k, v in value.items()}
    return value


def enforce_affinity():
    allowed = set(range(32)) - {4, 5}
    current = os.sched_getaffinity(0)
    selected = allowed & current
    if not selected:
        raise RuntimeError('No permitted CPUs available')
    os.sched_setaffinity(0, selected)


def gpu_environment(arguments):
    """Reuse the existing verified process-local driver bundle, if required."""
    from tools.environment_stability_eval.gpu_environment import nvml_probe
    probe = nvml_probe()
    if probe.returncode == 0:
        return
    detail = probe.stdout + probe.stderr
    if probe.returncode == 18 and 'driver/library version mismatch' in detail.lower() and not os.environ.get('OMNIDRIVE_DRIVER_COMPAT_MANIFEST'):
        from tools.worldstream_ablation.driver_compat import DEFAULT_ROOT, environment, kernel_version
        bundle = DEFAULT_ROOT / kernel_version()
        if (bundle / 'manifest.json').is_file():
            env = environment(bundle)
            if nvml_probe(env).returncode == 0:
                os.execve(PYTHON, [PYTHON, '-B', '-u', '-m',
                    'tools.experiments.resource_overhead.run', *arguments], env)
    raise RuntimeError('GPU driver preflight failed; no model inference performed: ' + detail.strip())


def numeric_gpu_binding(physical_gpu, inventory):
    """Keep NVML's physical index, but expose a numeric CUDA PCI-order ordinal.

    torch 1.13's NVML device_count parser cannot handle GPU UUIDs in
    CUDA_VISIBLE_DEVICES, even though the CUDA driver accepts them.
    """
    def pci_key(row):
        domain, bus, slot = row['pci.bus_id'].strip().split(':')
        device, function = slot.split('.')
        return tuple(int(v, 16) for v in (domain, bus, device, function))
    ordered = sorted(inventory, key=pci_key)
    for ordinal, row in enumerate(ordered):
        if int(row['index']) == physical_gpu:
            return dict(physical_gpu=physical_gpu, uuid=row['uuid'].strip(),
                        pci_bus_id=row['pci.bus_id'].strip(), cuda_device_order='PCI_BUS_ID',
                        cuda_visible_devices=str(ordinal))
    raise ValueError('Requested physical GPU is absent: ' + str(physical_gpu))


def configure_gpu_visibility(physical_gpu):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
        '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.DictReader(output.splitlines(), fieldnames=['index', 'uuid', 'pci.bus_id']))
    binding = numeric_gpu_binding(physical_gpu, rows)
    os.environ['CUDA_DEVICE_ORDER'] = binding['cuda_device_order']
    os.environ['CUDA_VISIBLE_DEVICES'] = binding['cuda_visible_devices']
    return binding


def validate_runtime_config(component, actual, frozen):
    """Allow relocated policy files only when their bytes are identical."""
    expected = dict(frozen)
    policy = None
    if component == 'M3' and actual.get('selective_answer_policy') != expected.get('selective_answer_policy'):
        actual_path = actual['selective_answer_policy']
        frozen_path = expected['selective_answer_policy']
        actual_hash, frozen_hash = sha(actual_path), sha(frozen_path)
        if actual_hash != frozen_hash:
            raise ValueError('M3 policy contents differ from main: ' + actual_path + ' vs ' + frozen_path)
        policy = dict(actual_path=actual_path, main_path=frozen_path, sha256=actual_hash,
                      byte_identical=True)
        expected['selective_answer_policy'] = actual_path
    differing = {k: dict(actual=actual.get(k), main=expected.get(k))
                 for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k)}
    if differing or actual.keys() != expected.keys():
        raise ValueError(component + ' differs from main: ' + json.dumps(differing, ensure_ascii=False))
    return dict(status='PASS', component=component, relocated_policy=policy)


def worker_error_tail(path):
    lines = Path(path).read_text(errors='replace').splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('Traceback (most recent call last):')]
    lines = lines[starts[-1]:] if starts else lines[-18:]
    return '\n'.join(line[:1000] for line in lines[-24:])


def wait_gpu(gpu, check_only=False):
    while True:
        cmd = ['nvidia-smi', '--id=' + str(gpu)]
        result = subprocess.run(cmd + ['--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                text=True, capture_output=True, check=True)
        usage = subprocess.check_output(cmd + ['--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip().split(',')
        if not result.stdout.strip() and float(usage[0]) < 1000 and float(usage[1]) < 10:
            return
        print('WAIT_GPU_IDLE', now(), result.stdout.strip(), usage, flush=True)
        if check_only:
            raise GPUBusy('GPU is busy; no other processes were changed')
        time.sleep(15)


@contextmanager
def gpu_reservation(check_only):
    # Same cooperative lock as the existing benchmark. Nonblocking diagnostics
    # must also work while a job holds its lock during CPU checkpoint loading.
    if not check_only:
        from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
        with gpu_job_lock():
            yield
        return
    import fcntl
    with Path('/tmp/omnidrive_m123_latency_gpu.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GPUBusy('GPU reservation busy; check-only did not wait or change any other workload') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare(config, out, smoke):
    from tools.worldstream_m123.source_provenance import source_revision
    from tools.worldstream_m123.frozen_main_benchmark import WORKLOADS
    required = [config[key] for key in ('infos', 'scene_manifest', 'frozen_config', 'checkpoint')]
    required += ['configs/worldstream_r0.json', 'outputs/m123_final_quality_validation_run01/reference_policy.json',
                 'ckpts/pretrain_qformer/config.json', 'ckpts/pretrain_qformer/pytorch_model-00001-of-00002.bin',
                 'ckpts/pretrain_qformer/pytorch_model-00002-of-00002.bin']
    required.append(read(config['frozen_config'])['M3']['selective_answer_policy'])
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    actual = sha(config['checkpoint'])
    if actual != config['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from configured main experiment weight hash')
    scenes = read(config['scene_manifest'])['selected']
    if config['max_scenes']:
        scenes = scenes[:config['max_scenes']]
    if smoke:
        scenes = scenes[:1]
    limit = 3 if smoke else config['max_frames_per_scene']
    scenes = [dict(s, tokens=s['tokens'][:limit] if limit else s['tokens']) for s in scenes]
    if not scenes or any(not s['tokens'] for s in scenes):
        raise ValueError('Empty scene selection')
    workloads = {w[0]: dict(id=w[0], name=w[1], workload=w[2], frequency=w[3]) for w in WORKLOADS}
    if not config['workloads'] or set(config['workloads']) - workloads.keys():
        raise ValueError('Choose S1 S2 S3 S4 S5')
    if not config['methods'] or set(config['methods']) - {'native', 'worldstream'}:
        raise ValueError('Choose native and/or worldstream')
    if len(config['workloads']) != len(set(config['workloads'])) or len(config['methods']) != len(set(config['methods'])):
        raise ValueError('Duplicate workload/method names are not allowed')
    if config['repeats'] < 1 or config['warmup_frames'] < 1:
        raise ValueError('Positive repeats and warmup_frames required')
    if any(config[k] is not None and config[k] < 1 for k in ('max_scenes', 'max_frames_per_scene')):
        raise ValueError('Scene/frame limits must be positive or null')
    if config['sample_interval_seconds'] <= 0 or config['uss_interval_seconds'] <= 0:
        raise ValueError('Sampling intervals must be positive')
    sources = [p for folder in ('worldstream', 'projects', 'tools/worldstream_m123',
        'tools/experiments/resource_overhead') for p in Path(folder).rglob('*.py')]
    sources += [HERE/'run.sh', HERE/'config.json', Path('configs/worldstream_r0.json')]
    revision = source_revision()
    detached = read('docs/provenance/git_detachment.json') if Path('docs/provenance/git_detachment.json').exists() else None
    manifest = dict(status='PREPARED', created_at=now(), config=config, smoke=smoke,
        scenes=scenes, workloads=[workloads[w] for w in config['workloads']],
        frame_count=sum(len(s['tokens']) for s in scenes), source_revision=revision,
        git_commit=revision if not revision.startswith('standalone-') else None,
        exported_original_commit=detached.get('original_local_commit') if detached else None,
        git_note='Exported checkout has no live Git database; use source SHA256 inventory' if revision.startswith('standalone-') else None,
        checkpoint_sha256=actual, input_config_hashes={str(p): sha(p) for p in required},
        source_hashes={str(p): sha(p) for p in sorted(set(sources))},
        cpu_affinity=sorted(os.sched_getaffinity(0)), python=PYTHON,
        units=dict(GB='10^9 bytes', MB='10^6 bytes'),
        protocol=dict(batch_size=1, cameras=6, image_size=[640, 640],
            precision='Inherited unchanged from effective_config and load_model; recorded per worker',
            baseline='After strict weight load, before dataset construction and warmup',
            warmup='First scene prefix, then standard benchmark scene reset; allocator cache is retained',
            measurement='All selected chronological frames, including cold scene starts; peaks reset once after warmup',
            module_ownership='Deduplicate globally in OGSM/DAVP/QCSE order; subset columns are not additive',
            average_cpu='Mean of interval RSS samples plus frame-boundary samples',
            repeats='Fresh subprocess for each method/workload/repeat; paired same request seeds'))
    write(out/'manifest.json', manifest)
    return manifest


def worker(args):
    # No dataset/plugin import until the driver has been checked and CUDA initialized.
    manifest = read(args.output/'manifest.json')
    cfg_run = manifest['config']
    if args.check_gpu_once:
        wait_gpu(cfg_run['gpu'], True)
    with gpu_reservation(args.check_gpu_once):
        wait_gpu(cfg_run['gpu'], args.check_gpu_once)
        for filename, expected in manifest['source_hashes'].items():
            if sha(filename) != expected:
                raise RuntimeError('Source changed since experiment preparation: ' + filename)
        for filename, expected in manifest['input_config_hashes'].items():
            if sha(filename) != expected:
                raise RuntimeError('Input/config/weights changed since preparation: ' + filename)
        import torch
        count = torch.cuda.device_count()
        if count != 1:
            raise RuntimeError('Expected one visible CUDA device before model loading; '
                f'got {count}; CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}')
        torch.cuda.set_device(0)
        torch.cuda.init()
        device_name = torch.cuda.get_device_name(0)
        from .memory import ProcessMonitor, cuda_snapshot, module_snapshot
        monitor = ProcessMonitor(cfg_run['gpu'], cfg_run['sample_interval_seconds'], cfg_run['uss_interval_seconds'])
        monitor.start()
        started = now()
        import projects.mmdet3d_plugin
        from mmcv.parallel import DataContainer, collate, scatter
        from mmdet3d.datasets import build_dataset
        from tools.worldstream_m123.benchmark.worker import effective_config, load_model, reset
        from tools.worldstream_m123.frozen_main_benchmark import frame_rng
        from tools.worldstream_m123.continuous_stream import question_schedule
        from tools.worldstream_m123.selective_answer_reuse import encode_questions
        from worldstream.m3.request_rng import request_seed
        dest = args.output/'runs'/f'{args.workload}_r{args.repeat:02d}_{args.method}'
        dest.mkdir(parents=True, exist_ok=False)
        config = effective_config('B0' if args.method == 'native' else 'B3', 'language', dest)
        config.data.test.ann_file = cfg_run['infos']
        torch.manual_seed(cfg_run['seed'])
        model = load_model(config, cfg_run['checkpoint'], dest).eval().requires_grad_(False)
        if args.method == 'native':
            assert all(getattr(model, k) is None for k in ('m1_manager', 'm2_runtime', 'm3_runtime'))
        else:
            frozen = read(cfg_run['frozen_config'])
            validation = []
            for key, attr in [('M1', 'm1_manager'), ('M2', 'm2_runtime'), ('M3', 'm3_runtime')]:
                validation.append(validate_runtime_config(key, asdict(getattr(model, attr).config), frozen[key]))
            write(dest/'runtime_validation.json', validation)
        model.worldstream_allow_idle_frames = True
        model.worldstream_question_rng_enabled = True
        model.worldstream_profile = False
        model.worldstream_request_rng_context = dict(repeat=args.repeat, experiment_seed=cfg_run['seed'])
        model.worldstream_audit_identity = dict(checkpoint_sha256=manifest['checkpoint_sha256'])
        baseline = dict(**cuda_snapshot(), **monitor.sample())
        baseline_modules = module_snapshot(model)
        write(dest/'baseline.json', dict(resources=baseline, modules=baseline_modules))
        dataset = build_dataset(config.data.test)
        original = {i['token']: i for i in dataset.data_infos}
        tokens = [t for s in manifest['scenes'] for t in s['tokens']]
        dataset.data_infos = [original[t] for t in tokens]
        index = {i['token']: j for j, i in enumerate(dataset.data_infos)}
        workload = next(w for w in manifest['workloads'] if w['id'] == args.workload)
        runtime = {k: asdict(getattr(model, a).config) if getattr(model, a) is not None else None
                   for k, a in [('OGSM', 'm1_manager'), ('DAVP', 'm2_runtime'), ('QCSE', 'm3_runtime')]}
        effective = dict(runtime=runtime, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=device_name, physical_gpu=cfg_run['gpu'], cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            gpu_binding=cfg_run.get('gpu_binding'),
            cpu_affinity=sorted(os.sched_getaffinity(0)), batch_size=1,
            parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
            fp16=config.get('fp16'), pipeline=config.data.test.pipeline,
            generation=read(cfg_run['frozen_config'])['generation'])
        write(dest/'runtime.json', effective)

        def frame(scene, fi, token, phase):
            context = dict(scene_id=scene['scene_id'], frame_id=token,
                           repeat=args.repeat, experiment_seed=cfg_run['seed'])
            with frame_rng(request_seed(context, -1)):
                data = scatter(collate([dataset[index[token]]], samples_per_gpu=1), [0])[0]
                meta = data['img_metas'][0][0]
                if len(meta['cam_infos']) != 6:
                    raise ValueError('Expected exactly six cameras')
                natural = list(meta['vlm_labels'].data)[1:4]
                questions = question_schedule(workload['workload'], workload['frequency'], fi, natural, natural)
                ids = encode_questions(model, questions, original[token]['location']) if questions else []
                data['input_ids'][0][0] = ids
                meta['vlm_labels'] = DataContainer(questions, stack=False)
                # A digest, not an accumulating copy of GPU inputs. This is
                # outside the forward timer; its memory is included in peaks.
                input_digest = input_fingerprint(data['img'])
                prompt_digest = hashlib.sha256(json.dumps([v.tolist() for v in ids]).encode()).hexdigest()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.no_grad():
                    outputs = model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
                forward_ms = (time.perf_counter() - tick) * 1000
                answers = outputs[0]['text_out']
                if len(answers) != len(questions) or any(a['Q'] != q for a, q in zip(answers, questions)):
                    raise RuntimeError('Incomplete query workload')
                trace = getattr(model, 'worldstream_execution', {})
                if not trace.get('completed'):
                    raise RuntimeError('Incomplete model execution')
                if args.method == 'worldstream':
                    assert trace['m1']['commit_completed']
                    assert model.m2_diagnostics['execution_completed'] and model.m3_diagnostics['execution_completed']
                gpu = cuda_snapshot()
                process = monitor.sample()
                tick = time.perf_counter()
                modules = module_snapshot(model)
                accounting_ms = (time.perf_counter() - tick) * 1000
                return dict(phase=phase, method=args.method, workload=args.workload, repeat=args.repeat,
                    scene_id=scene['scene_id'], frame_id=token, frame_index=fi,
                    questions=questions, input_digest=input_digest, prompt_digest=prompt_digest,
                    request_seeds=[request_seed(context, q) for q in range(len(questions))],
                    camera_ids=list(meta['cam_infos']), image_shape=list(data['img'][0].shape),
                    input_dtype=str(data['img'][0].dtype), forward_ms=forward_ms,
                    accounting_ms=accounting_ms, **gpu, **process, modules=modules)

        monitor.set_phase('warmup')
        scene = manifest['scenes'][0]
        reset(model)
        warm_path = dest/'answers'/'warmup'
        warm_path.mkdir(parents=True)
        model.save_path = str(warm_path) + '/'
        # S1/S2 first frames have queries; warmup is never idle-only.
        with (dest/'warmup.jsonl').open('w') as stream:
            for fi, token in enumerate(scene['tokens'][:cfg_run['warmup_frames']]):
                stream.write(json.dumps(frame(scene, fi, token, 'warmup'), allow_nan=False) + '\n')
        warmup_end = dict(**cuda_snapshot(), **monitor.sample())
        # Keep CUDA allocator/cache state; only normal benchmark scene lifecycle resets.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        monitor.set_phase('measurement')
        measured_start = time.perf_counter()
        final = None
        with (dest/'frames.jsonl').open('w') as stream:
            for scene in manifest['scenes']:
                reset(model)
                scene_path = dest/'answers'/'measurement'/scene['scene_id']
                scene_path.mkdir(parents=True)
                model.save_path = str(scene_path) + '/'
                for fi, token in enumerate(scene['tokens']):
                    final = frame(scene, fi, token, 'measurement')
                    stream.write(json.dumps(final, allow_nan=False) + '\n')
                    stream.flush()
                    print('FRAME', args.method, args.workload, args.repeat, fi, token, flush=True)
        final_gpu = cuda_snapshot()
        final_process = monitor.sample()
        sampling = monitor.finish()
        measured = sampling['phases']['measurement']
        metrics = dict(baseline_gpu_allocated_bytes=baseline['gpu_allocated_bytes'],
            baseline_gpu_reserved_bytes=baseline['gpu_reserved_bytes'],
            peak_gpu_allocated_bytes=final_gpu['gpu_peak_allocated_bytes'],
            peak_gpu_reserved_bytes=final_gpu['gpu_peak_reserved_bytes'],
            incremental_peak_gpu_allocated_bytes=max(0, final_gpu['gpu_peak_allocated_bytes'] - baseline['gpu_allocated_bytes']),
            incremental_peak_gpu_reserved_bytes=max(0, final_gpu['gpu_peak_reserved_bytes'] - baseline['gpu_reserved_bytes']),
            baseline_cpu_rss_bytes=baseline['cpu_rss_bytes'], peak_cpu_rss_bytes=measured['rss_peak'],
            average_cpu_rss_bytes=measured['rss_sum'] / measured['samples'],
            incremental_peak_cpu_rss_bytes=max(0, measured['rss_peak'] - baseline['cpu_rss_bytes']),
            baseline_cpu_uss_bytes=baseline['cpu_uss_bytes'], peak_cpu_uss_bytes=measured['uss_peak'],
            nvml_peak_process_gpu_bytes=measured['nvml_peak'],
            process_lifetime_peak_cpu_rss_bytes=sampling['process_lifetime_peak_rss_bytes'],
            process_peak_gpu_allocated_bytes=max(warmup_end['gpu_peak_allocated_bytes'], final_gpu['gpu_peak_allocated_bytes']),
            process_peak_gpu_reserved_bytes=max(warmup_end['gpu_peak_reserved_bytes'], final_gpu['gpu_peak_reserved_bytes']))
        if measured['rss_peak'] <= 0 or final_gpu['gpu_peak_allocated_bytes'] <= 0:
            raise RuntimeError('Resource collection returned invalid measurements')
        write(dest/'result.json', dict(status='COMPLETED', method=args.method, workload=args.workload,
            repeat=args.repeat, pid=os.getpid(), started_at=started, finished_at=now(), frames=len(tokens),
            measurement_seconds=time.perf_counter()-measured_start, runtime=effective,
            baseline=baseline, baseline_modules=baseline_modules, warmup_end=warmup_end,
            final_resources=dict(**final_gpu, **final_process), final_modules=final['modules'], sampling=sampling, metrics=metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'config.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--workloads', nargs='+')
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--max-scenes', type=int)
    parser.add_argument('--max-frames-per-scene', type=int)
    parser.add_argument('--warmup-frames', type=int)
    parser.add_argument('--smoke', action='store_true', help='One scene x three frames; default one repeat and S4')
    parser.add_argument('--cpu-smoke', action='store_true', help='Real CPU storage/sampler checks; no model claims')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--check-gpu-once', action='store_true', help='Exit if busy; default wait without touching workloads')
    parser.add_argument('--summarize', type=Path, help='Regenerate outputs from an existing experiment')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--method', help=argparse.SUPPRESS)
    parser.add_argument('--workload', help=argparse.SUPPRESS)
    parser.add_argument('--repeat', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    enforce_affinity()
    if args.cpu_smoke:
        from .smoke import run
        run(args.output or Path('outputs/experiments/resource_overhead/cpu_smoke'))
        return
    if args.summarize:
        from .report import summarize
        summarize(args.summarize)
        return
    if args.worker:
        worker(args)
        return
    cfg = read(args.config)
    for key in ('repeats', 'workloads', 'methods', 'gpu', 'max_scenes', 'max_frames_per_scene', 'warmup_frames'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.smoke:
        if args.repeats is None:
            cfg['repeats'] = 1
        if args.workloads is None:
            cfg['workloads'] = ['S4']
    if not args.prepare_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu'])
        gpu_environment(sys.argv[1:])
        cfg['gpu_binding'] = configure_gpu_visibility(cfg['gpu'])
    out = args.output or Path(cfg['output_root']) / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    manifest = prepare(cfg, out, args.smoke)
    print('PREPARED', out, manifest['frame_count'], 'frames', flush=True)
    if args.prepare_only:
        return
    write(out/'status.json', dict(status='RUNNING', started_at=now()))
    for repeat in range(cfg['repeats']):
        for wi, workload in enumerate(cfg['workloads']):
            methods = cfg['methods'] if (repeat + wi) % 2 == 0 else list(reversed(cfg['methods']))
            for method in methods:
                cmd = ['taskset', '-c', '0-3,6-31', PYTHON, '-B', '-u', '-m',
                    'tools.experiments.resource_overhead.run', '--worker', '--output', str(out),
                    '--method', method, '--workload', workload, '--repeat', str(repeat)]
                if args.check_gpu_once:
                    cmd.append('--check-gpu-once')
                with (out/f'{workload}_r{repeat:02d}_{method}.log').open('w') as log:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    write(out/'status.json', dict(status='GPU_BUSY' if result.returncode == 75 else 'FAILED',
                        finished_at=now(), command=cmd, returncode=result.returncode))
                    if result.returncode == 75:
                        raise GPUBusy('Smoke not run: GPU busy; see ' + str(log.name))
                    # Show the actual worker exception without printing the
                    # long initialization warnings from the checkpoint loader.
                    tail = worker_error_tail(log.name)
                    raise RuntimeError('Worker failed; see ' + str(log.name) + '\n' + tail)
    write(out/'status.json', dict(status='INFERENCE_COMPLETED', finished_at=now()))
    from .report import summarize
    summarize(out)
    write(out/'status.json', dict(status='COMPLETED', finished_at=now()))


if __name__ == '__main__':
    try:
        main()
    except GPUBusy as exc:
        print(str(exc), flush=True)
        sys.exit(75)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
