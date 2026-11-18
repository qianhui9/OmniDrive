"""Remeasure paired full-model forward latency without preprocessing or export.

Uses frozen GT/scenes and the unchanged existing evaluator. New instrumentation
is installed in the worker process only; historical source/receipts stay valid.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

from .common import (LEVELS, METHODS, PYTHON, ROOT, csv_write, gpu_lease, jsonl,
                     read, require_affinity, write)

DEFINITION = (
    'CUDA-synchronized host wall time for one root model(...) call on already '
    'prepared GPU inputs (one six-camera frame, all five queries); excludes '
    'dataset loading/preprocessing/collation/H2D, prompt encoding, model loading, '
    'warmup, final answer batch_decode and answer-file writes. Includes full '
    'vision/detection/map heads, OGSM/DAVP/QCSE, internal state adapters/transfers, '
    'prefill/autoregressive decode, model-required host work and existing read-only '
    'profiling overhead. Not one Transformer token forward or GPU kernel-only time.'
)
MARKER_START = '<!-- FORWARD_LATENCY_PRIMARY_START -->'
MARKER_END = '<!-- FORWARD_LATENCY_PRIMARY_END -->'


def _sha(path):
    from tools.worldstream_m123.r0_task_evaluation import sha
    return sha(path)


def gpu_environment():
    from .gpu_environment import nvml_probe, ensure_gpu_environment
    probe = nvml_probe()
    message = probe.stdout + probe.stderr
    if probe.returncode == 18 and 'driver/library version mismatch' in message.lower():
        from tools.worldstream_ablation import driver_compat
        bundle = driver_compat.DEFAULT_ROOT / driver_compat.kernel_version()
        env = driver_compat.environment(bundle)
        print('FORWARD_DRIVER_COMPAT_REEXEC', bundle, flush=True)
        os.execve(PYTHON, [PYTHON, '-B', '-u', '-m',
                          'tools.environment_stability_eval.forward_latency', *sys.argv[1:]], env)
    if probe.returncode:
        raise RuntimeError('NVML preflight failed: ' + message)
    return ensure_gpu_environment(sys.argv[1:])


def prepare(source, out, smoke, driver_environment=None):
    from .evaluator import verify_manifest
    completion = read(source / 'completion.json')
    if completion['status'] != 'COMPLETED' or completion['smoke']:
        raise ValueError('Source must be the completed full paired experiment')
    if out.exists():
        raise FileExistsError('Forward measurement directory must be new: ' + str(out))
    print('VERIFY_FROZEN_SOURCE_AND_INPUTS', flush=True)
    manifest = deepcopy(verify_manifest(source))
    instrumentation = [Path(__file__).resolve(), Path(__file__).with_name('forward_timing.py').resolve()]
    for path in instrumentation:
        manifest['source_and_GT_hashes'][str(path)] = _sha(path)
    manifest.update(parent_run=str(source), parent_manifest_sha256=_sha(source / 'manifest.json'),
                    forward_timing_definition=DEFINITION, smoke=smoke,
                    driver_environment=driver_environment or {},
                    cpu_affinity=sorted(os.sched_getaffinity(0)))
    if smoke:
        first = deepcopy(manifest['scenes'][0])
        first['tokens'] = first['tokens'][:manifest['config']['smoke_frames']]
        manifest['scenes'] = [first]
        manifest['frames'] = len(first['tokens'])
    out.mkdir(parents=True, exist_ok=False)
    for name in manifest['frozen_artifact_hashes']:
        shutil.copy2(source / name, out / name)
    write(out / 'manifest.json', manifest)
    write(out / 'forward_status.json', dict(status='PREPARED', source=str(source),
        smoke=smoke, expected_frames_per_method=manifest['frames'], definition=DEFINITION))
    return manifest


def worker(out, method, lock_fd):
    import torch
    from . import evaluator
    from .forward_timing import ForwardClock, observe_forward
    from tools.worldstream_ablation import instrumentation

    OriginalMeter = instrumentation.Meter

    class ForwardMeter(OriginalMeter):
        def __init__(self, model):
            # Existing worker has acquired the lease, waited for idle GPU, loaded
            # CUDA/model and validated the runtime before constructing this meter.
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(20261030)
                witness = torch.ones((32, 32), device='cuda')
                result = witness @ witness
                torch.cuda.synchronize()
                if not torch.all(result == 32).item():
                    raise RuntimeError('Real CUDA kernel witness failed')
                write(out / method / 'kernel_witness.json', dict(status='PASS',
                    seed=20261030, device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                    witness_sum=result.sum().item(), cpu_affinity=sorted(os.sched_getaffinity(0))))
            super().__init__(model)
            self.clock = ForwardClock(torch.cuda.synchronize)
            self.observer = observe_forward(model, self.clock)
            self.observer.__enter__()

        def collect(self):
            values = super().collect()
            if self.clock.last is None:
                raise RuntimeError('No synchronized forward timing for this frame')
            return dict(**values, **self.clock.last)

        def clear(self):
            super().clear()
            if hasattr(self, 'clock'):
                self.clock.last = None

        def close(self):
            self.observer.__exit__(None, None, None)
            super().close()

    instrumentation.Meter = ForwardMeter
    try:
        evaluator.worker(out, method, lock_fd)
    finally:
        instrumentation.Meter = OriginalMeter


def report(out):
    import math
    import numpy as np
    from .evaluator import verify_manifest
    from .metrics import cluster_intervals, validate_pair
    manifest = verify_manifest(out)
    rows, runtimes = {}, {}
    flat = []
    agreement = {}
    source = Path(manifest['parent_run'])
    for method in METHODS:
        directory = out / method
        complete = read(directory / 'inference_complete.json')
        if (complete['status'] != 'COMPLETED' or
                _sha(directory / 'receipts.jsonl') != complete['receipts_sha256'] or
                _sha(directory / 'processing_segments.json') != complete['segments_sha256']):
            raise ValueError('Incomplete/modified measurement: ' + method)
        rows[method] = list(jsonl(directory / 'receipts.jsonl'))
        if len(rows[method]) != complete['frames']:
            raise ValueError('Measurement frame count mismatch')
        runtimes[method] = read(directory / 'effective_runtime.json')
        original = {(r['repeat'], r['scene_id'], r['frame_id']): r
                    for r in jsonl(source / method / 'receipts.jsonl')}
        matched_answers = 0
        for row in rows[method]:
            value = row['forward_ms']
            if not math.isfinite(value) or value <= 0 or row['forward_excluded_ms'] < 0:
                raise ValueError('Invalid root forward measurement')
            if abs(value + row['forward_excluded_ms'] - row['forward_raw_ms']) > 1e-6:
                raise ValueError('Forward timing exclusion accounting mismatch')
            if row['forward_excluded_ms'] != sum(row['forward_excluded_by_label_ms'].values()):
                raise ValueError('Invalid nonoverlapping export accounting')
            if row['forward_raw_ms'] > row['e2e_ms'] + 1:
                raise ValueError('Forward interval outside measured frame')
            calls = row['forward_exclusion_calls']
            if calls.get('answer_text_decode') != len(row['answers']) or calls.get('answer_file_write') != 1:
                raise ValueError('Final answer decode/file export was not fully excluded')
            original_row = original[row['repeat'], row['scene_id'], row['frame_id']]
            for field in ('questions', 'fairness'):
                if row[field] != original_row[field]:
                    raise ValueError('Remeasurement changed frozen inputs: ' + field)
            matched_answers += sum(a == b for a, b in zip(row['answers'], original_row['answers']))
            flat.append({k: row[k] for k in ('method', 'repeat', 'scene_id', 'frame_id', 'timestamp',
                'stability_level', 'forward_ms', 'forward_raw_ms', 'forward_excluded_ms',
                'forward_excluded_by_label_ms', 'forward_exclusion_calls', 'e2e_ms')})
        agreement[method] = dict(exactly_matching_answers=matched_answers,
                                compared_answers=sum(len(r['answers']) for r in rows[method]))
    validate_pair(manifest, list(jsonl(out / 'test_windows.jsonl')), rows, runtimes)

    def stats(part):
        values = [r['forward_ms'] for r in part]
        return dict(mean_forward_ms=float(np.mean(values)) if values else None,
                    p95_forward_ms=float(np.percentile(values, 95)) if values else None)

    summary = []
    for level in (*LEVELS, 'Unclassified', 'ALL'):
        for method in METHODS:
            part = [r for r in rows[method] if level == 'ALL' or r['stability_level'] == level]
            values = stats(part)
            values.update(cluster_intervals(part, stats, list(values),
                samples=manifest['config']['bootstrap_samples'], seed=manifest['config']['seed']))
            summary.append(dict(method=method, stability_level=level, frames=len(part),
                independent_sequences=len({r['scene_id'] for r in part}), **values,
                mean_raw_model_call_ms=float(np.mean([r['forward_raw_ms'] for r in part])) if part else None,
                mean_excluded_export_ms=float(np.mean([r['forward_excluded_ms'] for r in part])) if part else None))
    csv_write(out / 'raw_forward_latency.csv', flat)
    csv_write(out / 'forward_latency_summary.csv', summary)
    result = dict(status='COMPLETED', smoke=manifest['smoke'], source=str(source),
        generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        definition=DEFINITION, summary=summary, answers_vs_original=agreement,
        paired_inputs_verified=True, cuda_synchronized=True,
        instrumentation='Original read-only MACs/profile observers retained in both methods; no GPU-kernel-only claim.',
        source_manifest_sha256=_sha(out / 'manifest.json'))
    write(out / 'forward_latency_report.json', result)
    _write_markdown(out, result)
    if not result['smoke']:
        update_source_report(source, out, result)
    write(out / 'forward_completion.json', dict(status='COMPLETED', smoke=manifest['smoke'],
        frames_per_method=len(rows['native']), definition=DEFINITION,
        report_sha256=_sha(out / 'forward_latency_report.json')))
    write(out / 'forward_status.json', dict(status='COMPLETED', report=str(out / '结果.md')))
    print('FORWARD_REPORT_COMPLETED', out, flush=True)
    return result


def _write_markdown(out, result):
    lines = ['# 模型纯前向推理时延', '', '**状态：COMPLETED**' +
             ('；Smoke 执行验证，不代表完整实验。' if result['smoke'] else '；完整场景配对测量。'), '',
             result['definition'], '',
             '一帧六相机及全部 5 个查询为一次根模型调用；自回归生成计入，不是单 token Transformer forward。'
             '输入预处理、外部 H2D 和 prompt 编码位于同步计时之前。模型内部必要的状态适配/传输仍计入，'
             '答案最终文本解码和文件写入经非重叠同步区间独立扣除；MACs/诊断观察器开销保留。', '',
             '| 等级 | 方法 | 帧 | 纯前向均值(ms) | 95%CI | P95(ms) | P95 95%CI | 调用原始均值(ms) | 排除导出均值(ms) |',
             '|---|---|---:|---:|---|---:|---|---:|---:|']
    for row in result['summary']:
        def fmt(v):
            return 'N/A' if v is None else f'{v:.3f}'
        lines.append('| ' + ' | '.join([row['stability_level'], row['method'], str(row['frames']),
            fmt(row['mean_forward_ms']),
            f'[{fmt(row["mean_forward_ms_ci95_low"])}, {fmt(row["mean_forward_ms_ci95_high"])}]',
            fmt(row['p95_forward_ms']),
            f'[{fmt(row["p95_forward_ms_ci95_low"])}, {fmt(row["p95_forward_ms_ci95_high"])}]',
            fmt(row['mean_raw_model_call_ms']), fmt(row['mean_excluded_export_ms'])]) + ' |')
    lines += ['', '原实验答案一致性（不替换原准确率/变化响应指标）：', '',
              '| 方法 | 与原结果逐字一致的回答 | 比较回答数 |', '|---|---:|---:|']
    for method, value in result['answers_vs_original'].items():
        lines.append(f'| {method} | {value["exactly_matching_answers"]} | {value["compared_answers"]} |')
    lines += ['', '[逐帧前向时间](raw_forward_latency.csv)；[统计CSV](forward_latency_summary.csv)；'
              '[机器可读定义/汇总](forward_latency_report.json)。', '']
    (out / '结果.md').write_text('\n'.join(lines))


def update_source_report(source, out, result=None):
    path = source / '结果.md'
    if not path.exists():
        return
    text = path.read_text()
    if result is None:
        body = ('## 当前主时延口径：模型纯前向（等待实测）\n\n'
                'WorldStream 的主时延改为：输入已预处理并传入 GPU 后，一次完整模型调用的 CUDA 同步墙钟时延，'
                '排除最终答案解码/文件写入；OGSM/DAVP/QCSE 和模型内部必要适配计入。Native 采用相同边界。\n\n'
                '**当前没有该口径的完整实测值，不能用原 2,972.17 ms 替代。** 补充测量已排队；'
                'GPU 有其他任务时只等待。原 E2E 数字在后文保留为历史口径，不作为本次主前向时延。\n\n'
                f'新测量目录：`{out}`；完成后本段自动写入正式数值。\n')
    else:
        chosen = {r['method']: r for r in result['summary'] if r['stability_level'] == 'ALL'}
        a, b = chosen['native'], chosen['worldstream']
        body = ('## 当前主时延口径：模型纯前向（已完成实测）\n\n'
                '输入预处理、外部 H2D、prompt 编码和模型加载不计入；排除最终答案解码/写文件。'
                '完整 OGSM/DAVP/QCSE 与自回归生成计入。Native 使用相同边界。\n\n'
                '| 方法 | 正式帧 | 平均纯前向(ms/帧) | P95(ms/帧) |\n|---|---:|---:|---:|\n'
                f'| Native | {a["frames"]} | {a["mean_forward_ms"]:.3f} | {a["p95_forward_ms"]:.3f} |\n'
                f'| WorldStream | {b["frames"]} | {b["mean_forward_ms"]:.3f} | {b["p95_forward_ms"]:.3f} |\n\n'
                f'同口径平均前向加速比：{a["mean_forward_ms"]/b["mean_forward_ms"]:.3f}×。'
                f'[分等级/95%CI及逐帧记录]({os.path.relpath(out / "结果.md", source)})。'
                '后文 E2E、持续 FPS、原窗口加速比是原测量历史口径，不能与本节混算。\n')
    block = MARKER_START + '\n' + body + MARKER_END
    if MARKER_START in text:
        start, tail = text.split(MARKER_START, 1)
        _, end = tail.split(MARKER_END, 1)
        text = start + block + end
    else:
        heading, rest = text.split('\n', 1)
        text = heading + '\n\n' + block + '\n' + rest
    text = text.replace('## 3. 总体结果', '## 3. 原实验总体结果（E2E 为历史口径，非纯前向）')
    text = text.replace('## 4. 延迟与吞吐：全等级及 95% CI', '## 4. 历史 E2E 延迟与持续吞吐：全等级及 95% CI')
    text = text.replace('## 9. 配对窗口完整明细', '## 9. 原 E2E 配对窗口完整明细（非纯前向）')
    temporary = path.with_name(path.name + '.forward.tmp')
    temporary.write_text(text)
    temporary.replace(path)
    write(source / 'forward_latency_link.json', dict(status='COMPLETED' if result else 'QUEUED',
        forward_run=str(out), definition=DEFINITION, updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        current_markdown_sha256=_sha(path)))


def launch(out):
    print('WAIT_PROJECT_GPU_LEASE (other jobs are never interrupted)', flush=True)
    with gpu_lease() as lease:
        manifest = read(out / 'manifest.json')
        if not manifest['smoke']:
            smoke_out = out / 'smoke_validation'
            prepare(Path(manifest['parent_run']), smoke_out, True, manifest['driver_environment'])
            _launch_locked(smoke_out, lease)
            report(smoke_out)
            print('REAL_FORWARD_SMOKE_PASS; starting full paired measurement', flush=True)
        _launch_locked(out, lease)
    report(out)


def _launch_locked(out, lease):
    for method in METHODS:
        folder = out / method
        folder.mkdir(exist_ok=False)
        command = ['taskset', '-c', '0-3,6-31', PYTHON, '-B', '-u', '-X', 'faulthandler',
                   '-m', 'tools.environment_stability_eval.forward_latency', '--_worker', method,
                   '--output', str(out), '--_lock-fd', str(lease)]
        environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        with (folder / 'runtime.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, pass_fds=(lease,))
            write(folder / 'process.json', dict(pid=child.pid, command=command))
            print('STARTED_FORWARD', method, child.pid, folder / 'runtime.log', flush=True)
            status = child.wait()
        if status:
            detail = '\n'.join((folder / 'runtime.log').read_text(errors='replace').splitlines()[-20:])
            raise RuntimeError(f'{method} forward worker exited {status}: {detail}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'tools/environment_stability_eval/outputs/run_20261117_113951_740642')
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--run-full', action='store_true')
    parser.add_argument('--_worker', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--_lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    require_affinity()
    source = args.source.resolve()
    if args._worker:
        gpu_environment()
        worker(args.output.resolve(), args._worker, args._lock_fd)
        return
    if args.check_only or not (args.smoke or args.run_full):
        from .evaluator import verify_manifest
        manifest = verify_manifest(source)
        print('FORWARD_CHECK_PASS', manifest['frames'], 'frames/method; CPU-only; no inference', flush=True)
        return
    environment = gpu_environment()
    out = (args.output or source / ('forward_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    try:
        manifest = prepare(source, out, args.smoke, environment)
        print('PREPARED_FORWARD', manifest['frames'], 'frames/method', out, flush=True)
        if not args.smoke:
            update_source_report(source, out)
        launch(out)
    except Exception:
        if out.exists():
            write(out / 'forward_failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
