"""Summarize only completed inference receipts; never synthesizes missing data."""
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics

from .run import read, write


def stats(values):
    values = [v for v in values if v is not None]
    return dict(mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else (0. if values else None),
                peak=max(values) if values else None, n=len(values),
                std_defined=len(values) > 1)


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def unit_metrics(values, scale, suffix):
    return {k.removesuffix('_bytes') + suffix if k.endswith('_bytes') else k:
            v / scale if k.endswith('_bytes') and v is not None else v for k, v in values.items()}


def flatten(prefix, values):
    return {prefix + key + '_' + stat: val for key, group in values.items() for stat, val in group.items()}


def overall_by_repeat(rows, keys):
    """Worst workload peak per repeat; equal workload weight for mean/baseline."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    for identity, group in grouped.items():
        merged = dict(zip(keys, identity))
        metric_keys = set().union(*(r['metrics'] for r in group))
        merged['metrics'] = {}
        for key in metric_keys:
            values = [r['metrics'].get(key) for r in group if r['metrics'].get(key) is not None]
            peak = 'peak' in key or key.endswith('_temporal_max') or key.endswith('_final')
            merged['metrics'][key] = (max(values) if peak else statistics.mean(values)) if values else None
        result.append(merged)
    return result


def summarize(root):
    root = Path(root)
    manifest = read(root/'manifest.json')
    results, frames, module_runs = [], [], []
    expected = {(m, w, r) for m in manifest['config']['methods']
                for w in manifest['config']['workloads'] for r in range(manifest['config']['repeats'])}
    found = set()
    receipts = {}
    for path in sorted((root/'runs').glob('*/result.json')):
        run = read(path)
        if run['status'] != 'COMPLETED':
            continue
        identity = (run['method'], run['workload'], run['repeat'])
        if identity in found:
            raise ValueError('Duplicate independent run: ' + str(identity))
        found.add(identity)
        local_frames = [json.loads(line) for line in (path.parent/'frames.jsonl').read_text().splitlines() if line]
        if len(local_frames) != run['frames'] or len(local_frames) != manifest['frame_count']:
            raise ValueError('Frame count mismatch: ' + str(path))
        expected_frames = [(s['scene_id'], t, fi) for s in manifest['scenes'] for fi, t in enumerate(s['tokens'])]
        if [(f['scene_id'], f['frame_id'], f['frame_index']) for f in local_frames] != expected_frames:
            raise ValueError('Chronological frame selection differs from manifest')
        results.append(run)
        for f in local_frames:
            receipt = {k: f[k] for k in ('questions', 'camera_ids', 'input_digest', 'prompt_digest',
                                        'request_seeds', 'image_shape', 'input_dtype')}
            key = (run['workload'], run['repeat'], f['scene_id'], f['frame_id'])
            if key in receipts and receipts[key] != receipt:
                raise ValueError('Unpaired inputs/questions: ' + str(key))
            receipts[key] = receipt
            flat = {k: v for k, v in f.items() if k != 'modules'}
            for module in f['modules']:
                flat.update({module['component'] + '_' + k: v for k, v in module.items() if k != 'component'})
            frames.append(flat)
        for component in ('OGSM', 'DAVP', 'QCSE', 'Total'):
            snapshots = []
            for f in local_frames:
                if component == 'Total':
                    keys = ('cpu_bytes', 'gpu_bytes', 'estimated_python_bytes', 'cpu_storage_bytes', 'entries', 'elements')
                    snapshot = {k: sum(m.get(k, 0) for m in f['modules']) for k in keys}
                else:
                    snapshot = next(m for m in f['modules'] if m['component'] == component)
                snapshots.append(snapshot)
            baseline = (dict(cpu_bytes=sum(m['cpu_bytes'] for m in run['baseline_modules']),
                             gpu_bytes=sum(m['gpu_bytes'] for m in run['baseline_modules'])) if component == 'Total'
                        else next(m for m in run['baseline_modules'] if m['component'] == component))
            keys = {k for s in snapshots for k, v in s.items() if isinstance(v, (int, float)) and k != 'enabled'}
            metrics = {}
            for key in keys:
                vals = [s.get(key, 0) for s in snapshots]
                label = key.removesuffix('_bytes') + '_MB' if key.endswith('_bytes') else key
                scale = 1e6 if key.endswith('_bytes') else 1
                metrics[label + '_temporal_mean'] = statistics.mean(vals) / scale
                metrics[label + '_temporal_max'] = max(vals) / scale
                metrics[label + '_final'] = vals[-1] / scale
            for key in ('cpu_bytes', 'gpu_bytes'):
                metrics[key.removesuffix('_bytes') + '_incremental_peak_MB'] = max(0, max(s[key] for s in snapshots) - baseline[key]) / 1e6
            module_runs.append(dict(method=run['method'], workload=run['workload'], repeat=run['repeat'],
                                    component=component, metrics=metrics))
    if found != expected:
        raise ValueError('Incomplete experiment; missing=' + str(sorted(expected-found)) + ', unexpected=' + str(sorted(found-expected)))
    # Pipeline, precision, device, and generation contract must also agree.
    common = None
    for r in results:
        config = {k: r['runtime'][k] for k in ('pipeline', 'parameter_dtypes', 'fp16', 'batch_size', 'physical_gpu', 'gpu', 'generation')}
        if common is not None and config != common:
            raise ValueError('Methods used different preprocessing/precision/device/generation')
        common = config
    run_rows = [dict(method=r['method'], workload=r['workload'], repeat=r['repeat'],
                     metrics=unit_metrics(r['metrics'], 1e9, '_GB')) for r in results]
    all_runs = overall_by_repeat(run_rows, ['method', 'repeat'])
    for r in all_runs:
        r['workload'] = 'ALL'
    overall_modules = overall_by_repeat(module_runs, ['method', 'repeat', 'component'])
    for r in overall_modules:
        r['workload'] = 'ALL'
    summaries, breakdowns = [], []
    for source, keys, destination in [(run_rows+all_runs, ['method', 'workload'], summaries),
                                      (module_runs+overall_modules, ['method', 'workload', 'component'], breakdowns)]:
        groups = defaultdict(list)
        for r in source:
            groups[tuple(r[k] for k in keys)].append(r['metrics'])
        for identity, group in sorted(groups.items()):
            metrics = {key: stats([r.get(key) for r in group]) for key in sorted(set().union(*group))}
            destination.append(dict(zip(keys, identity), metrics=metrics))
    paired = defaultdict(list)
    paired_source = {(r['method'], r['workload'], r['repeat']): r['metrics'] for r in run_rows + all_runs}
    for (method, workload, repeat), metrics in paired_source.items():
        native = paired_source.get(('native', workload, repeat))
        if method == 'worldstream' and native is not None:
            paired[workload].append({k: metrics[k]-native[k] if metrics[k] is not None and native[k] is not None else None
                                     for k in metrics})
    deltas = [dict(workload=workload, metrics={key: stats([r.get(key) for r in group])
              for key in sorted(set().union(*group))}) for workload, group in sorted(paired.items())]
    csv_write(root/'summary.csv', [dict(method=r['method'], workload=r['workload'], **flatten('', r['metrics'])) for r in summaries])
    csv_write(root/'per_frame_memory.csv', frames)
    csv_write(root/'module_memory_breakdown.csv', [dict(method=r['method'], workload=r['workload'],
        component=r['component'], **flatten('', r['metrics'])) for r in breakdowns])
    csv_write(root/'paired_overhead.csv', [dict(workload=r['workload'], **flatten('', r['metrics'])) for r in deltas])
    summary = dict(status='COMPLETED', manifest=manifest, independent_runs=results, summary=summaries,
        module_breakdown=breakdowns, paired_worldstream_minus_native=deltas, input_pairing_verified=True,
        statistics=dict(std='Sample standard deviation across independent repeat-level values (ddof=1); n=1 std=0 is undefined, std_defined=false',
            overall='ALL: per repeat take maximum across workloads for peak/final metrics; average other metrics equally across workloads; then mean/std/max across repeats',
            module='Per run temporal mean/max/final; across-repeat sample statistics. Total is same-snapshot globally deduplicated sum, then temporal reduction.',
            cpu_memory='Exact tensor/numpy storage + Estimated Memory for Python objects; opaque native internals may be undercounted',
            gpu_peak='CUDA allocator high-water marks; NVML is interval-sampled process usage, not guaranteed instantaneous peak'))
    write(root/'summary.json', summary)
    render_report(root, summary)
    plot(root, summaries, breakdowns, frames)
    if manifest.get('protocol_version', 1) >= 2:
        from .extended_statistics import extend
        summary['extended_statistics'] = extend(root, manifest, results, frames)
        write(root/'summary.json', summary)
        (root/'结果.md').write_text((root/'report.md').read_text()+'\n'+(root/'extended_report.md').read_text())
    print('SUMMARIZED', root)


def fmt(value, unit=''):
    if not value or value['mean'] is None:
        return 'N/A'
    return f"{value['mean']:.4f} ± {value['std']:.4f}{unit} (peak {value['peak']:.4f}; n={value['n']})"


def render_report(root, result):
    manifest = result['manifest']
    text = ['# WorldStream 资源开销评测', '',
        f"运行创建时间：{manifest['created_at']}。Smoke：{manifest['smoke']}。每种负载独立重复次数：{manifest['config']['repeats']}。",
        f"源码版本：`{manifest['source_revision']}`；Git Commit：`{manifest['git_commit']}`；导出前 Commit：`{manifest['exported_original_commit']}`。",
        '', '单位使用十进制 GB/MB；表中为 Mean ± Sample SD，并列出重复实验中的 Peak。n=1 的 SD 不可估计，显示 0 仅为占位。',
        'ALL 在每次重复内取各负载的最大峰值，再统计重复间分布；基础占用及平均 RSS 对负载等权平均。', '',
        '| Method | Peak GPU Allocated (GB) | Incremental GPU (GB) | Peak CPU RSS (GB) | Incremental CPU RSS (GB) |',
        '|---|---:|---:|---:|---:|']
    for row in result['summary']:
        if row['workload'] == 'ALL':
            m = row['metrics']
            text.append('| ' + ('Native VLM' if row['method'] == 'native' else 'WorldStream') + ' | ' + ' | '.join(fmt(m[k]) for k in (
                'peak_gpu_allocated_GB', 'incremental_peak_gpu_allocated_GB', 'peak_cpu_rss_GB', 'incremental_peak_cpu_rss_GB')) + ' |')
    text += ['', '模块表采用每次运行的逐帧平均，再在负载及重复间聚合。CPU 包含 Estimated Memory；共享存储只计入第一个拥有者（OGSM → DAVP → QCSE）。',
        '模块 Total 从同一帧的去重总量计算。world_state/window/global/candidate 子集单列于 CSV，不能再次加到 Total。', '',
        '| Component | CPU Memory (MB) | GPU Memory (MB) | Cache Entries | Cache Elements |',
        '|---|---:|---:|---:|---:|']
    for row in result['module_breakdown']:
        if row['method'] == 'worldstream' and row['workload'] == 'ALL':
            m = row['metrics']
            text.append('| ' + row['component'] + ' | ' + ' | '.join(fmt(m[k]) for k in (
                'cpu_MB_temporal_mean', 'gpu_MB_temporal_mean', 'entries_temporal_mean', 'elements_temporal_mean')) + ' |')
    text += ['', '## 分负载资源峰值与增量（GB）', '',
        '| Workload | Method | Peak GPU Allocated | Incremental GPU | Peak CPU RSS | Incremental CPU RSS |',
        '|---|---|---:|---:|---:|---:|']
    for row in result['summary']:
        if row['workload'] == 'ALL':
            continue
        text.append('| '+row['workload']+' | '+row['method']+' | '+' | '.join(fmt(row['metrics'][k]) for k in (
            'peak_gpu_allocated_GB', 'incremental_peak_gpu_allocated_GB', 'peak_cpu_rss_GB',
            'incremental_peak_cpu_rss_GB'))+' |')
    text += ['', '## 各负载基础与完整指标', '',
        '| Method | Workload | Baseline GPU (GB) | Peak Reserved (GB) | Baseline RSS (GB) | Average RSS (GB) |',
        '|---|---|---:|---:|---:|---:|']
    for row in result['summary']:
        m = row['metrics']
        text.append('| ' + row['method'] + ' | ' + row['workload'] + ' | ' + ' | '.join(fmt(m[k]) for k in (
            'baseline_gpu_allocated_GB', 'peak_gpu_reserved_GB', 'baseline_cpu_rss_GB', 'average_cpu_rss_GB')) + ' |')
    text += ['', '## 采集范围与限制', '',
        '- 模型加载完成后记录 baseline；数据集、预处理及后续状态属于增量。预热使用相同负载的首场景前缀，随后按主实验执行正常场景 reset；不调用 empty_cache，不修改状态预算。',
        '- CUDA 同步后读 allocated/reserved，预热后仅重置一次峰值统计；另保留包含加载/预热的进程 GPU 峰值。CPU 后台采样和逐帧快照取峰值，并用 Linux ru_maxrss 保留整个进程生命周期峰值。',
        '- RSS 采样均值含定时及帧边界样本；USS 低频读取，帧值为最近一次 USS；采样峰值可能漏掉短暂尖峰。NVML 不可用时为 null，不能解释成 0。',
        '- OGSM 包含世界锚点、特征、历史图像、掩码、验证结果和提交快照；DAVP 包含实际窗口缓存、全局锚点/残差及规划器/执行器缓存。',
        '- QCSE 按实际 AnswerMemory 的 Token ID、历史回答/世界快照、统计及绑定元数据测量；persistent KV=0 源自当前实现。单次生成的临时 KV 由 CUDA 峰值涵盖。',
        '- Entries/Elements：OGSM 为锚点+特征记录/锚点；DAVP 为窗口槽位+全局锚点/窗口槽位（跨缓存层）；QCSE 为历史回答条目/历史回答 Token 数。Total 的计数是异构单位之和。',
        '- Tensor 使用真实 storage 字节数，NumPy 追踪底层 owner；Python 用 sys.getsizeof 递归估算。模型参数/缓冲区 storage 被排除，nn.Module 不递归；C 扩展内部堆可能低估。',
        '- 逐帧 accounting_ms 记录统计成本。输入摘要复制和 Python 统计本身也可能影响进程峰值；此实验不作为时延基准。',
        '- summary.json 保留基础/预热/最终资源、采样健康状态、所有配置、源文件及输入配置 SHA256。每帧原始输入与 Prompt 摘要已经成对核验。', '']
    for r in result['independent_runs']:
        sampling = r['sampling']
        text.append(f"- 采集状态 {r['method']}/{r['workload']}/r{r['repeat']}: NVML={sampling['nvml_error'] or 'available'}; errors={sampling['errors']}")
    if manifest.get('protocol_version', 1) >= 2:
        text += ['', '## 协议 v2 的采集口径', '',
            '- 上述通用说明中的旧采样口径由本段更新：帧/基线 RSS 和 USS 来自同次 memory_full_info，未复用旧 USS；单次读取并非跨内核文件的原子快照，保留读取时长。',
            '- 模型加载后 gc.collect 回收已无引用的加载对象，等待 RSS/USS 达到配置的平台阈值；未达到时失败并保留 trace。不调用 malloc_trim、empty_cache，不裁剪任何 live 模块状态。',
            '- 新进程使用固定的 CPU allocator 环境（见 runtime.json），这是统一的实验设置，不保证不同进程的模型基线数值完全一致。',
            '- 平均 RSS 为采样间隔的时间加权积分。USS 的后台采样低频，但帧快照强制 fresh full read。',
            '- CUDA peak 计数器每场景重置，整次运行 peak 为所有场景 peak 的最大值；只重置计数器，保留 allocator 和模块实际状态。',
            '- 预热后另记基线及其 CPU 增量用于敏感性分析，主表仍相对模型加载基线。逐场景真峰值、帧快照 P95/P99、按日志聚类的场景差值置信区间见 extended_report.md。']
    (root/'report.md').write_text('\n'.join(text) + '\n')


def plot(root, summaries, breakdowns, frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#4676A9', '#D78B43']
    overall = [r for r in summaries if r['workload'] == 'ALL']
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    for ax, title, metrics in zip(axes, ['GPU memory', 'CPU resident memory'], [
            ['baseline_gpu_allocated_GB', 'peak_gpu_allocated_GB', 'incremental_peak_gpu_allocated_GB'],
            ['baseline_cpu_rss_GB', 'peak_cpu_rss_GB', 'incremental_peak_cpu_rss_GB']]):
        x = np.arange(3)
        for i, row in enumerate(overall):
            ax.bar(x + (i-(len(overall)-1)/2)*.35, [row['metrics'][m]['mean'] for m in metrics],
                   yerr=[row['metrics'][m]['std'] for m in metrics], width=.33, capsize=2,
                   label='Native VLM' if row['method'] == 'native' else 'WorldStream', color=colors[i % 2])
        ax.set_xticks(x, ['Baseline', 'Peak', 'Incremental'])
        ax.set_ylabel('Memory (GB)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(root/f'resource_comparison.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    modules = [next(r for r in breakdowns if r['component'] == c and r['method'] == 'worldstream'
                    and r['workload'] == 'ALL') for c in ('OGSM', 'DAVP', 'QCSE')] if any(r['method'] == 'worldstream' for r in breakdowns) else []
    if modules:
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
        for ax, metric, title in zip(axes, ['cpu_MB_temporal_mean', 'gpu_MB_temporal_mean'],
                                     ['CPU state (includes Estimated Memory)', 'GPU retained state']):
            ax.bar([r['component'] for r in modules], [r['metrics'][metric]['mean'] for r in modules],
                   yerr=[r['metrics'][metric]['std'] for r in modules], capsize=3,
                   color=['#4676A9', '#70A591', '#D78B43'])
            ax.set_ylabel('Memory (MB)')
            ax.set_title(title, fontsize=9)
            ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(root/f'module_cache_breakdown.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    # First independent repeat only, one line per actual workload/method.
    groups = defaultdict(list)
    for f in frames:
        if f['repeat'] == 0:
            groups[f['method'], f['workload']].append(f)
    for (method, workload), rows in sorted(groups.items()):
        for ax, metric in zip(axes, ['gpu_allocated_bytes', 'cpu_rss_bytes']):
            ax.plot(range(len(rows)), [r[metric]/1e9 for r in rows], label=method + '/' + workload, lw=1)
            ax.set_xlabel('Chronological frame (scene boundaries retained)')
    axes[0].set_ylabel('Frame-end GPU allocated (GB)')
    axes[1].set_ylabel('Frame-end CPU RSS (GB)')
    axes[1].legend(fontsize=5, ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(root/f'memory_over_frames.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
