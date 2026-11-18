"""Scene-level statistics and tails; repeats never masquerade as new scenes."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np


def distribution(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                p95=float(np.percentile(values, 95)) if values else None,
                p99=float(np.percentile(values, 99)) if values else None,
                maximum=max(values) if values else None)


def paired_log_ci(rows, repeats, samples=2000, seed=20261030):
    """CI for mean scene peak delta. Average repeats, resample whole logs.

    Scenes within a log and repeated inference on the same scene are correlated.
    These CIs concern a mean of scene peaks, not the maximum/worst-case bound.
    """
    paired = defaultdict(dict)
    for row in rows:
        paired[row['workload'], row['scene_id'], row['repeat']][row['method']] = row
    by_scene = defaultdict(list)
    metrics = ['peak_gpu_allocated_GB', 'peak_cpu_rss_GB', 'incremental_peak_cpu_rss_GB']
    for (workload, sid, repeat), methods in paired.items():
        if set(methods) != {'native', 'worldstream'}:
            continue
        a, b = methods['native'], methods['worldstream']
        for metric in metrics:
            by_scene[workload, sid, b['log_id'], metric].append(b[metric]-a[metric])
    groups = defaultdict(list)
    for (workload, sid, log, metric), values in by_scene.items():
        if len(values) != repeats:
            raise ValueError('Missing repeat in scene pairing: '+str((workload, sid, metric)))
        groups[workload, metric].append((log, statistics.mean(values)))
    result = []
    rng = np.random.default_rng(seed)
    for (workload, metric), values in sorted(groups.items()):
        logs = defaultdict(list)
        for log, value in values:
            logs[log].append(value)
        keys = sorted(logs)
        bootstrap = []
        if len(keys) >= 2:
            for _ in range(samples):
                chosen = rng.integers(0, len(keys), len(keys))
                bootstrap.append(statistics.mean(v for index in chosen for v in logs[keys[index]]))
        result.append(dict(workload=workload, metric=metric, mean_scene_peak_delta=statistics.mean(v for _, v in values),
            ci95_low=float(np.percentile(bootstrap, 2.5)) if bootstrap else None,
            ci95_high=float(np.percentile(bootstrap, 97.5)) if bootstrap else None,
            distinct_scenes=len(values), log_clusters=len(keys), repeats_averaged=repeats,
            bootstrap_samples=len(bootstrap), estimand='Mean of scene peak WorldStream-minus-Native; repeated scenes averaged first'))
    return result


def extend(root, manifest, runs, frames):
    from .report import csv_write, stats, fmt
    from .run import write
    root = Path(root)
    scene_meta = {s['scene_id']: s for s in manifest['scenes']}
    frame_groups = defaultdict(list)
    for f in frames:
        frame_groups[f['method'], f['workload'], f['repeat'], f['scene_id']].append(f)
    scene_rows, scene_modules, baselines = [], [], []
    for run in runs:
        key = run['method'], run['workload'], run['repeat']
        records = json.loads((root/'runs'/f'{key[1]}_r{key[2]:02d}_{key[0]}'/'scene_resources.json').read_text())
        if len(records) != len(scene_meta) or {r['scene_id'] for r in records} != set(scene_meta):
            raise ValueError('Scene resource receipts do not match manifest')
        for record in records:
            sid = record['scene_id']
            local = frame_groups[key+(sid,)]
            if len(local) != record['frames']:
                raise ValueError('Scene frames differ from receipt')
            peak = record['end']
            sample = record['sampling']
            base = run['baseline']
            row = dict(method=key[0], workload=key[1], repeat=key[2], scene_id=sid,
                log_id=scene_meta[sid]['log_id'], frames=len(local),
                selection_reason=scene_meta[sid].get('selection_reason'),
                peak_gpu_allocated_GB=peak['gpu_peak_allocated_bytes']/1e9,
                peak_gpu_reserved_GB=peak['gpu_peak_reserved_bytes']/1e9,
                incremental_peak_gpu_GB=max(0, peak['gpu_peak_allocated_bytes']-base['gpu_allocated_bytes'])/1e9,
                peak_cpu_rss_GB=sample['rss_peak']/1e9,
                average_cpu_rss_GB=(sample['rss_time_integral']/sample['sampled_seconds'] if sample['sampled_seconds']
                                    else sample['rss_sum']/sample['samples'])/1e9,
                peak_cpu_uss_GB=sample['uss_peak']/1e9 if sample['uss_peak'] is not None else None,
                incremental_peak_cpu_rss_GB=max(0, sample['rss_peak']-base['cpu_rss_bytes'])/1e9,
                incremental_cpu_rss_from_scene_start_GB=max(0, sample['rss_peak']-record['start']['cpu_rss_bytes'])/1e9,
                nvml_sampled_peak_GB=sample['nvml_peak']/1e9 if sample['nvml_peak'] is not None else None)
            scene_rows.append(row)
            for component in ('OGSM', 'DAVP', 'QCSE', 'Total'):
                metric_values = defaultdict(list)
                for f in local:
                    for metric in ('cpu_bytes', 'gpu_bytes', 'entries', 'elements'):
                        value = (sum(f[c+'_'+metric] for c in ('OGSM', 'DAVP', 'QCSE')) if component == 'Total'
                                 else f[component+'_'+metric])
                        label = metric.removesuffix('_bytes')+'_MB' if metric.endswith('_bytes') else metric
                        metric_values[label].append(value/1e6 if metric.endswith('_bytes') else value)
                for metric, values in metric_values.items():
                    scene_modules.append(dict(method=key[0], workload=key[1], repeat=key[2], scene_id=sid,
                        log_id=scene_meta[sid]['log_id'], component=component, metric=metric, **distribution(values)))
        base, warm = run['baseline'], run['warmup_end']
        baselines.append(dict(method=key[0], workload=key[1], repeat=key[2],
            stable=run['cpu_baseline_stable'], model_rss_GB=base['cpu_rss_bytes']/1e9,
            model_uss_GB=base['cpu_uss_bytes']/1e9 if base['cpu_uss_bytes'] is not None else None,
            model_rss_uss_same_read=base['rss_uss_same_read'],
            warmup_rss_GB=warm['cpu_rss_bytes']/1e9,
            warmup_uss_GB=warm['cpu_uss_bytes']/1e9 if warm['cpu_uss_bytes'] is not None else None,
            incremental_peak_rss_from_model_GB=run['metrics']['incremental_peak_cpu_rss_bytes']/1e9,
            incremental_peak_rss_from_warmup_GB=run['metrics']['incremental_peak_cpu_rss_from_warmup_bytes']/1e9,
            full_read_duration_ms=base['sample_duration_ms']))
    # Tails are calculated within each independent run; SD is over repeats.
    # Frame-end process memory tails never reuse cumulative CUDA peak columns.
    tail_groups = defaultdict(list)
    for f in frames:
        values = {k: f[k]/1e9 if f[k] is not None else None for k in
                  ('gpu_allocated_bytes', 'gpu_reserved_bytes', 'cpu_rss_bytes', 'cpu_uss_bytes')}
        values = {'process_'+k.removesuffix('_bytes')+'_GB': v for k, v in values.items()}
        for c in ('OGSM', 'DAVP', 'QCSE', 'Total'):
            for metric in ('cpu_bytes', 'gpu_bytes', 'entries', 'elements'):
                v = sum(f[k+'_'+metric] for k in ('OGSM', 'DAVP', 'QCSE')) if c == 'Total' else f[c+'_'+metric]
                label = metric.removesuffix('_bytes')+'_MB' if metric.endswith('_bytes') else metric
                values[c+'_'+label] = v/1e6 if metric.endswith('_bytes') else v
        for workload in (f['workload'], 'ALL'):
            for metric, v in values.items():
                tail_groups[f['method'], workload, f['repeat'], metric].append(v)
    tail_repeats = defaultdict(list)
    for (method, workload, repeat, metric), values in tail_groups.items():
        tail_repeats[method, workload, metric].append(distribution(values))
    tails = []
    for (method, workload, metric), repeats in sorted(tail_repeats.items()):
        row = dict(method=method, workload=workload, metric=metric,
                   scope='Frame-end process snapshots / same-frame deduplicated retained module states')
        for reduction in ('mean', 'p95', 'p99', 'maximum'):
            row.update({reduction+'_'+k: v for k, v in stats([r[reduction] for r in repeats]).items()})
        tails.append(row)
    # Scene units are collapsed over repeats before reporting coverage variation.
    scene_groups = defaultdict(list)
    for row in scene_rows:
        scene_groups[row['method'], row['workload'], row['scene_id']].append(row)
    scene_units = []
    for (method, workload, sid), rows in sorted(scene_groups.items()):
        if len(rows) != manifest['config']['repeats']:
            raise ValueError('Incomplete scene repeats')
        row = dict(method=method, workload=workload, scene_id=sid, log_id=rows[0]['log_id'])
        for metric in ('peak_gpu_allocated_GB', 'peak_cpu_rss_GB', 'incremental_peak_cpu_rss_GB'):
            row[metric] = statistics.mean(r[metric] for r in rows)
        scene_units.append(row)
    ci = paired_log_ci(scene_rows, manifest['config']['repeats'], manifest['config'].get('bootstrap_samples', 2000),
                       manifest['config']['seed'])
    between_scenes = defaultdict(list)
    for row in scene_units:
        for metric in ('peak_gpu_allocated_GB', 'peak_cpu_rss_GB', 'incremental_peak_cpu_rss_GB'):
            between_scenes[row['method'], row['workload'], metric].append(row[metric])
    scene_distributions = [dict(method=method, workload=workload, metric=metric,
        scope='Distinct scene peaks after averaging repeated runs', **distribution(values))
        for (method, workload, metric), values in sorted(between_scenes.items())]
    for name, rows in [('scene_memory.csv', scene_rows), ('scene_module_memory.csv', scene_modules),
                       ('scene_mean_over_repeats.csv', scene_units), ('tail_memory.csv', tails),
                       ('baseline_quality.csv', baselines), ('paired_scene_ci.csv', ci),
                       ('scene_distribution.csv', scene_distributions)]:
        csv_write(root/name, rows)
    write(root/'extended_statistics.json', dict(scene_resources=scene_rows, scene_module_distributions=scene_modules,
        frame_tails=tails, repeated_scene_units=scene_units, paired_log_cluster_confidence_intervals=ci,
        scene_peak_distributions=scene_distributions,
        baseline_quality=baselines,
        definitions=dict(bootstrap='Resample original log clusters, keeping scenes together; repeats averaged first',
            scene_gpu='Actual CUDA scene high-water marks; reset counters only, keep allocated state/caches',
            tails='Per-run frame snapshot empirical percentiles, then mean/sample SD across repeats',
            total='Sum globally deduplicated attribution within the same frame before any reduction')))
    text = ['# 扩样评测：场景分布与尾部内存', '',
        f"实际场景数 {len(scene_meta)}；日志数 {len({s['log_id'] for s in scene_meta.values()})}；负载 {manifest['config']['workloads']}。",
        '旧18场景全部保留。短日志不足3场景时，剩余名额分配给其他可用日志；具体规则和全部候选特征见 scene_selection.json。', '',
        '| Method | Workload | Metric | Frame Mean ± SD | P95 ± SD | P99 ± SD | Max |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in tails:
        if row['workload'] == 'ALL' or row['metric'] not in ('process_gpu_allocated_GB', 'process_cpu_rss_GB',
                    'QCSE_cpu_MB', 'QCSE_elements', 'Total_cpu_MB', 'Total_gpu_MB'):
            continue
        def cell(label):
            return 'N/A' if row[label+'_mean'] is None else f"{row[label+'_mean']:.4f} ± {row[label+'_std']:.4f}"
        text.append(f"| {row['method']} | {row['workload']} | {row['metric']} | {cell('mean')} | {cell('p95')} | {cell('p99')} | {row['maximum_peak']:.4f} |")
    text += ['', '进程列是帧结束时的快照分布，不能称为瞬时峰值分布。逐场景真正的 CUDA peak 和后台 RSS peak 见 scene_memory.csv。',
        '表中 SD 是重复间波动；scene_mean_over_repeats.csv 先平均同场景重复，才形成场景样本。', '',
        '| Method | Workload | Scene peak metric | Mean | P95 | P99 | Max | Distinct scenes |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in scene_distributions:
        text.append(f"| {r['method']} | {r['workload']} | {r['metric']} | {r['mean']:.4f} | {r['p95']:.4f} | {r['p99']:.4f} | {r['maximum']:.4f} | {r['n']} |")
    text += ['',
        '| Workload | Metric | Mean scene peak delta (GB) | Log-cluster bootstrap 95% CI | Scenes / Logs |',
        '|---|---|---:|---|---|']
    for r in ci:
        interval = 'N/A' if r['ci95_low'] is None else f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
        text.append(f"| {r['workload']} | {r['metric']} | {r['mean_scene_peak_delta']:.4f} | {interval} | {r['distinct_scenes']} / {r['log_clusters']} |")
    text += ['', '区间的对象是“场景峰值差的平均值”，不是全局最大值，也不证明未见场景的上界。日志内相关性按日志聚类处理；短日志名额重分配导致的覆盖偏差仍需考虑。',
        'CPU baseline 达到平台只证明当前进程短时稳定，不保证不同进程 allocator 驻留量相同。主表 Incremental 始终相对模型加载基线；另在 baseline_quality.csv 报告相对预热后基线的敏感性结果。',
        '新旧协议不同，不将旧18场景结果直接混入本次均值/SD；新运行中的原18场景与新增场景可通过 selection_reason 区分。', '']
    (root/'extended_report.md').write_text('\n'.join(text))
    return dict(scene_count=len(scene_meta), log_clusters=len({s['log_id'] for s in scene_meta.values()}),
                tail_statistics='tail_memory.csv', scene_resources='scene_memory.csv',
                paired_ci='paired_scene_ci.csv', report='extended_report.md')
