"""Rebuild a source-anchored audit before changing any execution algorithm."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def stats(values):
    return dict(count=len(values), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95))) if values else None


def main(output):
    dest = Path(output)
    if (dest / 'audit_manifest.json').exists():
        raise ValueError('Audit already frozen; choose a fresh output')
    sources = [Path('outputs/module3_refinement') / name for name in (
        'performance_results.json', 'quality_results.json', 'state_feasibility.json',
        'admission_review.json', 'multi_20261017_005514/frames.jsonl',
        'single_20261017_005514/frames.jsonl')]
    sources += [Path('outputs/module2_rearchitecture') / name for name in (
        'pipeline_profile.json', 'feasibility_raw.json', 'real_reuse_coverage.json')]
    sources += [Path('outputs/module1_optimization/holdout_summary.json')]
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    all_rows, breakdown, bounds = {}, {}, {}
    for workload, p in zip(('single', 'multi'), (sources[5], sources[4])):
        rows = [json.loads(line) for line in p.open()]
        rows = [r for r in rows if not r['warmup']]
        all_rows[workload] = rows
        base = [r for r in rows if r['mode'] == 'baseline']
        breakdown[workload] = dict(source=str(p), evidence='historical_real_gpu',
            frames=len(base), unique_frames=len({r['token'] for r in base}),
            e2e_wall_ms=stats([r['wall_ms'] for r in base]),
            components={k: stats([r['components'][k]['gpu_ms'] for r in base])
                        for k in ('eva', 'detection', 'map', 'prefill', 'decode')},
            note='CUDA intervals include launch gaps; projections are nested in heads. '
                 'Do not add component percentiles or call intervals kernel time.')
        # Amdahl-style counterfactual on each measured frame, then aggregate.
        # It is an ideal bound, never an actual or quality-admitted speedup.
        values = {}
        for stage in ('decode', 'prefill', 'eva'):
            residual = [max(1e-9, r['wall_ms']-r['components'][stage]['gpu_ms']) for r in base]
            values['remove_' + stage] = dict(
                residual_wall_ms=stats(residual),
                aggregate_time_ratio=sum(r['wall_ms'] for r in base)/sum(residual),
                assumption='remove entire measured interval, including its launch gaps; zero replacement cost')
        maintained = [r for r in rows if r['mode'] == 'm123']
        if maintained:
            residual = [r['wall_ms']-r['m1'].get('m1_total_including_adapters_ms', 0) for r in maintained]
            values['remove_m1'] = dict(residual_wall_ms=stats(residual),
                aggregate_time_ratio=sum(r['wall_ms'] for r in maintained)/sum(residual),
                assumption='Changes persistent-state service semantics; hypothetical bound only')
        bounds[workload] = values
    feasibility = json.loads(sources[7].read_text())
    state = json.loads(sources[2].read_text())
    write(dest/'audit_manifest.json', dict(head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), sources=manifest,
        evidence='Historical measurements only; new code has not been measured by these runs'))
    write(dest/'baseline_profile.json', breakdown)
    write(dest/'latency_breakdown.json', dict(workloads=breakdown,
        historical_kernel_profile=json.loads(sources[6].read_text()),
        idealized_bounds=bounds, visual_conditional_bounds=feasibility['upper_bounds']))
    write(dest/'state_consumption.json', dict(evidence='historical_real_gpu', **state))
    write(dest/'reuse_coverage.json', dict(evidence='historical_real_gpu',
        **json.loads(sources[8].read_text())))
    lines = ['# 系统瓶颈审计（算法修改前）', '',
        '来源为已有真实 GPU 原始逐帧记录；SHA256 见 audit_manifest.json。'
        '这些记录不代表本次新代码的性能。单问题 Sampling 与五问题 Greedy 分开。', '',
        '| 工作负载 | E2E P50/P95 ms | Decode interval P50 ms | Prefill interval P50 ms |',
        '|---|---:|---:|---:|']
    for name, row in breakdown.items():
        wall = row['e2e_wall_ms']
        lines.append(f"| {name} | {wall['p50']:.2f}/{wall['p95']:.2f} | "
                     f"{row['components']['decode']['p50']:.2f} | {row['components']['prefill']['p50']:.2f} |")
    lines += ['', '首要热点为 Decode。完整消除单问题 Prefill 也只是理想上限；'
        '同帧只有一个问题，不能从同帧共享 Prefix 获益。理想上限按每帧残余时间计算后汇总，'
        '没有相加组件 P50。', '',
        'EVA 前两层的受控理想节省仅 %.2f ms，尚未包含认证和缓存；'
        '真实移动场景 Camera/Window 跳算为零。Camera 上限仅是条件性收益，'
        '不能外推到真实场景。' % feasibility['upper_bounds'][0]['saving_ms'], '',
        'M1 延长历史实验拒绝原因：`%s`。任务激活仍为零；M1→M3 No-Go，'
        '保留可靠性条件。' % json.dumps(state['extended_history']['verification_rejections'], ensure_ascii=False), '',
        'Prefix 与 Batch 均未通过冻结的完整 Token 等价标准；Auto 必须回退。'
        '不继续堆叠精度补丁或 Window Kernel 优化。优先处理拒绝路径成本、真实来源记录、'
        '跨模块失效及公平的单问题后端评估。', '',
        '新增插桩文件：tools/system_optimization/audit.py（只读历史证据并输出审计）。'
        '完整预处理、H2D/D2H、分配及 launch-gap 的本轮数值需新的 GPU trace；未知值不填零。']
    (dest/'workload_audit.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/system_optimization')
    main(parser.parse_args().output)
