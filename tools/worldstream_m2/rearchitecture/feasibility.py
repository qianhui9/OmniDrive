"""Stage decisions from executed stage A and preserved M1 cost evidence."""
import json
from .common import ROOT,write


def main():
    raw=json.loads((ROOT/'feasibility_raw.json').read_text())
    previous=json.loads(open('outputs/module2_refinement/summary.json').read())
    m1=previous['real_optimized']['m1_ms'][0]
    rows=[]
    for r in raw['upper_bounds']:
        s=r['saving_ms'];rows.append(dict(scope=r['scope'],saving_upper_bound_wall_ms=s,
            validation_ms=m1,validation_source='preserved_previous_four_scene_M1_P50',
            maximum_possible_net_ms=s-m1,go_under_full_m1=s>m1,
            cache_maintenance='included only output assembly in idealized bound; actual runtime measured later',
            measured_reuse_timing=r['timing'],kind=r['kind']))
    write(ROOT/'cost_break_even.json',dict(rows=rows,m1_previous_p50_ms=m1,
        camera_fraction_lower_bound=m1/raw['full']['wall_ms'],
        notes='Linear camera coverage ratio is a model inference, not an admission guarantee. Measured subset latencies are used for actual decisions.'))
    table='\n'.join(f"| {r['scope']} | {r['saving_upper_bound_wall_ms']:.2f} | {m1:.2f} | {r['maximum_possible_net_ms']:.2f} |" for r in rows)
    (ROOT/'feasibility_report.md').write_text(f'''# 阶段 A：收益上限与可行性

RTX 3090、原始权重、相同 FlashAttention；3 次预热、10 次测量。Full EVA wall P50 {raw['full']['wall_ms']:.2f} ms。
逐层 CUDA interval、CPU enqueue wall P50/P95 在 feasibility_raw.json；CPU enqueue 不代表同步完成时间。

| 理想复用边界 | 计算节省上限 ms | 历史 M1 P50 ms | 扣除 M1 后最大余量 ms |
|---|---:|---:|---:|
{table}

上表是 Idealized Upper Bound，直接使用已算当前 oracle 的中间输出，并未认证历史状态；
不把这些数当成实际系统收益。正式 cache/equality/同步成本由后续执行器另测。
仅跳两层即使覆盖 100% 也不能支付现有同步 M1，Gate A 为 No-Go；保留 Packing 组件，停止继续微调。
按线性相机模型，覆盖至少约 {m1/raw['full']['wall_ms']:.1%} 才可能支付 M1，实际须用相机子批次成本复核。
跳到 Global 之后的更多层不具有局部精确条件，不进入运行时。相机之间没有跨 batch attention，
因此将 D1 作为条件候选；不同 batch shape 的数值质量仍需审计，不据数学独立直接批准。

完整视觉 heads、carrier、LLM prefill/decode 分解将补充在 pipeline_profile.json；
本报告决策先由 backbone 的实际收益上限得出，M3 不在本轮修改范围内。
''')
    numerical=json.loads((ROOT/'numerical_raw.json').read_text())['cases']
    write(ROOT/'reuse_boundary_analysis.json',dict(window_go=False,camera_candidate=True,
        independent_cameras='EVAViT treats camera as batch; all attention reshapes preserve batch sequences',
        dense_packed_max_abs=max(r['B_vs_A']['max_abs'] for r in numerical),
        sparse_max_abs=max(r['C_vs_A']['max_abs'] for r in numerical),
        dense_microbatch_max_abs=max(r['camera_microbatch_vs_A']['max_abs'] for r in numerical),
        gate_B='no promotion without downstream frozen-target checks',gate_C='pending actual camera tests'))

if __name__=='__main__':main()
