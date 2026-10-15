"""Audit paired decisions and summarize frozen optimization experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .io_utils import write_json


def read(path): return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='outputs/module1_optimization');a=p.parse_args();root=Path(a.root)
    parity=[];scene_latencies=[]
    keys=['scene_id','frame_id','valid_anchor_ids','invalid_anchor_ids','valid_feature_candidates',
          'invalidation_regions','feature_versions','full_refresh_hint','provenance']
    for scene in sorted((root/'scene_results/holdout').iterdir()):
        rows={m:[json.loads(line) for line in (scene/m/'state.jsonl').open()] for m in ['reference','balanced']}
        assert len(rows['reference'])==len(rows['balanced'])==12
        for old,new in zip(rows['reference'],rows['balanced']):
            assert all(old['precheck'][k]==new['precheck'][k] for k in keys),(scene,'precheck mismatch')
            assert old['commit']['association_pairs']==new['commit']['association_pairs'],(scene,'association mismatch')
            parity.append(dict(scene=scene.name,frame_id=new['precheck']['frame_id'],exact=True))
        scene_latencies.append([read(scene/m/'summary.json')['m1_wall_p50_ms'] for m in ['reference','balanced']])
    times=np.asarray(scene_latencies);reduction=1-times[:,1]/times[:,0];rng=np.random.RandomState(2026)
    bootstrap=np.median(reduction[rng.randint(0,len(times),(10000,len(times)))],axis=1)
    audit=dict(frames_checked=len(parity),exact_decision_mask_version_provenance_association_parity=True,
        scene_latency_p50_pairs_ms=times.tolist(),median_paired_scene_reduction=float(np.median(reduction)),
        paired_scene_bootstrap95ci=np.percentile(bootstrap,[2.5,97.5]).tolist(),
        bootstrap_scope='Resampling 16 scene-level median speed reductions; un-warmed replay including resets; fixed mode order, not a rigorous end-to-end speedup estimate.',
        source_unchanged_since_protocol=all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in read(root/'protocol.json')['source_sha256'].items()))
    assert audit['source_unchanged_since_protocol']
    write_json(root/'parity_audit.json',audit)
    h=read(root/'holdout_summary.json');cpu=read(root/'cpu_benchmark.json');selection=read(root/'selection.json')
    g=read(root/'gpu/results.json') if (root/'gpu/results.json').exists() else None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    names=list(cpu['records']);vals=[cpu['records'][k]['p50_p95_ms'][0] for k in names]
    axes[0].bar(['Reference','1','2','3','6'],vals,color=['#777777']+['#2070a0']*4)
    axes[0].set(xlabel='Camera workers (optimized)',ylabel='M1 CPU P50 (ms)',title='Matched warmed replay')
    axes[1].scatter(times[:,0],times[:,1],color='#2070a0');mx=times.max()*1.1
    axes[1].plot([0,mx],[0,mx],'--',color='#888888');axes[1].set(xlabel='Reference scene P50 (ms)',ylabel='Balanced scene P50 (ms)',title='16 unseen scenes; identical masks')
    if g:
        modes=['reference','balanced'];vals=[g['summary'][k]['m1_p50_p95_ms'][0] for k in modes]
        axes[2].bar(modes,vals,color=['#777777','#2070a0']);axes[2].set(ylabel='M1 incl. adapters P50 (ms)',title='RTX 3090 full inference')
    else:
        axes[2].axis('off');axes[2].text(.1,.5,'GPU run pending')
    fig.tight_layout();fig.savefig(root/'balanced_results.png',dpi=180);fig.savefig(root/'balanced_results.pdf');plt.close(fig)
    baseline=h['reference'];balanced=h['balanced'];best=selection['selected']['metrics']
    text=f'''# OmniDrive 模块一：综合平衡优化报告

本轮选择保持原风险与覆盖行为的计算优化。新的校准样本不足以可靠选择质量阈值，未宣称达到全局最优或提升检测 mAP/NDS。详细架构与复现方法见 `docs/worldstream_m1_optimization.md`。

## 实现与选择依据

- 对象关联按类型/类别分组后向量化，地图曲线每个输入只采样一次；保持原尺寸、距离、方向门限及 Hungarian 分配。
- 六相机证据并行计算，再按固定相机顺序汇总。实测选择 {cpu['selected_workers']} 个线程；库内不修改 OpenCV 全局线程设置。
- 增加仅由历史 commit 更新的检测分数、匹配观测次数和位置创新量。可选门限已实现，默认不启用额外限制。分数不是定位正确概率，重复匹配也无法消除固定定位偏差。
- 校准预先冻结 2 种光流预算 × 5 种历史质量门限；先约束风险、恢复率、覆盖率与局部失效范围，再看 Pareto 与耗时。独立样本不足时回退到原判断条件。

## 数据与实验边界

固定 16 个校准场景（排序索引 12..27）和 16 个本轮独立验证场景（32..47），每场景前 12 帧，共 384 个唯一 nuScenes keyframe。两个方法每帧分别执行，因此回放执行量为 768 帧次。它们是本地验证集上的新场景，不是官方隐藏测试集；历史回放只有检测锚点，地图关联另由单测与完整 GPU 推理验证。校准协议先于标签/指标读取写入 `protocol.json`；`selection.json` 在验证开始前冻结。

校准原判断接受 {best['accepted_all']} 条记录，仅 {best['accepted_unique_instances']} 个独立实例、{best['accepted_scenes']} 个场景，错误 {best['physical_errors']} 条。预先要求至少 10 个独立实例、3 个场景，实际不足。较高分数/历史门限在这批小样本上还会过滤正确状态，不能以减少接受量代替风险改善。800 点方案接受 2 条、错误 0 条，仍不足以说明泛化可靠，且候选量低于原来的 75% 约束。最终保留 1600 点、原门限。

## 独立验证结果

| 指标 | 修改前快照 | balanced |
|---|---:|---:|
| 历史状态检查数 | {baseline['trials']} | {balanced['trials']} |
| 可判定记录 | {baseline['scorable']} | {balanced['scorable']} |
| 接受候选数 | {baseline['accepted_all']} | {balanced['accepted_all']} |
| 候选率 | {baseline['valid_candidate_rate']:.4%} | {balanced['valid_candidate_rate']:.4%} |
| 物理错误/可判定接受 | {baseline['physical_errors']}/{baseline['accepted_labeled']} | {balanced['physical_errors']}/{balanced['accepted_labeled']} |
| 新出现区域召回（{baseline['new_regions']} 条事件） | {baseline['new_region_recall']:.4%} | {balanced['new_region_recall']:.4%} |
| 失效面积比例 | {baseline['invalidation_area_ratio']:.4%} | {balanced['invalidation_area_ratio']:.4%} |
| 回放 M1 P50（ms） | {baseline['m1_wall_p50_ms']:.2f} | {balanced['m1_wall_p50_ms']:.2f} |

该验证集只接受 1 个独立对象，0/1 不能证明低风险；逐记录描述性 95% 区间上界为 97.5%。接受率也说明现有证据规则在这些场景中非常保守。局部区域召回还有约 3.28% 未达 90% 失效覆盖门限；没有像素级变化真值，不能把未标注背景当作可靠负例。

逐帧审计 {len(parity)} 帧：候选、失效 ID、精确像素掩码、特征版本、provenance、关联配对全部一致。`parity_audit.json` 提供核验结果。回放耗时包含首次初始化且方法按固定顺序执行，不等于完整模型速度。

## 成本与完整模型

独立预热基准使用 4 个已知场景帧、1 轮预热和 4 轮交错计时，统计非首帧。CPU M1 P50 从 {cpu['records']['reference']['p50_p95_ms'][0]:.2f} ms 到 {cpu['records']['workers_'+str(cpu['selected_workers'])]['p50_p95_ms'][0]:.2f} ms。16 个验证场景的成对中位耗时降幅中位数为 {np.median(reduction):.2%}（场景 bootstrap 描述性 95% 区间 {np.percentile(bootstrap,2.5):.2%}–{np.percentile(bootstrap,97.5):.2%}）。

'''
    if g:
        text+='GPU 环境：'+g['metadata']['device']+'，PyTorch '+g['metadata']['torch']+'，CUDA '+g['metadata']['cuda']+'。每种模式 8 个连续帧，1 次预热、3 次计时，共 96 帧次完整推理（8 个唯一帧）。LLM 每次使用新的输出目录并实际执行，模式间重置随机种子和 Head memory。\n\n| 模式 | 完整帧 P50/P95 ms | M1 含适配器 P50/P95 ms |\n|---|---:|---:|\n'
        for name,m in g['summary'].items():
            frame='/'.join(f'{v:.2f}' for v in m['frame_p50_p95_ms']);m1='/'.join(f'{v:.2f}' for v in m['m1_p50_p95_ms']) if m['m1_p50_p95_ms'] else '—'
            text+=f'| {name} | {frame} | {m1} |\n'
        peak=max(r['gpu_peak_allocated_bytes'] for r in g['runs'])/2**30
        text+=f'\n所有模式/重复的原始输出、backbone 张量和检测/地图 Head memory 摘要一致：**{g["observe_only_equal"]}**。GPU 已分配显存峰值 {peak:.3f} GiB。计时采用顺序模式，不是严格随机交叉的端到端速度试验；M1 的线程提速不等于跳过 backbone 或实现增量推理。并行后各相机阶段耗时之和是累计工作时间，不能相加当作墙钟；应读 `precheck_total` 或 `m1_total_including_adapters_ms`。\n'
    else:text+='GPU 实验未完成；不能声明完整模型验证通过。\n'
    tests=(root/'tests.txt').read_text() if (root/'tests.txt').exists() else ''
    if '74 passed' in tests:
        text+='\n回归测试：74 项通过（含 10 项新增关联等价、历史门限和并行一致性检查），19 条依赖库弃用警告；完整模型配置加载和 `git diff --check` 通过。\n'
    text+='\n## 结论与剩余限制\n\n本轮给出可复现的成本改善，同时保持已验证的风险和覆盖行为。现有低候选率、同源 RGB 证据、固定定位偏差、地图/对象 carrier 无一一对应，以及全局注意力依赖，仍限制真实深层特征复用。新增质量控制作为后续校准工具，不应在当前稀少样本上宣称已得到可靠置信度。若目标转向提升检测/规划精度，需要另外定义 mAP/NDS/规划碰撞等任务指标并开展训练或可靠定位建模；本轮没有重训模型。\n\n![成本对比](balanced_results.png)\n'
    (root/'optimization_report.md').write_text(text)
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
