"""Render WR-CVMS evidence and enforce the pretraining cost gate."""
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import numpy as np
from tools.worldstream_m123.benchmark.common import write, sha256
from tools.worldstream_m123.compact_visual_memory import ROOT, OLD, dataset, stats, carrier_path


def paired_scene_ci(values):
    """Bootstrap whole scene means, not the correlated per-request repeats."""
    x=np.asarray(list(values),dtype=float)
    rng=np.random.RandomState(20261028)
    sample=x[rng.randint(len(x),size=(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=[float(v) for v in np.percentile(sample,[2.5,97.5])],
                scenes=len(x),method='10000 paired scene bootstrap samples; three cost scenes, exploratory only')


def run(root):
    coverage=json.loads((root/'real_memory_coverage.json').read_text())
    if coverage['status']!='REAL_FEATURE_ACTUAL_PREFIX_VERIFIED':raise ValueError('Actual prefix verification required')
    cost=json.loads((root/'compact_prefix_cost.json').read_text())
    rows=[json.loads(s) for s in (root/'controlled_cost/records.jsonl').read_text().splitlines()]
    overhead=json.loads((root/'controlled_cost/overheads.json').read_text())
    bridge={}
    for name in ('natural','enriched'):
        _,frames,_,_=dataset(root,name)
        records=[dict(scene_id=r['scene_id'],frame_id=r['frame_id'],event=r['compact_bridge']['event'],
            baseline_eager_bridge_wall_ms=r['bridge_total_wall_ms'],**r['compact_bridge']['costs']) for r in frames]
        keys=sorted(set(k for r in records for k,v in r.items() if isinstance(v,(int,float))))
        bridge[name]=dict(records=records,overall={k:stats([r.get(k,0.) for r in records]) for k in keys},
            by_event={event:{k:stats([r.get(k,0.) for r in records if r['event'].split(':')[0]==event]) for k in keys}
                      for event in ('no_memory','create','keep','revoke')},
            comparison_scope='Same newly computed EVA/M1 frame, eager then lazy path. Stream completion synchronization included; not independent service runs.',
            raw_capture_metadata_note='Executed run inherited legacy memory_or_LLM_tokens_constructed=False even when the added compact manager created RAW memory. compact_bridge.event records actual creations/keeps and is authoritative. The source now corrects that legacy metadata flag; original measured JSONL retained unchanged. No LLM projection/generation occurred in the live bridge audit.',
            timing_scope='Host wall. memory_create/keep/invalidation and bridge_total are inclusive; do not sum nested fields.',
            zero_cost_fields='Zero denotes an instrumented stage not entered. GPU sync D2H is inside creation; completion_sync is an explicit measuring barrier.',
            qualification_and_geometry_thresholds_unchanged=True)
    stage_probe=json.loads((root/'memory_stage_probe.json').read_text()) if (root/'memory_stage_probe.json').exists() else None
    write(root/'bridge_cost_breakdown.json',dict(status='COMPLETED',**bridge,
        exclusive_memory_and_projection_stage_probe=stage_probe,
        retained_version_check_note='Included in live memory_keep. Separately instrumented in memory_stage_probe.json; not a zero-time claim and not double charged.'))
    paired=defaultdict(dict)
    for row in rows:
        if row['K']==8:paired[(row['scene_id'],row['question_index'],row['repeat'])][row['kind']]=row
    differences=defaultdict(list)
    by_scene=defaultdict(lambda:defaultdict(list))
    for (scene,_,_),p in paired.items():
        for label,a,b,metric in (
            ('B_minus_C_wall','compact_zero','compact_exact','executor_wall_ms'),
            ('A_minus_C_wall','native','compact_exact','executor_wall_ms'),
            ('stable8_only_wall','compact_text_only','compact_exact','executor_wall_ms'),
            ('stable8_only_cuda','compact_text_only','compact_exact','model_prefill_cuda_ms'),
            ('A_minus_C_cuda','native','compact_exact','model_prefill_cuda_ms')):
            value=p[a][metric]-p[b][metric]
            differences[label].append(value);by_scene[scene][label].append(value)
    effects={label:dict(request_pair_distribution=stats(values),
        paired_scene=paired_scene_ci([np.mean(s[label]) for s in by_scene.values()])) for label,values in differences.items()}
    natural=coverage['natural']['8']
    q=natural['memory_coverage'];p=natural['one_question_hit_probability']
    cold=float(np.mean([r['cold']['wall_ms'] for r in overhead if r['K']==8]))
    warm=float(np.mean([r['warm']['wall_ms'] for r in overhead if r['K']==8]))
    prep=float(np.mean([r['expanded_preparation_ms'] for r in rows if r['K']==8]))
    plan=float(np.mean([r['planner_wall_ms'] for r in rows if r['K']==8]))
    no_bridge=float(np.mean([r['complete_bridge_wall_ms'] for r in bridge['natural']['records'] if r['event']=='no_memory']))
    create_bridge=float(np.mean([r['complete_bridge_wall_ms'] for r in bridge['natural']['records'] if r['event'].startswith('create:')]))
    keep_bridge=float(np.mean([r['complete_bridge_wall_ms'] for r in bridge['natural']['records'] if r['event']=='keep']))
    # Every cache capture/branch/validation is already inside executor wall.
    # Planner includes its matcher: matcher_wall must not be charged again.
    hit_raw=effects['A_minus_C_wall']['paired_scene']['mean']
    native_minus_zero=float(np.mean([r['native']['executor_wall_ms']-r['compact_zero']['executor_wall_ms'] for r in paired.values()]))
    gh=hit_raw-warm-prep-plan-keep_bridge
    gm=native_minus_zero-cold-prep-plan-create_bridge
    net=p*gh+(q-p)*gm-(1-q)*no_bridge
    denominator=gh-gm
    required=((1-q)*no_bridge-q*gm)/denominator if denominator>0 else None
    all_memory_required=-gm/denominator if denominator>0 else None
    net_scene=[]
    for scene,s in by_scene.items():
        scene_miss=float(np.mean([r['native']['executor_wall_ms']-r['compact_zero']['executor_wall_ms']
            for (sid,_,_),r in paired.items() if sid==scene]))-cold-prep-plan-create_bridge
        net_scene.append(p*(float(np.mean(s['A_minus_C_wall']))-warm-prep-plan-keep_bridge)
               +(q-p)*scene_miss-(1-q)*no_bridge)
    net_ci=paired_scene_ci(net_scene)
    write(root/'break_even_analysis.json',dict(status='COMPLETED',natural_memory_coverage=q,actual_hit_probability=p,
        incremental_hit_net_ms=gh,memory_miss_net_ms=gm,no_memory_extra_ms=no_bridge,
        expected_net_ms_per_single_question_frame=net,paired_scene_net=net_ci,
        P_break_even=required,maximum_hit_probability_at_observed_coverage=q,
        break_even_status='NO_FINITE_PROBABILITY' if gh<=0 and gm<=0 else 'SOLVED' if required is not None else 'NO_POSITIVE_HIT_ADVANTAGE',
        break_even_null_reason='No finite break-even: both the memory hit and miss have nonpositive measured net gain.' if gh<=0 and gm<=0 else None,
        break_even_feasible_at_observed_coverage=required is not None and 0<=required<=q,
        all_requests_have_memory_P_break_even=all_memory_required,
        formula='p*g_hit + (q-p)*g_miss - (1-q)*no_memory_extra',
        single_question_extra=dict(composer_cold_ms=cold,composer_keep_ms=warm,expanded_preparation_ms=prep,
            planner_including_matching_ms=plan,bridge_create_ms=create_bridge,bridge_keep_ms=keep_bridge),
        effects=effects,three_question_crossframe_hit_probability=natural['three_question_crossframe_opportunities']/natural['frames'],
        three_question_frame_net_optimistic_ms=3*(p*(hit_raw-prep-plan)+(q-p)*(native_minus_zero-prep-plan))
            -p*(warm+keep_bridge)-(q-p)*(cold+create_bridge)-(1-q)*no_bridge,
        three_question_scope='Upper model treating all three questions as isolated crossframe controls. Same-frame full-visual KV gain is never credited to world memory. Not measured E2E.',
        accounting='Executor wall already contains branch, validation, output KV capture. Planner already includes equality matching. Bridge includes M1 qualification and D2H. Expansion/composer outside executor are charged once. Source KV setup is optimistic/excluded; a negative result remains NO_GO.',
        limits='Three controlled scenes, one natural memory-bearing scene. Conditional costs are not a deployment population estimate. Decode and complete M1/M2 request service were not timed.'))
    go=net_ci['ci95'][0]>0 and gh>0 and required is not None and 0<=required<=q
    decision=dict(status='GO' if go else 'NO_GO',cost_gate_passed=bool(go),training_allowed=bool(go),
        expected_single_question_net_ms=net,optimistic_hit_net_ms=gh,observed_actual_prefix_probability=p,
        P_break_even=required,paired_scene_ci95=net_ci['ci95'],
        rationale='The measured net is negative; even a memory hit costs more than Native. No finite hit probability breaks even under these costs.' if gh<=0 and gm<=0 else 'Only train if measured scene-paired net lower bound is positive and observed coverage exceeds break-even.',
        quality='NOT_RUN',formal_E2E='NOT_RUN')
    write(root/'feasibility_decision.json',decision)
    if go:
        raise RuntimeError('GO requires conditional training workflow; do not mark task delivered yet')
    write(root/'deployment_decision.json',dict(status='NO_GO',enabled=False,trained_adapter=None,
        production_config_changed=False,reason='Pretraining cost gate failed. Retain native full visual deployment.'))
    write(root/'crossframe_reuse_summary.json',dict(status='CONTROL_ONLY',
        real_memory_coverage=natural['memory_coverage'],real_memory_hold_rate=natural['hold_rate_given_memory'],
        one_question_exact_opportunities=natural['one_question_exact_opportunities'],
        three_question_boundary_opportunities=natural['three_question_crossframe_opportunities'],
        stable_prefix_tokens=8,controlled_rows=len(rows),
        executed_control_visual_tokens=sum(r['controlled_crossframe_visual_tokens'] for r in rows),
        actual_control_skipped_prefill_tokens=sum(r['controlled_skipped_tokens'] for r in rows),
        scheduler_approved_rows=sum(r['scheduler_approved_tokens']>0 for r in rows),
        formal_crossframe_visual_KV_tokens=0,formal_skipped_prefill_tokens=0,formal_scheduler_approvals=0,
        trained_quality='NOT_RUN',formal_prefill_cuda_ms=None,decode_cuda_ms=None,
        formal_M3_wall_ms=None,formal_memory_composer_wall_ms=None,
        free_generation_E2E='NOT_RUN',equal_decode_steps_E2E='NOT_RUN',
        single_question_formal_E2E='NOT_RUN',three_question_formal_E2E='NOT_RUN',
        full_request_wall_ms=None,FPS=None,QPS=None,peak_control_gpu_bytes=cost['peak_gpu_allocated_bytes'],
        cache_bytes_max=max(r['cache_bytes'] for r in rows),
        note='Untrained control lengths are explicitly forced after exact matching, not policy admission. Existing Scheduler remains fail closed without a compatible cost/quality policy.'))
    texts={
      'adapter_training_report.md':f'# Adapter training\n\nNOT_RUN. Cost gate NO_GO; no optimizer, epochs, updates or trained checkpoint.\nTraining seconds: null; training peak memory: null.\nThe existing manifest is a source only; Core-Valid does not imply K8 eligibility.\nManifest SHA256: {sha256("outputs/m3_sf_wrcps_feasibility_run01/training_dataset_manifest.json")}.\nNo K64/128/256 training, no K8 expansion into more learned memory tokens.\n',
      'core_valid_quality_report.md':'# Memory subset quality\n\nNOT_RUN. No cost-qualified trained compact adapter. No teacher/student or task quality claim. Old semantic probes are not new LLM quality.\n',
      'quality_report.md':'# Quality\n\nTraining and memory-bearing quality: NOT_RUN. mAP/NDS/DET_l/Token F1: null. Spatial, motion, orientation, signals, counts and map QA: NOT_RUN.\nNo-memory real frozen LLM identity: see native_identity_cuda.json.\nCompression is lossy; information contribution and untouched slots do not establish semantic preservation. No final heldout data was used.\n',
      'compact_memory_design.md':'''# WR-CVMS compact candidate

One existing M3Runtime → ReusePlanner → PrefixMatcher → PrefixKVCache → PrefillExecutor path. Two representation modes: native_full_visual and world_compact_memory. Default deployed configuration unchanged.

K8 reads eight distinct real EVA cells after unchanged M1 independent identity qualification and current geometric exclusions. Deterministic farthest coverage combines normalized region position and normalized feature distance, only at creation. Historical IDs only control lifetime. No class embeddings, copies, learned token expansion, old kinematics or new encoder.

Current geometric support is checked every frame. Lazy RegionFeature views postpone GPU gathering until creation. No candidates short-circuit before scanning projections. Original raw observations and projected embedding objects are retained, not recomputed. Tensor pointer/version and source binding detect ordinary in-place mutation, tensor replacement, and metadata changes; projected embedding version also checked. This follows PyTorch versioned ownership, not protection against raw-storage/.data version-counter bypass. Source feature hash is computed once on creation; no repeated full-history D2H/hash on keep. PrefixMatcher still proves actual embeddings, positions and masks exactly. Revocation revisions never reset, so identical values cannot revive revoked visual KV.

Stable8 + Object252 + Ego1 + Map252 = 513. Four distributed fixed adjacent pairs merge within each current native query group using trainable positive weights. Removed query evidence contributes to its successor. Native order retained; untouched slots copied exactly. Queries have no certified object/pixel correspondence. 248 object and 248 map slots plus ego remain bitwise unchanged; eight destination slots are weighted merges. All 512 object/map slots contribute, but this is lossy and is not semantic equivalence. K16 is diagnostic only.

No-memory runtime bypasses the composer, casts, permutations and projection entirely, preserving native storage, layout, expansion and generation. No-placeholder stable tokens. Memory path requires a trained provenance-bound adapter with passed cost and quality gates; cost scaffold cannot load as deployment artifact. Cache lifetime binds scene/epoch/model/adapter/qualification/layout/memory revision.

Pretraining protocol: three scene pairs; three captured native questions; nine measured repeats over three alternating rounds after two warmups. Native513, Compact513 ZeroKV, Compact513 text+Stable8 ExactKV. Additional fixed-text-only control isolates the eight visual tokens from shared prompt reuse. One generated token means one prefill and zero decode forwards. No quality interpretation of untrained outputs. K16 only one enriched pair for coverage/cost diagnostics. Cache clones and capture are timed. GO requires positive paired-scene net lower bound and feasible measured hit probability; otherwise stop before training.
''',
      'performance_report.md':f'''# Controlled feasibility performance

Cost decision: NO_GO. Not a formal deployment or E2E benchmark.

K8 B−C executor wall: {effects['B_minus_C_wall']['paired_scene']['mean']:.4f} ms. K8 A−C executor wall: {hit_raw:.4f} ms. Stable8-only CUDA (text-only minus text+8): {effects['stable8_only_cuda']['paired_scene']['mean']:.4f} ms. These differ: fixed text savings cannot be attributed to world memory.

After extra costs, conditional hit net {gh:.4f} ms; natural coverage-weighted single-question net {net:.4f} ms/frame. Whole-scene bootstrap CI95: {net_ci['ci95']}. Three cost scenes are a bounded exploratory control, not a population confidence guarantee. Break-even probability {required}; maximum under observed coverage {q:.6f}; actual opportunity probability {p:.6f}.

No-memory bridge mean {no_bridge:.4f} ms; create bridge {create_bridge:.4f} ms; keep bridge {keep_bridge:.4f} ms. Composer cold {cold:.4f} ms, retained {warm:.4f} ms; preparation {prep:.4f} ms; planner including matching {plan:.4f} ms. All figures are newly measured. Component nesting is documented in bridge_cost_breakdown.json.

Natural coverage is 24 finite development frames across three logs, only one memory-bearing scene. Enriched scenes are availability-selected and cannot stand in for natural deployment coverage. Actual prefixes are checked on original native IDs/carriers for all frames, including no-memory/layout/scene transitions. No formal scheduler approvals are substituted with controlled reuse lengths.

Training, memory-bearing task quality, free generation A/B/C, equal-step decode E2E and final acceptance: NOT_RUN due to cost NO_GO. No M1/M2 full-request FPS/QPS claim. Investigate larger-compute reusable vision-language intermediate states with their own causal validity and accuracy gates; these results establish no such alternative speedup.
''',
      'result_answers.md':f'''# 本轮十六问

结论：NO_GO，停止高成本训练。结构与真实前缀成立，完整系统加速尚未成立。以下均为本轮新采集/测量结果；控制性跳过量不记为正式服务复用。

1. **K8 可合法创建吗？** 可以。15 个真实、唯一位置中选择 8 个；未放宽原 M1 身份资格、当前观测或几何排除门槛。
2. **寿命、持续次数和覆盖率？** 自然数据 24 帧/3 场景：创建 {natural['creates']} 次、保持 {natural['continuous_keeps']} 次、平均观察寿命 {natural['mean_lifetime_observed_frames']} 帧、覆盖 {q:.2%}。实际 PrefixMatcher 确认 {natural['one_question_exact_opportunities']}/24 次单问跨帧机会。富集结果见 real_memory_coverage.json，不代替自然覆盖。
3. **K16 有优势吗？** 自然场景覆盖为 0；15 个位置不能冒充 16 个。只做一个富集场景成本诊断，没有支持另开训练路线的证据。
4. **桥接开销下降多少？** 同帧配对 eager/lazy 均值：自然 {bridge['natural']['overall']['baseline_eager_bridge_wall_ms']['mean']:.4f}→{bridge['natural']['overall']['complete_bridge_wall_ms']['mean']:.4f} ms；富集 {bridge['enriched']['overall']['baseline_eager_bridge_wall_ms']['mean']:.4f}→{bridge['enriched']['overall']['complete_bridge_wall_ms']['mean']:.4f} ms。后者包含新增创建/保持管理，并未整体下降；不得宣传普遍桥接加速。
5. **重复同步与处理消除了吗？** 无候选时跳过投影扫描，保持时不 Gather、不 D2H、不 Clone、不重算历史 Hash/投影；创建时仍有一次真实区域 D2H、选择和 Hash。每帧仍复核当前几何。单独记录测量完成同步及版本检查；版本计数方案不防御绕过 PyTorch 计数器的 raw-storage/.data 写入。
6. **513 布局保留多少？** Stable8 + Object252 + Ego1 + Map252；497 个原生槽位逐值不变，8 个目的槽位合并当前信息，8 个移除源槽位在组内有贡献。全体 512 个 Object/Map 槽位参与，但合并有损，不等于全部信息可逆保留。
7. **有记忆时是否破坏动态语义？** NOT_RUN。成本门失败，未训练、未做有记忆任务质量实验；无法宣称没有破坏。
8. **Stable8 节省多少 CUDA？** 用“仅固定文本”对比“固定文本+Stable8”，平均 {effects['stable8_only_cuda']['paired_scene']['mean']:.4f} ms；场景级 CI 见 break_even_analysis.json。不是完整请求节省。
9. **扣除额外开销后是否有正收益？** 没有。自然覆盖加权模型为 {net:.4f} ms/帧，CI95 {net_ci['ci95']}。这是有限场景成本估算，未伪称实测完整请求 E2E。
10. **盈亏平衡需要多高命中率？** 当前命中和失配净收益均为负，没有有限可行命中概率；JSON 的 P_break_even=null、break_even_status=NO_FINITE_PROBABILITY。即便假设每次都有记忆并命中，当前测得路径仍不能盈亏平衡。
11. **自然场景达标吗？** 未达标。真实机会率 {p:.2%}，记忆覆盖 {q:.2%}；这些不是人群部署覆盖估计。
12. **Composer 训练成功了吗？** NOT_RUN。按前置门主动停止，未创建优化器、未训练任何 K 配置；没有训练 Adapter。
13. **训练后空间/运动/自然问答是否可靠？** NOT_RUN。mAP/NDS/DET_l/Token F1 均为 null。无记忆真实 LLM 的 Embedding、Position、Mask、同 RNG 生成 Token 严格相同，见 native_identity_cuda.json。
14. **正式 M3 跳过多少跨帧 Prefill？** 0。控制性 C 每次确实跳过 8 个视觉 Token，另有固定文本；原始 forward token 计数见 controlled_cost/records.jsonl。Scheduler 无相容质量/成本策略，未批准正式复用。
15. **B→C 和 A→C 是否稳定 E2E 加速？** 未成立。控制性 B−C Wall={effects['B_minus_C_wall']['paired_scene']['mean']:.4f} ms，A−C Wall={hit_raw:.4f} ms，均为负；正式自由生成及固定 Decode E2E 为 NOT_RUN。
16. **主要瓶颈及后续方向？** 8 个 Token 的 CUDA 收益太小，不能覆盖缓存、资格/几何、Composer 与输入准备成本；自然低覆盖进一步稀释收益。未测质量，不能把失败归因于语义压缩。建议先剖析计算占比更大的视觉—语言中间状态，再定义其因果依赖、世界关联、当前动态更新与失效边界；只有独立质量和净收益通过才探索复用。不能把 8 个真实观测扩成 64 个学习 Token 制造节省。
'''
    }
    for name,text in texts.items():(root/name).write_text(text)
    bindings={}
    pair_sources=[]
    for name in ('natural','enriched'):
        source,frames,_,qualification=dataset(root,name)
        scenes=set()
        for event in coverage[name]['8']['events']:
            if event['event']!='keep' or event['scene_id'] in scenes:continue
            scenes.add(event['scene_id'])
            pair_sources.append(dict(sample=name,scene_id=event['scene_id'],
                source_frame=event['generating_frame'],target_frame=event['frame_id'],
                selected_grid_indices=event['selected_grid_indices'],historical_memory_sha256=event['content_sha256'],
                qualification_sha256=qualification,source_carriers=str(carrier_path(source,event['generating_frame'])),
                target_carriers=str(carrier_path(source,event['frame_id']))))
    pair_sources=pair_sources[:3]
    for index,pair in enumerate(pair_sources):
        pair['pair']=index
        for key in ('source_carriers','target_carriers'):
            pair[key+'_sha256']=sha256(pair[key]);bindings[pair[key]]=pair[key+'_sha256']
    write(root/'controlled_pair_sources.json',dict(status='COMPLETED',pairs=pair_sources,
        selection='Exactly first K8 keep per scene in natural then enriched order, first three; same as executed cost tool.',
        untrained_scaffold_note='Cost controls construct untrained projection instances in the recorded seed/order. untrained_composer_8.pt retains the final K8 pair instance; K16 retains its single diagnostic pair. Earlier K8 parameter instances were overwritten by the executed tool, so no separate adapter-file SHA is claimed for them. No trained adapter exists.'))
    for name in ('natural','enriched'):
        source,_,_,_=dataset(root,name)
        for relative in ('experiment_plan.json','bridge_sequence.json','bridge_capture/region_features.pt','bridge_capture/frames.jsonl'):
            path=source/relative;bindings[str(path)]=sha256(path)
    for path in (Path('ckpts/iter_10548.pth'),Path('outputs/m3_sf_wrcps_feasibility_run01/training_dataset_manifest.json'),
                 Path('outputs/m3_sf_wrcps_optimization_run01/semantic_qualification.json')):
        bindings[str(path)]=sha256(path)
    for path in (root/'controlled_cost').glob('untrained_composer_*.pt'):bindings[str(path)]=sha256(path)
    for path in root.glob('executed_*.py'):bindings[str(path)]=sha256(path)
    for folder in ('worldstream/m3','tools/worldstream_m123'):
        for path in Path(folder).glob('*.py'):bindings[str(path)]=sha256(path)
    write(root/'source_and_artifact_bindings.json',dict(status='COMPLETED',sha256=bindings,
        base_commit='7e26bc10a6861caf29727817241f011e75b7b521',trained_adapter_sha256=None,
        source_note='New live captures; earlier CPU-only lifecycle used old enriched tensors while live capture was pending. Final coverage/cost/report bind only the new captures. Memory provenance guards were strengthened during live capture; M1/M2/geometry/selection numerics unchanged.'))
    diff=subprocess.check_output(['taskset','-c','0-3,6-31','git','diff','--unified=0','7e26bc10a6861caf29727817241f011e75b7b521','--','worldstream','tools','tests','.gitignore'],text=True)
    (root/'implementation.diff').write_text(diff)
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    run(parser.parse_args().root)
