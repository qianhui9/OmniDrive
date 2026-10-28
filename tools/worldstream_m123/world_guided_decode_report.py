"""Render measured WSG-SD evidence, retaining failed gates and NOT_RUN scopes."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import numpy as np
from tools.worldstream_m123.benchmark.common import write


def read(path):return json.loads(Path(path).read_text())
def mean(xs):return float(np.mean(xs)) if xs else None
def ratio(a,b):return a/b if b else None


def report(root):
    profile=read(root/'native_decode_profile.json')
    overlap=read(root/'historical_answer_overlap.json')
    bench=read(root/'decode_cuda_benchmark.json')
    sampling=read(root/'sampling_distribution_validation.json')
    numeric=read(root/'gpu_numeric_probe.json')
    schedule=read(root/'gpu_schedule_numeric_probe.json')
    regression=read(root/'gpu_regression.json')
    assert profile['status']==bench['status']==numeric['status']=='COMPLETED'
    assert regression['status']=='PASS'
    assert regression['weights_sha256']==profile['model_sha256']
    trials=bench['trials']
    assert len(trials)==len(read(root/'cuda_tasks.json')['tasks'])*3*3*2
    # The native loader prepends a random overview prompt. It is part of
    # native profiling, but must not inflate original dataset QA coverage.
    trials=[dict(t,workload=('native_template_'+t['workload']) if t['question_index']==0 else t['workload']) for t in trials]
    raw_events=[e for e in overlap['events'] if e.get('question_origin')=='dataset_conversation']
    raw_available=sum(e['exact_question'] for e in raw_events)
    groups=defaultdict(dict)
    for t in trials:
        groups[(t['contract'],t['workload'],t['round'],t['frame_id'],t['question_index'])][t['mode']]=t
    comparisons=[]
    for key,g in groups.items():
        assert set(g)=={'native','history4','history8'}
        for mode in ('history4','history8'):
            a,b=g['native'],g[mode]
            comparisons.append(dict(contract=key[0],workload=key[1],round=key[2],scene_id=a['scene_id'],
                frame_id=key[3],question_index=key[4],mode=mode,identical=a['tokens']==b['tokens'],
                native_tokens=a['generated_tokens'],spec_tokens=b['generated_tokens'],
                native_decode_wall_ms=a['decode_wall_ms'],spec_decode_wall_ms=b['decode_wall_ms'],
                native_decode_cuda_ms=a['decode_cuda_ms'],spec_decode_cuda_ms=b['decode_cuda_ms'],
                native_generation_wall_ms=a['wall_ms'],spec_generation_wall_ms=b['wall_ms'],
                native_calls=a['decode_forward_calls'],spec_calls=b['decode_forward_calls']))
    summaries=[];acceptance=[]
    for contract in ('greedy','sampling'):
        for workload in ('natural','fixed','native_template_natural','native_template_fixed'):
            for mode in ('history4','history8'):
                ps=[p for p in comparisons if (p['contract'],p['workload'],p['mode'])==(contract,workload,mode)]
                ts=[t for t in trials if (t['contract'],t['workload'],t['mode'])==(contract,workload,mode)]
                per_round=[]
                for i in range(3):
                    rr=[p for p in ps if p['round']==i]
                    per_round.append(ratio(sum(p['native_decode_wall_ms'] for p in rr),sum(p['spec_decode_wall_ms'] for p in rr)))
                summaries.append(dict(contract=contract,workload=workload,mode=mode,pairs=len(ps),
                    identical_pairs=sum(p['identical'] for p in ps),
                    decode_wall_speedup=ratio(sum(p['native_decode_wall_ms'] for p in ps),sum(p['spec_decode_wall_ms'] for p in ps)),
                    decode_cuda_speedup=ratio(sum(p['native_decode_cuda_ms'] for p in ps),sum(p['spec_decode_cuda_ms'] for p in ps)),
                    generation_wall_ratio_diagnostic=ratio(sum(p['native_generation_wall_ms'] for p in ps),sum(p['spec_generation_wall_ms'] for p in ps)),
                    per_round_decode_speedup=per_round,
                    native_tokens=sum(p['native_tokens'] for p in ps),spec_tokens=sum(p['spec_tokens'] for p in ps),
                    native_calls=sum(p['native_calls'] for p in ps),spec_calls=sum(p['spec_calls'] for p in ps),
                    peak_gpu_bytes=max(t['peak_gpu_bytes'] for t in ts),
                    selection_tuple_wall_ms=mean([t['draft_selection_wall_ms'] for t in ts]),
                    crop_host_wall_ms=mean([t['diagnostics'].get('kv_crop_host_ms',0) for t in ts])))
                logs=[l for t in ts for l in t['diagnostics']['verification_log']]
                acceptance.append(dict(contract=contract,workload=workload,mode=mode,
                    candidate_tokens=sum(len(l['proposed']) for l in logs),accepted_tokens=sum(l['accepted'] for l in logs),
                    verification_calls=len(logs),accepted_per_verification=mean([l['accepted'] for l in logs]),
                    rejection_rate=mean([l['rejection_position'] is not None for l in logs]),
                    rejection_positions=dict(Counter(str(l['rejection_position']) for l in logs if l['rejection_position'] is not None)),
                    first_verification_accepted=mean([t['diagnostics']['verification_log'][0]['accepted'] for t in ts]),
                    total_draft_tokens_accepted_per_request=mean([sum(l['accepted'] for l in t['diagnostics']['verification_log']) for t in ts])))
    forward_costs=defaultdict(list)
    for t in trials:
        for e in t['forward_events']:
            if not e['prefill']:forward_costs[e['tokens']].append(e['cuda_ms'])
    costs={str(k):dict(count=len(v),mean_cuda_ms=mean(v),median_cuda_ms=float(np.median(v))) for k,v in sorted(forward_costs.items())}
    write(root/'draft_acceptance_analysis.json',dict(status='COMPLETED_DIAGNOSTIC',summaries=acceptance,
        warning='Actual target verification, separated by contract. Batched FP16 is not qualified as native numerical equivalence. Accepted counts exclude target correction/bonus tokens.'))
    write(root/'decode_comparison_summary.json',dict(status='COMPLETED_DIAGNOSTIC',summaries=summaries,comparisons=comparisons,
        forward_costs=costs,warning='Generation wall A includes native multimodal assembly; B uses pre-expanded embeddings. This is not a fair complete M3 or M123 speedup. Decode spans are paired; output lengths and identity are reported explicitly.'))
    exact=all(p['identical'] for p in comparisons if p['contract']=='greedy')
    if exact:
        raise RuntimeError('New target passed greedy gate: conditional M1/quality/E2E work is required; do not automatically copy this failed-run conclusion')
    reason='Real RTX3090 native greedy token mismatch for both block lengths; batched numerical contract not qualified. Stop before complex M1 scheduling and formal sampling/E2E deployment.'
    write(root/'go_no_go.json',dict(status='NO_GO' if not exact else 'REQUIRES_FURTHER_VALIDATION',
        strict_greedy_gate=exact,reason=reason,history_decode_cuda_potential='MEASURED_DIAGNOSTIC_ONLY',
        original_dataset_greedy_pairs=dict(total=sum(p['contract']=='greedy' and p['workload'] in ('natural','fixed') for p in comparisons),
            identical=sum(p['contract']=='greedy' and p['workload'] in ('natural','fixed') and p['identical'] for p in comparisons)),
        sampling_transition_test=sampling['status'],production_sampling_distribution='NOT_QUALIFIED',
        m1_world_guidance='NOT_RUN',formal_m3_speedup=None,formal_m123_speedup=None,
        approximate_quality_acceptance='NOT_RUN',official_config_modified=False))
    notrun=dict(status='NOT_RUN',reason=reason,metrics=None)
    write(root/'world_delta_accuracy.json',notrun)
    write(root/'draft_scheduler_analysis.json',dict(status='HISTORY_ONLY',
        world_guided=notrun,natural_exact_available=raw_available,natural_requests=len(raw_events),
        native_loader_template=dict(questions=24,exact_available=overlap['exact_available']-raw_available),
        fixed_bounded_stream=dict(frames=9,questions=18,source_frame_questions=6,available_drafts=12,
            availability=12/18,scope='First three distinct frames per scene, first two original dataset questions. Native-loader overview templates reported separately. Available-target trials are conditional on history; first-frame outputs come from original profiling.'),
        full_history_management_cost=None,m1_diff_wall_ms=None,
        tuple_proposal_materialization_only=True))
    qs=[q for f in profile['frames'] if not f['warmup'] for q in f['requests']]
    pref=mean([q['prefill_cuda_ms'] for q in qs]);dec=mean([q['decode_cuda_ms'] for q in qs])
    raw_qs=[q for f in profile['frames'] if not f['warmup'] for q in f['requests'][1:]]
    native_summary=dict(prefill_cuda_ms=pref,decode_cuda_ms=dec,
        prefill_fraction=ratio(pref,pref+dec),decode_fraction=ratio(dec,pref+dec),
        decode_wall_ms=mean([q['decode_wall_ms'] for q in qs]),m3_wall_ms=mean([q['m3_wall_ms'] for q in qs]),
        generated_tokens=mean([q['generated_tokens'] for q in qs]),
        frame_wall_ms=mean([f['frame_wall_ms'] for f in profile['frames'] if not f['warmup']]),
        m1_wall_ms=mean([f['m1_wall_ms'] for f in profile['frames'] if not f['warmup']]),
        m2_wall_ms=mean([f['m2_wall_ms'] for f in profile['frames'] if not f['warmup']]),
        heads_wall_ms=mean([f['head_wall_ms'] for f in profile['frames'] if not f['warmup']]),
        original_dataset_qa=dict(questions=48,matched_drafts=raw_available,
            mean_prefill_cuda_ms=mean([q['prefill_cuda_ms'] for q in raw_qs]),
            mean_decode_cuda_ms=mean([q['decode_cuda_ms'] for q in raw_qs]),
            mean_generated_tokens=mean([q['generated_tokens'] for q in raw_qs])))
    native_summary.update(native_model_frame_fps=1000/native_summary['frame_wall_ms'],
        native_three_question_qps=3000/native_summary['frame_wall_ms'],
        peak_gpu_bytes=max(f['peak_gpu_bytes'] for f in profile['frames']))
    write(root/'native_profile_summary.json',native_summary)
    raw_pairs=[p for p in comparisons if p['contract']=='greedy' and p['workload'] in ('natural','fixed')]
    all_pairs=[p for p in comparisons if p['contract']=='greedy']
    raw_identical=sum(p['identical'] for p in raw_pairs)
    failed_total=sum(not p['identical'] for p in all_pairs)
    flipped=[e for e in schedule['events'] if e['serial_argmax']!=e['block_argmax']]
    if flipped:
        f=flipped[0]
        schedule_finding=(f"The first argmax disagreement in the mirrored schedule is at output index {f['predicted_output_index']}: "
            f"independent serial token {f['serial_argmax']} versus block-derived K/V token {f['block_argmax']}; "
            f"serial top-two margin {f['serial_top2_margin']}, max absolute logit difference {f['max_abs_logit_difference']}. "
            'This demonstrates a numerical decision difference with the same conditioned tokens and rejection crop.')
    else:
        schedule_finding='The mirrored schedule did not isolate an argmax flip; the failed free-running identity remains unresolved.'
    def accepted(contract,workload):
        return next(a for a in acceptance if (a['contract'],a['workload'],a['mode'])==(contract,workload,'history8'))
    def speed(contract,workload):
        return next(a for a in summaries if (a['contract'],a['workload'],a['mode'])==(contract,workload,'history8'))['decode_wall_speedup']
    table=['| Contract | Workload | K | Decode wall ratio A/B | CUDA ratio A/B | Greedy identical / pairs | Output tokens A / B |',
           '|---|---|---:|---:|---:|---|---|']
    for s in summaries:
        table.append(f"| {s['contract']} | {s['workload']} | {s['mode'][-1]} | {s['decode_wall_speedup']:.3f} | {s['decode_cuda_speedup']:.3f} | {str(s['identical_pairs']) if s['contract']=='greedy' else 'RNG differs'} / {s['pairs']} | {s['native_tokens']} / {s['spec_tokens']} |")
    text=f'''# WSG-SD decode evidence: NO_GO for exact deployment

Current base: `{profile['code_sha']}`; research branch `research/m3-world-guided-decode`. Actual RTX3090, unchanged checkpoint `{profile['model_sha256']}`. All 513 native visual tokens retained. No training or extra draft model.

Native instrumented 24-frame/72-question profile includes one built-in loader overview template and two original dataset questions per frame. Excluding one cold frame, the combined means are: prefill **{pref:.3f} ms**, decode forward CUDA **{dec:.3f} ms**; prefill/decode shares **{native_summary['prefill_fraction']:.2%}/{native_summary['decode_fraction']:.2%}** of their combined forward time. Decode wall **{native_summary['decode_wall_ms']:.3f} ms**, M3 question wall **{native_summary['m3_wall_ms']:.3f} ms**, generated tokens **{native_summary['generated_tokens']:.2f}**. Original dataset QA alone averages **{native_summary['original_dataset_qa']['mean_generated_tokens']:.2f} tokens**, prefill **{native_summary['original_dataset_qa']['mean_prefill_cuda_ms']:.3f} ms**, decode **{native_summary['original_dataset_qa']['mean_decode_cuda_ms']:.3f} ms**. Mean three-question frame **{native_summary['frame_wall_ms']:.3f} ms**. M1/M2/heads host wall means **{native_summary['m1_wall_ms']:.3f}/{native_summary['m2_wall_ms']:.3f}/{native_summary['heads_wall_ms']:.3f} ms**. These host durations can contain original synchronization; they are not isolated kernel times.

Original natural dataset QA strict question availability: **{raw_available}/{len(raw_events)} = {raw_available/len(raw_events):.2%}**. The native loader's 24 generated overview-template prompts have one further exact match; mixed-loader availability is 3/72. These are distinct workload scopes, not full-dataset coverage. Repeated-question application workload uses the two actual scene-first dataset questions on distinct subsequent images: bounded 9-frame/18-question stream, 12 history opportunities (**66.67%**). Native overview-template repetition is a separately labeled diagnostic and is not counted as original QA.

## Paired real decode cost gate

378 measured trials: 21 history-available question/frame tasks, 3 modes, 3 rotating orders, greedy and native sampling separately. These comprise two original natural QA opportunities, twelve original fixed-QA opportunities, and seven loader-template diagnostics, explicitly separated below. Fixed frames are the first two subsequent distinct frames per scene. Selection is by input opportunity, not measured outcome. Source drafts are prior real **sampled** native answers; they are never ground truth or the current answer.

'''+ '\n'.join(table)+'''

The table contains raw diagnostic ratios, including changed-output cases. It is **not evidence of exact acceleration**. Sampling output lengths differ naturally, and the known block-versus-serial numerical difference prevents declaring target-law equivalence on the deployed model. Per-round ratios, paired tokens, forward counts and raw logs are in `decode_comparison_summary.json` and `decode_trials.jsonl`.

Forward cost by input length is retained in the JSON. A nominal K=8 round normally forwards 9 tokens because it incorporates one pending correction/bonus; the initial round forwards 8. Likewise K=4 normally forwards 5. Compare actual costs, not theoretical counts. Rejection K/V cropping is a view operation; its host enqueue time is measured, not a fabricated isolated CUDA cost. Rejected suffix storage lives only until the next concatenation/request ends. Native decode uses no historical K/V. Cache capture is disabled equally for cost trials (zero persistent cache budget); tests separately verify owned-prefix lifecycle.

Generation-wall A includes native multimodal assembly while B has embeddings prepared before timing; therefore those generation ratios cannot qualify complete M3 speedup. Decode CUDA and post-prefill stream spans are the fair local comparison. Draft tuple materialization is recorded; full runtime metadata/hash management is not measured in this bounded executor gate.

## Decision

'''+reason+'''

M1 question-aware diff, world-guided scheduler, B→C comparisons, formal single-/three-question M123 trials and approximate task-quality acceptance are **NOT_RUN/null**. No complete M3 or M123 acceleration is claimed. Formal source configuration, M1/M2, EVA/heads, native visual carrier and pretrained weights are unchanged.

## Instrumentation changelog

| File | Change | Purpose |
|---|---|---|
| worldstream/m3/prefill_executor.py | modified | Native-forward counts/events and optional request-local history verifier |
| tools/worldstream_m123/world_guided_decode.py | new | Actual continuous native profiling, history audit and paired CUDA gate |
| tools/worldstream_m123/speculative_numeric_probe.py | new | Forced-prefix real-model causal/numeric audit |
| tools/worldstream_m123/speculative_sampling_validation.py | new | Actual small-Llama joint-law and residual boundary diagnostics |
| tools/worldstream_m123/world_guided_decode_checks.py | new | Serialized actual-GPU regression and witness |
| tools/worldstream_m123/world_guided_decode_report.py | new | Reproducible evidence aggregation, no new measurements |

The profiling launcher is independent of serving; ordinary native generation does not create CUDA timing events unless profile=True.
'''
    (root/'decode_performance_report.md').write_text(text)
    paired_failures=[p for p in comparisons if p['contract']=='greedy' and not p['identical']]
    (root/'speculative_verifier_correctness.md').write_text(f'''# Correctness gates

CPU tests and actual tiny-Llama CUDA tests cover causal verification, wrong/invalid proposals, first and middle rejection, all acceptance, EOS/stopping/length, processors, GenerationConfig, padding positions, owned prefill K/V, scene/version/question/expiry and multiple same-frame questions. Empty/invalid draft executes the original HF loop, preserving its seed behavior.

Actual large-model adversarial smoke cases: {read(root/'gpu_verifier_correctness.json')['status']}. Broader paired large-model greedy cases: **FAIL**, {len(paired_failures)} nonidentical pairs across both block configurations and three rounds. A smoke pass cannot override this failure. Raw first failure is preserved in `strict_gate_failure/`; repeated comparisons are in the final CUDA JSON.

Real forced-prefix probe: future-only suffix changes preserve earlier logits: **{numeric['future_suffix_changed_prefix_logits_identical']}**. Same forced prefix with serial vs block forwards has **{numeric['argmax_disagreements']}** argmax disagreements over {len(numeric['events'])} positions. Full logits deltas, margins and K/V differences are saved in `gpu_numeric_probe.json`. This isolates numerical differences from free-running answer drift; it does not authorize approximate deployment.

The additional exact-schedule probe mirrors the actual rejection/crop schedule and reports **{schedule['argmax_disagreements']}** argmax disagreement(s). {schedule_finding} This result must not be substituted by the earlier fixed-chunk probe.

Original dataset QA pairs alone pass {raw_identical}/{len(raw_pairs)} token comparisons in this sample. Across the whole native workload, {failed_total}/{len(all_pairs)} comparisons fail. Workload-specific counts remain in the comparison table. This post-capture input-provenance subgroup does not establish universal exactness or deployed sampling equivalence.

Speculative sampling q=delta derivation: `sampling_derivation.md`. Transition and real small-model joint-law diagnostic: {sampling['status']}. Deployed-model native numerical distribution equivalence: **NOT_QUALIFIED**. Formal sampling remains native.
''')
    (root/'e2e_performance_report.md').write_text('''# End-to-end status

Formal A/B/C E2E: **NOT_RUN**. M1-guided C: **NOT_RUN** because the actual-target exact verifier prerequisite failed. Single-question and three-question paired M123 acceptance: **NOT_RUN/null**. The original three-question native profile is measured, but has no equivalent full-pipeline B/C comparison. FPS/QPS speedups, B→C, A→C complete M3 and A→C M123 speedups: **null**.

Three rotating rounds were executed for the local same-carrier A/B decode diagnostic. They are not formal M123 rounds. No visual-prefill sharing is credited to WSG-SD, and no E2E speedup is inferred by adding local stage times or old experiments. Approximate deployment needs independent quality acceptance and complete request measurement; these were not run.
''')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('ckpts/pretrain_qformer',local_files_only=True,use_fast=False)
    diffs=[]
    for key,g in groups.items():
        if key[0]!='greedy' or key[2]!=0:continue
        for mode in ('history4','history8'):
            if g['native']['tokens']!=g[mode]['tokens']:
                diffs.append(dict(scene_id=g[mode]['scene_id'],frame_id=key[3],mode=mode,
                    question=g[mode]['question'],native=tokenizer.decode(g['native']['tokens'],skip_special_tokens=True),
                    speculative=tokenizer.decode(g[mode]['tokens'],skip_special_tokens=True)))
    write(root/'answer_divergences.json',dict(status='OBSERVED',cases=diffs,task_accuracy=None,
        scope='Explicit same-greedy-input output differences, not independently scored accuracy. No annotation answers used.'))
    quality='''# Quality status

Strict token identity fails on the actual pretrained model for both block lengths. Therefore the exact-model quality guarantee is not established, even though the ideal-arithmetic verifier rule and small-model distribution tests pass. User permission for some accuracy loss does not establish that this loss is acceptable; independent approximate accuracy acceptance and full M123 performance are NOT_RUN. No approximate deployment is enabled.

Changed greedy answers are reproduced below, rather than using text similarity to call them equal. Object/category, position, distance, motion, signal, count, road structure, safety and temporal semantic inheritance benchmark scores are NOT_RUN/null. M1 scheduling is not implemented after the prerequisite failure.
'''
    for d in diffs:
        quality+=f"\n## {d['frame_id']} / {d['mode']}\n\nQuestion: {d['question']}\n\nNative: {d['native']}\n\nSpeculative diagnostic: {d['speculative']}\n"
    (root/'quality_report.md').write_text(quality)
    answers=f'''# 最终问题逐项回答

1. 最新原生加载器混合流：Prefill CUDA {pref:.3f} ms/问题，Decode CUDA {dec:.3f} ms/问题，分别占两者之和 {native_summary['prefill_fraction']:.2%} 与 {native_summary['decode_fraction']:.2%}。24 帧/72 问，均值排除首个冷帧。严格原始数据集 QA 子集平均 {native_summary['original_dataset_qa']['mean_generated_tokens']:.2f} tokens，Prefill {native_summary['original_dataset_qa']['mean_prefill_cuda_ms']:.3f} ms、Decode {native_summary['original_dataset_qa']['mean_decode_cuda_ms']:.3f} ms；长概览模板不能冒充所有真实 QA 的长度。
2. 严格数据集自然 QA 历史草稿可用 {raw_available}/{len(raw_events)}（{raw_available/len(raw_events):.2%}）。原生加载器还插入了 24 个概览模板问题，额外产生一个匹配；混合流才是 3/72。样本范围仅三个开发 scene 的前两个真实 QA。
3. 固定重复的真实数据集问题有界流 9 帧/18 问，12 个可用历史草稿（66.67%）；它不代表自然 QA 覆盖率。加载器模板问题另列诊断。
4. K8 的实际目标验证平均接受：原始自然 QA Greedy {accepted('greedy','natural')['accepted_per_verification']:.3f} tokens/次、Sampling {accepted('sampling','natural')['accepted_per_verification']:.3f}；固定真实 QA Greedy {accepted('greedy','fixed')['accepted_per_verification']:.3f}、Sampling {accepted('sampling','fixed')['accepted_per_verification']:.3f}。它们排除了目标模型 bonus/correction，不是历史答案文本相似率。K4、每请求总草稿接受量及首失配位置完整保存在 draft_acceptance_analysis.json。
5. 实测单 Token Forward 平均 {costs['1']['mean_cuda_ms']:.3f} ms；首轮纯 8-token 验证 {costs['8']['mean_cuda_ms']:.3f} ms，通常含 pending 的 9-token 验证 {costs['9']['mean_cuda_ms']:.3f} ms（约单 Token 的 {costs['9']['mean_cuda_ms']/costs['1']['mean_cuda_ms']:.2f} 倍）。不同长度受缓存/算子形态影响，并非单调；原始事件全部保留。
6. 原始自然 QA 历史可用子集，Greedy K8 Decode wall 比值 A/B={speed('greedy','natural'):.3f}；固定真实 QA 为 {speed('greedy','fixed'):.3f}。Sampling 对应诊断比值 {speed('sampling','natural'):.3f}/{speed('sampling','fixed'):.3f}，但采样输出长度不同且真实数值分布未资格化，不能直接归为等价加速。原始 QA Greedy 子集 {raw_identical}/{len(raw_pairs)} Token 一致，全部原生问题仍有 {failed_total}/{len(all_pairs)} 失败，不能认定通用 Exact 加速。
7. 确定性草稿按 p(d) 接受，拒绝后采样去除 d 的残差。边界测试及实际小模型两 Token 联合分布测试 {sampling['status']}；部署模型批量/串行数值等价未通过，正式 Sampling 未启用。
8. 已审计：VerifiedWorldState 是感知提交前历史快照。真正当前状态应从已提交 store 读取并绑定当前帧/版本。实现与精度实验 NOT_RUN。
9. 候选依赖包括对象属性/关系、运动、车道；数量、否定、信号、安全及 UNKNOWN 缺乏完整覆盖。M1 贡献尚无实测，不能声称有效。
10. M1 对接受率的增量：NOT_RUN/null。
11. M1 对 Forward 次数的增量：NOT_RUN/null。
12. M1 diff 与完整历史管理额外时延：NOT_RUN/null；仅记录了预选 Token tuple 构建和 KV 视图裁剪开销，不冒充完整调度成本。
13. B→C 可重复加速：NOT_RUN/null。
14. A→C 完整 M3 加速：NOT_RUN/null。
15. A→C 完整 M123 加速：NOT_RUN/null。
16. 当前停止原因是通用实际模型严格 greedy/numerical 合同失败；镜像实际拒绝/裁剪顺序的独立串行 K/V 参照发现 {schedule['argmax_disagreements']} 次 argmax 分歧，详情见正确性报告。因果隔离检查通过。自然重复问题覆盖率也低。用户允许一定精度损失后已保留近似性能与答案差异诊断，但没有完成独立质量接受，不能把局部 Decode 潜力当成正式端到端成功。M1 后续复杂开发按前置门停止。
'''
    (root/'result_answers.md').write_text(answers)
    print('REPORT',root,'NO_GO exact deployment',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/m3_world_guided_decode_run01'))
    report(parser.parse_args().root)
