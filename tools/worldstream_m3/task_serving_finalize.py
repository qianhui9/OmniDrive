"""Evidence-linked reports; missing quality evidence always remains Unknown."""
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from worldstream.m3.task_evidence import TaskEvidenceContract,GRAMMAR,population_task
from tools.worldstream_m1.io_utils import write_json

ROOT=Path(os.environ.get('OMNIDRIVE_TASK_SERVING_OUTPUT','outputs/m2_freeze_m3_task_serving'))
TYPES=['object_relation','local_metric','map_topology','global_safety','planning','unknown']


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def stats(values):
    xs=list(values)
    return dict(count=len(xs),p50=float(np.percentile(xs,50)),p95=float(np.percentile(xs,95)),mean=float(np.mean(xs))) if xs else dict(count=0,p50=None,p95=None,mean=None)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stage(r,name,key='wall_ms'):return r.get('stages',{}).get(name,{}).get(key,0.)
def save(name,value):write_json(ROOT/name,value)
def change_type(path):
    return '修改' if subprocess.run(['git','ls-files','--error-unmatch',str(path)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 else '新增'
def md(name,text):(ROOT/name).write_text(text)


def census(initial,service,supplement):
    corpus=[];manifest={}
    for folder in ('data/nuscenes/conv/val','data/nuscenes/vqa/val'):
        for path in sorted(Path(folder).glob('*.json')):
            rows=read(path);manifest[str(path)]=digest(path)
            for i,p in enumerate(rows):corpus.append(dict(frame=path.stem,source=str(path),index=i,question=p['question'],offline_reference_answer=p.get('answer')))
    native=read(ROOT/'service/native_questions.json');unique={}
    for row in native:
        for i,q in enumerate(row['questions']):unique[row['frame'],i,q]=dict(frame=row['frame'],index=i,question=q,prompt_tokens=row['prompt_lengths'][i])
    actual=supplement['rows'];by_type={}
    for t in TYPES:
        a=[r for r in actual if population_task(r['question'])==t]
        by_type[t]=dict(requests=len(a),prompt_tokens=stats(r['prompt_tokens'] for r in a),generated_tokens=stats(r['generated_tokens'] for r in a),
            query_wall_ms=stats(r['wall_ms'] for r in a),prefill_wall_ms=stats(stage(r,'prefill') for r in a),decode_wall_ms=stats(stage(r,'decode') for r in a),
            visual_dependency='Unrestricted native multimodal context; state sufficiency not established',verified_state_complete_requests=0,global_inference_required=t in ('planning','global_safety'))
    qcounts=Counter(r['question'] for r in corpus)
    return dict(population='exploratory_reused_population',deployment_default='One native planning query per frame; additional native QA are a separately declared supported workload',
        corpus=dict(questions=len(corpus),counts=dict(Counter(population_task(r['question']) for r in corpus)),unique_question_texts=len(qcounts),
            repeated_text_fraction=1-len(qcounts)/len(corpus) if corpus else None,repeat_is_not_cache_eligibility=True,
            explicit_supported_grammar=sum(bool(GRAMMAR.fullmatch(r['question'].strip())) for r in corpus),manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()),
        service_native_inventory=dict(unique_frame_question_pairs=len(unique),counts=dict(Counter(population_task(r['question']) for r in unique.values())),prompt_tokens=stats(r['prompt_tokens'] for r in unique.values())),
        fully_timed_native_all_requests=len(actual),timed_by_type=by_type,independent_grounded_referent_annotations_available=False,
        potential_local_text_examples=[dict(r,admission='Unknown: no independently grounded current visual referent; matching words do not establish field/answer-schema sufficiency') for r in corpus if re.search(r'\b(distance|far|relative position|located|position of)\b',r['question'].lower())][:50],
        classification_is_descriptive_not_admission=True,whole_corpus_timing_not_claimed=True,
        corpus_manifest=manifest)


def summarize(runs,quality):
    result={}
    for pop in sorted({r['population'] for r in runs}):
        result[pop]={}
        for mode in sorted({r['mode'] for r in runs}):
            a=[r for r in runs if r['population']==pop and r['mode']==mode and not r['warmup']]
            q=[q for q in quality if q['population']==pop and q['mode']==mode]
            result[pop][mode]=dict(requests=len(a),query_requests=sum(len(r['questions']) for r in a),E2E_wall_ms=stats(r['E2E_wall_ms'] for r in a),E2E_GPU_ms=stats(r['E2E_GPU_ms'] for r in a),
                prefill_wall_ms=stats(stage(r,'prefill') for r in a),decode_wall_ms=stats(stage(r,'decode') for r in a),M1_total_ms=stats(r['m1_commit'].get('m1_total_including_adapters_ms',0.) for r in a),
                M2_wall_ms=stats(r['m2'].get('M2_wall_ms',0.) for r in a),actual_skipped_block_computations=sum(r['m2'].get('actual_skipped_block_computations',0) for r in a),
                actual_direct_requests=sum(r['m3'].get('actual_direct_requests',0) for r in a),peak_gpu_allocated=max((r['peak_gpu_allocated'] for r in a),default=0),
                peak_gpu_reserved=max((r['peak_gpu_reserved'] for r in a),default=0),cpu_rss_high_water_kib=max((r['cpu_rss_high_water_kib'] for r in a),default=0),quality_checks=len(q),quality_passed=sum(q['frozen_targets_pass'] for q in q),frozen_quality_pass=all(q['frozen_targets_pass'] for q in q) if q else None)
    return result


def main():
    assert read(ROOT/'batch_status.json')['status']=='completed'
    reconstructed=ROOT/'measured_service_source_reconstruction.json'
    if not reconstructed.exists():
        manifest=read(ROOT/'service/source_manifest.json');verified={}
        for name,h in manifest.items():
            p=Path(name);verified[name]=digest(p)==h
            if verified[name]:
                relative=p.relative_to(Path.cwd()) if p.is_absolute() else p
                target=ROOT/'measured_service_source'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        assert all(verified.values()),'Measured source must be archived before further edits'
        save('measured_service_source_reconstruction.json',dict(all_hashes_match=True,files=verified,note='Exact current source matches pre-execution manifest; copied without transformation'))
    assert read(ROOT/'final_validation_status.json')['status']=='completed'
    if (ROOT/'final_source_integrity.json').exists():assert read(ROOT/'final_source_integrity.json')['all_sources_unchanged']
    from . import m2_freeze_oracle
    old_funnel=read(ROOT/'m2_admission_funnel.json')
    if 'unit' not in old_funnel['stages'][0]:save('m2_admission_funnel_initial_window_liveness.json',old_funnel)
    save('m2_admission_funnel.json',dict(m2_freeze_oracle.funnel(),oracle=old_funnel['oracle'],source_audit=str(ROOT/'m2_oracle/source_audit.json')))
    runs=lines(ROOT/'service/runs.jsonl');quality=read(ROOT/'service/quality.json');initial=lines(ROOT/'initial_profile/runs.jsonl')
    final=lines(ROOT/'final_source_validation/runs.jsonl');final_q=read(ROOT/'final_source_validation/quality.json')
    supplement=read(ROOT/'service/prefix_checks/supplement/native_all_questions.json');negatives=read(ROOT/'service/prefix_checks/supplement/negative_execution.json')
    prefix=read(ROOT/'service/prefix_checks/results.json');recovery=read(ROOT/'service/recovery/results.json');oracle=lines(ROOT/'m2_oracle/frames.jsonl')
    frozen=read(ROOT/'baseline.json');changed=[p for p,h in frozen['sha256'].items() if not Path(p).exists() or digest(p)!=h]
    m1_frozen=not any(p.startswith('worldstream/m1/') or p=='configs/worldstream_m1_engineering_frozen.yaml' for p in changed)
    original_unchanged=not any(p.startswith('projects/mmdet3d_plugin/') for p in changed)
    formal_m2_changes=[p for p in changed if p.startswith('worldstream/m2/')]
    allowed_m2_changes={'worldstream/m2/runtime.py','worldstream/m2/conditional_prefix.py'}
    m2_frozen=set(formal_m2_changes)<=allowed_m2_changes
    summary=summarize(runs,quality);final_summary=summarize(final,final_q)
    corpus=census(initial,runs,supplement);save('task_workload_census.json',corpus)
    latency={}
    for pop in ('natural_planning','native_multi'):
        a=[r for r in initial if r['population']==pop]
        names=sorted({k for r in a for k in r['stages']})
        latency[pop]=dict(requests=len(a),E2E_wall_ms=stats(r['E2E_wall_ms'] for r in a),stages={n:dict(wall_ms=stats(stage(r,n) for r in a),GPU_ms=stats(stage(r,n,'gpu_ms') for r in a)) for n in names},
            measured_prefill_lengths=[x['input_length'] for r in a for x in r['invocations'] if x['stage']=='prefill'],
            per_decode_step_wall_ms=stats(x['wall_ms'] for r in a for x in r['invocations'] if x['stage']=='decode'),
            hypothetical_eliminate_prefill_speedup=stats(r['E2E_wall_ms']/(r['E2E_wall_ms']-stage(r,'prefill')) for r in a),
            hypothetical_eliminate_generation_speedup=stats(r['E2E_wall_ms']/(r['E2E_wall_ms']-stage(r,'generation')) for r in a),
            hypothetical_bounds_actual_execution=False)
    final_stages={pop:{mode:{name:dict(wall_ms=stats(stage(r,name) for r in final if r['population']==pop and r['mode']==mode),GPU_ms=stats(stage(r,name,'gpu_ms') for r in final if r['population']==pop and r['mode']==mode)) for name in sorted({n for r in final for n in r['stages']})} for mode in final_summary[pop]} for pop in final_summary}
    save('vlm_latency_breakdown.json',dict(final_source_separate_adapter_cache_parser_timers=final_stages,initial_profile=latency,service_stage_breakdown=summary,native_all_by_type=corpus['timed_by_type'],
        timers='Each actual full causal-LM prefill/decode call has separate synchronized CUDA + wall timing; no percentile subtraction. CUDA event spans include host enqueue gaps, not pure SM activity. Nested head/memory/generation spans overlap and are not additive.',
        observer_effect='Per-invocation synchronization charged to all service arms. Results apply to this instrumented exploratory protocol; no production cost certificate.',
        decode_trace=prefix['trace'],visual_tokens=513,visual_token_reduction='No new compact mechanism admitted. Its zero-prefill upper bound is above; no token-ratio latency extrapolation.',
        original_data=str(ROOT/'initial_profile/runs.jsonl'),service_data=str(ROOT/'service/runs.jsonl')))
    evidence_rows=[d for r in runs for d in r['m3'].get('requests',[]) if 'contract' in d]
    direct_count=sum(r['m3'].get('actual_direct_requests',0) for r in runs)
    save('task_answerability_analysis.json',dict(native_timed_queries=len(supplement['rows']),verified_input_complete=0,independently_quality_qualified=0,true_answerability_of_unrestricted_questions='Unknown',
        corpus_explicit_supported_grammar=corpus['corpus']['explicit_supported_grammar'],service_contract_checks=len(evidence_rows),
        local_candidate_review=corpus['potential_local_text_examples'],scope='No-Go for current evidence interfaces and frozen explicit-ID prototype. It does not prove that unrestricted local questions are inherently unanswerable from any future independently grounded state.',
        required_evidence_missing=['Independent current visual referent identity','Independent task accuracy certificate','Explicit current source/version and supported answer schema on unrestricted questions'],
        related_history_is_not_input_completeness=True,default_planning_direct_supported=False,physical_error_reference='No independent grounded local query reference exists for accepted requests because accepted population is empty; not replaced with VLM agreement',
        potential_direct_generation_savings_on_native_admitted_population_ms=0,controlled_questions_excluded_from_native_denominator=True))
    save('task_evidence_contract.json',dict(default=asdict(TaskEvidenceContract()),grammar=GRAMMAR.pattern,
        input_binding='Actual input token IDs must decode to the unchanged original single-turn OmniDrive Vicuna prompt containing exactly the declared question and one image marker; unknown wrappers/history reject',
        metrics={'distance':'Planar Euclidean distance from ego origin to current observed object center, meters, two decimals','relative_position':'Current ego x/y/z meters, JSON x_m/y_m/z_m, two decimals'},
        default_shadow=True,production_enabled=False,experimental_execution_requires=['offline_experiment','task_evidence_enabled','experimental_direct_query','shadow_evaluation=False','complete independent current evidence','matching independently heldout task artifact','native prompt + output schema binding'],
        forbidden=['global safety','planning/driving decisions','unknown existence','incomplete detection counts','ambiguous/stale/no-current observations','online GT','head-history agreement as independent truth']))
    save('direct_query_shadow_quality.json',dict(population='native_service',checks=len(evidence_rows),candidate_answers=sum(d.get('candidate') is not None for d in evidence_rows),
        complete_contracts=sum(d.get('evidence_complete',False) for d in evidence_rows),quality_qualified=sum(d.get('quality_qualified',False) for d in evidence_rows),
        disagreements_with_full_vlm=None,numeric_reference_error=None,reason='No native complete/independently qualified candidate. Undefined accuracy is not 100%.',
        false_direct_count=direct_count,false_direct_rate=None,false_reject_rate=None,correctly_answerable_rate_verified=0.,false_reject_reason='Unknown without independently established answerable references',
        controlled_negative_original_model_checks=negatives,controlled_fixture_tests='tests/worldstream_m3/test_task_evidence_serving.py; functionality only, never a quality certificate',shadow_savings_credited_ms=0))
    save('direct_query_execution_results.json',dict(native_actual_direct_requests=direct_count,native_actual_avoided_prefill_requests=direct_count,native_actual_avoided_decode_requests=direct_count,saved_decode_tokens=0,
        original_model_direct_execution_quality_approved=False,actual_direct_branch_proof='Controlled unit fixture forbids multimodal expansion and model forward/generate; independent real-data quality not established, so no original-model shortcut authorized',
        fallback_reasons=dict(Counter(d.get('fallback_reason') for d in evidence_rows)),experimental_population_includes_all_rejected_requests=True,shadow_is_not_execution_saving=True))
    prefix_rows=[]
    for r in runs:
        diag=r['m3'];diag=diag.get('visual_fallback',diag)
        for d in diag.get('requests',[]):prefix_rows.append(dict(population=r['population'],mode=r['mode'],frame=r['frame'],reused_tokens=d.get('reused_tokens',0),effective_prefill_tokens=d.get('effective_prefill_tokens'),input_tokens=d.get('input_tokens'),cache_bytes=d.get('cache_bytes',0),reason=d.get('prefix_admission')))
    prefix_query_denominator=sum(len(r['questions']) for r in runs if r['mode'] in ('exact_prefix','direct_prefix_fallback'))
    save('exact_prefix_integration.json',dict(query_denominator=prefix_query_denominator,actual_prefix_hit_rate=sum(d['reused_tokens']>0 for d in prefix_rows)/prefix_query_denominator,complete_prefill_elisions_via_prefix=sum(d['reused_tokens']>0 and d['effective_prefill_tokens']==0 for d in prefix_rows),saved_decode_tokens_via_prefix=0,corrected_final_identity_witness=read(ROOT/'final_source_validation/identity_metadata.json'),original_model_checks=prefix,service_rows=prefix_rows,service_prefix_hits=sum(d['reused_tokens']>0 for d in prefix_rows),
        single_question_fastpath_preserved=True,common_prefix_seed_and_batched_decode_algorithms_unchanged=True,changed_generation_contract='Existing beam/unsupported mask tests remain native; original full-model service uses frozen greedy controls. No generation algorithm replaced.',
        direct_prefix_route_exclusive=True,production_auto_approved=False,note='Identical expanded input is dependency-exact; suffix prefill shapes may have floating-point differences. Frozen whole-output quality gates are not relaxed.'))
    paired={}
    for pop in summary:
        paired[pop]={}
        for mode in summary[pop]:
            a=[r for r in runs if r['population']==pop and r['mode']==mode]
            pairs={}
            for baseline in ('original','full_maintain','frozen_exact_m2'):
                refs={(r['scene'],r['repeat'],r['ordinal']):r for r in runs if r['population']==pop and r['mode']==baseline}
                savings=[refs[r['scene'],r['repeat'],r['ordinal']]['E2E_wall_ms']-r['E2E_wall_ms'] for r in a]
                ratios=[refs[r['scene'],r['repeat'],r['ordinal']]['E2E_wall_ms']/r['E2E_wall_ms'] for r in a]
                pairs[baseline]=dict(paired_saved_wall_ms=stats(savings),paired_speedup=stats(ratios))
            paired[pop][mode]=pairs
    observed_prefix_profitability={pop:dict(mean_saved_ms=paired[pop]['exact_prefix']['frozen_exact_m2']['paired_saved_wall_ms']['mean'],frozen_quality_pass=summary[pop]['exact_prefix']['frozen_quality_pass'],production_approved=False) for pop in paired}
    save('runtime_cost_benefit.json',dict(exact_prefix_observed_by_population=observed_prefix_profitability,service_contract='All non-original arms maintain identical frozen M1 full persistent service, frozen Exact M2 where enabled, original heads/memory/513 carriers/full generation on rejection; Original lacks M1 maintenance',
        paired=paired,direct_native_net_opportunity=False,exact_prefix_production_profitable=False,
        oracle=dict(frames=len(oracle),mean_ideal_token_update_fraction=float(np.mean([r['ideal_token_update_fraction'] for r in oracle])),legal_window_update_fraction=1.,
            feature_max_abs=max(r['final_feature_max_abs'] for r in oracle),impossible_all_prefix_deletion_upper_bound=read(ROOT/'m2_oracle/prefix_only_profile.json'),authorized_window_saving_ms=0,actual_oracle_execution_wall_lower_bound_ms=stats(r['oracle_actual_execution_wall_lower_bound_ms'] for r in oracle),
            ideal_token_splicing_is_unimplementable=True,offline_full_forward_never_online=True),
        limitations='One replay per scene, two exposed scenes, instrumented synchronized stage timers, no new independent quality population. Descriptive paired timings cannot approve production.',
        no_token_ratio_speedup_estimates=True))
    observed_protocol=dict(population_label='exploratory_reused_population',requests=len(runs),modes=sorted({r['mode'] for r in runs}),populations=sorted({r['population'] for r in runs}),scenes=sorted({r['scene'] for r in runs}),frames={s:sorted({(r['timestamp'],r['frame']) for r in runs if r['scene']==s}) for s in {r['scene'] for r in runs}},warmup=0,repeats=1,source_archive=str(ROOT/'measured_service_source'),source_archive_matches_pre_execution_manifest=read(ROOT/'measured_service_source_reconstruction.json')['all_hashes_match'],note='Reconstructed protocol from original request records and recorded launch arguments; no independent-test claim')
    save('service_protocol_reconstruction.json',observed_protocol)
    save('end_to_end_comparison.json',dict(protocol=observed_protocol,population='exploratory_reused_population',main=summary,final_source_verification=final_summary,paired=paired,
        requests=len(runs),queries=sum(len(r['questions']) for r in runs),all_rejections_included=True,failed_request_exclusions=0,
        immutable_final_source_confirmation=read(ROOT/'final_source_integrity.json') if (ROOT/'final_source_integrity.json').exists() else None,old_source_versus_final_guard='Main exploratory service source pinned in service/source_manifest.json; final adds native prompt/output-budget binding and adapter/parser timers for quality-qualified candidates (none exist in main population). Final source separately rerun on all seven arms; main timings are not reissued as a final-source production certificate.'))
    suite=ET.parse(ROOT/'final_regression_cuda.xml').getroot();cases=suite.findall('.//testcase')
    failures=suite.findall('.//failure')+suite.findall('.//error');skipped=suite.findall('.//skipped')
    save('regression_results.json',dict(tests=len(cases),failed=len(failures),skipped=len(skipped),xml=str(ROOT/'final_regression_cuda.xml'),
        earlier_full_cuda=read(ROOT/'batch_status.json')['steps'][0],negative_original_model_requests=len(negatives['rows']),
        final_source_requests=len(final),m2_recovery=recovery,all_passed=not failures and not skipped,
        failure_history=['initial_profile_failed_attempt1.log: runner missing no_grad, OOM before completed requests; fixed without model/weight changes','targeted.log: Python 3.9 annotation compatibility collection failure, fixed with future annotations']))
    checkpoint=Path('ckpts/iter_10548.pth');weight_hash=hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):weight_hash.update(chunk)
    expected_checkpoint=read('outputs/system_optimization_v5/model_manifest.json')['checkpoint_sha256']
    assert weight_hash.hexdigest()==expected_checkpoint,'Original checkpoint changed'
    final_manifest=read(ROOT/'final_source_validation/source_manifest.json')
    assert all(digest(p)==h for p,h in final_manifest.items() if p.startswith('worldstream/')),'Core source changed after final validation'
    save('original_model_parity.json',dict(checkpoint_sha256=weight_hash.hexdigest(),checkpoint_matches_previous_original_sha=True,m1_frozen=m1_frozen,original_neural_source_unchanged=original_unchanged,baseline_head=frozen['head'],
        protected_changes=changed,formal_M2_changes_diagnostics_only=formal_m2_changes,per_arm_quality=summary,final_source_quality=final_summary,
        quality_checks=len(quality),quality_passed=sum(q['frozen_targets_pass'] for q in quality),final_quality_checks=len(final_q),final_quality_passed=sum(q['frozen_targets_pass'] for q in final_q),
        raw_final_outputs=str(ROOT/'final_source_validation/raw_outputs'),frozen_targets='outputs/module2_refinement/quality_targets.json',targets_sha256=digest('outputs/module2_refinement/quality_targets.json'),recovery=recovery,
        all_main_frozen_targets_pass=all(q['frozen_targets_pass'] for q in quality),checkpoint_unmodified=True,checkpoint='ckpts/iter_10548.pth',old_outputs_preserved=True))
    prefix_quality_pass=all(q['frozen_targets_pass'] for q in quality if q['mode'] in ('exact_prefix','direct_prefix_fallback'))
    decision=dict(m1_frozen=m1_frozen,m2_exact_frozen=m2_frozen,m2_conditional_production_enabled=False,m3_task_answerable_population=0,
        m3_direct_query_quality_pass=False,m3_direct_query_profitable=False,m3_exact_prefix_profitable=False,
        full_vlm_e2e_speedup=paired['natural_planning']['direct_prefix_fallback']['original']['paired_speedup']['mean'],
        raw_descriptive_speedup_by_population={pop:paired[pop]['direct_prefix_fallback']['original']['paired_speedup'] for pop in paired},
        production_quality_constrained_speedup_established=False,remaining_system_bottleneck='Original autoregressive decode; actual per-token decoder execution dominates prefill. Kernel trace gives execution evidence, not a memory-bandwidth measurement.',
        next_research_priority='Profile and optimize the original per-token decode execution under unchanged whole-output quality, including launch/LoRA/GEMM/KV traffic. No new state approximation or compact cache approved.',
        gates=dict(G1_m1_m2_freeze=m1_frozen and m2_frozen and original_unchanged,G2_conditional='No-Go; zero complete source windows',G3_actual_workload_profile=True,G4_complete_evidence='Interface tested; zero verified native population',G5_independent_direct_quality='No-Go; absent independent accepted population',G6_exclusive_routes_and_prefix=dict(routes_exclusive=True,prefix_whole_output_quality_pass=prefix_quality_pass,go=prefix_quality_pass,reason='Input/KV dependency identity is checked; observed full-output disagreement blocks production even when mathematical prefix matches'),G7_same_service_net_gain='No-Go for production; descriptive per-population timings only',G8_regression=not failures and not skipped),
        independent_final_test=False,exposed_population_label='exploratory_reused_population')
    save('formal_go_nogo.json',decision)
    recovery_errors=[r['detection']['max_center_difference_m'] for r in recovery['rows'] if r['case']=='unqualified_prefix_input_then_full_EVA' and r['ordinal']>0]
    md('m2_freeze_decision.md',f'''# M2 冻结结论\n\nM1 冻结：{m1_frozen}。M2 正式 Exact 算法保留，仅规范诊断字段；条件复用生产开关始终为 false。\n\n固定 8 场景 / 96 帧 / 576 相机的来源链、patch、window 漏斗见 m2_admission_funnel.json；同一窗口分母不变。未形成完整可复用窗口，停止条件复用优化。\n\n新鲜 96 帧 Full Forward oracle 仅在离线用当前真实 block-1 输出补齐未知 token，并运行原始 blocks 2..23 测量误差；既不能在线获得，也不能授权部分窗口跳算。来源 patch 的支持率不是注意力窗口的可跳算率。\n\n研究路径顶层标记 experimental_conditional_forced；即使其中包含同坐标 Exact 子集，也保留 quality_qualified=false、performance_qualified=false。正式缓存和研究缓存隔离。\n\n恢复实验证据见 service/recovery/results.json。仅当前 EVA 完整计算不代表下游记忆已恢复；本轮受控扰动后第 1/2 帧 EVA 误差为 0，检测中心最大差仍分别约 {recovery_errors[0]:.2f}/{recovery_errors[1]:.2f}m。可靠恢复先合法丢弃下游记忆，再从序列起点重放，本轮 {sum(r['frozen_targets_pass'] for r in recovery['rows'] if r['case']=='legal_reset_and_reliable_replay')}/{sum(r['case']=='legal_reset_and_reliable_replay' for r in recovery['rows'])} 帧恢复冻结目标。\n''')
    files=[p for p in Path('worldstream').rglob('*.py') if str(p) not in frozen['sha256'] or digest(p)!=frozen['sha256'][str(p)]]
    files += [Path('tests/worldstream_m2/test_source_aligned_conditional.py'),Path('tests/worldstream_m3/test_task_evidence_serving.py'),Path('projects/configs/OmniDrive/worldstream_m3_task_evidence_research.py'),*Path('tools/worldstream_m3').glob('task_serving_*.py'),Path('tools/worldstream_m3/m2_freeze_oracle.py')]
    md('implementation_changes.md','# 实现与插桩变更\n\n新增 TaskEvidenceContract、严格拒绝门、默认影子模式及显式实验直接执行分支；直接拒绝后可复用现有 Exact Prefix。未改变原始网络、权重、解码算法、M1 配置、M2 准入/执行算法。旧 post_perception.direct_query 入口统一调用同一严格门，不能只靠一个布尔值批准。\n\n| 文件 | 类型 | 用途 | 行数 |\n|---|---|---|---|\n'+''.join(f'| {p} | {change_type(p)} | {"运行时接口/测试" if not str(p).startswith("tools/") else "独立实验、计时或报告脚本；无模型内联插桩"} | 1–{len(p.read_text().splitlines())} |\n' for p in files)+'\n计时位于独立 runner；各 prefill/decode 调用分别同步计时，所有臂均包含 observer 成本。模型服务结束恢复原方法。阶段嵌套时间不可相加。GPU trace 属诊断实验，不作为完整请求性能证明。\n')
    table='| 工作负载 | 方法 | 请求 | E2E P50/P95 ms | 冻结质量通过/检查 |\n|---|---|---:|---:|---:|\n'
    for pop,modes in summary.items():
        for mode,r in modes.items():table+=f'| {pop} | {mode} | {r["requests"]} | {r["E2E_wall_ms"]["p50"]:.2f}/{r["E2E_wall_ms"]["p95"]:.2f} | {r["quality_passed"]}/{r["quality_checks"]} |\n'
    p=latency['natural_planning'];pref=p['stages']['prefill']['wall_ms']['p50'];dec=p['stages']['decode']['wall_ms']['p50']
    md('experiment_report.md',f'''# OmniDrive M2 冻结与 M3 任务条件化服务\n\n**结论：M2 条件复用 No-Go；M3 直接回答生产 No-Go；Exact Prefix 保留为受限实验接口，没有签发新的 Auto 证书。**\n\n冻结 M1、原始网络与权重不变。新增严格 TaskEvidenceContract 和影子/实验执行入口，默认仍完整运行原始 VLM。拒绝样本始终留在分母。\n\n真实默认规划 profile：prefill P50 {pref:.2f} ms，decode P50 {dec:.2f} ms；两者独立计时，未用 P50 相减。513 个视觉 token 全部保留。逐 token 执行与输出长度共同构成 decode 成本，详细 CUDA/wall、token 数、原生全问题分类和短 decode trace 见 vlm_latency_breakdown.json。\n\n七组对照：{len(runs)} 次完整模型前向、{sum(len(r['questions']) for r in runs)} 个查询；原生全问题补充 {len(supplement['rows'])} 个，受控负例 {len(negatives['rows'])} 个。全部属于已暴露开发人口 exploratory_reused_population，不能重新命名为独立最终测试。默认任务、原生多问、同帧完全相同请求分别报告。\n\n{table}\n\n原生输入完备且独立质量合格的直接查询为 0；真正免除的 prefill/decode 为 0，节省 decode token 为 0。没有用关键词、语义相似、当前 Head 与历史一致性批准直接回答。没有独立可接受对象指称参考，误拒率和直接回答物理误差保持 Unknown，零接受不等于准确率 100%。受控 fixture 只证明函数执行分支，不作为真实质量证书。\n\n固定 96 帧 M2 新鲜离线 oracle、来源漏斗和恢复实验见 m2_admission_funnel.json、m2_freeze_decision.md。部分 token 支持不会成为完整窗口复用证书。\n\n完整最终 CUDA 回归 {len(cases)} 项，失败 {len(failures)}，跳过 {len(skipped)}；固定全部源码后另执行 {len(final)} 次七臂原模型前向，启动前与结束后哈希逐项一致。前次抽验及快照脚本版本审计保留在 final_source_validation_attempt1。初次 runner OOM 和 Python 3.9 注解失败日志保留。研究性能主实验源码与最后新增的输入提示/输出预算绑定守卫与细分计时分别固定并验证，不把前者重新签发为最终源码生产成本证书。\n\n同服务边际收益与相对 Original 的总收益分列 runtime_cost_benefit.json；Exact Prefix 的正收益若仅出现在受控重复请求，不能外推默认单规划任务。单次重放、两场景与阶段同步插桩只支持描述性比较。所有数值失败均保留并阻止部署准入。原生局部问题的文本示例及离线参考在 task_workload_census.json，仅用于分析；当前显式 ID 原型与现有证据接口的 No-Go 不等于证明所有局部问题在理论上不可由未来状态回答。\n\n下一优先级是原始逐 token decode 的执行分析与优化，继续保持整模型检测、地图、记忆、文本和规划门。完整结果与证据链接见 formal_go_nogo.json、original_model_parity.json；所有旧输出保留。\n''')
    required=['implementation_changes.md','m2_admission_funnel.json','m2_freeze_decision.md','vlm_latency_breakdown.json','task_workload_census.json','task_answerability_analysis.json','task_evidence_contract.json','direct_query_shadow_quality.json','direct_query_execution_results.json','exact_prefix_integration.json','runtime_cost_benefit.json','end_to_end_comparison.json','original_model_parity.json','regression_results.json','formal_go_nogo.json','experiment_report.md','commands.sh']
    assert all((ROOT/p).is_file() and (ROOT/p).stat().st_size>0 for p in required)
    import tarfile
    with tarfile.open(ROOT/'final_source.tar.gz','w:gz') as archive:
        for p in set(files)|set(Path('worldstream').rglob('*.py')):archive.add(p)
    current={str(p):digest(p) for p in files}
    save('delivery_manifest.json',dict(status='completed',final_source_integrity=read(ROOT/'final_source_integrity.json') if (ROOT/'final_source_integrity.json').exists() else None,files=current,m1_frozen=m1_frozen,original_neural_source_unchanged=original_unchanged,required_outputs_present=True,final_source_manifest=str(ROOT/'final_source_validation/source_manifest.json')))
    print(json.dumps(dict(regressions=len(cases),requests=len(runs),queries=sum(len(r['questions']) for r in runs),quality_passed=sum(q['frozen_targets_pass'] for q in quality),quality_checks=len(quality),m1_frozen=m1_frozen,original_unchanged=original_unchanged),indent=2))

if __name__=='__main__':main()
