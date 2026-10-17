"""Offline inventory of ORIGINAL questions; lexical bins never grant admission.

The corpus inventory reads question text only, never answers or annotations.
Semantic candidates are an optimistic screening bound, not certified coverage.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import re

from .audit import write

PLANNING = 'Please provide the planning trajectory for the ego car without reasons.'
ACTION = 'Please shortly describe your driving action.'
# Read and semantically reviewed original corpus questions. These are genuine
# dependency opportunities CONDITIONAL on referent grounding; not online hits.
REVIEWED_METRIC_TASKS = {
    'What is the approximate distance between the vehicle and the car in front?',
    'What is the approximate distance between the vehicle and the delivery truck?',
}
CATEGORIES = (
    'global_planning', 'global_safety_scene', 'count_absence_existence',
    'local_spatial_relation', 'local_distance_position', 'local_motion',
    'local_map_candidate', 'other_visual_semantics',
)
# These are dependency specifications, not tuned geometric thresholds.
REQUIREMENTS = {
    'global_planning': ('yes', 'no', 'complete current agents, free space, signals, route, ego dynamics',
        'current request; dynamic agents need current motion', 'trajectory quality unchanged; no local surrogate'),
    'global_safety_scene': ('yes', 'no', 'all relevant hazards, scene semantics, signs, lighting, intent',
        'current request', 'complete safety/semantic evidence absent from local anchors'),
    'count_absence_existence': ('yes (for arbitrary yes/no and exact counts)', 'no', 'complete requested spatial domain and identity deduplication',
        'current request', 'no false negatives; partial history cannot prove absence'),
    'local_spatial_relation': ('object grounding required; full global scene not intrinsically required', 'conditional',
        'unique referent, identities and signed relative direction interval', 'verify at request time; age plus motion error bounded',
        'signed interval must exclude zero'),
    'local_distance_position': ('object grounding required; nearest selection needs domain coverage', 'conditional',
        'unique referent, independent metric localization error bound, ego pose', 'verify at request time',
        'frozen prior direct numeric error <= 0.5 m; detector score is insufficient'),
    'local_motion': ('unique object grounding and temporal evidence required', 'conditional',
        'unique referent, time baseline, independently supported object motion and ego motion', 'current velocity relation with bounded lag',
        'independent velocity interval excludes decision boundary; no new tolerance frozen'),
    'local_map_candidate': ('local domain and map semantic type required', 'conditional',
        'local lane geometry, topology/marking semantics when requested', 'static map plus current ego pose and current closure observations',
        'lane_control_points alone do not certify topology, markings or traffic rules'),
    'other_visual_semantics': ('question specific; unresolved', 'no established substitution',
        'appearance, text, buildings, weather or unconstrained language semantics', 'current request',
        'not represented in current state interface'),
}


def dependency(question):
    """Conservative, reviewable screening only; no online consumer uses this."""
    q = question.lower().strip()
    if q == PLANNING.lower() or 'planning trajectory' in q:
        category = 'global_planning'
    elif q == ACTION.lower() or re.search(r'\b(safe|safety|risk|collision|should|would|might|why|happen|affect|influence|impact|decision|action|move|scenario|environment|conditions|summary|overview)\b', q):
        category = 'global_safety_scene'
    elif re.search(r'\b(how many|count|number of|any|no|none|absence)\b', q) or re.search(r'\b(is|are) there\b', q):
        category = 'count_absence_existence'
    elif re.search(r'\b(distance|how far|position|location of)\b', q):
        category = 'local_distance_position'
    elif re.search(r'\b(moving|stationary|motion|speed|velocity|approaching|receding)\b', q):
        category = 'local_motion'
    elif re.search(r'\b(left of|right of|relative|in front of|behind)\b', q):
        category = 'local_spatial_relation'
    elif re.search(r'\b(lane|lanes|road|intersection|markings|topology|curve|curvature)\b', q):
        category = 'local_map_candidate'
    else:
        category = 'other_visual_semantics'
    candidate = category in CATEGORIES[3:7]
    # Do not confuse semantic screening with grounded, input-complete tasks.
    return dict(category=category, semantic_candidate=candidate,
        input_complete=False, direct_query_eligible=False, compact_input_eligible=False,
        reason=('referent/domain and requested semantics not certified by current state; candidate requires manual grounding and independent quality evidence'
                if candidate else 'full visual context or facts outside current local state required'),
        classification_scope='lexical screening; not semantic proof or online admission')


def inventory(root, data_root, previous):
    root, data_root, previous = map(Path, (root, data_root, previous))
    root.mkdir(parents=True, exist_ok=True)
    annotation = data_root / 'nuscenes2d_ego_temporal_infos_val.pkl'
    infos = pickle.load(annotation.open('rb'))['infos']
    counts = defaultdict(Counter); missing = Counter(); examples = defaultdict(list)
    digest = hashlib.sha256(); total = 0; reviewed=[]
    with (root / 'task_dependency_records.jsonl').open('w') as handle:
        for info in infos:
            sources = [('default_planning', PLANNING), ('short_action', ACTION)]
            for source in ('conv', 'vqa'):
                path = data_root / source / 'val' / (info['token'] + '.json')
                if not path.exists():
                    missing[source] += 1
                    continue
                raw = path.read_bytes(); digest.update(str(path).encode()); digest.update(raw)
                sources.extend((source, row['question']) for row in json.loads(raw))
            for source, question in sources:
                result = dependency(question)
                record = dict(sample_token=info['token'], scene_token=info['scene_token'], source=source,
                    question=question, **result)
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                if question in REVIEWED_METRIC_TASKS:reviewed.append(record)
                counts[source][result['category']] += 1; total += 1
                if len(examples[result['category']]) < 8:
                    examples[result['category']].append(dict(question=question, source=source, sample_token=info['token']))
    # The actually executed v3 questions are a distinct workload. Do not count
    # method repeats as additional questions or silently replace the first five.
    executed = Counter(); requests = 0
    for run in ('main', 'long', 'auto', 'auto_supported', 'cost_support', 'multi'):
        for record in json.loads((previous / run / 'questions.json').read_text()).values():
            for q in record['questions']:
                executed[dependency(q)['category']] += 1; requests += 1
    aggregate = sum(counts.values(), Counter())
    candidate_count = sum(aggregate[c] for c in CATEGORIES[3:7])
    result = dict(status='completed_offline_audit', annotation=str(annotation),
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(), question_corpus_sha256=digest.hexdigest(),
        frames=len(infos), independent_scenes=len({i['scene_token'] for i in infos}),
        counts_by_source={k:dict(v) for k,v in counts.items()}, categories=dict(aggregate),
        audited_question_occurrences=total, missing_question_files=dict(missing),
        default_planning=dict(total=len(infos), potential_direct_query_fraction=0., potential_compact_input_fraction=0.,
            full_visual_fallback_fraction=1., reason='global planning depends on complete current scene and policy, not a local object fact'),
        vqa_screening=dict(total=sum(sum(counts[s].values()) for s in ('conv','vqa')),
            semantic_candidates=sum(counts[s][c] for s in ('conv','vqa') for c in CATEGORIES[3:7]),
            established_input_complete_tasks=0, potential_coverage_certified=0.,
            unresolved_candidate_scope='Lexical screening includes appearance questions and ego-motion questions; not all are local-object opportunities'),
        all_sources_semantic_candidate_count=candidate_count,
        manually_reviewed_conditional_metric_opportunities=reviewed,
        semantic_screening_upper_fraction=candidate_count/total,
        potential_direct_query_fraction_established=0., potential_compact_input_fraction_established=0.,
        certified_full_visual_fallback_fraction=1.,
        executed_v3_workload=dict(unique_run_question_occurrences=requests, categories=dict(executed),
            input_complete_established=0, note='Run-level occurrences; overlapping frames across runs not independent samples'),
        requirements={c:dict(zip(('complete_scene','partial_history_allowed','facts','freshness','precision'),REQUIREMENTS[c]),
            current_head_can_verify='prediction agreement only; no independent truth/error bounds',
            reduce_llm_input='conditional research possibility' if c in CATEGORIES[3:7] else 'not supported',
            direct_query='not currently justified', examples=examples[c]) for c in CATEGORIES},
        gate_a=dict(decision='No-Go for original Planning historical substitution; unresolved VQA candidates remain research-only',
            online_bc_allowed=False, reason='No grounded input-complete original task with independent fact and task quality evidence; do not equate lexical bins with opportunities',
            offline_b_allowed=bool(reviewed),
            offline_b_reason='Two manually reviewed original metric questions have conditional local-fact dependencies; test available evidence offline before any consumption',
            continue_d_e=True),
        limitations=['No semantic parser or manual labeling of 72k unconstrained QA; candidate counts are screening only',
            'Zero established coverage is not a proof that every original VQA task is impossible',
            'Local controlled probes cannot replace the Planning/VQA workload',
            'conv random summary template excluded from full corpus denominator; actual frozen run templates included separately'])
    write(root/'task_state_feasibility.json', result)
    # Use individual measured stage durations; never add component percentiles.
    measurements=[]
    for line in (previous/'main/frames.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['warmup'] or row['mode']!='original': continue
        stages=row['frame']['stages']
        prefill=[s for s in stages if s['name']=='llm_prefill']
        decode=[s for s in stages if s['name']=='llm_decode']
        measurements.append(dict(frame=row['dataset_identity']['sample_token'],
            e2e_ms=row['frame']['e2e_wall_ms'], prefill_wall_ms=sum(s['wall_ms'] for s in prefill),
            decode_wall_ms=sum(s['wall_ms'] for s in decode), effective_prefill_tokens=sum(s.get('forward_tokens',0) for s in prefill)))
    bounds=dict(status='analytical_upper_bounds_not_speedup', native_measurement_source=str(previous/'main/frames.jsonl'),
        native_measurements=measurements, original_carrier_tokens=513,
        default_planning_legal_skipped_tokens_upper=0, established_original_task_validation_budget_ms=0.,
        established_state_substitution_e2e_speedup_upper=1.,
        screening_only_max_carrier_tokens_if_every_candidate_were_solved=candidate_count*513,
        manually_reviewed_conditional_metric_tasks=len(reviewed),
        manually_reviewed_conditional_fraction_of_original_vqa=len(reviewed)/result['vqa_screening']['total'],
        reviewed_metric_ideal_carrier_reduction_upper_tokens=len(reviewed)*513,
        reviewed_metric_bound_scope='Conditional on independently grounding both original referents and certifying metric facts plus unchanged answer quality; not actual activation',
        screening_only_token_bound_excludes_compact_context_cost=True,
        all_workload_absolute_ideal='Per measured request T, zero-cost removal of measured LLM forward wall intervals L gives T/(T-L). This excludes generation/sampling work outside forwards, is unattainable for global planning, and is not observed speedup',
        per_request_unattainable_language_removal=[dict(frame=m['frame'],
            speedup_upper=m['e2e_ms']/max(1e-9,m['e2e_ms']-m['prefill_wall_ms']-m['decode_wall_ms']),
            validation_break_even_ms=m['prefill_wall_ms']+m['decode_wall_ms']) for m in measurements],
        candidate_verification_budget='Must be below actually avoided native language work, minus retrieval/context/scheduling; candidate coverage and quality unestablished',
        prefill_cost_model='No linear ms/token extrapolation: native attention is nonlinear and context changes need real inference',
        source_generation_contract='v3 original single native sampling temp=.1 top_p=.75 beams=1 max_new_tokens=320 use_cache=True')
    write(root/'theoretical_benefit_bound.json',bounds)
    lines=['# 原始任务依赖与收益可行性审计', '',
        f'实际读取验证集 {len(infos)} 帧、{result["independent_scenes"]} 个场景；原始 conv/VQA 文件无替换。',
        f'共 {total} 条任务出现（含逐帧 Planning/短动作）。完整问题证据见 task_dependency_records.jsonl。', '',
        '**Gate A：原始 Planning 历史状态替代 No-Go；VQA 局部候选未获得输入完备性证据，B/C 在线开发不启动。D/E 可独立推进。**', '',
        '以下分类为可复查的词法筛查，不构成任务依赖证明或线上准入。潜在局部候选不能直接计为命中。', '',
        '| 类别 | 数量 | 比例 | 完整覆盖 | 不完整历史 | 物理事实 / 精度 / 新鲜度 |',
        '|---|---:|---:|---|---|---|']
    for c in CATEGORIES:
        r=REQUIREMENTS[c]
        lines.append(f'| {c} | {aggregate[c]} | {aggregate[c]/total:.4%} | {r[0]} | {r[1]} | {r[2]}; {r[4]}; {r[3]} |')
    lines += ['', f'原始 Planning {len(infos)} 条：可合法替代覆盖 0%，强制 Full Visual 100%，可合法跳过 Prefill Token 上限 0。',
        f'conv/VQA 筛查候选 {result["vqa_screening"]["semantic_candidates"]}/{result["vqa_screening"]["total"]}；已建立完整依赖映射 0。不能把未解决的问题报告为已证明不可能。',
        '当前状态只有类别、几何、未认证速度和 lane_control_points；没有问题指称到唯一目标的独立映射、完整搜索域、外观/标志语义或独立测距误差界。当前 Head 一致性也不是物理真值。',
        '局部方向在有独立前景归属、身份、投影符号证据时可能无需完整 3D；本轮没有这样的独立认证证据。',
        f'人工复核确认 {len(reviewed)} 条原始距离问题有条件的局部事实依赖（原文与 sample_token 保存在 JSON），允许最小离线 B 诊断。它们仍缺少唯一目标映射与测距误差界，不能在线替代，也不能作为 Planning 主路径。',
        '理论收益文件保留原生 Prefill/Decode 实测逐请求值，并单列删除全部语言工作的不可达上限；所有上限均不是实际加速。',
        '按用户要求不改写 M3 Prompt/Carrier，不强制制造状态命中；保留严格回退与旧风险协议。',
        '', '复现：`taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.task_dependency_audit`']
    (root/'task_dependency_audit.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/system_optimization_v4')
    p.add_argument('--data-root',default='data/nuscenes');p.add_argument('--previous',default='outputs/system_optimization_v3')
    args=p.parse_args();r=inventory(args.output,args.data_root,args.previous)
    print(json.dumps(dict(gate_a=r['gate_a'],vqa_screening=r['vqa_screening']),ensure_ascii=False))
