"""Offline, audited main-protocol scoring for completed ablation/stability runs.

Original frozen inference sources, manifests, answers and timings are never edited.
Reports and derived metrics are archived before their current versions are updated.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np

from .registry import GROUPS, VARIANTS
from .run import ROOT, read, sha, write
from tools.environment_stability_eval.metrics import accuracy, score_rows

MAIN = Path('outputs/m123_r0_only_reproduction_run01')
ABLATION = Path('outputs/worldstream_ablation_full_run01')
STABILITY = Path('tools/environment_stability_eval/outputs/run_20261117_113951_740642')
VERSION = 'main_protocol_v1'


def records(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(path, rows):
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def table(headers, rows):
    return ['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|',
            *['|'+'|'.join(map(str, r))+'|' for r in rows], '']


def fmt(value, digits=3):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def archive(root, files, timestamp):
    destination = root/'protocol_history'/timestamp
    destination.mkdir(parents=True)
    hashes = {}
    for f in files:
        if not f.exists():
            continue
        relative = f.relative_to(root)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        hashes[str(relative)] = sha(target)
    write(destination/'archive.json', dict(reason='Before unifying with main benchmark statistics', files=hashes))
    return destination


def quality(events):
    result = accuracy(events)
    known = [e for e in events if e['scorable']]
    return dict(**result, score_sum=sum(e['score'] for e in known),
        fully_correct=sum(e['correct'] for e in known),
        Factual_QA_Accuracy_percent=result['overall_accuracy_percent'],
        Stale_Answer_Rate_percent=result['stale_answer_rate_percent'],
        factual_correct=sum(e['correct'] for e in known), factual_scored=len(known),
        factual_unscored=len(events)-len(known), questions=len(events),
        changed_query_denominator=result['queries_after_changes'],
        unknown_reasons=dict(Counter(e['quality'].get('reason') or 'unscorable' for e in events if not e['scorable'])))


def throughput(rows, samples=2000):
    """Use exactly main FPS/QPS denominators; cluster entire scenes for intervals."""
    values = {}
    for r in rows:
        values.setdefault(r['scene_id'], np.zeros(3))
        values[r['scene_id']] += (r['e2e_ms'], 1, len(r['questions']))
    ms, frames, queries = np.stack(list(values.values())).sum(axis=0)
    result = dict(FPS=1000*frames/ms, QPS=1000*queries/ms, E2E_seconds=ms/1000)
    if len(values) >= 2:
        v = np.stack(list(values.values()))
        rng = np.random.default_rng(20261030)
        sums = v[rng.integers(0, len(v), size=(samples, len(v)))].sum(axis=1)
        for field, col in [('FPS', 1), ('QPS', 2)]:
            result[field+'_CI95'] = np.quantile(1000*sums[:, col]/sums[:, 0], [.025, .975]).tolist()
    return result


def lean(row):
    requests = row['requests']
    keep = ('executor', 'direct_answer', 'llm_skipped', 'draft_candidate_tokens', 'draft_accepted_tokens',
            'source_frame', 'reused_tokens', 'decode_forward_calls')
    return dict(scene_id=row['scene_id'], frame_id=row['frame_id'], questions=row['questions'],
                groups=row['groups'], answers=row['answers'], e2e_ms=row['e2e_ms'],
                timings=row['timings'], executed_macs=row['executed_macs'],
                ogsm_precheck_commit_ms=row.get('ogsm_precheck_commit_ms'),
                ogsm_including_adapters_ms=row.get('ogsm_including_adapters_ms'),
                requests=[{k:q[k] for k in keep if k in q} for q in requests])


def load_verified(root, name, expected):
    done = read(root/name/'inference_complete.json')
    assert done['status'] == 'COMPLETED' and not done['smoke']
    digest = hashlib.sha256()
    rows = []
    with (root/name/'receipts.jsonl').open('rb') as handle:
        for line in handle:
            digest.update(line)
            rows.append(lean(json.loads(line)))
    if digest.hexdigest() != done['receipts_sha256']:
        raise RuntimeError('Receipt hash mismatch: '+str(root/name))
    if [r['frame_id'] for r in rows] != expected or len(rows) != done['frames']:
        raise RuntimeError('Wrong frame workload: '+str(root/name))
    if read(root/name/'checkpoint_validation.json')['status'] != 'PASS':
        raise RuntimeError('Checkpoint validation failed: '+str(root/name))
    return rows, digest.hexdigest()


def execution_metrics(rows, native):
    nqueries = sum(len(r['questions']) for r in rows)
    out = dict(frames=len(rows), **throughput(rows),
        E2E_Latency_ms=float(np.mean([r['e2e_ms'] for r in rows])),
        P95_E2E_Latency_ms=float(np.percentile([r['e2e_ms'] for r in rows], 95)))
    for label, prefix in [('visual', 'Visual'), ('prefill', 'LLM_Prefill'), ('decode', 'LLM_Decode')]:
        values = [r['timings'].get(label, {}).get('cuda_ms', 0.) for r in rows]
        out[prefix+'_Latency_ms'] = sum(values)/len(rows)
        out[prefix+'_Latency_ms_per_query'] = sum(values)/nqueries
        if label == 'visual':
            out['P95_Visual_Backbone_Latency_ms'] = float(np.percentile(values, 95))
            out['Visual_Backbone_Latency_ms'] = out[prefix+'_Latency_ms']
    for label in ['visual', 'llm']:
        actual = sum(r['executed_macs'][label] for r in rows)
        baseline = sum(r['executed_macs'][label] for r in native)
        prefix = 'Visual' if label == 'visual' else 'LLM'
        out[prefix+'_Compute_Reduction_percent'] = 100*(1-actual/baseline)
        out['actual_'+label+'_MACs'], out['native_'+label+'_MACs'] = actual, baseline
    present = [r['ogsm_including_adapters_ms'] for r in rows if r['ogsm_including_adapters_ms'] is not None]
    out['OGSM_Latency_ms'] = float(np.mean(present)) if len(present) == len(rows) else (0. if not present and all(r['ogsm_precheck_commit_ms'] == 0 for r in rows) else None)
    out['OGSM_Precheck_Commit_Only_ms'] = float(np.mean([r['ogsm_precheck_commit_ms'] for r in rows]))
    requests = [q for r in rows for q in r['requests']]
    skipped = sum(q.get('llm_skipped', False) for q in requests)
    state_direct = sum(q.get('direct_answer', q.get('executor') == 'QuestionReferencedFactReuse') for q in requests)
    proposed = sum(q.get('draft_candidate_tokens', 0) for q in requests)
    accepted = sum(q.get('draft_accepted_tokens', 0) for q in requests)
    out.update(LLM_Skip_Rate_percent=100*skipped/nqueries,
        Direct_Answer_Rate_percent=100*state_direct/nqueries,
        Candidate_Token_Acceptance_Rate_percent=100*accepted/proposed if proposed else None,
        proposed_candidate_tokens=proposed, accepted_candidate_tokens=accepted)
    return out


def render_ablation(root, metrics, status, history, comparison, old_extra):
    full = metrics['worldstream']
    acc, stale, end = 'Factual_QA_Accuracy_percent', 'Stale_Answer_Rate_percent', 'E2E_Latency_ms'
    def label(n):
        return 'WorldStream / Full' if n == 'worldstream' else VARIANTS[n].label
    def cells(names, fields):
        return [[label(n), *[fmt(metrics[n][k]) for k in fields]] for n in names]
    lines = ['# WorldStream 消融实验结果（统一主实验评估口径）', '',
        f"更新时间：{status['generated_at']}。**14/14 组推理完整，当前结果为既有真实回答的离线重统计，无新增推理。**", '',
        '本次统一评分器、UNKNOWN 规则、平均得分公式、FPS/QPS 分母和模块时延单位。'
        '旧的含无目标二元评分报告已归档到 `'+str(history.relative_to(root))+'`，原始推理收据及 manifest 保持原字节。', '',
        '## 评估协议与负载差异', '',
        '- **Factual QA Accuracy = 100 × 主实验原评分器在全部可评分查询上的平均事实得分。** '
        '包括原始 QA 的有限事实部分分；缺失可见目标、标注或语义歧义保留 UNKNOWN，不新增负例。完全正确率另列。',
        '- 每组仍为 18 个 Scene / 723 帧 / 3615 次查询（三个固定事实问题加两个原始 QA）。'
        '统计公式已统一，但现有负载不等于主实验五负载合并；不能强行把 Full 改成历史 85.267%。',
        '- Stale 为共同正目标/原始 QA 规则下当前 GT 已变化、当前错误且同答符合前次有效 GT 的比例；'
        '沿用环境稳定性实验的可比 GT 状态与前次有效历史，GT UNKNOWN 不覆盖历史。'
        '原主实验没有定义 Stale、变化响应或 MACs 主指标，这些仍为明确标识的补充指标。',
        '- FPS = 帧数 / ΣE2E 秒；QPS = 查询数 / ΣE2E 秒，与主实验一致。E2E 包含每帧全部五问、预处理和答案导出，'
        '包含 Scene 冷首帧，排除权重加载与离线评分。不能从五问 E2E 扣减得到三问或其他负载的反事实时延。',
        '- OGSM 使用主实验的含适配器时延；LLM Prefill / Decode 统一按全部视觉帧平均，单位 ms/帧。'
        '原 ms/查询 和 Precheck+Commit 时延在机器可读结果中保留为辅助值；原始 CUDA 计时跨度保持不变。',
        '- 计算量依然来自实际执行 dense MACs，与同批 Native 配对；原主实验未提供该 MACs 计数，不能把它与历史 Forward 次数或 GPU 时延混为同一指标。', '',
        '### 主结果、共同负载与当前五问负载', '']
    lines += table(['范围', '可评分数', 'WorldStream 平均事实得分 (%)', '解释'], comparison)
    lines += ['## 1. Module Ablation', '']
    flags = {'native':'×|×|×', 'worldstream':'✓|✓|✓', 'without_ogsm':'×|✓*|✓*', 'without_davp':'✓|×|✓', 'without_qcse':'✓|✓|×'}
    lines += table(['配置', 'OGSM', 'DAVP', 'QCSE', 'Factual QA Accuracy (%)', 'Δ vs Full (百分点)'],
        [[label(n), *flags[n].split('|'), fmt(metrics[n][acc]), f"{metrics[n][acc]-full[acc]:+.3f}"] for n in GROUPS['module']])
    lines += ['* w/o OGSM 的 DAVP 使用相邻图像差异；QCSE 禁用状态事实直答，保留独立候选验证。', '',
        '## 2. Alternative Design — OGSM', '']
    lines += table(['配置','Factual QA Accuracy (%)','Stale Answer Rate (%)','OGSM含适配器 (ms/帧)','E2E (ms/帧)'],
        cells(GROUPS['ogsm'], [acc, stale, 'OGSM_Latency_ms', end]))
    lines += ['## 3. Alternative Design — DAVP', '',
        'Periodic 和 Feature-Drift 采用相同逐摄像头预算 3 个前缀窗口及最大复用年龄 8 帧。', '']
    lines += table(['配置','Visual Backbone (ms/帧)','Visual Compute Reduction (%)','Factual QA Accuracy (%)','E2E (ms/帧)','P95 Visual (ms/帧)'],
        cells(GROUPS['davp'], ['Visual_Backbone_Latency_ms','Visual_Compute_Reduction_percent', acc, end,'P95_Visual_Backbone_Latency_ms']))
    lines += ['## 4. Alternative Design — QCSE', '']
    lines += table(['配置','Factual QA Accuracy (%)','Stale Answer Rate (%)','Prefill (ms/帧)','Decode (ms/帧)','E2E (ms/帧)','LLM Compute Reduction (%)'],
        cells(GROUPS['qcse'], [acc, stale,'LLM_Prefill_Latency_ms','LLM_Decode_Latency_ms',end,'LLM_Compute_Reduction_percent']))
    lines += ['Prefill/Decode 按每帧全部五问累计后平均；不是单请求服务时间，不含 Python 采样循环的全部开销。', '',
        '### 辅助执行指标', '',
        '主实验 FAST coverage 对应同时跳过 Prefill 和 Decode 的查询比例；普通答案缓存命中也跳过 LLM。'
        '“世界状态事实直答率”只计状态直答，因此与 LLM 跳过率分别报告。候选未启用时为 N/A。', '']
    lines += table(['配置','世界状态事实直答率 (%)','LLM跳过率 (%)','候选接受率 (%)','接受/提出'],
        [[label(n),fmt(metrics[n]['Direct_Answer_Rate_percent']),fmt(metrics[n]['LLM_Skip_Rate_percent']),
          fmt(metrics[n]['Candidate_Token_Acceptance_Rate_percent']),
          f"{metrics[n]['accepted_candidate_tokens']}/{metrics[n]['proposed_candidate_tokens']}" if metrics[n]['proposed_candidate_tokens'] else '—（未启用）'] for n in GROUPS['qcse']])
    lines += ['## 5. 分母、完全正确率及共同核心问题', '']
    lines += table(['配置','得分和/可评分','UNKNOWN','完全正确率 (%)','三问已知GT得分 (%)','过期/变化机会'],
        [[label(n),f"{fmt(m['score_sum'])}/{m['scorable_queries']}",m['unknown_queries'],
          fmt(m['fully_correct_accuracy_percent']), fmt(m['core_main_accuracy_percent']),
          f"{m['stale_answers']}/{m['queries_after_changes']}"] for n,m in metrics.items()])
    lines += ['## 6. 吞吐与模块辅助时延', '']
    lines += table(['配置','FPS','QPS','E2E (ms/帧)','OGSM Precheck+Commit (ms/帧)','Prefill (ms/查询)','Decode (ms/查询)'],
        cells(list(metrics), ['FPS','QPS',end,'OGSM_Precheck_Commit_Only_ms','LLM_Prefill_Latency_ms_per_query','LLM_Decode_Latency_ms_per_query']))
    lines += ['## 7. 变化场景分层', '', '离线 GT 场景标签可重叠，使用主实验正目标/原始 QA 评分器；不同标签的分母不能相加。', '']
    for tag, title in [('vehicle_motion','车辆运动'),('object_disappearance','对象消失'),('local_occlusion','局部遮挡'),('ego_motion','自车运动')]:
        lines += [f'### {title}', '']
        lines += table(['配置','Factual QA Accuracy (%)','得分和/可评分','Stale (%)','过期/机会'],
            [[label(n),fmt(m['scenarios'][tag]['overall_accuracy_percent']),
              f"{fmt(m['scenarios'][tag]['score_sum'])}/{m['scenarios'][tag]['scorable_queries']}",
              fmt(m['scenarios'][tag]['stale_answer_rate_percent']),
              f"{m['scenarios'][tag]['stale_answers']}/{m['scenarios'][tag]['queries_after_changes']}"] for n,m in metrics.items()])
    lines += ['## 8. 本次结果与适用范围', '',
        f"WorldStream 当前五问平均事实得分为 **{full[acc]:.3f}%**，相比 Native 增加 "
        f"**{full[acc]-metrics['native'][acc]:.3f} 个百分点**。共同三问已知 GT 范围为 "
        f"**{full['core_main_accuracy_percent']:.3f}%**，与主实验 S4 相同。历史五负载合并 85.267% 保持原值。", '',
        '质量与效率逐项比较（正值的 Accuracy 差表示 Full 更高；负值的时延节省表示 Full 更慢）：', '']
    lines += table(['比较对象','Full Accuracy差 (百分点)','Full Stale差 (百分点)','Full E2E降低 (%)'],
        [[label(n),f"{full[acc]-m[acc]:+.3f}",f"{full[stale]-m[stale]:+.3f}",f"{100*(1-full[end]/m[end]):+.3f}"] for n,m in metrics.items() if n != 'worldstream'])
    lines += ['Full R0 采用 current_only 事实直答，历史状态主要通过视觉和候选路径影响回答；QA 不能独立证明状态有效性。'
        'DAVP 前缀的两个相同分区窗口层使区域映射和依赖闭包可能重合，替代设计还改变全局残差路径，不能单独归因于闭包。'
        '所有结果为固定暴露验证集、单次运行，未增加独立 TEST 或多 seed 证据。', '',
        '主报告未定义的 Stale、响应延迟与 dense MACs 是补充指标；本次统一公式不冒充补跑原五负载。'
        '旧 58.007% 为含无目标问题的另一套二元评估，已归档，不再作为主口径表中的 Accuracy。', '',
        '本次重统计证据：`main_protocol_v1/audit.json`；未舍入指标：`metrics.json`；四张 CSV 已同步更新。'
        '逐问题新评分见各组 `per_question.jsonl`，旧评分在 protocol_history 中完整保留。', '',
        '重统计命令：', '', '```bash',
        'taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\\n  /home/qh/miniconda3/envs/OmniDrive/bin/python -B -m tools.worldstream_ablation.unify_main_protocol', '```', '',
        '该入口保留已有版本，重复运行会拒绝覆盖 `main_protocol_v1`；重新分析时需显式创建新协议版本。', '']
    if old_extra:
        lines += ['以下独立环境稳定性/Change Response Delay 补充结果已沿用相同主评分器，'
                  '其事件分母与消融主表分开保留。', '', old_extra]
    (root/'结果.md').write_text('\n'.join(lines)+'\n')
    (root/'ABLATION_REPORT.md').write_text('# WorldStream 消融实验报告\n\n'
        '14/14 组完成，当前采用统一主实验评分口径。详见 [结果.md](结果.md)。\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.chdir(ROOT)
    if {4,5} & os.sched_getaffinity(0):
        raise RuntimeError('Use taskset -c 0-3,6-31')
    if any((r/VERSION).exists() for r in [ABLATION, STABILITY]):
        raise FileExistsError('Protocol output already exists; refusing to overwrite analysis evidence')
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    generated = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    # Check both frozen manifests, sharing a digest cache for identical inputs.
    checked = {}
    def verify(path, expected):
        key = str(Path(path).resolve())
        if key not in checked:
            checked[key] = sha(path)
        if checked[key] != expected:
            raise RuntimeError('Changed frozen source/data: '+str(path))
    am, em = read(ABLATION/'manifest.json'), read(STABILITY/'manifest.json')
    for manifest, groups in [(am, ['source_hashes','data_hashes']), (em,['source_and_GT_hashes','input_hashes'])]:
        for group in groups:
            for path, expected in manifest[group].items():
                verify(path, expected)
        print('VERIFIED original manifest inputs', flush=True)
    for name, expected in em['frozen_artifact_hashes'].items():
        verify(STABILITY/name, expected)
    verify(am['checkpoint'], am['checkpoint_sha256'])
    verify(MAIN/'current_gt_annotations.json', read(MAIN/'experiment_protocol.json')['GT_sha256'])
    expected = [t for scene in am['scenes'] for t in scene['tokens']]
    assert expected == [t for scene in em['scenes'] for t in scene['tokens']]
    objects = read(MAIN/'current_gt_annotations.json')
    gt = records(STABILITY/'query_ground_truth.jsonl')
    truth_by_frame = {}
    for q in gt:
        truth_by_frame.setdefault(q['frame_id'], []).append(q)
    tags = read(ABLATION/'scenario_tags.json')
    metrics, fingerprints, scored, lean_rows = {}, {}, {}, {}
    for name in am['variants']:
        rows, fingerprints[name] = load_verified(ABLATION, name, expected)
        converted = []
        for row in rows:
            truth = truth_by_frame[row['frame_id']]
            if row['questions'] != [q['question'] for q in truth]:
                raise RuntimeError('Different query schedule: '+name)
            converted.append(dict(row, method=name, repeat=0, timestamp=truth[0]['timestamp'],
                window_id='not_used', stability_level='not_used'))
        es = score_rows(converted, gt, objects)
        assert len(es) == 3615
        scored[name] = es
        lean_rows[name] = rows
        m = quality(es)
        m.update(execution_metrics(rows, lean_rows['native']))
        core = quality([e for e in es if e['group'] != 'other'])
        m.update(Factual_QA_Accuracy_percent=m['overall_accuracy_percent'],
                 Stale_Answer_Rate_percent=m['stale_answer_rate_percent'],
                 core_main_accuracy_percent=core['overall_accuracy_percent'], core_main_scored=core['scorable_queries'],
                 scenarios={tag:quality([e for e in es if tag in tags[e['frame_id']]])
                            for tag in sorted({t for ts in tags.values() for t in ts})})
        m['original_QA_limited_fact_score_percent'] = quality([e for e in es if e['group']=='other'])['overall_accuracy_percent']
        assert m['scorable_queries'] == 1969 and core['scorable_queries'] == 1214
        metrics[name] = m
        print(name, 'mean score', m['overall_accuracy_percent'], 'core', core['overall_accuracy_percent'],
              'Stale', m['stale_answer_rate_percent'], flush=True)
    # Verify matching core outputs/labels against original main S4, not just a rounded number.
    main_events = [e for e in records(MAIN/'per_question_metrics.jsonl') if e['arm'] == 'C' and e['workload_id'] == 'S4']
    index = {(e['frame_id'], e['question']):e for e in main_events}
    core = [e for e in scored['worldstream'] if e['group'] != 'other']
    assert all(index[e['frame_id'], e['question']]['answer'] == e['prediction'] for e in core)
    assert all(index[e['frame_id'], e['question']]['quality'] == json.loads(json.dumps(e['quality'])) for e in core)
    # Existing environment experiment already uses main scoring: independently regrade and check it.
    environment_throughput = []
    original_environment = records(STABILITY/'scored_queries.jsonl')
    for name in ['native','worldstream']:
        rows, digest = load_verified(STABILITY, name, expected)
        fingerprints['stability/'+name] = digest
        converted = [dict(r, method=name, repeat=0, timestamp=truth_by_frame[r['frame_id']][0]['timestamp'],
                          window_id='not_used', stability_level='not_used') for r in rows]
        es = score_rows(converted, gt, objects)
        old = [e for e in original_environment if e['method'] == name]
        assert [(e['scorable'],e['score'],e['stale_answer']) for e in es] == [(e['scorable'],e['score'],e['stale_answer']) for e in old]
        for level in ['S1','S2','S3','S4','Unclassified','ALL']:
            window_levels = {r['frame_id']:r['stability_level'] for r in old}
            part = [r for r in rows if level == 'ALL' or window_levels[r['frame_id']] == level]
            environment_throughput.append(dict(method=name, stability_level=level, **throughput(part)))
    print('VERIFIED stability grading is already identical to main evaluator', flush=True)
    for root in [ABLATION, STABILITY]:
        (root/VERSION).mkdir()
    abfiles = [f for f in ABLATION.iterdir() if f.is_file() and (f.suffix in ['.md','.csv'] or f.name in ['metrics.json','report_status.json','completion.json'])]
    abfiles += [ABLATION/n/f for n in am['variants'] for f in ['metrics.json','per_question.jsonl']]
    ah = archive(ABLATION, abfiles, timestamp)
    eh = archive(STABILITY, [STABILITY/f for f in ['结果.md','experiment_summary.json','latency_summary.csv','completion.json']], timestamp)
    old_text = (ABLATION/'结果.md').read_text()
    old_extra = old_text[old_text.index('## 9.'): ] if '## 9.' in old_text else ''
    synthesis = read(MAIN/'evidence_synthesis.json')['answer_quality']
    comparison = [
        ['主实验：五负载合并', synthesis['scored_per_arm'],fmt(100*synthesis['conditional_known_answer_accuracy_C']), '历史正式值，负载重复权重不同'],
        ['主实验：S4 三问/帧',1214,fmt(100*sum(e['quality']['score'] for e in main_events if e['quality']['known'])/1214),'同一核心问题范围'],
        ['消融：共同三问/帧',1214,fmt(metrics['worldstream']['core_main_accuracy_percent']),'2169 个核心回答与原 S4 逐字一致'],
        ['消融：实际五问/帧',1969,fmt(metrics['worldstream']['overall_accuracy_percent']),'主评分器，含原始 QA 部分分'],
        ['环境稳定性：实际五问/帧',1969,fmt(quality([e for e in original_environment if e['method']=='worldstream'])['overall_accuracy_percent']),'原来已使用主评分器，值不变']]
    status = dict(status='COMPLETE', generated_at=generated, protocol=VERSION,
        original_manifest_sha256=sha(ABLATION/'manifest.json'), new_inference_calls=0,
        planned_variants=14, completed_variants=14,
        definition='Original main grade; 100*mean score over known queries, including limited original QA fractional scores; no new absence labels',
        throughput_definition='FPS=frames/sum(frame E2E seconds); QPS=queries/sum(frame E2E seconds)',
        raw_receipts_sha256=fingerprints, scorer_sha256=sha('tools/worldstream_m123/continuous_stream_report.py'),
        script_sha256=sha(__file__), history=str(ah), verified_inputs=len(checked),
        load_scope='Existing five-query stream, not historical five-workload aggregate',
        variants={n:dict(status='COMPLETED',frames=m['frames'],queries=m['queries'],scorable=m['scorable_queries']) for n,m in metrics.items()})
    write(ABLATION/VERSION/'audit.json',status)
    write(ABLATION/'report_status.json',status)
    write(ABLATION/'metrics.json',dict(protocol=VERSION,definitions=status,metrics=metrics))
    for name,m in metrics.items():
        write(ABLATION/name/'metrics.json',m)
        dump(ABLATION/name/'per_question.jsonl',scored[name])
    for group,names in GROUPS.items():
        save_csv(ABLATION/(group+'.csv'),[dict(variant=n,label=VARIANTS[n].label,status='COMPLETED',
            **{k:v for k,v in metrics[n].items() if not isinstance(v,(dict,list))}) for n in names])
    save_csv(ABLATION/VERSION/'protocol_comparison.csv', [dict(scope=r[0],scored=r[1],accuracy=r[2],note=r[3]) for r in comparison])
    render_ablation(ABLATION,metrics,status,ah,comparison,old_extra)
    completed = read(ABLATION/'completion.json')
    completed.update(report='结果.md',statistics_protocol=VERSION,statistics_audit=VERSION+'/audit.json')
    write(ABLATION/'completion.json',completed)
    update_environment(STABILITY,environment_throughput,comparison,eh,status)
    print('UPDATED',ABLATION/'结果.md',STABILITY/'结果.md',flush=True)


def update_environment(root, new_throughput, comparison, history, audit):
    summary = read(root/'experiment_summary.json')
    lookup = {(r['method'],r['stability_level']):r for r in new_throughput}
    for row in summary['latency']:
        row.update(lookup[row['method'],row['stability_level']])
    summary['main_protocol'] = dict(version=VERSION, accuracy_verified_identical=True,
        FPS='frames / sum(frame E2E seconds)', QPS='queries / sum(frame E2E seconds)',
        original_sustained_fps='Retained as supplementary processing-segment-wall throughput',
        original_accuracy_and_stale_and_response_unchanged=True,
        comparison_to_main_load='Same grade and mean-score formula; five-query load differs from original five workload aggregate')
    write(root/'experiment_summary.json',summary)
    save_csv(root/'latency_summary.csv',summary['latency'])
    write(root/VERSION/'audit.json',dict(audit,history=str(history),scope='Stability source scoring verified unchanged; primary FPS/QPS aligned with main',
        supplemental_response='Existing Change Response Delay values retained; main experiment has no identical response metric'))
    save_csv(root/VERSION/'throughput.csv',new_throughput)
    text = (root/'结果.md').read_text()
    start = text.index('## 4.')
    end = text.index('## 5.',start)
    lat = ['## 4. 延迟与吞吐：统一主实验分母', '',
        '**主 FPS/QPS 改为主对比实验定义：帧数或查询数 / 累计帧 E2E 秒数。** E2E 测量值及边界不变。'
        '区间仍按完整 Scene 聚类 bootstrap 2000 次，不把五个查询看作独立序列。', '']
    lat += table(['等级','方法','帧','平均 E2E ms [95%CI]','P95 ms [95%CI]','FPS [95%CI]','QPS [95%CI]'],
        [[r['stability_level'],r['method'],r['frames'],
          fmt(r['average_e2e_ms'],2)+' ['+fmt(r['average_e2e_ms_ci95_low'],2)+', '+fmt(r['average_e2e_ms_ci95_high'],2)+']',
          fmt(r['p95_e2e_ms'],2)+' ['+fmt(r['p95_e2e_ms_ci95_low'],2)+', '+fmt(r['p95_e2e_ms_ci95_high'],2)+']',
          fmt(r['FPS'],4)+' ['+', '.join(fmt(v,4) for v in r['FPS_CI95'])+']',
          fmt(r['QPS'],4)+' ['+', '.join(fmt(v,4) for v in r['QPS_CI95'])+']'] for r in summary['latency']])
    lat += ['累计处理区间墙钟还包含收据等工具开销，与主实验 E2E 分母不同。下表仅保留为辅助指标，'
            '不再将其称为主 FPS；原所有 E2E/吞吐置信区间另保存在归档和机器可读结果中。', '']
    lat += table(['等级','方法','辅助持续吞吐 (FPS)','处理区间秒'],
        [[r['stability_level'],r['method'],fmt(r['sustained_fps'],4),fmt(r['measured_processing_seconds'],2)] for r in summary['latency']])
    text = text[:start]+'\n'.join(lat)+'\n'+text[end:]
    a,b = lookup['native','ALL'],lookup['worldstream','ALL']
    olda = next(r for r in summary['latency'] if r['method']=='native' and r['stability_level']=='ALL')
    oldb = next(r for r in summary['latency'] if r['method']=='worldstream' and r['stability_level']=='ALL')
    # Only the primary summary label/figures change; keep supplementary throughput honest.
    for line in text.splitlines():
        if line.startswith('| 持续吞吐 |'):
            text=text.replace(line,f"| 主 FPS（累计 E2E 分母） | {a['FPS']:.4f} | {b['FPS']:.4f} | FPS |\n"
                f"| 主 QPS（累计 E2E 分母） | {a['QPS']:.4f} | {b['QPS']:.4f} | QPS |\n"
                f"| 辅助持续吞吐（处理区间墙钟） | {olda['sustained_fps']:.4f} | {oldb['sustained_fps']:.4f} | FPS |")
    text=text.replace('持续吞吐提高 34.78%',f"主 FPS 提高 {100*(b['FPS']/a['FPS']-1):.2f}%（辅助持续吞吐提高 34.78%）")
    intro = ['## 统计口径更新（与主对比实验统一）', '',
        f"更新时间：{audit['generated_at']}。本次无新增推理。旧文档及吞吐表已归档到 `{history.relative_to(root)}`。", '',
        '本实验原准确率已经使用主评分器及 mean-score 公式，本次逐问题重新评分核验后保持不变：'
        '**WorldStream 84.459%（得分和/1969），共同三问已知 GT 为 86.244%**。'
        '完全正确率仍是独立辅助列；主实验 85.267% 来自五负载合并，不能直接当成本实验相同负载的参照。', '',
        '本次统一主 FPS/QPS 的累计 E2E 分母；原处理区间持续吞吐仍保留为辅助值。'
        'Stable/Changed、Stale、MACs 和 Change Response Delay 是主报告没有完全对应定义的补充指标，'
        '继续沿用已冻结的 GT/时间域/实际执行计数规则，不混成主实验已有指标。', '']
    intro += table(['范围','可评分数','WorldStream平均事实得分 (%)','比较范围'],comparison)
    text=text.replace('\n## 1.', '\n'+'\n'.join(intro)+'\n## 1.',1)
    (root/'结果.md').write_text(text)


if __name__ == '__main__':
    main()
