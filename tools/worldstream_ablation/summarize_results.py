"""Audit and score completed configurations without claiming an incomplete suite passed.

Uses the frozen report.py evaluators, never changes the inference manifest or receipts,
and intentionally does not write the suite's completion.json.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import pickle
from zoneinfo import ZoneInfo

from .registry import GROUPS, VARIANTS
from .report import (grade_answer, scenario_tags, summarize, summarize_events,
                     target_signature)
from .run import read, sha, write


def evaluate(out):
    manifest = read(out / 'manifest.json')
    if manifest['smoke']:
        raise ValueError('This summary requires a full run, not smoke results')
    print('Verifying all frozen source, assets, input data and checkpoint hashes', flush=True)
    verified = {}
    for label in ('source_hashes', 'data_hashes'):
        for path, digest in manifest[label].items():
            if sha(path) != digest:
                raise RuntimeError('Frozen input changed: ' + path)
        verified[label] = len(manifest[label])
        print('Verified', label, verified[label], flush=True)
    if sha(manifest['checkpoint']) != manifest['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    print('Verified checkpoint', flush=True)
    tokens = [t for s in manifest['scenes'] for t in s['tokens']]
    rows, status = {}, {}
    for name in manifest['variants']:
        directory = out / name
        marker = directory / 'inference_complete.json'
        if not marker.exists():
            failure = directory / 'exception.json'
            status[name] = dict(status='FAILED' if failure.exists() else 'NOT_STARTED',
                                error=read(failure)['error'] if failure.exists() else None)
            continue
        complete = read(marker)
        if (complete['status'] != 'COMPLETED' or complete['smoke'] or
                complete['receipts_sha256'] != sha(directory / 'receipts.jsonl')):
            raise RuntimeError('Incomplete or changed inference: ' + name)
        part = [json.loads(line) for line in (directory / 'receipts.jsonl').read_text().splitlines()]
        if [r['frame_id'] for r in part] != tokens or len(part) != complete['frames']:
            raise RuntimeError('Missing, repeated or reordered frames: ' + name)
        if any(r['variant'] != name for r in part):
            raise RuntimeError('Wrong variant receipt: ' + name)
        if read(directory / 'checkpoint_validation.json')['status'] != 'PASS':
            raise RuntimeError('Strict checkpoint validation did not pass: ' + name)
        for row in part:
            if not (len(row['questions']) == len(row['answers']) == len(row['groups'])
                    and len(row['requests']) in (0, len(row['questions']))):
                raise RuntimeError('Missing question/answer/route: ' + name)
        rows[name] = part
        status[name] = dict(status='COMPLETED', frames=len(part),
                            questions=sum(len(r['questions']) for r in part),
                            receipts_sha256=complete['receipts_sha256'])
    if 'native' not in rows:
        raise RuntimeError('A complete Native reference is required')
    for name, part in rows.items():
        for native, row in zip(rows['native'], part):
            if any(native[k] != row[k] for k in ('scene_id', 'frame_id', 'frame_index', 'questions', 'groups')):
                raise RuntimeError('Unpaired workload: ' + name)
    print('Verified complete variants:', ', '.join(rows), flush=True)
    from tools.worldstream_m123.answer_validity_quality import load_current_annotations
    objects = load_current_annotations(manifest['infos'], set(tokens), 'data/nuscenes')
    with Path(manifest['infos']).open('rb') as handle:
        infos = {i['token']: i for i in pickle.load(handle)['infos'] if i['token'] in objects}
    tags = {}
    for scene in manifest['scenes']:
        previous = None
        for token in scene['tokens']:
            tags[token] = scenario_tags(previous, token, objects, infos)
            previous = token
    write(out / 'scenario_tags.json', tags)
    metrics = {}
    for name, part in rows.items():
        events, history = [], {}
        for row in part:
            token = row['frame_id']
            for qi, (question, answer, group) in enumerate(zip(row['questions'], row['answers'], row['groups'])):
                key = (row['scene_id'], question)
                signature = target_signature(question, objects[token])
                quality = grade_answer(question, answer, token, objects)
                prev = history.get(key)
                change = bool(prev and prev['signature'] is not None and signature is not None
                              and prev['signature'] != signature)
                stale = bool(change and quality['known'] and quality['score'] < 1 and
                             grade_answer(question, answer, prev['token'], objects).get('score') == 1)
                source = row['requests'][qi].get('source_frame') if row['requests'] else None
                source_sig = target_signature(question, objects[source]) if source in objects else None
                source_changed = bool(source_sig is not None and signature is not None and source_sig != signature)
                events.append(dict(variant=name, scene_id=row['scene_id'], frame_id=token,
                    question=question, answer=answer, group=group, signature=signature,
                    quality=quality, change_opportunity=change, stale=stale,
                    previous_frame=prev['token'] if prev else None, scenario_tags=tags[token],
                    cache_source_frame=source, cache_source_changed=source_changed,
                    cache_source_stale=bool(source_changed and quality['known'] and quality['score'] < 1
                        and grade_answer(question, answer, source, objects).get('score') == 1)))
                history[key] = dict(token=token, signature=signature)
        metric = summarize(part, events, rows['native'])
        metric['factual_correct'] = sum(e['signature'] is not None and e['quality']['known']
                                       and e['quality']['score'] == 1 for e in events)
        metric['scenarios'] = {tag: summarize_events([e for e in events if tag in e['scenario_tags']])
                              for tag in sorted({t for ts in tags.values() for t in ts})}
        metric['question_groups'] = {group: summarize_events([e for e in events if e['group'] == group])
                                    for group in sorted({e['group'] for e in events})}
        metrics[name] = metric
        with (out / name / 'per_question.jsonl').open('w') as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        write(out / name / 'metrics.json', metric)
        print(name, 'Accuracy', metric['Factual_QA_Accuracy_percent'], 'Stale',
              metric['Stale_Answer_Rate_percent'], flush=True)
    audit = dict(status='PARTIAL' if len(rows) != len(manifest['variants']) else 'COMPLETE',
                 generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                 planned_variants=len(manifest['variants']), completed_variants=len(rows),
                 variants=status, verified=verified, checkpoint_verified=True,
                 manifest_sha256=sha(out / 'manifest.json'),
                 evaluator_sha256=sha(Path(__file__).with_name('report.py')),
                 summary_script_sha256=sha(__file__))
    write(out / 'report_status.json', audit)
    write(out / 'metrics.json', dict(smoke=False, report_status=audit['status'], metrics=metrics,
          definitions={k: manifest[k] for k in ('compute_definition', 'accuracy_definition', 'stale_definition')}))
    return manifest, metrics, audit, tags


def render(out, manifest, metrics, audit, tags):
    def value(name, field):
        v = metrics.get(name, {}).get(field)
        return 'N/A' if v is None else f'{v:.3f}'

    def table(headers, records):
        return ['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|',
                *['|' + '|'.join(map(str, row)) + '|' for row in records], '']

    def label(name, full='WorldStream'):
        return full if name == 'worldstream' else VARIANTS[name].label

    acc = 'Factual_QA_Accuracy_percent'
    stale = 'Stale_Answer_Rate_percent'
    e2e = 'E2E_Latency_ms'
    full = metrics['worldstream']
    native = metrics['native']
    missing = [n for n, s in audit['variants'].items() if s['status'] != 'COMPLETED']
    if not missing:
        status_lines = [f"**整套任务已完成：{audit['completed_variants']}/{audit['planned_variants']} 组。** "
                        'Module、OGSM、DAVP、QCSE 四类结果均已通过完整性核验。', '']
        if (out / 'recovery_history').exists():
            status_lines += ['原队列的失败记录和恢复启动审计保留在 `recovery_history/`；下表使用各组完整的正式推理结果。', '']
    else:
        status_lines = [f"**整套任务未全部完成：计划 {audit['planned_variants']} 组，已验证完成 {audit['completed_variants']} 组。**", '',
                        '缺失配置：' + '、'.join(label(n) for n in missing) + '。缺失指标记为 N/A，不使用 smoke 或估计值补齐。', '']
        for name in missing:
            error = audit['variants'][name].get('error')
            if error:
                status_lines += [f"- {label(name)}：`{error.replace('`', '')}`；详见该组 `exception.json`。", '']
    lines = ['# WorldStream 消融实验结果', '',
        f"结果目录：`{out}`。汇总时间：{audit['generated_at']}（Asia/Shanghai）。", '',
        *status_lines,
        '## 评估范围与指标口径', '',
        f"- 每个已完成配置使用相同的 {len(manifest['scenes'])} 个完整 Scene、{manifest['frames']} 帧、"
        f"{native['questions']} 个查询；每帧 3 个固定事实问题及 2 个原始 QA 问题。随机种子 `{manifest['seed']}`，单次运行，无多种子误差条。",
        '- Factual QA Accuracy：仅对受支持且当前 GT 可判定的事实问题，统计完全正确回答比例；原始 QA 的有限事实分数单列，不能解释为通用开放问答准确率。',
        '- Stale Answer Rate：同 Scene 内同问题的相邻观测中，GT 回答语义发生已知变化后，当前错误回答仍完全符合前次 GT 的比例。它衡量行为，不直接判断错误由缓存引起；分母在后文列出。',
        '- OGSM / Visual Backbone / E2E 时延单位均为 ms/帧；E2E 包含本帧全部 5 个查询、预处理及导出，包含各 Scene 首帧，但不含模型加载。LLM Prefill / Decode 为实际前向 CUDA 跨度，按全部查询平均（ms/查询），缓存命中和直答贡献零。',
        '- Visual Backbone 为 CUDA 流跨度，包含窗口调度/缓存导致的主机空隙；同时报告 P95。计算量按真实执行的 Linear / Conv / Attention 形状统计 dense MACs，相对相同负载 Native 归一化，包含输出长度差异；不含逐元素、内存搬移和采样，不以窗口复用率代替计算节省。',
        '- 原 manifest 中全部源码/资产、输入数据和 checkpoint 哈希，以及每组完成标记、原始收据 SHA256、严格权重加载和帧/问题配对均已通过核验。原始 manifest、收据及失败日志保留不变。', '',
        '## 1. Module Ablation', '']
    flags = {'native': ['×', '×', '×'], 'worldstream': ['✓', '✓', '✓'],
             'without_ogsm': ['×', '✓*', '✓*'], 'without_davp': ['✓', '×', '✓'],
             'without_qcse': ['✓', '✓', '×']}
    lines += table(['实验配置', 'OGSM', 'DAVP', 'QCSE', 'Factual QA Accuracy (%)', '相对 WorldStream Δ (百分点)'],
        [[label(n), *flags[n], value(n, acc), f"{metrics[n][acc] - full[acc]:+.3f}"] for n in GROUPS['module']])
    lines += ['* 关闭 OGSM 后，DAVP 使用相邻帧图像差异；QCSE 禁用世界状态事实直答，保留独立候选 Token 验证。', '',
        f"WorldStream 相对 Native 的准确率变化为 **{full[acc] - native[acc]:+.3f} 个百分点**；"
        f"相对变化 **{100 * (full[acc] / native[acc] - 1):+.3f}%**。各移除实验的实际差值如表；"
        '差值为负表示移除后下降，差值为正表示移除后提高，不预设三个模块均已得到准确率支持。', '',
        '## 2. Alternative Design — OGSM', '']
    ogfields = [acc, stale, 'OGSM_Latency_ms', e2e]
    lines += table(['实验配置', 'Factual QA Accuracy (%)', 'Stale Answer Rate (%)', 'OGSM Latency (ms/帧)', 'E2E Latency (ms/帧)'],
        [[label(n, 'Full OGSM'), *[value(n, f) for f in ogfields]] for n in GROUPS['ogsm']])
    lines += ['## 3. Alternative Design — DAVP', '',
        'Periodic 与 Feature-Drift 采用相同的逐摄像头刷新预算（3 个前缀窗口/帧）与最大复用年龄（8 帧）。', '']
    davfields = ['Visual_Backbone_Latency_ms', 'Visual_Compute_Reduction_percent', acc, e2e,
                 'P95_Visual_Backbone_Latency_ms']
    lines += table(['实验配置', 'Visual Backbone Latency (ms/帧)', 'Visual Compute Reduction (%)',
                    'Factual QA Accuracy (%)', 'E2E Latency (ms/帧)', 'P95 Visual Backbone Latency (ms/帧)'],
        [[label(n, 'Full DAVP'), *[value(n, f) for f in davfields]] for n in GROUPS['davp']])
    lines += [f"Full DAVP 实际减少视觉 MACs **{full['Visual_Compute_Reduction_percent']:.3f}%**；"
        f"相对关闭 DAVP，视觉时延减少 **{100 * (1 - full['Visual_Backbone_Latency_ms'] / metrics['without_davp']['Visual_Backbone_Latency_ms']):.3f}%**，"
        f"但事实准确率下降 **{metrics['without_davp'][acc] - full[acc]:.3f} 个百分点**。"
        'Region-to-Window Mapping 在本次负载上的实际计算量节省为 0%，不能仅按策略名称认定减少了 GPU 计算。', '']
    lines += ['## 4. Alternative Design — QCSE', '']
    qcfields = [acc, stale, 'LLM_Prefill_Latency_ms', 'LLM_Decode_Latency_ms', e2e, 'LLM_Compute_Reduction_percent']
    lines += table(['实验配置', 'Factual QA Accuracy (%)', 'Stale Answer Rate (%)', 'LLM Prefill Latency (ms/查询)',
                    'LLM Decode Latency (ms/查询)', 'E2E Latency (ms/帧)', 'LLM Compute Reduction (%)', '运行状态'],
        [[label(n, 'Full QCSE'), *[value(n, f) for f in qcfields], audit['variants'][n]['status']]
         for n in GROUPS['qcse']])
    lines += ['辅助指标：Direct Answer Rate 仅计状态事实直答，不将普通历史答案缓存命中计入；'
              'Candidate Token Acceptance Rate 无候选提案时为 N/A。', '']
    lines += table(['配置', 'Direct Answer Rate (%)', 'Candidate Token Acceptance Rate (%)', '接受/提出候选 Token'],
        [[label(n, 'Full QCSE'), value(n, 'Direct_Answer_Rate_percent'), value(n, 'Candidate_Token_Acceptance_Rate_percent'),
          f"{metrics[n]['accepted_candidate_tokens']}/{metrics[n]['proposed_candidate_tokens']}" if n in metrics else 'N/A']
         for n in GROUPS['qcse']])
    exact = metrics['qcse_exact_cache']
    lines += [f"Full QCSE 比 Exact Answer Cache 的事实准确率高 **{full[acc] - exact[acc]:.3f} 个百分点**，"
        f"但 E2E 时延增加 **{100 * (full[e2e] / exact[e2e] - 1):.3f}%**。"
        f"当前行为性 Stale 指标为 Full **{full[stale]:.3f}%（{full['stale_answers']}/{full['changed_query_denominator']}）**、"
        f"Exact **{exact[stale]:.3f}%（{exact['stale_answers']}/{exact['changed_query_denominator']}）**，"
        '**本次结果不支持 Full QCSE 降低该口径的过期率这一结论**。', '',
        'Stale 主指标只识别符合前次 GT 的错误回答；更早的缓存事实若与前次 GT 也不一致，不一定计入。'
        '因此低 Stale 不能替代高 Accuracy，也不能覆盖所有历史过期事实。独立的缓存来源诊断如下，分母不同，不能直接替换主指标：', '']
    lines += table(['配置', '当前 GT 与所记录缓存来源 GT 不同的查询数', '其中仍符合缓存来源 GT 的错误回答比例 (%)'],
        [[label(n, 'Full QCSE'), metrics[n]['reused_changed_source_denominator'] if n in metrics else 'N/A',
          value(n, 'stale_on_reused_changed_source_percent')] for n in GROUPS['qcse']])
    lines += ['该诊断依赖收据顶层 `source_frame`，缺少这一来源记录时为 N/A；分母为 0 不表示已证明没有任何历史来源错误。', '']
    lines += ['## 5. 样本分母与原始结果', '']
    lines += table(['配置', '完成帧/查询', '事实完全正确/可评分', '排除查询数', '过期回答/GT变化机会', '原始 QA 有限事实分数 (%)'],
        [[label(n), f"{m['frames']}/{m['questions']}", f"{m['factual_correct']}/{m['factual_scored']}",
          m['factual_unscored'], f"{m['stale_answers']}/{m['changed_query_denominator']}",
          value(n, 'original_QA_limited_fact_score_percent')] for n, m in metrics.items()])
    lines += ['排除查询包括不属于受支持事实语法和当前 GT 不确定的问题；不等于模型答错数。'
              '逐问题答案、GT 语义、评分理由和过期判定见各配置的 `per_question.jsonl`。', '',
              '## 6. 变化场景分层', '',
              '标签由离线 GT 生成，未用于推理或事后挑样；标签可以重叠，不能相加作为独立样本总数。', '']
    scenarios = [('vehicle_motion', '车辆运动'), ('object_disappearance', '对象消失'),
                 ('local_occlusion', '局部遮挡'), ('ego_motion', '自车运动')]
    counts = Counter(t for ts in tags.values() for t in ts)
    lines += table(['场景', '标签', '覆盖帧数'], [[cn, tag, counts[tag]] for tag, cn in scenarios])
    for tag, cn in scenarios:
        lines += [f'### {cn}', '']
        lines += table(['实验配置', 'Factual QA Accuracy (%)', '正确/可评分', 'Stale Answer Rate (%)', '过期/变化机会'],
            [[label(n), f"{m['scenarios'][tag][acc]:.3f}" if m['scenarios'][tag][acc] is not None else 'N/A',
              f"{round((m['scenarios'][tag][acc] or 0) * m['scenarios'][tag]['factual_scored'] / 100)}/{m['scenarios'][tag]['factual_scored']}",
              f"{m['scenarios'][tag][stale]:.3f}" if m['scenarios'][tag][stale] is not None else 'N/A',
              f"{m['scenarios'][tag]['stale_answers']}/{m['scenarios'][tag]['changed_query_denominator']}"]
             for n, m in metrics.items()])
    lines += ['## 7. 结果比较与可支持的结论', '',
        f"1. 整体系统：WorldStream 相对 Native 准确率增加 {full[acc] - native[acc]:.3f} 个百分点，"
        f"E2E 时延减少 {100 * (1 - full[e2e] / native[e2e]):.3f}%。",
        f"2. Module：去掉 OGSM 或 QCSE 后准确率分别下降 {full[acc] - metrics['without_ogsm'][acc]:.3f}、"
        f"{full[acc] - metrics['without_qcse'][acc]:.3f} 个百分点；去掉 DAVP 后准确率提高 "
        f"{metrics['without_davp'][acc] - full[acc]:.3f} 个百分点，不能声称三个模块均提高准确率。",
        '3. OGSM：完整设计在本次三个替代方案中获得最高准确率、最低行为性 Stale；其状态维护时延明显高于 Tracking 和 Geometry，历史状态有效性仍需独立标注评估。',
        '4. DAVP：完整设计比 Periodic 和 Feature-Drift 获得更高准确率、更低视觉时延和更多实际 MACs 节省；相对关闭 DAVP 及 Region Mapping 存在准确率损失，且没有得到更低的平均 E2E 时延。',
        '5. QCSE：Full 比 Exact Cache 更准确，但 Exact Cache 更快、LLM 计算量更少，且当前 Stale 主指标更低；'
        + ('尚有配置缺失，' if any(n in missing for n in GROUPS['qcse']) else '')
        + '不能据此声称 Full 全面优于所有替代方案。其余替代设计按 QCSE 表中的实测指标判断。', '',
        '与各配置的未舍入指标计算得到的差值：', '']
    for name in ['without_ogsm', 'without_davp', 'without_qcse', 'ogsm_tracking', 'ogsm_geometry',
                 'ogsm_flow', 'davp_periodic', 'davp_feature_drift', 'davp_region_mapping', 'qcse_exact_cache',
                 'qcse_semantic_cache', 'qcse_prompt_kv']:
        if name not in metrics:
            continue
        m = metrics[name]
        lines.append(f"- Full 相对 {label(name)}：Accuracy {full[acc] - m[acc]:+.3f} 个百分点，"
                     f"Stale {full[stale] - m[stale]:+.3f} 个百分点，"
                     f"E2E 时延减少 {100 * (1 - full[e2e] / m[e2e]):+.3f}%（负值表示 Full 更慢）。")
    lines += ['', '这些差异是单次、固定 18 场景上的描述性结果，不提供统计显著性结论。'
        'OGSM 的现行 Full R0 采用 current_only 事实直答，历史验证主要经视觉执行及候选路径影响回答；'
        'QA/Stale 差异不能独立证明世界状态有效性判断能力。', '',
        'DAVP 当前可缓存前缀为两个相同分区的独立窗口层，区域映射与逐层依赖闭包在此拓扑中可能重合；'
        '替代方案还禁用了全局锚点残差复用，所以本表不能单独归因于依赖闭包。', '',
        ('QCSE 尚有缺失配置，需要先补齐推理。' if any(n in missing for n in GROUPS['qcse']) else 'QCSE 的所有替代配置已齐全。')
        + '若要对总体收益或机制进行论文级归因，还需扩大场景并报告多次运行或按 Scene 的不确定性。', '',
        '## 8. 可复核产物', '',
        '- `module.csv`、`ogsm.csv`、`davp.csv`、`qcse.csv`：四类结果，缺失配置保留状态和空指标。',
        '- `metrics.json`：未舍入指标、实际执行 MACs、分场景/问题组统计和分母。',
        f"- `report_status.json`：{audit['completed_variants']}/{audit['planned_variants']} 完成状态、输入/源码/权重核验及报告脚本哈希；恢复入口仅在全部评分通过后写入整套 `completion.json`。",
        '- `<variant>/receipts.jsonl`：原始推理证据；`<variant>/per_question.jsonl`：离线逐问题评分。',
        '- `manifest.json`：运行前固定的场景、负载、种子和哈希，未修改。', '',
        '重算本报告（只执行 CPU 离线评分，不重新推理）：', '', '```bash',
        'taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \\\n  /home/qh/miniconda3/envs/OmniDrive/bin/python -B -m tools.worldstream_ablation.summarize_results \\\n  --output outputs/worldstream_ablation_full_run01', '```', '']
    for group, names in GROUPS.items():
        columns = sorted({k for m in metrics.values() for k, v in m.items() if not isinstance(v, (dict, list))})
        with (out / (group + '.csv')).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['variant', 'label', 'status', *columns])
            writer.writeheader()
            for name in names:
                writer.writerow(dict(variant=name, label=label(name), status=audit['variants'][name]['status'],
                                     **{k: metrics.get(name, {}).get(k) for k in columns}))
    (out / '结果.md').write_text('\n'.join(lines) + '\n')
    (out / 'ABLATION_REPORT.md').write_text('# WorldStream 消融实验报告\n\n'
        f"完整分类结果和评估口径见 [结果.md](结果.md)。当前核验完成 {audit['completed_variants']}/{audit['planned_variants']} 组。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest, metrics, audit, tags = evaluate(args.output)
    render(args.output, manifest, metrics, audit, tags)
    print('Wrote', args.output / '结果.md', 'suite status:', audit['status'], flush=True)


if __name__ == '__main__':
    main()
