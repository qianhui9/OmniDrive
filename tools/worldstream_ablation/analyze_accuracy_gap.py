"""Explain historical-main versus ablation accuracy from existing question receipts."""
from collections import Counter
import json
from pathlib import Path

from .run import sha, write

MAIN = Path('outputs/m123_r0_only_reproduction_run01')
ABLATION = Path('outputs/worldstream_ablation_full_run01')


def events(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summary(part, select_signature=False):
    scored = [e for e in part if e['quality']['known'] and
              (not select_signature or e['signature'] is not None)]
    total = sum(e['quality']['score'] for e in scored)
    correct = sum(e['quality']['score'] == 1 for e in scored)
    return dict(requests=len(part), scored=len(scored), sum_score=total, fully_correct=correct,
                mean_score_percent=100*total/len(scored) if scored else None,
                binary_accuracy_percent=100*correct/len(scored) if scored else None)


def main():
    main_events = [e for e in events(MAIN/'per_question_metrics.jsonl') if e['arm'] == 'C']
    full = events(ABLATION/'worldstream/per_question.jsonl')
    key = lambda e: (e['scene_id'], e['frame_id'], e['question'])
    s4 = {key(e): e for e in main_events if e['workload_id'] == 'S4'}
    core = [e for e in full if e['group'] != 'other']
    assert len(core) == len(s4) == 2169 and {key(e) for e in core} == set(s4)
    assert all(e['answer'] == s4[key(e)]['answer'] for e in core)
    common = [e for e in core if s4[key(e)]['quality']['known']]
    assert len(common) == 1214
    assert all(e['quality']['known'] and e['quality']['score'] == s4[key(e)]['quality']['score'] for e in common)
    assert all(not s4[key(e)]['quality']['known'] for e in core if e['quality'].get('absence_label'))
    absence = [e for e in core if e['quality'].get('absence_label')]
    others_scored = [e for e in full if e['group'] == 'other' and e['signature'] is not None and e['quality']['known']]
    assert len(absence) == 694 and len(others_scored) == 9
    original_scenes = json.loads((MAIN/'scene_selection.json').read_text())['selected']
    manifest = json.loads((ABLATION/'manifest.json').read_text())
    assert [(s['scene_id'], s['tokens']) for s in original_scenes] == [(s['scene_id'], s['tokens']) for s in manifest['scenes']]
    variants = {}
    known_keys = {key(e) for e in common}
    for name in manifest['variants']:
        es = events(ABLATION/name/'per_question.jsonl')
        part = [e for e in es if e['group'] != 'other' and key(e) in known_keys]
        assert len(part) == 1214 and all(e['quality']['known'] and not e['quality'].get('absence_label') for e in part)
        variants[name] = dict(core_main_known_mask=summary(part),
                              ablation_expanded=summary(es, True))
    stages = dict(main_all_five_workloads=summary(main_events), main_S4=summary(list(s4.values())),
                  ablation_same_S4_known_queries=summary(common),
                  added_core_absence=summary(absence), ablation_core_expanded=summary(core, True),
                  added_other_supported=summary(others_scored), ablation_full_expanded=summary(full, True))
    absent_groups = {g: summary([e for e in absence if e['group'] == g]) for g in sorted({e['group'] for e in absence})}
    evidence = dict(status='DIAGNOSED_PROTOCOL_MISMATCH', inference_performed=False,
        main_report_value_percent=stages['main_all_five_workloads']['mean_score_percent'],
        ablation_value_percent=stages['ablation_full_expanded']['binary_accuracy_percent'],
        scenes_and_frame_order_identical=True, same_core_questions=2169, exact_same_core_answers=2169,
        same_known_core_scores=1214, stages=stages, absence_groups=absent_groups,
        diagnostic_all_variants=variants,
        sources={str(p):sha(p) for p in [MAIN/'per_question_metrics.jsonl', MAIN/'evidence_synthesis.json',
            MAIN/'scene_selection.json', ABLATION/'manifest.json', ABLATION/'worldstream/per_question.jsonl',
            Path('tools/worldstream_ablation/report.py'), Path('tools/worldstream_m123/frozen_main_report.py')]})
    write(ABLATION/'accuracy_gap_diagnosis.json', evidence)
    lines = ['# 主实验与消融实验 Factual QA Accuracy 差异溯源', '',
        '**结论：85.267% 与 58.007% 使用了不同的负载、评分范围和分母，不能直接比较。主要问题是评估协议未对齐，'
        '相同指标名称掩盖了任务变化；没有发现 Full WorldStream 在共同三问负载上推理退化的证据。**', '',
        '本次仅重分析现有逐问题收据，无新推理，未覆盖主实验和消融实验的既有指标或表格。', '',
        '## 数值分解', '',
        '|统计范围|分子|可评分分母|结果 (%)|', '|---|---:|---:|---:|',
        '|主实验五种负载合并，原 mean-score 口径|2630.5|3085|85.267|',
        '|主实验 S4：每帧三个固定事实问题|1047|1214|86.244|',
        '|消融 Full：同一 S4 可评分问题|1047|1214|86.244|',
        '|消融新增的核心无可见目标问题|57|694|8.213|',
        '|消融三个核心问题，加入无目标评分|1104|1908|57.862|',
        '|消融额外两个原始 QA 中进入事实主指标的问题|8|9|88.889|',
        '|消融最终事实主指标|1112|1917|58.007|', '',
        '精确桥接：85.267 → 86.244（固定 S4 负载），→ 57.862（增加 694 个无目标样本），'
        '→ 58.007（加入额外 QA 中 9 个受支持的可评分事实问题）。', '',
        '## 直接一致性证据', '',
        '- 两次评估使用相同 18 个 Scene、相同 723 帧及相同帧序；不是换了另一批场景。',
        '- 主实验 S4 与消融 Full 的三个固定事实问题逐帧配对，共 2169 个回答，**2169/2169 回答逐字相同**。',
        '- 原 S4 的 1214 个可评分问题，两次评分也逐项相同，均为 1047 个完全正确。',
        '- 因此共同三问负载上的落差并非权重损坏、模块未启用、GPU 驱动兼容处理或生成随机性造成。'
        'Full 组在驱动兼容补跑之前就已完成；兼容补跑仅用于两组 QCSE 替代方法。', '',
        '## 口径改变发生在哪里', '',
        '1. 主指标来源：`evidence_synthesis.json → answer_quality.conditional_known_answer_accuracy_C`。'
        '它合并 Every-4、Every-2、Every-1、Three-per-frame、Mixed-three 五种负载，'
        '5613 个查询中评分 3085 个，其余 2528 个 UNKNOWN。它不是某个单一固定负载的准确率。',
        '2. 主评分 `tools/worldstream_m123/fact_bypass_identity.py::question_target` 将 '
        '`no_visible_GT_target` 保持 UNKNOWN，不评分否定事实。',
        '3. 消融 `tools/worldstream_ablation/report.py::target_signature / grade_answer` 将部分 '
        '`no_visible_GT_target` 变为 `absent` 标签，并要求回答明确否定，原因字段为 '
        '`official_GT_no_visible_target`。这一扩展纳入了主实验没有评分的困难样本。',
        '4. 消融 `summarize_events` 只纳入有受支持 signature 的问题，并统计 `score == 1` 的二元准确率；'
        '主实验 `frozen_main_report.py::quality_summary` 对全部 known 问题取 score 均值，'
        '包含部分自然问题的有限事实部分分。主实验的 2630.5 是得分之和，不是完全正确答案数；'
        '其完全正确数为 2618，对应二元率 84.862%。这也是同名不同定义。', '',
        '## 新增无目标问题的分布', '',
        '|问题组|新增可评分数|完全正确数|准确率 (%)|', '|---|---:|---:|---:|']
    for group, result in absent_groups.items():
        lines.append(f"|{group}|{result['scored']}|{result['fully_correct']}|{result['binary_accuracy_percent']:.3f}|")
    lines += ['', '分类问题和位置问题各有 402、170 个新增无目标样本，完全正确数均为零。'
        '这两类原本询问目标类别或位置，现在额外要求在目标缺失时给出明确否定。'
        '它揭示当前模型在扩展任务上的不足，但不能用来认定原正目标事实任务从 85% 退化到 58%。', '',
        '“GT 查询区域内没有满足可见性条件的标注目标”也不自动等同于“图像中确定不存在该对象”。'
        '需要审查类别/距离/方位限制、标注覆盖与遮挡条件；无目标标签应作为单独、有明确边界的评估协议。', '',
        '## 各配置在主实验 S4 已知问题掩码下的诊断值', '',
        '下表仅用于说明评估口径差异，不替代现有正式结果。所有配置使用同一 1214 个问题，'
        '按原有非缺失目标事实评分统计。Stale 等其他指标未重定义。', '',
        '|配置|完全正确/1214|诊断准确率 (%)|现有扩展准确率 (%)|', '|---|---:|---:|---:|']
    for name, result in variants.items():
        old, new = result['core_main_known_mask'], result['ablation_expanded']
        lines.append(f"|{name}|{old['fully_correct']}/1214|{old['binary_accuracy_percent']:.3f}|{new['binary_accuracy_percent']:.3f}|")
    lines += ['', '## 修正方向', '',
        '1. 固定同一个主评估协议，并对所有方法统一使用同一问题清单、评分器和 GT 掩码。'
        '对当前收据可先离线重算“原 S4 正目标事实准确率”，Full 应为 86.244%，不是强行对齐为 85.27%。',
        '2. 若要与原报告的 85.267% 严格同负载比较，所有消融配置都必须运行原五种负载并按同样权重汇总。'
        '现有每帧五问的收据不包含其他负载下各配置的执行轨迹，不能靠复制或重加权现有回答补造结果。',
        '3. 保留 58.007% 作为扩展协议的历史结果，改用清晰名称标识“含无目标问题的二元事实准确率”；'
        '将无目标/否定回答能力单独列为补充评估。主表和消融表必须明确列出共同协议。',
        '4. 所有 14 组统一重算，而非只改 Full 单元格；原始答案和旧报告保留，新的协议在执行前固定。', '',
        '证据及未舍入数值见 `accuracy_gap_diagnosis.json`。', '']
    (ABLATION/'准确率差异溯源.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stages=stages, identical_answers=2169, common_known=1214), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
