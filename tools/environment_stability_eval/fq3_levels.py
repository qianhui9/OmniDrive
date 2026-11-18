"""Offline S1-S4 x per-query Stable/Changed metrics for completed FQ-3 runs.

Reuses pre-inference GT thresholds and the complete chronological response
tracking. Does not rerun models or change historical artifacts.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from .common import (LEVELS, METHODS, csv_write, dump_jsonl, jsonl, read,
                     require_affinity, write)
from .metrics import accuracy, cluster_intervals
from .change_response import statistics, paired_statistics

LEVEL_NAMES = dict(S1='Highly Stable', S2='Moderately Stable',
                   S3='Dynamic', S4='Highly Dynamic', Unclassified='Incomplete tail window', ALL='All measured frames')


def aggregate(scored, events, frames, samples=2000, seed=20261030):
    """Group completed events by onset level, not by filtered follow-up streams."""
    rows, pairs = [], []
    for level in (*LEVELS, 'Unclassified', 'ALL'):
        select = lambda e: level == 'ALL' or e['stability_level'] == level
        level_events = [e for e in events if select(e)]
        pairs.append(dict(stability_level=level, **paired_statistics(level_events)))
        for method in METHODS:
            qs = [e for e in scored if e['method'] == method and select(e)]
            ds = [e for e in level_events if e['method'] == method]
            fs = [f for f in frames[method] if select(f)]
            stable = [e for e in qs if e['scorable'] and e['fact_status'] == 'Stable']
            changed = [e for e in qs if e['scorable'] and e['fact_status'] == 'Changed']
            stale = [e for e in qs if e['stale_eligible']]
            a, d = accuracy(qs), statistics(ds)
            if len(ds) != len(changed):
                raise ValueError('Changed GT/event count mismatch')
            # CIs use the scenes actually contributing to each metric, rather
            # than allowing stable-only scenes to create a spurious Changed CI.
            a.update(cluster_intervals(stable, accuracy, ['stable_fact_accuracy_percent'], samples=samples, seed=seed))
            a.update(cluster_intervals(changed, accuracy, ['changed_fact_accuracy_percent'], samples=samples, seed=seed))
            a.update(cluster_intervals(stale, accuracy, ['stale_answer_rate_percent'], samples=samples, seed=seed))
            d.update(cluster_intervals(ds, statistics,
                ['response_delay_mean_ms', 'observed_response_rate_percent'], samples=samples, seed=seed))
            rows.append(dict(stability_level=level, level_name=LEVEL_NAMES[level], method=method,
                frames=len(fs), query_scenes=len({e['scene_id'] for e in qs}),
                stable_scenes=len({e['scene_id'] for e in stable}),
                changed_scenes=len({e['scene_id'] for e in changed}),
                stable_score_sum=sum(e['score'] for e in stable), stable_fully_correct=sum(e['correct'] for e in stable),
                changed_score_sum=sum(e['score'] for e in changed), changed_fully_correct=sum(e['correct'] for e in changed),
                **a, **d))
    # The unclassified TAIL level differs from a query lacking valid temporal GT.
    # Both must remain explicit, and ALL must reconcile to all 723 real frames.
    for method in METHODS:
        total = next(r for r in rows if r['method'] == method and r['stability_level'] == 'ALL')
        parts = [r for r in rows if r['method'] == method and r['stability_level'] != 'ALL']
        for key in ('frames', 'queries', 'scorable_queries', 'unknown_queries', 'stable_queries',
                    'changed_queries', 'queries_after_changes', 'stale_answers', 'changed_events',
                    'responded_events', 'censored_events', 'superseded_events', 'horizon_exhausted_events',
                    'stale_outputs', 'stable_fully_correct', 'changed_fully_correct'):
            if sum(r[key] for r in parts) != total[key]:
                raise ValueError('Level totals do not reconcile: ' + key)
    return rows, pairs


def verify_level_assignments(windows, manifest):
    from copy import deepcopy
    from .stability_classifier import assign_levels
    thresholds = manifest['thresholds']['thresholds']
    expected = assign_levels(deepcopy(windows), thresholds)
    for a, b in zip(windows, expected):
        if a['stability_level'] != b['stability_level']:
            raise ValueError('Saved level differs from frozen GT threshold')
    calibration_ids = set(manifest['thresholds']['calibration_scene_ids'])
    evaluation_ids = {s['scene_id'] for s in manifest['scenes']}
    if calibration_ids & evaluation_ids:
        raise ValueError('Calibration/evaluation scenes overlap')
    return thresholds


def _normal(value):
    return json.loads(json.dumps(value, allow_nan=False))


def generate(source, output, samples=None):
    from .evaluator import verify_manifest
    from .fq3 import PROTOCOL, check_core, strict_transitions
    from .metrics import score_rows, validate_pair
    from .change_response import change_events
    from .stability_classifier import distribution, fit_thresholds
    from tools.worldstream_m123.r0_task_evaluation import sha
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new level-report directory: ' + str(output))
    completion = read(source/'completion.json')
    if completion['status'] != 'COMPLETED' or completion['smoke'] or completion['protocol'] != PROTOCOL:
        raise ValueError('Requires the completed true FQ-3 run')
    print('VERIFY_COMPLETED_FQ3_INPUTS_AND_RECEIPTS', flush=True)
    if sha(source/'experiment_summary.json') != completion['report_sha256']:
        raise ValueError('Original FQ-3 summary changed')
    for name, digest in completion['artifacts'].items():
        if sha(source/name) != digest:
            raise ValueError('Original FQ-3 artifact changed: ' + name)
    manifest = verify_manifest(source)
    if manifest['config']['other_questions'] != 0:
        raise ValueError('Original QA must not be included')
    windows = list(jsonl(source/'test_windows.jsonl'))
    thresholds = verify_level_assignments(windows, manifest)
    cal = list(jsonl(source/'calibration_windows.jsonl'))
    recomputed = fit_thresholds(cal, manifest['config']['threshold_quantiles'], manifest['config']['thresholds'])
    if recomputed['thresholds'] != thresholds:
        raise ValueError('Frozen calibration quartiles do not reproduce')
    frames, runtimes = {}, {}
    for method in METHODS:
        folder = source/method
        status = read(folder/'inference_complete.json')
        if status['status'] != 'COMPLETED' or any(sha(folder/n) != status[k] for n,k in (
                ('receipts.jsonl','receipts_sha256'),('processing_segments.json','segments_sha256'))):
            raise ValueError('Modified/incomplete receipt')
        frames[method] = list(jsonl(folder/'receipts.jsonl'))
        if len(frames[method]) != completion['frames_per_method']:
            raise ValueError('Missing completed frames')
        check_core(frames[method])
        runtimes[method] = read(folder/'effective_runtime.json')
    validate_pair(manifest, windows, frames, runtimes)
    gt = list(jsonl(source/'query_ground_truth.jsonl'))
    if strict_transitions(gt, manifest['scenes'], manifest['config']['max_transition_gap_seconds']) != gt:
        raise ValueError('Saved FQ-3 adjacent-known labels do not reproduce')
    objects = {r['frame_id']:r['objects'] for r in jsonl(source/'evaluation_objects.jsonl')}
    scored = [q for method in METHODS for q in score_rows(frames[method],gt,objects)]
    if _normal(scored) != list(jsonl(source/'scored_queries.jsonl')):
        raise ValueError('Regraded queries do not match original completed scores')
    # Track on FULL streams before grouping; a response may legitimately occur
    # in a different GT window/level from its originating change.
    events = change_events(scored,frames,objects,manifest['config']['response_horizon_seconds'],
                           manifest['config']['max_transition_gap_seconds'])
    if _normal(events) != list(jsonl(source/'change_response_events.jsonl')):
        raise ValueError('Chronological response events do not reproduce')
    samples = manifest['config']['bootstrap_samples'] if samples is None else samples
    if samples < 0:
        raise ValueError('Bootstrap count must be nonnegative')
    print('AGGREGATE_S1_S4_CORE_METRICS', flush=True)
    rows, pairs = aggregate(scored,events,frames,samples,manifest['config']['seed'])
    original = read(source/'experiment_summary.json')
    for method in METHODS:
        row = next(r for r in rows if r['method']==method and r['stability_level']=='ALL')
        old = next(r for r in original['primary'] if r['method']==method and r['environment_state']=='ALL')
        old_response = next(r for r in original['change_response'] if r['method']==method and r['cohort']=='ALL_CHANGED')
        for key in ('stable_fact_accuracy_percent','changed_fact_accuracy_percent','stale_answer_rate_percent'):
            if row[key] != old[key]:
                raise ValueError('Grouped totals differ from original metric: ' + key)
        for key in ('response_delay_mean_ms','responded_events','changed_events'):
            if row[key] != old_response[key]:
                raise ValueError('Grouped response differs from original metric: ' + key)
    dist = distribution(windows,thresholds)
    output.mkdir(parents=True, exist_ok=False)
    csv_write(output/'core_metrics_by_stability.csv',rows)
    csv_write(output/'paired_response_by_stability.csv',pairs)
    csv_write(output/'stability_distribution.csv',dist)
    dump_jsonl(output/'change_response_events.jsonl',events)
    csv_write(output/'raw_qa_results.csv', [{k:v for k,v in q.items() if k!='quality'} for q in scored])
    figures = plot_core(output,rows)
    consumed = ['manifest.json','completion.json','experiment_summary.json','query_ground_truth.jsonl',
        'evaluation_objects.jsonl','test_windows.jsonl','calibration_windows.jsonl',
        'stability_thresholds.json','scored_queries.jsonl','change_response_events.jsonl']
    consumed += [f'{m}/{n}' for m in METHODS for n in
        ('receipts.jsonl','inference_complete.json','processing_segments.json','effective_runtime.json')]
    code = [Path(__file__),Path(__file__).with_name('metrics.py'),Path(__file__).with_name('change_response.py'),
            Path(__file__).with_name('stability_classifier.py')]
    report = dict(status='COMPLETED', source=str(source), new_model_inference=False,
        generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        protocol=PROTOCOL, bootstrap_samples=samples, thresholds=thresholds,
        calibration_scene_count=len(manifest['thresholds']['calibration_scene_ids']),
        calibration_frames=read(source/'data_coverage.json')['calibration_frames'],
        calibration_complete_windows=len([w for w in cal if w['complete'] and w['change_rate'] is not None]),
        definitions=dict(level_axis='Frozen 5s GT semantic-rate windows; calibration quartiles, not model accuracy/delay.',
            fact_axis='FQ-3 same-query adjacent-known GT; Stable/Changed distinct from S1-S4 window levels.',
            response='First fully correct response on original complete chronological stream; grouped by change onset level; sampled-video ms, not live wall time.',
            censoring='Unknown GT, missing query, observation gap, scene end stop follow-up; later state change supersedes. Unresolved=null.',
            confidence_interval='Scene-cluster bootstrap on metric-eligible scenes only; <2 contributing scenes gives N/A.',
            pooling='Ratio of summed numerators/denominators and event-weighted response mean; never unweighted mean of S1-S4 percentages.'),
        verified=dict(source_artifacts=True,frozen_calibration_thresholds=True,pairing=True,
            regraded_queries_match=True,response_events_match=True,all_level_totals_reconcile=True),
        metrics=rows, paired_response=pairs, distribution=dist,figures=figures,
        source_hashes={name:sha(source/name) for name in consumed},
        implementation_hashes={str(p.resolve()):sha(p) for p in code})
    write(output/'experiment_summary.json',report)
    markdown(output,report)
    write(output/'completion.json',dict(status='COMPLETED',new_model_inference=False,
        generated_at=report['generated_at'],source=str(source),
        output_hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('FQ3_FOUR_LEVEL_REPORT_COMPLETED',output,flush=True)
    return report


def plot_core(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    panels=[('stable_fact_accuracy_percent','Stable Factual QA Accuracy (%)'),
        ('changed_fact_accuracy_percent','Changed Factual QA Accuracy (%)'),
        ('stale_answer_rate_percent','Stale Answer Rate (%)'),
        ('response_delay_mean_ms','Change Response Delay (sampled-video ms)')]
    with plt.rc_context({'font.size':10,'pdf.fonttype':42}):
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for ax,(key,title) in zip(axes.flat,panels):
            for method,offset,color in [('native',-.19,'#777777'),('worldstream',.19,'#1b9e77')]:
                part=[next(r for r in rows if r['method']==method and r['stability_level']==level) for level in LEVELS]
                values=[r[key] if r[key] is not None else np.nan for r in part]
                ax.bar(np.arange(4)+offset,values,width=.36,label=method,color=color)
                for i,r in enumerate(part):
                    if r[key] is None:
                        ax.text(i+offset,.04,'N/A',transform=ax.get_xaxis_transform(),ha='center',rotation=90,fontsize=8)
            ax.set_title(title)
            ax.set_xticks(range(4),LEVELS)
            if key!='response_delay_mean_ms': ax.set_ylim(0,110)
            ax.grid(axis='y',alpha=.2); ax.legend()
        counts='; '.join(f'{level}={next(r for r in rows if r["method"]=="native" and r["stability_level"]==level)["changed_queries"]}' for level in LEVELS)
        fig.suptitle('True FQ-3 run; Changed n: '+counts+'\nDelay is conditional on observed responses; response counts in report',fontsize=11)
        fig.tight_layout()
        names=[]
        for suffix in ('png','pdf'):
            name='four_level_core_metrics.'+suffix
            fig.savefig(output/name,dpi=180,bbox_inches='tight'); names.append(name)
        plt.close(fig)
        return names


def markdown(output,report):
    def fmt(v,n=2): return 'N/A' if v is None else f'{v:.{n}f}'
    def table(headers,records):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
                         ['| '+' | '.join(map(str,r))+' |' for r in records])+'\n'
    rows=report['metrics']; core=[r for r in rows if r['stability_level'] in LEVELS]
    lookup={(r['stability_level'],r['method']):r for r in rows}
    n4,w4=lookup['S4','native'],lookup['S4','worldstream']
    n3,w3=lookup['S3','native'],lookup['S3','worldstream']
    def core_table(selected):
        return table(['等级','方法','Stable Accuracy(%)','Stable查询数','Changed Accuracy(%)','Changed查询数',
            'Stale Answer Rate(%)','过期/机会','Change Response Delay均值(ms)','已响应/变化事件'],[
            [r['stability_level'],r['method'],fmt(r['stable_fact_accuracy_percent']),r['stable_queries'],
             fmt(r['changed_fact_accuracy_percent']),r['changed_queries'],fmt(r['stale_answer_rate_percent']),
             f'{r["stale_answers"]}/{r["queries_after_changes"]}',fmt(r['response_delay_mean_ms']),
             f'{r["responded_events"]}/{r["changed_events"]}'] for r in selected])
    threshold=report['thresholds']
    intervals=[f'[0, {threshold[0]:.9f}]',f'({threshold[0]:.9f}, {threshold[1]:.9f}]',
               f'({threshold[1]:.9f}, {threshold[2]:.9f}]',f'({threshold[2]:.9f}, 1]']
    lines=['# FQ-3 四种环境稳定性等级：四项核心指标', '',
        '**状态：COMPLETED。基于已完成的真实三问推理离线分层统计，无需重跑模型。**', '',
        '原始FQ-3实验：`'+report['source']+'`。原有模型回答、GT、整体Stable/Changed结果和完成摘要保持不变。', '',
        '## 两个独立分组轴', '',
        '- **S1–S4** 描述5秒环境窗口的GT语义变化率，标签在模型推理前冻结。S1中仍可发生某条问题所需事实的变化，S4中也可有大量Stable查询。',
        '- **Stable / Changed** 描述具体FQ-3问题在相邻已知GT间所需回答事实是否变化；不把一个窗口的所有查询统一设为Stable或Changed。首帧、UNKNOWN和时间断点不强行分类。',
        '- FQ-3始终是q0车辆类别、q1车辆存在性、q2最近行人位置，不添加原始生成QA。', '',
        '## 1. 四等级定义与样本分布', '',
        f'沿用{report["calibration_scene_count"]}个其他val场景/{report["calibration_frames"]}帧、'
        f'{report["calibration_complete_windows"]}个完整校准窗口的GT变化率25%、50%、75%分位数；'
        '已复算校验，不用模型输出、性能或新分组结果调整阈值。同属既有val，不是独立TEST。', '',
        '`R_change = changed atomic facts / valid atomic fact transitions`；原子事实包括标注集合存在性、类别、非空属性、可见ego方向和对象对相对方向。'
        '该环境语义分组与FQ-3问题的可评分GT范围不同，标注集合进出不等于物理出现/消失的完整QA认证。', '',
        table(['等级','名称','变化率区间（显示舍入）','窗口','帧','变化率均值','中位数','实测最小','实测最大','有效原子转换','变化原子事实'],[
            [r['stability_level'],LEVEL_NAMES[r['stability_level']],intervals[i] if i<4 else '不足5秒尾窗',
             r['windows'],r['frames'],fmt(r['change_rate_mean'],6),fmt(r['change_rate_median'],6),
             fmt(r['change_rate_min'],6),fmt(r['change_rate_max'],6),r['valid_fact_transitions'],r['changed_facts']]
            for i,r in enumerate(report['distribution'])]),
        '未舍入阈值：`'+str(threshold)+'`。', '',
        table(['等级','appearance','disappearance','relative_position','category','attribute','spatial_relation'],[
            [r['stability_level']]+[r[k+'_events'] for k in
            ('appearance','disappearance','relative_position','category','attribute','spatial_relation')]
            for r in report['distribution']]),
        '上表为GT环境原子事件计数，不是FQ-3可评分Changed查询数。', '',
        '## 2. S1–S4 × 四项核心指标（论文主表）', '',core_table(core),
        'Stable Accuracy与Changed Accuracy：既有事实评分器在对应可评分查询上的平均得分；原始完全正确计数同时保存。'
        'Stale：变化帧当前回答得分<1且对变化之前GT得分=1。Delay只对观察到完全正确响应的事件求均值，必须结合响应率/删失阅读。', '',
        '## 3. Change Response Delay：响应覆盖与连续历史错误', '',
        '`T_response=t_first_correct-t_change`。使用视频采样时间，非在线墙钟。按变化起点等级分组，完整跨窗追踪不截断；'
        '跟踪期间GT再次变化、UNKNOWN、缺问、时间断点、场景结束及5秒上限各有标记。未响应为N/A，不能填0。', '',
        table(['等级','方法','响应率%','即时正确数','Median(ms)','P95(ms)','删失','再变化','观察上限',
               '历史错误输出','最长连续历史错误次数','最长段跨度下界均值ms'],[
            [r['stability_level'],r['method'],fmt(r['observed_response_rate_percent']),r['immediate_response_events'],
             fmt(r['response_delay_median_ms']),fmt(r['response_delay_p95_ms']),r['censored_events'],
             r['superseded_events'],r['horizon_exhausted_events'],r['stale_outputs'],r['max_consecutive_stale_outputs'],
             fmt(r['mean_max_stale_observed_span_ms'])] for r in core]),
        '历史错误在每个后续帧对**本次变化前冻结GT**重新评分，后续Stable帧也检测；普通错误不等于历史错误。'
        '单次历史错误的首末采样跨度为0，不证明真实持续时间为0。', '',
        table(['等级','方法','停止原因'],[[r['stability_level'],r['method'],r['stop_reasons']] for r in core]),
        '## 4. 相同变化事件上的配对响应', '',
        table(['等级','变化事件','共同响应','仅Native响应','仅WS响应','双方未观察响应','共同Native均值ms','共同WS均值ms','WS更快/Native更快/相同'],[
            [r['stability_level'],r['paired_events'],r['both_responded'],r['native_only_responded'],r['worldstream_only_responded'],
             r['neither_responded'],fmt(r['common_native_mean_ms']),fmt(r['common_worldstream_mean_ms']),
             f'{r["common_worldstream_faster"]}/{r["common_native_faster"]}/{r["common_tied"]}'] for r in report['paired_response']]),
        '共同响应子集不代表全部事件；不得只比较各方法不同成功子集的均值而省略响应比例。', '',
        '## 5. 分母、GT覆盖和置信区间', '',
        table(['等级','方法','总查询','可评分','UNKNOWN','已知但无可比历史','Stable Scene','Changed Scene'],[
            [r['stability_level'],r['method'],r['queries'],r['scorable_queries'],r['unknown_queries'],
             r['unclassified_scorable_queries'],r['stable_scenes'],r['changed_scenes']] for r in rows]),
        f'{report["bootstrap_samples"]}次Scene-cluster bootstrap，以每项指标实际贡献的Scene为聚类单位；'
        '少于2个贡献Scene时区间为N/A，不用大量Stable样本伪造Changed不确定性。点估计均未改变。', '',
        table(['等级','方法','Stable% 95%CI','Changed% 95%CI','Stale% 95%CI','响应均值ms 95%CI','响应率% 95%CI'],[
            [r['stability_level'],r['method']]+[
                f'[{fmt(r[k+"_ci95_low"])}, {fmt(r[k+"_ci95_high"])}]'
                for k in ('stable_fact_accuracy_percent','changed_fact_accuracy_percent','stale_answer_rate_percent',
                          'response_delay_mean_ms','observed_response_rate_percent')] for r in rows]),
        '## 6. 尾窗与总体对账', '',
        '7个不足5秒的尾窗共62帧保持Unclassified，不塞入S1–S4。该“窗口未分类”与“查询无可比历史”的Unclassified不同。'
        '尾窗仍可有Stable查询；ALL包括所有723帧。', '',
        core_table([r for r in rows if r['stability_level'] in ('Unclassified','ALL')]),
        'S1–S4与尾窗的原始计数/分母/分子相加后与原FQ-3总体结果严格一致；总体百分比不能由四级百分比简单平均得到。', '',
        '## 7. 结果解释', '',
        '1. 不假定Changed表现随等级单调变化，也不假定每级都优于Native。四级内的Stable/Changed分母各自独立，不能用环境等级替代逐查询GT标签。',
        f'2. S4有{n4["changed_queries"]}个Changed事件，WorldStream变化帧正确{w4["changed_fully_correct"]}个、观察到响应{w4["responded_events"]}个；'
        f'Native分别为{n4["changed_fully_correct"]}和{n4["responded_events"]}。Changed Accuracy：WS {fmt(w4["changed_fact_accuracy_percent"])}% / Native {fmt(n4["changed_fact_accuracy_percent"])}%；'
        f'Stale：{fmt(w4["stale_answer_rate_percent"])}% / {fmt(n4["stale_answer_rate_percent"])}%；成功样本平均响应间隔：{fmt(w4["response_delay_mean_ms"])} / {fmt(n4["response_delay_mean_ms"])} ms。',
        f'3. S3有{n3["changed_queries"]}个Changed事件：WS Accuracy {fmt(w3["changed_fact_accuracy_percent"])}%、Stale {fmt(w3["stale_answer_rate_percent"])}%，'
        f'Native分别为{fmt(n3["changed_fact_accuracy_percent"])}%和{fmt(n3["stale_answer_rate_percent"])}%；观察到响应分别{w3["responded_events"]}和{n3["responded_events"]}个。'
        '若某级出现退化，应如实保留，不写所有等级都改善的结论。',
        f'4. S1/S2分别有{lookup["S1","native"]["changed_queries"]}/{lookup["S2","native"]["changed_queries"]}个Changed事件。'
        '小样本的0%/100%不代表可靠概率估计；N/A表示未观察到响应，0ms表示已响应事件在变化帧即正确，两者不可混淆。',
        '5. 响应时延来自约2Hz采样视频；许多当帧正确事件使均值小于采样周期。它与本次三问纯前向墙钟时延不同，不能当成在线毫秒级系统反应能力。',
        f'6. 当前{lookup["ALL","native"]["changed_queries"]}个Changed事件不足以给每级提供稳定的变化能力估计，更不支持独立TEST或OGSM/QCSE因果协同声明。'
        '下一步应增加具有可靠连续GT的变化事件，并在统一四等级协议下做模块消融；不按性能事后改阈值或补造GT。', '',
        '## 8. 图表与原始数据', '',
        '![四等级四项核心指标](four_level_core_metrics.png)', '',
        '[PDF](four_level_core_metrics.pdf)、[核心指标CSV](core_metrics_by_stability.csv)、'
        '[分布与事件数](stability_distribution.csv)、[配对响应](paired_response_by_stability.csv)、'
        '[逐问题](raw_qa_results.csv)、[逐变化事件](change_response_events.jsonl)、'
        '[完整定义/来源摘要](experiment_summary.json)、[完成记录](completion.json)。', '',
        '没有新的模型推理。已复核冻结阈值/分组、原始收据、输入配对、重新评分、完整时序响应事件以及原总体对账。', '']
    (output/'结果.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--bootstrap-samples',type=int)
    args=parser.parse_args()
    require_affinity()
    destination=args.output or args.source/('stability_levels_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    generate(args.source,destination,args.bootstrap_samples)


if __name__=='__main__':
    main()
