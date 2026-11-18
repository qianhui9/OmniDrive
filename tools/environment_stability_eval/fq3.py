"""FQ-3-only Stable/Changed evaluation, with real paired chronological inference."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

from .common import (METHODS, PYTHON, ROOT, csv_write, dump_jsonl, gpu_lease,
                     jsonl, read, require_affinity, write)

GROUPS = ('vehicle_category', 'vehicle_existence', 'pedestrian_position')
PROTOCOL = 'fq3_adjacent_known_v1'
SOURCE_PAGE = ROOT / 'tools/environment_stability_eval/outputs/run_20261117_113951_740642/结果.md'
START, END = '<!-- FQ3_PRIMARY_START -->', '<!-- FQ3_PRIMARY_END -->'


def strict_transitions(rows, scenes, max_gap):
    """GT-only classification on immediately adjacent frames, never bridge UNKNOWN.

    Preserves the existing finite fact scorer and core queries. First frames,
    unknown GT, different schemas and temporal gaps stay Unclassified.
    """
    position = {t: (s['scene_id'], i) for s in scenes for i, t in enumerate(s['tokens'])}
    history, result = {}, []
    for original in rows:
        row = dict(original)
        if row['group'] not in GROUPS or row['query_index'] not in range(3):
            raise ValueError('FQ-3 must exclude all original QA')
        scene, index = position[row['frame_id']]
        if scene != row['scene_id']:
            raise ValueError('GT scene mismatch')
        previous = history.get((scene, row['question']))
        comparable = bool(previous and index == previous['_index'] + 1 and
            row['ground_truth']['known'] and previous['ground_truth']['known'] and
            0 < row['timestamp'] - previous['timestamp'] <= max_gap and
            set(row['ground_truth']['state']) == set(previous['ground_truth']['state']))
        row.update(fact_status=('Stable' if row['ground_truth']['state'] == previous['ground_truth']['state']
                                else 'Changed') if comparable else 'Unclassified',
            previous_frame_id=previous['frame_id'] if comparable else None,
            previous_timestamp=previous['timestamp'] if comparable else None,
            previous_ground_truth=previous['ground_truth'] if comparable else None)
        row['change_type'] = row['ground_truth'].get('fact_type') if row['fact_status'] == 'Changed' else None
        history[scene, row['question']] = dict(row, _index=index)
        result.append(row)
    return result


def check_core(rows):
    for row in rows:
        if tuple(row['groups']) != GROUPS or len(row['questions']) != 3:
            raise ValueError('Expected exactly the three fixed core fact questions')
        if len(row['answers']) != 3 or len(row['requests']) != 3:
            raise ValueError('Expected three real completed query outputs')
        if row.get('forward_exclusion_calls') != {'answer_text_decode': 3, 'answer_file_write': 1}:
            raise ValueError('Pure-forward export boundary was not observed for all three queries')
        if (not math.isfinite(row['forward_ms']) or row['forward_ms'] <= 0 or
                abs(row['forward_ms'] + row['forward_excluded_ms'] - row['forward_raw_ms']) > 1e-6):
            raise ValueError('Invalid pure-forward timing')


def sha(path):
    from tools.worldstream_m123.r0_task_evaluation import sha as hash_file
    return hash_file(path)


def prepare(args, out, smoke):
    from types import SimpleNamespace
    from .run import load_config, prepare as legacy_prepare
    from .data_adapter import coverage_report
    config = load_config(args.config)
    if config['other_questions'] != 0:
        raise ValueError('FQ-3 requires other_questions=0')
    if out.exists():
        raise FileExistsError(out)
    params = SimpleNamespace(output=out, config=args.config, smoke=smoke,
                             driver_environment=args.driver_environment)
    manifest = legacy_prepare(params, config)
    rows = strict_transitions(list(jsonl(out / 'query_ground_truth.jsonl')),
                              manifest['scenes'], config['max_transition_gap_seconds'])
    # Reclassify from authoritative GT BEFORE inference; preserve no legacy
    # nonadjacent labels in the primary event set.
    tmp = out / 'fq3_ground_truth.tmp'
    dump_jsonl(tmp, rows)
    tmp.replace(out / 'query_ground_truth.jsonl')
    objects = {r['frame_id']: r['objects'] for r in jsonl(out / 'evaluation_objects.jsonl')}
    coverage = read(out / 'data_coverage.json')
    coverage.update(coverage_report(rows, objects))
    coverage.update(protocol=PROTOCOL, queries_per_frame=3,
                    known_transition_counts=dict(Counter(r['fact_status'] for r in rows)))
    write(out / 'data_coverage.json', coverage)
    for name in manifest['frozen_artifact_hashes']:
        manifest['frozen_artifact_hashes'][name] = sha(out / name)
    manifest.update(protocol=PROTOCOL,
        query_definition='FQ-3 only: fixed q0 vehicle category; q1 car existence; q2 nearest pedestrian position. No original QA.',
        primary_groups=['Stable', 'Changed'],
        transition_definition='Immediately adjacent frames of the same scene and exact query; both known and same state schema; gap<=1s. Unknown never bridged.',
        forward_definition='CUDA synchronized one model call per six-camera frame with exactly three queries. Prepared GPU input; exclude final text decoding/file export; retain OGSM/DAVP/QCSE and autoregressive generation.')
    write(out / 'manifest.json', manifest)
    write(out / 'experiment_summary.json', dict(status='PREPARED_NO_INFERENCE',
        protocol=PROTOCOL, metrics_available=False, manifest=manifest, data_coverage=coverage))
    write(out / 'status.json', dict(status='PREPARED', smoke=smoke, frames=manifest['frames'],
                                   queries=3 * manifest['frames']))
    (out / '结果.md').write_text('# FQ-3 环境事实变化响应实验\n\n状态：PREPARED，尚无本次新性能结果。\n\n'
        '仅三个核心事实任务 q0/q1/q2，无原始 QA。Native 与完整 WorldStream 新解释器重新运行。\n\n'
        '论文主表仅 Stable/Changed；缺失 GT、首帧、时间断点不强行分类。纯前向时延沿用新同步计时边界。\n')
    print('FQ3_PREPARED', out, coverage['known_transition_counts'], flush=True)
    return manifest


def primary_page(out, completed=False):
    text = SOURCE_PAGE.read_text()
    history = out / 'historical_five_query_report.md'
    if not history.exists():
        history.write_text(text)
    body = ('## 当前协议：仅 FQ-3 核心事实任务\n\n'
        '按 2026-11-18 更新要求，仅 q0 前方车辆类别、q1 前方是否有汽车、q2 最近行人相对位置；'
        '移除两个原始 QA。不是 q0/q1/q3 的三问估算子集。Stable 与 Changed 都执行全部三个任务；'
        '状态按相同查询的相邻已知 GT 定义，未知/断点不强行归类。\n\n'
        f'新实验状态：**{"COMPLETED" if completed else "正在运行或等待 GPU，尚无本次完整结果"}**。'
        f'[FQ-3 正式报告]({os.path.relpath(out / "结果.md", SOURCE_PAGE.parent)})。'
        '完成后报告自动列出 Stable/Changed Accuracy、Stale Answer Rate、Change Response Delay（含响应率/删失），'
        '以及三问实际纯前向时延。\n\n'
        '本页以下五问结果及此前五问纯前向时延仅为历史记录，不作为新 FQ-3 主表。原始数据保留。\n')
    block = START + '\n' + body + END
    if START in text:
        a, b = text.split(START, 1)
        _, c = b.split(END, 1)
        text = a + block + c
    else:
        a, b = text.split('\n', 1)
        text = a + '\n\n' + block + '\n' + b
    temp = SOURCE_PAGE.with_suffix('.fq3.tmp')
    temp.write_text(text)
    temp.replace(SOURCE_PAGE)


def report(out):
    import numpy as np
    from .evaluator import verify_manifest
    from .metrics import validate_pair, score_rows, accuracy, cluster_intervals
    from .change_response import change_events, statistics, paired_statistics, plot
    manifest = verify_manifest(out)
    config = manifest['config']
    frames, runtimes = {}, {}
    for method in METHODS:
        d = out / method
        status = read(d / 'inference_complete.json')
        if status['status'] != 'COMPLETED' or any(sha(d / filename) != status[field] for filename, field in (
                ('receipts.jsonl', 'receipts_sha256'), ('processing_segments.json', 'segments_sha256'))):
            raise ValueError('Modified/incomplete inference')
        frames[method] = list(jsonl(d / 'receipts.jsonl'))
        if len(frames[method]) != manifest['frames']:
            raise ValueError('Frame coverage mismatch')
        check_core(frames[method])
        runtimes[method] = read(d / 'effective_runtime.json')
    validate_pair(manifest, list(jsonl(out / 'test_windows.jsonl')), frames, runtimes)
    gt = list(jsonl(out / 'query_ground_truth.jsonl'))
    objects = {r['frame_id']: r['objects'] for r in jsonl(out / 'evaluation_objects.jsonl')}
    scored = [q for method in METHODS for q in score_rows(frames[method], gt, objects)]
    delays = change_events(scored, frames, objects, config['response_horizon_seconds'],
                           config['max_transition_gap_seconds'])
    samples = config['bootstrap_samples']
    seed = config['seed']
    primary, task_rows, delay_rows, forward_rows = [], [], [], []
    for method in METHODS:
        method_events = [e for e in scored if e['method'] == method]
        for state in ('Stable', 'Changed', 'Unclassified', 'ALL'):
            selected = [e for e in method_events if state == 'ALL' or e['fact_status'] == state]
            values = accuracy(selected)
            values.update(cluster_intervals(selected, accuracy,
                ['overall_accuracy_percent', 'stale_answer_rate_percent'], samples=samples, seed=seed))
            primary.append(dict(method=method, environment_state=state, **values))
            for group in GROUPS:
                part = [e for e in selected if e['group'] == group]
                task_rows.append(dict(method=method, environment_state=state, task=group, **accuracy(part)))
        method_delays = [d for d in delays if d['method'] == method]
        for cohort, choose in [('ALL_CHANGED', lambda e: True),
                ('SAME_PERSISTENT_TARGET', lambda e: e['same_persistent_target'] is True)]:
            selected = [e for e in method_delays if choose(e)]
            values = statistics(selected)
            values.update(cluster_intervals(selected, statistics,
                ['observed_response_rate_percent', 'response_delay_mean_ms',
                 'mean_max_consecutive_stale_outputs'], samples=samples, seed=seed))
            delay_rows.append(dict(method=method, cohort=cohort, **values))
        # A frame can contain both Stable and Changed tasks; never double-count
        # frame latency into two primary environmental states.
        times = frames[method]
        def latency(part):
            values = [r['forward_ms'] for r in part]
            return dict(mean_forward_ms=float(np.mean(values)), p95_forward_ms=float(np.percentile(values, 95)))
        values = latency(times)
        values.update(cluster_intervals(times, latency, ['mean_forward_ms', 'p95_forward_ms'],
                                        samples=samples, seed=seed))
        forward_rows.append(dict(method=method, frames=len(times), queries=sum(len(r['questions']) for r in times), **values))
    dump_jsonl(out / 'scored_queries.jsonl', scored)
    dump_jsonl(out / 'change_response_events.jsonl', delays)
    csv_write(out / 'stable_changed_summary.csv', primary)
    csv_write(out / 'accuracy_by_task_and_state.csv', task_rows)
    csv_write(out / 'change_response_summary.csv', delay_rows)
    csv_write(out / 'change_response_events.csv', delays)
    csv_write(out / 'forward_latency_summary.csv', forward_rows)
    csv_write(out / 'raw_forward_latency.csv', [dict(method=m, **{k:r[k] for k in
        ('scene_id','frame_id','timestamp','forward_ms','forward_raw_ms','forward_excluded_ms')})
        for m in METHODS for r in frames[m]])
    paired = paired_statistics(delays)
    subtype = []
    for change_type in sorted({e['change_type'] for e in delays}):
        for method in METHODS:
            selected = [e for e in delays if e['method'] == method and e['change_type'] == change_type]
            subtype.append(dict(method=method, change_type=change_type, **statistics(selected)))
    csv_write(out / 'supported_change_type_summary.csv', subtype)
    figures = plot(delays, out, config['response_horizon_seconds'])
    result = dict(status='COMPLETED', smoke=manifest['smoke'], protocol=PROTOCOL,
        generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        primary=primary, by_task=task_rows, change_response=delay_rows,
        paired_response=paired, supported_change_types=subtype, forward_latency=forward_rows,
        coverage=read(out / 'data_coverage.json'), figures=figures,
        definitions=dict(transition=manifest['transition_definition'], query=manifest['query_definition'],
            delay='Sampled-video t_first_correct - t_first_observed_change; full correctness; no bridge across unknown GT or later change; unresolved=null.',
            accuracy='100*mean existing factual score over known GT, fully-correct rate also saved.',
            stale='On changed frame: wrong for current GT and fully matches frozen preceding GT. Continued historical errors also tracked after change.',
            forward=manifest['forward_definition']),
        fair_inputs_verified=True, live_wall_response_delay_available=False,
        source_manifest_sha256=sha(out / 'manifest.json'),
        limitations=['Existing exposed validation scenes; not independent TEST.',
            'State labels are per exact task query, not a claim that the whole scene is globally unchanged.',
            'Same nearest-query answer may change when target identity switches; persistent-target subset separately given.',
            'Appearance/disappearance/motion/occlusion cause labels not fully identifiable from these three answer states; no negative GT invented.',
            'Two-method comparison does not isolate OGSM/QCSE causal synergy.'])
    write(out / 'experiment_summary.json', result)
    markdown(out, result)
    write(out / 'completion.json', dict(status='COMPLETED', smoke=manifest['smoke'],
        protocol=PROTOCOL, frames_per_method=manifest['frames'], queries_per_method=3*manifest['frames'],
        fair_inputs_verified=True, generated_at=result['generated_at'],
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.csv','.jsonl','.md')},
        report_sha256=sha(out / 'experiment_summary.json')))
    write(out / 'status.json', dict(status='COMPLETED', smoke=manifest['smoke']))
    if not manifest['smoke']:
        primary_page(out, True)
    print('FQ3_REPORT_COMPLETED', out, flush=True)
    return result


def markdown(out, r):
    def fmt(v):
        return 'N/A' if v is None else f'{v:.2f}'
    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+
            ['| '+' | '.join(str(v) for v in values)+' |' for values in rows])+'\n'
    main = [e for e in r['primary'] if e['environment_state'] in ('Stable','Changed')]
    response = {e['method']:e for e in r['change_response'] if e['cohort']=='ALL_CHANGED'}
    lines = ['# FQ-3 环境事实稳定性与变化响应：正式结果', '',
        '**'+('Smoke 执行验证；不能作为论文完整结果。' if r['smoke'] else '完整真实 FQ-3 配对实验已完成。')+'**', '',
        '三个任务固定为前方车辆类别、前方是否有汽车、最近行人相对位置（q0/q1/q2）；无原始生成 QA。'
        'Stable/Changed 都对全部三个任务执行，不根据 GT 为模型选择任务。', '',
        '## 论文核心表', '',
        '环境状态是同一问题在连续已知 GT 帧间的回答事实状态。相邻帧未知、首帧、schema不一致或间隔>1秒为 Unclassified。'
        'Accuracy 分母是该组已知 GT 查询；Stale 与响应时延只在 Changed 上定义。时延均值仅对成功响应样本计算，必须结合响应率阅读。', '',
        table(['状态','方法','可评分查询','Factual QA Accuracy(%)','Stale Answer Rate(%)',
               'Change Response Delay均值(ms)','观察到响应/变化事件'], [
            [e['environment_state'],e['method'],e['scorable_queries'],fmt(e['overall_accuracy_percent']),
             fmt(e['stale_answer_rate_percent']) if e['environment_state']=='Changed' else 'N/A',
             fmt(response[e['method']]['response_delay_mean_ms']) if e['environment_state']=='Changed' else 'N/A',
             f'{response[e["method"]]["responded_events"]}/{response[e["method"]]["changed_events"]}'
             if e['environment_state']=='Changed' else 'N/A'] for e in main]),
        '## Change Response Delay 与连续历史错误', '',
        '`T_response = t_first_correct - t_change`，单位为视频采样 ms，不是在线墙钟。'
        '变化帧就正确时为0，不代表推理耗时为0；再次变化、UNKNOWN、缺问、时间断点、场景结束和5秒上限分别标记。', '',
        table(['集合','方法','变化','响应率%','Mean(ms)','Median(ms)','P95(ms)','删失','再次变化','上限',
               '历史错误输出','最长连续次数','最长段次数均值'], [
            [e['cohort'],e['method'],e['changed_events'],fmt(e['observed_response_rate_percent']),
             fmt(e['response_delay_mean_ms']),fmt(e['response_delay_median_ms']),fmt(e['response_delay_p95_ms']),
             e['censored_events'],e['superseded_events'],e['horizon_exhausted_events'],e['stale_outputs'],
             e['max_consecutive_stale_outputs'],fmt(e['mean_max_consecutive_stale_outputs'])] for e in r['change_response']]),
        '连续历史错误要求对当前GT错、对本次变化之前冻结GT完全正确，后续Stable帧也检测。'
        '每段首末错误时间跨度仅为持续时间下界；单次错误跨度为0不表示真实持续时间为0。', '',
        '观察停止原因：', '',
        table(['集合','方法','原因计数'],[[e['cohort'],e['method'],e['stop_reasons']] for e in r['change_response']]),
        '相同变化事件的配对（共同响应均值只对两方法都响应的事件）：', '',
        table(['配对事件','双方响应','仅Native','仅WorldStream','双方未响应','共同Native均值ms','共同WorldStream均值ms'],[[
            r['paired_response']['paired_events'],r['paired_response']['both_responded'],
            r['paired_response']['native_only_responded'],r['paired_response']['worldstream_only_responded'],
            r['paired_response']['neither_responded'],fmt(r['paired_response']['common_native_mean_ms']),
            fmt(r['paired_response']['common_worldstream_mean_ms'])]]),
        '共同响应事件中更快次数：WorldStream='+str(r['paired_response']['common_worldstream_faster'])+
        '，Native='+str(r['paired_response']['common_native_faster'])+'，相同='+str(r['paired_response']['common_tied'])+'。', '',
        '## 每任务、每状态完整结果', '',
        table(['状态','任务','方法','总查询','可评分','UNKNOWN','Accuracy%','完全正确%','Stale/机会'],[
            [e['environment_state'],e['task'],e['method'],e['queries'],e['scorable_queries'],e['unknown_queries'],
             fmt(e['overall_accuracy_percent']),fmt(e['fully_correct_accuracy_percent']),
             f'{e["stale_answers"]}/{e["queries_after_changes"]}'] for e in r['by_task']]),
        '## 95%置信区间与覆盖率', '',
        '2,000次完整Scene聚类bootstrap，非多seed误差条；零分母为N/A。', '',
        table(['状态','方法','Accuracy% 95%CI','Stale% 95%CI'],[
            [e['environment_state'],e['method'],
             f'[{fmt(e["overall_accuracy_percent_ci95_low"])}, {fmt(e["overall_accuracy_percent_ci95_high"])}]',
             f'[{fmt(e["stale_answer_rate_percent_ci95_low"])}, {fmt(e["stale_answer_rate_percent_ci95_high"])}]']
            for e in r['primary']]),
        table(['响应集合','方法','响应率% 95%CI','成功响应均值ms 95%CI'],[
            [e['cohort'],e['method'],
             f'[{fmt(e["observed_response_rate_percent_ci95_low"])}, {fmt(e["observed_response_rate_percent_ci95_high"])}]',
             f'[{fmt(e["response_delay_mean_ms_ci95_low"])}, {fmt(e["response_delay_mean_ms_ci95_high"])}]']
            for e in r['change_response']]),
        table(['覆盖项','数量'],[[k,v] for k,v in r['coverage'].items() if isinstance(v,(int,float))]),
        'GT 转换标签计数：'+str(r['coverage']['known_transition_counts'])+'。', '',
        '## 三问实际纯前向时延（辅助）', '',
        '输入加载/预处理/H2D/prompt编码位于计时外，排除最终文本解码和答案文件写入。'
        '完整OGSM/DAVP/QCSE、内部必要适配和自回归生成计入，前后CUDA同步。每帧一次六相机+三问模型调用。', '',
        table(['方法','帧','查询','Mean(ms/帧)','Mean 95%CI','P95(ms/帧)','P95 95%CI'],[
            [e['method'],e['frames'],e['queries'],fmt(e['mean_forward_ms']),
             f'[{fmt(e["mean_forward_ms_ci95_low"])}, {fmt(e["mean_forward_ms_ci95_high"])}]',fmt(e['p95_forward_ms']),
             f'[{fmt(e["p95_forward_ms_ci95_low"])}, {fmt(e["p95_forward_ms_ci95_high"])}]']
            for e in r['forward_latency']]),
        '## 变化原因细分与结论边界', '',
        '本协议不把答案类别变化伪装成Appearance/Disappearance，也不把最近目标切换必然解释为同一物体运动。'
        '现有正例可见性评分在没有可靠目标时为UNKNOWN；不能据此造出“消失”的负例标签。'
        '遮挡、真实运动与自车运动的因果标签不足，故不进入论文主表。可支持的回答事实变化类型仅作附录：', '',
        table(['回答事实变化类型','方法','事件','已响应','Mean(ms)'],[
            [e['change_type'],e['method'],e['changed_events'],e['responded_events'],fmt(e['response_delay_mean_ms'])]
            for e in r['supported_change_types']]),
        '完整系统与Native的比较不能独立证明OGSM/QCSE因果协同；需匹配模块消融。既有暴露验证集，非独立TEST。'
        '后续优先检查未响应/直答历史错误事件，并扩展带可靠事件GT的数据；不为美化结果补造GT。', '',
        '## 原始证据', '',
        '[主表CSV](stable_changed_summary.csv)、[任务分组](accuracy_by_task_and_state.csv)、'
        '[响应统计](change_response_summary.csv)、[逐变化事件](change_response_events.jsonl)、'
        '[逐问题评分](scored_queries.jsonl)、[前向统计](forward_latency_summary.csv)、'
        '[完整汇总](experiment_summary.json)。原始答案与运行收据在native/worldstream子目录。', '',
        '![响应曲线与连续历史错误](change_response.png)', '']
    (out/'结果.md').write_text('\n'.join(lines))


def launch_locked(out, lease):
    for method in METHODS:
        folder = out/method
        folder.mkdir(exist_ok=False)
        command = ['taskset','-c','0-3,6-31',PYTHON,'-B','-u','-X','faulthandler',
            '-m','tools.environment_stability_eval.fq3','--_worker',method,'--output',str(out),'--_lock-fd',str(lease)]
        env = dict(os.environ, HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',
                   OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        with (folder/'runtime.log').open('x') as log:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease,))
            write(folder/'process.json',dict(pid=process.pid,command=command))
            print('STARTED_FQ3',method,process.pid,folder/'runtime.log',flush=True)
            status=process.wait()
        if status:
            raise RuntimeError(f'{method} exited {status}: '+(folder/'runtime.log').read_text(errors='replace')[-4000:])


def environment():
    from .gpu_environment import nvml_probe
    probe = nvml_probe()
    if probe.returncode == 0:
        return dict(mode='system',nvml_driver_version=probe.stdout.strip())
    if probe.returncode==18 and 'driver/library version mismatch' in (probe.stdout+probe.stderr).lower():
        from tools.worldstream_ablation import driver_compat
        bundle=driver_compat.DEFAULT_ROOT/driver_compat.kernel_version()
        env=driver_compat.environment(bundle)
        if 'FQ3_DRIVER_REEXEC' in os.environ:
            raise RuntimeError('Matching process driver still failed')
        env['FQ3_DRIVER_REEXEC']='1'
        os.execve(PYTHON,[PYTHON,'-B','-u','-m','tools.environment_stability_eval.fq3',*sys.argv[1:]],env)
    raise RuntimeError('Cannot query NVML: '+probe.stdout+probe.stderr)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('fq3.yaml'))
    parser.add_argument('--output',type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--run-full',action='store_true')
    parser.add_argument('--_worker',choices=METHODS,help=argparse.SUPPRESS)
    parser.add_argument('--_lock-fd',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    os.chdir(ROOT)
    require_affinity()
    if args._worker:
        environment()
        from .forward_latency import worker
        worker(args.output.resolve(),args._worker,args._lock_fd)
        return
    args.output=(args.output or ROOT/'tools/environment_stability_eval/outputs'/
        ('fq3_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    args.driver_environment=environment() if args.run_full or args.smoke else {}
    try:
        prepare(args,args.output,args.smoke)
        if not (args.smoke or args.run_full):
            print('FQ3_CHECK_PASS',args.output,flush=True)
            return
        if args.run_full:
            primary_page(args.output)
        print('WAIT_PROJECT_GPU_LEASE',flush=True)
        with gpu_lease() as lease:
            if args.run_full:
                write(args.output/'status.json',dict(status='SMOKE_VALIDATION'))
                smoke=args.output/'smoke_validation'
                prepare(args,smoke,True)
                launch_locked(smoke,lease)
                report(smoke)
                print('FQ3_REAL_SMOKE_PASS',flush=True)
            write(args.output/'status.json',dict(status='RUNNING_FULL' if args.run_full else 'RUNNING_SMOKE'))
            launch_locked(args.output,lease)
        report(args.output)
    except Exception:
        if args.output.exists():
            write(args.output/'failure.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
