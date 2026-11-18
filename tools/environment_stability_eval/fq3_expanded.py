"""Expanded, predeclared FQ-3 evaluation with disjoint GT calibration scenes.

CPU preparation is separate from actual paired model inference. Historical
experiments and their hash-bound source files are never rewritten.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
import sys
import traceback
from types import SimpleNamespace

from .common import (LEVELS, METHODS, PYTHON, ROOT, csv_write, dump_jsonl,
                     gpu_lease, jsonl, read, require_affinity, write)

PROTOCOL = 'fq3_expanded_disjoint_v1'
BASE = Path(__file__).with_name('fq3.yaml')


def stamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def sha(path):
    from tools.worldstream_m123.r0_task_evaluation import sha as digest
    return digest(path)


def split_scenes(scenes, original_ids, calibration_count=30, seed=20261118):
    """Preserve the old evaluation scenes; hash-rank other IDs without GT/results."""
    ids = [s['scene_id'] for s in scenes]
    if len(ids) != len(set(ids)) or not set(original_ids) <= set(ids):
        raise ValueError('Duplicate scenes or missing original evaluation scenes')
    candidates = [s for s in scenes if s['scene_id'] not in original_ids]
    if not 1 <= calibration_count < len(candidates):
        raise ValueError('Reserve calibration scenes and at least one new evaluation scene')
    key = lambda s: (hashlib.sha256(f'{seed}:{s["scene_id"]}'.encode()).hexdigest(), s['scene_id'])
    calibration = sorted(sorted(candidates, key=key)[:calibration_count], key=lambda s:s['scene_id'])
    excluded = {s['scene_id'] for s in calibration}
    evaluation = sorted([s for s in scenes if s['scene_id'] not in excluded], key=lambda s:s['scene_id'])
    return evaluation, calibration


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def fmt(value):
    return 'N/A' if value is None else f'{value:.2f}'


def gt_audit(manifest, gt, windows, target_events=50, target_scenes=10):
    """Count GT events and follow-up opportunities, never model correctness."""
    levels = {t:w['stability_level'] for w in windows for t in w['frame_ids']}
    streams = defaultdict(list)
    for q in gt:
        streams[q['scene_id'],q['question']].append(q)
    followup = {}
    horizon = manifest['config']['response_horizon_seconds']
    gap = manifest['config']['max_transition_gap_seconds']
    for stream in streams.values():
        stream.sort(key=lambda q:q['timestamp'])
        for i,q in enumerate(stream):
            if q['fact_status'] != 'Changed':
                continue
            last, samples, stop = q['timestamp'], 1, 'SCENE_END'
            for nxt in stream[i+1:]:
                if nxt['timestamp']-last > gap:
                    stop='OBSERVATION_GAP'; break
                if nxt['timestamp']-q['timestamp'] > horizon:
                    stop='HORIZON'; break
                if not nxt['ground_truth']['known']:
                    stop='UNKNOWN_GT'; break
                if nxt['ground_truth']['state'] != q['ground_truth']['state']:
                    stop='SUPERSEDED'; break
                samples += 1
                last=nxt['timestamp']
            followup[q['frame_id'],q['question']] = dict(samples=samples,
                observable_span_ms=1000*(last-q['timestamp']), stop=stop)
    rows=[]
    for level in (*LEVELS,'Unclassified','ALL'):
        part=[q for q in gt if level=='ALL' or levels[q['frame_id']]==level]
        changed=[q for q in part if q['fact_status']=='Changed']
        scenes={q['scene_id'] for q in changed}
        follows=[followup[q['frame_id'],q['question']] for q in changed]
        rows.append(dict(stability_level=level, queries=len(part),
            known_queries=sum(q['ground_truth']['known'] for q in part),
            stable_queries=sum(q['fact_status']=='Stable' for q in part),
            changed_events=len(changed), changed_scenes=len(scenes),
            changed_by_task=dict(Counter(q['group'] for q in changed)),
            events_with_followup=sum(f['samples']>1 for f in follows),
            events_with_1s_followup=sum(f['observable_span_ms']>=999 for f in follows),
            gt_followup_stop_reasons=dict(Counter(f['stop'] for f in follows)),
            planning_target_events=target_events, planning_target_scenes=target_scenes,
            planning_target_met=len(changed)>=target_events and len(scenes)>=target_scenes))
    return rows


def prepare(root, calibration_count=30, split_seed=20261118, driver=None,
            target_events=50, target_scenes=10):
    import yaml
    from .data_adapter import load_infos
    from .fq3 import prepare as prepare_fq3
    from tools.worldstream_m123.r0_task_evaluation import selected_scenes
    if root.exists():
        raise FileExistsError('Use a new --output, or --resume an existing prepared run: '+str(root))
    config=yaml.safe_load(BASE.read_text())
    infos,_=load_infos(config)
    all_scenes=selected_scenes(infos,'full-val',ROOT,None)
    original=read(ROOT/config['main_manifest'])['selected']
    evaluation,calibration=split_scenes(all_scenes,{s['scene_id'] for s in original},calibration_count,split_seed)
    root.mkdir(parents=True)
    setup=root/'plan'; setup.mkdir()
    selection=setup/'scene_selection.json'
    write(selection,dict(selected=evaluation))
    plan=dict(protocol=PROTOCOL,split_seed=split_seed,
        selection='sha256(seed:scene_id); no GT or model performance used; original evaluation scenes retained',
        calibration_scene_ids=[s['scene_id'] for s in calibration],
        evaluation_scene_ids=[s['scene_id'] for s in evaluation],
        original_evaluation_scene_ids=[s['scene_id'] for s in original],
        evaluation_frames=sum(len(s['tokens']) for s in evaluation),
        calibration_frames=sum(len(s['tokens']) for s in calibration),
        thresholds='New version: quartiles fitted ONLY on disjoint calibration scenes before inference',
        independent_TEST=False,planning_target_events=target_events,planning_target_scenes=target_scenes)
    write(setup/'split.json',plan)
    config.update(main_manifest=str(selection),thresholds=None,other_questions=0,repeats=1)
    config_path=setup/'config.yaml'
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    write(root/'status.json',dict(status='PREPARING_GT',protocol=PROTOCOL))
    args=SimpleNamespace(config=config_path,driver_environment=driver or {})
    manifest=prepare_fq3(args,root/'evaluation',False)
    # The unchanged evaluator selects the complement as calibration. Verify this
    # contract instead of silently trusting a new, different split.
    if set(manifest['thresholds']['calibration_scene_ids']) != set(plan['calibration_scene_ids']):
        raise ValueError('Calibration complement differs from the frozen split')
    manifest.update(expansion_protocol=PROTOCOL,split_plan_sha256=sha(setup/'split.json'),
                    independent_TEST=False)
    manifest['source_and_GT_hashes'][str(setup/'split.json')]=sha(setup/'split.json')
    write(root/'evaluation'/'manifest.json',manifest)
    windows=list(jsonl(root/'evaluation'/'test_windows.jsonl'))
    gt=list(jsonl(root/'evaluation'/'query_ground_truth.jsonl'))
    audit=gt_audit(manifest,gt,windows,target_events,target_scenes)
    write(root/'gt_audit.json',dict(status='GT_ONLY_NO_MODEL_RESULTS',plan=plan,
        thresholds=manifest['thresholds']['thresholds'],by_level=audit))
    csv_write(root/'gt_audit.csv',audit)
    for artifact in (root/'gt_audit.json',root/'gt_audit.csv',Path(__file__).with_name('run_expanded.sh')):
        manifest['source_and_GT_hashes'][str(artifact.resolve())]=sha(artifact)
    write(root/'evaluation'/'manifest.json',manifest)
    lines=['# FQ-3 扩充实验：推理前覆盖统计','',
        '**当前仅完成 GT 准备；尚未产生新模型指标。**','',
        f'评估 {len(evaluation)} 个完整场景 / {plan["evaluation_frames"]} 帧；'
        f'校准 {len(calibration)} 个互斥场景 / {plan["calibration_frames"]} 帧。',
        '场景划分只依赖固定种子与场景 ID；保留原18场景。此次重新校准四级阈值，属于新协议版本，不能与旧表逐级直接混算。',
        '使用既有且已暴露的 val；场景互斥不等于独立未见 TEST。','',
        table(['等级','Stable查询','Changed事件','Changed场景','有后续已知采样的事件','至少约1秒已知跟踪','达到规划目标'],[
            [r['stability_level'],r['stable_queries'],r['changed_events'],r['changed_scenes'],
             r['events_with_followup'],r['events_with_1s_followup'],r['planning_target_met']] for r in audit]),
        f'规划目标为每级 {target_events} 个变化事件、{target_scenes} 个变化场景，不是统计功效保证；未达到时如实标注并继续完整评估。',
        'GT后续跨度不代表模型响应时延；无负例认证的存在性仍为 UNKNOWN。仅增加样本不能补造出现/消失标签。','',
        '执行结束后，本文件自动汇总四级四指标、响应覆盖、置信区间、纯前向时延及原始证据链接。','']
    (root/'结果.md').write_text('\n'.join(lines))
    write(root/'status.json',dict(status='PREPARED_NO_INFERENCE',protocol=PROTOCOL,
        frames_per_method=manifest['frames'],queries_per_method=3*manifest['frames']))
    for row in audit:
        print('GT_AUDIT',row['stability_level'],'changed=',row['changed_events'],
              'scenes=',row['changed_scenes'],'followup=',row['events_with_followup'],flush=True)
    return manifest


def load_completed(out):
    """Reject incomplete, reordered or unpaired receipts before scoring."""
    from .evaluator import verify_manifest
    from .fq3 import check_core, strict_transitions
    from .metrics import validate_pair
    manifest=verify_manifest(out)
    windows=list(jsonl(out/'test_windows.jsonl'))
    gt=list(jsonl(out/'query_ground_truth.jsonl'))
    if strict_transitions(gt,manifest['scenes'],manifest['config']['max_transition_gap_seconds']) != gt:
        raise ValueError('Frozen strict adjacent-known GT does not reproduce')
    rows,runtime={},{}
    for method in METHODS:
        folder=out/method
        done=read(folder/'inference_complete.json')
        if done['status']!='COMPLETED' or done['frames']!=manifest['frames']:
            raise ValueError('Incomplete method: '+method)
        for name,key in [('receipts.jsonl','receipts_sha256'),('processing_segments.json','segments_sha256')]:
            if sha(folder/name)!=done[key]: raise ValueError('Changed receipt: '+method+'/'+name)
        rows[method]=list(jsonl(folder/'receipts.jsonl'))
        check_core(rows[method])
        runtime[method]=read(folder/'effective_runtime.json')
    validate_pair(manifest,windows,rows,runtime)
    return manifest,windows,gt,rows


def report(root, out=None, smoke=False):
    import numpy as np
    from .metrics import accuracy, cluster_intervals, score_rows
    from .change_response import change_events
    from .fq3_levels import aggregate, plot_core, verify_level_assignments
    from .stability_classifier import fit_thresholds, distribution
    out=out or root/'evaluation'
    manifest,windows,gt,frames=load_completed(out)
    if bool(manifest['smoke']) != smoke:
        raise ValueError('Smoke and formal reports must remain distinct')
    thresholds=verify_level_assignments(windows,manifest)
    calibration=list(jsonl(out/'calibration_windows.jsonl'))
    if fit_thresholds(calibration,manifest['config']['threshold_quantiles'])['thresholds']!=thresholds:
        raise ValueError('Calibration thresholds do not reproduce')
    objects={r['frame_id']:r['objects'] for r in jsonl(out/'evaluation_objects.jsonl')}
    scored=[q for method in METHODS for q in score_rows(frames[method],gt,objects)]
    events=change_events(scored,frames,objects,manifest['config']['response_horizon_seconds'],
                         manifest['config']['max_transition_gap_seconds'])
    rows,pairs=aggregate(scored,events,frames,manifest['config']['bootstrap_samples'],manifest['config']['seed'])
    folder=root/(('smoke_report_' if smoke else 'report_')+stamp())
    folder.mkdir()
    dump_jsonl(folder/'scored_queries.jsonl',scored)
    dump_jsonl(folder/'change_response_events.jsonl',events)
    csv_write(folder/'core_metrics_by_stability.csv',rows)
    csv_write(folder/'paired_response_by_stability.csv',pairs)
    csv_write(folder/'stability_distribution.csv',distribution(windows,thresholds))
    tasks=[]
    for method in METHODS:
        for level in (*LEVELS,'Unclassified','ALL'):
            for group in ('vehicle_category','vehicle_existence','pedestrian_position'):
                part=[q for q in scored if q['method']==method and q['group']==group and
                      (level=='ALL' or q['stability_level']==level)]
                tasks.append(dict(method=method,stability_level=level,task=group,**accuracy(part)))
    csv_write(folder/'accuracy_by_task.csv',tasks)
    times=[]
    def latency(part):
        return dict(mean_forward_ms=float(np.mean([r['forward_ms'] for r in part])),
                    p95_forward_ms=float(np.percentile([r['forward_ms'] for r in part],95)))
    for method in METHODS:
        times.append(dict(method=method,frames=len(frames[method]),**latency(frames[method]),
            **cluster_intervals(frames[method],latency,['mean_forward_ms','p95_forward_ms'],
                samples=manifest['config']['bootstrap_samples'],seed=manifest['config']['seed'])))
    csv_write(folder/'forward_latency_summary.csv',times)
    figures=plot_core(folder,rows)
    audit=read(root/'gt_audit.json')
    summary=dict(status='SMOKE_VALIDATED' if smoke else 'COMPLETED',protocol=PROTOCOL,
        smoke=smoke,new_model_inference=True,independent_TEST=False,
        generated_at=datetime.now().astimezone().isoformat(),thresholds=thresholds,
        frames_per_method=manifest['frames'],queries_per_method=3*manifest['frames'],
        calibration_scene_count=len(manifest['thresholds']['calibration_scene_ids']),
        evaluation_scene_count=len(manifest['scenes']),metrics=rows,paired_response=pairs,
        by_task=tasks,forward_latency=times,coverage=read(out/'data_coverage.json'),
        planning_targets=dict(events=audit['plan']['planning_target_events'],scenes=audit['plan']['planning_target_scenes']),
        definitions=dict(delay='Sampled-video milliseconds; conditional on observed correct responses; not live wall-clock latency.',
            stale='Changed-frame current answer wrong and fully matches frozen preceding GT.',
            confidence='2000 scene-cluster bootstrap replicates on eligible scenes; sparse/degenerate intervals are not certainty.',
            forward=manifest['forward_definition']),
        verified=dict(frozen_inputs=True,paired_inputs=True,strict_GT=True,calibration_disjoint=True,
            full_chronological_response_tracking=True),figures=figures,
        source_manifest_sha256=sha(out/'manifest.json'),
        input_receipt_hashes={f'{m}/{n}':sha(out/m/n) for m in METHODS for n in
            ('receipts.jsonl','inference_complete.json','processing_segments.json','effective_runtime.json')})
    write(folder/'experiment_summary.json',summary)
    markdown(folder,summary,audit)
    write(folder/'completion.json',dict(status=summary['status'],smoke=smoke,
        artifacts={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    if not smoke:
        relative=os.path.relpath(folder,root)
        (root/'结果.md').write_text('# FQ-3 扩充实验：已完成\n\n'
            f'[完整四级评估结果]({relative}/结果.md)；[机器可读汇总]({relative}/experiment_summary.json)。\n\n'+
            core_table(rows)+'\n响应时延为视频时间，必须结合响应数与未响应原因阅读。\n')
        write(root/'completion.json',dict(status='COMPLETED',protocol=PROTOCOL,smoke=False,
            frames_per_method=manifest['frames'],report_directory=str(folder),
            report_completion_sha256=sha(folder/'completion.json'),index_sha256=sha(root/'结果.md')))
        write(root/'status.json',dict(status='COMPLETED',report_directory=str(folder)))
    print('EXPANDED_REPORT',folder,flush=True)
    return summary


def core_table(rows):
    return table(['等级','方法','Stable准确率%','Stable数','Changed准确率%','Changed数',
        'Stale率%','响应时延均值ms','已响应/变化数'],[
        [r['stability_level'],r['method'],fmt(r['stable_fact_accuracy_percent']),r['stable_queries'],
         fmt(r['changed_fact_accuracy_percent']),r['changed_queries'],fmt(r['stale_answer_rate_percent']),
         fmt(r['response_delay_mean_ms']),f'{r["responded_events"]}/{r["changed_events"]}']
         for r in rows if r['stability_level'] in (*LEVELS,'ALL')])


def markdown(folder,summary,audit):
    rows=summary['metrics']
    targets=summary['planning_targets']
    lines=['# FQ-3 扩充实验：四级环境稳定性与变化响应','',
        '**'+summary['status']+'**'+('：仅执行验证，禁止作为正式结果。' if summary['smoke'] else '：真实配对模型推理。'),'',
        f'每方法 {summary["evaluation_scene_count"]} 场景、{summary["frames_per_method"]} 帧、'
        f'{summary["queries_per_method"]} 条问题；校准 {summary["calibration_scene_count"]} 个互斥场景。',
        'FQ-3 固定为 q0 车辆类别、q1 汽车存在性、q2 最近行人位置。原始生成 QA 不参与。',
        '保留原评估场景；从其余场景按固定哈希选校准集，其余全部评估。没有按模型表现筛选样本。',
        '本版本在新校准集上拟合阈值，不能与旧版四级百分比直接混算；既有 val 已暴露，不宣称独立未见 TEST。',
        '冻结阈值：`'+str(summary['thresholds'])+'`。5秒完整窗口分级；不足5秒尾窗保持 Unclassified。','',
        core_table(rows),
        'Stable/Changed 是逐问题相邻已知 GT 标签，与 S1–S4 窗口等级相互独立。未知 GT 不跨越、不填负例。',
        'Change Response Delay 是视频采样时间差，仅对已正确响应事件求均值；N/A 是无可计算响应，0 是变化帧即正确。'
        '所有响应先在完整时间流中追踪，再按变化起点分级，不在窗口边界截断。','',
        table(['等级','方法','响应率%','即时正确','删失','再变化','观察上限','停止原因','历史错误输出','最长连续历史错误'],[
            [r['stability_level'],r['method'],fmt(r['observed_response_rate_percent']),r['immediate_response_events'],
             r['censored_events'],r['superseded_events'],r['horizon_exhausted_events'],r['stop_reasons'],
             r['stale_outputs'],r['max_consecutive_stale_outputs']] for r in rows]),
        '置信区间按每项指标实际贡献的场景聚类重采样；少于两个场景时为 N/A。'
        '极少样本或全0/全1时 bootstrap 可能退化为零宽区间，不能据此声称没有不确定性。','',
        table(['等级','方法','Stable 95%CI','Changed 95%CI','Stale 95%CI','响应时延95%CI','响应率95%CI','变化场景'],[
            [r['stability_level'],r['method']]+[
                f'[{fmt(r[k+"_ci95_low"])}, {fmt(r[k+"_ci95_high"])}]' for k in
                ('stable_fact_accuracy_percent','changed_fact_accuracy_percent','stale_answer_rate_percent',
                 'response_delay_mean_ms','observed_response_rate_percent')]+[r['changed_scenes']] for r in rows]),
        table(['等级','共同响应','仅Native响应','仅WS响应','均未观察响应','共同Native均值ms','共同WS均值ms'],[
            [p['stability_level'],p['both_responded'],p['native_only_responded'],p['worldstream_only_responded'],
             p['neither_responded'],fmt(p['common_native_mean_ms']),fmt(p['common_worldstream_mean_ms'])]
             for p in summary['paired_response']]),
        '双方均响应子集不代表全部事件；不得只比较不同成功子集的时延均值。','',
        table(['等级','GT变化数','变化场景','有后续已知采样','达到规划目标'],[
            [a['stability_level'],a['changed_events'],a['changed_scenes'],a['events_with_followup'],a['planning_target_met']]
             for a in audit['by_level']]),
        f'规划目标：每级 {targets["events"]} 个事件 / {targets["scenes"]} 个变化场景。'
        '未达到的等级继续视为探索性结果；扩大到可用数据不保证每级都达到目标或一定有统计显著性。',
        '存在性负例未获认证时仍为 UNKNOWN；出现、消失、遮挡原因不能仅靠三问答案变化推断。',
        '只能比较完整 Native 与 WorldStream；没有模块消融，不能独立归因 OGSM/QCSE 协同效应。','',
        table(['方法','纯前向均值ms/帧','P95 ms/帧'],[
            [t['method'],fmt(t['mean_forward_ms']),fmt(t['p95_forward_ms'])] for t in summary['forward_latency']]),
        '纯前向包含每帧六相机及三个问题的完整模型计算，排除数据读取、预处理、外部搬运、加载、预热和最终文本/文件导出。'
        '它与视频时间的环境变化响应时延是不同指标。','',
        '![四级四指标](four_level_core_metrics.png)','',
        '[核心指标CSV](core_metrics_by_stability.csv)、[逐任务](accuracy_by_task.csv)、'
        '[逐问题](scored_queries.jsonl)、[逐变化事件](change_response_events.jsonl)、'
        '[配对响应](paired_response_by_stability.csv)、[完整汇总](experiment_summary.json)。','']
    (folder/'结果.md').write_text('\n'.join(lines))


def environment():
    """Re-exec THIS entrypoint if existing cached driver libraries are required."""
    from .gpu_environment import nvml_probe, ensure_gpu_environment
    probe=nvml_probe()
    if probe.returncode==0:
        return ensure_gpu_environment([])
    if probe.returncode==18 and 'driver/library version mismatch' in (probe.stdout+probe.stderr).lower():
        from tools.worldstream_ablation import driver_compat
        if os.environ.get('FQ3_EXPANDED_DRIVER_REEXEC'):
            raise RuntimeError('Matching process driver still cannot initialize NVML')
        env=driver_compat.environment(driver_compat.DEFAULT_ROOT/driver_compat.kernel_version())
        env['FQ3_EXPANDED_DRIVER_REEXEC']='1'
        os.execve(PYTHON,[PYTHON,'-B','-u','-m','tools.environment_stability_eval.fq3_expanded',*sys.argv[1:]],env)
    raise RuntimeError('NVML unavailable: '+probe.stdout+probe.stderr)


def launch(root, driver, retry_incomplete=False):
    from .evaluator import verify_manifest
    from .fq3 import prepare as prepare_fq3
    evaluation=root/'evaluation'
    manifest=verify_manifest(evaluation)
    # Reuse an entire completed method after validating its hash-bound receipts.
    # A partial method is archived and restarted from scene one with clean state.
    def method_done(out,method):
        folder=out/method
        if not folder.exists(): return False
        marker=folder/'inference_complete.json'
        if marker.exists():
            done=read(marker)
            if done['status']=='COMPLETED' and done['frames']==read(out/'manifest.json')['frames']:
                for name,key in [('receipts.jsonl','receipts_sha256'),('processing_segments.json','segments_sha256')]:
                    if sha(folder/name)!=done[key]: raise ValueError('Completed receipt changed: '+str(folder/name))
                return True
        if not retry_incomplete:
            raise RuntimeError('Partial method exists: '+str(folder)+'; use --resume --retry-incomplete to archive and restart it')
        destination=root/'interrupted'/stamp()/out.name/method
        destination.parent.mkdir(parents=True)
        shutil.move(str(folder),str(destination))
        return False

    def paired(out,lease):
        # launch_locked always launches both methods. Invoke its existing worker
        # command directly so validated completed methods can be resumed safely.
        import subprocess
        for method in METHODS:
            if method_done(out,method):
                print('REUSE_COMPLETED_METHOD',out,method,flush=True); continue
            folder=out/method; folder.mkdir()
            command=['taskset','-c','0-3,6-31',PYTHON,'-B','-u','-X','faulthandler',
                '-m','tools.environment_stability_eval.fq3','--_worker',method,'--output',str(out),'--_lock-fd',str(lease)]
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            with (folder/'runtime.log').open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lease,))
                write(folder/'process.json',dict(pid=process.pid,command=command))
                print('STARTED_EXPANDED',method,process.pid,folder/'runtime.log',flush=True)
                status=process.wait()
            if status:
                raise RuntimeError(f'{method} exited {status}: '+(folder/'runtime.log').read_text(errors='replace')[-4000:])

    print('WAIT_PROJECT_GPU_LEASE (existing jobs are not interrupted)',flush=True)
    with gpu_lease() as lease:
        smoke=root/'smoke_validation'
        if not smoke.exists():
            args=SimpleNamespace(config=root/'plan'/'config.yaml',driver_environment=driver)
            prepare_fq3(args,smoke,True)
        verify_manifest(smoke)
        write(root/'status.json',dict(status='SMOKE_VALIDATION'))
        paired(smoke,lease)
        report(root,smoke,True)
        print('EXPANDED_REAL_SMOKE_PASS',flush=True)
        write(root/'status.json',dict(status='RUNNING_FULL',frames_per_method=manifest['frames']))
        paired(evaluation,lease)
    report(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only',action='store_true',help='Prepare real GT, split and coverage; no model/CUDA')
    mode.add_argument('--run-full',action='store_true',help='Prepare, real paired smoke, full paired inference and all reports')
    mode.add_argument('--report-only',action='store_true',help='Verify completed receipts and regenerate a new report directory')
    parser.add_argument('--resume',action='store_true',help='Use an existing frozen prepared output; never redo the split')
    parser.add_argument('--retry-incomplete',action='store_true',help='Archive partial method directories and rerun those methods')
    parser.add_argument('--calibration-scenes',type=int,default=30)
    parser.add_argument('--split-seed',type=int,default=20261118)
    parser.add_argument('--target-events',type=int,default=50)
    parser.add_argument('--target-scenes',type=int,default=10)
    args=parser.parse_args()
    os.chdir(ROOT); require_affinity()
    if args.target_events<1 or args.target_scenes<1: parser.error('Planning targets must be positive')
    if (args.resume or args.report_only) and args.output is None: parser.error('Existing output must be explicit')
    if args.retry_incomplete and not (args.resume and args.run_full): parser.error('--retry-incomplete requires --resume --run-full')
    root=(args.output or ROOT/'tools/environment_stability_eval/outputs'/('fq3_expanded_'+stamp())).resolve()
    driver=environment() if args.run_full else {}
    try:
        if args.report_only:
            report(root); return
        if args.resume:
            from .evaluator import verify_manifest
            manifest=verify_manifest(root/'evaluation')
            if manifest.get('expansion_protocol')!=PROTOCOL: raise ValueError('Not an expanded prepared run')
            if (root/'completion.json').exists():
                raise FileExistsError('Already completed; use --report-only for a new derived report')
        else:
            prepare(root,args.calibration_scenes,args.split_seed,driver,args.target_events,args.target_scenes)
        print('EXPANDED_OUTPUT',root,flush=True)
        if args.run_full: launch(root,driver,args.retry_incomplete)
    except Exception:
        if root.exists():
            write(root/'failure.json',dict(traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
