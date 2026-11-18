"""Run frozen R0 factual QA and 1/2/3-second planning, then score raw outputs.

No inference is performed by --check-only, --report-only, or importing this file.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

from .task_metrics import (FACT_GROUPS, PLANNING_QUESTION, fact_summary,
                           performance, planning_summary, score_planning, valid_gt)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INFO = Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
DEFAULT_DATA = Path('data/nuscenes')
MAIN = Path('outputs/m123_r0_only_reproduction_run01')
CHECKPOINT_SHA = '6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d'


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def records(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stage(out, name, **details):
    """Flush a durable marker before crossing a native extension boundary."""
    row = dict(stage=name, pid=os.getpid(), time=datetime.now().astimezone().isoformat(), **details)
    write(Path(out)/'runtime_stage.json', row)
    with (Path(out)/'runtime_stages.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')
    print('STAGE', name, json.dumps(details, ensure_ascii=False), flush=True)


def worker_command(args):
    command = [sys.executable, '-B', '-u', '-X', 'faulthandler', '-m',
               'tools.worldstream_m123.r0_task_evaluation', '--_inference-worker',
               '--output', str(args.output.resolve())]
    if args.runtime_check_only:
        command.append('--runtime-check-only')
    return command


def launch_worker(args):
    """Use exec of a fresh interpreter, never inherit preflight native state."""
    out = args.output
    command = worker_command(args)
    print('START_FRESH_GPU_WORKER', flush=True)
    with (out/'runtime.log').open('x') as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            write(out/'worker_process.json', dict(pid=child.pid, parent_pid=os.getpid(),
                  command=command, cpu_affinity=sorted(os.sched_getaffinity(0))))
            for line in child.stdout:
                log.write(line);log.flush()
                print(line, end='', flush=True)
            code = child.wait()
    if code:
        last = json.loads((out/'runtime_stage.json').read_text()) if (out/'runtime_stage.json').exists() else None
        write(out/'worker_failure.json', dict(returncode=code, last_stage=last,
              runtime_log=str(out/'runtime.log'), native_crash_log=str(out/'native_crash.log')))
        raise RuntimeError(f'GPU worker exited with code {code}; last stage={last}. '
                           f'See {out}/runtime.log and native_crash.log')


def inference_worker(args):
    import faulthandler
    out = args.output
    manifest = json.loads((out/'run_manifest.json').read_text())
    args.infos = Path(manifest['infos'])
    args.data_root = Path(manifest['data_root'])
    args.checkpoint = Path(manifest['checkpoint'])
    args.seed = manifest['seed']
    args.other_questions = manifest['other_questions']
    with (out/'native_crash.log').open('a') as crash:
        faulthandler.enable(file=crash, all_threads=True)
        try:
            stage(out, 'WORKER_STARTED', inherited_preflight_state=False,
                  cpu_affinity=sorted(os.sched_getaffinity(0)))
            run_inference(args, manifest['scenes'])
        finally:
            faulthandler.disable()


def facts_questions(token, data_root, other_count):
    from .selective_answer_reuse import QUESTIONS
    qs = list(QUESTIONS)
    for folder in ('conv', 'vqa'):
        p = Path(data_root)/folder/'val'/(token+'.json')
        if p.exists():
            for item in json.loads(p.read_text()):
                question = item['question']
                if question not in qs and len(qs) < 3+other_count:
                    qs.append(question)
    return qs, list(FACT_GROUPS[:3])+['other']*(len(qs)-3)


def selected_scenes(infos, scope, manifest, max_scenes):
    grouped = defaultdict(list)
    for info in infos:
        grouped[info['scene_token']].append(info)
    for values in grouped.values():
        values.sort(key=lambda x:x['timestamp'])
    if scope == 'main-scenes':
        selection = json.loads((Path(manifest)/'scene_selection.json').read_text())['selected']
        scenes = []
        for scene in selection:
            by_token = {i['token']:i for i in grouped.get(scene['scene_id'], [])}
            if set(scene['tokens'])-set(by_token):
                raise ValueError('Selected main Scene is missing from info pickle')
            # Preserve every sample of each selected scene and its original order.
            actual = grouped[scene['scene_id']]
            if [i['token'] for i in actual] != scene['tokens']:
                raise ValueError('Main selection must contain the complete chronological Scene')
            scenes.append(dict(scene_id=scene['scene_id'], tokens=scene['tokens']))
    else:
        scenes = [dict(scene_id=sid, tokens=[i['token'] for i in values])
                  for sid, values in sorted(grouped.items())]
    if max_scenes:
        scenes = scenes[:max_scenes]
    for scene in scenes:
        by = {i['token']:i for i in grouped[scene['scene_id']]}
        frames = [by[t] for t in scene['tokens']]
        if any(a['timestamp'] >= b['timestamp'] or a.get('next') != b['token'] or
               b.get('prev') != a['token'] for a, b in zip(frames, frames[1:])):
            raise ValueError('Non-contiguous scene stream: '+scene['scene_id'])
    if not scenes:
        raise ValueError('No scenes selected')
    return scenes


def preflight(args):
    if {4, 5} & os.sched_getaffinity(0):
        raise RuntimeError('Launch with taskset -c 0-3,6-31')
    if args.data_root.resolve() != DEFAULT_DATA.resolve():
        raise ValueError('Current frozen loaders/QA references require data/nuscenes; relocate its symlink instead.')
    required = [args.infos, args.checkpoint, Path('ckpts/pretrain_qformer/config.json'),
                Path('ckpts/pretrain_qformer/pytorch_model-00001-of-00002.bin'),
                Path('ckpts/pretrain_qformer/pytorch_model-00002-of-00002.bin'),
                Path('configs/worldstream_r0.json'),
                args.data_root/'v1.0-trainval/sample_annotation.json',
                args.data_root/'v1.0-trainval/instance.json',
                args.data_root/'v1.0-trainval/category.json',
                args.data_root/'conv/val', args.data_root/'vqa/val']
    if args.scope == 'main-scenes':
        required.append(args.main_manifest/'scene_selection.json')
    for city in ('boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown'):
        required.append(args.data_root/'maps/expansion'/(city+'.json'))
    missing = [str(p) for p in required if not p.exists()]
    dependencies = ('torch', 'mmcv', 'mmdet', 'mmdet3d', 'transformers', 'peft',
                    'nuscenes', 'pyquaternion', 'cv2', 'skimage', 'shapely', 'descartes')
    missing += ['Python module: '+n for n in dependencies if importlib.util.find_spec(n) is None]
    if missing:
        raise FileNotFoundError('Missing dependencies: '+repr(missing))
    with args.infos.open('rb') as f:
        infos = pickle.load(f)['infos']
    scenes = selected_scenes(infos, args.scope, args.main_manifest, args.max_scenes)
    by = {i['token']:i for i in infos}
    counts = {str(s)+'s':0 for s in (1, 2, 3)}
    for scene in scenes:
        for token in scene['tokens']:
            info = by[token]
            if len(info['cams']) != 6:
                raise ValueError('Expected six cameras: '+token)
            for cam in info['cams'].values():
                if not Path(cam['data_path']).is_file():
                    raise FileNotFoundError(cam['data_path'])
            for key in ('gt_boxes', 'gt_velocity', 'gt_fut_traj', 'gt_fut_traj_mask',
                        'gt_fut_yaw', 'gt_fut_idx', 'ego2global_rotation', 'ego2global_translation', 'location'):
                if key not in info:
                    raise ValueError('Missing planning GT '+key+': '+token)
            _, mask = valid_gt(info)
            for s in (1, 2, 3):
                counts[str(s)+'s'] += int(mask[:2*s].all())
    if not counts['3s']:
        raise ValueError('No valid 3-second planning GT')
    result = dict(status='PREFLIGHT_PASS_NO_INFERENCE', scope=args.scope,
                  scenes=len(scenes), frames=sum(len(s['tokens']) for s in scenes),
                  planned_full_model_calls=2*sum(len(s['tokens']) for s in scenes),
                  planning_GT_valid=counts, facts_per_frame='3 fixed factual questions + up to '+str(args.other_questions)+' original questions',
                  planning_per_frame=1, checkpoint=str(args.checkpoint),
                  GPU='physical GPU 0, exclusive project lock; waits for other workloads',
                  independent_TEST=False)
    return result, scenes, by


def run_inference(args, scenes):
    out = args.output
    stage(out, 'IMPORT_TORCH')
    import torch
    from .benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu

    stage(out, 'WAIT_GPU_LOCK')
    with gpu_job_lock():
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '0'):
            raise RuntimeError('This runner waits for physical GPU 0; use CUDA_VISIBLE_DEVICES=0')
        while True:
            stage(out, 'WAIT_GPU_IDLE')
            wait_gpu()
            active = subprocess.check_output(['nvidia-smi', '--id=0',
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            if not active:
                break
            print('WAIT GPU 0 has active compute processes; waiting without stopping them', flush=True)
            time.sleep(30)
        stage(out, 'CUDA_INITIALIZATION')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: no model inference executed')
        torch.cuda.set_device(0)
        stage(out, 'CUDA_READY', device=torch.cuda.get_device_name(0))
        # Initialize CUDA before plugin/dataset imports and large pickle reads.
        # The parent has already checked all data in a separate interpreter.
        stage(out, 'IMPORT_MODEL_PLUGINS')
        import projects.mmdet3d_plugin
        from mmcv.parallel import DataContainer, collate, scatter
        from mmdet3d.datasets import build_dataset
        from .benchmark.worker import effective_config, load_model, reset
        from .benchmark.common import diagnostic_json
        from .frozen_main_benchmark import frame_rng
        from .selective_answer_reuse import encode_questions
        from worldstream.m3.request_rng import request_seed
        from worldstream.r0 import model_settings
        stage(out, 'LOAD_CONFIG')
        cfg = effective_config('B3', 'planning', out)
        # Only the annotation path/output are evaluation overrides.
        cfg.data.test.ann_file = str(args.infos)
        if args.runtime_check_only:
            stage(out, 'RUNTIME_CHECK_PASS_NO_MODEL_INFERENCE')
            write(out/'runtime_check.json', dict(status='PASS', fresh_interpreter=True,
                  CUDA_initialized=True, plugins_imported=True, config_loaded=True,
                  model_loaded=False, model_forward_calls=0,
                  device=torch.cuda.get_device_name(0)))
            return
        stage(out, 'VERIFY_CHECKPOINT')
        actual = sha(args.checkpoint)
        if actual != CHECKPOINT_SHA:
            raise ValueError('Checkpoint differs from frozen R0 checkpoint')
        stage(out, 'LOAD_MODEL_WEIGHTS')
        model = load_model(cfg, str(args.checkpoint), out).eval().requires_grad_(False)
        stage(out, 'MODEL_READY')
        expected = model_settings()
        if not (model.m1_manager and model.m2_runtime and model.m3_runtime):
            raise RuntimeError('Full M1/M2/M3 R0 did not load')
        from dataclasses import asdict
        effective = dict(M1=asdict(model.m1_manager.config), M2=asdict(model.m2_runtime.config),
                         M3=asdict(model.m3_runtime.config))
        for label, key in [('M1','m1_config'), ('M2','m2_config'), ('M3','m3_config')]:
            if diagnostic_json(effective[label]) != expected[key]:
                raise ValueError('Non-R0 runtime configuration: '+label)
        write(out/'effective_runtime.json', dict(**diagnostic_json(effective), checkpoint_sha256=actual,
              device=torch.cuda.get_device_name(0), generation=dict(do_sample=True, temperature=.1, top_p=.75, max_new_tokens=320)))
        model.worldstream_audit_identity = dict(checkpoint_sha256=actual)
        model.worldstream_question_rng_enabled = True
        model.worldstream_request_rng_context = dict(repeat=0, experiment_seed=args.seed, rng_audit=True)
        stage(out, 'BUILD_DATASET')
        dataset = build_dataset(cfg.data.test)
        original = {i['token']:i for i in dataset.data_infos}
        tokens = [t for s in scenes for t in s['tokens']]
        dataset.data_infos = [original[t] for t in tokens]
        index = {i['token']:k for k,i in enumerate(dataset.data_infos)}
        calls = 0
        write(out/'inference_status.json', dict(status='RUNNING', full_model_calls=0))
        for task in ('facts', 'planning'):
            with (out/(task+'_receipts.jsonl')).open('x') as receipt_file:
                for si, scene in enumerate(scenes):
                    reset(model)
                    answers = out/'answers'/task/scene['scene_id']
                    answers.mkdir(parents=True, exist_ok=False)
                    model.save_path = str(answers)+'/'
                    for fi, token in enumerate(scene['tokens']):
                        stage(out, 'INFERENCE_FRAME', task=task, scene_index=si, frame_index=fi, frame_id=token)
                        context = dict(scene_id=scene['scene_id'], frame_id=token, repeat=0, experiment_seed=args.seed)
                        with frame_rng(request_seed(context, -1)):
                            torch.cuda.synchronize()
                            tick = time.perf_counter()
                            data = scatter(collate([dataset[index[token]]], samples_per_gpu=1), [0])[0]
                            meta = data['img_metas'][0][0]
                            if task == 'facts':
                                questions, groups = facts_questions(token, args.data_root, args.other_questions)
                                data['input_ids'][0][0] = encode_questions(model, questions, original[token]['location'])
                                meta['vlm_labels'] = DataContainer(questions, stack=False)
                            else:
                                questions, groups = [PLANNING_QUESTION], ['planning']
                                if list(meta['vlm_labels'].data) != questions:
                                    raise RuntimeError('Native planning prompt changed')
                            torch.cuda.synchronize()
                            prep_ms = (time.perf_counter()-tick)*1000
                            model_tick = time.perf_counter()
                            with torch.no_grad():
                                outputs = model(return_loss=False, rescale=True, **data)
                            torch.cuda.synchronize()
                            end = time.perf_counter()
                            model_ms, e2e_ms = (end-model_tick)*1000, (end-tick)*1000
                        answers_out = outputs[0]['text_out']
                        if len(answers_out) != len(questions) or any(a['Q'] != q for a,q in zip(answers_out,questions)):
                            raise RuntimeError('Model answers do not match requested questions')
                        if not model.worldstream_execution['m1']['commit_completed'] or not model.m2_diagnostics['execution_completed']:
                            raise RuntimeError('Incomplete R0 execution')
                        diag = model.m3_diagnostics
                        if not diag['execution_completed']:
                            raise RuntimeError('Incomplete M3 execution')
                        requests = diag['requests']
                        row = dict(task=task, frame_id=token, scene_id=scene['scene_id'], frame_index=fi,
                                   questions=questions, groups=groups,
                                   answers=[a['A'][0] for a in answers_out],
                                   e2e_ms=e2e_ms, input_preparation_ms=prep_ms, model_ms=model_ms,
                                   seed=args.seed, llm_skipped=[r['llm_skipped'] for r in requests],
                                   llm_forward_calls=[r['llm_forward_calls'] for r in requests])
                        receipt_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        receipt_file.flush()
                        calls += 1
                        if fi % 8 == 0:
                            print('PROGRESS', task, 'scene', si+1, '/', len(scenes), 'frame', fi+1, '/', len(scene['tokens']), 'calls', calls, flush=True)
                        write(out/'progress.json', dict(task=task, scene_index=si, frame_index=fi, full_model_calls=calls))
        write(out/'inference_status.json', dict(status='COMPLETED', full_model_calls=calls,
              receipt_hashes={name:sha(out/name) for name in ('facts_receipts.jsonl','planning_receipts.jsonl')},
              scopes='Facts and planning are separate chronological passes; reset at every task/Scene boundary.'))
        import gc
        del model, dataset
        gc.collect()
        torch.cuda.empty_cache()
        stage(out, 'INFERENCE_COMPLETED', full_model_calls=calls)


def report(args):
    from .answer_validity_quality import load_current_annotations
    from .continuous_stream_report import grade
    from .task_planning import PlanningBackend
    out = args.output
    status = json.loads((out/'inference_status.json').read_text())
    if status['status'] != 'COMPLETED':
        raise ValueError('Cannot score incomplete inference')
    manifest = json.loads((out/'run_manifest.json').read_text())
    for group in ('input_hashes','reference_hashes'):
        for name, expected_hash in manifest[group].items():
            if sha(name) != expected_hash:
                raise ValueError('Code/config/GT/reference changed since run preparation: '+name)
    for name, expected_hash in status['receipt_hashes'].items():
        if sha(out/name) != expected_hash:
            raise ValueError('Raw inference receipt changed: '+name)
    expected = [t for scene in manifest['scenes'] for t in scene['tokens']]
    if status['full_model_calls'] != 2*len(expected):
        raise ValueError('Unexpected model call count')
    frames = {name:list(records(out/(name+'_receipts.jsonl'))) for name in ('facts','planning')}
    for name, rows in frames.items():
        if [r['frame_id'] for r in rows] != expected:
            raise ValueError('Missing/duplicated/reordered '+name+' receipts')
    expected_set = set(expected)
    infos = {i['token']:i for i in pickle.load(args.infos.open('rb'))['infos'] if i['token'] in expected_set}
    objects = load_current_annotations(str(args.infos), set(expected), str(args.data_root))
    events = []
    with (out/'per_question_metrics.jsonl').open('w') as handle:
        for row in frames['facts']:
            if not len(row['questions']) == len(row['groups']) == len(row['answers']):
                raise ValueError('Question/answer length mismatch')
            for q, group, answer in zip(row['questions'], row['groups'], row['answers']):
                quality = grade(q, answer, row['frame_id'], objects)
                event = dict(frame_id=row['frame_id'], group=group, question=q, answer=answer, quality=quality)
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
                events.append(event)
    facts = dict(performance=performance(frames['facts']), quality=fact_summary(events))
    write(out/'fact_qa_metrics.json', facts)
    backend = PlanningBackend(args.data_root)
    planning_rows = []
    with (out/'per_frame_planning_metrics.jsonl').open('w') as handle:
        for index, row in enumerate(frames['planning']):
            scored = dict(frame_id=row['frame_id'], **score_planning(row['answers'][0], infos[row['frame_id']], backend))
            handle.write(json.dumps(scored, allow_nan=False)+'\n')
            planning_rows.append(scored)
            if index % 50 == 0:
                print('SCORING_PLANNING', index+1, '/', len(expected), flush=True)
    planning = dict(performance=performance(frames['planning']), **planning_summary(planning_rows))
    write(out/'planning_metrics.json', planning)
    write(out/'metrics.json', dict(model='OmniDrive full frozen R0 M1+M2+M3', scope=manifest['scope'],
          facts=facts, planning=planning, source_manifest='run_manifest.json', independent_TEST=False))
    render_report(out, facts, planning, manifest)
    write(out/'completion.json', dict(status='COMPLETED', frames_per_task=len(expected),
          facts_scored=facts['quality']['overall']['scored'], planning_3s_scored=planning['horizons']['3s']['scored_trajectories'],
          partial_planning_scoring=any(not p['collision_denominator'] or not p['road_boundary_denominator'] for p in planning['horizons'].values()),
          report='TASK_METRICS.md'))


def render_report(out, facts, planning, manifest):
    import csv
    show = lambda x: f'{x:.4f}' if isinstance(x, (int,float)) else 'N/A'
    p = facts['performance']
    lines = ['# R0 事实问答与规划评估', '',
             f"范围：{manifest['scope']}，{len(manifest['scenes'])} 个连续 Scene；QA 与规划分别运行并重置状态。",
             '时延按单个视觉帧及其全部问题统计，含预处理、H2D、完整推理和答案导出；含冷启动帧，不含模型加载、排队与离线评分。', '',
             '| 事实问答平均E2E(ms) | p95(ms) | FPS | 可评分事实准确率(%) | 可评分问题/全部问题 |',
             '|---:|---:|---:|---:|---:|',
             f"| {show(p['mean_e2e_ms'])} | {show(p['p95_e2e_ms'])} | {show(p['FPS'])} | {show(facts['quality']['overall']['scorable_fact_accuracy_percent'])} | {facts['quality']['overall']['scored']}/{facts['quality']['overall']['questions']} |", '',
             '| 问题组 | 有限事实评分(%) | 可评分数 | 未评分数 |', '|---|---:|---:|---:|']
    names = dict(vehicle_category='车辆类别', vehicle_existence='车辆存在性', pedestrian_position='行人相对位置', other='其他原始问题')
    for name, result in facts['quality']['groups'].items():
        lines.append(f"| {names[name]} | {show(result['scorable_fact_accuracy_percent'])} | {result['scored']} | {result['unscored']} |")
    lines += ['', '其他问题使用当前帧原始参考答案中的有限事实召回评分，可为小数；不代表通用自由问答准确率。UNKNOWN不填为正确。', '',
              '| 规划时域 | L2/ADE(m) | 端点L2/FDE(m) | 碰撞率(%) | 道路越界率(%) | 有效预测/GT有效 | 碰撞/道路分母 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    with (out/'planning_metrics.csv').open('w', newline='') as f:
        writer = None
        for horizon, result in planning['horizons'].items():
            flat = dict(horizon=horizon, **result)
            if writer is None:
                writer=csv.DictWriter(f, fieldnames=list(flat));writer.writeheader()
            writer.writerow(flat)
            lines.append(f"| {horizon} | {show(result['L2_ADE_m'])} | {show(result['endpoint_L2_FDE_m'])} | {show(result['collision_rate_percent'])} | {show(result['road_boundary_rate_percent'])} | {result['scored_trajectories']}/{result['GT_valid_frames']} | {result['collision_denominator']}/{result['road_boundary_denominator']} |")
    lines += ['', 'L2/ADE是前2/4/6个0.5秒轨迹点的平均欧氏距离；端点误差单列。碰撞采用原评估器车辆占用及自车包围框，主值排除GT已碰撞的时间点，另保存未掩码碰撞率。道路越界检查旋转自车矩形是否离开drivable_area，初始已越界样本单独计数并排除。解析失败、GT缺失及超出栅格的覆盖率详见JSON，未评分项显示N/A。']
    (out/'TASK_METRICS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--scope', choices=('main-scenes','full-val'), default='main-scenes')
    parser.add_argument('--main-manifest', type=Path, default=MAIN)
    parser.add_argument('--infos', type=Path, default=DEFAULT_INFO)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--checkpoint', type=Path, default=Path('ckpts/iter_10548.pth'))
    parser.add_argument('--other-questions', type=int, default=2)
    parser.add_argument('--max-scenes', type=int, default=None)
    parser.add_argument('--seed', type=int, default=20261030)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--report-only', action='store_true')
    mode.add_argument('--runtime-check-only', action='store_true',
                      help='Run full CPU preflight plus fresh-process CUDA/plugin/config initialization, without loading model weights or inference.')
    parser.add_argument('--_inference-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if {4, 5} & os.sched_getaffinity(0):
        raise RuntimeError('Launch with taskset -c 0-3,6-31')
    if args.other_questions < 0 or (args.max_scenes is not None and args.max_scenes <= 0):
        parser.error('other-questions must be nonnegative and max-scenes positive')
    if args.output is None:
        args.output = Path('outputs')/('r0_task_metrics_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if args._inference_worker:
        inference_worker(args)
        return
    if args.report_only:
        manifest = json.loads((args.output/'run_manifest.json').read_text())
        args.infos, args.data_root = Path(manifest['infos']), Path(manifest['data_root'])
        report(args)
        return
    check, scenes, _ = preflight(args)
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'preflight.json', check)
    paths = [args.infos, Path('configs/worldstream_r0.json'), Path('projects/configs/OmniDrive/worldstream_m123_auto.py')]
    paths += [Path(__file__), Path('tools/worldstream_m123/task_metrics.py'), Path('tools/worldstream_m123/task_planning.py'),
              Path('evaluation/planning_utils.py'), Path('evaluation/map_api.py')]
    paths += [p for folder in ('projects','worldstream','tools/worldstream_m123')
              for p in Path(folder).rglob('*.py')]
    references = [args.data_root/'v1.0-trainval'/name for name in
                  ('sample_annotation.json','category.json','instance.json')]
    references += list((args.data_root/'maps/expansion').glob('*.json'))
    references += [args.data_root/folder/'val'/(t+'.json') for s in scenes for t in s['tokens']
                   for folder in ('conv','vqa') if (args.data_root/folder/'val'/(t+'.json')).exists()]
    manifest = dict(scope=args.scope, scenes=scenes, infos=str(args.infos), data_root=str(args.data_root),
                    checkpoint=str(args.checkpoint),
                    other_questions=args.other_questions, seed=args.seed,
                    created_at=datetime.now().astimezone().isoformat(), input_hashes={str(p):sha(p) for p in sorted(set(paths))},
                    reference_hashes={str(p):sha(p) for p in sorted(set(references))},
                    expected_checkpoint_sha256=CHECKPOINT_SHA,
                    native_or_R0_parameters_changed=False, independent_TEST=False)
    write(args.output/'run_manifest.json', manifest)
    try:
        launch_worker(args)
        if args.runtime_check_only:
            print('RUNTIME_CHECK_PASS_NO_MODEL_INFERENCE', args.output, flush=True)
            return
        else:
            stage(args.output, 'OFFLINE_SCORING')
            report(args)
    except BaseException:
        write(args.output/'failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise
    print('COMPLETED', args.output/'TASK_METRICS.md', flush=True)


if __name__ == '__main__':
    main()
