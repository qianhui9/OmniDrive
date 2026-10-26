"""Frozen-policy, scene-disjoint M3 validation. No calibration or policy writes.

Use prepare, run, then analyze in an unused run directory. Existing results are
never replaced. All GPU workers are sequential and inherit safe CPU affinity.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from .common import sha256, write
from .sequence import validate_sequence
from worldstream.m3.task_conditioner import task_profile

POLICY = Path('outputs/final_route/m3_reuse_policy.json')
CONFIG = Path('projects/configs/OmniDrive/worldstream_m123_auto.py')
PYTHON = '/home/qh/miniconda3/envs/OmniDrive/bin/python'


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def source_hashes():
    paths = {p for folder in ('worldstream', 'projects/mmdet3d_plugin',
        'projects/configs/OmniDrive', 'tools/worldstream_m123/benchmark')
        for p in Path(folder).rglob('*.py')}
    paths.update((POLICY, CONFIG, Path('configs/worldstream_m1_final.yaml'),
        Path('outputs/final_route/window_costs.json'), Path('tools/test.py')))
    return {str(p): sha256(p) for p in sorted(paths)}


def historical_audit(outputs, current, infos):
    """All historical language runs plus every M3/M123 sequence/evidence file.

    Full-split perception-only M2 predictions are separately disclosed; they
    cannot have selected M3 thresholds or assessed KV output quality.
    """
    by_token = {str(i['token']): i for i in infos}
    scenes, frames, questions, texts = set(), set(), set(), set()
    sources, ignored = [], []
    for p in sorted(Path(outputs).rglob('*')):
        if not p.is_file() or current.resolve() in p.resolve().parents:
            continue
        relevant = (p.name in ('per_question_metrics.jsonl', 'paired_records.jsonl', 'service_evidence.jsonl')
            or ('sequence' in p.name and p.suffix == '.json' and
                any(part.startswith(('m3_', 'm123_')) for part in p.parts)))
        if not relevant:
            if p.name == 'sequence.json':
                ignored.append(str(p))
            continue
        values = read_rows(p) if p.suffix == '.jsonl' else [json.loads(p.read_text())]
        local_frames, local_scenes, local_questions = set(), set(), set()
        def visit(v):
            if isinstance(v, list):
                for x in v: visit(x)
            elif isinstance(v, dict):
                token = v.get('frame_id') or v.get('sample_token')
                if token in by_token: local_frames.add(token)
                scene = v.get('scene_id') or v.get('scene')
                if scene: local_scenes.add(str(scene))
                local_scenes.update(v.get('scene_ids', []))
                local_scenes.update(v.get('scene_splits', {}))
                for key in ('tokens', 'sample_tokens'):
                    local_frames.update(t for t in v.get(key, []) if t in by_token)
                if token and isinstance(v.get('question'), str):
                    local_questions.add((str(token), v['question']))
                    texts.add(v['question'])
                for x in v.values():
                    if isinstance(x, (dict, list)): visit(x)
        visit(values)
        local_scenes.update(by_token[t]['scene_token'] for t in local_frames)
        scenes.update(local_scenes); frames.update(local_frames); questions.update(local_questions)
        sources.append(dict(path=str(p), sha256=sha256(p), frames=sorted(local_frames),
            scenes=sorted(local_scenes), question_pairs=sorted(local_questions)))
    # Policy split membership excludes scenes even if raw execution is missing.
    policy = json.loads(POLICY.read_text())
    scenes.update(policy.get('scene_splits', {}))
    return dict(excluded_scenes=sorted(scenes), excluded_frames=sorted(frames),
        excluded_frame_questions=sorted(questions), seen_question_texts=sorted(texts),
        sources=sources, non_m3_sequence_files_not_used_for_exclusion=ignored,
        scope='All historical labelled language evaluations; M3/M123 sequences, calibration and service records. '
            'Full perception-only evaluations do not establish unseen perception data, only unseen M3 quality/development scenes.')


def labelled_pairs(token):
    pairs = []
    for name in ('conv', 'vqa'):
        path = Path('data/nuscenes')/name/'val'/(token+'.json')
        if path.is_file():
            pairs.extend(dict(pair, source=str(path), source_sha256=sha256(path))
                for pair in json.loads(path.read_text()))
    counts = Counter(p['question'] for p in pairs)
    return [p for p in pairs if counts[p['question']] == 1 and p.get('answer')]


def choose_questions(pairs, ordinal, count):
    profiles = ['map_relations', 'object_attributes', 'object_ego_motion', 'full_visual']
    # Rotate metadata profiles before observing embeddings, risk or outputs.
    profiles = profiles[ordinal % 4:] + profiles[:ordinal % 4]
    selected = []
    for profile in profiles:
        eligible = [p for p in pairs if task_profile(p['question'])[1] == profile]
        if eligible: selected.append(eligible[ordinal % len(eligible)])
        if len(selected) == count: break
    selected += [p for p in pairs if p not in selected][:count-len(selected)]
    return selected


def prepare(root, args):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'sequence.json').exists():
        raise FileExistsError('Prepared sequence already exists; use a new run directory')
    (root/'raw_logs').mkdir(exist_ok=True)
    (root/'frozen').mkdir(exist_ok=True)
    from mmcv import Config
    cfg = Config.fromfile(str(CONFIG))
    annotation = Path(cfg.data.test.ann_file)
    infos = pickle.load(annotation.open('rb'))['infos']
    audit = historical_audit('outputs', root, infos)
    write(root/'historical_data_audit.json', audit)
    by_scene = defaultdict(list)
    for info in infos:
        if info['scene_token'] not in audit['excluded_scenes']:
            by_scene[info['scene_token']].append(info)
    eligible, rejected = [], []
    for scene, rows in sorted(by_scene.items()):
        rows.sort(key=lambda r: r['timestamp'])
        segment = rows[:args.frames_per_scene]
        if len(segment) < args.frames_per_scene: continue
        try:
            continuity = validate_sequence(segment, scene)
            if any(a.get('next') != b['token'] or b.get('prev') != a['token']
                    for a, b in zip(segment, segment[1:])):
                raise ValueError('nuScenes prev/next links are not consecutive')
            for r in segment:
                if len(r['cams']) != 6 or not all(Path(c['data_path']).is_file() for c in r['cams'].values()):
                    raise ValueError('Missing real camera images')
                if len(labelled_pairs(r['token'])) < args.questions_per_frame:
                    raise ValueError('Insufficient unambiguous labelled questions')
            eligible.append((continuity['ego_sampled_path_length_m'], scene, segment, continuity))
        except ValueError as e:
            rejected.append(dict(scene_id=scene, reason=str(e)))
    eligible.sort(key=lambda r: (r[0], r[1]))
    n = min(args.scenes, len(eligible))
    indices = [min(len(eligible)-1, int((j+.5)*len(eligible)/n)) for j in range(n)]
    picked = [eligible[i] for i in indices]
    if not picked: raise ValueError('No independent natural scenes available')
    sequence = dict(tokens=[], questions={}, scene_splits={}, frame_splits={},
        selection='First consecutive linked segment from metadata motion quantiles of scene-disjoint pool; '
            'question profiles rotated before model execution; no embedding/output filtering',
        checkpoint=args.checkpoint, seed=args.seed, scenes=[])
    manifest = []
    seen_texts = set(audit['seen_question_texts'])
    for motion, scene, segment, continuity in picked:
        sequence['scene_splits'][scene] = 'independent_validation'
        sequence['scenes'].append(dict(scene_id=scene, continuity=continuity,
            full_scene_frames=len(by_scene[scene]), selected_frames=len(segment), full_scene_selected=False))
        for info in segment:
            pairs = choose_questions(labelled_pairs(info['token']), len(sequence['tokens']), args.questions_per_frame)
            sequence['tokens'].append(info['token'])
            sequence['frame_splits'][info['token']] = 'independent_validation'
            sequence['questions'][info['token']] = [p['question'] for p in pairs]
            manifest.append(dict(scene_id=scene, scene_name=info.get('scene_name'),
                frame_id=info['token'], timestamp_us=int(info['timestamp']),
                prev=info.get('prev'), next=info.get('next'),
                scene_overlap=False, frame_overlap=False, frame_question_overlap=False,
                camera_paths={k:c['data_path'] for k,c in info['cams'].items()},
                object_classes=dict(Counter(map(str, info.get('gt_names', [])))),
                questions=[dict(p, task_type=task_profile(p['question'])[0],
                    task_profile=task_profile(p['question'])[1], question_text_seen=p['question'] in seen_texts)
                    for p in pairs]))
    write(root/'sequence.json', sequence)
    write(root/'data_manifest.json', dict(annotation=str(annotation), annotation_sha256=sha256(annotation),
        historical_audit_sha256=sha256(root/'historical_data_audit.json'), frames=manifest,
        requested_scenes=args.scenes, eligible_scenes=len(eligible), excluded_scenes=len(audit['excluded_scenes']),
        rejected_scenes=rejected, actual_scenes=n,
        limitation='Scene-isolated M3 evidence; short contiguous segments, not entire 20-second scenes. '
            'Shared generic question text is disclosed and does not imply shared frame-question inputs.'))
    initial = root/'freeze_initial.json'
    if not initial.exists():
        write(initial, dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            git_status=subprocess.check_output(['git','status','--porcelain'],text=True),
            checkpoint=dict(path=args.checkpoint, sha256=sha256(args.checkpoint)), source_hashes=source_hashes()))
    frozen = json.loads(initial.read_text())
    if frozen['checkpoint']['path'] != args.checkpoint:
        raise ValueError('Checkpoint differs from initial freeze')
    policy = json.loads(POLICY.read_text())
    if frozen['checkpoint']['sha256'] != policy['checkpoint_sha256']:
        raise ValueError('Actual checkpoint hash differs from frozen policy binding')
    for path in (POLICY, CONFIG):
        target = root/'frozen'/path.name
        if target.exists() and sha256(target) != sha256(path): raise ValueError('Frozen artifact changed')
        if not target.exists(): shutil.copy2(path, target)
    changed = [p for p,h in frozen['source_hashes'].items() if sha256(p) != h]
    if any(not p.startswith('tools/worldstream_m123/benchmark/') for p in changed):
        raise ValueError('Protected production code changed: '+repr(changed))
    write(root/'freeze_execution.json', dict(initial_manifest_sha256=sha256(initial),
        checkpoint=frozen['checkpoint'], source_hashes=source_hashes(),
        sequence_sha256=sha256(root/'sequence.json'), data_manifest_sha256=sha256(root/'data_manifest.json'),
        policy_sha256=sha256(POLICY), config_sha256=sha256(CONFIG),
        benchmark_only_changes=changed, production_logic_changed=False,
        no_recalibration=True, quality_budget=policy['quality_budget']))
    print('PREPARED', n, 'scenes', len(manifest), 'frames', sum(len(r['questions']) for r in manifest), 'questions', flush=True)


def verify_freeze(root, checkpoint=False):
    frozen = json.loads((root/'freeze_execution.json').read_text())
    changed = [p for p,h in frozen['source_hashes'].items() if not Path(p).is_file() or sha256(p) != h]
    for name,key in [('sequence.json','sequence_sha256'), ('data_manifest.json','data_manifest_sha256')]:
        if sha256(root/name) != frozen[key]: changed.append(name)
    if sha256(root/'frozen/m3_reuse_policy.json') != frozen['policy_sha256']: changed.append('frozen policy')
    if checkpoint and sha256(frozen['checkpoint']['path']) != frozen['checkpoint']['sha256']:
        changed.append('checkpoint')
    if changed: raise ValueError('Frozen artifacts changed: '+repr(changed))
    return frozen


def gpu_state():
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,power.draw',
        '--format=csv,noheader,nounits'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return dict(returncode=r.returncode, output=r.stdout, unix_time=time.time())


def run(root, args):
    verify_freeze(root, checkpoint=True)
    sequence = json.loads((root/'sequence.json').read_text())
    commands = []
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='0')
    plan = [dict(repeat=r, order=order, strategy=s, name=f'r{r}_{s}')
        for r in range(args.repeats) for order,s in enumerate(
            ('native_full','wsdr_kv') if r%2==0 else ('wsdr_kv','native_full'))]
    plan_path = root/'run_plan.json'
    plan_value = dict(jobs=plan, repeats=args.repeats, warmup_frames=args.warmup, seed=args.seed,
        generation='Unchanged native sampling; same per-frame hash-derived seed in every repeat and path',
        warmup='First N selected frames; all temporal state/cache reset before measurement',
        timing_units=dict(prefill='question', decode='question', m3_runtime='frame', end_to_end='frame'),
        environment={k:environment[k] for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES')})
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan_value:
        raise ValueError('Cannot change a resumed experiment plan')
    write(plan_path, plan_value)
    old = root/'commands.json'
    if old.exists(): commands = json.loads(old.read_text())
    for job in plan:
        dest = root/'runs'/job['name']
        if dest.exists():
            status_path = dest/'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('status') == 'COMPLETED':
                print('PRESERVED', job['name'], flush=True); continue
            raise RuntimeError('Prior incomplete/failed worker retained at '+str(dest))
        verify_freeze(root)
        while True:
            gpu = gpu_state()
            if gpu['returncode']: raise RuntimeError('GPU unavailable: '+gpu['output'])
            memory = float(gpu['output'].splitlines()[0].split(',')[3].strip())
            if memory < 500: break
            write(root/'driver_status.json', dict(status='WAITING_FOR_GPU', gpu=gpu))
            print('WAITING_FOR_GPU', memory, 'MiB; existing processes untouched', flush=True)
            time.sleep(20)
        command = ['taskset','-c','0-3,6-31',PYTHON,'-u','-m','tools.worldstream_m123.benchmark.worker',
            '--run-root',str(root),'--output',str(dest),'--condition','B3','--workload','language',
            '--limit',str(len(sequence['tokens'])),'--warmup',str(args.warmup),
            '--seed',str(args.seed),'--repeat',str(job['repeat']),'--measurement-set','efficiency',
            '--checkpoint',args.checkpoint,'--sequence-file',str(root/'sequence.json'),
            '--m3-strategy',job['strategy'],'--m3-policy',str(root/'frozen/m3_reuse_policy.json')]
        if job['strategy'] == 'wsdr_kv': command.append('--trace-m3-decisions')
        record = dict(job, argv=command, gpu_before=gpu, affinity=sorted(os.sched_getaffinity(0)),
            python=sys.executable, platform=platform.platform(), start_unix=time.time())
        commands.append(record); write(root/'commands.json', commands)
        write(root/'driver_status.json', dict(status='RUNNING', job=job))
        print('START', job['name'], flush=True)
        with (root/'raw_logs'/(job['name']+'.log')).open('x') as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(10)
                progress = dest/'progress.json'
                print('PROGRESS', job['name'], progress.read_text().strip() if progress.exists() else 'loading', flush=True)
        record.update(returncode=process.returncode, end_unix=time.time(), gpu_after=gpu_state())
        write(root/'commands.json', commands)
        if process.returncode: raise RuntimeError('Worker failed; raw logs retained: '+job['name'])
    verify_freeze(root, checkpoint=True)
    write(root/'driver_status.json', dict(status='COMPLETED', jobs=len(plan)))
    from .independent_analysis import analyze
    analyze(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','run','analyze'])
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--checkpoint', default='ckpts/iter_10548.pth')
    parser.add_argument('--scenes', type=int, default=4)
    parser.add_argument('--frames-per-scene', type=int, default=6)
    parser.add_argument('--questions-per-frame', type=int, choices=[2,3], default=3)
    parser.add_argument('--repeats', type=int, choices=[3,4,5], default=3)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if set(os.sched_getaffinity(0)) & {4,5}: parser.error('Use taskset -c 0-3,6-31')
    if min(args.scenes, args.frames_per_scene, args.warmup) < 1: parser.error('Counts must be positive')
    root = Path(args.run_dir)
    try:
        if args.stage == 'prepare': prepare(root, args)
        elif args.stage == 'run': run(root, args)
        else:
            from .independent_analysis import analyze
            analyze(root)
    except BaseException:
        if root.exists():
            with (root/'driver_errors.jsonl').open('a') as f:
                f.write(json.dumps(dict(stage=args.stage, error=traceback.format_exc(), unix_time=time.time()))+'\n')
            write(root/'driver_status.json', dict(status='FAILED', stage=args.stage, error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
