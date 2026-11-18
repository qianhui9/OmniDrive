"""Reuse main-experiment sequence, six-camera input and authoritative QA GT."""
from datetime import datetime
from pathlib import Path
import os
import subprocess

from .protocol import (ROOT, STAGES, PROTOCOL_VERSION, dump_jsonl, stage_counts,
                       write, validate_workload)


def prepare(out, config, smoke, config_path):
    from tools.environment_stability_eval.data_adapter import (
        load_infos, validate_frames, load_objects, build_query_records, coverage_report)
    from tools.worldstream_m123.r0_task_evaluation import selected_scenes, sha, CHECKPOINT_SHA, facts_questions
    infos, by = load_infos(config)
    pool = selected_scenes(infos, config['sequence_pool'],
                          (ROOT / config['main_manifest']).parent, None)
    duration = lambda s: (by[s['tokens'][-1]]['timestamp'] - by[s['tokens'][0]]['timestamp']) / 1e6
    ranked = sorted(pool, key=lambda s: (-duration(s), -len(s['tokens']), s['scene_id']))
    if config['scene_ids']:
        wanted = set(config['scene_ids'])
        if len(wanted) != len(config['scene_ids']) or wanted - {s['scene_id'] for s in pool}:
            raise ValueError('Duplicate or unavailable explicit scene IDs')
        selected = [s for s in ranked if s['scene_id'] in wanted]
    else:
        selected = ranked[:config['sequence_count']]
    validate_frames(selected, by)
    # Verify complete log boundaries against authoritative sample links as well.
    from tools.worldstream_m1.io_utils import iter_json_array
    table = ROOT / config['data_root'] / 'v1.0-trainval'
    scene_table = {s['token']: s for s in iter_json_array(table / 'scene.json')}
    for scene in selected:
        gt = scene_table[scene['scene_id']]
        if (gt['nbr_samples'] != len(scene['tokens']) or gt['first_sample_token'] != scene['tokens'][0]
                or gt['last_sample_token'] != scene['tokens'][-1]):
            raise ValueError('Sequence is not a complete authoritative scene')
    originals = [dict(s, name=scene_table[s['scene_id']]['name'], duration_seconds=duration(s),
                      stage_counts=stage_counts(len(s['tokens']))) for s in selected]
    measured = [dict(s, tokens=s['tokens'][:config['smoke_frames']] if smoke else s['tokens']) for s in originals]
    for scene in measured:
        scene['timestamps'] = [by[t]['timestamp'] / 1e6 for t in scene['tokens']]
        scene['stage_counts'] = stage_counts(len(scene['tokens']))
        scene['measured_duration_seconds'] = scene['timestamps'][-1] - scene['timestamps'][0]
    tokens = {t for s in measured for t in s['tokens']}
    # Validate online workload before preparing GT or loading any GPU model.
    # facts_questions reuses the original conv/vqa order without generating QA.
    for token in tokens:
        validate_workload(*facts_questions(token, ROOT / config['data_root'], 2))
    print('LOAD_AUTHORITATIVE_GT', len(tokens), flush=True)
    objects = load_objects(config, tokens)
    queries = build_query_records(config, measured, by, objects)
    dump_jsonl(out / 'query_ground_truth.jsonl', queries)
    dump_jsonl(out / 'evaluation_objects.jsonl', [dict(frame_id=t, objects=objects[t]) for t in sorted(tokens)])
    coverage = coverage_report(queries, objects)
    write(out / 'gt_coverage.json', coverage)
    # Record unavailable commit honestly for distributed source archives.
    git = subprocess.run(['taskset', '-c', '0-3,6-31', 'git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True)
    sources = {p for folder in ('worldstream', 'projects', 'tools/worldstream_m123',
                               'tools/environment_stability_eval', 'tools/worldstream_ablation',
                               'tools/experiments/long_run_state_aging', 'tools/worldstream_m1')
               for p in (ROOT / folder).rglob('*.py') if 'outputs' not in p.parts}
    sources.add(config_path.resolve())
    sources.add(ROOT / config['r0_config'])
    sources.add(ROOT / config['main_manifest'])
    sources.add(ROOT / 'outputs/m123_final_quality_validation_run01/reference_policy.json')
    sources.add(ROOT / config['infos'])
    sources.update(table / p for p in ('sample.json', 'sample_annotation.json', 'category.json',
                                     'instance.json', 'attribute.json', 'scene.json'))
    inputs = {ROOT / c['data_path'] for t in tokens for c in by[t]['cams'].values()}
    for t in tokens:
        for folder in ('conv', 'vqa'):
            p = ROOT / config['data_root'] / folder / 'val' / (t + '.json')
            if not p.exists():
                raise FileNotFoundError(p)
            inputs.add(p)
    print('HASH_FROZEN_SOURCES_WEIGHTS_AND_INPUTS', flush=True)
    checkpoint_sha = sha(ROOT / config['checkpoint'])
    if checkpoint_sha != CHECKPOINT_SHA:
        raise ValueError('Checkpoint differs from frozen main experiment')
    manifest = dict(protocol_version=PROTOCOL_VERSION,
        created_at=datetime.now().astimezone().isoformat(), config=config, smoke=smoke,
        scenes=measured, full_selected_scenes=originals, stages=STAGES,
        candidate_sequences=[dict(scene_id=s['scene_id'], frames=len(s['tokens']), duration_seconds=duration(s)) for s in ranked],
        selection='Longest elapsed video duration in configured pool; no concatenation or repetition of frames.',
        query_workload='Exactly five per frame: original three fixed facts then two original QA; fail if unavailable. All five executed in one timed frame call.',
        quality_definition='Main factual score uses all five known queries under the existing main evaluator. Fixed facts and original-QA limited factual scores also reported separately; no validated holistic generation-quality scorer.',
        peak_memory_definition='Per-frame CUDA allocator peak allocated/reserved reset outside E2E; stage maximum includes resident model and transient inference memory. Host RSS sampled at frame endpoints; process lifetime high-water separately labeled. Cache peak is max observed retained payload after each frame.',
        git_commit=git.stdout.strip() if git.returncode == 0 else None,
        git_unavailable_reason=git.stderr.strip() if git.returncode else None,
        cpu_affinity=sorted(os.sched_getaffinity(0)), checkpoint_sha256=checkpoint_sha,
        source_and_GT_hashes={str(p): sha(p) for p in sorted(sources)},
        input_hashes={str(p): sha(p) for p in sorted(inputs)},
        artifact_hashes={name: sha(out / name) for name in
                         ('query_ground_truth.jsonl', 'evaluation_objects.jsonl', 'gt_coverage.json')},
        timing_definition='CUDA-synchronized wall time: image read/preprocessing, H2D, prompt encoding, full model, final answer export. Excludes loading, offline scoring and evaluation instrumentation.',
        fps_definition='Frames / sum(frame E2E seconds), same as main_protocol_v1. Processing wall throughput is an auxiliary metric.',
        initialization='Cold first frame included; no measured-prefix warmup or stage reset. Each method/repeat is a fresh process; reset only at scene boundary. Native perception temporal memories reset before each frame.',
        response_delay_available=False,
        response_delay_reason='GT labels sampled ~2 Hz; no exact physical change timestamps. Change Response Delay and stale duration are N/A; no change_response.csv.',
        stale_definition='Same-question previous comparable known GT changes; stale iff fully matches previous GT and fails current GT. Sampled queried transitions only; denominator and UNKNOWN reported.',
        horizon_limitation='nuScenes scenes are approximately 20 seconds. Independent repeats cannot establish hour-scale stability.')
    write(out / 'manifest.json', manifest)
    return manifest


def verify_manifest(out):
    from .protocol import read
    from tools.worldstream_m123.r0_task_evaluation import sha
    manifest = read(out / 'manifest.json')
    if manifest.get('protocol_version') != PROTOCOL_VERSION or list(manifest.get('stages', [])) != list(STAGES):
        raise ValueError('New three-stage protocol requires a new output directory; legacy runs are unchanged')
    for group in ('source_and_GT_hashes', 'input_hashes'):
        for path, expected in manifest[group].items():
            if sha(path) != expected:
                raise ValueError('Frozen source/input changed: ' + path)
    for name, expected in manifest['artifact_hashes'].items():
        if sha(out / name) != expected:
            raise ValueError('Frozen GT artifact changed: ' + name)
    return manifest
