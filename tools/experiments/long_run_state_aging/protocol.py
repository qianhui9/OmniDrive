"""CPU-only protocol, durable records and continuity checks."""
import json
import math
import os
from pathlib import Path

from tools.environment_stability_eval.common import (ROOT, PYTHON, METHODS, read,
    write, jsonl, dump_jsonl, require_affinity)

PROTOCOL_VERSION = 3
STAGES = ('Early', 'Middle', 'Late')


def stage_index(index, length):
    if length < len(STAGES) or not 0 <= index < length:
        raise ValueError('Need at least three chronological frames')
    return min(len(STAGES) - 1, len(STAGES) * index // length)


def validate_workload(questions, groups):
    """Reject missing original QA rather than silently measuring fewer queries."""
    from tools.worldstream_m123.selective_answer_reuse import QUESTIONS
    from tools.worldstream_m123.task_metrics import FACT_GROUPS
    if (len(questions) != 5 or list(questions[:3]) != list(QUESTIONS)
            or list(groups) != list(FACT_GROUPS[:3]) + ['other', 'other']
            or len(set(questions)) != 5
            or any(not isinstance(q, str) or not q.strip() for q in questions)):
        raise ValueError('Every frame requires exactly three fixed facts + two distinct original QA')


def stage_counts(length):
    return {s: sum(stage_index(i, length) == k for i in range(length))
            for k, s in enumerate(STAGES)}


def lifecycle(model):
    if model.m1_manager is None:
        return None
    store = model.m1_manager.store
    m3 = model.m3_runtime
    return dict(epoch=store.generation, commits=store.commit_count,
                frame_id=model.m1_manager._previous_frame_id,
                cache_version=model.m2_runtime.cache.version,
                cache_frame=(model.m2_runtime.cache.context or {}).get('frame_id'),
                candidate_frame_index=m3._frame_index,
                candidate_frame=m3._frame_id,
                candidate_entries=len(m3.answer_memory.records))


def check_continuity(previous, before, after, token):
    # Store itself is replaced by atomic Commit: pointer equality is inappropriate.
    if previous is not None:
        if before != previous:
            raise RuntimeError('WorldStream state changed between measured frames')
        if after['epoch'] != previous['epoch'] or after['commits'] != previous['commits'] + 1:
            raise RuntimeError('OGSM reset or failed commit inside a continuous log')
        if after['cache_version'] != previous['cache_version'] + 1:
            raise RuntimeError('DAVP failed to publish the next cache version')
        if after['candidate_frame_index'] != previous['candidate_frame_index'] + 1:
            raise RuntimeError('QCSE lifecycle reset inside a continuous log')
    if after['frame_id'] != token or after['cache_frame'] != token or after['candidate_frame'] != token:
        raise RuntimeError('State/cache provenance does not match the current frame')


def append_record(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()
    os.fsync(handle.fileno())


def read_journal(path):
    """Recover only newline-terminated commits; ignore a crash-torn last line."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    result = []
    for i, line in enumerate(lines):
        if not line.endswith(b'\n'):
            if i != len(lines) - 1:
                raise ValueError('Malformed journal interior')
            break
        result.append(json.loads(line))
    return result


def validate_rows(manifest, rows, method, repeat, allow_partial=False):
    if manifest.get('protocol_version') != PROTOCOL_VERSION or list(manifest.get('stages', [])) != list(STAGES):
        raise ValueError('Protocol changed: use a new v3 output directory; previous runs are preserved')
    expected = [(s['scene_id'], i, t) for s in manifest['scenes']
                for i, t in enumerate(s['tokens'])]
    actual = [(r['scene_id'], r['frame_index'], r['frame_id']) for r in rows]
    if actual != expected[:len(actual)] or len(actual) > len(expected):
        raise ValueError('Missing, duplicate or reordered chronological frames')
    if not allow_partial and len(actual) != len(expected):
        raise ValueError('Incomplete stream')
    by_scene = {s['scene_id']: s for s in manifest['scenes']}
    previous = None
    last_scene = None
    for row in rows:
        scene = by_scene[row['scene_id']]
        si = stage_index(row['frame_index'], len(scene['tokens']))
        if row['method'] != method or row['repeat'] != repeat or row['stage'] != STAGES[si]:
            raise ValueError('Incorrect method/repeat/stage label')
        if row['timestamp'] != scene['timestamps'][row['frame_index']]:
            raise ValueError('Frame timestamp differs from source metadata')
        if not math.isfinite(row['e2e_ms']) or row['e2e_ms'] <= 0:
            raise ValueError('Invalid E2E duration')
        if not len(row['questions']) == len(row['answers']) == len(row['requests']):
            raise ValueError('Incomplete query receipt')
        validate_workload(row['questions'], row['groups'])
        if method == 'worldstream':
            if last_scene != row['scene_id']:
                previous = None
            check_continuity(previous, row['state_before'], row['state_after'], row['frame_id'])
            previous = row['state_after']
        last_scene = row['scene_id']
    return len(actual) == len(expected)
