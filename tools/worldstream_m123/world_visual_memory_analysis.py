"""Deterministic pretraining WRVMP evidence; no synthetic language quality."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import torch

from tools.worldstream_m123.benchmark.common import write, sha256


def distribution(values):
    x = np.asarray(values, float)
    if not len(x):
        return dict(n=0, mean=None, median=None, min=None, max=None, p95=None)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite audit measurement')
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
        min=float(x.min()), max=float(x.max()), p95=float(np.quantile(x, .95)))


def analyze(root):
    plan = json.loads((root/'experiment_plan.json').read_text())
    dest = root/'bridge_capture'
    status = json.loads((dest/'status.json').read_text())
    assert status['status'] == 'COMPLETED'
    binding = json.loads((dest/'capture_hashes.json').read_text())
    assert sha256(dest/'frames.jsonl') == binding['frames_sha256']
    assert sha256(dest/'region_features.pt') == binding['region_features_sha256']
    frames = [json.loads(l) for l in (dest/'frames.jsonl').open()]
    features = torch.load(dest/'region_features.pt', map_location='cpu', weights_only=False)
    qualified = [x for x in features if x['qualified']]
    gate = plan['bridge']['gate']
    known = [x for x in qualified if x['offline_projected_gt_cell_support'] is not None]
    supported = [x for x in known if x['offline_projected_gt_cell_support'] >= gate['min_projected_box_cell_support']]
    unknown_fraction = (len(qualified)-len(known))/len(qualified) if qualified else None
    supported_fraction = len(supported)/len(known) if known else None
    previous = {}; edges = []
    for frame in frames:
        now = {}
        for x in frame['regions']:
            if not x['qualified']:
                continue
            key = (frame['scene_id'], x['anchor_id'])
            if key in previous:
                a, b = previous[key]['offline_gt_identity'], x['offline_gt_identity']
                edges.append(dict(scene_id=frame['scene_id'], anchor_id=x['anchor_id'],
                    frame_id=frame['frame_id'], status='unknown' if a is None or b is None
                    else 'correct' if a == b else 'wrong'))
            now[key] = x
        previous = now
    wrong = sum(e['status'] == 'wrong' for e in edges)
    checks = dict(scored=len(known) >= gate['min_scored'],
        scenes=len({x['scene_id'] for x in known}) >= gate['min_scenes'],
        unknown=unknown_fraction is not None and unknown_fraction <= gate['max_unknown_fraction'],
        geometric_region_support=supported_fraction is not None and supported_fraction >= gate['min_supported_fraction'],
        identity_continuations=wrong <= gate['max_wrong_continuations'])
    by_scene = {}
    for scene in sorted({x['scene_id'] for x in qualified}):
        xs = [x for x in qualified if x['scene_id'] == scene]
        ks = [x['offline_projected_gt_cell_support'] for x in xs if x['offline_projected_gt_cell_support'] is not None]
        by_scene[scene] = dict(regions=len(xs), known=len(ks),
            support_distribution=distribution(ks), supported=sum(x >= .8 for x in ks))
    costs = {name: distribution([r[name] for r in frames]) for name in
        ('m1_read_qualification_ms', 'extraction_wall_ms', 'bridge_total_wall_ms', 'bridge_stream_elapsed_ms')}
    costs['core_present_wall_ms'] = distribution([r['bridge_total_wall_ms'] for r in frames if r['qualified_ids']])
    costs['no_core_wall_ms'] = distribution([r['bridge_total_wall_ms'] for r in frames if not r['qualified_ids']])
    evidence = dict(status='PASS' if all(checks.values()) else 'NO_GO', checks=checks,
        thresholds=gate, frames=len(frames), scenes=len(by_scene),
        qualified_objects=sum(len(f['qualified_ids']) for f in frames),
        qualified_regions=len(qualified), scored=len(known), unknown_fraction=unknown_fraction,
        supported=len(supported), supported_fraction=supported_fraction,
        qualified_rejections=dict(Counter(reason for r in frames for reason in r['bridge_rejections'].values())),
        identity_edges=dict(Counter(e['status'] for e in edges)), by_scene=by_scene,
        grid_cells=distribution([len(x['grid_indices']) for x in qualified]),
        K_native_cell_availability={str(k):sum(len(x['grid_indices']) >= k for x in qualified) for k in (64, 128, 256)},
        feature_shapes=sorted({tuple(f['feature_shape']) for f in frames}), costs=costs,
        gpu_kernel_cost_ms=None, gpu_cost_note='Stream event intervals include CPU launch/projection gaps, not summed kernel time.',
        deployment_coverage=None, selection_bias=plan['selection'],
        foreground_or_appearance_certified=False, online_GT=False,
        failures=[dict(scene_id=x['scene_id'], frame_id=x['source_frame'], anchor_id=x['anchor_id'],
            support=x['offline_projected_gt_cell_support'], camera_id=x['camera_id'], cells=x['cells'])
            for x in known if x['offline_projected_gt_cell_support'] < .8])
    write(root/'bridge_analysis.json', evidence)
    write(root/'bridge_identity_edges.json', edges)
    # Feature distinctness is a provenance diagnostic, not a semantic accuracy
    # gate or proof that the representation is stable in the current scene.
    vectors = torch.stack([x['vector'] for x in qualified]) if qualified else torch.empty(0, 0)
    unique = len({x['feature_sha256'] for x in qualified})
    matched = []; different = []; same_camera = []
    for i, x in enumerate(qualified):
        same_camera.append(float(torch.nn.functional.cosine_similarity(x['vector'][None], x['same_camera_global_vector'][None])))
        for j in range(i):
            y = qualified[j]
            if x['offline_gt_identity'] is None or y['offline_gt_identity'] is None:
                continue
            cosine = float(torch.nn.functional.cosine_similarity(vectors[i:i+1], vectors[j:j+1]))
            if (x['scene_id'] == y['scene_id'] and x['anchor_id'] == y['anchor_id']
                    and 0 < x['timestamp']-y['timestamp'] <= 1.):
                matched.append(cosine)
            elif x['source_frame'] == y['source_frame'] and x['offline_gt_identity'] != y['offline_gt_identity']:
                different.append(cosine)
    write(root/'feature_content_diagnostics.json', dict(regions=len(qualified),
        unique_feature_hashes=unique, real_features_differ=unique > 1,
        same_identity_within_1s_cosine=distribution(matched),
        different_identity_same_frame_cosine=distribution(different),
        region_vs_same_camera_global_cosine=distribution(same_camera),
        learned_class_or_appearance_probe='NOT_RUN', memory_embedding_retention='NOT_RUN',
        dynamic_state_leakage='Contextual grid includes position/background/current appearance; not certified invariant.',
        language_quality='NOT_RUN', semantic_gate='NOT_RUN' if not all(checks.values()) else 'PENDING'))
    print(json.dumps({k:evidence[k] for k in ('status', 'checks', 'qualified_regions', 'scored', 'supported_fraction', 'identity_edges')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/m3_world_visual_memory_run01'))
    a = p.parse_args()
    analyze(a.root)
