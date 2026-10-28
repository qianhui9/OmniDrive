"""Bounded real WRVMP bridge audit. No language generation or new online GT."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import perf_counter

from tools.worldstream_m123.benchmark.common import sha256, write

ROOT = Path('outputs/m3_world_visual_memory_run01')
PRIOR = Path('outputs/m3_sf_wrcps_feasibility_run01')


def prepare(root):
    root.mkdir(exist_ok=False, parents=True)
    sequence = json.loads((PRIOR/'audit_sequence.json').read_text())
    counts = defaultdict(Counter)
    for line in (PRIOR/'state_audit/frames.jsonl').open():
        row = json.loads(line)
        counts[row['scene_id']]['hits'] += row['anchor_hit']
    train = sorted(s for s, role in sequence['scene_splits'].items()
                   if role == 'train' and counts[s]['hits'] >= 6)[:3]
    dev = sorted(s for s, role in sequence['scene_splits'].items()
                 if role == 'validation' and counts[s]['hits'])
    scenes = train + dev
    assert len(train) == 3 and len(dev) == 2
    # Positive selection is for bridge diagnosis, never deployment coverage.
    import pickle
    from mmcv import Config
    cfg = Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    infos = pickle.load(open(cfg.data.train.ann_file, 'rb'))['infos']
    token_scene = {r['token']: r['scene_token'] for r in infos}
    seq = dict(tokens=[t for t in sequence['tokens'] if token_scene[t] in scenes],
        scene_splits={s: sequence['scene_splits'][s] for s in scenes},
        scene_logs={s: sequence['scene_logs'][s] for s in scenes})
    write(root/'bridge_sequence.json', seq)
    write(root/'experiment_plan.json', dict(base_commit='25d782296ff119fad945c1f9a029689a0ded3389',
        seed=20261028, stages=['bridge', 'semantic', 'cost', 'conditional_training'],
        selection='Three sorted TRAIN scenes with >=6 previous extensions; two positive development scenes, all causal first32. Biased for availability, not coverage.',
        frames=len(seq['tokens']), sequence_sha256=sha256(root/'bridge_sequence.json'),
        source_manifest=str(PRIOR/'training_dataset_manifest.json'),
        source_manifest_sha256=sha256(PRIOR/'training_dataset_manifest.json'),
        qualification_path='outputs/m3_sf_wrcps_optimization_run01/semantic_qualification.json',
        qualification_sha256=sha256('outputs/m3_sf_wrcps_optimization_run01/semantic_qualification.json'),
        bridge=dict(patch_stride=16, minimum_cells=4, maximum_diagnostic_objects=8,
            exclusions='Other currently observed object projection AABBs, all depths',
            gate=dict(min_scored=30, min_scenes=3, max_unknown_fraction=.05,
                min_projected_box_cell_support=.8, min_supported_fraction=.95,
                max_wrong_continuations=0),
            gate_scope='Conservative preregistered geometric region suitability; GT box support is not foreground purity. No tuning on audit results.'),
        language_decode_requests=0, trained_adapter=None,
        training_guard='Do not train unless real bridge, semantic content and net cost all pass.'))


def audit(root):
    import random
    import hashlib
    import numpy as np
    import torch
    import projects.mmdet3d_plugin  # noqa: F401
    from mmcv.parallel import collate, scatter
    from mmdet3d.datasets import build_dataset
    from tools.worldstream_m123.benchmark.worker import effective_config, load_model, clear_questions, reset
    from tools.worldstream_m123.benchmark.observe import Observer
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    from tools.worldstream_m1.dataset import load_oracle
    from tools.worldstream_m1.identity_metrics import strict_labels
    from worldstream.m1.geometry import project_points
    from worldstream.m3.semantic_core import SemanticQualifier, committed_semantic_candidates
    from worldstream.m3.visual_feature_bridge import projected_region_features, inside_convex
    plan = json.loads((root/'experiment_plan.json').read_text())
    sequence = json.loads((root/'bridge_sequence.json').read_text())
    assert sha256(root/'bridge_sequence.json') == plan['sequence_sha256']
    assert sha256(plan['qualification_path']) == plan['qualification_sha256']
    dest = root/'bridge_capture'
    dest.mkdir(exist_ok=False)
    write(dest/'status.json', dict(status='RUNNING'))
    features_saved, timeline = [], []
    with gpu_job_lock():
        wait_gpu()
        cfg = effective_config('B3', 'language', dest, 'native_full')
        cfg.data.test.ann_file = cfg.data.train.ann_file
        for node in cfg.data.test.pipeline:
            if node['type'] == 'LoadAnnoatationVQATest':
                node['base_conv_path'] = './data/nuscenes/conv/train/'
                node['base_vqa_path'] = './data/nuscenes/vqa/train/'
        write(dest/'effective_config.json', dict(config=cfg._cfg_dict.to_dict(),
            scope='Read-only feature bridge, zero LLM questions, native M1/M2; not service latency'))
        dataset = build_dataset(cfg.data.test)
        lookup = {r['token']: r for r in dataset.data_infos}
        infos = [lookup[t] for t in sequence['tokens']]
        dataset.data_infos = infos
        model = load_model(cfg, 'ckpts/iter_10548.pth', dest).eval().requires_grad_(False)
        observer = Observer(model)
        observer.capture_m1_controls = True
        qualifier = SemanticQualifier(plan['qualification_path'])
        annotations = load_oracle(infos, 'data/nuscenes')
        previous_scene = None
        with (dest/'frames.jsonl').open('w') as log:
            for ordinal, info in enumerate(infos):
                seed = int(hashlib.sha256(info['token'].encode()).hexdigest()[:8], 16) % (2**31)
                random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                observer.clear()
                if previous_scene != info['scene_token']:
                    reset(model)
                data = scatter(collate([dataset[ordinal]], samples_per_gpu=1), [0])[0]
                clear_questions(data)
                with torch.no_grad():
                    model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
                assert len(observer.commits) == 1 and not observer.native_generations
                frame = observer.commits[0][0]
                manager = model.m1_manager
                outputs = observer.m1_control['outputs']
                feature = next(f.tensor for f in outputs.features if f.feature_kind == 'backbone')
                context = dict(scene_id=frame.scene_id, frame_id=frame.frame_id,
                    timestamp=frame.timestamp, state_epoch=manager.store.generation,
                    state_version=manager.store.commit_count,
                    state_execution_signature=manager.store.execution_signature)
                # Timed online path reads only existing state and current features.
                torch.cuda.synchronize()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                tick = perf_counter(); start.record()
                candidates = committed_semantic_candidates(manager, context)
                selection = qualifier.select(candidates)
                qualified = [r[0] for r in selection['facts']]
                anchors = [manager.store.anchors[a] for a in manager.last_commit_diagnostics['committed_anchor_ids']]
                read_ms = (perf_counter()-tick)*1000
                extraction_start = perf_counter()
                regions, rejected = projected_region_features(frame, anchors, feature,
                    manager.config, camera_ids=list(frame.cameras), selected_ids=qualified)
                end.record(); end.synchronize()
                wall_ms = (perf_counter()-tick)*1000
                extraction_wall_ms = (perf_counter()-extraction_start)*1000
                cuda_ms = start.elapsed_time(end)
                # Extra objects for semantic controls are OUTSIDE the online cost.
                diagnostic_ids = [a.anchor_id for a in sorted(anchors,
                    key=lambda a: (-(a.source_score or 0), a.anchor_id))
                    if a.anchor_type == 'object' and a.anchor_id not in qualified][:8]
                controls, _ = projected_region_features(frame, anchors, feature,
                    manager.config, camera_ids=list(frame.cameras), selected_ids=diagnostic_ids)
                # GT begins only here; it cannot alter extraction or eligibility.
                labels = strict_labels(anchors, annotations.get(frame.frame_id, []), gate=2.)
                identities = {a.anchor_id: labels.get(i) for i, a in enumerate(anchors)}
                gt_by_id = {g['instance_id']: g for g in annotations.get(frame.frame_id, [])}
                records = []
                for region in regions + controls:
                    identity = identities.get(region.anchor_id)
                    camera = frame.cameras[region.camera_id]
                    support = None
                    if identity is not None:
                        uv, _, good = project_points(gt_by_id[identity]['geometry'].points,
                            camera.intrinsic, camera.world_to_camera)
                        if good.all():
                            idx = np.asarray(region.grid_indices)
                            points = np.stack(((idx % feature.shape[-1]+.5)*16,
                                               (idx // feature.shape[-1]+.5)*16), -1)
                            support = float(inside_convex(points, uv).mean())
                    values = region.features.detach().cpu()
                    vector = values.float().mean(0)
                    # Different image region from the SAME contextual camera grid.
                    grid = feature[0, list(frame.cameras).index(region.camera_id)].flatten(1).T
                    background = grid.float().mean(0).detach().cpu()
                    record = dict(anchor_id=region.anchor_id, semantic_class=region.semantic_class,
                        camera_id=region.camera_id, source_frame=region.frame_id,
                        source_state_version=region.source_state_version, region=region.region,
                        grid_indices=region.grid_indices, cells=len(region.grid_indices),
                        excluded_cells=region.excluded_cells, candidate_cells=region.candidate_cells,
                        qualified=region.anchor_id in qualified, offline_gt_identity=identity,
                        offline_projected_gt_cell_support=support,
                        feature_sha256=hashlib.sha256(values.numpy().tobytes()).hexdigest(),
                        source_kind=region.source_kind, foreground_certified=False)
                    records.append(record)
                    features_saved.append(dict(**record, scene_id=frame.scene_id,
                        split=sequence['scene_splits'][frame.scene_id], timestamp=frame.timestamp,
                        features=values, vector=vector, same_camera_global_vector=background))
                row = dict(ordinal=ordinal, **context, split=sequence['scene_splits'][frame.scene_id],
                    log_id=sequence['scene_logs'][frame.scene_id], qualified_ids=qualified,
                    regions=records, bridge_rejections=rejected, feature_shape=list(feature.shape),
                    feature_dtype=str(feature.dtype), camera_order=list(frame.cameras),
                    image_shapes={cid:list(c.image.shape) for cid,c in frame.cameras.items()},
                    m1_read_qualification_ms=read_ms, extraction_wall_ms=extraction_wall_ms,
                    bridge_total_wall_ms=wall_ms, bridge_stream_elapsed_ms=cuda_ms,
                    stream_elapsed_scope='CUDA events encompass host projection gaps; not GPU kernel time',
                    memory_or_LLM_tokens_constructed=False, LLM_decode_requests=0, GT_online=False)
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
                timeline.append(row)
                previous_scene = frame.scene_id
                if (ordinal+1) % 8 == 0:
                    print('VISUAL_BRIDGE', ordinal+1, '/', len(infos), flush=True)
        torch.save(features_saved, dest/'region_features.pt')
        write(dest/'capture_hashes.json', dict(region_features_sha256=sha256(dest/'region_features.pt'),
            frames_sha256=sha256(dest/'frames.jsonl'),
            sources={str(p):sha256(p) for p in [Path(__file__), Path('worldstream/m3/visual_feature_bridge.py')]}))
        write(dest/'status.json', dict(status='COMPLETED', frames=len(timeline),
            features=len(features_saved), LLM_decode_requests=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--stage', choices=['prepare', 'audit'], required=True)
    a = p.parse_args()
    if {4, 5} & set(os.sched_getaffinity(0)):
        p.error('Use taskset -c 0-3,6-31')
    {'prepare': prepare, 'audit': audit}[a.stage](a.root)


if __name__ == '__main__':
    main()
