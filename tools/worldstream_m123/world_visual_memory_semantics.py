"""Real-feature probes and one-memory causal playback, never LLM quality."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
import torch

from tools.worldstream_m123.benchmark.common import write
from tools.worldstream_m123.world_visual_memory_analysis import distribution
from worldstream.m3.visual_feature_bridge import RegionFeature
from worldstream.m3.visual_memory import VisualMemoryManager, VisualMemoryProjection


def context(row):
    return {**{k:row[k] for k in ('scene_id', 'frame_id', 'timestamp', 'state_epoch', 'state_version')},
        'model_sha256':'6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d',
        'adapter_sha256':'UNTRAINED_COST_ONLY', 'memory_version':'wrvmp_region_v1'}


def regions(values):
    return [RegionFeature(x['anchor_id'], x['semantic_class'], x['camera_id'], x['source_frame'],
        x['source_state_version'], tuple(x['region']), tuple(x['grid_indices']), x['features'],
        x['candidate_cells'], x['excluded_cells']) for x in values if x['qualified']]


def replay(root, tokens, *, capture='bridge_capture'):
    rows = [json.loads(l) for l in (root/capture/'frames.jsonl').open()]
    xs = torch.load(root/capture/'region_features.pt', map_location='cpu', weights_only=False)
    by_frame = defaultdict(list)
    for x in xs:
        by_frame[x['source_frame']].append(x)
    manager = VisualMemoryManager(tokens)
    # Actual tensors held by the manager, not regenerated approximate features.
    projection = VisualMemoryProjection().eval().requires_grad_(False)
    events = []; previous_embedding = None; previous_memory = None
    qualification = json.loads((root/'experiment_plan.json').read_text())['qualification_sha256']
    with torch.no_grad():
        for row in rows:
            values = by_frame[row['frame_id']]
            memory, event = manager.update(context(row), regions(values), row['qualified_ids'], qualification)
            retained = False
            if memory is not None:
                embedding = projection(memory)
                if event == 'keep':
                    assert memory is previous_memory and embedding is previous_embedding
                    retained = torch.equal(embedding, previous_embedding)
                previous_memory, previous_embedding = memory, embedding
            else:
                previous_memory = previous_embedding = None
            events.append(dict(scene_id=row['scene_id'], frame_id=row['frame_id'], split=row['split'],
                event=event, memory_valid=memory is not None,
                anchor_id=memory.anchor_id if memory else None, revision=memory.revision if memory else None,
                generating_frame=memory.generating_frame if memory else None,
                source_camera=memory.camera_id if memory else None,
                source_state_version=memory.source_state_version if memory else None,
                latest_frame=memory.latest_frame if memory else None,
                age=memory.age if memory else None, content_sha256=memory.content_sha256 if memory else None,
                scope=list(memory.scope) if memory else None,
                qualification_sha256=qualification, exact_same_embedding_tensor=retained,
                raw_observations=tokens if memory else 0,
                historical_context_only=True, current_visual_tokens=513, actual_skipped_KV_tokens=0))
    return events


def run(root):
    assert json.loads((root/'bridge_analysis.json').read_text())['status'] == 'PASS'
    # Freeze the semantic check before computing any probe outcome.
    protocol = dict(category_probe='Untrained cosine nearest centroid, raw mean regional features. TRAIN centroids -> log-disjoint development. Only independently GT-matched detections; conditional on detector-correct class.',
        retrieval='Adjacent frame object retrieval among qualified GT-known cars, cosine; score GT equality offline only. Do not use GT for memory.',
        go='Real feature hashes differ; actual historical embedding is held exactly; category macro accuracy and identity retrieval each exceed their uniform candidate chance. These are pretraining content checks, not task quality or appearance invariance.',
        dynamic_leakage='Historical EVA regions contain old appearance, background and absolute image position. Never replace current full513 or claim current state equals history.')
    write(root/'semantic_probe_protocol.json', protocol)
    xs = torch.load(root/'bridge_capture/region_features.pt', map_location='cpu', weights_only=False)
    xs = [x for x in xs if x['offline_gt_identity'] is not None]
    train = [x for x in xs if x['split'] == 'train']
    val = [x for x in xs if x['split'] == 'validation']
    classes = sorted({x['semantic_class'] for x in train} & {x['semantic_class'] for x in val})
    normalize = lambda a: torch.nn.functional.normalize(a.float(), dim=-1)
    centers = torch.stack([normalize(torch.stack([normalize(x['vector']) for x in train
        if x['semantic_class'] == c]).mean(0)) for c in classes])
    category = []
    for x in val:
        if x['semantic_class'] in classes:
            pred = classes[int((normalize(x['vector']) @ centers.T).argmax())]
            category.append(dict(frame_id=x['source_frame'], truth=x['semantic_class'], predicted=pred))
    recalls = {c:float(np.mean([r['truth'] == r['predicted'] for r in category if r['truth'] == c])) for c in classes}
    macro = float(np.mean(list(recalls.values())))
    by_frame = defaultdict(list)
    for x in xs:
        if x['qualified']:
            by_frame[(x['scene_id'], x['source_frame'])].append(x)
    timeline = [json.loads(l) for l in (root/'bridge_capture/frames.jsonl').open()]
    previous = []; previous_scene = None; retrieval = []
    for row in timeline:
        current = by_frame[(row['scene_id'], row['frame_id'])]
        if row['scene_id'] == previous_scene and len(previous) > 1:
            source = torch.stack([normalize(x['vector']) for x in previous])
            for x in current:
                if not any(x['offline_gt_identity'] == y['offline_gt_identity'] for y in previous):
                    continue
                j = int((normalize(x['vector']) @ source.T).argmax())
                retrieval.append(dict(scene_id=row['scene_id'], frame_id=row['frame_id'],
                    correct=x['offline_gt_identity'] == previous[j]['offline_gt_identity'],
                    chance=1/len(previous), candidates=len(previous)))
        previous, previous_scene = current, row['scene_id']
    accuracy = float(np.mean([x['correct'] for x in retrieval])) if retrieval else None
    chance = float(np.mean([x['chance'] for x in retrieval])) if retrieval else None
    events = {str(k):replay(root, k) for k in (64,128,256)}
    exact = sum(e['exact_same_embedding_tensor'] for e in events['64'])
    unique = len({x['feature_sha256'] for x in xs if x['qualified']})
    checks = dict(distinct_real_features=unique > 1, actual_tensor_retention=exact > 0,
        category_information=macro > 1/len(classes),
        identity_information=accuracy is not None and accuracy > chance)
    write(root/'memory_lifecycle.json', dict(status='REAL_FEATURE_RESEARCH_PLAYBACK',
        main_semantic_tokens=64, finite_K_availability_only=True, events=events,
        scope='Positive-enriched bridge scenes. Finite K playback is not deployment coverage, LLM quality or actual KV reuse. One state per budget, no multi-level cache.',
        actual_service_KV_tokens=0))
    report = dict(status='PASS' if all(checks.values()) else 'NO_GO', checks=checks,
        classes=classes, category_examples=len(category), category_macro_accuracy=macro,
        category_uniform_chance=1/len(classes), category_recalls=recalls,
        category_predictions=category, retrieval_examples=len(retrieval), retrieval_accuracy=accuracy,
        retrieval_chance=chance, retrieval=retrieval, exact_held_embeddings=exact,
        finite_K_playback={k:dict(events=dict(Counter(e['event'] for e in es)),
            valid_frames=sum(e['memory_valid'] for e in es), extensions=sum(e['event']=='keep' for e in es)) for k,es in events.items()},
        current_state_invariance_certified=False, LLM_quality='NOT_RUN', trained_projection=False,
        semantics_scope='Raw real regional feature information and historical context retention only. Projection parameters are untrained cost scaffolding; no language/appearance quality attributed to them.')
    write(root/'memory_semantic_probe.json', report)
    print(json.dumps({k:report[k] for k in ('status','checks','category_macro_accuracy','retrieval_accuracy','retrieval_chance','finite_K_playback')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/m3_world_visual_memory_run01'))
    run(p.parse_args().root)
