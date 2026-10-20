"""Deterministic scene-local keyframe selection and honest micro/macro statistics."""
from collections import Counter
import numpy as np


def ratio(n, d):
    return n / d if d else None


def distribution(values):
    xs=[float(x) for x in values if x is not None and np.isfinite(x)]
    if not xs:return dict(n=0,mean=None,median=None,p05=None,p95=None,min=None,max=None,std=None)
    return dict(n=len(xs),mean=float(np.mean(xs)),median=float(np.median(xs)),
                p05=float(np.percentile(xs,5)),p95=float(np.percentile(xs,95)),
                min=min(xs),max=max(xs),std=float(np.std(xs)))


def select_scenes(infos, available_tokens=None, scene_limit=0, frame_limit=0, context=None):
    """Use contiguous prefixes of available keyframes, never hide missing predictions.

    Limits of zero mean all available targets. Frame metadata contains no labels.
    Scene sorting is frozen and explicit; no selection depends on observed quality.
    """
    if scene_limit<0 or frame_limit<0:raise ValueError('Limits must be nonnegative')
    grouped={};seen=set()
    for info in infos:
        token=info['token']
        if token in seen:raise ValueError('Repeated sample token in dataset')
        seen.add(token);grouped.setdefault(info['scene_token'],[]).append(info)
    sequences={};specs=[];eligible=[]
    for scene,rows in sorted(grouped.items()):
        rows=sorted(rows,key=lambda x:(x['timestamp'],x['token']))
        times=[i['timestamp'] for i in rows]
        if len(set(times))!=len(times):raise ValueError('Repeated timestamp inside scene')
        available=[i for i in rows if available_tokens is None or i['token'] in available_tokens]
        if available:eligible.append(scene)
        sequences[scene]=available[:frame_limit] if frame_limit else available
        indices={r['token']:n for n,r in enumerate(rows)}
        specs.append(dict(scene_id=scene,log_token=context.log_for_scene(scene) if context else None,
            dataset_frames=len(rows),available_frames=len(available),
            missing_prediction_frames=[i['token'] for i in rows if available_tokens is not None and i['token'] not in available_tokens],
            omitted_by_frame_limit=[i['token'] for i in available[len(sequences[scene]):]],
            selected_frames=[dict(frame_id=i['token'],timestamp=i['timestamp']/1e6,
                dataset_sequence_index=indices[i['token']]) for i in sequences[scene]]))
    selected=eligible[:scene_limit] if scene_limit else eligible
    for s in specs:
        s['scheduled']=s['scene_id'] in selected
        s['status']='Scheduled' if s['scheduled'] else 'Not Run'
        s['reason']=None if s['scheduled'] else ('scene_limit' if s['available_frames'] else 'no_actual_predictions')
    return {s:sequences[s] for s in selected},dict(
        sampling_rule='Lexicographic scene_token, timestamp-sorted available keyframe prefix; zero limit means all',
        sequence_type='nuScenes keyframes, not high-frequency video',
        dataset_scene_count=len(grouped),available_target_scene_count=len(eligible),scheduled_scene_count=len(selected),
        scene_limit=scene_limit,frame_limit=frame_limit,scenes=specs,
        unexecuted_scene_ids=[s['scene_id'] for s in specs if not s['scheduled']])


RATIOS={'observation_anchor_coverage':('observation_supported_anchor_events','historical_projection_candidates'),
        'observed_pixel_coverage':('observation_supported_pixels','camera_pixels'),
        'certification_coverage':('certified_anchor_events','actual_current_observed_anchor_events'),
        'identity_candidate_coverage':('identity_candidate_proposals','identity_proposals'),
        'identity_unknown_fraction':('identity_unknown_proposals','identity_proposals')}


def aggregate(scenes):
    counts=Counter()
    for s in scenes:
        counts.update({k:v for k,v in s['counts'].items() if k!='unique_scenes'})
    counts['unique_scenes']=len(scenes)
    micro={name:ratio(counts[a],counts[b]) for name,(a,b) in RATIOS.items()}
    macro={name:distribution([s.get(name) for s in scenes]) for name in RATIOS}
    flags=[]
    for s in scenes:
        reasons=[]
        if s['counts'].get('historical_projection_candidates',0) and s['observation_anchor_coverage']==0:
            reasons.append('zero_anchor_observation_coverage')
        if s['counts'].get('identity_proposals',0) and s['identity_unknown_fraction']==1:
            reasons.append('all_identity_proposals_unknown')
        rejection=s.get('identity_rejection_reasons',{})
        if rejection and max(rejection.values())/sum(rejection.values())>=.9:
            reasons.append('identity_rejections_concentrated_at_least_90_percent')
        if s.get('transaction_failures',0):reasons.append('transaction_failures')
        flags.append(dict(scene_id=s['scene_id'],flags=reasons))
    from .scene_metrics import latency_stats
    timing_keys=sorted({k for s in scenes for k in s.get('latency_samples',{})})
    latency_micro={k:latency_stats([v for s in scenes for v in s.get('latency_samples',{}).get(k,[])]) for k in timing_keys}
    latency_macro={k:distribution([s.get('latency',{}).get(k,{}).get('p50') for s in scenes]) for k in timing_keys}
    return dict(micro=dict(counts=dict(counts),**micro),macro=macro,latency_micro=latency_micro,latency_macro=latency_macro,
        scene_flags=flags,scenes=scenes,independent_scene_statistics_are_not_independent_log_qualification=True)
