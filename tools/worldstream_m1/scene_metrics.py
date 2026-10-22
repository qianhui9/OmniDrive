"""Scene-local metrics. Null denominators stay unknown; pooled ratios use counts."""
from collections import Counter
import numpy as np
from .scene_protocol import RATIOS,ratio,distribution


def latency_stats(values):
    values=[float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(samples=len(values),p50=float(np.percentile(values,50)) if values else None,
                p95=float(np.percentile(values,95)) if values else None,scope='milliseconds')


def summarize_scene(rows,base=None):
    if not rows:raise ValueError('No scene rows')
    scene=rows[0]['precheck']['scene_id'];counts=Counter();cert=Counter();fg=Counter();identity=Counter();view=Counter()
    statuses_fg=Counter();statuses_identity=Counter();resets=Counter();funnel={};frames=[];logs=set();times={};known=set();last=None
    for row in rows:
        p=row['precheck'];c=row['commit'];s=row['committed_state'];d=p['diagnostics'];experiment=row.get('experiment') or {}
        if p['scene_id']!=scene or s['scene_id']!=scene:raise ValueError('Mixed scene sequence')
        assert d['transaction_status']=='committed' and s['transaction_status']=='committed'
        assert s['frame_id']==p['frame_id'] and s['timestamp']==p['timestamp']
        assert s['state_epoch']==d['state_epoch'] and s['source_digest']
        log=s.get('log_token',d.get('execution_context',{}).get('log_token'))
        if log is not None:logs.add(log)
        assert len(logs)<=1,'Scene cannot change log identity'
        dt=None if last is None else p['timestamp']-last['timestamp']
        if last is not None:assert dt>0 and p['frame_id']!=last['frame_id']
        for a in p['anchor_provenance'].values():
            assert (scene,a['source_frame_id']) in known and a['source_timestamp']<p['timestamp']
            assert a['source_state_epoch']==s['state_epoch'] and a['source_execution_signature']==s['execution_signature']
            assert a['state_version']
        for a in s['anchors']:
            assert a['source_timestamp']<=p['timestamp'] and a['state_version']
            assert not a['physical_or_geometry_certified']
        for f in s['features'].values():
            assert not f['network_reuse_authorized']
            assert f['tensor_captured']==(f['tensor_availability']=='captured')
        counts.update(frames=1,camera_observations=len(p['invalidation_regions']),
            historical_projection_candidates=len(p['association_candidate_ids']),
            observation_supported_anchor_events=len(p['observation_supported_anchor_ids']),
            actual_committed_anchor_events=len(c['committed_anchor_ids']),
            actual_current_observed_anchor_events=sum(a['observed'] and a['source_frame_id']==p['frame_id'] for a in s['anchors']),
            snapshot_observed_anchor_events=sum(a['observed'] for a in s['anchors']),
            certified_anchor_events=len(s['certifications']['certified_anchor_ids']))
        counts['identity_proposals']+=len(c.get('identity_continuity_proposals',[]))
        for x in c['post_perception_physical_certifications']:cert.update(x['rejection_reasons'])
        for x in c['foreground_correspondence_proposals']:
            statuses_fg[x['status']]+=1;fg.update(x['rejection_reasons'])
        for x in c.get('identity_continuity_proposals',[]):
            statuses_identity[x['status']]+=1;identity.update(x['rejection_reasons'])
            view.update(v['reason'] for v in x['measurements'].get('views',[]))
            counts['identity_candidate_proposals']+=x['status']=='candidate'
            counts['identity_unknown_proposals']+=x['status']=='unknown'
        for e in d.get('temporal_evidence',{}).values():
            for key in ('points','bidirectional_points','geometric_points','supported_points','supported_points_after_exclusion'):
                counts['measured_'+key]+=e.get(key,0)
        for r in p['invalidation_regions'].values():
            from worldstream.m1.mask_codec import decode_mask
            mask=decode_mask(r['observation_stable_mask']);counts['camera_pixels']+=mask.size
            counts['observation_supported_pixels']+=int(mask.sum())
        if d.get('reset_reason'):resets[d['reset_reason']]+=1
        for aid,a in d.get('anchors',{}).items():
            for cid,v in a['cameras'].items():
                for name,item in v.get('evidence',{}).get('observation_funnel',{}).items():
                    counter=funnel.setdefault(name,Counter(total=0,passed=0,rejected=0,unknown=0))
                    counter['total']+=1;counter['passed' if item['passed'] is True else 'rejected' if item['passed'] is False else 'unknown']+=1
        timing={'precheck_wall_ms':d.get('timings_ms',{}).get('precheck_total'),
                'commit_wall_ms':c.get('commit_total_ms'),**(experiment.get('latency') or {})}
        for k,v in timing.items():
            if k.endswith('_ms'):times.setdefault(k,[]).append(v)
        frames.append(dict(frame_id=p['frame_id'],timestamp=p['timestamp'],delta_s=dt,
            dataset_sequence_index=experiment.get('dataset_sequence_index'),epoch=s['state_epoch'],
            state_version_check='PASS',source_binding='PASS',transaction_status=d['transaction_status'],
            reset_reason=d.get('reset_reason'),source_digest=s['source_digest'],
            candidates=len(p['association_candidate_ids']),observations=len(p['observation_supported_anchor_ids']),
            certified=len(s['certifications']['certified_anchor_ids'])))
        known.add((scene,p['frame_id']));last=p
    for key in ('identity_proposals','identity_candidate_proposals','identity_unknown_proposals','camera_pixels',
                'observation_supported_pixels'):counts.setdefault(key,0)
    deltas=[f['delta_s'] for f in frames if f['delta_s'] is not None]
    missing=[dict(previous=frames[i-1]['frame_id'],current=f['frame_id'],
        skipped_dataset_keyframes=f['dataset_sequence_index']-frames[i-1]['dataset_sequence_index']-1)
        for i,f in enumerate(frames) if i and f['dataset_sequence_index'] is not None and frames[i-1]['dataset_sequence_index'] is not None
        and f['dataset_sequence_index']>frames[i-1]['dataset_sequence_index']+1]
    result=dict(base or {})
    result.update(scene_id=scene,log_token=next(iter(logs)) if logs else None,counts=dict(counts),
        time_range_s=[frames[0]['timestamp'],frames[-1]['timestamp']],frame_intervals=distribution(deltas),
        missing_keyframe_events=missing,long_gap_frames=[f['frame_id'] for f in frames if f['reset_reason']=='long_frame_gap'],
        sequence_type='nuScenes keyframes, not high-frequency video',frames=frames,binding_checks='PASS',transaction_failures=0,
        certificate_rejections=dict(cert),foreground_proposals=dict(statuses_fg),identity_proposals=dict(statuses_identity),
        foreground_rejection_reasons=dict(fg),identity_rejection_reasons=dict(identity),identity_view_reasons=dict(view),resets=dict(resets),
        observation_funnel={k:dict(v) for k,v in funnel.items()},latency_samples=times,latency={k:latency_stats(v) for k,v in times.items()},
        qualification_blockers=['independent_producer_qualification_missing'] if not counts['certified_anchor_events'] else [],
        empirical_false_acceptance_risk=None,empirical_risk_upper_95=None)
    result.update({name:ratio(counts[a],counts[b]) for name,(a,b) in RATIOS.items()})
    return result


def reproducibility_signature(summary):
    return dict(scene=summary['scene_id'],log=summary['log_token'],counts=summary['counts'],
        ratios={name:summary[name] for name in RATIOS},funnel=summary['observation_funnel'],
        rejection=summary['certificate_rejections'],identity=summary['identity_rejection_reasons'],frames=summary['frames'])


def validate_rows(rows,context=None):
    known={};seen_frames=set();scene=None;log=None;counts=Counter();last_time=None
    for row in rows:
        p,c,s=row['precheck'],row['commit'],row['committed_state'];d=p['diagnostics']
        if scene is None:scene=p['scene_id'];log=s.get('log_token')
        assert p['scene_id']==s['scene_id']==scene,'scene binding'
        assert p['frame_id']==s['frame_id'] and p['frame_id'] not in seen_frames,'frame identity'
        assert p['timestamp']==s['timestamp'] and (last_time is None or p['timestamp']>last_time),'time binding'
        assert d['transaction_status']==s['transaction_status']=='committed','transaction boundary'
        assert s['state_epoch']==d['state_epoch'] and s['source_digest'],'epoch/digest binding'
        assert s.get('log_token')==log,'log binding'
        if context is not None:
            expected=context.metadata_for(scene,p['frame_id'])
            metadata=d['execution_context']
            assert metadata['log_token']==log==expected['log_token'],'trusted log binding'
            assert metadata['log_source']==expected['log_source'],'source context binding'
        for aid,source in p['anchor_provenance'].items():
            key=(source['source_frame_id'],aid)
            assert key in known,'source must be a prior actual committed version'
            actual=known[key]
            assert source['state_version']==actual['state_version'],'historical state version mismatch'
            assert source['source_timestamp']==actual['source_timestamp']<p['timestamp'],'historical source time'
            assert source['source_state_epoch']==actual['source_epoch']==s['state_epoch'],'historical source epoch'
            assert source['source_execution_signature']==actual['execution_signature']==s['execution_signature'],'execution source binding'
            counts['historical_versions_verified']+=1
            for view in d['anchors'][aid]['cameras'].values():
                e=view.get('evidence',{})
                if 'source_state_version' in e:
                    assert e['source_state_version']==actual['state_version'],'camera evidence version mismatch'
                    assert e['source_frame_id']==source['source_frame_id'],'camera evidence source mismatch'
                    counts['camera_evidence_versions_verified']+=1
        committed=set(c['committed_anchor_ids'])
        current={a['anchor_id']:a for a in s['anchors']}
        for aid,a in current.items():
            key=(a['source_frame_id'],aid)
            if a['source_frame_id']==p['frame_id']:
                assert aid in committed and a['source_timestamp']==p['timestamp'],'uncomputed current state'
                assert a['source_epoch']==s['state_epoch'] and a['execution_signature']==s['execution_signature'],'current execution binding'
                known[key]=a;counts['actual_current_versions_verified']+=1
            else:
                assert key in known and known[key]['state_version']==a['state_version'],'retained version changed without computation'
        for cert in s['certifications']['certifications']:
            a=current[cert['anchor_id']]
            assert cert['current_state_version']==a['state_version'] and cert['current_frame_id']==p['frame_id'],'certificate version binding'
            assert cert['evidence_provenance']['risk_qualification_sha256'],'qualification missing'
        seen_frames.add(p['frame_id']);last_time=p['timestamp'];counts['frames']+=1
    return dict(status='PASS',scene_id=scene,log_token=log,checks=dict(counts))
