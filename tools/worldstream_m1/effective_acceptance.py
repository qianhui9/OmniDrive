"""Independent read-only acceptance of M1 logs; optional actual-RGB CPU replay.

Replay uses historical full-model detections, never annotations. Full-model GPU
validation is run separately through the existing model_smoke entry point.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import cv2
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m1.omnidrive_adapter import append_result
from .dataset import frame_from_info


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def replay(output,frames=12,scenes=2):
    dest=Path(output);dest.mkdir(parents=True,exist_ok=False)
    predictions=Path('outputs/module1_evidence_optimization/model_observations.pkl')
    selection=json.loads(Path('outputs/module1_evidence_optimization/data_selection.json').read_text())
    infos_path=Path(selection['info_file'])
    with predictions.open('rb') as f:outputs=pickle.load(f)
    with infos_path.open('rb') as f:infos=pickle.load(f)['infos']
    available={i['scene_token'] for i in infos if i['token'] in outputs}
    selected=[]
    for scene in sorted(available)[:scenes]:
        selected.extend(sorted((i for i in infos if i['scene_token']==scene and i['token'] in outputs),
                               key=lambda i:i['timestamp'])[:frames])
    if len(selected)<2:raise ValueError('Insufficient actual sequential images and detections')
    protocol=dict(population='Exploratory previously exposed scenes; no qualification',
        coordinates='resized_raw_image 640px; separate from full-model preprocessed_image',
        detection_source=str(predictions),detection_sha256=sha(predictions),
        infos_sha256=sha(infos_path),selection=[dict(scene=i['scene_token'],frame=i['token']) for i in selected],
        m2_enabled=False,m3_enabled=False,online_GT=False,full_model_executed=False)
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cv2.setNumThreads(1)
    settings=replace(M1Config.load('configs/worldstream_m1_effective.yaml'),logging=False)
    for repeat in range(2):
        manager=WorldStateManager(settings)
        for info in selected:
            frame=frame_from_info(info,'data/nuscenes')
            result=manager.precheck(frame)
            # Existing records retain their explicit historical-model replay source.
            manager.commit(frame,outputs[info['token']],result)
            append_result(dest/f'repeat_{repeat}.jsonl',result,manager.last_commit_diagnostics,
                          manager.export_state_audit())
    summaries=[verify(dest/f'repeat_{i}.jsonl') for i in range(2)]
    comparable=lambda r:{k:v for k,v in r.items() if k not in ('log','log_sha256')}
    equal=comparable(summaries[0])==comparable(summaries[1])
    (dest/'reproducibility.json').write_text(json.dumps(dict(repeats=summaries,counts_and_bindings_reproduced=equal),indent=2)+'\n')
    if not equal:raise AssertionError('Replay output counts/bindings changed')


def verify(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    totals=Counter();rejections=Counter();foreground=Counter();identity=Counter();resets=Counter()
    foreground_reasons=Counter();identity_reasons=Counter();identity_view_reasons=Counter()
    frames=[];previous=None;observed_versions=set()
    for row in rows:
        p=row['precheck'];c=row['commit'];s=row['committed_state']
        assert p['diagnostics']['transaction_status']=='committed'
        assert s['transaction_status']=='committed' and s['frame_id']==p['frame_id']
        assert s['scene_id']==p['scene_id'] and s['timestamp']==p['timestamp']
        assert s['state_epoch']==p['diagnostics']['state_epoch']
        assert p['diagnostics']['no_gpu_computation_skipped'] and s['no_gpu_computation_skipped']
        for aid,source in p['anchor_provenance'].items():
            assert source['source_timestamp']<p['timestamp']
            assert source['source_execution_signature']==s['execution_signature']
            assert source['source_state_epoch']==s['state_epoch']
            assert (p['scene_id'],source['source_frame_id']) in observed_versions
        for a in s['anchors']:
            assert a['source_timestamp']<=p['timestamp'] and a['state_version']
            assert not a['physical_or_geometry_certified']
        for f in s['features'].values():
            assert f['tensor_captured']==(f['tensor_availability']=='captured')
            assert not f['network_reuse_authorized']
        for cert in s['certifications']['certifications']:
            assert cert['current_frame_id']==p['frame_id'] and cert['current_state_version']
            assert cert['evidence_provenance']['risk_qualification_sha256']
            assert cert['evidence_provenance']['scene_level_risk']['confidence_upper_95']<=.05
        if previous and previous['scene_id']==p['scene_id']:
            assert previous['timestamp']<p['timestamp'] and previous['frame_id']!=p['frame_id']
        totals['frames']+=1;totals['camera_observations']+=len(p['invalidation_regions'])
        totals['historical_projection_candidates']+=len(p['association_candidate_ids'])
        totals['observation_supported_anchor_events']+=len(p['observation_supported_anchor_ids'])
        totals['current_observed_anchor_events']+=sum(a['observed'] for a in s['anchors'])
        totals['certified_anchor_events']+=len(s['certifications']['certified_anchor_ids'])
        totals['post_perception_certificate_attempts']+=len(c['post_perception_physical_certifications'])
        totals['actual_tensor_bytes']+=c['tensor_bytes']
        for r in c['post_perception_physical_certifications']:rejections.update(r['rejection_reasons'])
        for r in c['foreground_correspondence_proposals']:
            foreground.update([r['status']]);foreground_reasons.update(r['rejection_reasons'])
        for r in c['identity_continuity_proposals']:
            identity.update([r['status']]);identity_reasons.update(r['rejection_reasons'])
            identity_view_reasons.update(v['reason'] for v in r['measurements'].get('views',[]))
        reset=p['diagnostics'].get('reset_reason')
        if reset:resets[reset]+=1
        for e in p['diagnostics'].get('temporal_evidence',{}).values():
            for key in ('points','bidirectional_points','geometric_points','supported_points'):
                totals['measured_'+key]+=e.get(key,0)
        for r in p['invalidation_regions'].values():
            # Exact run-length masks provide measured pixel counts; no area inference.
            from worldstream.m1.mask_codec import decode_mask
            mask=decode_mask(r['observation_stable_mask'])
            totals['camera_pixels']+=mask.size
            totals['observation_supported_pixels']+=int(mask.sum())
        frames.append(dict(scene=p['scene_id'],frame=p['frame_id'],timestamp=p['timestamp'],
            epoch=s['state_epoch'],committed=len(c['committed_anchor_ids']),
            candidates=len(p['association_candidate_ids']),observations=len(p['valid_anchor_ids']),
            certified=len(s['certifications']['certified_anchor_ids'])))
        observed_versions.add((p['scene_id'],p['frame_id']));previous=p
    totals['unique_scenes']=len({r['scene'] for r in frames})
    return dict(log=str(path),log_sha256=sha(path),binding_checks='PASS',counts=dict(totals),
        observation_anchor_coverage=(totals['observation_supported_anchor_events']/totals['historical_projection_candidates']
                                     if totals['historical_projection_candidates'] else None),
        observed_pixel_coverage=totals['observation_supported_pixels']/totals['camera_pixels'] if totals['camera_pixels'] else None,
        certification_coverage=totals['certified_anchor_events']/totals['current_observed_anchor_events'] if totals['current_observed_anchor_events'] else None,
        certificate_rejections=dict(rejections),foreground_proposals=dict(foreground),identity_proposals=dict(identity),resets=dict(resets),
        foreground_rejection_reasons=dict(foreground_reasons),identity_rejection_reasons=dict(identity_reasons),
        identity_view_reasons=dict(identity_view_reasons),
        empirical_false_acceptance_risk=None,empirical_risk_upper_95=None,
        risk_status='Not Run: no independently qualified producer and no accepted population',frames=frames)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-output');parser.add_argument('--log');parser.add_argument('--summary')
    a=parser.parse_args()
    if a.replay_output:replay(a.replay_output)
    if a.log:
        out=Path(a.summary)
        if out.exists():raise ValueError('Choose fresh summary')
        out.write_text(json.dumps(verify(a.log),indent=2)+'\n')


if __name__=='__main__':main()
