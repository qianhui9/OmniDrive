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


def replay(output,frames=12,scenes=2,config='configs/worldstream_m1_effective.yaml',repeats=2):
    from time import perf_counter, process_time
    from .scene_protocol import select_scenes,aggregate
    from .dataset import log_context_from_root
    dest=Path(output);dest.mkdir(parents=True,exist_ok=False)
    predictions=Path('outputs/module1_evidence_optimization/model_observations.pkl')
    selection=json.loads(Path('outputs/module1_evidence_optimization/data_selection.json').read_text())
    infos_path=Path(selection['info_file'])
    with predictions.open('rb') as f:outputs=pickle.load(f)
    with infos_path.open('rb') as f:infos=pickle.load(f)['infos']
    context=log_context_from_root('data/nuscenes')
    sequences,protocol=select_scenes(infos,set(outputs),scenes,frames,context)
    protocol.update(population='Exploratory previously exposed scenes; no qualification',
        coordinates='resized_raw_image 640px; not full-model preprocessed_image',
        detection_source=str(predictions),detection_sha256=sha(predictions),
        infos_path=str(infos_path),infos_sha256=sha(infos_path),config=config,
        log_context_sha256=context.digest,source_sha256=source_manifest(),
        m2_enabled=False,m3_enabled=False,online_GT=False,full_model_executed=False,repeats=repeats)
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cv2.setNumThreads(1)
    settings=replace(M1Config.load(config),logging=False)
    summaries=[];reproduced=True
    ledger={scene:'Not Run' for scene in sequences}
    (dest/'run_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
    for scene,sequence in sequences.items():
        folder=dest/'scenes'/scene;folder.mkdir(parents=True)
        ledger[scene]='Running'
        (dest/'run_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
        repeat_summaries=[]
        selected_spec=next(x for x in protocol['scenes'] if x['scene_id']==scene)
        ordinals={x['frame_id']:x['dataset_sequence_index'] for x in selected_spec['selected_frames']}
        for repeat in range(repeats):
            manager=WorldStateManager(settings)  # no state/history crosses this boundary
            trace=folder/f'repeat_{repeat}.jsonl'
            errors=[]
            for index,info in enumerate(sequence):
                wall=perf_counter();cpu=process_time()
                try:
                    frame=frame_from_info(info,'data/nuscenes',source_context=context)
                    manager._diagnostic_source_images={cid:c.image for cid,c in manager._previous_cameras.items()}
                    m1_cpu=process_time()
                    t=perf_counter();result=manager.precheck(frame);pre=(perf_counter()-t)*1000
                    t=perf_counter();manager.commit(frame,outputs[info['token']],result);commit=(perf_counter()-t)*1000
                    m1_cpu_ms=(process_time()-m1_cpu)*1000
                    audit=manager.export_state_audit()
                    timing=dict(precheck_wall_ms=pre,commit_wall_ms=commit,m1_cpu_process_ms=m1_cpu_ms,
                        measurement_cpu_process_ms=(process_time()-cpu)*1000,
                        end_to_end_wall_ms=(perf_counter()-wall)*1000,
                        end_to_end_scope='frame read/decode, M1 precheck, commit, audit; excludes persistence and visualization')
                    append_result(trace,result,manager.last_commit_diagnostics,audit,
                        dict(latency=timing,scene_sequence_index=index,dataset_sequence_index=ordinals[info['token']]))
                    if repeat==0:
                        from .observation_diagnostics import visualize_case
                        visualize_case(manager,frame,folder,index)
                except Exception as exc:
                    manager.abort_pending()
                    ledger[scene]='Failed'
                    (dest/'run_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
                    errors.append(dict(frame=info['token'],type=type(exc).__name__,reason=str(exc),
                        audit=manager.export_state_audit()))
                    (folder/f'exceptions_{repeat}.json').write_text(json.dumps(errors,indent=2)+'\n')
                    raise
            summary=verify(trace)
            (folder/f'metrics_{repeat}.json').write_text(json.dumps(summary,indent=2)+'\n')
            repeat_summaries.append(summary)
        from .scene_metrics import reproducibility_signature
        equal=all(reproducibility_signature(x)==reproducibility_signature(repeat_summaries[0]) for x in repeat_summaries)
        reproduced &= equal
        (folder/'reproducibility.json').write_text(json.dumps(dict(repeats=repeats,reproduced=equal),indent=2)+'\n')
        summaries.append(repeat_summaries[0])
        ledger[scene]='Completed' if equal else 'Reproducibility Failed'
        (dest/'run_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(scene,'frames',len(sequence),'coverage',summary['observation_anchor_coverage'],'reproduced',equal,flush=True)
    combined=aggregate(summaries)
    combined.update(reproduced=reproduced,executed_scene_ids=[s['scene_id'] for s in summaries],
        unexecuted_scene_ids=protocol['unexecuted_scene_ids'],full_dataset_validated=False,
        source_unchanged=protocol['source_sha256']==source_manifest())
    (dest/'summary.json').write_text(json.dumps(combined,indent=2)+'\n')
    if not reproduced or not combined['source_unchanged']:raise AssertionError('Reproducibility/source check failed')


def source_manifest():
    files=[*Path('worldstream/m1').glob('*.py'),Path(__file__),Path('tools/worldstream_m1/dataset.py'),
           Path('tools/worldstream_m1/scene_protocol.py'),Path('tools/worldstream_m1/scene_metrics.py'),
           Path('tools/worldstream_m1/observation_diagnostics.py')]
    return {str(p):sha(p) for p in files}


def _verify_sequence(path):
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


def verify(path):
    from .scene_metrics import summarize_scene
    from .scene_protocol import aggregate
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    grouped={}
    for row in rows:grouped.setdefault(row['precheck']['scene_id'],[]).append(row)
    if len(grouped)==1:
        base=_verify_sequence(path)
        return summarize_scene(rows,base)
    # Legacy traces can be analyzed, but are not evidence of independent instances.
    scenes=[summarize_scene(v,None) for v in grouped.values()]
    return dict(aggregate(scenes),independent_scene_execution=False,legacy_mixed_trace=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=int,default=12)
    parser.add_argument('--scenes',type=int,default=2,help='0: all available target scenes')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--config',default='configs/worldstream_m1_effective.yaml')
    parser.add_argument('--replay-output');parser.add_argument('--log');parser.add_argument('--summary')
    a=parser.parse_args()
    if a.replay_output:replay(a.replay_output,a.frames,a.scenes,a.config,a.repeats)
    if a.log:
        out=Path(a.summary)
        if out.exists():raise ValueError('Choose fresh summary')
        out.write_text(json.dumps(verify(a.log),indent=2)+'\n')


if __name__=='__main__':main()
