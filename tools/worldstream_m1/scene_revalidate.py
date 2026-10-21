"""Independent ledger checks against earlier actual committed state versions."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .dataset import log_context_from_root
from .effective_acceptance import sha


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
