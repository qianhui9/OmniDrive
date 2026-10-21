"""Paired off/shadow replay on identical actual OmniDrive detections.

No detector or tracker is run. Existing calibration/confirmation scenes remain
exposed development data. Frame observations are correlated, not IID trials.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import copy
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m1.state_types import json_value


def evaluate(batches, config, trajectories, annotations=None):
    report=dict(frames=len(batches),scene_count=len({f.scene_id for f,_ in batches}),scenes=sorted({f.scene_id for f,_ in batches}),
        sampling='correlated nuScenes keyframes; scenes/instances are dependence groups',
        split='exposed_development_replay',methods={})
    for mode in ('off','shadow'):
        manager=WorldStateManager(replace(config,dqtrack_mode=mode,dqtrack_results_path=trajectories))
        records=[]; previous_gt={};errors=0;scored=0;switches=0;instance_anchor={}
        previous_external={};consistent=0;comparable=0;conflicting=0
        start=perf_counter()
        for frame,outputs in batches:
            r=manager.precheck(frame)
            manager.commit(frame,copy.deepcopy(outputs),r)
            d=manager.last_commit_diagnostics
            bridge_diagnostics=d.get('dqtrack') or {}
            current_external={}
            committed=d.get('committed_anchor_ids',[])
            for j,hint in bridge_diagnostics.get('alignments',{}).items():
                if int(j)>=len(committed):continue
                aid=committed[int(j)]
                key=(tuple(hint['namespace']),hint['tracking_id'])
                prior=previous_external.get(aid)
                if prior is not None and prior[0]==key[0]:
                    comparable+=1;consistent+=prior==key;conflicting+=prior!=key
                current_external[aid]=key
            previous_external=current_external
            # GT is evaluated only after the online transaction has completed.
            if annotations is not None:
                from .evaluate import oracle_assignment
                current=[a for a in manager.store.anchors.values() if a.source_frame_id==frame.frame_id]
                labels=oracle_assignment(current,annotations.get(frame.frame_id,[]))
                for i,instance in labels.items():
                    a=current[i];key=(frame.scene_id,instance)
                    if a.anchor_id in previous_gt:
                        errors+=previous_gt[a.anchor_id]!=instance;scored+=1
                    if key in instance_anchor:switches+=instance_anchor[key]!=a.anchor_id
                    previous_gt[a.anchor_id]=instance;instance_anchor[key]=a.anchor_id
            records.append(dict(frame_id=frame.frame_id,scene_id=frame.scene_id,
                pairs=d.get('association_pairs',[]),dqtrack=d.get('dqtrack'),
                rejected=d.get('rejected_ambiguous_associations',[]),
                certified=list(manager.get_current_certifications().certified_anchor_ids)))
        bridge=[r['dqtrack'] for r in records if r['dqtrack'] is not None]
        available=sum(d['available_tracks'] for d in bridge);aligned=sum(d['aligned'] for d in bridge)
        report['methods'][mode]=dict(records=records,elapsed_ms=(perf_counter()-start)*1000,
            available_tracks=available,aligned=aligned,alignment_rate=aligned/available if available else None,
            rejected_records=sum(len(d['rejected']) for d in bridge),
            ambiguous_assignments=sum(d['ambiguous'] for d in bridge),
            unaligned_tracks=sum(d.get('unaligned_tracks',0) for d in bridge),
            consistent_associations=consistent,comparable_external_associations=comparable,
            external_identity_conflicts=conflicting,
            association_consistency_rate=consistent/comparable if comparable else None,
            wrong_identity_continuations=errors if annotations is not None else None,
            scored_continuations=scored if annotations is not None else None,
            identity_switches=switches if annotations is not None else None,
            gt_metric_scope='partial GT matching within 2m; unmatched unknown, not official nuScenes metrics')
    baseline=report['methods']['off']['records']
    for mode in ('shadow',):
        rows=report['methods'][mode]['records']
        report['methods'][mode]['pairs_equal_baseline']=all(a['pairs']==b['pairs'] for a,b in zip(baseline,rows))
        report['methods'][mode]['certifications_equal_baseline']=all(a['certified']==b['certified'] for a,b in zip(baseline,rows))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('infos','data-root','trajectories','scene','output'):
        p.add_argument('--'+name,required=True)
    candidates=p.add_mutually_exclusive_group(required=True)
    candidates.add_argument('--detections',help='Standard nuScenes detection JSON')
    candidates.add_argument('--candidate-pickle',help='Trusted local FullOutputs pickle produced by collect_evidence_observations')
    p.add_argument('--config',default='configs/worldstream_m1_balanced.yaml')
    p.add_argument('--frames',type=int,default=8)
    p.add_argument('--with-gt',action='store_true')
    a=p.parse_args()
    if not 1<=a.frames<=40:raise ValueError('Small replay only: 1..40 frames')
    from .dataset import select_infos,frame_from_info,prediction_outputs,load_oracle
    from .io_utils import read_prediction_subset
    config=M1Config.load(a.config)
    infos=select_infos(a.infos,scene=a.scene,limit=a.frames)
    if a.candidate_pickle:
        import pickle
        with open(a.candidate_pickle,'rb') as handle:candidates=pickle.load(handle)
        batches=[(frame_from_info(i,a.data_root),candidates[i['token']]) for i in infos]
    else:
        detections=read_prediction_subset(a.detections,{i['token'] for i in infos})
        batches=[(frame_from_info(i,a.data_root),prediction_outputs(detections[i['token']],config)) for i in infos]
    # Trusted project infos only: existing loader uses pickle. Never use untrusted files.
    result=evaluate(batches,config,a.trajectories,load_oracle(infos,a.data_root) if a.with_gt else None)
    result['inputs']=vars(a)
    with open(a.output,'x') as handle:json.dump(json_value(result),handle,indent=2,allow_nan=False)


if __name__=='__main__':main()
