"""M1 evidence/region experiments with scene-frozen acquisition and oracle isolation."""
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
from time import perf_counter
import cv2
import numpy as np
from scipy.stats import beta

from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.association import associate,association_ambiguities,assignment_edge_alternatives
from worldstream.m1.geometry import region_slices
from worldstream.m1.mask_codec import decode_mask
from .dataset import frame_from_info,prediction_outputs,load_oracle
from .coverage_oracle import world_lidar,prepare_pair,score_camera
from .evidence_oracle import identity_population,lidar_reference,score_foreground_points
from .io_utils import read_prediction_subset,write_json
from .independent_refinement import distribution,sha256

ROOT=Path('outputs/module1_evidence_optimization')
BASE=Path('outputs/module1_independent_refinement')
METHODS=dict(sparse=dict(dense_local_support=False,dense_motion_guard=False),
    dense_original=dict(dense_guard_radius_px=2,dense_motion_guard=False),
    conservative=dict(),local_r4=dict(dense_region_policy='local_visibility',dense_guard_radius_px=2,local_boundary_radius_px=4),
    local_r8=dict(dense_region_policy='local_visibility',dense_guard_radius_px=2,local_boundary_radius_px=8),
    local_r12=dict(dense_region_policy='local_visibility',dense_guard_radius_px=2,local_boundary_radius_px=12))


def append_json(path,value):
    from worldstream.m1.state_types import json_value
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:handle.write(json.dumps(json_value(value),allow_nan=False)+'\n')


def summary(rows):
    pixels=sum(r['pixels'] for r in rows);events=[e for r in rows for e in r.get('events',[])]
    accepted=sum(r.get('accepted_points',0) for r in rows);false=sum(r.get('false_points',0) for r in rows)
    critical=sum(r.get('critical_points',0) for r in rows);missed=sum(r.get('missed_critical_points',0) for r in rows)
    units=defaultdict(lambda:dict(accepted=0,false=0,critical=0,missed=0))
    for r in rows:
        u=units[r['scene']]
        for key,source in [('accepted','accepted_points'),('false','false_points'),('critical','critical_points'),('missed','missed_critical_points')]:
            u[key]+=r.get(source,0)
    return dict(frames=len({r['frame'] for r in rows}),cameras=len(rows),
        observation_stable_area=sum(r['stable_pixels'] for r in rows)/pixels if pixels else None,
        effective_stable_area=sum(r['effective_pixels'] for r in rows)/pixels if pixels else None,
        refresh_area=1-sum(r['effective_pixels'] for r in rows)/pixels if pixels else None,
        changed_area=sum(r['changed_pixels'] for r in rows)/pixels if pixels else None,
        unknown_area=sum(r['unknown_pixels'] for r in rows)/pixels if pixels else None,
        scorable_lidar_points=sum(r.get('scorable_points',0) for r in rows),
        accepted_scorable_points=accepted,false_stable_points=false,
        sparse_reference_far=false/accepted if accepted else None,
        far_upper95_iid_descriptive=float(beta.ppf(.95,false+1,accepted-false)) if accepted and false<accepted else 1. if accepted else None,
        accepted_scene_support=sum(v['accepted']>0 for v in units.values()),
        critical_events=len(events),recalled_events=sum(e['recalled'] for e in events),
        critical_change_recall=sum(e['recalled'] for e in events)/len(events) if events else None,
        critical_lidar_points=critical,missed_critical_points=missed,
        critical_point_recall=1-missed/critical if critical else None,
        scene_units=dict(units),scoring_scope='Annotation-owned sparse LiDAR support; unannotated background unscorable, no dense instance mask truth',
        by_event={reason:dict(count=sum(reason in e['reasons'] for e in events),
            recalled=sum(e['recalled'] for e in events if reason in e['reasons']))
            for reason in ('newly_visible','moving_object','visibility_drop')})


def save_failure_visual(frame,reference,invalid,evidence,path,points):
    camera=frame.cameras[reference['camera']];image=camera.image.copy()
    overlay=image.copy();stable=~invalid
    overlay[stable]=(.4*overlay[stable]+.6*np.array([40,220,70])).astype(np.uint8)
    for point in points:
        xy=tuple(map(int,point['pixel']));cv2.circle(image,xy,3,(255,30,30),1)
        cv2.circle(overlay,xy,3,(255,30,30),1)
    local=np.zeros_like(stable) if evidence is None or evidence.local_rejection_mask is None else evidence.local_rejection_mask
    boundary=camera.image.copy();boundary[local]=(.4*boundary[local]+.6*np.array([220,170,30])).astype(np.uint8)
    panel=np.concatenate([image,overlay,boundary],axis=1)
    cv2.putText(panel,'Red: independently referenced missed critical points | green: accepted stable | amber: local rejection',
                (6,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))


def run_scene(split,spec,infos,outputs,oracle,methods,foreground=False,old_reproduction=False):
    scene=spec['scene_id'];folder=ROOT/'traces'/split/scene
    if (folder/'complete.json').exists():raise RuntimeError('Preserve completed scene: '+str(folder))
    if (folder/'pixels.jsonl').exists():raise RuntimeError('Partial scene found; investigate, do not append a second run')
    cfg=M1Config.load('configs/worldstream_m1_independent.yaml')
    managers={name:WorldStateManager(replace(cfg,foreground_correspondence=foreground and name=='conservative',**METHODS[name])) for name in methods}
    identities={name:{} for name in methods};previous=None;prev_points=None
    pixels=[];timings=[];population=[];association=[];foreground_rows=[];certifications=[];sizes=[];failures=[]
    parity=Counter();old_records={}
    if old_reproduction:
        path=BASE/'traces'/spec['original_split']/scene/'current/state.jsonl'
        old_records={r['precheck']['frame_id']:r['precheck'] for r in map(json.loads,path.read_text().splitlines())}
    for number,info in enumerate(infos):
        frame=frame_from_info(info,'data/nuscenes')
        # The complete current detection objects are not exposed to precheck.
        current_output=outputs[frame.frame_id]
        current_anns=oracle[frame.frame_id]
        points=world_lidar(info)
        reference=prepare_pair(previous,frame,prev_points,points,oracle[previous.frame_id],current_anns,cfg) if previous else None
        known,statuses=identity_population(current_output.anchors,current_anns)
        independent_lidar=lidar_reference(points,frame,current_anns) if foreground else None
        for name,m in managers.items():
            start=perf_counter();r=m.precheck(frame);pre_ms=(perf_counter()-start)*1000
            source_truth=identities[name].copy();old_anchors=list(m.store.anchors.values())
            for cid,camera in frame.cameras.items():
                meta=r.invalidation_regions[cid];invalid=m.last_masks[cid]
                row=dict(method=name,split=split,scene=scene,frame=frame.frame_id,camera=cid,pixels=invalid.size,
                    stable_pixels=int(decode_mask(meta['observation_stable_mask']).sum()),
                    effective_pixels=int((~invalid).sum()),changed_pixels=int(decode_mask(meta['observed_changed_mask']).sum()),
                    unknown_pixels=int(decode_mask(meta['unknown_mask']).sum()),periodic_or_forced_refresh=r.full_refresh_hint,
                    ego_translation_m=float(np.linalg.norm(frame.ego_to_world[:3,3]-previous.ego_to_world[:3,3])) if previous else None,
                    interval_s=frame.timestamp-previous.timestamp if previous else None,
                    camera_evidence=m.last_evidence[cid].diagnostics if cid in m.last_evidence else {})
                assert meta['invalidation_area_ratio']==float(invalid.mean())
                assert row['stable_pixels']+row['changed_pixels']+row['unknown_pixels']==invalid.size
                if reference:
                    ref=dict(reference[cid]);ref['scorable']=ref['scorable']&ref['annotated']
                    row.update(score_camera(ref,invalid,m.last_evidence.get(cid),m.config))
                    indices=np.flatnonzero(ref['critical']);xy=ref['xy'][indices]
                    missed=indices[~invalid[xy[:,1],xy[:,0]]]
                    if name in ('conservative','local_r8','local_r12') and len(missed):
                        e=m.last_evidence.get(cid);case=[]
                        for idx in missed:
                            x,y=map(int,ref['xy'][idx]);predicted=(e.dense_source_map[y,x] if e is not None and e.dense_source_map is not None else np.array([np.nan,np.nan]))
                            reason='new_annotation_instance' if not ref['known'][idx] else 'annotated_object_motion' if ref['moving'][idx] else 'independent_lidar_disocclusion'
                            case.append(dict(pixel=[x,y],cause_reference=reason,annotated=bool(ref['annotated'][idx]),
                                predicted_source_uv=predicted,reference_source_uv=ref['source_uv'][idx],
                                correspondence_error_px=float(np.linalg.norm(predicted-ref['source_uv'][idx])),
                                local_rejected=bool(e.local_rejection_mask[y,x]) if e is not None and e.local_rejection_mask is not None else False))
                        failure=dict(method=name,scene=scene,frame=frame.frame_id,camera=cid,points=case,
                            actual_critical_point_count=len(indices),missed_count=len(missed))
                        failures.append(failure)
                        if number in (1,2,5) or row.get('events') and any(not v['recalled'] for v in row['events']):
                            path=ROOT/'visualizations'/split/scene/f'{number:03d}_{cid}_{name}.jpg'
                            save_failure_visual(frame,dict(camera=cid),invalid,e,path,case)
                            failure['visualization']=str(path)
                if old_reproduction and name=='conservative':
                    expected=decode_mask(old_records[frame.frame_id]['invalidation_regions'][cid]['refresh_required_mask'])
                    parity['compared_camera_masks']+=1;parity['different_camera_masks']+=int(not np.array_equal(expected,invalid))
                pixels.append(row);append_json(folder/'pixels.jsonl',row)
                if name=='conservative':
                    full=dict(row,method='full_refresh',effective_pixels=0)
                    if reference:full.update(score_camera(ref,np.ones_like(invalid),m.last_evidence.get(cid),m.config))
                    pixels.append(full);append_json(folder/'pixels.jsonl',full)
            before=r.to_dict()
            start=perf_counter();m.commit(frame,current_output,r);commit_ms=(perf_counter()-start)*1000
            assert r.to_dict()['physically_certified_anchor_ids']==before['physically_certified_anchor_ids']
            diag=m.last_commit_diagnostics
            selected=diag['selected_prediction_indices'];raw_preds=[current_output.anchors[j] for j in selected]
            # Cost alternatives and all identities stay on this offline side.
            pairs,costs=associate(old_anchors,raw_preds,frame.timestamp,m.config)
            local=association_ambiguities(pairs,costs)
            global_cost=assignment_edge_alternatives(pairs,costs) if name=='conservative' else {}
            continued={(aid,j) for aid,j in diag['association_pairs']}
            if name=='conservative':
                for i,j in pairs:
                    aid=old_anchors[i].anchor_id;raw=selected[j];old=source_truth.get(aid);new=known.get(raw)
                    association.append(dict(scene=scene,frame=frame.frame_id,anchor=aid,prediction=raw,
                        source_frame=old_anchors[i].source_frame_id,source_status=old[1] if old else 'source_identity_unscorable',
                        current_status=statuses[raw]['status'],scorable=bool(old and new),
                        correct=(old[0]['instance_id']==new['instance_id']) if old and new else None,
                        source_center=old_anchors[i].geometry_world.center,current_center=raw_preds[j].geometry_world.center,
                        source_instance=old[0]['instance_id'] if old else None,current_instance=new['instance_id'] if new else None,
                        continued=(aid,raw) in continued,local_margin_m=local[i,j]['margin_m'],**global_cost[i,j]))
            for proposal in diag['foreground_correspondence_proposals']:
                source=source_truth.get(proposal['anchor_id']);expected=source[0]['instance_id'] if source else None
                scores=score_foreground_points(proposal,independent_lidar,expected) if independent_lidar else []
                item=dict(scene=scene,frame=frame.frame_id,**proposal,offline_points=scores)
                foreground_rows.append(item);append_json(folder/'foreground.jsonl',item)
            for result in diag['post_perception_physical_certifications']:
                assert not result['certified']  # no approved qualification artifact in this experiment
                certifications.append(dict(scene=scene,frame=frame.frame_id,**result))
            for j,aid in zip(selected,diag['committed_anchor_ids']):
                identities[name].pop(aid,None)
                if j in known:identities[name][aid]=(known[j],statuses[j]['status'])
                if name=='conservative':
                    population.append(dict(scene=scene,frame=frame.frame_id,prediction=j,anchor=aid,
                        semantic_class=current_output.anchors[j].semantic_class,
                        center_error_m=float(np.linalg.norm(current_output.anchors[j].geometry_world.center-known[j]['geometry'].center)) if j in known else None,
                        residual_xy=(current_output.anchors[j].geometry_world.center-known[j]['geometry'].center)[:2] if j in known else None,
                        size_error_m=(current_output.anchors[j].geometry_world.size-known[j]['geometry'].size) if j in known else None,
                        **statuses[j]))
            identities[name]={aid:truth for aid,truth in identities[name].items() if aid in m.store.anchors}
            timing=dict(method=name,scene=scene,frame=frame.frame_id,precheck_ms=pre_ms,commit_ms=commit_ms,
                total_cpu_ms=pre_ms+commit_ms,foreground_cpu_ms=sum(p['producer_cpu_ms'] for p in diag['foreground_correspondence_proposals']),
                physical_certification_ms=diag['physical_certification_ms'],association_ms=diag['association_ms'],
                **r.diagnostics['timings_ms'])
            timings.append(timing);append_json(folder/'timing.jsonl',timing)
            if name=='conservative':
                sizes.append(dict(frame=frame.frame_id,anchors=len(m.store.anchors),features=len(m.store.features),
                    tensor_bytes=m.store.tensor_bytes,history_entries=sum(len(a.association_history) for a in m.store.anchors.values()),
                    state_pickle_bytes=len(pickle.dumps(m.store.__dict__))))
                append_json(folder/'state.jsonl',dict(precheck=r.to_dict(),commit=diag))
        previous=frame;prev_points=points
    for name,value in [('association',association),('population',population),('certification',certifications),('sizes',sizes),('failures',failures)]:
        write_json(folder/(name+'.json'),value)
    write_json(folder/'summary.json',{name:summary([r for r in pixels if r['method']==name]) for name in [*methods,'full_refresh']})
    write_json(folder/'complete.json',dict(split=split,scene=scene,frames=len(infos),source_sha256=sha256(__file__),mask_reproduction=dict(parity)))
    print(split,scene,'frames',len(infos),'foreground candidates',sum(p['status']=='candidate' for p in foreground_rows),flush=True)


def load_inputs(stage):
    selection=json.loads((ROOT/'data_selection.json').read_text())
    old=json.loads((BASE/'experiment_protocol.json').read_text())
    if stage=='development':
        # All old groups are explicitly development/reproduction this round.
        specs=[dict(s,original_split=split,split='development') for split,values in old['scenes'].items() for s in values]
        infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
        needed={token for s in specs for token in s['samples']}
        predictions=read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json',needed)
        outputs={token:prediction_outputs(value,M1Config()) for token,value in predictions.items()}
    else:
        specs=selection['splits'][stage]
        needed={token for s in specs for token in s['samples']}
        infos=pickle.load(open(selection['info_file'],'rb'))['infos']
        with (ROOT/'model_observations.pkl').open('rb') as handle:outputs=pickle.load(handle)
        outputs={token:outputs[token] for token in needed}
    infos=[i for i in infos if i['token'] in needed]
    by_token={i['token']:i for i in infos};oracle=load_oracle(infos,'data/nuscenes')
    return specs,by_token,outputs,oracle


def summarize_stage(stage):
    paths=list((ROOT/'traces'/stage).glob('*/pixels.jsonl'));rows=[]
    for path in paths:rows.extend(map(json.loads,path.read_text().splitlines()))
    result={method:summary([r for r in rows if r['method']==method]) for method in sorted({r['method'] for r in rows})}
    write_json(ROOT/(stage+'_summary.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['development','calibration','heldout','final_test'],required=True)
    args=parser.parse_args();cv2.setNumThreads(1)
    if not (ROOT/'experiment_protocol.json').exists():
        if args.stage not in ('development','calibration'):raise RuntimeError('Freeze protocol before evaluation')
        protocol=dict(frozen_before_new_quality_labels=True,methods=METHODS,source='raw real RGB and actual original checkpoint detections',
            stage_scope=dict(development='All previously exposed validation scenes',calibration='New-to-M1 model-train scenes',
                             heldout='Unseen-to-M1 scenes/logs, model-training population',final_test='Unseen-to-M1 scenes/logs, model-training population'),
            region_gate=dict(critical_change_recall_min=.998,event_refresh_fraction=.9,critical_point_recall_min=.998,
                sparse_annotated_far_max=.05,minimum_accepted_scenes=3,
                scope='Sparse annotated-volume LiDAR reference; no claim about unscorable background',
                selection='Calibration only; maximize effective stable area among eligible methods; no candidate if constraints fail'),
            physical_gate=dict(far_one_sided95_max=.05,independent_unit='scene/log',zero_acceptance_far=None,
                               qualification_required=True,online_qualification=None),
            local_parameters_fixed=dict(parallax_jump_px=1.,foreground_depth_gap_m=1.,local_radii=[4,8,12],global_image_guard_increase=False),
            nonzero_association_margins='Calibration diagnostic grid only: 0,0.1,0.25,0.5; no online promotion without proof',
            cpu_affinity=sorted(os.sched_getaffinity(0)),OpenCV_threads=1,camera_workers=6,
            code_sha256={str(p):sha256(p) for p in [*Path('worldstream/m1').glob('*.py'),Path(__file__),Path('tools/worldstream_m1/evidence_oracle.py')]},
            data_selection_sha256=sha256(ROOT/'data_selection.json'),checkpoint_sha256=(BASE/'checkpoint.sha256').read_text().split()[0],
            command=['taskset','-c','0-3,6-31',*__import__('sys').argv])
        write_json(ROOT/'experiment_protocol.json',protocol)
    if args.stage in ('heldout','final_test') and not (ROOT/'algorithm_selection.json').exists():
        raise RuntimeError('Freeze calibration selection before evaluation')
    specs,infos,outputs,oracle=load_inputs(args.stage)
    # Development compares all fixed candidates. Calibration is the only place
    # where a method can be selected. Evaluation retains the full frozen grid
    # as predeclared ablations; its outcomes do not change selection.
    for spec in specs:
        stream=[infos[t] for t in spec['samples']]
        run_scene(args.stage,spec,stream,outputs,oracle,list(METHODS),foreground=True,old_reproduction=args.stage=='development')
    result=summarize_stage(args.stage)
    if args.stage=='calibration':
        eligible=[]
        for name,value in result.items():
            if not name.startswith('local_'):continue
            if (value['critical_change_recall'] is not None and value['critical_change_recall']>=.998
                    and value['critical_point_recall'] is not None and value['critical_point_recall']>=.998
                    and value['sparse_reference_far'] is not None and value['sparse_reference_far']<=.05
                    and value['accepted_scene_support']>=3
                    and value['effective_stable_area']>=result['conservative']['effective_stable_area']):eligible.append(name)
        selected=max(eligible,key=lambda name:result[name]['effective_stable_area']) if eligible else None
        write_json(ROOT/'algorithm_selection.json',dict(selected_local_method=selected,eligible=eligible,
            frozen_before_heldout=True,quality_protocol='experiment_protocol.json',
            online_default_unchanged=True,reason='Calibration-constrained experimental candidate only; independent qualification still required'))


if __name__=='__main__':main()
