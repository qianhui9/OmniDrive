"""Frozen, real-data coverage experiment with offline-only independent references."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager,WorldAnchor
from .dataset import frame_from_info,prediction_outputs,load_oracle
from .evaluate import oracle_assignment
from .refinement_common import physical_details,summarize_rows,ratio
from .io_utils import read_prediction_subset,write_json
from .coverage_oracle import world_lidar,prepare_pair,score_camera

VARIANTS={'original_m1':dict(temporal_visibility=False,local_support=False),
          'sparse_m1':{}, **{f'dense_guard_{r}':dict(dense_local_support=True,dense_guard_radius_px=r) for r in (2,4,8)}}


def run_scene(path,frames,infos,predictions,oracle,configs):
    if (path/'complete.json').exists():raise ValueError('Completed path exists')
    path.mkdir(parents=True,exist_ok=True)
    managers={name:WorldStateManager(cfg) for name,cfg in configs.items()}
    identities={name:{} for name in managers};source_anns={name:{} for name in managers}
    records={name:dict(anchors=[],pixels=[],lidar=[],events=[],timings=[]) for name in managers}
    previous=None;previous_points=None
    for index,(frame,info) in enumerate(zip(frames,infos)):
        # All online prechecks finish before any current annotation/LiDAR evaluation.
        results={};runtime={}
        for name,m in managers.items():
            start=perf_counter();results[name]=m.precheck(frame);runtime[name]=(perf_counter()-start)*1000
        annotations=oracle[frame.frame_id];current_points=world_lidar(info)
        refs=prepare_pair(previous,frame,previous_points,current_points,oracle[previous.frame_id],annotations,next(iter(configs.values()))) if previous else {}
        by_id={a['instance_id']:a for a in annotations}
        for name,m in managers.items():
            result=results[name];cfg=configs[name];rows=records[name]
            for aid,check in result.diagnostics['anchors'].items():
                anchor=m.store.anchors[aid];eligible=[v for v in check['cameras'].values() if v['geometry_valid']]
                fixed=[m.validator.residual(obs.descriptor,m.validator.descriptor(frame.cameras[cid].image,obs.region)) for cid,obs in anchor.observation_support.items() if cid in frame.cameras]
                detail=physical_details(anchor,annotations,identities[name].get(aid),source_anns[name].get(aid),frame.timestamp)
                rows['anchors'].append(dict(scene_id=frame.scene_id,frame_id=frame.frame_id,anchor_id=aid,
                    identity=identities[name].get(aid),accepted=check['valid'],reasons=check['reasons'],
                    geometry_only=bool(check['within_horizon'] and check['motion_known'] and eligible),
                    appearance_only=bool(fixed and max(fixed)<=cfg.residual_threshold),**detail))
            for cid,cam in frame.cameras.items():
                invalid=m.last_masks[cid];meta=result.invalidation_regions[cid]
                # Certified image support and scheduled reuse opportunity are distinct.
                full_blocks=(~invalid)[:invalid.shape[0]//16*16,:invalid.shape[1]//16*16].reshape(invalid.shape[0]//16,16,invalid.shape[1]//16,16).all((1,3))
                rows['pixels'].append(dict(frame_id=frame.frame_id,camera=cid,
                    stable=meta['verified_stable_area_ratio'],invalid=float(invalid.mean()),
                    effective_stable=float((~invalid).mean()),changed=meta['changed_area_ratio'],unknown=meta['unknown_area_ratio'],
                    image_pixels=invalid.size,complete_16px_tiles=int(full_blocks.sum()),tile_total=full_blocks.size,
                    refresh_override=result.full_refresh_hint))
                if previous:
                    data=score_camera(refs[cid],invalid,m.last_evidence.get(cid),cfg)
                    events=data.pop('events');rows['lidar'].append(dict(frame_id=frame.frame_id,camera=cid,**data))
                    rows['events'].extend(dict(frame_id=frame.frame_id,camera=cid,**e) for e in events)
                if index in (1,5,10):
                    folder=path/name/'masks';folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/f'{index:03d}_{cid}.npz',invalid=invalid)
            outputs=prediction_outputs(predictions[frame.frame_id],cfg)
            current=[WorldAnchor(str(j),frame.scene_id,p.anchor_type,p.semantic_class,p.geometry_world,frame.timestamp) for j,p in enumerate(outputs.anchors)]
            assignments=oracle_assignment(current,annotations)
            start=perf_counter();m.commit(frame,outputs,result);elapsed=(perf_counter()-start)*1000
            for j,aid in enumerate(m.last_commit_diagnostics['committed_anchor_ids']):
                identities[name][aid]=assignments.get(j);source_anns[name][aid]=by_id.get(assignments.get(j))
            rows['timings'].append(dict(frame_id=frame.frame_id,precheck_ms=runtime[name],commit_ms=elapsed,
                overall_ms=runtime[name]+elapsed,**result.diagnostics['timings_ms'],
                association_ms=m.last_commit_diagnostics['association_ms'],state_update_ms=m.last_commit_diagnostics['state_update_ms'],
                dense_flow_ms=sum(e.diagnostics.get('dense_flow_ms',0.) for e in m.last_evidence.values()),
                dense_validation_ms=sum(e.diagnostics.get('dense_validation_ms',0.) for e in m.last_evidence.values())))
            with (path/(name+'_state.jsonl')).open('a') as f:
                # Exact masks and version records; dense maps remain bounded per-frame memory.
                from worldstream.m1.state_types import json_value
                f.write(json.dumps(json_value(result.to_dict()),allow_nan=False)+'\n')
        previous=frame;previous_points=current_points
    for name,data in records.items():
        for key,value in data.items():write_json(path/name/(key+'.json'),value)
    write_json(path/'complete.json',dict(samples=[f.frame_id for f in frames],methods=list(configs)))
    print(path.name,'complete',flush=True)


def collect(root,split,method):
    data={k:[] for k in ['anchors','pixels','lidar','events','timings']}
    for path in sorted((root/split).glob('*/'+method)):
        for key in data:data[key]+=json.loads((path/(key+'.json')).read_text())
    sums={k:sum(r[k] for r in data['lidar']) for k in ['scorable_points','accepted_points','false_points','critical_points','missed_critical_points']}
    events=data['events'];pixels=data['pixels'];summary=dict(physical=summarize_rows(data['anchors']),lidar=dict(sums,
        false_acceptance_rate=ratio(sums['false_points'],sums['accepted_points']),critical_recall=ratio(sums['critical_points']-sums['missed_critical_points'],sums['critical_points'])),
        stable=float(np.mean([p['stable'] for p in pixels])),effective_stable=float(np.mean([p['effective_stable'] for p in pixels])),
        invalid=float(np.mean([p['invalid'] for p in pixels])),complete_tiles=sum(p['complete_16px_tiles'] for p in pixels),total_tiles=sum(p['tile_total'] for p in pixels),
        critical_events=len(events),critical_event_recall=ratio(sum(e['recalled'] for e in events),len(events)),
        new_event_recall=ratio(sum(e['recalled'] for e in events if 'newly_visible' in e['reasons']),sum('newly_visible' in e['reasons'] for e in events)),
        latency_p50_p95_ms=np.percentile([t['overall_ms'] for t in data['timings']],[50,95]).tolist())
    return summary,data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='outputs/module1_refinement/coverage_round3')
    p.add_argument('--confirmation',action='store_true',help='Use exposed scenes for revised development and fresh confirmation scenes')
    args=p.parse_args();root=Path(args.root);root.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    if (root/'protocol.json').exists():raise ValueError('Frozen protocol already exists; choose fresh root')
    cfg=replace(M1Config.load('configs/worldstream_m1_balanced.yaml'),logging=False,visualization=False)
    infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos'];scenes=sorted({i['scene_token'] for i in infos})
    scene_splits=([('calibration',list(range(48,56))+list(range(64,80))),('holdout',range(80,96))]
                  if args.confirmation else [('calibration',range(48,56)),('holdout',range(64,80))])
    variants={name:dict(values,**({'dense_motion_guard':True} if args.confirmation and name.startswith('dense') else {})) for name,values in VARIANTS.items()}
    specs={split:{f'scene_{idx:02d}':sorted([i for i in infos if i['scene_token']==scenes[idx]],key=lambda i:i['timestamp'])[:12]
         for idx in indices} for split,indices in scene_splits}
    protocol=dict(methods=variants,confirmation=args.confirmation,config=cfg.to_dict(),splits={s:{n:[i['token'] for i in v] for n,v in names.items()} for s,names in specs.items()},
        selection='Maximize effective stable pixels subject to no additional critical event misses, no additional critical LiDAR point misses, no increase LiDAR correspondence false acceptance rate and no increase anchor physical errors versus sparse_m1 on calibration. If no dense variant passes, retain sparse_m1; still evaluate all frozen dense ablations on holdout without changing selection.',
        critical_event_gate=.9,lidar_correspondence_gate_px=2.,lidar_visibility_tolerance_m=.5,
        limitation='LiDAR is offline sparse evidence, not dense pixel GT. Unannotated LiDAR points assumed static; asynchronous sensors may affect measured error. No hidden test set.',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('worldstream/m1').glob('*.py')})
    write_json(root/'protocol.json',protocol)
    all_infos=[i for names in specs.values() for v in names.values() for i in v]
    predictions=read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json',{i['token'] for i in all_infos})
    oracle=load_oracle(all_infos,'data/nuscenes')
    configs={name:replace(cfg,**values) for name,values in variants.items()}
    for name,chosen in specs['calibration'].items():run_scene(root/'calibration'/name,[frame_from_info(i,'data/nuscenes') for i in chosen],chosen,predictions,oracle,configs)
    stats={name:collect(root,'calibration',name)[0] for name in configs};base=stats['sparse_m1'];_,base_data=collect(root,'calibration','sparse_m1')
    def key(e):return (e['frame_id'],e['camera'],e['instance'])
    baseline_misses={key(e) for e in base_data['events'] if not e['recalled']}
    choices=[]
    for name,metric in stats.items():
        if not name.startswith('dense'):continue
        _,data=collect(root,'calibration',name);misses={key(e) for e in data['events'] if not e['recalled']}
        ok=(misses<=baseline_misses and metric['lidar']['missed_critical_points']<=base['lidar']['missed_critical_points']
            and metric['lidar']['false_acceptance_rate'] is not None and base['lidar']['false_acceptance_rate'] is not None
            and metric['lidar']['false_acceptance_rate']<=base['lidar']['false_acceptance_rate']
            and metric['physical']['physical_errors']<=base['physical']['physical_errors'])
        if ok:choices.append(name)
    selected=max(choices,key=lambda n:stats[n]['effective_stable']) if choices else 'sparse_m1'
    write_json(root/'selection.json',dict(selected=selected,calibration=stats,qualified=choices,frozen_before_holdout=True))
    import yaml
    (root/'selected_config.yaml').write_text(yaml.safe_dump(configs[selected].to_dict(),sort_keys=False))
    print('selected',selected,json.dumps(stats),flush=True)
    for name,chosen in specs['holdout'].items():run_scene(root/'holdout'/name,[frame_from_info(i,'data/nuscenes') for i in chosen],chosen,predictions,oracle,configs)
    write_json(root/'holdout_summary.json',{name:collect(root,'holdout',name)[0] for name in configs})

if __name__=='__main__':main()
