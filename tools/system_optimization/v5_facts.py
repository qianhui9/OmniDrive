"""New-scene evidence inventory and annotation-isolated direction diagnostic."""
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

import numpy as np

from worldstream.m1.fact_certifier import calibrate_scene_maxima, certify_relative_direction
from worldstream.m1.geometry import predict_geometry
from worldstream.m1.task_facts import extract_task_facts
from worldstream.m1.state_types import json_value
from .audit import write
from .streaming_replay import distribution
from .v3_physical import upper95
from .v5 import ROOT, sha256


def summarize(rows):
    accepted=[r for r in rows if r['certification']['certified']]
    scored=[r for r in rows if r['oracle_correct'] is not None]
    scored_accept=[r for r in accepted if r['oracle_correct'] is not None]
    errors=sum(r['oracle_correct'] is False for r in scored_accept)
    valid=sum(r['oracle_correct'] is True for r in scored)
    recovered=sum(r['oracle_correct'] is True for r in scored_accept)
    scenes=len({r['scene'] for r in scored_accept})
    bound=upper95(errors,len(scored_accept))
    return dict(candidate_count=len(rows),certified_count=len(accepted),scorable_candidates=len(scored),
        scorable_certified_count=len(scored_accept),oracle_correct_candidates=valid,
        unscorable_certified_count=len(accepted)-len(scored_accept),
        oracle_direction_value_correct_candidates=sum(r.get('oracle_value_correct') is True for r in rows),
        false_acceptance=errors/len(scored_accept) if scored_accept else None,
        false_acceptance_upper95=bound,valid_fact_recovery=recovered/valid if valid else None,
        independent_accepted_scene_support=scenes,
        physical_quality_pass=bool(bound is not None and bound<.05 and scenes>=3 and
            len(scored_accept)>=59 and len(accepted)==len(scored_accept)),
        decision_status_counts=dict(Counter(r['value_status'] for r in rows)),
        rejection_reasons=dict(Counter(r['certification']['rejection_reason'] for r in rows)))


def main():
    fact_protocol=json.loads((ROOT/'relative_direction_protocol.json').read_text())
    if fact_protocol['physical_risk']['far_upper95_max']!=.05 or fact_protocol['online_consumption_allowed'] is not False:
        raise ValueError('Frozen physical risk/online gate changed')
    if json.loads((ROOT/'system_status.json').read_text())['status']!='completed':
        raise RuntimeError('Wait for new real model captures')
    paths=sorted(ROOT.glob('*/oracle_frames/*.pkl.gz'))
    if not paths:raise ValueError('No real snapshots')
    selected=set();online=[];availability=Counter();snapshot_hashes={};frames={}
    # This complete pass has no annotation loader or Oracle imported.
    with (ROOT/'traces/relative_direction_online.jsonl').open('x') as handle:
        for path in paths:
            data=pickle.load(gzip.open(path,'rb'));frame=data['frame'];frames[frame.frame_id]=frame
            snapshot_hashes[str(path)]=sha256(path)
            selected.add(frame.frame_id)
            availability['frames']+=1
            availability['current_rgb_frames']+=bool(frame.cameras)
            availability['ego_pose_frames']+=frame.ego_to_world is not None
            availability['historical_rgb_frames']+=bool(data.get('previous_cameras'))
            availability['current_detection_frames']+=bool(data.get('current_head_anchors'))
            availability['calibrated_camera_frames']+=bool(frame.cameras) and all(
                c.intrinsic is not None and c.world_to_camera is not None for c in frame.cameras.values())
            availability['temporal_correspondence_frames']+=bool(data.get('temporal_evidence'))
            availability['independent_foreground_mask_frames']+=any(c.visibility_mask is not None for c in frame.cameras.values())
            for temporal in data.get('temporal_evidence',{}).values():
                availability['tracked_correspondence_points']+=len(temporal.source_points)
                availability['geometry_valid_correspondence_points']+=int(np.asarray(temporal.geometry_valid).sum())
                availability['supported_correspondence_points']+=int(np.asarray(temporal.supported).sum())
            for aid,anchor in data['history'].items():
                if anchor.anchor_type!='object':continue
                check=data['result'].diagnostics.get('anchors',{}).get(aid,{})
                candidate=next(f for f in extract_task_facts(anchor,frame,check) if f.fact_type=='relative_direction')
                point=(np.linalg.inv(frame.ego_to_world)@np.r_[predict_geometry(anchor,frame.timestamp).center,1.])[:2]
                evidence=dict(identity={},foreground={},
                    observation_consistent=bool(check.get('valid')),source='current_RGB_and_historical_head',
                    geometry_diagnostics={cid:c.get('evidence',{}).get('bidirectional_geometry',{}) for cid,c in check.get('cameras',{}).items()})
                row=dict(scene=frame.scene_id,frame=frame.frame_id,split=data['split'],candidate=asdict(candidate),
                    center_xy=point.tolist(),evidence=evidence,
                    certification=asdict(certify_relative_direction(candidate,frame,point,evidence,None,associations=())),
                    certification_frozen_before_any_oracle_load=True,
                    historical_observations=anchor.history_observations,
                    has_source_frame=bool(anchor.source_frame_id),has_current_detection=bool(data.get('current_head_anchors')),
                    source_execution_matches=anchor.source_execution_signature==frame.execution_signature)
                handle.write(json.dumps(json_value(row),allow_nan=False)+'\n')
                online.append((path,row));selected.add(anchor.source_frame_id)
                availability['object_candidates']+=1
                availability['candidate_source_frames']+=bool(anchor.source_frame_id)
                availability['observation_consistent_candidates']+=bool(check.get('valid'))
                availability['history_association_candidates']+=anchor.history_observations>1
    extraction_sha=sha256(ROOT/'traces/relative_direction_online.jsonl')
    # Annotation data enters only independent scoring from this point onward.
    from tools.worldstream_m1.dataset import load_oracle
    infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    def score_split(split):
        part=[(path,row) for path,row in online if row['split']==split]
        tokens={r['frame'] for _,r in part}|{r['candidate']['source_frame'] for _,r in part}
        oracle=load_oracle([r for r in infos if r['token'] in tokens],'data/nuscenes')
        scored=[];cache=None;last_path=None
        for path,row in part:
            if path!=last_path:cache=pickle.load(gzip.open(path,'rb'));last_path=path
            frame=cache['frame'];anchor=cache['history'][row['candidate']['anchor_id']]
            eligible=[ann for ann in oracle.get(anchor.source_frame_id,[]) if ann['class']==anchor.semantic_class and
                      ann['has_sensor_support'] and np.linalg.norm(anchor.geometry_world.center-ann['geometry'].center)<=2.]
            identity=None
            if len(eligible)==1:
                source=eligible[0]
                competitors=[a for a in cache['history'].values() if a.anchor_type=='object' and
                    a.source_frame_id==anchor.source_frame_id and a.semantic_class==anchor.semantic_class and
                    np.linalg.norm(a.geometry_world.center-source['geometry'].center)<=2.]
                if len(competitors)==1:identity=source['instance_id']
            ann=next((a for a in oracle.get(frame.frame_id,[]) if a['instance_id']==identity),None)
            correct=None;value_correct=None;residual=None;visible=False
            if identity is not None:
                correct=False
                if ann:
                    point=(np.linalg.inv(frame.ego_to_world)@np.r_[ann['geometry'].center,1.])[:2]
                    value_correct=bool((point!=0).all() and row['candidate']['fact_value']==dict(longitudinal=int(np.sign(point[0])),lateral=int(np.sign(point[1]))))
                    visible=bool(ann['has_sensor_support'] and str(ann['visibility'])=='4')
                    correct=visible and value_correct
                    residual=(np.asarray(row['center_xy'])-point).tolist()
            scored.append(dict(row,oracle_correct=correct,oracle_value_correct=value_correct,
                oracle_visible=visible,offline_identity_known=identity is not None,xy_error_m=residual))
        return scored
    records=score_split('calibration')
    scene_errors=defaultdict(list)
    for row in records:
        if row['xy_error_m'] is not None and row['oracle_visible']:scene_errors[row['scene']].append(row['xy_error_m'])
    signatures={frames[r['frame']].execution_signature for r in records}
    # Adapter signatures contain per-camera calibration/shape; never pool across
    # incompatible signatures and call that a portable error bound.
    calibration=calibrate_scene_maxima(scene_errors,'diagnostic_only_not_portable','v5-scene-max')
    write(ROOT/'direction_calibration.json',dict(asdict(calibration),
        usable_online=False,signature_count=len(signatures),
        reason='Too few scene units for finite 95% bound; offline 2m identity censoring also limits population',
        scene_residual_counts={s:len(r) for s,r in scene_errors.items()},
        xy_absolute_error_m=distribution([abs(v) for values in scene_errors.values() for xy in values for v in xy]),
        calibration_gt_scope='Calibration-only diagnostic; frozen before heldout/final Oracle load; never current online intervals'))
    records.extend(score_split('heldout'))
    records.extend(score_split('final_test'))
    with (ROOT/'traces/relative_direction_scored.jsonl').open('x') as handle:
        for row in records:
            row['value_status']=('candidate_value_missing' if row['candidate']['fact_value'] is None else
                'candidate_correct_proof_missing' if row['oracle_value_correct'] is True else
                'candidate_incorrect' if row['oracle_value_correct'] is False else
                'candidate_current_instance_missing' if row['oracle_correct'] is False else 'candidate_unscorable')
            handle.write(json.dumps(json_value(row),allow_nan=False)+'\n')
    results=dict(status='completed_new_scene_offline_diagnostic',online_gt_used=False,
        fact_protocol_sha256=sha256(ROOT/'relative_direction_protocol.json'),
        candidate_algorithm_implemented=True,upstream_identity_foreground_producer_available=False,
        certification_algorithm_not_implemented_count=0,
        online_evidence_insufficient_count=sum(r['certification']['rejection_reason'] in (
            'identity_unverified','foreground_unverified','calibrated_coordinate_error_bound_missing',
            'insufficient_independent_calibration_scenes') for r in records),
        frames=len(paths),scenes=len({f.scene_id for f in frames.values()}),availability=dict(availability),
        extraction_sha256=extraction_sha,snapshot_sha256=snapshot_hashes,
        overall=summarize(records),by_split={s:summarize([r for r in records if r['split']==s]) for s in ('calibration','heldout','final_test')},
        certification_latency_ms=distribution([r['certification']['certification_latency'] for r in records]),
        cost_scope='Certifier only; RGB evidence/history construction costs remain in measured persistent service E2E',
        source_assignment='Offline unique same-class within 2m; truncated source population, never online identity truth',
        decision='No-Go: independent identity/foreground unavailable; do not tune ROI or score thresholds')
    write(ROOT/'relative_direction_candidate_results.json',results)
    write(ROOT/'relative_direction_risk_coverage.json',dict(overall=results['overall'],by_split=results['by_split'],
        scene_risk={s:summarize([r for r in records if r['scene']==s]) for s in sorted({r['scene'] for r in records})},
        risk_coverage=[dict(policy='frozen evidence requirements',coverage=0.,far=None,upper95=None)],
        independent_quality_pass=False,deployment_approved=False))
    write(ROOT/'identity_foreground_diagnostics.json',dict(availability=dict(availability),
        identity_from_roi_similarity_is_proof=False,geometry_is_independent_foreground=False,
        certifier_implemented=True,producer_missing=True,
        nonobservability='Same background texture and the same projected ROI can occur with/without a foreground object. Epipolar consistency constrains a ray, not which instance owns the correspondence. Moving targets further break static triangulation.'))
    (ROOT/'online_evidence_audit.md').write_text(
        '# 在线证据与方向认证审计\n\n'+json.dumps(dict(availability),ensure_ascii=False,indent=2)+
        '\n\n历史 Source Frame、当前 RGB、标定和相机时间 Pose、历史三维几何、当前 Head、历史关联、双向对应点及残差均可获得。'
        '它们来自实际传感器/预测接口；历史关联和 Head 一致性仍是模型自身证据。'
        '当前 adapter 从未提供独立实例前景 mask，也没有问题指称到实例的独立映射。'
        '背景可穿过同一 ROI 并满足双向、极线与纹理检查，因此无法由 ROI 相似性确定物理身份。'
        '现有三角化对动态对象主动拒绝，对静态对象仍不证明前景归属。\n\n'
        '已实现独立认证分支；当前数据缺少必要证据而拒绝，区别于此前认证算法本身不存在。'
        '候选正确但缺证明、候选错误、候选无值、不可评分分别计数。'
        '6 个校准场景不足以得到 95% 场景单位 Conformal 有限界；2m 离线关联还存在截断选择，不能外推。'
        '未将 Detector Score 转为米级误差，未使用测试 GT 构造在线区间。'
        '零接受时 FAR 和上界为 null，不能报告零风险或部署通过。\n')


if __name__=='__main__':main()
