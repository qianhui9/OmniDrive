"""Dependency-light adaptation of DQTrack's decoded LiDAR outputs to nuScenes.

Contract audited against DQTrack-main's datasets/nuscenes_dataset_track.py,
output_to_nusc_box / lidar_nusc_box_to_global / _format_bbox. This independently
implemented adapter preserves original indices when range filtering, including
track_ids and track_scores. No tracking algorithm or model weights are copied.
Only the >=1.0 MMDetection3D box convention is accepted.
"""
import numpy as np
from pyquaternion import Quaternion


def _array(value):
    if hasattr(value, 'detach'):
        value=value.detach().cpu().numpy()
    return np.asarray(value)


def _quaternion(value):
    vector=np.asarray(value,dtype=float)
    if vector.shape!=(4,) or not np.isfinite(vector).all() or abs(np.linalg.norm(vector)-1)>1e-3:
        raise ValueError('Invalid calibration quaternion')
    return Quaternion(vector/np.linalg.norm(vector))


def _translation(value):
    vector=np.asarray(value,dtype=float)
    if vector.shape!=(3,) or not np.isfinite(vector).all():
        raise ValueError('Invalid calibration translation')
    return vector


def export_frame(result,info,track_classes,class_ranges,box_convention='mmdet3d_1_lidar'):
    """Return standard world-coordinate rows and explicit rejected source indices.

    None is the official empty tracking result. Inputs are a single decoded
    tracklet2results dict and its exact dataset.data_infos record, never a numeric
    sample ordinal. Detector-only outputs and unverified conventions reject.
    """
    if box_convention!='mmdet3d_1_lidar':
        raise ValueError('Unverified decoded box convention')
    for key in ('token','scene_token'):
        if not isinstance(info[key],str) or not info[key]:raise ValueError('Missing frame identity')
    if not np.isfinite(info['timestamp']):raise ValueError('Invalid frame timestamp')
    diagnostics=dict(input_tracks=0,exported=0,range_filtered=0,rejected=[])
    if result is None:return [],diagnostics
    if result.get('eval_det') is not False:
        raise ValueError('Detector-only or unknown output is not a tracking result')
    boxes=result['boxes_3d']
    centers=_array(boxes.gravity_center);sizes=_array(boxes.dims);yaw=_array(boxes.yaw)
    tensor=_array(boxes.tensor)
    ids=_array(result['track_ids']);scores=_array(result['track_scores']);labels=_array(result['labels_3d'])
    count=len(centers)
    if (centers.shape!=(count,3) or sizes.shape!=(count,3) or yaw.shape!=(count,)
            or tensor.ndim!=2 or tensor.shape[0]!=count or tensor.shape[1]<9
            or any(a.shape!=(count,) for a in (ids,scores,labels))):
        raise ValueError('Decoded arrays do not correspond one-to-one')
    q_le=_quaternion(info['lidar2ego_rotation']);q_eg=_quaternion(info['ego2global_rotation'])
    t_le=_translation(info['lidar2ego_translation']);t_eg=_translation(info['ego2global_translation'])
    q_lg=q_eg*q_le
    rows=[];diagnostics['input_tracks']=count
    for i in range(count):
        try:
            if (not np.isfinite(np.r_[centers[i],sizes[i],yaw[i],tensor[i,7:9],scores[i],ids[i],labels[i]]).all()
                    or np.any(sizes[i]<=0) or not 0<=scores[i]<=1
                    or ids[i]<0 or float(ids[i])!=int(ids[i])
                    or float(labels[i])!=int(labels[i]) or not 0<=labels[i]<len(track_classes)):
                raise ValueError('Invalid track geometry, score, ID or label')
            if np.count_nonzero(ids==ids[i])!=1:raise ValueError('Duplicate tracking ID')
            name=track_classes[int(labels[i])]
            limit=float(class_ranges[name])
            if not np.isfinite(limit) or limit<=0:raise ValueError('Invalid class range')
            ego_center=q_le.rotate(centers[i])+t_le
            if np.linalg.norm(ego_center[:2])>limit:
                diagnostics['range_filtered']+=1
                continue
            world_center=q_eg.rotate(ego_center)+t_eg
            orientation=q_lg*Quaternion(axis=[0,0,1],radians=float(yaw[i]))
            velocity=q_lg.rotate(np.r_[tensor[i,7:9],0.])
            rows.append(dict(sample_token=info['token'],tracking_id=str(int(ids[i])),
                tracking_name=name,tracking_score=float(scores[i]),translation=world_center.tolist(),
                size=sizes[i,[1,0,2]].tolist(),rotation=orientation.elements.tolist(),
                velocity=velocity[:2].tolist()))
        except (ValueError,KeyError,TypeError,OverflowError) as exc:
            diagnostics['rejected'].append(dict(source_index=i,reason=str(exc)))
    diagnostics['exported']=len(rows)
    return rows,diagnostics
