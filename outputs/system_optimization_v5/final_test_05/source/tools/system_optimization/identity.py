"""Join independent dataset/annotation arrays only by stable nuScenes identity."""
from collections import Counter
import math
import numpy as np


def token_index(infos):
    result={}
    for info in infos:
        token=str(info['token'])
        if token in result:raise ValueError('Duplicate sample_token: '+token)
        result[token]=info
    return result


def validate_sequence(selected, scene, max_gap_s=2., max_speed_mps=100., max_angular_speed_rad_s=math.pi):
    if not selected:raise ValueError('Empty sequence')
    tokens=[str(x['token']) for x in selected]
    if len(set(tokens))!=len(tokens):raise ValueError('Repeated sample within sequence')
    if any(str(x['scene_token'])!=str(scene) for x in selected):raise ValueError('Cross-scene sequence contamination')
    times=np.asarray([x['timestamp']/1e6 for x in selected])
    positions=np.asarray([x['ego2global_translation'] for x in selected],dtype=float)
    if positions.shape!=(len(selected),3) or not np.isfinite(positions).all() or not np.isfinite(times).all():
        raise ValueError('Non-finite timestamp or ego pose')
    gaps=np.diff(times);steps=np.linalg.norm(np.diff(positions,axis=0),axis=1)
    if np.any(gaps<=0):raise ValueError('Sequence timestamp not strictly increasing')
    if np.any(gaps>max_gap_s):raise ValueError('Discontinuous timestamp gap')
    if np.any(steps/gaps>max_speed_mps):raise ValueError('Implausible ego pose jump')
    rotation_checked=any('ego2global_rotation' in x for x in selected);angles=None
    if rotation_checked:
        if not all('ego2global_rotation' in x for x in selected):raise ValueError('Missing ego rotation within sequence')
        quaternions=np.asarray([x['ego2global_rotation'] for x in selected],dtype=float)
        norms=np.linalg.norm(quaternions,axis=-1)
        if quaternions.shape!=(len(selected),4) or not np.isfinite(quaternions).all() or np.any(abs(norms-1)>.01):
            raise ValueError('Invalid ego rotation quaternion')
        quaternions=quaternions/norms[:,None]
        # q and -q are the same rotation; a sign change is not a pose jump.
        angles=2*np.arccos(np.clip(abs(np.sum(quaternions[1:]*quaternions[:-1],axis=1)),0,1))
        if np.any(angles/gaps>max_angular_speed_rad_s):raise ValueError('Implausible ego rotation jump')
    counts=[dict(Counter(map(str,x.get('gt_names',[])))) for x in selected]
    return dict(sample_tokens=tokens,timestamps_s=times.tolist(),duration_s=float(times[-1]-times[0]),
        measured_frame_gaps_s=gaps.tolist(),ego_sampled_path_length_m=float(steps.sum()),
        class_counts=counts,classes_observed=sorted({k for c in counts for k in c}),
        scene_name=selected[0].get('scene_name'),description=selected[0].get('description'),
        unique_frames=len(tokens),identity_valid=True,rotation_pose_validated=rotation_checked,
        ego_angular_steps_rad=angles.tolist() if angles is not None else None)


def resolve_sequence(spec, by_token, executed=None):
    tokens=spec.get('tokens') or spec.get('sample_tokens')
    if not tokens:raise ValueError('Stable sample tokens required; integer-index fallback is forbidden')
    selected=[by_token[str(t)] for t in tokens]
    result=validate_sequence(selected,spec['scene'])
    if executed is not None:
        for token,info in zip(tokens,selected):
            records=executed.get(str(token),[])
            if not records:raise ValueError('Missing actual execution identity: '+str(token))
            for identity in records:
                if identity['scene_id']!=spec['scene'] or not math.isclose(identity['timestamp'],info['timestamp']/1e6,rel_tol=0,abs_tol=1e-5):
                    raise ValueError('Actual execution scene/timestamp mismatch: '+str(token))
    return selected,result
