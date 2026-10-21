"""Offline-only population, foreground and uncertainty diagnostics. Never runtime."""
import numpy as np
from scipy.spatial import cKDTree
from worldstream.m1.association import legal_assignment,association_ambiguities
from worldstream.m1.geometry import project_points,oriented_box_membership


def identity_population(predictions,annotations,gate=2.,margin=.25):
    costs=np.full((len(predictions),len(annotations)),np.inf);nearest={}
    statuses={}
    for i,p in enumerate(predictions):
        candidates=[(float(np.linalg.norm(p.geometry_world.center-a['geometry'].center)),j)
                    for j,a in enumerate(annotations) if p.semantic_class==a['class'] and a['has_sensor_support']]
        if not candidates:
            statuses[i]=dict(status='no_supported_GT_same_class',nearest_distance_m=None);continue
        distance,index=min(candidates);nearest[i]=(distance,index)
        statuses[i]=dict(status='outside_2m_gate' if distance>gate else 'identity_competition',nearest_distance_m=distance,
            nearest_reference_only=annotations[index]['instance_id'])
        for distance,j in candidates:
            if distance<=gate:costs[i,j]=distance
    pairs=legal_assignment(costs);ambiguity=association_ambiguities(pairs,costs,margin);known={}
    for i,j in pairs:
        if not ambiguity[i,j]['ambiguous']:
            statuses[i]['status']='independently_scorable_unique_2m';known[i]=annotations[j]
    return known,statuses


def lidar_reference(points,frame,annotations):
    """Nearest actual LiDAR returns, not GT box projection masks."""
    owner=np.full(len(points),-1,int);multiple=np.zeros(len(points),bool)
    for j,ann in enumerate(annotations):
        inside=oriented_box_membership(points,ann['geometry'],.05)
        multiple|=inside&(owner>=0);owner[inside]=j
    owner[multiple]=-2
    result={}
    for cid,cam in frame.cameras.items():
        uv,z,good=project_points(points,cam.intrinsic,cam.world_to_camera)
        h,w=cam.image.shape[:2]
        good&=np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        indices=np.flatnonzero(good)
        result[cid]=dict(tree=cKDTree(uv[indices]) if len(indices) else None,indices=indices,
                         depth=z,owner=owner,points=points,annotations=annotations)
    return result


def score_foreground_points(proposal,reference,expected_instance):
    """Sparse geometric volume reference; inside-box is NOT a full instance mask.

    Even positive volume support is weaker than pixel ownership. Unmatched and
    multiply owned returns stay unscorable; no unknown background true negatives.
    """
    rows=[]
    for stage,key in [('proposal','candidate_points'),('selected','selected_points')]:
        for p in proposal[key]:
            ref=reference[p['camera']];tree=ref['tree'];status='no_independent_lidar_near_pixel'
            distance=None;identity=None
            if tree is not None:
                distance,index=tree.query(np.asarray(p['current_uv']),k=1)
                if distance<=1.5:
                    idx=ref['indices'][index];owner=ref['owner'][idx]
                    if owner>=0:
                        identity=ref['annotations'][owner]['instance_id']
                        if expected_instance is None:status='source_identity_unscorable'
                        elif identity==expected_instance:status='target_volume_supported_not_pixel_mask'
                        else:status='other_instance_volume'
                    elif owner==-2:status='overlapping_annotation_volumes'
                    else:
                        target=next((a for a in ref['annotations'] if a['instance_id']==expected_instance),None)
                        if target is None:status='source_identity_unscorable'
                        elif not oriented_box_membership(ref['points'][idx:idx+1],target['geometry'],.5)[0]:
                            status='outside_target_volume_background_candidate'
            rows.append(dict(stage=stage,status=status,independent_instance=identity,
                             expected_instance=expected_instance,lidar_distance_px=float(distance) if distance is not None else None,**p))
    return rows
