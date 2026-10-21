"""Unqualified, short-lived search hypotheses, isolated from world motion state.

Only an explicitly supplied physical-measurement search hint is projected.
This module does not generate identity candidates or allocate world identities.
"""
import numpy as np
from .geometry import state_geometry_version,project_points


def image_search_initial(anchor,old,new,points,hypothesis,config):
    """Box-centre optical depth is only a search hypothesis, not per-pixel truth."""
    if hypothesis is None or hypothesis.get('status')!='search_only':return None
    if (hypothesis['source_state_version']!=state_geometry_version(anchor)
            or hypothesis['source_frame_id']!=anchor.source_frame_id or old.timestamp>=new.timestamp):return None
    from .ego_motion import camera_pair_reason,project_source_depth
    if camera_pair_reason(old,new,config):return None
    center=anchor.geometry_world.center
    z=float((old.world_to_camera@np.r_[center,1])[2])
    if not .5<z<200:return None
    _,valid,world,_=project_source_depth(old,new,points,np.full(len(points),z))
    delta=np.asarray(hypothesis['velocity_world'])*(new.timestamp-old.timestamp)
    uv,target_z,front=project_points(world+delta,new.intrinsic,new.world_to_camera)
    uncertainty=np.abs(new.intrinsic[0,0])*hypothesis['uncertainty_m']/np.maximum(target_z,.5)
    uv[~valid|~front]=np.nan
    return uv,uncertainty
