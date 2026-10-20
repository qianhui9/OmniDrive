import numpy as np
from tools.system_optimization.physical_oracle import historical_anchor
from worldstream.m1.geometry import predict_geometry


def test_frozen_history_geometry_inverts_actual_translation_predictor():
    pose=np.eye(4);pose[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];pose[:3,3]=[100,50,0]
    source=np.array([10.,2.,0.]);velocity=np.array([2.,0.,0.]);current=source+.5*velocity
    points=np.array([current+[-1,-1,-1],current+[1,1,1]])
    record=dict(anchor_id='a',anchor_type='object',semantic_class='car',source_timestamp=1.,verified_timestamp=1.5,
        center_ego_m=(pose[:3,:3].T@(current-pose[:3,3])).tolist(),
        geometry_ego_m=((pose[:3,:3].T@(points-pose[:3,3]).T).T).tolist(),
        velocity_ego_mps=(pose[:3,:3].T@velocity).tolist(),source_frame='actual-source')
    anchor=historical_anchor(record,pose,'scene')
    np.testing.assert_allclose(anchor.geometry_world.center,source)
    np.testing.assert_allclose(predict_geometry(anchor,1.5).points,points)
    assert anchor.source_frame_id=='actual-source'
