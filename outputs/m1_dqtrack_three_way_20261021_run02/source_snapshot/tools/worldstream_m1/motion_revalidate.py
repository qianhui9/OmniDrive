"""Read-only CPU replay depth ledger, reconstructed from real image/calibration inputs.

This checks provenance and conditional projection arithmetic, not qualification.
No annotation table or GT is read. Every used depth must originate in an earlier
committed two-view measurement; a current frame cannot supply its own prediction.
"""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import cKDTree
from .dataset import frame_from_info,log_context_from_root
from .effective_acceptance import sha
from .scene_revalidate import validate_rows
from worldstream.m1 import M1Config
from worldstream.m1.temporal_visibility import gray_image
from worldstream.m1.identity_temporal import camera_snapshot_digest
from worldstream.m1.ego_motion import project_source_depth


def validate_trace(rows, infos, config, context):
    base=validate_rows(rows,context);checks=Counter();previous=None;prior_depth={}
    for sequence_index,row in enumerate(rows):
        p=row['precheck'];s=row['committed_state']
        frame=frame_from_info(infos[p['frame_id']],'data/nuscenes',source_context=context)
        gray_cameras={}
        for cid,camera in frame.cameras.items():
            saved=copy.copy(camera);saved.image=gray_image(camera);gray_cameras[cid]=saved
        for cid,e in p['diagnostics'].get('temporal_evidence',{}).items():
            m=e.get('point_measurements',{});source=np.asarray(m.get('source_points',[]),float)
            depths=np.asarray(m.get('source_depth_m',[]),float)
            if not len(depths):continue
            finite=np.isfinite(depths)
            if finite.any():
                assert previous is not None and cid in prior_depth,'future/uncommitted source depth'
                hp,hz=prior_depth[cid];tree=cKDTree(hp)
                predictions,valid,_,_=project_source_depth(previous[cid],frame.cameras[cid],source[finite],depths[finite])
                supplied=np.asarray(m['predicted_points'],float)[finite]
                assert valid.all() and np.allclose(predictions,supplied,atol=1e-6),'point projection mismatch'
                for pixel,z in zip(source[finite],depths[finite]):
                    nearby=tree.query_ball_point(pixel,config.flow_fb_max_px+1e-6)
                    assert nearby and np.isclose(hz[nearby],z,atol=1e-6).any(),'depth was not a prior actual measured ray'
                checks['prior_depth_predictions_reconstructed']+=int(finite.sum())
        for cid,h in s.get('depth_history',{}).items():
            assert h['frame_id']==p['frame_id'] and h['scene_id']==p['scene_id']
            assert h['epoch']==s['state_epoch'] and h['execution_signature']==s['execution_signature']
            assert h['commit_version']==sequence_index+1,'depth commit version mismatch'
            assert h['timestamp']==frame.cameras[cid].timestamp
            assert h['camera_sha256']==camera_snapshot_digest(gray_cameras[cid]),'actual camera/calibration digest mismatch'
            assert not h['metric_geometry_certified']
            checks['depth_commit_bindings_verified']+=1
        next_depth={}
        if previous is not None:
            import cv2
            for cid,e in p['diagnostics'].get('temporal_evidence',{}).items():
                m=e.get('point_measurements',{});pp=np.asarray(m.get('source_points',[]),float)
                qq=np.asarray(m.get('current_points',[]),float)
                if not len(pp):continue
                a,b=previous[cid],frame.cameras[cid]
                x=cv2.triangulatePoints(a.intrinsic@a.world_to_camera[:3],b.intrinsic@b.world_to_camera[:3],pp.T,qq.T)
                with np.errstate(divide='ignore',invalid='ignore'):xyz=(x[:3]/x[3]).T
                z=(xyz@b.world_to_camera[:3,:3].T+b.world_to_camera[:3,3])[:,2]
                good=np.asarray(m['correspondence_valid'],bool)&np.asarray(m['geometry_valid'],bool)
                good&=np.asarray(m['observation_supported_after_exclusion'],bool)&np.isfinite(z)&(z>=.5)&(z<=200)
                if good.any():next_depth[cid]=(qq[good],z[good])
        prior_depth=next_depth;previous=gray_cameras
    return dict(**base,depth_checks=dict(checks))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists():raise ValueError('Choose fresh verification output')
    protocol=json.loads((root/'protocol.json').read_text())
    with open(protocol['infos_path'],'rb') as f:infos={i['token']:i for i in pickle.load(f)['infos']}
    config=M1Config.load(protocol['config']);context=log_context_from_root('data/nuscenes');results=[]
    for trace in sorted((root/'scenes').glob('*/repeat_*.jsonl')):
        result=validate_trace([json.loads(x) for x in trace.read_text().splitlines()],infos,config,context)
        results.append(dict(trace=str(trace),sha256=sha(trace),**result))
    out.write_text(json.dumps(dict(all_pass=bool(results) and all(x['status']=='PASS' for x in results),
        validator_sha256=sha(__file__),annotation_tables_read=False,traces=results),indent=2)+'\n')


if __name__=='__main__':main()
