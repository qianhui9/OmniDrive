"""Reconstruct search hypotheses and pixel rejection budgets from prior commits.

No labels, certificates or state updates. This also audits the zero-attempt
case, rather than treating a configured dynamic arm as an activated tracker.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import numpy as np
from .dataset import frame_from_info
from .effective_acceptance import sha
from .scene_protocol import distribution,ratio
from worldstream.m1 import M1Config
from worldstream.m1.ego_motion import project_source_depth
from worldstream.m1.geometry import project_points


def run(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text());config=M1Config.load(protocol['config'])
    with open(protocol['infos_path'],'rb') as f:infos={i['token']:i for i in pickle.load(f)['infos']}
    summaries=[];pooled=Counter();all_u=[]
    for trace in sorted((root/'scenes').glob('*/repeat_0.jsonl')):
        known={};counts=Counter();values=[];previous=None;old_frame=None
        with (out/(trace.parent.name+'.jsonl')).open('w') as audit,trace.open() as stream:
            for line in stream:
                row=json.loads(line);p,c,s=row['precheck'],row['commit'],row['committed_state']
                hypotheses=c.get('search_motion',{}).get('hypotheses',{})
                current=frame_from_info(infos[p['frame_id']],'data/nuscenes')
                for aid,h in hypotheses.items():
                    if h['status']!='search_only':continue
                    assert not h['qualified'] and not h['identity_claim'] and not h['metric_geometry_claim']
                    witnesses=[known[(w['frame_id'],aid)] for w in h['sources']]
                    for w,a in zip(h['sources'],witnesses):
                        assert w['state_version']==a['state_version'] and w['epoch']==a['source_epoch']
                        assert w['execution_signature']==a['execution_signature']
                    times=np.array([a['source_timestamp'] for a in witnesses]);centers=np.array([a['center_world'] for a in witnesses])
                    v=np.diff(centers,axis=0)/np.diff(times)[:,None];dt=p['timestamp']-times[-1]
                    u=config.velocity_uncertainty_mps*dt+np.linalg.norm(v[-1]-v[-2])*dt
                    assert h['source_state_version']==witnesses[-1]['state_version']
                    assert np.allclose(h['velocity_world'],v[-1]) and np.allclose(h['center_world'],centers[-1]+v[-1]*dt)
                    assert np.isclose(u,h['uncertainty_m']) and times[-1]<p['timestamp'];counts['reconstructed_prior_hypotheses']+=1
                for prop in c['identity_continuity_proposals']:
                    h=hypotheses.get(prop['anchor_id'],{})
                    if h.get('status')!='search_only':continue
                    a=known[(h['source_frame_id'],prop['anchor_id'])]
                    for view in prop['measurements']['views']:
                        m=view.get('ego_motion',{});candidate=m.get('candidates',{}).get('dynamic_search')
                        if candidate is None:counts['view_not_reached_search_'+view['reason']]+=1;continue
                        assert old_frame is not None and previous==h['source_frame_id']
                        cid=view['camera'];old,new=old_frame.cameras[cid],current.cameras[cid];pixels=np.asarray(m['source_points'],float)
                        depth=float((old.world_to_camera@np.r_[a['center_world'],1])[2])
                        _,valid,world,_=project_source_depth(old,new,pixels,np.full(len(pixels),depth))
                        uv,z,front=project_points(world+np.asarray(h['velocity_world'])*(new.timestamp-old.timestamp),new.intrinsic,new.world_to_camera)
                        uv[~valid|~front]=np.nan
                        uncertainty=abs(new.intrinsic[0,0])*h['uncertainty_m']/np.maximum(z,.5)
                        expected=np.isfinite(uv).all(1)&(uncertainty<=config.search_radius_px)
                        assert np.array_equal(expected,np.asarray(candidate['attempted'],bool))
                        finite=np.isfinite(uv).all(1);counts['search_image_points']+=len(pixels)
                        counts['invalid_camera_projection']+=int((~finite).sum())
                        counts['uncertainty_exceeds_existing_pixel_budget']+=int((finite&~expected).sum())
                        counts['actual_image_initialization_attempts']+=int(expected.sum());values+=uncertainty[np.isfinite(uncertainty)].tolist()
                        audit.write(json.dumps(dict(frame_id=p['frame_id'],source_frame=h['source_frame_id'],source_version=h['source_state_version'],
                            anchor_id=prop['anchor_id'],camera=cid,total=len(pixels),attempted=int(expected.sum()),
                            pixel_uncertainty=distribution(uncertainty),existing_budget_px=config.search_radius_px,verified=True))+'\n')
                for a in s['anchors']:
                    if a['source_frame_id']==p['frame_id']:known[(p['frame_id'],a['anchor_id'])]=a
                previous=p['frame_id'];old_frame=current
        summary=dict(scene_id=trace.parent.name,counts=dict(counts),pixel_uncertainty=distribution(values),all_reconstructions_pass=True,
            attempt_rate=ratio(counts['actual_image_initialization_attempts'],counts['search_image_points']),trace_sha256=sha(trace))
        summaries.append(summary);pooled.update(counts);all_u+=values
    (out/'summary.json').write_text(json.dumps(dict(scenes=summaries,micro=dict(counts=dict(pooled),pixel_uncertainty=distribution(all_u)),
        all_pass=True,GT_read=False,states_updated=False,script_sha256=sha(__file__)),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.root,a.output)


if __name__=='__main__':main()
