"""OFFLINE ONLY: sparse sensor-depth initializer diagnostic after all commits.

No manager/commit/certificate is called. Oracle outputs are never imported by
online M1. Sensor samples constrain sparse rays, not dense truth, identity or a
guaranteed mathematical upper bound. Unknowns are retained in every denominator.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .effective_acceptance import sha
from .dataset import frame_from_info,load_oracle
from .coverage_oracle import world_lidar,depth_map,prepare_pair
from .motion_scoring import score_points,counts
from .scene_protocol import distribution,ratio
from .source_anchored_scoring import score_source_anchored
from worldstream.m1 import M1Config
from worldstream.m1.ego_motion import project_source_depth,lk_measure
from worldstream.m1.geometry_support import calibrated_correspondence,measured_patch_support
from worldstream.m1.temporal_visibility import gray_image


def sensor_ray_depth(points,camera,pixels):
    uv,z,xy,front,nearest=depth_map(points,camera)
    ids=np.flatnonzero(front)
    keep=(z[ids]<=nearest[xy[ids,1],xy[ids,0]]+.5)&(z[ids]>=.5)&(z[ids]<=200)
    ids=ids[keep];depth=np.full(len(pixels),np.nan);reason=np.full(len(pixels),'no_sparse_reference',dtype=object)
    if not len(ids):return depth,reason
    tree=cKDTree(uv[ids]);distance,index=tree.query(pixels)
    for j in np.flatnonzero(distance<=1.):
        neighbours=tree.query_ball_point(pixels[j],1.);zs=z[ids[neighbours]]
        if np.ptp(zs)>1.:
            reason[j]='multiple_sensor_depths';continue
        depth[j]=z[ids[index[j]]];reason[j]='sparse_sensor_ray'
    return depth,reason


def evaluate(old,new,p,q,fb,valid,config):
    geometry=calibrated_correspondence(old,new,p,q,config.flow_epipolar_max_px)
    _,_,supported,residual=measured_patch_support(gray_image(old),gray_image(new),p,q,valid&geometry,config)
    supported&=residual<=config.support_residual_max
    return dict(source_points=p.tolist(),current_points=q.tolist(),fb_error_px=fb.tolist(),
        correspondence_valid=valid.tolist(),geometry_valid=geometry.tolist(),patch_supported=supported.tolist(),patch_residual=residual.tolist())


def run(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text());config=M1Config.load(protocol['config'])
    traces=sorted((root/'scenes').glob('*/repeat_0.jsonl'))
    rows={t.parent.name:[json.loads(x) for x in t.read_text().splitlines()] for t in traces}
    for rs in rows.values():
        if not rs or any(x['precheck']['diagnostics']['transaction_status']!='committed' for x in rs):
            raise ValueError('Oracle diagnostic requires completed online commits')
    with open(protocol['infos_path'],'rb') as f:infos={i['token']:i for i in pickle.load(f)['infos']}
    selected={x['precheck']['frame_id'] for rs in rows.values() for x in rs}
    (out/'protocol.json').write_text(json.dumps(dict(offline_only=True,online_GT=False,all_online_commits_completed_before_sensor_read=True,
        source_trace_sha256={str(t):sha(t) for t in traces},code_sha256=sha(__file__),
        sensor_source='Actual source LiDAR, sparse ray <=1px and depth discontinuity <=1m; no future sensor for initialization',
        source_depth_range_m=[.5,200],prediction_residual_limit_px=config.search_radius_px,
        evaluation='Original unchanged two-frame sparse LiDAR/annotation scoring; future/current sensor used only as offline reference',
        qualification_generated=False,online_states_updated=False,limitation='Sparse diagnostic comparison; not dense identity truth or a guaranteed upper bound'),indent=2)+'\n')
    annotations=load_oracle([infos[t] for t in selected],'data/nuscenes')
    totals={k:[] for k in ('baseline','rgb_depth','oracle_depth')};summaries=[]
    for scene,rs in rows.items():
        population={k:[] for k in totals};funnel=Counter();previous=None;prior_points=None
        folder=out/scene;folder.mkdir()
        with (folder/'diagnostic_tracks.jsonl').open('w') as log:
            for row in rs:
                token=row['precheck']['frame_id'];current=frame_from_info(infos[token],'data/nuscenes');sensor=world_lidar(infos[token])
                if previous is not None:
                    reference=prepare_pair(previous,current,prior_points,sensor,annotations[previous.frame_id],annotations[token],config)
                    for cid,e in row['precheck']['diagnostics'].get('temporal_evidence',{}).items():
                        m=e.get('point_measurements',{});p=np.asarray(m.get('source_points',[]),float)
                        if not len(p):continue
                        old,new=previous.cameras[cid],current.cameras[cid];depth,reasons=sensor_ray_depth(prior_points,old,p)
                        known=np.isfinite(depth);funnel['all_source_points']+=len(p);funnel['reliable_sparse_source_depth']+=int(known.sum())
                        funnel.update(reasons.tolist())
                        initializer=p.copy();projection_valid=np.zeros(len(p),bool)
                        if known.any():initializer[known],projection_valid[known],_,_=project_source_depth(old,new,p[known],depth[known])
                        in_view=known&projection_valid&np.isfinite(initializer).all(1)&(initializer[:,0]>=0)&(initializer[:,0]<new.image.shape[1])&(initializer[:,1]>=0)&(initializer[:,1]<new.image.shape[0])
                        initializer=np.where(in_view[:,None],initializer,p)
                        oq,of,ov=lk_measure(gray_image(old),gray_image(new),p,config,initializer)
                        ov&=in_view
                        fb_valid=int(ov.sum())
                        ov&=np.linalg.norm(oq-initializer,axis=1)<=config.search_radius_px
                        rawq=np.asarray(m.get('raw_points',m['current_points']),float)
                        rawfb=np.asarray(m.get('raw_fb_error_px',m['fb_error_px']),float)
                        rawvalid=np.asarray(m.get('raw_valid',m['correspondence_valid']),bool)
                        candidates=dict(baseline=evaluate(old,new,p,rawq,rawfb,rawvalid,config),rgb_depth=m,
                            oracle_depth=evaluate(old,new,p,oq,of,ov,config))
                        funnel['oracle_initialized']+=int(in_view.sum());funnel['oracle_FB_valid']+=fb_valid
                        funnel['oracle_prediction_residual_valid']+=int(ov.sum())
                        # Reconstruct the unchanged historical dynamic/occlusion
                        # rejection footprint from the committed online trace.
                        exclusion=np.zeros(new.image.shape[:2],bool)
                        from worldstream.m1.geometry import region_slices
                        for a in row['precheck']['diagnostics'].get('anchors',{}).values():
                            view=a['cameras'].get(cid,{})
                            dynamic=a['anchor_type']=='object' and a.get('motion_hypothesis')!='static_class_prior'
                            occlusion=view.get('evidence',{}).get('observation_funnel',{}).get('projected_occlusion',{}).get('passed') is False
                            region=view.get('predicted_region')
                            if region is not None and (dynamic or occlusion):exclusion[region_slices(region,exclusion.shape)]=True
                        for name,original in list(candidates.items()):
                            measured=dict(original)
                            xy=np.rint(np.asarray(measured['current_points'])).astype(int)
                            xy[:,0]=np.clip(xy[:,0],0,exclusion.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,exclusion.shape[0]-1)
                            measured['observation_supported_after_exclusion']=(np.asarray(measured['patch_supported'],bool)&~exclusion[xy[:,1],xy[:,0]]).tolist()
                            # local dict below is consumed independently; source
                            # JSON and online states remain byte-for-byte intact.
                            candidates[name]=measured
                        for name,measured in candidates.items():
                            # Score exactly the same source population in all arms;
                            # missing oracle references remain unknown, never removed.
                            scored=score_points(reference[cid],measured)
                            anchored=score_source_anchored(reference[cid],measured)
                            for j,r in enumerate(scored):
                                r.update(arm=name,camera=cid,source_frame=previous.frame_id,current_frame=token,
                                    oracle_reference_available=bool(known[j]),oracle_reference_reason=str(reasons[j]),
                                    additional_source_anchored_score=anchored[j]['score'],
                                    additional_source_anchored_error_px=anchored[j]['correspondence_error_px'])
                                if name=='oracle_depth' and not known[j]:r.update(score='unscorable',measured=False,observation_supported=False)
                                log.write(json.dumps(r)+'\n')
                            population[name]+=scored
                previous=current;prior_points=sensor
        summary=dict(scene_id=scene,funnel=dict(funnel),arms={})
        for name,scored in population.items():
            summary['arms'][name]=dict(all_source_tracks=counts(scored),
                measured=counts([r for r in scored if r['measured']]),
                observation_supported=counts([r for r in scored if r['observation_supported']]),
                additional_source_anchored_measured=counts([dict(r,score=r['additional_source_anchored_score']) for r in scored if r['measured']]),
                common_sparse_reference=counts([r for r in scored if r['oracle_reference_available'] and r['measured']]),
                common_reference_source_denominator=sum(r['oracle_reference_available'] for r in scored))
            totals[name]+=scored
        summaries.append(summary);(folder/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(scene,summary['funnel'],flush=True)
    micro={name:dict(all_source_tracks=counts(xs),measured=counts([r for r in xs if r['measured']]),
        observation_supported=counts([r for r in xs if r['observation_supported']]),
        additional_source_anchored_measured=counts([dict(r,score=r['additional_source_anchored_score']) for r in xs if r['measured']]),
        common_sparse_reference=counts([r for r in xs if r['oracle_reference_available'] and r['measured']]),
        common_reference_source_denominator=sum(r['oracle_reference_available'] for r in xs)) for name,xs in totals.items()}
    (out/'summary.json').write_text(json.dumps(dict(scenes=summaries,micro=micro,
        macro={name:distribution([s['arms'][name]['measured']['correspondence_accuracy'] for s in summaries]) for name in totals},
        no_online_state_or_certificate_modified=True,anchor_support_counterfactual='Not Run: low-level offline correspondence diagnostic does not commit or certify Anchors'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cv2.setNumThreads(1);run(a.root,a.output)


if __name__=='__main__':main()
