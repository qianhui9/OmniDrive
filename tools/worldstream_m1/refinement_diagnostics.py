"""Measured pre-method diagnostics; GT is read only in this offline runner."""
from collections import Counter,defaultdict
from dataclasses import replace
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.foreground_correspondence import triangulate_correspondences
from worldstream.m1.local_surface import _project
from .dataset import frame_from_info
from .coverage_oracle import world_lidar,prepare_pair,score_camera
from .evidence_experiments import load_inputs,append_json
from .identity_surface_audit import flow_fields
from .io_utils import write_json

ROOT=Path('outputs/module1_quality_coverage_refinement')
OLD=Path('outputs/module1_quality_coverage_optimization')


def patch_search(old,current,p,q):
    """Independent direct-patch search, diagnostic only, without GT-guided selection."""
    from worldstream.m1.temporal_visibility import gray_image
    a=gray_image(old).astype(np.float32)/255;b=gray_image(current).astype(np.float32)/255
    patch=cv2.getRectSubPix(b,(7,7),tuple(map(float,q)));costs=[]
    for dy in range(-4,5):
        for dx in range(-4,5):
            xy=(float(p[0]+dx),float(p[1]+dy));sample=cv2.getRectSubPix(a,(7,7),xy)
            delta=np.clip(np.median(patch-sample),-.03,.03)
            costs.append((float(np.mean(np.abs(patch-sample-delta))),xy))
    costs.sort(key=lambda t:t[0]);best=costs[0]
    from worldstream.m1.identity_temporal import _lk
    direct,fb,valid=_lk(b.mul(255).astype(np.uint8) if hasattr(b,'mul') else (b*255).astype(np.uint8),
                         (a*255).astype(np.uint8),np.asarray(q,np.float32).reshape(1,2))
    return dict(best_source_uv=list(best[1]),best_cost=best[0],second_cost=costs[1][0],
        direct_LK_source_uv=direct[0].tolist(),direct_LK_fb_error=float(fb[0]),direct_LK_valid=bool(valid[0]),
        direct_LK_DIS_difference=float(np.linalg.norm(direct[0]-p)))


def unit_for(units,x,y):
    return next((u for u in units or [] if u['bounds'][0]<=x<u['bounds'][2] and u['bounds'][1]<=y<u['bounds'][3]),None)


def point_panel(previous,current,row,path):
    x,y=row['pixel'];unit=row['unit']['bounds'];old=previous.image.copy();new=current.image.copy()
    for name,uv,color in [('DIS',row['measured_source_uv'],(255,180,10)),('plane',row['plane_source_uv'],(20,230,80)),('reference',row['reference_source_uv'],(255,30,30))]:
        xy=tuple(np.rint(uv).astype(int));cv2.circle(old,xy,3,color,1)
        cv2.putText(old,name,(xy[0]+4,xy[1]),0,.35,color,1)
    cv2.rectangle(new,tuple(unit[:2]),tuple(unit[2:]),(20,230,80),1);cv2.circle(new,(x,y),3,(255,20,20),1)
    for route in row['unit']['attempts']:
        if route.get('reason')=='accepted':
            for uv,train in zip(route['key_current_uv'],route['train']):cv2.circle(new,tuple(map(int,uv)),1,(20,230,80) if train else (20,100,255),-1)
    # Local crops preserve exact measured pixels. No generated/interpolated evidence.
    sx,sy=np.rint(row['measured_source_uv']).astype(int);rad=38
    def crop(im,cx,cy):
        h,w=im.shape[:2];return cv2.resize(im[max(0,cy-rad):min(h,cy+rad),max(0,cx-rad):min(w,cx+rad)],(304,304),interpolation=cv2.INTER_NEAREST)
    panel=np.concatenate([crop(old,sx,sy),crop(new,x,y)],axis=1)
    canvas=np.zeros((352,608,3),np.uint8);canvas[:304]=panel
    cv2.putText(canvas,'Source: measured / plane / independent reference | Current: fit green, withheld blue',(5,322),0,.34,(255,255,255),1)
    cv2.putText(canvas,f"{row['case_id']} false={row['false']} source error={row['reference_error_px']:.3f}px",(5,342),0,.4,(255,255,255),1)
    path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))


def scene(spec):
    infos,outputs,oracle=_INPUTS;folder=ROOT/'diagnostics'/spec['scene_id'];folder.mkdir(parents=True,exist_ok=False)
    cfg=replace(M1Config.load('configs/worldstream_m1_independent.yaml'),dense_region_policy='surface_local_plane',
        dense_guard_radius_px=2,surface_diagnostics=True,identity_continuity=True,identity_method='temporal',
        identity_tracking_policy='experimental',identity_diagnostics=True)
    manager=WorldStateManager(cfg);previous=None;oldpoints=None;casecount=0;parity=Counter()
    historical={(r['frame'],r['camera']):r for r in map(json.loads,(OLD/'traces/development'/spec['scene_id']/'pixels.jsonl').read_text().splitlines()) if r['method']=='surface_local_plane'}
    for token in spec['samples']:
        frame=frame_from_info(infos[token],'data/nuscenes');points=world_lidar(infos[token]);r=manager.precheck(frame)
        if previous:
            refs=prepare_pair(previous,frame,oldpoints,points,oracle[previous.frame_id],oracle[token],cfg)
            for cid,ref in refs.items():
                ref['scorable']&=ref['annotated'];e=manager.last_evidence.get(cid);invalid=manager.last_masks[cid]
                stats=score_camera(ref,invalid,e,cfg)
                expected=historical[token,cid]
                parity['compared']+=1;parity['different']+=any(stats[k]!=expected[k] for k in ['accepted_points','false_points','critical_points','missed_critical_points'])
                append_json(folder/'pixels.jsonl',dict(scene=frame.scene_id,frame=token,camera=cid,**stats))
                if e is None or e.dense_source_map is None:continue
                xy=ref['xy'];scorable=ref['scorable'];critical=ref['critical']
                for unit in e.surface_units or []:
                    x0,y0,x1,y1=unit['bounds'];inside=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
                    append_json(folder/'units.jsonl',dict(scene=frame.scene_id,frame=token,camera=cid,**unit,
                        scorable_reference_points=int((inside&scorable).sum()),
                        reference_static_points=int((inside&scorable&~ref['moving']).sum()),critical_points=int((inside&critical).sum())))
                idx=np.flatnonzero(scorable);accepted=idx[~invalid[xy[idx,1],xy[idx,0]]]
                if not len(accepted):continue
                fields=e.dense_measurements
                for index in accepted:
                    x,y=map(int,xy[index]);measured=e.dense_source_map[y,x];unit=unit_for(e.surface_units,x,y)
                    assert unit is not None and unit['primary_reason']=='accepted'
                    route=next(a for a in unit['attempts'] if a.get('reason')=='accepted');normal=np.asarray(route['normal'])
                    ray=np.r_[x,y,1.]@np.linalg.inv(frame.cameras[cid].intrinsic).T
                    cc=ray/(ray@normal);ww=(cc-frame.cameras[cid].world_to_camera[:3,3])@frame.cameras[cid].world_to_camera[:3,:3]
                    predicted,z=_project(ww[None],previous.cameras[cid]);error=float(np.linalg.norm(measured-ref['source_uv'][index]))
                    false=error>2 or not np.isfinite(error) or bool(ref['moving'][index])
                    h,w=invalid.shape;boundary=e.surface_boundary_mask
                    distance=cv2.distanceTransform((~boundary).astype(np.uint8),cv2.DIST_L2,3)[y,x] if boundary is not None else None
                    row=dict(case_id=f'{token}_{cid}_{casecount:04d}',scene=frame.scene_id,frame=token,camera=cid,
                        source_frame=previous.frame_id,pixel=[x,y],false=false,measured_source_uv=measured.tolist(),
                        plane_source_uv=predicted[0].tolist(),reference_source_uv=ref['source_uv'][index].tolist(),
                        reference_error_px=error,reference_motion=bool(ref['moving'][index]),unit=unit,
                        plane_flow_residual=float(np.linalg.norm(predicted[0]-measured)),
                        withheld_error_max=max(np.asarray(route['key_reprojection_errors'])[~np.asarray(route['train'],bool)]),
                        fb_error_px=float(fields['fb'][y,x]),photometric_residual=float(fields['photo'][y,x]),
                        gradient_residual=float(fields['gradient'][y,x]),texture=float(fields['texture'][y,x]),
                        jacobian=float(fields['jacobian'][y,x]),boundary_distance_px=distance,
                        source_pose=previous.cameras[cid].world_to_camera,current_pose=frame.cameras[cid].world_to_camera,
                        source_intrinsic=previous.cameras[cid].intrinsic,current_intrinsic=frame.cameras[cid].intrinsic,
                        online_source_digest=manager._previous_frame_digest,primary_physical_cause='Unknown_Root_Cause',
                        independent_scope='Annotation-owned sparse LiDAR correspondence; no dense instance ownership reference',
                        direct_match=patch_search(previous.cameras[cid],frame.cameras[cid],measured,[x,y]))
                    from types import SimpleNamespace
                    xyz,angle,rep,depth,valid=triangulate_correspondences(previous.cameras[cid],frame.cameras[cid],SimpleNamespace(source_points=measured[None],current_points=np.array([[x,y]],float)))
                    row.update(triangulated_world=xyz[0],parallax_deg=float(angle[0]),triangulation_reprojection_px=float(rep[0]),depths=depth[0],triangulation_valid=bool(valid[0]))
                    if false:
                        path=ROOT/'visualizations/old_plane_errors'/(row['case_id']+'.png');point_panel(previous.cameras[cid],frame.cameras[cid],row,path);row['visualization']=str(path)
                    append_json(folder/'accepted_points.jsonl',row);casecount+=1
        manager.commit(frame,outputs[token],r)
        for proposal in manager.last_commit_diagnostics['identity_continuity_proposals']:
            append_json(folder/'identity_observations.jsonl',dict(scene=frame.scene_id,frame=token,**proposal))
        previous=frame;oldpoints=points
    write_json(folder/'complete.json',dict(scene=spec['scene_id'],point_records=casecount,parity=dict(parity)))
    print('diagnostics',spec['scene_id'],casecount,dict(parity),flush=True)


def summarize():
    points=[];units=[];observations=[]
    for path in (ROOT/'diagnostics').glob('*/complete.json'):
        assert not json.loads(path.read_text())['parity']['different']
        for name,dest in [('accepted_points',points),('units',units),('identity_observations',observations)]:
            p=path.parent/(name+'.jsonl')
            if p.exists():dest.extend(map(json.loads,p.read_text().splitlines()))
    bad=[r for r in points if r['false']];good=[r for r in points if not r['false']]
    assert len(bad)==39 and len(good)==155,(len(bad),len(good))
    write_json(ROOT/'surface_false_stable_analysis.json',dict(false_points=len(bad),confirmed_primary_physical_causes=0,
        unknown_root_cause=39,rule='Error confirmed by unchanged independent reference; fitting wrong observed coordinates is measured, but surface ownership/physical cause remains unknown',records=bad))
    keys=['plane_flow_residual','withheld_error_max','fb_error_px','photometric_residual','gradient_residual','texture','jacobian','boundary_distance_px','parallax_deg']
    compare={k:{name:dict(count=len(rows),quantiles=np.percentile([r[k] for r in rows],[0,25,50,75,100]).tolist()) for name,rows in [('correct',good),('false',bad)]} for k in keys}
    compare['direct_LK_DIS_difference']={name:np.percentile([r['direct_match']['direct_LK_DIS_difference'] for r in rows],[0,25,50,75,100]).tolist() for name,rows in [('correct',good),('false',bad)]}
    write_json(ROOT/'surface_true_false_comparison.json',dict(correct_points=155,false_points=39,measurements=compare,records=points))
    categories=sorted({u['primary_reason'] for u in units})
    write_json(ROOT/'surface_acceptance_funnel.json',dict(primary_rule='Furthest measured failure across the two legal paths; one primary cause per nonempty unit',
        mutually_exclusive={k:dict(units=sum(u['primary_reason']==k for u in units),candidate_pixels=sum(u['candidate_pixels'] for u in units if u['primary_reason']==k),
            scorable_static_reference_points=sum(u['reference_static_points'] for u in units if u['primary_reason']==k),
            critical_reference_points=sum(u['critical_points'] for u in units if u['primary_reason']==k)) for k in categories},
        overlapping_route_failures=dict(Counter(a['reason'] for u in units for a in u['attempts'])),
        note='Static sparse reference is potential support, not proof of correct flow. Raw per-unit records retained.'))
    write_json(ROOT/'identity_observability_analysis.json',dict(evaluated_edges=len(observations),
        view_reasons=dict(Counter(v['reason'] for r in observations for v in r['measurements']['views'])),
        raw='diagnostics/*/identity_observations.jsonl',no_online_GT=True))
    (ROOT/'root_cause_audit.md').write_text('# Pre-method measured audit\n\nState isolation reproduced four failures before repair; the raw test log and source snapshot are retained. The repaired default keeps proposals as shadow relations and enforces one-to-one assignments for explicitly experimental tracking.\n\nThe old surface is numerically reproduced: 39 false and 155 correct sparse accepts. Every false point has a real RGB visualization and measured plane/withheld/flow/pose/reference fields. Primary physical ownership causes remain Unknown; direct patch/LK differences are diagnostics, not independent truth. Unit-level mutually exclusive and overlapping failures are recorded separately. New method design must use these observations, without changing old thresholds or the oracle.\n')
    write_json(ROOT/'diagnostic_freeze.json',{p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in [ROOT/'surface_false_stable_analysis.json',ROOT/'surface_true_false_comparison.json',ROOT/'surface_acceptance_funnel.json',ROOT/'identity_observability_analysis.json',ROOT/'root_cause_audit.md']})
    print('frozen diagnostics',len(bad),len(good),len(units))


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--summarize-only',action='store_true');args=p.parse_args()
    if args.summarize_only:summarize();return
    cv2.setNumThreads(1);specs,infos,outputs,oracle=load_inputs('calibration')
    global _INPUTS
    _INPUTS=(infos,outputs,oracle)
    if args.workers>1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:list(pool.map(scene,specs))
    else:
        for spec in specs:scene(spec)
    summarize()


if __name__=='__main__':main()
