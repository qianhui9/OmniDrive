"""Phase-one replay diagnostics. All annotations/LiDAR remain offline."""
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
from time import perf_counter
import cv2
import numpy as np
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m1.mask_codec import decode_mask
from worldstream.m1.foreground_correspondence import triangulate_correspondences
from worldstream.m1.state_types import OnlineVisibilityEvidence
from .evidence_experiments import load_inputs, append_json, summary
from .coverage_oracle import world_lidar, prepare_pair, score_camera
from .dataset import frame_from_info
from .io_utils import write_json

ROOT=Path('outputs/module1_identity_surface_refinement')
PRIOR=Path('outputs/module1_evidence_optimization')


def flow_fields(old, new):
    def gray(c):return cv2.cvtColor(c.image,cv2.COLOR_RGB2GRAY) if c.image.ndim==3 else c.image
    a,b=gray(old),gray(new);h,w=b.shape
    f=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(a,b,None)
    r=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(b,a,None)
    y,x=np.mgrid[:h,:w].astype(np.float32);source=np.stack((x,y),-1)+r
    def sample(im):return cv2.remap(im,source[...,0],source[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    fb=np.linalg.norm(r+sample(f),axis=2)
    af,bf=a.astype(np.float32)/255,b.astype(np.float32)/255
    photo=np.abs(sample(af)-bf);grad=np.zeros_like(bf)
    for axis in (0,1):
        grad+=np.abs(sample(cv2.Sobel(af,cv2.CV_32F,int(axis==0),int(axis==1),ksize=3)/8)-
                          cv2.Sobel(bf,cv2.CV_32F,int(axis==0),int(axis==1),ksize=3)/8)
    dx=np.gradient(source,axis=1);dy=np.gradient(source,axis=0)
    jac=dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    texture=np.minimum(sample(cv2.cornerMinEigenVal(af,7,ksize=3)),cv2.cornerMinEigenVal(bf,7,ksize=3))
    return dict(source=source,forward=f,reverse=r,fb=fb,photo=photo,gradient=grad,jacobian=jac,texture=texture)


def point_diagnostics(old,cam,e,invalid,ref,method,scene,frame_id,fields=None):
    idx=np.flatnonzero(ref['scorable']);xy=ref['xy'][idx]
    idx=idx[~invalid[xy[:,1],xy[:,0]]];xy=ref['xy'][idx]
    if not len(idx):return []
    predicted=e.dense_source_map[xy[:,1],xy[:,0]]
    errors=np.linalg.norm(predicted-ref['source_uv'][idx],axis=1)
    chosen=idx[(~np.isfinite(errors))|(errors>2)|ref['moving'][idx]]
    if not len(chosen):return []
    d=flow_fields(old,cam) if fields is None else fields
    xy=ref['xy'][chosen];p=e.dense_source_map[xy[:,1],xy[:,0]]
    n=len(p);proxy=OnlineVisibilityEvidence(cam.camera_id,old.timestamp,cam.timestamp,p,xy.astype(np.float32),
        np.zeros(n),np.ones(n,bool),np.ones(n,bool),np.ones(n,bool),np.zeros(n),invalid,invalid)
    xyz,angle,reproj,depth,valid=triangulate_correspondences(old,cam,proxy)
    cells=np.rint(e.dense_source_map).astype(np.int32);h,w=invalid.shape
    inside=(cells[...,0]>=0)&(cells[...,0]<w)&(cells[...,1]>=0)&(cells[...,1]<h)
    hits=np.zeros((h,w),int);np.add.at(hits,(cells[...,1][inside],cells[...,0][inside]),1)
    rows=[]
    for k,(i,(x,y)) in enumerate(zip(chosen,xy)):
        sx,sy=np.clip(cells[y,x],[0,0],[w-1,h-1]);flags=[]
        if not valid[k] or angle[k]<.5:flags.append('C_insufficient_ray_observability')
        if hits[sy,sx]>1 or not .5<=d['jacobian'][y,x]<=2:flags.append('E_measured_many_to_one_or_fold')
        if ref['moving'][i]:flags.append('annotation_confirms_object_surface_motion')
        if not ref['known'][i]:flags.append('D_new_instance_reference')
        # A/B surface ownership cannot be inferred from flow error alone.
        root='C_geometric_degeneracy' if 'C_insufficient_ray_observability' in flags else (
            'E_mapping_nonunique' if 'E_measured_many_to_one_or_fold' in flags else 'Unknown_Root_Cause')
        rows.append(dict(method=method,scene=scene,frame=frame_id,camera=cam.camera_id,pixel=[int(x),int(y)],
            source_pixel=p[k],reference_source_uv=ref['source_uv'][i],interval_s=cam.timestamp-old.timestamp,
            forward_flow_at_source=[float(cv2.getRectSubPix(d['forward'][...,axis],(1,1),tuple(map(float,p[k])))[0,0]) for axis in (0,1)],
            backward_flow=d['reverse'][y,x],fb_error_px=float(d['fb'][y,x]),photometric_residual=float(d['photo'][y,x]),
            gradient_residual=float(d['gradient'][y,x]),texture_eigenvalue=float(d['texture'][y,x]),jacobian=float(d['jacobian'][y,x]),
            source_pose=old.world_to_camera,current_pose=cam.world_to_camera,world_point=xyz[k],depths=depth[k],
            parallax_deg=float(angle[k]),reprojection_error_px=float(reproj[k]),observable=bool(valid[k]),
            local_visibility_rejected=bool(e.local_rejection_mask[y,x]) if e.local_rejection_mask is not None else False,
            dynamic_swept_rejected=False,reference_object_moving=bool(ref['moving'][i]),
            correspondence_error_px=float(np.linalg.norm(p[k]-ref['source_uv'][i])),verified_flags=flags,root_cause=root,
            surface_ownership_A_vs_B='Unscorable: no independent source/current instance surface labels',
            oracle_scope='annotation-owned synchronized sparse LiDAR, same unchanged 2px correspondence/motion error rule'))
    return rows


def panels(old,cam,e,meta,rows,path):
    h,w=cam.image.shape[:2];a=old.image.copy();b=cam.image.copy()
    if a.ndim==2:a=cv2.cvtColor(a,cv2.COLOR_GRAY2RGB)
    if b.ndim==2:b=cv2.cvtColor(b,cv2.COLOR_GRAY2RGB)
    flow=b.copy();points=b.copy();truth=b.copy()
    for row in rows:
        xy=tuple(row['pixel']);p=tuple(np.rint(row['source_pixel']).astype(int));q=tuple(np.rint(row['reference_source_uv']).astype(int))
        cv2.arrowedLine(flow,xy,p,(255,210,10),1,tipLength=.2)
        cv2.circle(points,xy,3,(255,20,20),1);cv2.arrowedLine(truth,xy,q,(255,20,20),1,tipLength=.2)
    pictures=[a,b,flow,points]
    names=['Source RGB','Current RGB','Measured backward correspondence','False stable sparse points']
    for key,label in [('observation_stable_mask','Stable'),('unknown_mask','Unknown'),('refresh_required_mask','Refresh')]:
        mask=decode_mask(meta[key]);pic=b.copy();pic[mask]=(.4*pic[mask]+.6*np.array([20,230,80])).astype(np.uint8)
        pictures.append(pic);names.append(label)
    pictures.append(truth);names.append('Offline LiDAR/annotation source reference')
    for pic,name in zip(pictures,names):
        cv2.rectangle(pic,(0,0),(w,23),(0,0,0),-1);cv2.putText(pic,name,(5,16),0,.42,(255,255,255),1)
    out=np.concatenate([np.concatenate(pictures[:4],axis=1),np.concatenate(pictures[4:],axis=1)],axis=0)
    path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),cv2.cvtColor(out,cv2.COLOR_RGB2BGR))


def main():
    cv2.setNumThreads(1);ROOT.mkdir(exist_ok=True)
    if (ROOT/'phase1_diagnostics.json').exists():raise RuntimeError('Phase-one results already exist')
    if not (ROOT/'baseline_source').exists():shutil.copytree('worldstream/m1',ROOT/'baseline_source/worldstream/m1')
    specs,infos,outputs,oracle=load_inputs('calibration');allrows=[];pixels=[];visuals=[]
    cfg=replace(M1Config.load('configs/worldstream_m1_independent.yaml'),dense_region_policy='local_visibility',dense_guard_radius_px=2,local_boundary_radius_px=12)
    for spec in specs:
        m=WorldStateManager(cfg);previous=None;oldpoints=None
        for token in spec['samples']:
            frame=frame_from_info(infos[token],'data/nuscenes');points=world_lidar(infos[token]);r=m.precheck(frame)
            if previous:
                refs=prepare_pair(previous,frame,oldpoints,points,oracle[previous.frame_id],oracle[token],cfg)
                for cid,ref in refs.items():
                    ref['scorable']&=ref['annotated'];e=m.last_evidence.get(cid);invalid=m.last_masks[cid]
                    pixels.append(dict(scene=spec['scene_id'],frame=token,camera=cid,**score_camera(ref,invalid,e,cfg)))
                    if e is None or e.dense_source_map is None:continue
                    rows=point_diagnostics(previous.cameras[cid],frame.cameras[cid],e,invalid,ref,'local_r12',spec['scene_id'],token)
                    allrows.extend(rows)
                    for row in rows:append_json(ROOT/'traces/phase1_false_stable.jsonl',row)
                    if rows and len(visuals)<12:
                        path=ROOT/'visualizations'/f'phase1_{len(visuals):02d}.jpg'
                        panels(previous.cameras[cid],frame.cameras[cid],e,r.invalidation_regions[cid],rows,path)
                        visuals.append(dict(path=str(path),scene=spec['scene_id'],frame=token,camera=cid,count=len(rows)))
            m.commit(frame,outputs[token],r);previous=frame;oldpoints=points
        print('phase1',spec['scene_id'],'false points so far',len(allrows),flush=True)
    result=dict(false_stable_points=len(allrows),primary_taxonomy=dict(Counter(r['root_cause'] for r in allrows)),
        verified_flag_counts=dict(Counter(f for r in allrows for f in r['verified_flags'])),visualizations=visuals,
        accepted_scorable_points=sum(r['accepted_points'] for r in pixels),
        scope='Prior calibration is now development. Unlabelled surfaces remain unscorable; A/B ownership not fabricated.')
    assert result['false_stable_points']==169
    write_json(ROOT/'phase1_diagnostics.json',result)
    n=math.ceil(math.log(.05)/math.log(.95))
    write_json(ROOT/'qualification_sample_size.json',dict(rule='One scene/log fails if any identity false acceptance',
        alpha=.05,FAR_upper_max=.05,minimum_zero_error_independent_units=n,
        zero_error_upper={str(k):1-.05**(1/k) for k in (6,12,20,38,59,60)},
        repeated_frames_or_entities_are_not_independent=True,unscorable_acceptances_not_success=True))


if __name__=='__main__':main()
