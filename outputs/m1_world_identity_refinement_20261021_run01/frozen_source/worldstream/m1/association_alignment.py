"""Association-only adapter over existing measured temporal identity tools.

No image is retained here or in dormant records. Source snapshots belong to the
existing commit-only bounded image history. No physical evidence is issued.
"""
from dataclasses import replace
import numpy as np
from .geometry import project_anchor, state_geometry_version, intersect_area, region_area
from .identity_temporal import camera_snapshot_digest, _measure_view, quadrants
from .state_types import WorldAnchor


class AlignmentSession:
    def __init__(self, history, store, frame, config, validator, predictions):
        self.history=history; self.store=store; self.frame=frame
        self.config=config; self.validator=validator; self.predictions=predictions
        self.correspondences={}; self.sources={}; self.projections={}; self.results={}; self.occlusions={}
        self.calls=0; self.snapshot_valid={}

    def source(self, anchor):
        key=(anchor.anchor_id,anchor.source_frame_id)
        if key in self.sources:return self.sources[key]
        record=next((r for r in reversed(self.history) if r['frame_id']==anchor.source_frame_id),None)
        reason='actual_source_frame_unavailable'
        saved=None
        track=self.store.association_tracks.get(anchor.anchor_id)
        if record is not None:
            saved=record['anchors'].get(anchor.anchor_id)
            sample=track['samples'][-1] if track and track.get('samples') else {}
            valid=(record['scene_id']==anchor.scene_id==self.frame.scene_id
                and record['epoch']==anchor.source_state_epoch==self.store.generation
                and record['execution_signature']==anchor.source_execution_signature==self.frame.execution_signature
                and record['timestamp']==anchor.source_timestamp
                and 0<self.frame.timestamp-record['timestamp']<=min(self.config.max_frame_gap_s,1.)
                and 0<record['commit_version']<=self.store.commit_count
                and record.get('frame_digest') and sample.get('frame_digest')==record['frame_digest']
                and sample.get('commit_version')==record['commit_version']
                and saved is not None and saved.evidence_origin==anchor.evidence_origin
                and state_geometry_version(saved)==state_geometry_version(anchor))
            if valid:
                valid=all(cid in saved.observation_support
                    and np.array_equal(o['descriptor'],saved.observation_support[cid].descriptor)
                    and tuple(o['region'])==tuple(saved.observation_support[cid].region)
                    and o['timestamp']==saved.observation_support[cid].timestamp
                    for cid,o in sample.get('rois',{}).items())
                if anchor.observation_support:
                    valid=valid and all(cid in saved.observation_support
                        and np.array_equal(o.descriptor,saved.observation_support[cid].descriptor)
                        and o.region==saved.observation_support[cid].region and o.timestamp==saved.observation_support[cid].timestamp
                        for cid,o in anchor.observation_support.items())
            if valid:
                # Digests include pixels, intrinsics, pose, time and coordinate space.
                key_digest=id(record)
                if key_digest not in self.snapshot_valid:
                    self.snapshot_valid[key_digest]=all(camera_snapshot_digest(c)==record['camera_sha256'].get(cid)
                                                       for cid,c in record['cameras'].items())
                valid=self.snapshot_valid[key_digest]
            reason='ok' if valid else 'actual_source_binding_mismatch'
        result=(saved,record,reason)
        self.sources[key]=result
        return result

    def projection(self,j,cid):
        key=j,cid
        if key not in self.projections:
            p=self.predictions[j]
            a=WorldAnchor('',self.frame.scene_id,p.anchor_type,p.semantic_class,p.geometry_world,self.frame.timestamp)
            self.projections[key]=project_anchor(a,self.frame.cameras[cid],self.config)
        return self.projections[key]

    def occluded(self,j,cid):
        key=j,cid
        if key not in self.occlusions:self.occlusions[key]=self._occluded(j,cid)
        return self.occlusions[key]

    def _occluded(self,j,cid):
        cam=self.frame.cameras[cid];p=self.projection(j,cid)
        if not p.valid:return None
        z=(cam.world_to_camera@np.r_[self.predictions[j].geometry_world.center,1])[2]
        for k,other in enumerate(self.predictions):
            if k==j or other.anchor_type!='object':continue
            oz=(cam.world_to_camera@np.r_[other.geometry_world.center,1])[2]
            if oz+self.config.occlusion_depth_margin_m>=z:continue
            q=self.projection(k,cid)
            if q.valid and intersect_area(p.region,q.region)>self.config.occlusion_overlap_fraction*region_area(p.region):return True
        return False

    def measure(self,anchor,j,require_metric=False):
        key=anchor.anchor_id,j,require_metric
        if key in self.results:return self.results[key]
        saved,record,reason=self.source(anchor)
        out=dict(status='unknown',reason=reason,views=[],metric_supported=False,metric_budget_m=None)
        if reason!='ok':return out
        if self.calls>=self.config.association_measurement_limit:
            return dict(out,reason='measurement_budget_exhausted')
        self.calls+=1
        pred=self.predictions[j]
        target=WorldAnchor('',self.frame.scene_id,pred.anchor_type,pred.semantic_class,pred.geometry_world,self.frame.timestamp)
        budget=[]
        for cid,cam in self.frame.cameras.items():
            p=self.projection(j,cid);old=record['cameras'].get(cid)
            occluded=self.occluded(j,cid)
            view=dict(camera=cid,status='unknown',reason='roi_projection_invalid',projection_valid=p.valid,
                region=list(p.region) if p.region else None,coverage=float(p.coverage),occluded=occluded,
                descriptor_available=False,aligned=False,tracks=0,source_quadrants=0,current_quadrants=0,visibility='unknown',
                descriptor_residual=None,photometric_residual=None,affine_residual_px=None)
            if p.region is not None:
                view['roi_size']=[p.region[2]-p.region[0],p.region[3]-p.region[1]]
            if old is None:view['reason']='source_camera_unavailable'
            elif abs(cam.timestamp-self.frame.timestamp)>self.config.max_camera_skew_s:view['reason']='camera_time_mismatch'
            elif occluded:view['reason']='occluded'
            elif p.valid:
                value=_measure_view(saved,target,old,cam,self.validator,
                                    replace(self.config,max_state_age_s=min(1.,self.config.association_track_ttl_s)),
                                    correspondence_cache=self.correspondences)
                # Dense numeric witnesses stay ephemeral; bounded log is scalars.
                view.update({k:v for k,v in value.items() if k in ('status','reason','tracks','total_tracks',
                    'measured_tracks','descriptor_residual','source_quadrants','current_quadrants')})
                view['descriptor_available']=cid in saved.observation_support
                view['aligned']='motion_matrix' in value
                for field,raw in [('photometric_residual','patch_residual'),('affine_residual_px','affine_residual_px')]:
                    values=value.get(raw)
                    if values is not None and len(values):view[field]=float(np.max(values))
                if value['status']=='contradicted' and value['reason']=='tracks_exclude_target':
                    pp=np.asarray(value.get('exclusion_source_points',[]),float).reshape(-1,2)
                    qq=np.asarray(value.get('exclusion_current_points',[]),float).reshape(-1,2)
                    # Exclusion from scattered/partial background flow is unknown.
                    obs=saved.observation_support[cid]
                    coherent=False
                    if len(pp)>=8 and quadrants(pp,obs.region)>=3:
                        design=np.c_[pp,np.ones(len(pp))]
                        mat,_,rank,_=np.linalg.lstsq(design,qq,rcond=None)
                        coherent=rank==3 and np.max(np.linalg.norm(design@mat-qq,axis=1))<=self.config.flow_fb_max_px
                    if not coherent:view.update(status='unknown',reason='exclusion_motion_unreliable')
                if value['status']=='supported' or value['reason']=='motion_aligned_appearance_conflict':
                    # The existing descriptor helper fits half the points and
                    # validates the other half, plus real warp coverage.
                    from types import SimpleNamespace
                    from .correspondence_recovery import measured_aligned_descriptor
                    pp=np.asarray(value['source_points']);qq=np.asarray(value['current_points'])
                    evidence=SimpleNamespace(source_points=pp,current_points=qq,supported=np.ones(len(pp),bool),
                        source_timestamp=old.timestamp,timestamp=cam.timestamp)
                    descriptor,alignment=measured_aligned_descriptor(saved.observation_support[cid],old,cam,p,evidence,self.validator)
                    residual=self.validator.residual(saved.observation_support[cid].descriptor,descriptor)
                    view.update(aligned=descriptor is not None,withheld_alignment=alignment,
                                descriptor_residual=float(residual) if np.isfinite(residual) else None)
                    if descriptor is None:view.update(status='unknown',reason=alignment['reason'])
                    elif residual>self.config.residual_threshold:view.update(status='contradicted',reason='aligned_appearance_conflict')
                    elif not require_metric:
                        view.update(status='supported',reason='withheld_aligned_observation')
                    else:
                        view.update(status='supported',reason='withheld_aligned_observation')
                        # Bootstrap beyond the old world-distance gate requires
                        # independent measured object-surface AND background rays.
                        from .identity_object import measure_object_view
                        metric=measure_object_view(saved,target,old,cam,self.validator,self.config,value)
                        view['motion_compensation_status']=metric['status'];view['motion_compensation_reason']=metric['reason']
                        if metric['status']=='supported':
                            from .identity_object import surface_geometry
                            belongs,ring,gap,geometry=surface_geometry(saved,target,old,cam,
                                np.asarray(metric['source_points']),np.asarray(metric['current_points']),
                                np.asarray(metric['ring_source_points']),np.asarray(metric['ring_current_points']),self.config)
                            # Reprojection error plus declared localization budget;
                            # conditional, NOT a calibrated motion covariance.
                            depth=float((cam.world_to_camera@np.r_[target.geometry_world.center,1])[2])
                            error=float(np.max(geometry['reprojection_error_px']))
                            u=error*depth/max(float(abs(cam.intrinsic[0,0])),1.)+saved.geometry_uncertainty+pred.geometry_uncertainty
                            view['compensated_geometry_budget_m']=u
                            if belongs.all() and ring.all() and gap is not None and u<=self.config.association_gate_m:budget.append(u)
                            else:view.update(status='unknown',reason='compensated_geometry_uncertain')
                        else:view.update(status='unknown',reason='metric_motion_unavailable')
            if view['status']=='supported':view['visibility']='local_correspondence_supported'
            out['views'].append(view)
        statuses=[v['status'] for v in out['views']]
        if 'contradicted' in statuses:out.update(status='contradicted',reason='appearance_conflict')
        elif 'supported' in statuses and (not require_metric or budget):
            out.update(status='supported',reason='aligned_measured_correspondence',metric_supported=bool(budget),
                       metric_budget_m=max(budget) if budget else None)
        else:out['reason']='occluded' if any(v.get('occluded') for v in out['views']) else 'appearance_unknown'
        self.results[key]=out
        return out
