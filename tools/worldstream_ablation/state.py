"""Simple state baselines. Ground truth is never an input to these engines."""
import copy
from dataclasses import replace
from types import SimpleNamespace
from time import perf_counter
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from worldstream.m1.geometry import project_anchor, region_slices
from worldstream.m1.state_types import M1Result, WorldAnchor
from worldstream.m3.world_state import CommittedAnchor, CommittedWorldState
from .vision import source_bundle


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def flow_support(previous, current, fb_limit=1.0, residual_limit=.08):
    """Dense bidirectional Farneback and local appearance residual, no geometry."""
    old, now = gray(previous), gray(current)
    fwd = cv2.calcOpticalFlowFarneback(old, now, None, .5, 3, 15, 3, 5, 1.2, 0)
    bwd = cv2.calcOpticalFlowFarneback(now, old, None, .5, 3, 15, 3, 5, 1.2, 0)
    yy, xx = np.indices(now.shape, dtype=np.float32)
    sx, sy = xx+bwd[..., 0], yy+bwd[..., 1]
    reverse = cv2.remap(fwd, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped = cv2.remap(old.astype(np.float32)/255, sx, sy, cv2.INTER_LINEAR)
    residual = cv2.blur(np.abs(warped-now.astype(np.float32)/255), (5,5))
    good = ((sx >= 0) & (sy >= 0) & (sx < now.shape[1]-1) & (sy < now.shape[0]-1) &
            (np.linalg.norm(bwd+reverse, axis=-1) <= fb_limit) & (residual <= residual_limit))
    return good, fwd


def iou(a,b):
    x0,y0 = max(a[0],b[0]),max(a[1],b[1])
    x1,y1 = min(a[2],b[2]),min(a[3],b[3])
    area = max(0,x1-x0)*max(0,y1-y0)
    return area/max(1.e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


class AlternativeState:
    def __init__(self, config, policy):
        self.config, self.policy = config, policy
        self.runtime = None
        self.reset()

    def reset(self, scene_id=None):
        epoch = getattr(getattr(self, 'store', None), 'generation', 0)+1
        self.store = SimpleNamespace(scene_id=scene_id, generation=epoch, commit_count=0,
            last_timestamp=None, execution_signature=None, anchors={})
        self.records = {}
        self.last_result = None
        self.last_commit_diagnostics = {}
        self._previous_cameras = {}
        self._previous_frame_id = None
        self._pending = None
        self.counter = 0
        self.frame = None

    def abort_pending(self):
        self._pending = None

    def _boxes(self, anchor, frame):
        result = {}
        geometry_only = replace(self.config,velocity_uncertainty_mps=0.,uncertainty_px=0.)
        for cid, cam in frame.cameras.items():
            p = project_anchor(anchor, cam, geometry_only)
            if p.valid and p.region is not None:
                result[cid] = tuple(p.region)
        return result

    def precheck_with_observation(self, frame, validate_observation=True):
        tick = perf_counter()
        if (self.store.scene_id != frame.scene_id or
            (self.store.last_timestamp is not None and frame.timestamp <= self.store.last_timestamp)):
            self.reset(frame.scene_id)
        self.frame = frame
        self._pending = frame.frame_id
        expired = [k for k,r in self.records.items() if frame.timestamp-r['timestamp'] > self.config.max_state_age_s]
        for k in expired:
            del self.records[k]
        masks = {cid:np.ones(c.image.shape[:2],bool) for cid,c in frame.cameras.items()}
        flows = {}
        if self.policy == 'flow':
            for cid, cam in frame.cameras.items():
                old = self._previous_cameras.get(cid)
                if old is not None and old.shape == cam.image.shape:
                    supported, flows[cid] = flow_support(old, cam.image)
                    masks[cid] = ~supported
        valid, invalid, rows, checks = [], [], [], {}
        for key, record in self.records.items():
            if self.policy == 'geometry':
                boxes = self._boxes(record['anchor'], frame)
                accepted = bool(boxes)
            elif self.policy == 'tracking':
                boxes = record['boxes']
                accepted = True  # Fixed TTL, no optical or observation validation.
            else:
                boxes = {}
                for cid, box in record['boxes'].items():
                    if cid not in flows:
                        continue
                    sl = region_slices(box, flows[cid].shape[:2])
                    shift = np.median(flows[cid][sl].reshape(-1,2),axis=0) if flows[cid][sl].size else np.array([0.,0.])
                    moved = np.asarray(box)+np.tile(shift,2)
                    h,w = masks[cid].shape
                    moved[[0,2]] = moved[[0,2]].clip(0,w)
                    moved[[1,3]] = moved[[1,3]].clip(0,h)
                    sl = region_slices(moved, (h,w))
                    if moved[2]>moved[0] and moved[3]>moved[1] and masks[cid][sl].size and (~masks[cid][sl]).mean() >= .8:
                        boxes[cid] = tuple(moved)
                accepted = bool(boxes)
            record['validated'] = accepted
            record['current_boxes'] = boxes
            (valid if accepted else invalid).append(key)
            checks[key] = dict(valid=accepted, method=self.policy, source_frame=record['frame_id'],
                               source_timestamp=record['timestamp'])
            if accepted:
                for cid, box in boxes.items():
                    if cid not in masks:
                        continue
                    masks[cid][region_slices(box,masks[cid].shape)] = False
                    rows.append(dict(camera_id=cid, track_id=key, bbox_xyxy=list(box),
                        coordinate_space='preprocessed_image', state_signature=(record['class'],),
                        valid_until=record['timestamp']+self.config.max_state_age_s,
                        visible=True, event_flags=[], state_source='experimental_'+self.policy))
        self.masks, self.region_rows = masks, rows
        result = M1Result(frame.scene_id,frame.frame_id,frame.timestamp,
            valid_anchor_ids=valid, invalid_anchor_ids=invalid, association_candidate_ids=list(self.records),
            full_refresh_hint=self.store.commit_count == 0,
            diagnostics=dict(transaction_status='pending', state_epoch=self.store.generation,
                execution_signature=frame.execution_signature, expired_anchor_ids=expired,
                anchors=checks, timings_ms=dict(precheck_total=(perf_counter()-tick)*1000),
                alternative=self.policy))
        self.last_result = result
        return result

    def export_m2_inputs(self, expected_context=None, computation_only=False):
        context = expected_context
        if self.frame is None or context['frame_id'] != self.frame.frame_id or self._pending != self.frame.frame_id:
            raise RuntimeError('Alternative state request/frame mismatch')
        images = self.runtime._ablation_images
        bundle = source_bundle(self.runtime, images, context,
            [self.masks[c] for c in context['camera_ids']], self.config.max_state_age_s)
        bundle.region_states = self.region_rows
        bundle.region_events = [dict(track_id=k,event='association_invalid') for k in self.last_result.invalid_anchor_ids]
        bundle.region_events += [dict(track_id=k,event='state_expired') for k in self.last_result.diagnostics['expired_anchor_ids']]
        self.last_result.computation_state = bundle
        return self.last_result

    def commit(self, frame, outputs, result):
        tick = perf_counter()
        if self._pending != frame.frame_id or result is not self.last_result:
            raise RuntimeError('Alternative state commit without matching precheck')
        # Only current decoded perception, no GT or benchmark answers.
        preds = [p for p in outputs.anchors if p.anchor_type == 'object']
        old = list(self.records)
        pairs = {}
        if self.policy == 'tracking' and old and preds:
            cost = np.full((len(old),len(preds)),1.e6)
            for i,k in enumerate(old):
                for j,p in enumerate(preds):
                    a = self.records[k]['anchor']
                    ratio = np.maximum(p.geometry_world.size/a.geometry_world.size,a.geometry_world.size/p.geometry_world.size).max()
                    if a.semantic_class == p.semantic_class and ratio <= self.config.box_size_ratio:
                        cost[i,j] = np.linalg.norm(a.geometry_world.center-p.geometry_world.center)
            rr,cc = linear_sum_assignment(cost)
            pairs = {j:old[i] for i,j in zip(rr,cc) if cost[i,j] <= self.config.association_gate_m}
        inverse = np.linalg.inv(frame.ego_to_world)
        for j,p in enumerate(preds):
            key = pairs.get(j)
            a = WorldAnchor('',frame.scene_id,p.anchor_type,p.semantic_class,p.geometry_world,frame.timestamp,
                            source_score=p.source_score)
            boxes = self._boxes(a,frame)
            if not boxes:
                continue
            if self.policy in ('geometry','flow'):
                # Image-space occupancy, no geometric cross-frame association.
                duplicate = any(r.get('validated') and r['class'] == p.semantic_class and
                    any(cid in r.get('current_boxes',{}) and iou(box,r['current_boxes'][cid])>.5 for cid,box in boxes.items())
                    for r in self.records.values())
                if duplicate:
                    continue
            if key is None:
                self.counter += 1
                key = f'{self.policy}:{self.counter}'
            center = inverse[:3,:3]@p.geometry_world.center+inverse[:3,3]
            record = dict(id=key, **{'class':p.semantic_class}, boxes=boxes, current_boxes=boxes,
                center_ego=tuple(map(float,center)), timestamp=frame.timestamp, frame_id=frame.frame_id,
                score=float(p.source_score or 0), validated=True)
            if self.policy != 'flow':
                # Flow baseline retains only image regions and semantic readout;
                # no world geometry, metric anchor, ego pose or velocity history.
                record['anchor'] = a
            self.records[key] = record
        if self.policy == 'flow':
            for record in self.records.values():
                if record['validated']:
                    record['boxes'] = record['current_boxes']
        if len(self.records)>self.config.max_anchors:
            keep = sorted(self.records,key=lambda k:(-self.records[k]['timestamp'],-self.records[k]['score'],k))[:self.config.max_anchors]
            self.records = {k:self.records[k] for k in keep}
        self._previous_cameras = {cid:cam.image.copy() for cid,cam in frame.cameras.items()}
        self._previous_frame_id = frame.frame_id
        self.store.commit_count += 1
        self.store.last_timestamp = frame.timestamp
        self.store.execution_signature = frame.execution_signature
        self._pending = None
        result.diagnostics['transaction_status'] = 'committed'
        self.last_commit_diagnostics = dict(transaction_status='committed', commit_total_ms=(perf_counter()-tick)*1000,
            method=self.policy, records=len(self.records))
        return result

    def snapshot(self):
        """Preserve historical provenance; an alternative validation is not perception."""
        frame = self.frame
        inverse = np.linalg.inv(frame.ego_to_world)
        anchors = []
        for k,r in self.records.items():
            if not r['validated']:
                continue
            center = (inverse[:3,:3]@r['anchor'].geometry_world.center+inverse[:3,3]
                      if 'anchor' in r else r['center_ego'])
            anchors.append(CommittedAnchor(k,'object',r['class'],tuple(map(float,center)),(),
                True,tuple(r['current_boxes']),False,False,r['frame_id'],r['timestamp'],str(self.store.commit_count),
                'actual_perception' if r['frame_id']==frame.frame_id else 'experimental_'+self.policy+'_validation',
                r['score'],'experimental_'+self.policy))
        return CommittedWorldState(frame.scene_id,frame.frame_id,frame.timestamp,self.store.generation,
            self.store.commit_count,frame.execution_signature,tuple(anchors),tuple(frame.ego_to_world.reshape(-1)))
