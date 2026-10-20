"""Precheck verifies historical observations; only atomic commit publishes state.

valid_anchor_ids is the compatibility alias for observation support. It is
never an assertion of physical entity identity, stationarity or metric truth.
"""
import copy
from dataclasses import asdict
from time import perf_counter
import numpy as np
from .association import associate, association_ambiguities
from .input_contract import frame_digest, camera_layout
from .physical_certifier import certify_available_state
from .geometry import immutable_array, immutable_geometry, state_geometry_version
from .physical_certifier import EvidenceIssuer, certify_physical_evidence
from .config import M1Config
from .geometry import intersect_area, region_area, region_slices, project_anchor, valid_transform, valid_projection_camera, predict_geometry
from .novelty_detector import unexplained_regions
from .observation_validator import ObservationValidator
from .provenance import export_provenance
from .state_store import StateStore
from .state_types import AnchorObservation, M1Result, WorldAnchor, M1CommitCertification

STATIC_CLASSES = {"barrier", "traffic_cone"}


class WorldStateManager:
    def __init__(self, config=None):
        self.config = M1Config.load(config)
        self.store = StateStore(self.config)
        self.validator = ObservationValidator(self.config)
        self.calibration = None
        if self.config.calibration_path:
            import json
            with open(self.config.calibration_path) as handle:
                self.calibration = json.load(handle)
        self.last_result = None
        self.last_masks = {}
        self.last_commit_diagnostics = {}
        self._last_precheck_timestamp = None
        self._previous_cameras = {}
        self.last_evidence = {}
        self._pending = None
        self._pending_store = None
        self._camera_layout = None
        self._recovery_required = False
        self._previous_frame_id = None
        self._previous_frame_digest = None
        self._surface_history = {}
        self._depth_history = {}
        self._identity_history = ()
        self._identity_hypotheses = ()
        self._computation_receipt = None
        self._committed_log_token = None
        qualification=None
        if self.config.physical_qualification_path:
            import json
            with open(self.config.physical_qualification_path) as handle:qualification=json.load(handle)
        self._physical_issuer=EvidenceIssuer(qualification,self.config.physical_qualification_sha256)

    def reset(self, scene_id=None):
        self.store.reset(scene_id)
        self.last_result = None
        self.last_masks = {}
        self.last_commit_diagnostics = {}
        self._pending = None
        self._pending_store = None
        self._last_precheck_timestamp = None
        self._previous_cameras = {}
        self.last_evidence = {}
        self._camera_layout = None
        self._recovery_required = False
        self._previous_frame_id = None
        self._previous_frame_digest = None
        self._surface_history = {}
        self._depth_history = {}
        self._identity_history = ()
        self._identity_hypotheses = ()
        self._computation_receipt = None
        self._committed_log_token = None

    def _prepare(self, frame):
        s = self.store
        reason = None
        if s.scene_id != frame.scene_id:
            reason = "scene_change"
        elif frame.frame_id in s.committed_frame_ids:
            reason = "duplicate_frame_id"
        elif s.execution_signature is not None and s.execution_signature != frame.execution_signature:
            reason = "execution_conditions_changed"
        elif self._camera_layout is not None and self._camera_layout != camera_layout(frame):
            reason = "camera_layout_changed"
        elif s.last_timestamp is not None or self._last_precheck_timestamp is not None:
            previous = max(t for t in (s.last_timestamp, self._last_precheck_timestamp) if t is not None)
            dt = frame.timestamp - previous
            if dt <= 0:
                reason = "timestamp_reversal_or_repeat"
            elif dt > self.config.max_frame_gap_s:
                reason = "long_frame_gap"
        if not np.isfinite(frame.timestamp) or not valid_transform(frame.ego_to_world):
            reason = "invalid_timestamp_or_pose"
        if not frame.cameras or any(cid != c.camera_id for cid, c in frame.cameras.items()):
            reason = "missing_or_misidentified_cameras"
        if any(not valid_projection_camera(c) or not np.isfinite(c.timestamp)
               or not c.timing_reliable for c in frame.cameras.values()):
            reason = "invalid_camera_calibration_or_timing"
        if not all(isinstance(v, str) and v.strip() for v in
                   (frame.scene_id, frame.frame_id, frame.execution_signature)):
            reason = "missing_frame_identity_or_execution_signature"
        self._input_rejection = reason if reason in {
            "invalid_timestamp_or_pose", "missing_or_misidentified_cameras",
            "invalid_camera_calibration_or_timing", "missing_frame_identity_or_execution_signature",
            "duplicate_frame_id"} else None
        from .source_context import log_binding_reason
        log_reason=log_binding_reason(frame.metadata,frame.scene_id)
        if (s.scene_id==frame.scene_id and self._committed_log_token is not None
                and frame.metadata.get('log_token')!=self._committed_log_token):
            log_reason='log_source_changed_or_missing'
        if log_reason:
            reason=log_reason;self._input_rejection=log_reason
        if self.calibration and (not self.calibration.get('observation_protocol')
                or self.calibration['observation_protocol'] != frame.metadata.get('observation_protocol')):
            reason = "calibration_protocol_incompatible"
        self._rejected_anchor_ids = list(s.anchors) if self._input_rejection else []
        if reason:
            self.reset(frame.scene_id)
        return reason

    def precheck(self, frame):
        return self.precheck_with_observation(frame)

    def precheck_with_observation(self, frame, validate_observation=True):
        """Compute on private metadata; abandonment cannot verify or expire history."""
        wall_start = perf_counter()
        recovery = self._pending is not None or self._recovery_required
        self.abort_pending()
        try:
            digest = frame_digest(frame)
        except Exception:
            self.reset(frame.scene_id)
            raise
        reset_reason = self._prepare(frame)
        # The public store remains the last committed snapshot even while cv2
        # releases the GIL. A private worker owns all provisional metadata.
        worker = copy.copy(self)
        worker._validate_observation = validate_observation
        fork_start=perf_counter()
        worker.store = self.store.fork()
        fork_ms=(perf_counter()-fork_start)*1000
        try:
            from .measurement_cache import measurement_cache
            with measurement_cache(self.config.efficient_state, self.config.measurement_reuse):
                result = worker._precheck(frame, reset_reason, recovery)
        except Exception:
            self.last_result = None
            self.last_masks = {}
            self.last_evidence = {}
            self._recovery_required = True
            raise
        self._pending_store = worker.store
        self.last_result = result
        self.last_masks = worker.last_masks
        self.last_evidence = worker.last_evidence
        self._pending = worker._pending
        self._last_precheck_timestamp = worker._last_precheck_timestamp
        self._pending_digest = digest
        self._pending_started = perf_counter()
        self._pending_generation = self.store.generation
        self._pending_source_digest = self.store.source_digest()
        self._pending_history_digest = self._history_digest()
        self._recovery_required = False
        timings = result.diagnostics.get('timings_ms', {})
        core_ms = timings.get('precheck_total', 0.)
        timings['precheck_total'] = (perf_counter() - wall_start) * 1000
        timings['transaction_and_frame_binding'] = max(0., timings['precheck_total'] - core_ms)
        timings['snapshot_fork'] = fork_ms
        return result

    def _history_digest(self):
        """Bind actual committed camera snapshots used to produce measurements."""
        import hashlib
        from .identity_temporal import camera_snapshot_digest
        import json
        from .state_types import json_value
        identities=[]
        for r in self._identity_history:
            identities.append((r['frame_id'],r['frame_digest'],r['scene_id'],r['timestamp'],r['epoch'],r['execution_signature'],
                r.get('commit_version'),r['camera_sha256'],
                [(cid,camera_snapshot_digest(c)) for cid,c in r['cameras'].items()],
                [(aid,state_geometry_version(a),[(cid,o.timestamp,o.region,hashlib.sha256(o.descriptor.tobytes()).hexdigest())
                    for cid,o in a.observation_support.items()]) for aid,a in r['anchors'].items()],
                r.get('depth_history',{})))
        value=(self._previous_frame_id,self._previous_frame_digest,
               tuple((cid,camera_snapshot_digest(c)) for cid,c in self._previous_cameras.items()),
               json.dumps(json_value((self._depth_history,identities)),sort_keys=True,allow_nan=False))
        return hashlib.sha256(repr(value).encode()).hexdigest()

    def _bound_depth_history(self, cid, frame):
        h=self._depth_history.get(cid)
        if h is None:return None
        if (h['scene_id']!=frame.scene_id or h['epoch']!=self.store.generation
                or h['execution_signature']!=frame.execution_signature
                or h['frame_id']!=self._previous_frame_id or h['frame_digest']!=self._previous_frame_digest
                or h['commit_version']!=self.store.commit_count):
            raise ValueError('Historical depth source binding mismatch')
        return h

    def abort_pending(self):
        """Explicit recovery when model execution fails after precheck."""
        if self._pending is not None and self.last_result is not None:
            self.last_result.diagnostics['transaction_status'] = 'aborted'
            if self.last_result.computation_state is not None:
                self.last_result.computation_state.diagnostics['transaction_status'] = 'aborted'
            self._recovery_required = True
        self._pending = None
        self._pending_store = None

    def _precheck(self, frame, reset_reason=None, recovery=False):
        start = perf_counter()
        if not self.config.enabled:
            result = M1Result(frame.scene_id, frame.frame_id, frame.timestamp,
                              diagnostics={"disabled": True})
            self.last_result = result
            return result
        if self._input_rejection:
            result = M1Result(frame.scene_id, frame.frame_id, frame.timestamp, full_refresh_hint=True,
                diagnostics=dict(reset_reason=reset_reason, input_rejection=self._input_rejection,
                    transaction_status='pending', state_epoch=self.store.generation,
                    timings_ms={'precheck_total': (perf_counter()-start)*1000},
                    no_gpu_computation_skipped=True))
            result.invalid_anchor_ids = list(self._rejected_anchor_ids)
            self.last_masks = {cid:np.ones(c.image.shape[:2],bool) for cid,c in frame.cameras.items()}
            self._pending = (frame.scene_id, frame.frame_id, frame.timestamp, frame.execution_signature)
            self.last_result = result
            return result
        self._pending_execution_signature = frame.execution_signature
        self._last_precheck_timestamp = frame.timestamp
        expired_ids = [k for k, a in self.store.anchors.items()
                       if frame.timestamp - a.source_timestamp > self.config.max_state_age_s]
        self.store.prune(frame.timestamp)
        periodic = self.store.commit_count % self.config.full_refresh_interval == 0
        no_history = not self.store.anchors and not (self.config.local_support and self._previous_cameras)
        result = M1Result(frame.scene_id, frame.frame_id, frame.timestamp,
                          full_refresh_hint=bool(reset_reason or periodic or no_history or recovery))
        if self.config.source_linked_state and result.full_refresh_hint:
            self._validate_observation = False
        refresh_reasons = ([reset_reason] if reset_reason else [])
        refresh_reasons += [name for name, yes in [('periodic', periodic), ('no_history', no_history),
                                                   ('aborted_frame_recovery', recovery)] if yes]
        result.invalid_anchor_ids.extend(expired_ids)
        projections, checks = {}, {}
        projection_start = perf_counter()
        validated_cameras = {cid: valid_projection_camera(cam) for cid, cam in frame.cameras.items()}
        for aid, anchor in self.store.anchors.items():
            projections[aid] = {cid: project_anchor(anchor, cam, self.config, validated_cameras[cid])
                                for cid, cam in frame.cameras.items()}
            if any(p.region is not None for p in projections[aid].values()):
                result.association_candidate_ids.append(aid)
        projection_ms = (perf_counter() - projection_start) * 1000
        occlusions=None
        if self.config.vectorized_occlusion:
            from .bidirectional_geometry import occlusion_table
            occlusions=occlusion_table(projections,frame.cameras,self.config)
        temporal_times = dict(temporal_correspondence=0., visibility_verification=0., geometry_support=0.)
        self.last_evidence = {}
        if self.config.temporal_visibility or self.config.local_support:
            from .temporal_visibility import build_evidence
            def camera_evidence(item):
                cid, camera = item
                # A static-scene epipolar match cannot establish trustworthy object motion.
                # Predicted uncertain dynamic/occluded regions remain unsupported.
                excluded = np.zeros(camera.image.shape[:2], bool)
                anchor_excluded = np.zeros_like(excluded)
                for aid, anchor in self.store.anchors.items():
                    p = projections[aid][cid]
                    if p.region is None: continue
                    dynamic = anchor.anchor_type == 'object' and anchor.semantic_class not in STATIC_CLASSES
                    occluded = occlusions[aid][cid] if occlusions is not None else any(other != aid and q[cid].region is not None
                        and q[cid].depth < p.depth-self.config.occlusion_depth_margin_m
                        and intersect_area(p.region,q[cid].region) > self.config.occlusion_overlap_fraction*region_area(p.region)
                        for other,q in projections.items())
                    if dynamic or occluded:
                        anchor_excluded[region_slices(p.region,camera.image.shape)] = True
                        region = p.region
                        if dynamic and self.config.dense_local_support and self.config.dense_motion_guard:
                            # Rejection-only swept footprint: uncertain velocity never certifies motion.
                            # Pixel uncertainty is not capped by the small acceptance search radius.
                            from .dense_support import swept_dynamic_region
                            region = swept_dynamic_region(anchor, camera, p, self.config)
                        excluded[region_slices(region,camera.image.shape)] = True
                from .measurement_cache import measurement_cache
                # ContextVars do not propagate to thread-pool workers. Each
                # camera owns a bounded readonly measurement scope.
                with measurement_cache(self.config.efficient_state, self.config.measurement_reuse):
                    evidence = build_evidence(self._previous_cameras.get(cid), camera, self.config,
                                              dense_exclusion=excluded,surface_history=self._surface_history.get(cid),
                                              validate_dense=getattr(self, '_validate_observation', True),
                                              depth_history=self._bound_depth_history(cid, frame))
                if self.config.motion_risk_revocation:
                    from .observation_refinement import motion_risk_masks
                    risk,uncertain,details=motion_risk_masks(self.store.anchors.values(),
                        self._previous_cameras.get(cid),camera,evidence,self.config,frame.scene_id,frame.execution_signature)
                    evidence.motion_risk_dirty_mask=risk
                    evidence.motion_risk_unknown_mask=uncertain
                    evidence.diagnostics['motion_risk']=details
                return evidence, excluded, anchor_excluded
            items = list(frame.cameras.items())
            if self.config.temporal_workers > 1 and len(items) > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(self.config.temporal_workers, len(items))) as pool:
                    evidence_list = list(pool.map(camera_evidence, items))
            else:
                evidence_list = [camera_evidence(item) for item in items]
            for (cid, camera), (evidence, excluded, anchor_excluded) in zip(items, evidence_list):
                evidence.observation_exclusion_mask = excluded
                before=int(evidence.stable_mask.sum())
                evidence.stable_mask &= ~excluded
                evidence.changed_mask &= ~excluded
                if len(evidence.current_points):
                    xy = np.rint(evidence.current_points).astype(int)
                    xy[:,0] = np.clip(xy[:,0],0,excluded.shape[1]-1)
                    xy[:,1] = np.clip(xy[:,1],0,excluded.shape[0]-1)
                    evidence.supported &= ~anchor_excluded[xy[:,1],xy[:,0]]
                evidence.diagnostics['supported_points_after_exclusion'] = int(evidence.supported.sum())
                if 'point_measurements' in evidence.diagnostics:
                    evidence.diagnostics['point_measurements']['observation_supported_after_exclusion']=evidence.supported.tolist()
                evidence.diagnostics['excluded_motion_or_occlusion_area_ratio'] = float(excluded.mean())
                # Exclusion is an acceptance gate. Preserve measured rejection
                # evidence separately, without changing legacy identity inputs.
                evidence.diagnostics.setdefault('acceptance_funnel',{})['dynamic_and_occlusion']=dict(
                    total_pixels=excluded.size,standalone_population=excluded.size,alone_rejected=int(excluded.sum()),
                    incremental_rejected=before-int(evidence.stable_mask.sum()),retained=int(evidence.stable_mask.sum()))
                self.last_evidence[cid] = evidence
                for k in temporal_times: temporal_times[k] += evidence.diagnostics.get(k+'_ms',0.)
        if self.config.dense_region_policy=='surface_constraints' and getattr(self, '_validate_observation', True):
            from .surface_visibility_constraints import apply_surface_constraints
            for cid, evidence in self.last_evidence.items():
                apply_surface_constraints(self._previous_cameras.get(cid), frame.cameras[cid],
                    evidence, frame.cameras, self._surface_history.get(cid), self.config)
        validation_start = perf_counter()
        certified = {cid: [] for cid in frame.cameras}
        changed = {cid: [] for cid in frame.cameras}
        for aid, anchor in self.store.anchors.items():
            age = frame.timestamp - anchor.source_timestamp
            horizon = 0 <= age <= self.config.max_state_age_s
            motion_known = (anchor.anchor_type != "object" or anchor.semantic_class in STATIC_CLASSES
                            or anchor.velocity_reliable or anchor.motion_prediction.get("reliable",False))
            camera_checks = {}
            # Every geometrically eligible view must agree; do not select one passing camera
            # while another camera observes a changed/occluded object.
            eligible = []
            for cid, camera in frame.cameras.items():
                p = projections[aid][cid]
                occluded = False
                if occlusions is not None:occluded=occlusions[aid][cid]
                elif p.region is not None:
                    for other, views in projections.items():
                        q = views[cid]
                        if (other != aid and q.region is not None
                                and q.depth < p.depth - self.config.occlusion_depth_margin_m
                                and intersect_area(p.region, q.region) > self.config.occlusion_overlap_fraction * region_area(p.region)):
                            occluded = True; break
                validation = self.validator.validate(anchor.observation_support.get(cid), camera, p, occluded,
                                                     self.last_evidence.get(cid), anchor=anchor,
                                                     previous_camera=self._previous_cameras.get(cid))
                if validation.valid and self.config.bidirectional_geometry:
                    from .bidirectional_geometry import verify_anchor_geometry
                    proof=verify_anchor_geometry(anchor,self._previous_cameras.get(cid),camera,self.last_evidence.get(cid),p,self.config)
                    validation.evidence['bidirectional_geometry']=proof
                    if not proof['passed']:
                        validation.valid=False;validation.reason=proof['reason']
                if self.config.observation_diagnostics:
                    from .observation_diagnostics import attach_observation_funnel
                    attach_observation_funnel(self,anchor,camera,p,occluded,validation)
                if p.region is None:
                    visibility_status='unknown_depth_or_near_plane' if p.reason=='incomplete_depth' else 'unknown_projection'
                elif p.coverage==0:visibility_status='outside_field_of_view'
                elif p.coverage<self.config.min_coverage:visibility_status='partially_outside_field_of_view'
                elif occluded:visibility_status='occluded_hypothesis'
                elif validation.valid:visibility_status='observed_supported'
                else:visibility_status='unknown_observation'
                validation.evidence['visibility_status']=visibility_status
                validation.evidence['absence_proves_disappearance']=False
                camera_checks[cid] = dict(asdict(validation), geometry_valid=p.valid,
                                          projected_points=p.uv,
                                          predicted_region=p.region, coverage=p.coverage,
                                          search_radius_px=p.search_radius_px, depth=p.depth)
                # A tiny/clipped observation is not affirmative evidence; eligible views must
                # have enough geometry. Calibration/timing failure invalidates globally below.
                if p.valid:
                    eligible.append(validation.valid)
                # A mismatched box descriptor is unknown, not localized change evidence.
            calibration_ok = all(p.reason not in ("invalid_calibration", "asynchronous_pose_unknown")
                                 for p in projections[aid].values())
            quality_reasons = []
            if self.config.certification_min_source_score > 0 and (anchor.source_score is None
                    or not np.isfinite(anchor.source_score)
                    or anchor.source_score < self.config.certification_min_source_score):
                quality_reasons.append("source_score_insufficient")
            if anchor.history_observations < self.config.certification_min_history:
                quality_reasons.append("history_insufficient")
            if self.config.certification_max_innovation_m > 0 and (anchor.last_innovation_m is None
                    or anchor.last_innovation_m > self.config.certification_max_innovation_m):
                quality_reasons.append("history_innovation_unverified")
            valid = bool(horizon and motion_known and calibration_ok and eligible and all(eligible)
                         and not quality_reasons)
            reasons = list(quality_reasons)
            if not horizon: reasons.append("outside_horizon")
            if not motion_known: reasons.append("motion_unknown")
            if not calibration_ok: reasons.append("calibration_or_camera_timing_unknown")
            if not eligible: reasons.append("no_reliable_view")
            reasons.extend(v["reason"] for v in camera_checks.values()
                           if v["geometry_valid"] and not v["valid"])
            checks[aid] = {"valid": valid, "age_s": age, "within_horizon": horizon,
                           "anchor_type": anchor.anchor_type,
                           "source_score": anchor.source_score,
                           "history_observations": anchor.history_observations,
                           "last_innovation_m": anchor.last_innovation_m,
                           "motion_known": motion_known, "reasons": sorted(set(reasons)),
                           "cameras": camera_checks}
            checks[aid]['physical_certification'] = certify_available_state(anchor, frame, valid)
            checks[aid]['motion_hypothesis'] = ('static_class_prior' if anchor.semantic_class in STATIC_CLASSES
                                                else 'predicted_velocity' if anchor.velocity_reliable else 'unknown')
            if valid:
                result.valid_anchor_ids.append(aid)
                anchor.last_verified_timestamp = frame.timestamp
                anchor.last_verified_source_frame = anchor.source_frame_id
                for cid, value in camera_checks.items():
                    if value["valid"]:
                        certified[cid].append(value["region"])
            else:
                result.invalid_anchor_ids.append(aid)
        validation_ms = (perf_counter() - validation_start) * 1000
        result.observation_supported_anchor_ids = list(result.valid_anchor_ids)
        # Unknown branch-level dependencies cannot be certified by a handful of local anchors.
        valid_ids = set(result.valid_anchor_ids)
        for fid, record in self.store.features.items():
            if (record.dependency_scope == "local_observation" and record.source_anchor_ids
                    and set(record.source_anchor_ids) <= valid_ids
                    and record.execution_signature == frame.execution_signature
                    and all(self.store.anchors[aid].source_timestamp == record.computation_timestamp
                            and all(checks[aid]['cameras'].get(cid, {}).get('valid', False)
                                    for cid in record.source_camera_ids)
                            for aid in record.source_anchor_ids)):
                record.last_verified_timestamp = frame.timestamp
                result.valid_feature_candidates.append({"feature_id": fid,
                    "computation_version": record.computation_version, "state_epoch": self.store.generation,
                    "evidence": "observation_support_only", "transformer_dependency_certified": False})
        novelty_start = perf_counter()
        self.last_masks = {}
        for cid, camera in frame.cameras.items():
            regions, mask = unexplained_regions(camera, certified[cid], changed[cid], self.config,
                                                 result.full_refresh_hint, self.last_evidence.get(cid))
            regions['full_refresh_reasons'] = list(refresh_reasons)
            result.invalidation_regions[cid], self.last_masks[cid] = regions, mask
            if cid in self.last_evidence:
                observation=regions['verified_stable_area_ratio']*mask.size
                self.last_evidence[cid].diagnostics.setdefault('acceptance_funnel',{})['effective_after_refresh']=dict(
                    total_pixels=mask.size,standalone_population=int(round(observation)),
                    alone_rejected=int(round(observation)) if result.full_refresh_hint else 0,
                    incremental_rejected=int(round(observation))-int((~mask).sum()),retained=int((~mask).sum()))
        novelty_ms = (perf_counter() - novelty_start) * 1000
        result.feature_versions = self.store.feature_versions()
        result.provenance = export_provenance(self.store)
        result.anchor_provenance = {aid: {name: copy.deepcopy(getattr(a, name)) for name in (
            'source_frame_id', 'source_timestamp', 'source_state_epoch', 'source_execution_signature', 'state_version',
            'evidence_origin', 'track_birth_timestamp', 'last_actual_observation_timestamp',
            'last_verified_timestamp', 'last_verified_source_frame', 'last_certified_timestamp',
            'association_count', 'history_observations', 'certified_observation_count',
            'association_history', 'geometry_uncertainty_calibrated', 'association_basis', 'identity_link_qualified')}
            for aid, a in self.store.anchors.items()}
        result.diagnostics = {"reset_reason": reset_reason, "anchors": checks,
            "expired_anchor_ids": expired_ids, "state_epoch": self.store.generation,
            "execution_signature": frame.execution_signature, "execution_context": copy.deepcopy(frame.metadata),
            "model_id": self.config.model_id,
            "thresholds_provisional": True,  # geometry/coverage gates still provisional
            "residual_threshold_calibrated": self.calibration is not None,
            "visibility_requires_independent_support": self.config.require_visibility_evidence and not self.config.temporal_visibility,
            "temporal_evidence": {cid: e.diagnostics for cid,e in self.last_evidence.items()},
            "transformer_dependency_certified": False,
            "timings_ms": {"projection": projection_ms, "observation_validation": validation_ms,
                           "unexplained_regions": novelty_ms, **temporal_times,
                           "precheck_total": (perf_counter() - start) * 1000},
            "tensor_bytes": self.store.tensor_bytes,
            "no_gpu_computation_skipped": True}
        result.diagnostics.update(transaction_status='pending',
            valid_anchor_ids_semantics='observation_support_only_not_physical_identity',
            association_candidate_semantics='historical_projection_only_current_detection_unavailable_until_commit',
            full_refresh_reasons=refresh_reasons)
        if self.config.source_linked_state:
            from .computation_state import build_bundle
            state_start=perf_counter()
            result.computation_state = build_bundle(self, frame, result,
                getattr(self, '_validate_observation', True))
            result.diagnostics['timings_ms']['computation_state_build']=(perf_counter()-state_start)*1000
            result.diagnostics['timings_ms']['precheck_total'] = (perf_counter()-start)*1000
        result.diagnostics['timings_ms'].update(
            dense_flow=sum(e.diagnostics.get('dense_flow_ms',0.) for e in self.last_evidence.values()),
            dense_validation=sum(e.diagnostics.get('dense_validation_ms',0.) for e in self.last_evidence.values()))
        if not self.config.profiling:
            result.diagnostics['timings_ms'] = {'precheck_total': result.diagnostics['timings_ms']['precheck_total']}
        self.last_result = result
        self._pending = (frame.scene_id, frame.frame_id, frame.timestamp, frame.execution_signature)
        return result

    def initialize(self, frame, full_outputs):
        self.reset(frame.scene_id)
        result = self.precheck(frame)
        return self.commit(frame, full_outputs, result)

    def commit(self, frame, full_outputs, precheck_result):
        """Publish only after all descriptors, tensors and provenance are constructed."""
        wall_start = perf_counter()
        if not self.config.enabled:
            return precheck_result
        identity = (frame.scene_id, frame.frame_id, frame.timestamp, frame.execution_signature)
        if (self._pending != identity or precheck_result is not self.last_result
                or self._pending_store is None):
            raise ValueError("commit requires this frame's pending precheck; repeated/stale commits rejected")
        source_validation_in_progress = False
        try:
            if self.store.generation != self._pending_generation:
                source_validation_in_progress = True
                raise ValueError('Pending transaction epoch changed')
            if (perf_counter() - self._pending_started > self.config.pending_timeout_s
                    or frame_digest(frame) != self._pending_digest):
                raise ValueError('Pending transaction expired or frame/calibration changed')
            source_validation_in_progress = True
            if (self.store.source_digest()!=self._pending_source_digest
                    or self._history_digest()!=self._pending_history_digest):
                raise ValueError('Committed source state changed during pending transaction')
            source_validation_in_progress = False
            if full_outputs.source not in ('actual_perception', 'historical_full_model_detection_replay',
                                           'invalid_pose_no_state_committed'):
                raise ValueError('Unsupported online computation source')
            receipt_start=perf_counter()
            receipt = copy.deepcopy(full_outputs.execution_receipt)
            receipt_ms=(perf_counter()-receipt_start)*1000
            worker = copy.copy(self)
            fork_start=perf_counter()
            worker.store = self._pending_store.fork()
            fork_ms=(perf_counter()-fork_start)*1000
            from .provenance import feature_source_snapshot, preserve_camera_computation_source
            previous_backbone = feature_source_snapshot(self.store.features.get('backbone'))
            from .measurement_cache import measurement_cache
            with measurement_cache(self.config.efficient_state, self.config.measurement_reuse):
                result = worker._commit(frame, full_outputs, precheck_result)
            if full_outputs.execution_receipt and full_outputs.execution_receipt.get('actual_reused_cameras'):
                worker.last_commit_diagnostics['backbone_computation_source'] = preserve_camera_computation_source(
                    worker.store, frame, full_outputs.execution_receipt, previous_backbone)
        except Exception:
            self.abort_pending()
            if source_validation_in_progress:
                # The public snapshot was modified externally; it is no longer
                # a trusted rollback target. Discard it and every image source.
                self.reset(frame.scene_id)
                precheck_result.diagnostics['source_state_discarded'] = True
                self.last_result = precheck_result
                self.last_commit_diagnostics = dict(transaction_status='aborted',
                    rejection_reason='committed_source_changed', source_state_discarded=True)
            self._recovery_required = True
            precheck_result.diagnostics['transaction_status'] = 'aborted'
            raise
        # One publication point: no partially written store is ever exposed.
        self.store = worker.store
        self._previous_cameras = worker._previous_cameras
        self._previous_frame_id = worker._previous_frame_id
        self._previous_frame_digest = worker._previous_frame_digest
        self._surface_history = worker._surface_history
        self._depth_history = worker._depth_history
        self._identity_history = worker._identity_history
        self._identity_hypotheses = worker._identity_hypotheses
        self.last_commit_diagnostics = worker.last_commit_diagnostics
        self.last_commit_diagnostics.update(snapshot_fork_ms=fork_ms,cache_provenance_receipt_ms=receipt_ms)
        self._computation_receipt = receipt
        self._pending = None
        self._pending_store = None
        self._camera_layout = camera_layout(frame)
        self._committed_log_token = worker._committed_log_token
        precheck_result.diagnostics['transaction_status'] = ('rejected' if self._input_rejection else 'committed')
        if precheck_result.computation_state is not None:
            precheck_result.computation_state.diagnostics['transaction_status'] = 'committed'
        core_ms = self.last_commit_diagnostics.get('commit_total_ms', 0.)
        self.last_commit_diagnostics['commit_total_ms'] = (perf_counter() - wall_start) * 1000
        self.last_commit_diagnostics['transaction_and_frame_binding_ms'] = max(
            0., self.last_commit_diagnostics['commit_total_ms'] - core_ms)
        return result

    def _commit(self, frame, full_outputs, precheck_result):
        start = perf_counter()
        if not self.config.enabled:
            return precheck_result
        identity = (frame.scene_id, frame.frame_id, frame.timestamp, frame.execution_signature)
        if (getattr(self, "_pending", None) != identity or precheck_result is not self.last_result):
            raise ValueError("commit requires this frame's pending precheck; repeated/stale commits rejected")
        self._pending = None
        if self._input_rejection:
            self.reset(frame.scene_id)
            self.last_result = precheck_result
            self.last_commit_diagnostics = dict(transaction_status='rejected',
                input_rejection=self._input_rejection, committed_anchor_ids=[],
                post_perception_physical_certifications=[], commit_total_ms=(perf_counter()-start)*1000)
            return precheck_result
        anchors = list(self.store.anchors.values())
        candidates = []
        malformed = 0
        for index, pred in enumerate(full_outputs.anchors):
            g = pred.geometry_world
            if (pred.anchor_type not in ("object", "map", "region") or g.points.ndim != 2
                    or g.points.shape[1] != 3 or len(g.points) < 2
                    or np.asarray(g.center).shape != (3,)
                    or not np.isfinite(g.points).all() or not np.isfinite(g.center).all()):
                malformed += 1
                continue
            if (not np.isfinite(pred.geometry_uncertainty) or pred.geometry_uncertainty < 0
                    or (g.size is not None and (np.asarray(g.size).shape != (3,)
                        or not np.isfinite(g.size).all() or np.any(g.size <= 0)))):
                malformed += 1
                continue
            candidates.append((index, pred))
        def selection_key(item):
            _, pred = item
            score = pred.source_score
            score = float(score) if score is not None and np.isfinite(score) else -float('inf')
            # Score ranks capacity only; it is not a localization probability.
            return (-score, pred.anchor_type, pred.semantic_class,
                    tuple(pred.geometry_world.center), tuple(pred.geometry_world.points.ravel()),
                    repr((pred.velocity_world, pred.velocity_reliable, pred.geometry_uncertainty)))
        selected = sorted(candidates, key=selection_key)[:self.config.max_anchors]
        predictions = [pred for _, pred in selected]
        assoc_start = perf_counter()
        pairs, costs = associate(anchors, predictions, frame.timestamp, self.config)
        ambiguities = association_ambiguities(pairs, costs, self.config.association_ambiguity_margin_m)
        accepted_pairs = []
        rejected_pairs = []
        for i, j in pairs:
            reasons = []
            if ambiguities[i, j]['ambiguous']:
                reasons.append('competing_geometric_candidates')
            if (anchors[i].anchor_type == 'object' and self.store.last_timestamp is not None
                    and anchors[i].source_timestamp < self.store.last_timestamp):
                reasons.append('reappearance_after_missing_observation')
            if reasons:
                rejected_pairs.append(dict(anchor_id=anchors[i].anchor_id, prediction_index=selected[j][0],
                    reasons=reasons, **ambiguities[i, j]))
                anchors[i].association_history = (anchors[i].association_history + [dict(
                    frame_id=frame.frame_id, timestamp=frame.timestamp, status='ambiguous_or_missing',
                    reasons=reasons, certified=False)])[-self.config.association_history_limit:]
            else:
                accepted_pairs.append((i, j))
        geometric_pairs=list(accepted_pairs)
        temporal_proposals={};proposal_certificates={}
        identity_protocol=None
        if self.config.identity_continuity and self.config.identity_method.startswith('temporal'):
            from .identity_temporal import temporal_identity_pairs
            from .identity_continuity import identity_protocol_digest,IDENTITY_CLAIM
            identity_protocol=identity_protocol_digest(self.config,frame)
            temporal_proposals=temporal_identity_pairs(anchors,predictions,costs,frame,self._identity_history,
                self.config,self.validator,self._physical_issuer,self.last_evidence,self._previous_frame_id,self._previous_frame_digest)
            # A proposal is a relation between actual observations. Check its
            # named claim before deciding whether it may affect ordinary IDs.
            for (i,j),proposal in temporal_proposals.items():
                pred=predictions[j];old=anchors[i]
                velocity_ok=(pred.velocity_world is not None and np.asarray(pred.velocity_world).shape==(3,)
                             and np.isfinite(pred.velocity_world).all())
                target=WorldAnchor(old.anchor_id,frame.scene_id,pred.anchor_type,pred.semantic_class,
                    immutable_geometry(pred.geometry_world),frame.timestamp,
                    velocity_world=immutable_array(pred.velocity_world) if velocity_ok else None,
                    velocity_reliable=bool(pred.velocity_reliable and velocity_ok),geometry_uncertainty=pred.geometry_uncertainty,
                    source_frame_id=frame.frame_id,source_state_epoch=old.source_state_epoch,
                    source_execution_signature=frame.execution_signature)
                proposal_certificates[i,j]=certify_physical_evidence(old,target,frame,proposal['evidence'],self._physical_issuer,
                    proposal.get('source_frame_digest',self._previous_frame_digest),identity_protocol,claim=IDENTITY_CLAIM)
            allowed=np.full_like(costs,np.inf)
            for edge,p in temporal_proposals.items():
                if p['status']=='candidate' and (self.config.identity_tracking_policy=='experimental'
                        or (self.config.identity_tracking_policy=='certified_only' and proposal_certificates[edge].certified)):
                    allowed[edge]=costs[edge]
            from .association import legal_assignment
            measured=legal_assignment(allowed)
            measured_rows={i for i,j in measured};measured_cols={j for i,j in measured}
            accepted_pairs=measured+[(i,j) for i,j in accepted_pairs if i not in measured_rows and j not in measured_cols]
        if (len({i for i,j in accepted_pairs})!=len(accepted_pairs)
                or len({j for i,j in accepted_pairs})!=len(accepted_pairs)):
            raise ValueError('Association must be one-to-one before updating state')
        assoc_ms = (perf_counter() - assoc_start) * 1000
        matched = {j: anchors[i].anchor_id for i, j in accepted_pairs}
        committed = []
        physical_results=[]
        foreground_results=[]
        identity_results=[]
        for (i,j),proposal in temporal_proposals.items():
            identity_results.append(dict(anchor_id=anchors[i].anchor_id,prediction_index=selected[j][0],
                source_frame_id=anchors[i].source_frame_id,current_frame_id=frame.frame_id,
                source_state_version=state_geometry_version(anchors[i]),source_epoch=anchors[i].source_state_epoch,
                tracking_policy=self.config.identity_tracking_policy,
                applied_to_tracking=(i,j) in accepted_pairs and (i,j) not in geometric_pairs,
                certified=proposal_certificates[i,j].certified,
                qualification_reasons=list(proposal_certificates[i,j].rejection_reasons),
                protocol_digest=identity_protocol,
                **{k:v for k,v in proposal.items() if k!='evidence'}))
        certification_ms=0.
        if self.config.identity_continuity:
            from .identity_continuity import identity_protocol_digest
            identity_protocol=identity_protocol_digest(self.config,frame)
        validated_cameras = {cid: valid_projection_camera(cam) for cid, cam in frame.cameras.items()}
        for j, pred in enumerate(predictions):
            aid = matched.get(j) or self.store.new_anchor_id()
            prior = self.store.anchors.get(aid)
            velocity_ok = (pred.velocity_world is not None and np.asarray(pred.velocity_world).shape == (3,)
                           and np.isfinite(pred.velocity_world).all())
            a = WorldAnchor(aid, frame.scene_id, pred.anchor_type, pred.semantic_class,
                            immutable_geometry(pred.geometry_world), frame.timestamp,
                            prior.last_verified_timestamp if prior else None,
                            immutable_array(pred.velocity_world) if velocity_ok else None,
                            bool(pred.velocity_reliable and velocity_ok and not self.config.dynamic_motion_prediction), pred.geometry_uncertainty)
            a.source_score = pred.source_score
            a.geometry_uncertainty_calibrated = pred.geometry_uncertainty_calibrated
            a.source_frame_id = frame.frame_id
            a.source_state_epoch = self.store.generation
            a.source_execution_signature = frame.execution_signature
            a.evidence_origin = full_outputs.source
            a.history_observations = prior.history_observations + 1 if prior else 1
            a.track_birth_timestamp = prior.track_birth_timestamp if prior else frame.timestamp
            a.last_actual_observation_timestamp = frame.timestamp
            # A new detection is a new concrete state version. A historical
            # certificate cannot attach to replacement geometry automatically.
            a.last_certified_timestamp = None
            a.last_verified_source_frame = prior.last_verified_source_frame if prior else None
            a.certified_observation_count = prior.certified_observation_count if prior else 0
            if prior is not None:
                edge=(next(i for i,old in enumerate(anchors) if old.anchor_id==aid),j)
                certificate=proposal_certificates.get(edge)
                a.association_basis=('qualified_identity' if certificate is not None and certificate.certified else
                    'experimental_identity_hypothesis' if edge not in geometric_pairs else 'unqualified_geometric_track')
                a.identity_link_qualified=bool(certificate is not None and certificate.certified)
                if a.association_basis=='experimental_identity_hypothesis':
                    a.last_verified_timestamp=None;a.last_verified_source_frame=None
                    a.certified_observation_count=0;a.history_observations=1
            a.association_count = prior.association_count + 1 if prior else 0
            a.association_history = ((prior.association_history if prior else []) + [dict(
                frame_id=frame.frame_id, timestamp=frame.timestamp, source_epoch=self.store.generation,
                prediction_index=selected[j][0], status=a.association_basis if prior else 'new_observation',
                certified=a.identity_link_qualified,
                source_state_version=state_geometry_version(prior) if prior else None)])[-self.config.association_history_limit:]
            a.last_innovation_m = (float(np.linalg.norm(
                pred.geometry_world.center - predict_geometry(prior, frame.timestamp).center))
                if prior else None)
            a.state_version=state_geometry_version(a)
            if prior is not None and self.config.foreground_correspondence:
                from .foreground_correspondence import propose_foreground_correspondence
                proposal=propose_foreground_correspondence(prior,a,frame,self._previous_cameras,
                    self.last_evidence,self.config,self._physical_issuer,self._previous_frame_digest,
                    self._previous_frame_id, association_alternatives=int(np.isfinite(np.r_[
                        np.delete(costs[next(i for i,old in enumerate(anchors) if old.anchor_id==aid)],j),
                        np.delete(costs[:,j],next(i for i,old in enumerate(anchors) if old.anchor_id==aid))]).sum()))
                certification_start=perf_counter()
                certificate=certify_physical_evidence(prior,a,frame,proposal['evidence'],self._physical_issuer,
                    self._previous_frame_digest,proposal['protocol_digest'])
                certification_ms+=(perf_counter()-certification_start)*1000
                physical_results.append(asdict(certificate))
                foreground_results.append(dict(anchor_id=aid,source_frame_id=prior.source_frame_id,
                    current_frame_id=frame.frame_id,**{k:v for k,v in proposal.items() if k!='evidence'}))
                if certificate.certified:
                    a.physical_certification=certificate
                    a.last_certified_timestamp=frame.timestamp
                    a.certified_observation_count+=1
            if (prior is not None and self.config.identity_continuity
                    and (not self.config.identity_method.startswith('temporal') or prior.anchor_type=='object')):
                from .identity_continuity import propose_identity_continuity, IDENTITY_CLAIM
                i=next(i for i,old in enumerate(anchors) if old.anchor_id==aid)
                competing=np.r_[np.delete(costs[i],j),np.delete(costs[:,j],i)]
                identity=(temporal_proposals[i,j] if self.config.identity_method.startswith('temporal') else
                    propose_identity_continuity(prior,a,frame,self.last_evidence,self.config,self.validator,
                    self._physical_issuer,self._previous_frame_digest,self._previous_frame_id,competing))
                source_digest=identity.get('source_frame_digest',self._previous_frame_digest)
                certification_start=perf_counter()
                certificate=(proposal_certificates[i,j] if self.config.identity_method.startswith('temporal') else
                    certify_physical_evidence(prior,a,frame,identity['evidence'],self._physical_issuer,
                    source_digest,identity_protocol,claim=IDENTITY_CLAIM))
                certification_ms+=(perf_counter()-certification_start)*1000
                if self.config.identity_method.startswith('temporal'):
                    row=next(r for r in identity_results if r['anchor_id']==aid and r['prediction_index']==selected[j][0])
                    row.update(protocol_digest=identity_protocol,qualification_reasons=list(certificate.rejection_reasons))
                else:
                    identity_results.append(dict(anchor_id=aid,source_frame_id=prior.source_frame_id,current_frame_id=frame.frame_id,
                        **{k:v for k,v in identity.items() if k!='evidence'},protocol_digest=identity_protocol))
                physical_results.append(asdict(certificate))
                if certificate.certified:
                    a.physical_certification=certificate;a.last_certified_timestamp=frame.timestamp
                    a.certified_observation_count+=1
                    a.identity_link_qualified=True;a.association_basis='qualified_identity'
                    a.association_history[-1].update(status='qualified_identity',certified=True)
            if self.config.dynamic_motion_prediction:
                from .world_motion import update_world_motion
                update_world_motion(prior,a,self.config)
            # New support always comes from an actually computed frame, never chained crops.
            for cid, camera in frame.cameras.items():
                p = project_anchor(a, camera, self.config, validated_cameras[cid])
                if p.valid:
                    descriptor = self.validator.descriptor(camera.image, p.region)
                    if descriptor is not None:
                        a.observation_support[cid] = AnchorObservation(cid, p.region, immutable_array(descriptor),
                                                                      camera.timestamp, p.coverage)
            self.store.anchors[aid] = a
            committed.append(aid)
        for row in identity_results:
            if 'prediction_index' in row:
                j=next(j for j,(raw,_) in enumerate(selected) if raw==row['prediction_index'])
                row['target_anchor_id']=committed[j]
                row['target_state_version']=state_geometry_version(self.store.anchors[committed[j]])
        self.store.prune(frame.timestamp)
        # Per-anchor observation records have known crop provenance; branch records remain
        # conservative. Only computed feature slots advance their own versions.
        from .state_types import ComputedFeature
        computed = []
        for aid in committed:
            a = self.store.anchors.get(aid)
            if a is None: continue
            for cid, support in a.observation_support.items():
                fid = aid + ":observation:" + cid
                computed.append(ComputedFeature(fid, "observation_descriptor", support.descriptor.shape,
                    str(support.descriptor.dtype), "cpu", "raw_observation/" + self.config.descriptor,
                    support.descriptor, [cid], {cid: [support.region]}, "local_observation", [aid]))
                a.associated_feature_ids.append(fid)
        for feature in full_outputs.features:
            feature = copy.copy(feature)
            if not feature.source_anchor_ids:
                kind = "map" if feature.feature_kind == "map_carrier" else "object"
                feature.source_anchor_ids = [aid for aid in committed if aid in self.store.anchors
                    and (feature.feature_kind == "backbone" or self.store.anchors[aid].anchor_type == kind)]
            computed.append(feature)
            for aid in feature.source_anchor_ids:
                if aid in self.store.anchors:
                    self.store.anchors[aid].associated_feature_ids.append(feature.feature_id)
        for feature in computed:
            self.store.put_feature(feature, frame)
        # This edge is directly visible in Petr3D's call graph. All other transformer
        # dependencies (including temporal memories) remain explicitly conservative.
        computed_ids = {f.feature_id for f in computed}
        if 'backbone' in computed_ids:
            for fid in ('object_carrier', 'map_carrier'):
                if fid in computed_ids:
                    self.store.provenance[fid].upstream_versions['backbone'] = self.store.features['backbone'].computation_version
        # A branch slot can be replaced while an unmatched old anchor is retained briefly.
        # Do not leave that anchor pointing to a newer feature with different provenance.
        for aid, anchor in self.store.anchors.items():
            anchor.associated_feature_ids = [fid for fid in anchor.associated_feature_ids
                if fid in self.store.features and aid in self.store.features[fid].source_anchor_ids]
        self._committed_log_token = frame.metadata.get('log_token')
        self.store.last_timestamp = frame.timestamp
        self.store.execution_signature = frame.execution_signature
        self.store.commit_count += 1
        self.store.committed_frame_ids = self.store.committed_frame_ids | {frame.frame_id}
        history_start=perf_counter()
        self._surface_history = {}
        if self.config.dense_region_policy in ('surface_visibility','surface_local_plane','surface_consensus','surface_constraints'):
            for cid,e in self.last_evidence.items():
                old=self._previous_cameras.get(cid)
                if old is not None and e.dense_source_map is not None:
                    self._surface_history[cid]=(copy.copy(old),frame.cameras[cid].timestamp,immutable_array(e.dense_source_map))
        old_cameras=self._previous_cameras
        if self.config.temporal_visibility or self.config.local_support:
            # Own only one bounded grayscale frame per camera, from actual computation.
            # precheck never replaces this source with an uncomputed/verified frame.
            from .temporal_visibility import gray_image
            self._previous_cameras = {}
            for cid, camera in frame.cameras.items():
                saved = copy.copy(camera)
                saved.image = immutable_array(gray_image(camera))
                saved.intrinsic = immutable_array(camera.intrinsic)
                saved.raw_intrinsic = immutable_array(camera.raw_intrinsic)
                saved.raw_to_image = immutable_array(camera.raw_to_image)
                saved.world_to_camera = immutable_array(camera.world_to_camera)
                saved.visibility_mask = None; saved.geometry_support_mask = None
                self._previous_cameras[cid] = saved
        self._depth_history = {}
        if self.config.ego_motion_mode == 'depth':
            from .ego_motion import measured_depth_history
            for cid,e in self.last_evidence.items():
                # Current actual detections are available only at Commit. Newly
                # visible moving objects must not enter NEXT frame's static-ray
                # search depth, even when absent from historical projections.
                dynamic_mask=np.zeros(frame.cameras[cid].image.shape[:2],bool)
                for aid in committed:
                    anchor=self.store.anchors.get(aid)
                    if anchor is None or anchor.anchor_type!='object' or anchor.semantic_class in STATIC_CLASSES:continue
                    projection=project_anchor(anchor,frame.cameras[cid],self.config)
                    if projection.region is not None:
                        dynamic_mask[region_slices(projection.region,dynamic_mask.shape)]=True
                self._depth_history[cid]=measured_depth_history(old_cameras.get(cid),self._previous_cameras[cid],e,
                    self.config,dict(scene_id=frame.scene_id,frame_id=frame.frame_id,frame_digest=self._pending_digest,
                        epoch=self.store.generation,execution_signature=frame.execution_signature,
                        commit_version=self.store.commit_count),additional_exclusion=dynamic_mask)
        if self.config.identity_continuity and self.config.identity_method.startswith('temporal'):
            # Only actual successful commits enter this bounded tuple. A failed
            # transaction cannot append an image or advance a source timestamp.
            from .identity_temporal import camera_snapshot_digest
            record=dict(frame_id=frame.frame_id,frame_digest=self._pending_digest,scene_id=frame.scene_id,
                timestamp=frame.timestamp,epoch=self.store.generation,execution_signature=frame.execution_signature,
                cameras={cid:copy.copy(c) for cid,c in self._previous_cameras.items()},
                camera_sha256={cid:camera_snapshot_digest(c) for cid,c in self._previous_cameras.items()},
                commit_version=self.store.commit_count,depth_history=copy.deepcopy(self._depth_history),
                anchors={aid:copy.deepcopy(self.store.anchors[aid]) for aid in committed if aid in self.store.anchors})
            self._identity_history=tuple(r for r in self._identity_history
                if 0<frame.timestamp-r['timestamp']<=min(self.config.max_state_age_s,1.))+(record,)
            self._identity_history=self._identity_history[-self.config.identity_history_frames:]
        self._previous_frame_id=frame.frame_id
        self._previous_frame_digest=self._pending_digest
        self._identity_hypotheses=tuple(copy.deepcopy(identity_results))
        self.last_commit_diagnostics = {"source": full_outputs.source, "committed_anchor_ids": committed,
            "association_pairs": [[anchors[i].anchor_id, selected[j][0]] for i, j in accepted_pairs],
            "geometric_candidate_pairs": [[anchors[i].anchor_id, selected[j][0]] for i, j in pairs],
            "rejected_ambiguous_associations": rejected_pairs,
            "selected_prediction_indices": [index for index, _ in selected],
            "excluded_prediction_count": len(candidates) - len(selected),
            "malformed_prediction_count": malformed,
            "association_is_identity_certification": False,
            "geometric_tracking_pairs": [[anchors[i].anchor_id,selected[j][0]] for i,j in geometric_pairs],
            "identity_tracking_policy": self.config.identity_tracking_policy,
            "post_perception_physical_certifications": physical_results,
            "foreground_correspondence_proposals": foreground_results,
            "identity_continuity_proposals": identity_results,
            "physical_certification_phase": "post_perception_commit_only",
            "physical_certification_ms": certification_ms,
            "association_ms": assoc_ms, "state_update_ms": (perf_counter() - start) * 1000 - assoc_ms,
            "commit_history_copy_ms": (perf_counter()-history_start)*1000,
            "commit_total_ms": (perf_counter() - start) * 1000,
            "anchor_count": len(self.store.anchors), "feature_count": len(self.store.features),
            "tensor_bytes": self.store.tensor_bytes}
        if not self.config.profiling:
            self.last_commit_diagnostics.pop('association_ms')
            self.last_commit_diagnostics.pop('state_update_ms')
        # Keep precheck_result frozen at the historical snapshot for M2; current versions are
        # available in store.feature_versions(), not silently substituted into old candidates.
        return precheck_result

    def export_m2_inputs(self):
        if self.last_result is None or self.last_result.diagnostics.get('transaction_status') == 'aborted':
            raise RuntimeError("Call precheck first")
        return copy.deepcopy(self.last_result)

    def export_state_audit(self):
        """Current committed lifecycle; precheck observations remain separately scoped."""
        from .provenance import export_state_audit
        return export_state_audit(self)

    def get_identity_hypotheses(self):
        """Audit relations from the last atomic commit; ordinary IDs are not proof."""
        return copy.deepcopy(self._identity_hypotheses)

    def get_current_certifications(self):
        """Read the last atomically committed state, never pending detections.

        The frozen bundle and defensive certificate copies cannot mutate Store.
        Identity continuity does not authorize pixels or metric-state consumers.
        """
        from .physical_certifier import CONTINUITY_CLAIM
        from .identity_continuity import IDENTITY_CLAIM
        certificates=[]
        for a in self.store.anchors.values():
            c=a.physical_certification
            if (c is None or not c.certified or c.certified_claim not in (CONTINUITY_CLAIM,IDENTITY_CLAIM)
                    or c.anchor_id!=a.anchor_id or c.scene_id!=self.store.scene_id
                    or c.current_frame_id!=self._previous_frame_id or a.source_frame_id!=self._previous_frame_id
                    or c.timestamp!=self.store.last_timestamp or a.source_timestamp!=self.store.last_timestamp
                    or c.state_epoch!=self.store.generation or a.source_state_epoch!=self.store.generation
                    or a.source_execution_signature!=self.store.execution_signature
                    or c.current_state_version!=state_geometry_version(a) or a.state_version!=c.current_state_version):
                continue
            certificates.append(copy.deepcopy(c))
        rejected=tuple(copy.deepcopy(r) for r in self.last_commit_diagnostics.get('post_perception_physical_certifications',[])
            if not r['certified'] and r.get('current_frame_id')==self._previous_frame_id)
        return M1CommitCertification(self.store.scene_id,self._previous_frame_id,self.store.last_timestamp,
            tuple(c.anchor_id for c in certificates),tuple(certificates),rejected,self.store.generation)

    def get_candidate_feature(self, feature_id, computation_version):
        """Read before commit; a replaced slot cannot masquerade as an older version."""
        if self._pending is None or self.last_result is None or not any(c['feature_id'] == feature_id
                                              for c in self.last_result.valid_feature_candidates):
            raise KeyError('Not a verified candidate')
        record = self.store.features.get(feature_id)
        if record is None or record.computation_version != computation_version:
            raise KeyError('Historical feature version has been replaced; acquire before commit')
        return self.store.feature_snapshot(record)
