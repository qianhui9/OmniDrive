"""Precheck certifies historical evidence; commit alone consumes new model outputs."""
import copy
from dataclasses import asdict
from time import perf_counter
import numpy as np
from .association import associate
from .config import M1Config
from .geometry import intersect_area, region_area, region_slices, project_anchor, valid_transform, valid_projection_camera, predict_geometry
from .novelty_detector import unexplained_regions
from .observation_validator import ObservationValidator
from .provenance import export_provenance
from .state_store import StateStore
from .state_types import AnchorObservation, M1Result, WorldAnchor

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

    def reset(self, scene_id=None):
        self.store.reset(scene_id)
        self.last_result = None
        self.last_masks = {}
        self._pending = None
        self._last_precheck_timestamp = None
        self._previous_cameras = {}
        self.last_evidence = {}

    def _prepare(self, frame):
        s = self.store
        reason = None
        if s.scene_id != frame.scene_id:
            reason = "scene_change"
        elif s.execution_signature is not None and s.execution_signature != frame.execution_signature:
            reason = "execution_conditions_changed"
        elif s.last_timestamp is not None or self._last_precheck_timestamp is not None:
            previous = max(t for t in (s.last_timestamp, self._last_precheck_timestamp) if t is not None)
            dt = frame.timestamp - previous
            if dt <= 0:
                reason = "timestamp_reversal_or_repeat"
            elif dt > self.config.max_frame_gap_s:
                reason = "long_frame_gap"
        if not np.isfinite(frame.timestamp) or not valid_transform(frame.ego_to_world):
            reason = "invalid_timestamp_or_pose"
        if self.calibration and (not self.calibration.get('observation_protocol')
                or self.calibration['observation_protocol'] != frame.metadata.get('observation_protocol')):
            reason = "calibration_protocol_incompatible"
        if reason:
            self.reset(frame.scene_id)
        return reason

    def precheck(self, frame):
        start = perf_counter()
        if not self.config.enabled:
            result = M1Result(frame.scene_id, frame.frame_id, frame.timestamp,
                              diagnostics={"disabled": True})
            self.last_result = result
            return result
        reset_reason = self._prepare(frame)
        self._last_precheck_timestamp = frame.timestamp
        expired_ids = [k for k, a in self.store.anchors.items()
                       if frame.timestamp - a.source_timestamp > self.config.max_state_age_s]
        self.store.prune(frame.timestamp)
        periodic = self.store.commit_count % self.config.full_refresh_interval == 0
        no_history = not self.store.anchors and not (self.config.local_support and self._previous_cameras)
        result = M1Result(frame.scene_id, frame.frame_id, frame.timestamp,
                          full_refresh_hint=bool(reset_reason or periodic or no_history))
        result.invalid_anchor_ids.extend(expired_ids)
        projections, checks = {}, {}
        projection_start = perf_counter()
        validated_cameras = {cid: valid_projection_camera(cam) for cid, cam in frame.cameras.items()}
        for aid, anchor in self.store.anchors.items():
            projections[aid] = {cid: project_anchor(anchor, cam, self.config, validated_cameras[cid])
                                for cid, cam in frame.cameras.items()}
        projection_ms = (perf_counter() - projection_start) * 1000
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
                    occluded = any(other != aid and q[cid].region is not None
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
                evidence = build_evidence(self._previous_cameras.get(cid), camera, self.config,
                                          dense_exclusion=excluded)
                return evidence, excluded, anchor_excluded
            items = list(frame.cameras.items())
            if self.config.temporal_workers > 1 and len(items) > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(self.config.temporal_workers, len(items))) as pool:
                    evidence_list = list(pool.map(camera_evidence, items))
            else:
                evidence_list = [camera_evidence(item) for item in items]
            for (cid, camera), (evidence, excluded, anchor_excluded) in zip(items, evidence_list):
                evidence.stable_mask &= ~excluded
                evidence.changed_mask &= ~excluded
                if len(evidence.current_points):
                    xy = np.rint(evidence.current_points).astype(int)
                    xy[:,0] = np.clip(xy[:,0],0,excluded.shape[1]-1)
                    xy[:,1] = np.clip(xy[:,1],0,excluded.shape[0]-1)
                    evidence.supported &= ~anchor_excluded[xy[:,1],xy[:,0]]
                evidence.diagnostics['excluded_motion_or_occlusion_area_ratio'] = float(excluded.mean())
                self.last_evidence[cid] = evidence
                for k in temporal_times: temporal_times[k] += evidence.diagnostics.get(k+'_ms',0.)
        validation_start = perf_counter()
        certified = {cid: [] for cid in frame.cameras}
        changed = {cid: [] for cid in frame.cameras}
        for aid, anchor in self.store.anchors.items():
            age = frame.timestamp - anchor.source_timestamp
            horizon = 0 <= age <= self.config.max_state_age_s
            motion_known = (anchor.anchor_type != "object" or anchor.semantic_class in STATIC_CLASSES
                            or anchor.velocity_reliable)
            camera_checks = {}
            # Every geometrically eligible view must agree; do not select one passing camera
            # while another camera observes a changed/occluded object.
            eligible = []
            for cid, camera in frame.cameras.items():
                p = projections[aid][cid]
                occluded = False
                if p.region is not None:
                    for other, views in projections.items():
                        q = views[cid]
                        if (other != aid and q.region is not None
                                and q.depth < p.depth - self.config.occlusion_depth_margin_m
                                and intersect_area(p.region, q.region) > self.config.occlusion_overlap_fraction * region_area(p.region)):
                            occluded = True; break
                validation = self.validator.validate(anchor.observation_support.get(cid), camera, p, occluded,
                                                     self.last_evidence.get(cid))
                camera_checks[cid] = dict(asdict(validation), geometry_valid=p.valid,
                                          projected_points=p.uv,
                                          predicted_region=p.region, coverage=p.coverage,
                                          search_radius_px=p.search_radius_px, depth=p.depth)
                # A tiny/clipped observation is not affirmative evidence; eligible views must
                # have enough geometry. Calibration/timing failure invalidates globally below.
                if p.valid:
                    eligible.append(validation.valid)
                if p.region is not None and np.isfinite(validation.residual) and validation.residual > self.config.novelty_threshold:
                    changed[cid].append(p.region)
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
            if valid:
                result.valid_anchor_ids.append(aid)
                anchor.last_verified_timestamp = frame.timestamp
                for cid, value in camera_checks.items():
                    if value["valid"]:
                        certified[cid].append(value["region"])
            else:
                result.invalid_anchor_ids.append(aid)
        validation_ms = (perf_counter() - validation_start) * 1000
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
                    "evidence": "physical_and_observation_only", "transformer_dependency_certified": False})
        novelty_start = perf_counter()
        self.last_masks = {}
        for cid, camera in frame.cameras.items():
            regions, mask = unexplained_regions(camera, certified[cid], changed[cid], self.config,
                                                 result.full_refresh_hint, self.last_evidence.get(cid))
            result.invalidation_regions[cid], self.last_masks[cid] = regions, mask
        novelty_ms = (perf_counter() - novelty_start) * 1000
        result.feature_versions = self.store.feature_versions()
        result.provenance = export_provenance(self.store)
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
        start = perf_counter()
        if not self.config.enabled:
            return precheck_result
        identity = (frame.scene_id, frame.frame_id, frame.timestamp, frame.execution_signature)
        if (getattr(self, "_pending", None) != identity or precheck_result is not self.last_result):
            raise ValueError("commit requires this frame's pending precheck; repeated/stale commits rejected")
        self._pending = None
        if not np.isfinite(frame.timestamp) or not valid_transform(frame.ego_to_world):
            self.reset(frame.scene_id)
            self.last_result = precheck_result
            return precheck_result
        anchors = list(self.store.anchors.values())
        predictions = []
        for pred in full_outputs.anchors:
            g = pred.geometry_world
            if (pred.anchor_type not in ("object", "map", "region") or g.points.ndim != 2
                    or g.points.shape[1] != 3 or len(g.points) < 2
                    or not np.isfinite(g.points).all() or not np.isfinite(g.center).all()):
                continue
            if (not np.isfinite(pred.geometry_uncertainty) or pred.geometry_uncertainty < 0
                    or (g.size is not None and (not np.isfinite(g.size).all() or np.any(g.size <= 0)))):
                continue
            predictions.append(pred)
        assoc_start = perf_counter()
        pairs, costs = associate(anchors, predictions, frame.timestamp, self.config)
        assoc_ms = (perf_counter() - assoc_start) * 1000
        matched = {j: anchors[i].anchor_id for i, j in pairs}
        committed = []
        validated_cameras = {cid: valid_projection_camera(cam) for cid, cam in frame.cameras.items()}
        for j, pred in enumerate(predictions[:self.config.max_anchors]):
            aid = matched.get(j) or self.store.new_anchor_id()
            prior = self.store.anchors.get(aid)
            velocity_ok = (pred.velocity_world is not None and np.asarray(pred.velocity_world).shape == (3,)
                           and np.isfinite(pred.velocity_world).all())
            a = WorldAnchor(aid, frame.scene_id, pred.anchor_type, pred.semantic_class,
                            copy.deepcopy(pred.geometry_world), frame.timestamp,
                            prior.last_verified_timestamp if prior else None,
                            np.asarray(pred.velocity_world).copy() if velocity_ok else None,
                            bool(pred.velocity_reliable and velocity_ok), pred.geometry_uncertainty)
            a.source_score = pred.source_score
            a.history_observations = prior.history_observations + 1 if prior else 1
            a.last_innovation_m = (float(np.linalg.norm(
                pred.geometry_world.center - predict_geometry(prior, frame.timestamp).center))
                if prior else None)
            # New support always comes from an actually computed frame, never chained crops.
            for cid, camera in frame.cameras.items():
                p = project_anchor(a, camera, self.config, validated_cameras[cid])
                if p.valid:
                    descriptor = self.validator.descriptor(camera.image, p.region)
                    if descriptor is not None:
                        a.observation_support[cid] = AnchorObservation(cid, p.region, descriptor,
                                                                      camera.timestamp, p.coverage)
            self.store.anchors[aid] = a
            committed.append(aid)
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
        self.store.last_timestamp = frame.timestamp
        self.store.execution_signature = frame.execution_signature
        self.store.commit_count += 1
        if self.config.temporal_visibility or self.config.local_support:
            # Own only one bounded grayscale frame per camera, from actual computation.
            # precheck never replaces this source with an uncomputed/verified frame.
            from .temporal_visibility import gray_image
            self._previous_cameras = {}
            for cid, camera in frame.cameras.items():
                saved = copy.copy(camera)
                saved.image = gray_image(camera)
                saved.intrinsic = np.array(camera.intrinsic,copy=True)
                saved.world_to_camera = None if camera.world_to_camera is None else np.array(camera.world_to_camera,copy=True)
                saved.visibility_mask = None; saved.geometry_support_mask = None
                self._previous_cameras[cid] = saved
        self.last_commit_diagnostics = {"source": full_outputs.source, "committed_anchor_ids": committed,
            "association_pairs": [[anchors[i].anchor_id, j] for i, j in pairs],
            "association_ms": assoc_ms, "state_update_ms": (perf_counter() - start) * 1000 - assoc_ms,
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
        if self.last_result is None:
            raise RuntimeError("Call precheck first")
        return copy.deepcopy(self.last_result)

    def get_candidate_feature(self, feature_id, computation_version):
        """Read before commit; a replaced slot cannot masquerade as an older version."""
        if self.last_result is None or not any(c['feature_id'] == feature_id
                                              for c in self.last_result.valid_feature_candidates):
            raise KeyError('Not a verified candidate')
        record = self.store.features.get(feature_id)
        if record is None or record.computation_version != computation_version:
            raise KeyError('Historical feature version has been replaced; acquire before commit')
        return record
