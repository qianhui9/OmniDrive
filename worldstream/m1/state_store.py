"""Bounded scene-local state; owned tensors are detached copies, never temporal Memory."""
from dataclasses import fields
import copy
import numpy as np
from .state_types import ComputedFeature, FeatureRecord
from .provenance import provenance_for
from .geometry import immutable_array


class StateStore:
    def __init__(self, config):
        self.config = config
        self.generation = 0
        self.reset()

    def reset(self, scene_id=None):
        self.generation += 1
        self.scene_id = scene_id
        self.anchors = {}
        self.features = {}
        self.provenance = {}
        self.tensor_bytes = 0
        self.last_timestamp = None
        self.execution_signature = None
        self.commit_count = 0
        self.next_anchor_id = 0
        self.committed_frame_ids = frozenset()

    def new_anchor_id(self):
        self.next_anchor_id += 1
        return "{}:g{}:a{}".format(self.scene_id, self.generation, self.next_anchor_id)

    def fork(self):
        """Private transaction metadata; immutable owned tensors/geometry are shared.

        Writers replace features/geometry rather than mutating their arrays. Copy
        mutable anchor metadata so failed verification/commit cannot leak writes.
        """
        staged = copy.copy(self)
        staged.anchors = {}
        for key, anchor in self.anchors.items():
            value = copy.copy(anchor)
            value.geometry_world = copy.copy(anchor.geometry_world)
            value.observation_support = {cid:copy.copy(obs) for cid,obs in anchor.observation_support.items()}
            value.associated_feature_ids = list(anchor.associated_feature_ids)
            value.association_history = copy.deepcopy(anchor.association_history)
            value.external_tracking = copy.deepcopy(anchor.external_tracking)
            value.motion_history = copy.deepcopy(anchor.motion_history)
            value.motion_prediction = copy.deepcopy(anchor.motion_prediction)
            value.physical_certification = copy.deepcopy(anchor.physical_certification)
            staged.anchors[key] = value
        staged.features = {key:self.feature_snapshot(value) for key,value in self.features.items()}
        staged.provenance = copy.deepcopy(self.provenance)
        return staged

    @staticmethod
    def feature_snapshot(record):
        """Protect internal allocations and mutable metadata at read/fork boundaries."""
        result=copy.copy(record)
        for name in ('source_anchor_ids','source_camera_ids','source_region','computation_sources'):
            setattr(result,name,copy.deepcopy(getattr(record,name)))
        if isinstance(record.tensor,np.ndarray):result.tensor=immutable_array(record.tensor)
        elif record.tensor is not None:result.tensor=record.tensor.detach().clone()
        return result

    def source_digest(self):
        """Detect public state replacement while a frame transaction is pending."""
        import hashlib
        import json
        from .geometry import state_geometry_version
        from .state_types import json_value
        anchors=[]
        for aid,a in self.anchors.items():
            anchors.append((aid,state_geometry_version(a),a.source_score,a.history_observations,
                a.last_verified_timestamp,a.last_certified_timestamp,a.association_history,
                repr(a.physical_certification),a.association_basis,a.identity_link_qualified,
                [(cid,obs.timestamp,obs.region,hashlib.sha256(np.ascontiguousarray(obs.descriptor).tobytes()).hexdigest())
                 for cid,obs in a.observation_support.items()]))
        features=[(fid,r.computation_version,r.computation_timestamp,r.source_anchor_ids,r.source_camera_ids,
                   r.source_region,getattr(r.tensor,'_version',None),self._nbytes(r.tensor)) for fid,r in self.features.items()]
        from dataclasses import asdict
        # Bind all mutable metadata, not only geometry and tensor allocation size.
        metadata = {aid:asdict(a) for aid,a in self.anchors.items()}
        for row in metadata.values():
            if not row['external_tracking']:
                row.pop('external_tracking')
        feature_metadata = {}
        for fid,r in self.features.items():
            feature_metadata[fid] = {f.name:getattr(r,f.name) for f in fields(r) if f.name!='tensor'}
            if isinstance(r.tensor,np.ndarray):
                feature_metadata[fid]['tensor_sha256'] = hashlib.sha256(r.tensor.tobytes()).hexdigest()
        value=(self.generation,self.scene_id,self.execution_signature,self.next_anchor_id,sorted(self.committed_frame_ids),
               self.last_timestamp,self.commit_count,anchors,features,metadata,feature_metadata,
               {k:asdict(v) for k,v in self.provenance.items()})
        return hashlib.sha256(json.dumps(json_value(value),sort_keys=True,allow_nan=False).encode()).hexdigest()

    @staticmethod
    def _nbytes(tensor):
        if tensor is None:
            return 0
        return tensor.nbytes if isinstance(tensor, np.ndarray) else tensor.numel() * tensor.element_size()

    def put_feature(self, feature, frame):
        previous = self.features.get(feature.feature_id)
        values = {f.name: getattr(feature, f.name) for f in fields(ComputedFeature)}
        tensor = feature.tensor
        # Remove the old allocation from the budget before replacing the slot.
        self.tensor_bytes -= self._nbytes(previous.tensor) if previous else 0
        count = self._nbytes(tensor)
        capture = self.config.capture_tensors and (self.config.capture_backbone or feature.feature_kind != "backbone")
        if capture and count + self.tensor_bytes <= self.config.max_tensor_bytes and tensor is not None:
            if isinstance(tensor, np.ndarray):
                tensor = immutable_array(tensor)
            else:
                tensor = tensor.detach().to(self.config.storage_device).clone()
        else:
            tensor = None
        values["tensor"] = tensor
        values["source_anchor_ids"] = list(feature.source_anchor_ids)
        values["source_camera_ids"] = list(feature.source_camera_ids)
        values["source_region"] = {k: list(v) for k, v in feature.source_region.items()}
        values["computation_version"] = previous.computation_version + 1 if previous else 1
        values["computation_timestamp"] = frame.timestamp
        values["execution_signature"] = frame.execution_signature
        values["source_frame_id"] = frame.frame_id
        values["source_state_epoch"] = self.generation
        record = FeatureRecord(**values)
        self.features[feature.feature_id] = record
        self.provenance[feature.feature_id] = provenance_for(record, frame)
        self.tensor_bytes += self._nbytes(tensor)
        return record

    def prune(self, timestamp):
        expired = {k for k, a in self.anchors.items()
                   if timestamp - a.source_timestamp > self.config.max_state_age_s}
        remaining = sorted((a for k, a in self.anchors.items() if k not in expired),
                           key=lambda a: (-a.source_timestamp, a.anchor_id))
        expired.update(a.anchor_id for a in remaining[self.config.max_anchors:])
        for key in expired:
            del self.anchors[key]
        for key, record in list(self.features.items()):
            if (record.computation_timestamp is None or timestamp - record.computation_timestamp > self.config.max_state_age_s
                    or any(a not in self.anchors for a in record.source_anchor_ids)):
                self.tensor_bytes -= self._nbytes(record.tensor)
                del self.features[key]
                self.provenance.pop(key, None)

    def feature_versions(self):
        return {k: {"computation_version": r.computation_version,
                    "state_epoch": self.generation,
                    "computation_timestamp": r.computation_timestamp,
                    "last_verified_timestamp": r.last_verified_timestamp,
                    "shape": list(r.shape), "dtype": r.dtype, "source_device": r.device,
                    "storage_device": str(r.tensor.device) if hasattr(r.tensor, "device") else "cpu",
                    "tensor_captured": r.tensor is not None,
                    "tensor_availability": "captured" if r.tensor is not None else "unavailable",
                    "network_reuse_authorized": False,
                    "execution_signature": r.execution_signature,
                    "source_frame_id": r.source_frame_id, "source_state_epoch": r.source_state_epoch,
                    "computation_sources": [dict(s) for s in r.computation_sources]}
                for k, r in self.features.items()}
