from dataclasses import asdict
from .state_types import ComputationalProvenance


def provenance_for(record, frame):
    return ComputationalProvenance(
        record.feature_id, list(record.source_anchor_ids), list(record.source_camera_ids),
        dict(record.source_region), record.source_layer, record.dependency_scope,
        coordinate_space={k: v.coordinate_space for k, v in frame.cameras.items()})


def export_provenance(store):
    return {key: asdict(value) for key, value in store.provenance.items()}
