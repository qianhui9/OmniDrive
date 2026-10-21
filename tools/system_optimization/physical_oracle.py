"""Independent annotation oracle, applied OFFLINE to captured actual M1 inputs.

No annotation object reaches a model, worker, precheck, cache or online query.
Reuses the pre-existing frozen 2m source / 1m current physical protocol.
"""
import argparse,gzip,hashlib,json,pickle
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from worldstream.m1 import M1Config,WorldAnchor,Geometry
from worldstream.m1.geometry import project_anchor,region_slices
from tools.worldstream_m1.dataset import load_oracle
from tools.worldstream_m1.evaluate import oracle_assignment
from tools.worldstream_m1.refinement_common import physical_details,summarize_rows,interval
from .identity import token_index,validate_sequence
from .audit import write


def historical_anchor(record,ego_to_world,scene):
    """Invert the existing constant-velocity predictor; no new association."""
    pose=np.asarray(ego_to_world,dtype=float)
    center=pose[:3,:3]@np.asarray(record['center_ego_m'])+pose[:3,3]
    points=(pose[:3,:3]@np.asarray(record['geometry_ego_m']).T).T+pose[:3,3]
    velocity=record.get('velocity_ego_mps')
    velocity=pose[:3,:3]@np.asarray(velocity) if velocity is not None else None
    shift=velocity*(record['verified_timestamp']-record['source_timestamp']) if velocity is not None else np.zeros(3)
    anchor=WorldAnchor(record['anchor_id'],scene,record['anchor_type'],record['semantic_class'],
        Geometry(points-shift,center-shift),record['source_timestamp'],velocity_world=velocity,
        velocity_reliable=velocity is not None,source_frame_id=record['source_frame'])
    return anchor
