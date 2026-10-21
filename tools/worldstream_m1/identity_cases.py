"""Draw actual measured candidate/rejection examples; never a qualification plot."""
import argparse
import json
from pathlib import Path
import pickle
import cv2
import numpy as np
from .dataset import frame_from_info


def measured_tracks(record):
    """Do not draw failed LK placeholder coordinates as measured matches."""
    p=np.asarray(record.get('source_points',[]),float);q=np.asarray(record.get('current_points',[]),float)
    if p.ndim!=2 or p.shape[1:]!=(2,) or q.shape!=p.shape:return np.empty((0,2)),np.empty((0,2))
    valid=np.asarray(record.get('correspondence_valid',np.ones(len(p),bool)),bool)
    if valid.shape!=(len(p),):return np.empty((0,2)),np.empty((0,2))
    valid=valid & np.isfinite(p).all(1) & np.isfinite(q).all(1)
    return p[valid],q[valid]
