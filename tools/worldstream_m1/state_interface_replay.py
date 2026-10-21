"""Fresh CPU nuScenes replay in the original test pipeline coordinates.

Historical original-model detections are Commit inputs. GT is used after
Precheck only for sparse scoring, never to produce candidates or refresh masks.
This tool does not run EVA, heads, memory or the LLM; their results remain Not Run.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import resource
import cv2
import numpy as np
import torch
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.mask_codec import decode_mask
from worldstream.m1.novelty_detector import validate_refresh_partition
from worldstream.m1.omnidrive_adapter import frame_from_model
from .evidence_experiments import load_inputs,append_json,summary
from .coverage_oracle import world_lidar,prepare_pair,score_camera
from .io_utils import write_json


def preprocessed_frame(info,cfg,settings):
    from mmdet3d.datasets.pipelines import Compose
    from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
    obj=object.__new__(CustomNuScenesDataset)
    obj.data_infos=[info];obj.modality=dict(use_camera=True);obj.test_mode=True
    x=Compose(cfg.test_pipeline[:5])(obj.get_data_info(0))
    data={k:np.array([x[k]]) for k in ("intrinsics","extrinsics","img_timestamp","ego_pose")}
    data["img"]=torch.from_numpy(np.stack(x["img"]).transpose(0,3,1,2))[None]
    data["timestamp"]=np.array([x["timestamp"]])
    return frame_from_model([x],data,settings),data["img"][0]
