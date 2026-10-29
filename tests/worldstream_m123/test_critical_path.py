"""Compatibility guards for complete live-state and frame transaction hashes."""
import importlib.util
from pathlib import Path
from dataclasses import dataclass
import json

import numpy as np


def reference_module(name):
    path=Path('outputs/m123_critical_path_and_joint_tuning_run01/baseline_sources/worldstream/m1')/(name+'.py')
    spec=importlib.util.spec_from_file_location('worldstream.m1._critical_reference_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_complete_state_normalization_preserves_reference_bytes_and_live_mutations():
    from worldstream.m1.source_digest import normalized
    legacy=reference_module('source_digest').normalized
    @dataclass
    class Record:
        values: object
        score: float
    array=np.arange(24,dtype=np.float64).reshape(4,6)[:,::2]
    value=dict(record=Record([True,None,'事实',-1,1.25,np.float32(.5)],float('nan')),
        numeric=array,empty=np.empty((0,3),np.uint8),objects=np.array([dict(count=1)],dtype=object),
        tuple=(float('inf'),float('-inf'),np.int64(9)))
    encode=lambda x:json.dumps(x,sort_keys=True,allow_nan=False)
    before=encode(normalized(value));assert before==encode(legacy(value))
    array[0,0]=99
    assert encode(normalized(value))!=before
    assert encode(normalized(value))==encode(legacy(value))
    value['objects'][0]['count']=2
    assert encode(normalized(value))==encode(legacy(value))


def test_exact_frame_binding_preserves_bytes_and_rejects_input_mutation():
    from worldstream.m1.input_contract import frame_digest,camera_snapshot_digest
    from worldstream.m1.state_types import CameraFrame,Frame
    legacy=reference_module('input_contract')
    camera=CameraFrame('front',np.arange(6*8*3,dtype=np.uint8).reshape(6,8,3),np.eye(3),np.eye(4),1.,'preprocessed_image',None,True)
    frame=Frame('s','f',1.,np.eye(4),{'front':camera},'signature',{})
    initial=frame_digest(frame)
    assert initial==legacy.frame_digest(frame)
    assert camera_snapshot_digest(camera)==legacy.camera_snapshot_digest(camera)
    camera.image[0,0,0]=200
    assert frame_digest(frame)!=initial
    assert frame_digest(frame)==legacy.frame_digest(frame)
