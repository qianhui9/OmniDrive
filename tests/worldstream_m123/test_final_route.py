"""Final route contract without model weights or GPU inference."""
from pathlib import Path
from dataclasses import asdict
import json
import pytest
import torch
from mmcv import Config
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m2.config import M2Config
from worldstream.m2.camera_executor import CameraExecutor
from worldstream.m3 import M3Runtime
from worldstream.m3.scheduler import Scheduler
from worldstream.m3.config import M3Config

ENTRY='projects/configs/OmniDrive/worldstream_m123_auto.py'


def test_final_configuration_and_baseline_heads():
    cfg=Config.fromfile(ENTRY);base=Config.fromfile('projects/configs/OmniDrive/mask_eva_lane_det_vlm.py')
    m1=M1Config.load(cfg.model.m1_config);expected=M1Config()
    assert {k:v for k,v in asdict(m1).items() if k not in ('logging','visualization')}=={k:v for k,v in asdict(expected).items() if k not in ('logging','visualization')}
    WorldStateManager(m1)
    m2=M2Config(**cfg.model.m2_config);m3=M3Config(**cfg.model.m3_config)
    assert cfg.model.m1_enable and cfg.model.m2_enable and cfg.model.m3_enable
    assert m2.mode==m3.mode=='auto' and m2.m1_service=='maintain'
    assert not m2.experimental_conditional_reuse and not m2.experimental_exact_lease_separation
    assert not m3.offline_experiment and not m3.batched_decode and m3.state_context=='none'
    for key in ('img_backbone','pts_bbox_head','map_head','lm_head','tokenizer'):
        if key in base.model:assert cfg.model[key]==base.model[key]
    for path in (m2.camera_cost_table,m3.cost_table,m3.quality_certificate):assert Path(path).is_file()
    runtime=M3Runtime(m3)
    assert not runtime.scheduler.costs.get('approved') and not runtime.scheduler.quality.get('approved')


def test_final_m2_native_fallback():
    cfg=Config.fromfile(ENTRY);model=torch.nn.Identity().eval()
    executor=CameraExecutor(model,M2Config(**cfg.model.m2_config))
    with torch.no_grad():
        image=torch.zeros(1,3,8,8)
        value,d=executor.forward(image,dict(scene_id='s',frame_id='f',camera_ids=['front']),'signature')
    assert torch.equal(image,value) and d['mode']=='full' and d['actual_reused_cameras']==0


@pytest.mark.parametrize('mode',['hint','shadow'])
def test_retired_external_identity_modes_rejected(mode):
    with pytest.raises(ValueError):WorldStateManager(M1Config(dqtrack_mode=mode))


@pytest.mark.parametrize('content',[None,'broken','[]','{"approved": "invalid"}'])
def test_missing_or_invalid_m3_evidence_fails_closed(tmp_path,content):
    path=tmp_path/'evidence.json'
    if content is not None:path.write_text(content)
    scheduler=Scheduler(M3Config(cost_table=str(path),quality_certificate=str(path)))
    assert scheduler.costs==scheduler.quality=={}
