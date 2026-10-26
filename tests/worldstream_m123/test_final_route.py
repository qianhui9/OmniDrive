"""Final route contract without model weights or GPU inference."""
from pathlib import Path
from dataclasses import asdict
import json
import pytest
import torch
from mmcv import Config
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m2.config import M2Config
from worldstream.m3 import M3Runtime
from worldstream.m3.scheduler import Scheduler
from worldstream.m3.config import M3Config

ENTRY='projects/configs/OmniDrive/worldstream_m123_auto.py'


def test_final_configuration_and_baseline_heads():
    cfg=Config.fromfile(ENTRY);base=Config.fromfile('projects/configs/OmniDrive/mask_eva_lane_det_vlm.py')
    m1=M1Config.load(cfg.model.m1_config);expected=M1Config()
    assert {k:v for k,v in asdict(m1).items() if k not in ('logging','visualization','temporal_visibility','local_support','source_linked_state')}=={k:v for k,v in asdict(expected).items() if k not in ('logging','visualization','temporal_visibility','local_support','source_linked_state')}
    assert m1.temporal_visibility and m1.local_support and m1.source_linked_state
    WorldStateManager(m1)
    m2=M2Config(**cfg.model.m2_config);m3=M3Config(**cfg.model.m3_config)
    assert cfg.model.m1_enable and cfg.model.m2_enable and cfg.model.m3_enable
    assert cfg.model.worldstream_require_m123
    assert not base.model.get('worldstream_require_m123', False)
    assert m2.mode==m3.mode=='auto' and m2.m1_service=='maintain'
    assert m2.reuse_boundary=='window' and m2.cost_table
    assert m2.state_interface=='source_linked'
    assert m3.state_context=='augment'
    assert not hasattr(m3,'selector') and not hasattr(m3,'batched_decode')
    for key in ('img_backbone','pts_bbox_head','map_head','lm_head','tokenizer'):
        if key in base.model:assert cfg.model[key]==base.model[key]
    assert Path(m2.cost_table).is_file()
    assert m3.reuse_strategy=='wsdr_kv' and m3.reuse_policy
    runtime=M3Runtime(m3)
    assert not hasattr(runtime.scheduler,'costs') and not hasattr(runtime.scheduler,'quality')


def test_final_m2_native_fallback():
    from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT
    from worldstream.m2 import IncrementalVisionRuntime
    cfg=Config.fromfile(ENTRY)
    model=EVAViT(img_size=40,patch_size=4,embed_dim=32,depth=4,num_heads=4,
        window_size=4,window_block_indexes=[0,1,3],pt_hw_seq_len=4,
        pretrain_img_size=40,flash_attn=False,with_cp=False).eval()
    executor=IncrementalVisionRuntime(model,M2Config(**cfg.model.m2_config))
    with torch.no_grad():
        image=torch.zeros(1,3,40,40)
        value,d=executor.forward(image,execution_context=dict(scene_id='s',frame_id='f',camera_ids=['front']))
        reference=model(image)
    assert torch.equal(reference[0],value[0]) and d['mode']=='full_update' and d['actual_skipped_windows']==0
    assert d['cache_boundary']=='window' and d['cacheable_blocks']==[0,1]
    assert d['global_barrier_block']==2 and not d['cache_saved']


@pytest.mark.parametrize('mode',['hint','shadow'])
def test_retired_external_identity_modes_rejected(mode):
    with pytest.raises(TypeError):WorldStateManager(M1Config(dqtrack_mode=mode))


@pytest.mark.parametrize('content',[None,'broken','[]','{"approved": "invalid"}'])
def test_missing_or_invalid_m3_evidence_fails_closed(tmp_path,content):
    path=tmp_path/'evidence.json'
    if content is not None:path.write_text(content)
    scheduler=Scheduler(M3Config(reuse_policy=str(path)))
    assert scheduler.policy=={}


def test_only_one_formal_entry_and_no_alternate_runtime_files():
    assert list(Path('projects/configs/OmniDrive').glob('worldstream_*.py'))==[Path(ENTRY)]
    for name in ('worldstream/m2/camera_executor.py','worldstream/m2/conditional_prefix.py',
                 'worldstream/m3/carrier_selector.py','worldstream/m3/batched_executor.py'):
        assert not Path(name).exists()
