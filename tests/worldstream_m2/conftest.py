import pytest
import torch


@pytest.fixture
def eva():
    from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT
    torch.manual_seed(19)
    return EVAViT(img_size=40, patch_size=4, embed_dim=32, depth=4, num_heads=4,
                  window_size=4, window_block_indexes=[0, 1, 3], pt_hw_seq_len=4,
                  pretrain_img_size=40, flash_attn=False, with_cp=False).eval()


@pytest.fixture
def context():
    return dict(scene_id='scene', frame_id='0', camera_ids=['front', 'back'],
                execution_signature='test', coordinate_space='preprocessed_image',timestamp=1.,state_epoch=1)


@pytest.fixture
def admitted_runtime(eva,monkeypatch):
    from worldstream.m2 import IncrementalVisionRuntime
    runtime=IncrementalVisionRuntime(eva,dict(cache_capture='always'))
    # Unit fixture only: isolate execution/source contracts from measured cost
    # certification. No mocked costs are written to deployment artifacts.
    monkeypatch.setattr(runtime.costs,'beneficial',lambda key,count:(True,None))
    return runtime
