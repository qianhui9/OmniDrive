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
    runtime=IncrementalVisionRuntime(eva,dict())
    return runtime


@pytest.fixture
def exact_runtime(eva):
    """Exact prefix controls use offline A and refresh on any H2 drift."""
    from worldstream.m2 import IncrementalVisionRuntime
    runtime=IncrementalVisionRuntime(eva,dict(global_drift_threshold=0.))
    runtime.forward=lambda images,**kwargs:runtime.evaluate_offline(images,variant='A',**kwargs)
    return runtime


def synthetic_approval(tmp_path,key,count):
    """Synthetic unit artifacts, confined to pytest's temporary directory."""
    import hashlib
    artifact=tmp_path/'synthetic_evidence.txt'
    artifact.write_text('UNIT TEST ONLY: not a deployment quality assessment')
    digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
    return dict(approved_by='synthetic_unit_fixture',updated=count,
        key_sha256=hashlib.sha256(key.encode()).hexdigest(),
        candidate=str(artifact),candidate_sha256=digest,
        quality_report=str(artifact),quality_report_sha256=digest,
        numerical_evidence=str(artifact),numerical_evidence_sha256=digest)
