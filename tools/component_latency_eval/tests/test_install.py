"""Install all scopes on real classes; run a small real EVA forward on CPU."""
from types import SimpleNamespace
import torch
from tools.component_latency_eval.instrumentation import Instrumentation
from tools.component_latency_eval.timing import Recorder


@torch.no_grad()
def test_all_scopes_install_and_real_eva_path_restores():
    from projects.mmdet3d_plugin.models.backbones.eva_vit import EVAViT
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D
    from worldstream.m1 import WorldStateManager
    from worldstream.m1.state_types import ComputationStateBundle
    from worldstream.m2 import IncrementalVisionRuntime
    from worldstream.m3.runtime import M3Runtime
    torch.manual_seed(19)
    eva=EVAViT(img_size=40,patch_size=4,embed_dim=32,depth=4,num_heads=4,window_size=4,
               window_block_indexes=[0,1,3],pt_hw_seq_len=4,pretrain_img_size=40,
               flash_attn=False,with_cp=False).eval()
    runtime=IncrementalVisionRuntime(eva)
    model=SimpleNamespace(m1_manager=WorldStateManager(),m2_runtime=runtime,m3_runtime=M3Runtime(),
                          simple_test_pts=Petr3D.simple_test_pts)
    recorder=Recorder('cpu_contract')
    original=eva.patch_embed.forward.__func__.__code__
    instrumentation=Instrumentation(model,recorder)
    try:
        assert len(instrumentation.audit)==15
        image=torch.randn(2,3,40,40)
        context=dict(scene_id='seq',frame_id='0',camera_ids=['front','back'],execution_signature='test',
                     coordinate_space='preprocessed_image',timestamp=1.,state_epoch=1)
        bundle=ComputationStateBundle('seq','0',1.,1,'test',diagnostics=dict(full_refresh_hint=True))
        recorder.frame('seq','0')
        outputs,diag=runtime.forward(image,execution_context=context,computation_state=bundle)
        assert diag['execution_completed'] and outputs[0].shape==(2,32,10,10)
        names=[r['component_name'] for r in recorder.rows]
        assert names.count('DAVP.Patch Embedding')==1
        assert names.count('DAVP.Visual Planning')==2
        assert names.count('DAVP.Incremental EVA')==1
        assert names.count('DAVP.Global Execution')==1
    finally:
        instrumentation.close()
    assert eva.patch_embed.forward.__func__.__code__ is original
