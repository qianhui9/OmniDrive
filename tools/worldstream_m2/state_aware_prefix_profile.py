"""Separate original block 0/1 cost, excluding PE/ABS and unrelated work."""
import os
import torch
from worldstream.m1 import M1Config
from worldstream.m2.profiler import measure
from tools.worldstream_m1.source_state_refinement_replay import preprocessed_frame
from tools.worldstream_m1.evidence_experiments import load_inputs
from tools.worldstream_m1.io_utils import write_json
from .state_aware_feasibility import ROOT
from .source_state_refinement_cuda_batch import wait_idle
from .common import backbone


@torch.no_grad()
def main():
    target=ROOT/'prefix_only_profile.json'
    if target.exists():raise RuntimeError('Preserve completed prefix measurement')
    gpu=wait_idle(ROOT);os.environ['CUDA_VISIBLE_DEVICES']=str(gpu['index'])
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    from mmcv import Config
    from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
    specs,infos,_,_=load_inputs('development');token=specs[0]['samples'][0]
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    frame,images=preprocessed_frame(infos[token],cfg,M1Config.load('configs/worldstream_m1_engineering_frozen.yaml'))
    model=backbone();x=model.patch_embed(images.cuda())
    if model.pos_embed is not None:x=x+get_abs_pos(model.pos_embed,model.pretrain_use_cls_token,x.shape[1:3])
    timing=measure(lambda:model.blocks[1](model.blocks[0](x)),warmup=5,repeats=20)
    write_json(target,dict(timing=timing,input_shape=list(x.shape),input_stride=list(x.stride()),dtype=str(x.dtype),
        frame=token,scene=frame.scene_id,GPU=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        scope='Only original block 0 + block 1, excludes PE and ABS addition. Actual execution, not reuse qualification.'))


if __name__=='__main__':main()
