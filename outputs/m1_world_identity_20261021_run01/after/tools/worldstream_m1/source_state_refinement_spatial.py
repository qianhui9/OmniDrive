"""Small isolated online replay for actual refined mask visualization."""
from pathlib import Path
import numpy as np
import cv2
import torch
from mmcv import Config
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.mask_codec import decode_mask
from .state_interface_replay import preprocessed_frame
from .evidence_experiments import load_inputs
from .io_utils import write_json


def main():
    root=Path('outputs/module1_source_state_refinement/profile_output')
    torch.set_num_threads(1);cv2.setNumThreads(1)
    specs,infos,outputs,_=load_inputs('development')
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    settings=M1Config.load('configs/worldstream_m1_source_state_refined.yaml')
    m=WorldStateManager(settings)
    for token in specs[0]['samples']:
        frame,_=preprocessed_frame(infos[token],cfg,settings);result=m.precheck(frame)
        for cid,contract in result.computation_state.refresh_contract.items():
            candidate=decode_mask(contract['reuse_candidate_mask'])
            if candidate.any():
                np.savez_compressed(root/'spatial_masks.npz',image=frame.cameras[cid].image,
                    candidate=candidate,dirty=decode_mask(contract['dirty_hint_mask']),
                    unknown=decode_mask(contract['unknown_mask']))
                write_json(root/'spatial_context.json',dict(frame=token,camera=cid,source=m._previous_frame_id,
                    population='exploratory_reused_population',online_GT=False))
                return
        m.commit(frame,outputs[token],result)
    raise RuntimeError('No candidate mask available for visualization')


if __name__=='__main__':main()
