"""Small native full-model latency probe with real language generation.

Delegates to model_smoke's exact baseline/observe comparison. Timing wrappers
change no tensors, weights, decoding parameters or generation cache decisions.
External DQTrack is file-fed; its GPU cost is measured separately, not hidden.
"""
import argparse
from functools import wraps
import json
from pathlib import Path
import sys
from time import perf_counter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--scene',required=True)
    p.add_argument('--trajectories',required=True);p.add_argument('--frames',type=int,default=2)
    a=p.parse_args()
    if not 2<=a.frames<=4:raise ValueError('Only 2..4 native VLM frames per mode')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    from .run_dqtrack_source import wait_for_gpu
    wait_for_gpu(60)
    import torch,yaml
    from projects.mmdet3d_plugin.models.dense_heads.llava_llama import LlavaLlamaForCausalLM
    from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D
    cfg=yaml.safe_load(Path('configs/worldstream_m1_balanced.yaml').read_text())
    cfg.update(dqtrack_mode='hint',dqtrack_results_path=str(Path(a.trajectories).resolve()),
        dqtrack_association_policy='conservative',logging=False)
    config=root/'hint.yaml';config.write_text(yaml.safe_dump(cfg))
    context={};calls=[];original_frame=Petr3D.simple_test;original_generate=LlavaLlamaForCausalLM.generate
    @wraps(original_frame)
    def frame_call(self,*args,**kwargs):
        meta=args[0] if args else kwargs['img_metas']
        context.update(frame_id=str(meta[0]['sample_idx']),language_directory=self.save_path)
        return original_frame(self,*args,**kwargs)
    @wraps(original_generate)
    def generate(self,*args,**kwargs):
        torch.cuda.synchronize();start=perf_counter()
        result=original_generate(self,*args,**kwargs)
        torch.cuda.synchronize();calls.append(dict(context,wall_ms=(perf_counter()-start)*1000))
        return result
    Petr3D.simple_test=frame_call;LlavaLlamaForCausalLM.generate=generate
    saved=sys.argv
    try:
        sys.argv=['model_smoke','--scene',a.scene,'--frames',str(a.frames),'--compare',
            '--m1-config',str(config),'--output',str(root/'native')]
        from .model_smoke import main as native
        native()
    finally:
        sys.argv=saved;Petr3D.simple_test=original_frame;LlavaLlamaForCausalLM.generate=original_generate
        (root/'language_timings.json').write_text(json.dumps(dict(calls=calls,
            scope='native Llava generate wall time; no DQTrack forward in this process',
            external_tracker_included=False),indent=2))


if __name__=='__main__':main()
