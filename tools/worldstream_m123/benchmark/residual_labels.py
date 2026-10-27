"""Offline exact global suffix labels. Never imported by serving modules."""
from collections import deque
import torch
from worldstream.m2.anchor_risk import feature_statistics


class ResidualLabels:
    def __init__(self,runtime,observer):
        self.rows=[];self.bank=deque(maxlen=3);self.steps=[];self.scene=None
        original=runtime._update_global
        def update(x,context,signature):
            if self.scene!=context['scene_id']:
                self.bank.clear();self.steps=[];self.scene=context['scene_id']
            step=context.get('_global_world_signal',{}).get('score',0.)
            self.steps.append(step);index=len(self.steps)-1
            event_count=len(observer.events)
            # Private exact reference. Its module hook events are removed from
            # observer accounting; whole shadow-run latency remains excluded.
            h2=x.detach().clone();exact=h2.clone()
            for block in runtime.model.blocks[len(runtime.prefix):]:exact=block(exact)
            del observer.events[event_count:]
            residual=exact-h2
            for old in self.bank:
                stats=feature_statistics(h2,old['h2'],runtime.config.global_tail_fraction)
                age=index-old['index'];world=min(1.,sum(self.steps[old['index']+1:index+1]))
                approximate=h2+old['residual'];err=approximate.float()-exact.float()
                relres=(old['residual'].float()-residual.float()).square().mean().div(residual.float().square().mean().clamp_min(1e-12)).sqrt()
                final=err.square().mean().div(exact.float().square().mean().clamp_min(1e-12)).sqrt()
                mae=err.abs().mean();maximum=err.abs().max()
                values=torch.stack([relres,final,mae,maximum]).cpu().tolist()
                self.rows.append(dict(scene_id=self.scene,frame_id=context['frame_id'],anchor_frame_id=old['frame_id'],
                    age=age,world_change=world,world_signal=context.get('_global_world_signal'),**stats,
                    residual_relative_rms=values[0],final_relative_rms=values[1],final_mae=values[2],final_max_abs=values[3],
                    eligible_temporally=age<runtime.config.global_anchor_interval and not context.get('_global_full_refresh')))
            self.bank.append(dict(h2=h2,residual=residual,index=index,frame_id=context['frame_id']))
            return original(x,context,signature)
        runtime._update_global=update

    def clear(self):self.rows.clear()

    def reset(self):
        self.rows.clear();self.bank.clear();self.steps=[];self.scene=None
