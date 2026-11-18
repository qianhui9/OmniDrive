"""Count actually executed dense MACs and synchronize CUDA timing receipts.

MACs cover Linear/Conv2d and attention QK/AV products, including functional EVA
projections and fused attention. Norms, activations, copies and sampling are not
MACs. This is a shape-derived count of executed arithmetic, not hardware FLOPs
or a window-reuse proxy. GPU latency is measured independently with CUDA events.
"""
from functools import wraps
from time import perf_counter
import math
import torch
from torch import nn
from worldstream.m3.visual_state import base_model


class Meter:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.restores = []
        self.question = -1
        self.clear()
        for root,label in ((model.img_backbone,'visual'),(base_model(model.lm_head),'llm')):
            for module in root.modules():
                if isinstance(module,(nn.Linear,nn.Conv2d)):
                    self.handles.append(module.register_forward_hook(self.linear_hook(label)))
        # EVA uses functional q/k/v projections even in its native forward.
        for block in model.img_backbone.blocks:
            self.handles.append(block.attn.register_forward_pre_hook(self.vision_attention))
        runtime = model.m2_runtime
        if runtime is not None:
            original = runtime.executor.execute
            @wraps(original)
            def execute(block,x,cached,ids,_original=original):
                size = block.window_size
                total = len(x)*math.ceil(x.shape[1]/size)*math.ceil(x.shape[2]/size)
                if 0 < len(ids) < total:
                    n = size*size
                    self.add('visual',len(ids)*n*sum(m.in_features*m.out_features for m in
                        (block.attn.q_proj,block.attn.k_proj,block.attn.v_proj)))
                    self.add('visual',2*len(ids)*n*n*block.attn.q_proj.out_features)
                return _original(block,x,cached,ids)
            self.replace(runtime.executor,'execute',execute)
            self.timed(runtime,'forward_from_m1','visual')
        else:
            # Hooks leave the frozen backbone topology intact.
            stack = []
            def start(module,args):
                stack.append(self.start('visual'))
            def stop(module,args,output):
                self.stop('visual',stack.pop())
            self.handles += [model.img_backbone.register_forward_pre_hook(start),
                             model.img_backbone.register_forward_hook(stop)]
        base = base_model(model.lm_head)
        original = base.forward
        @wraps(original)
        def llm(*args,_original=original,**kwargs):
            x = kwargs.get('inputs_embeds')
            if x is None:
                x = kwargs.get('input_ids',args[0] if args else None)
            n = x.shape[1]
            past = kwargs.get('past_key_values')
            previous = past[0][0].shape[-2] if past is not None else 0
            phase = 'prefill' if kwargs.get('inputs_embeds') is not None else 'decode'
            self.phase = phase
            hidden = base.config.hidden_size
            layers = base.config.num_hidden_layers
            self.add('llm',2*x.shape[0]*n*(previous+n)*hidden*layers)
            self.forward_lengths.append(dict(request_index=self.question,phase=phase,
                query_tokens=n,key_tokens=previous+n))
            event = self.start(phase)
            result = _original(*args,**kwargs)
            self.stop(phase,event)
            self.phase = None
            return result
        llm._worldstream_read_only_observer = True
        self.replace(base,'forward',llm)
        if model.m3_runtime is not None:
            self.timed(model.m3_runtime,'generate','language_total')
        else:
            original = model.lm_head.generate
            @wraps(original)
            def native(*args,_original=original,**kwargs):
                self.question += 1
                return _original(*args,**kwargs)
            self.replace(model.lm_head,'generate',native)

    def replace(self,obj,name,value):
        self.restores.append((obj,name,getattr(obj,name)))
        setattr(obj,name,value)

    def clear(self):
        self.macs = dict(visual=0,llm=0,prefill=0,decode=0)
        self.events = []
        self.forward_lengths = []
        self.phase = None
        self.question = -1

    def add(self,label,count):
        self.macs[label] += int(count)
        if label == 'llm' and self.phase is not None:
            self.macs[self.phase] += int(count)

    def linear_hook(self,label):
        def hook(module,args,output):
            if isinstance(module,nn.Linear):
                # LoRA subclass executes this dense projection plus its own A/B
                # Linear modules, which are independently observed.
                value = output.numel()*module.in_features
            else:
                value = output.numel()*(module.in_channels//module.groups)*math.prod(module.kernel_size)
            self.add(label,value)
        return hook

    def vision_attention(self,module,args):
        x = args[0]
        batch,n = x.shape[0],math.prod(x.shape[1:-1])
        self.add('visual',batch*n*sum(m.in_features*m.out_features for m in
            (module.q_proj,module.k_proj,module.v_proj)))
        self.add('visual',2*batch*n*n*module.q_proj.out_features)

    def start(self,label):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event,perf_counter()

    def stop(self,label,start):
        event,tick = start
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.events.append((label,event,end,(perf_counter()-tick)*1000))

    def timed(self,obj,name,label):
        original = getattr(obj,name)
        @wraps(original)
        def call(*args,**kwargs):
            start = self.start(label)
            result = original(*args,**kwargs)
            self.stop(label,start)
            return result
        call._worldstream_read_only_observer = True
        self.replace(obj,name,call)

    def collect(self):
        values = {}
        for label,a,b,wall in self.events:
            entry = values.setdefault(label,dict(cuda_ms=0.,host_ms=0.,calls=0))
            entry['cuda_ms'] += a.elapsed_time(b)
            entry['host_ms'] += wall
            entry['calls'] += 1
        return dict(executed_macs=dict(self.macs),timings=values,llm_forward_lengths=list(self.forward_lengths))

    def close(self):
        for handle in self.handles:
            handle.remove()
        for obj,name,original in reversed(self.restores):
            setattr(obj,name,original)
