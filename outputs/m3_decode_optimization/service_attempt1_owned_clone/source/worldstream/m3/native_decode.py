"""Opt-in native LLaMA decode probe: omit only provably zero additive masks.

No Q/K/V/LoRA, KV layout, sampling or generation loop is replaced. The all-ones
check is real work (including its GPU sync) and must be charged to benchmarks.
"""
from contextlib import contextmanager
from functools import wraps
import torch
from .visual_state import base_model


class StaticDecodeGraphs:
    """Request-scoped installation of persistent, KV-free native CUDA graphs.

    Only inspected eval single-token LLaMA MLPs and PEFT LoRA projections are
    captured. Attention, positions, masks, KV cat, sampling and stopping remain
    native. Graph outputs are cloned: callers never observe a replay overwrite.
    No parameter is merged, cast, copied or mutated. Failed captures fall back
    to the original callable; interrupted generations discard all graph buffers.
    One owner per model; no concurrent mutation/generation is supported.
    """
    def __init__(self):
        self.graphs = {}
        self.signature = None
        self.blocked = set()

    def clear(self):
        self.graphs.clear()
        self.blocked.clear()
        self.signature = None

    @staticmethod
    def _modules(base):
        import transformers
        from transformers.models.llama.modeling_llama import LlamaMLP
        try:
            import peft
            from peft.tuners.lora.layer import Linear
        except ImportError:
            return []
        if transformers.__version__ != '4.31.0' or peft.__version__ != '0.7.0':
            return []
        modules=[]
        for layer in getattr(getattr(base, 'model', None), 'layers', ()):
            mlp=getattr(layer, 'mlp', None)
            if type(mlp) is LlamaMLP:modules.append(mlp)
            for attr in ('q_proj','k_proj','v_proj','o_proj'):
                module=getattr(getattr(layer,'self_attn',None),attr,None)
                if type(module) is Linear:modules.append(module)
        return [m for m in modules if 'forward' not in vars(m) and not any(
            sub._forward_hooks or sub._forward_pre_hooks for sub in m.modules())]

    @staticmethod
    def _eligible(x,args,kwargs):
        return (isinstance(x,torch.Tensor) and x.is_cuda and x.ndim==3
                and tuple(x.shape[:2])==(1,1) and not args and not kwargs
                and not torch.is_grad_enabled() and not torch.is_autocast_enabled())

    def _capture(self,native,x):
        stream=torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        static=x.detach().clone()
        with torch.cuda.stream(stream):
            for _ in range(2):native(static)
        torch.cuda.current_stream(x.device).wait_stream(stream)
        torch.cuda.synchronize(x.device)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):output=native(static)
        graph.replay()
        reference=native(x)
        # Different cuBLAS capture algorithms must not silently pass a relaxed
        # tolerance. This is a capture guard, not a whole-task certificate.
        if not torch.equal(output,reference):
            raise ValueError('Captured native subgraph is not bitwise equal')
        return graph,static,output

    @contextmanager
    def activate(self,lm_head):
        from time import perf_counter
        from .visual_state import execution_signature
        base=base_model(lm_head)
        stats=dict(enabled=True,captured=0,replayed=0,native_calls=0,
                   capture_failures=[],setup_cpu_ms=0.,fallback_reason=None)
        if (not torch.cuda.is_available() or torch.is_grad_enabled()
                or any(m.training for m in base.modules())
                or getattr(base,'_worldstream_graph_owner',None) is not None):
            stats['fallback_reason']='unsupported_device_training_grad_or_concurrent_owner'
            yield stats
            return
        tick=perf_counter();modules=self._modules(base)
        if not modules:
            stats['fallback_reason']='uninspected_backend'
            yield stats
            return
        signature=execution_signature(lm_head)
        if signature!=self.signature:
            self.clear();self.signature=signature
        stats['setup_cpu_ms']=(perf_counter()-tick)*1000
        originals=[];base._worldstream_graph_owner=self
        policy='_worldstream_decode_policy'
        previous=[(m,policy in vars(m),getattr(m,policy,None)) for m in {id(m):m for m in (lm_head,base)}.values()]
        def wrap(module):
            native=module.forward
            @wraps(native)
            def forward(x,*args,**kwargs):
                if not self._eligible(x,args,kwargs) or module in self.blocked or module.training:
                    stats['native_calls']+=1
                    return native(x,*args,**kwargs)
                key=(module,tuple(x.shape),x.dtype,x.device,tuple(x.stride()))
                try:
                    if key not in self.graphs:
                        self.graphs[key]=self._capture(native,x);stats['captured']+=1
                    graph,static,output=self.graphs[key]
                    static.copy_(x);graph.replay();stats['replayed']+=1
                    return output.clone()
                except (RuntimeError,ValueError) as error:
                    # Capture/replay errors never keep a possibly invalid graph.
                    self.graphs.pop(key,None);self.blocked.add(module)
                    stats['capture_failures'].append(dict(type=type(error).__name__,reason=str(error)[:240]))
                    stats['native_calls']+=1
                    return native(x,*args,**kwargs)
            originals.append((module,native));module.forward=forward
        try:
            for m in modules:wrap(m)
            for m,_,_ in previous:setattr(m,policy,'native_static_decode_graphs_v1')
            yield stats
        except Exception:
            self.clear()
            raise
        finally:
            for m,_ in originals:delattr(m,'forward')
            for m,owned,value in previous:
                if owned:setattr(m,policy,value)
                else:delattr(m,policy)
            delattr(base,'_worldstream_graph_owner')
            stats['resident_graphs']=len(self.graphs)


@contextmanager
def zero_mask_decode(lm_head):
    import transformers
    from transformers.models.llama.modeling_llama import LlamaModel
    base=base_model(lm_head);decoder=base.model
    if not transformers.__version__.startswith('4.31.') or not isinstance(decoder,LlamaModel):
        raise ValueError('Only inspected native Transformers 4.31 LLaMA is supported')
    if any(m.training for m in base.modules()):raise ValueError('Decode probe requires eval mode')
    attr='_prepare_decoder_attention_mask';owned=attr in vars(decoder);original=getattr(decoder,attr)
    policy='_worldstream_decode_policy';previous={id(m):(policy in vars(m),getattr(m,policy,None)) for m in (lm_head,base)}
    stats=dict(skipped_zero_masks=0,full_masks=0,all_ones_host_checks=0)
    @wraps(original)
    def prepare(attention_mask,input_shape,inputs_embeds,past_key_values_length):
        if (tuple(input_shape)==(1,1) and past_key_values_length>0 and attention_mask is not None
                and tuple(attention_mask.shape)==(1,past_key_values_length+1)):
            stats['all_ones_host_checks']+=1
            if bool((attention_mask==1).all()):
                stats['skipped_zero_masks']+=1
                return None
        stats['full_masks']+=1
        return original(attention_mask,input_shape,inputs_embeds,past_key_values_length)
    try:
        setattr(decoder,attr,prepare)
        for module in (lm_head,base):setattr(module,policy,'native_zero_mask_v1')
        yield stats
    finally:
        if owned:setattr(decoder,attr,original)
        else:delattr(decoder,attr)
        for module in {id(m):m for m in (lm_head,base)}.values():
            existed,value=previous[id(module)]
            if existed:setattr(module,policy,value)
            else:delattr(module,policy)
