"""Actual native multimodal expansion, with explicit token provenance (batch one)."""
from dataclasses import dataclass, replace
import hashlib
import torch

IMAGE_TOKEN_INDEX = -200


@dataclass
class ExpandedInput:
    embeddings: torch.Tensor
    token_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    visual_mask: torch.Tensor
    scope: tuple = ()
    signature: tuple = ()
    source: tuple = ()

    @property
    def length(self):
        return self.embeddings.shape[1]

    def prefix(self, length, clone=False):
        fields = {}
        for name in ('embeddings', 'token_ids', 'position_ids', 'attention_mask', 'visual_mask'):
            value = getattr(self, name)[:, :length]
            fields[name] = value.detach().clone() if clone else value
        return replace(self, **fields)

    @property
    def nbytes(self):
        return sum(getattr(self, n).numel() * getattr(self, n).element_size() for n in
                   ('embeddings', 'token_ids', 'position_ids', 'attention_mask', 'visual_mask'))


@dataclass(frozen=True)
class CarrierLayout:
    object_tokens: int
    ego_tokens: int
    map_tokens: int

    @property
    def total(self):
        return self.object_tokens + self.ego_tokens + self.map_tokens


def base_model(lm_head):
    # PEFT LoRA leaves the injected layers in the returned original model.
    return lm_head.get_base_model() if hasattr(lm_head, 'get_base_model') else lm_head


def execution_signature(lm_head, model_version=''):
    base = base_model(lm_head)
    tensors = tuple((n, id(t), t.data_ptr(), t._version, tuple(t.shape), str(t.dtype), str(t.device))
                    for n, t in list(base.named_parameters()) + list(base.named_buffers())
                    if not n.endswith(('cos_cached', 'sin_cached')))
    # RoPE cache buffers grow during a forward, but their computed values are
    # determined by config. All trainable state and LoRA flags are included.
    adapters = tuple((n, repr(getattr(m, 'active_adapters', None)),
                      repr(getattr(m, 'disable_adapters', None)), repr(getattr(m, 'merged_adapters', None)),
                      repr(getattr(m, 'scaling', None)))
                     for n, m in base.named_modules() if hasattr(m, 'lora_A'))
    backends = tuple((n, type(m).__module__, type(m).__qualname__,
                      id(getattr(type(m), 'forward', None)),
                      id(getattr(m.forward, '__func__', m.forward)))
                     for n, m in base.named_modules())
    return (id(base), model_version, repr(base.config.to_dict()), tensors, adapters, backends,
            torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype(),
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def deployment_key(lm_head):
    """Stable profiler compatibility key; weights are identified separately by SHA/version.

    Cache validity itself uses the live versioned signature above. This stable key
    prevents loading a cost/quality certificate for a different backend or GPU.
    """
    import transformers
    base = base_model(lm_head)
    device = base.get_input_embeddings().weight.device
    hardware = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
    description = (base.config.to_dict(), torch.__version__, transformers.__version__, hardware,
        [(n, type(m).__module__, type(m).__qualname__) for n, m in base.named_modules()],
        [(n, tuple(p.shape), str(p.dtype), str(p.device)) for n, p in base.named_parameters()],
        torch.is_autocast_enabled(), str(torch.get_autocast_gpu_dtype()),
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    return hashlib.sha256(repr(description).encode()).hexdigest()


@torch.no_grad()
def expand_input(lm_head, input_ids, images, attention_mask=None, position_ids=None,
                 scope=(), signature=(), source=()):
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('M3 supports nonempty batch size one requests')
    base = base_model(lm_head)
    if images is None:
        if (input_ids < 0).any():
            raise ValueError('Image marker requires image features')
        embeds = base.get_input_embeddings()(input_ids)
        ids = input_ids.to(embeds.device)
        visual = torch.zeros_like(ids, dtype=torch.bool)
        positions = position_ids
        mask = attention_mask
    else:
        if input_ids.shape[1] == 1 and (input_ids == IMAGE_TOKEN_INDEX).any():
            raise ValueError('Native one-token image marker bypass is unsupported')
        _, positions, mask, _, embeds, _ = base.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, None, None, images, None)
        if embeds is None:
            embeds = base.get_input_embeddings()(input_ids)
        raw = input_ids[0] if attention_mask is None else input_ids[0][attention_mask[0].bool().to(input_ids.device)]
        count = images.reshape(images.shape[0], -1, base.config.hidden_size).shape[1]
        chunks = [t.reshape(1) if t.item() != IMAGE_TOKEN_INDEX else
                  raw.new_full((count,), IMAGE_TOKEN_INDEX) for t in raw]
        ids = torch.cat(chunks).unsqueeze(0).to(embeds.device)
        visual = ids == IMAGE_TOKEN_INDEX
    if ids.shape[1] != embeds.shape[1]:
        raise ValueError('Native expansion and provenance disagree')
    if mask is None:
        mask = torch.ones_like(ids)
    else:
        mask = mask.to(embeds.device)
    if positions is None:
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
    else:
        positions = positions.to(embeds.device)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
    if positions.shape != ids.shape or mask.shape != ids.shape:
        raise ValueError('Unsupported mask or position shape')
    if not torch.isfinite(embeds).all():
        raise ValueError('Nonfinite multimodal embeddings')
    return ExpandedInput(embeds, ids, positions, mask, visual, tuple(scope), signature, tuple(source))
