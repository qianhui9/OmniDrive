"""Only the immediately preceding frame and early independent block outputs live here."""
import copy
import math
import torch


class NonFiniteCacheError(RuntimeError):
    """A non-finite input or prefix cannot become a reusable source."""

class StateCache:
    def __init__(self, budget_bytes):
        self.budget_bytes = budget_bytes
        self.version = 0
        self.clear()

    def clear(self):
        self.images = None
        self.outputs = {}
        self.context = None
        self.signature = None
        self.output_stride = None
        self.input_stride = None
        self.sources = {}
        self.storage_revision = None
        self.metadata_revision = None

    @staticmethod
    def _revision(images, outputs):
        return tuple((name, value.data_ptr(), value._version, tuple(value.shape),
                      tuple(value.stride()), value.dtype, value.device)
                     for name,value in [('images',images)]+[(f'block:{i}',t) for i,t in sorted(outputs.items())])

    def commit(self, images, outputs, context, signature, output_stride=None, sources=None):
        """Publish a single completed forward atomically; retained outputs are immutable."""
        needed = sum(t.numel()*t.element_size() for t in [images, *outputs.values()])
        if needed > self.budget_bytes:
            raise RuntimeError('window_cache_budget_exceeded')
        finite=[torch.isfinite(value).all() for value in [images,*outputs.values()]]
        if not torch.stack(finite).all().item():
            raise NonFiniteCacheError('nonfinite_cache_features')
        owned_images = images.detach().clone()
        owned_outputs = {i: value.detach().clone() for i,value in outputs.items()}
        owned_context = copy.deepcopy(context)
        owned_sources = copy.deepcopy(sources) if sources is not None else {
            i: [dict(computation_frame_id=context['frame_id'], computation_timestamp=context.get('timestamp'),
                computation_version=self.version+1, camera_id=context['camera_ids'][j//(len(values)//len(images))],
                producer_layer=i, dependency_scope='closed_window_prefix',task_quality_certified=True) for j in range(len(values))]
            for i,values in outputs.items()}
        metadata_revision = repr((owned_context,owned_sources))
        self.images = owned_images
        self.outputs = owned_outputs
        self.context = owned_context
        self.sources = owned_sources
        self.signature = signature
        self.input_stride = tuple(images.stride())
        if output_stride is not None:self.output_stride = tuple(output_stride)
        self.version += 1
        self.storage_revision = self._revision(self.images, self.outputs)
        self.metadata_revision = metadata_revision

    @property
    def bytes(self):
        values = list(self.outputs.values()) + ([self.images] if self.images is not None else [])
        return sum(x.numel()*x.element_size() for x in values)

    def mismatch(self, images, context, signature):
        if self.images is None:
            return 'no_valid_cache'
        if self.metadata_revision != repr((self.context,self.sources)):
            return 'cache_metadata_modified'
        if self.storage_revision is not None and self.storage_revision != self._revision(self.images, self.outputs):
            return 'cache_storage_modified'
        if self.signature != signature:
            return 'execution_signature_mismatch'
        if self.images.shape != images.shape or self.images.device != images.device or self.images.dtype != images.dtype:
            return 'shape_unsupported'
        if self.input_stride != tuple(images.stride()):
            return 'execution_signature_mismatch'
        if self.context.get('coordinate_space') != context.get('coordinate_space'):
            return 'preprocessing_geometry_mismatch'
        if self.context.get('state_epoch') != context.get('state_epoch'):
            return 'cache_state_epoch_mismatch'
        if self.context['scene_id'] != context['scene_id']:
            return 'scene_change'
        if self.context['camera_ids'] != context['camera_ids']:
            return 'execution_signature_mismatch'
        if context.get('source_frame_id', self.context['frame_id']) != self.context['frame_id']:
            return 'cache_version_mismatch'
        if context.get('source_version', self.version) != self.version:
            return 'cache_version_mismatch'
        if context['frame_id'] == self.context['frame_id']:
            return 'frame_order_mismatch'
        now, old = context.get('timestamp'), self.context.get('timestamp')
        if now is not None and (not math.isfinite(now) or old is not None and now <= old):
            return 'frame_order_mismatch'
        return None
