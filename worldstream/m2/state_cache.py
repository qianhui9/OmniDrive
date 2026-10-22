"""Only the immediately preceding frame and early independent block outputs live here."""
import copy
import math
import torch


class NonFiniteCacheError(RuntimeError):
    """A non-finite input or prefix cannot become a reusable source."""


class CachePublicationError(RuntimeError):
    """Expected resource failure; programming/interface errors must propagate."""

class StateCache:
    def __init__(self, budget_bytes):
        self.budget_bytes = budget_bytes
        self.version = 0
        self.clear()

    def clear(self):
        self.images = None
        self.block0_input = None
        self.outputs = {}
        self.context = None
        self.signature = None
        self.output_stride = None
        self.input_stride = None
        self.sources = {}
        self.storage_revision = None
        self.metadata_revision = None
        self.region_states = []
        self.window_ages = {}
        self.descriptors = None

    @staticmethod
    def _revision(images, block0_input, outputs):
        return tuple((name, value.data_ptr(), value._version, tuple(value.shape),
                      tuple(value.stride()), value.dtype, value.device)
                     for name,value in [('images',images),('block0_input',block0_input)]+[(f'block:{i}',t) for i,t in sorted(outputs.items())])

    def commit(self, images, outputs, context, signature, output_stride=None, sources=None, *, block0_input,
               region_states=(), window_ages=None, descriptors=None):
        """Publish a single completed forward atomically; retained outputs are immutable."""
        window_ages = window_ages or {}
        values = [images, block0_input, *outputs.values(), *window_ages.values()]
        if descriptors is not None: values.append(descriptors)
        needed = sum(t.numel()*t.element_size() for t in values)
        if needed > self.budget_bytes:
            raise CachePublicationError('window_cache_budget_exceeded')
        finite=[torch.isfinite(value).all() for value in values]
        if not torch.stack(finite).all().item():
            raise NonFiniteCacheError('nonfinite_cache_features')
        owned_images = images.detach().clone()
        owned_input = block0_input.detach().clone()
        owned_outputs = {i: value.detach().clone() for i,value in outputs.items()}
        owned_context = copy.deepcopy(context)
        owned_regions = copy.deepcopy(region_states)
        owned_ages = {i:t.detach().clone() for i,t in window_ages.items()}
        owned_descriptors = descriptors.detach().clone() if descriptors is not None else None
        owned_sources = copy.deepcopy(sources) if sources is not None else {
            i: [dict(computation_frame_id=context['frame_id'], computation_timestamp=context.get('timestamp'),
                computation_version=self.version+1, camera_id=context['camera_ids'][j//(len(values)//len(images))],
                producer_layer=i, dependency_scope='closed_window_prefix',task_quality_certified=False) for j in range(len(values))]
            for i,values in outputs.items()}
        input_stride = tuple(images.stride())
        output_stride = tuple(output_stride) if output_stride is not None else None
        storage_revision = self._revision(owned_images, owned_input, owned_outputs) + self._policy_revision(owned_descriptors,owned_ages)
        metadata_revision = repr((owned_context,owned_sources,signature,input_stride,output_stride,self.version+1,owned_regions))
        # All potentially failing work precedes publication, including ownership,
        # metadata cloning and integrity bookkeeping.
        (self.images,self.block0_input,self.outputs,self.context,self.sources,self.signature,
         self.input_stride,self.output_stride,self.version,self.storage_revision,self.metadata_revision,
         self.region_states,self.window_ages,self.descriptors) = (
            owned_images,owned_input,owned_outputs,owned_context,owned_sources,signature,
            input_stride,output_stride,self.version+1,storage_revision,metadata_revision,
            owned_regions,owned_ages,owned_descriptors)

    @staticmethod
    def _policy_revision(descriptors, ages):
        values = ([descriptors] if descriptors is not None else []) + [t for _,t in sorted(ages.items())]
        return tuple((t.data_ptr(),t._version,tuple(t.shape),str(t.dtype),str(t.device)) for t in values)

    @property
    def bytes(self):
        values = list(self.outputs.values()) + ([self.images,self.block0_input] if self.images is not None else [])
        values += list(self.window_ages.values()) + ([self.descriptors] if self.descriptors is not None else [])
        return sum(x.numel()*x.element_size() for x in values)

    def mismatch(self, images, context, signature, block0_input):
        if self.images is None or self.block0_input is None:
            return 'no_valid_cache'
        if self.metadata_revision != repr((self.context,self.sources,self.signature,self.input_stride,self.output_stride,self.version,self.region_states)):
            return 'cache_metadata_modified'
        if self.storage_revision is not None and self.storage_revision != self._revision(self.images,self.block0_input,self.outputs)+self._policy_revision(self.descriptors,self.window_ages):
            return 'cache_storage_modified'
        if self.signature != signature:
            return 'execution_signature_mismatch'
        if self.images.shape != images.shape or self.images.device != images.device or self.images.dtype != images.dtype:
            return 'shape_unsupported'
        if (self.block0_input.shape != block0_input.shape or self.block0_input.device != block0_input.device
                or self.block0_input.dtype != block0_input.dtype or self.block0_input.stride() != block0_input.stride()):
            return 'execution_signature_mismatch'
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
