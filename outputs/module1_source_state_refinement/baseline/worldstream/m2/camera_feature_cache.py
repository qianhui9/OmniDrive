"""One owned scene-local camera batch from actual preceding execution."""
import torch
from .state_cache import StateCache


def layout_copy(tensor):
    out=torch.empty_strided(tensor.shape,tensor.stride(),dtype=tensor.dtype,device=tensor.device)
    return out.copy_(tensor)


class CameraFeatureCache(StateCache):
    def clear(self):
        super().clear()
        self.features=None;self.sources=[]

    def output_sources(self,context,changed):
        sources=[]
        for i,computed in enumerate(changed):
            source=(dict(camera_id=context['camera_ids'][i],computation_frame=context['frame_id'],
                computation_timestamp=context.get('timestamp'),computation_version=self.version+1,
                producer_layer='eva_backbone',dependency_scope='full_camera',source_cache_version=self.version+1)
                if computed else dict(self.sources[i]))
            source.update(verification_frame=context['frame_id'],verification_timestamp=context.get('timestamp'))
            sources.append(source)
        return sources

    def publish(self,images,features,context,signature,changed):
        # Build owned allocations before publishing anything. OOM cannot publish
        # a new frame with old features or half-updated source provenance.
        owned_images=layout_copy(images.detach())
        owned_features=layout_copy(features.detach())
        sources=self.output_sources(context,changed)
        owned_context=dict(context);owned_context['camera_ids']=list(context['camera_ids'])
        self.images=owned_images;self.features=owned_features;self.sources=sources
        self.context=owned_context;self.signature=signature
        self.input_stride=tuple(images.stride());self.output_stride=tuple(features.stride())
        self.version+=1

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.images,self.features) if t is not None)
