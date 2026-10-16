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

    def publish(self,images,features,context,signature,changed):
        # Build owned allocations before publishing anything. OOM cannot publish
        # a new frame with old features or half-updated source provenance.
        owned_images=layout_copy(images.detach())
        owned_features=layout_copy(features.detach())
        sources=[]
        for i in range(len(images)):
            sources.append(dict(camera_id=context['camera_ids'][i],
                computation_frame=context['frame_id'],computation_timestamp=context.get('timestamp'))
                if changed[i] else dict(self.sources[i]))
            sources[-1].update(verification_frame=context['frame_id'],verification_timestamp=context.get('timestamp'))
        owned_context=dict(context);owned_context['camera_ids']=list(context['camera_ids'])
        self.images=owned_images;self.features=owned_features;self.sources=sources
        self.context=owned_context;self.signature=signature
        self.input_stride=tuple(images.stride());self.output_stride=tuple(features.stride())
        self.version+=1

    @property
    def bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.images,self.features) if t is not None)
