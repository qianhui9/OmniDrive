"""A rejection-only exact-input gate, before expensive world-state validation."""
import torch


class CheapAdmission:
    def __init__(self):
        self.previous = None
        self.identity = None
        self.rows = self.cols = None
        self.version = 0

    def clear(self):
        self.previous = None
        self.identity = None
        self.rows = self.cols = None

    @staticmethod
    def required_bytes(images, pixel_window):
        """Owned samples, next samples, comparison workspace and index arrays."""
        h, w = images.shape[-2:]; ph, pw = pixel_window
        samples = ((h + ph - 1)//ph)*((w + pw - 1)//pw)*9
        return samples*(len(images)*images.shape[1]*(3*images.element_size()+2)+16)

    def check(self, images, context, pixel_window):
        """Snapshot only rejection samples. A possible window is NEVER an equality witness.

        Finite values at three deterministic rows/columns per window are necessary
        for whole-window equality. One differing sample rules out that window.
        The sole device->host transfer is the scalar final rejection decision.
        """
        if images.ndim != 4 or len(set(context.get('camera_ids',[]))) != len(images):
            self.clear()
            return dict(reject=True,reason='unsupported_gate_input',sample_acceptance=False)
        identity=(context.get('scene_id'),tuple(context.get('camera_ids',[])),
                  context.get('execution_signature'),tuple(images.shape),tuple(images.stride()),
                  images.dtype,images.device,pixel_window)
        compatible=self.previous is not None and self.identity==identity
        if not compatible:
            h,w=images.shape[-2:];ph,pw=pixel_window
            rows=[];cols=[]
            for y in range(0,h,ph):
                for x in range(0,w,pw):
                    yy=(y,(y+min(y+ph,h)-1)//2,min(y+ph,h)-1)
                    xx=(x,(x+min(x+pw,w)-1)//2,min(x+pw,w)-1)
                    rows.append([a for a in yy for b in xx]);cols.append([b for a in yy for b in xx])
            self.rows=torch.tensor(rows,device=images.device);self.cols=torch.tensor(cols,device=images.device)
        # Advanced indexing creates owned storage. Unsampled pixels have never
        # contributed to this rejection-only gate; exact execution still compares
        # the full input against its independent feature-cache source.
        current=images[:,:,self.rows,self.cols]
        if compatible:
            invalid=((current!=self.previous)|~torch.isfinite(current)).any(dim=1).any(dim=-1)
            rejected=bool(invalid.all().item())
            reason='all_windows_have_changed_samples' if rejected else 'full_verification_required'
        else:
            rejected=True;reason='gate_no_compatible_history'
        self.previous=current.detach();self.identity=identity;self.version+=1
        return dict(reject=rejected,reason=reason,sample_acceptance=False,snapshot_version=self.version,
                    snapshot_bytes=self.previous.numel()*self.previous.element_size(),
                    index_bytes=(self.rows.numel()+self.cols.numel())*self.rows.element_size(),
                    snapshot_kind='rejection_samples_only')
