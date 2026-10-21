"""One conditional local inverse-depth model, fitted without offline truth.

Corner support and axis holdout are disjoint actual pixel measurements. They
share the same RGB/flow, so they do not constitute independent physical truth.
Only the tested center pixel can be admitted; no patch or hull is filled.
"""
import cv2
import numpy as np
from .surface_observability import PatchObservation, measured_world, project


class GeometryAdaptivePatch(PatchObservation):
    def fit(self, other, uv, world, source_map, candidate_mask):
        uv = np.asarray(uv, float); world = np.asarray(world, float)
        n = len(uv); radius = self.config.support_patch_radius_px + 1
        offsets = radius * np.array([[-1,-1], [1,-1], [-1,1], [1,1], [-1,0], [1,0], [0,-1], [0,1]], float)
        grid = uv[:,None] + offsets
        h, w = source_map.shape[:2]
        xy = np.rint(grid).astype(int)
        inside = ((xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h))
        x = np.clip(xy[...,0], 0, w-1); y = np.clip(xy[...,1], 0, h-1)
        source = source_map[y,x]
        m = measured_world(other.camera, self.camera, source.reshape(-1,2), grid.reshape(-1,2))
        _, depth = project(m['world'], self.camera)
        _, center_depth = project(world, self.camera)
        with np.errstate(divide='ignore', invalid='ignore'):
            inverse = (1/depth).reshape(n,8)
            intercept = 1/center_depth
            slopes = (inverse[:,:4]-intercept[:,None]) @ np.linalg.pinv(offsets[:4]).T
        predicted_inverse = intercept[:,None] + slopes @ offsets.T
        rays = np.concatenate((grid, np.ones((n,8,1))), axis=2) @ np.linalg.inv(self.camera.intrinsic).T
        with np.errstate(divide='ignore', invalid='ignore'):
            cam_points = rays/predicted_inverse[...,None]
        fitted_world = (cam_points-self.camera.world_to_camera[:3,3]) @ self.camera.world_to_camera[:3,:3]
        predicted, _ = project(fitted_world, other.camera)
        residual = np.linalg.norm(predicted-source, axis=2)
        valid = m['valid'].reshape(n,8)&inside&candidate_mask[y,x]
        applicable = (valid.all(1)&np.isfinite(slopes).all(1)&np.isfinite(residual).all(1)
            &(predicted_inverse>.005).all(1)&(predicted_inverse<2).all(1)
            &(residual<=.5).all(1)&np.isfinite(intercept)&(intercept>.005)&(intercept<2))
        return dict(slopes=slopes, applicable=applicable, holdout_residual_px=residual[:,4:],
            support_residual_px=residual[:,:4], support_valid=valid, offsets=offsets)

    def compare(self, other, uv, world, geometry=None):
        # The original path remains exact and available for the ablation.
        baseline = super().compare(other, uv, world)
        if geometry is None:
            return baseline
        slopes = np.asarray(geometry['slopes'])
        adapted = super().compare(other, uv, world, inverse_depth_slopes=slopes)
        usable = geometry['applicable']
        supported = baseline['supported'] | (usable & adapted['supported'])
        return dict(**{k:v for k,v in adapted.items() if k!='supported'}, supported=supported,
            frontoparallel_supported=baseline['supported'],
            center_observation_supported=baseline['observed'], patch_geometry_supported=supported,
            patch_model_applicable=usable, patch_visibility_unknown=~supported,
            patch_conflict_observed=np.zeros(len(uv),bool),
            patch_rejection_reason=np.where(supported,'supported',np.where(usable,'appearance_not_supported','local_geometry_unverified')),
            local_inverse_depth_supported=usable&adapted['supported'])


def measured_third_correspondence(current, view, target, predicted, config):
    """Hypothesis-independent LK, with measured local spatial holdout.

    No predicted hypothesis initializes LK. A missing/occluded/low-texture track
    returns unknown. The affine is only a track validation, not a second patch
    geometry acceptance model. Both acquisitions remain correlated evidence.
    """
    from .identity_temporal import _lk
    from .surface_observability import gray
    target = np.asarray(target, float); n = len(target)
    r = config.support_patch_radius_px + 1
    offsets = r*np.array([[0,0],[-1,-1],[1,-1],[-1,1],[1,1],[-1,0],[1,0],[0,-1],[0,1]],float)
    points = target[:,None] + offsets
    track, fb, valid = _lk(gray(current),gray(view),points.reshape(-1,2).astype(np.float32))
    track=track.reshape(n,9,2); fb=fb.reshape(n,9); valid=valid.reshape(n,9)
    design=np.c_[offsets,np.ones(9)]
    affine=np.einsum('ij,njk->nik',np.linalg.pinv(design[1:5]),track[:,1:5])
    fitted=np.einsum('ij,njk->nik',design,affine)
    residual=np.linalg.norm(fitted-track,axis=2)
    h,w=view.image.shape[:2]
    visible=(track[...,0]>=1)&(track[...,0]<w-1)&(track[...,1]>=1)&(track[...,1]<h-1)
    reliable=(valid.all(1)&visible.all(1)&(fb<=.35).all(1)&(residual[:,[0,5,6,7,8]]<=.5).all(1))
    measured=measured_world(current,view,target,track[:,0])
    reliable &= measured['valid'] & (np.linalg.norm(track[:,0]-predicted,axis=1)<=.5)
    return dict(source_uv=track[:,0],supported=reliable,world=measured['world'],
        fb=fb[:,0],holdout_residual_px=residual[:,5:],
        dependency='Direct current-to-third RGB LK; not the original DIS flow or a statistical independence claim')
