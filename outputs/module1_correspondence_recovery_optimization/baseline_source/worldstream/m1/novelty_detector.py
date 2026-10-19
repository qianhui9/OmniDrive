"""Observed stable/changed/unknown partition; refresh is a separate policy output."""
import numpy as np
from .geometry import region_slices
from .mask_codec import encode_mask


def unexplained_regions(camera, certified_regions, changed_regions, config, force_refresh=False,
                        temporal_evidence=None):
    h, w = camera.image.shape[:2]
    explained = np.zeros((h, w), dtype=bool)
    for region in certified_regions:
        explained[region_slices(region, camera.image.shape)] = True
    # Object bounding rectangles include unmodelled background: conservatively never explain
    # them unless the caller has supplied dense reliable geometric support for those pixels.
    if camera.geometry_support_mask is None:
        explained[:] = False
    elif camera.geometry_support_mask.shape == (h, w):
        explained &= camera.geometry_support_mask.astype(bool)
    else:
        explained[:] = False
    changed = np.zeros((h, w), dtype=bool)
    if config.local_support:
        # Same-RGB local correspondence can survive a failed whole-anchor check.
        # No rectangle is used to mark stable pixels.
        explained[:] = False
        if temporal_evidence is not None:
            if (temporal_evidence.camera_id != camera.camera_id
                    or temporal_evidence.timestamp != camera.timestamp
                    or temporal_evidence.stable_mask.shape != (h, w)
                    or temporal_evidence.changed_mask.shape != (h, w)):
                raise ValueError('Temporal evidence frame/camera/mask mismatch')
            explained |= temporal_evidence.stable_mask
            changed |= temporal_evidence.changed_mask
    else:
        for region in changed_regions:
            changed[region_slices(region, camera.image.shape)] = True
    explained &= ~changed
    unknown = ~(explained | changed)
    invalid = changed | unknown
    observation_refresh_ratio = float(invalid.mean())
    if force_refresh:
        invalid[:] = True
    if temporal_evidence is not None and temporal_evidence.stable_candidate_mask is not None:
        temporal_evidence.refresh_required_mask = invalid.copy()
    # Grid is a conservative compact representation, with exact mask available separately.
    regions = []
    step = config.novelty_cell_px
    if invalid.all():
        regions = [(0, 0, w, h)]
    else:
        for y in range(0, h, step):
            for x in range(0, w, step):
                if invalid[y:y + step, x:x + step].any():
                    regions.append((x, y, min(w, x + step), min(h, y + step)))
    return {
        "coordinate_space": camera.coordinate_space,
        "raw_to_image": None if camera.raw_to_image is None else camera.raw_to_image.tolist(),
        "image_shape": [h, w], "regions": regions,
        "changed_regions": changed_regions,
        "unknown_uncovered_area_ratio": observation_refresh_ratio,  # legacy changed|unknown
        "observation_refresh_area_ratio": observation_refresh_ratio,
        "verified_stable_area_ratio": float(explained.mean()),
        "changed_area_ratio": float(changed.mean()),
        "unknown_area_ratio": float(unknown.mean()),
        "invalidation_area_ratio": float(invalid.mean()),
        "refresh_override": bool(force_refresh),
        "evidence_source": temporal_evidence.source if temporal_evidence is not None else "external_or_missing",
        "transformer_dependency_certified": False,
        "observation_stable_mask": encode_mask(explained),
        "observed_changed_mask": encode_mask(changed),
        "unknown_mask": encode_mask(unknown),
        "refresh_required_mask": encode_mask(invalid),
        "exact_invalidation_mask": encode_mask(invalid),
        "effective_stable_area_ratio": float((~invalid).mean()),
        "stable_semantics": "local_observation_correspondence_not_physical_object_truth",
        "regions_are_conservative_tile_envelope": True,
        "reason": "periodic_or_reset_refresh" if force_refresh else "uncovered_or_changed",
        "feature_mapping": "unknown; no Transformer token indices",
    }, invalid
