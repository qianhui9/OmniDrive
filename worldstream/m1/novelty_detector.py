"""Only certified local geometry explains pixels. Everything else stays invalidated."""
import numpy as np
from .geometry import region_slices


def unexplained_regions(camera, certified_regions, changed_regions, config, force_refresh=False):
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
    invalid = ~explained
    for region in changed_regions:
        invalid[region_slices(region, camera.image.shape)] = True
    ratio = float(invalid.mean())
    if force_refresh:
        invalid[:] = True
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
        "unknown_uncovered_area_ratio": ratio,
        "reason": "periodic_or_reset_refresh" if force_refresh else "uncovered_or_changed",
        "feature_mapping": "unknown; no Transformer token indices",
    }, invalid
