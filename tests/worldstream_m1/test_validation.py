from dataclasses import replace
import numpy as np
from worldstream.m1 import FullOutputs, WorldStateManager
from worldstream.m1.novelty_detector import unexplained_regions
from worldstream.m1.observation_validator import ObservationValidator


def initialized(config, make_frame, make_prediction):
    manager = WorldStateManager(config)
    manager.initialize(make_frame(), FullOutputs([make_prediction()]))
    return manager


def test_positive_and_changed_observation(config, make_frame, make_prediction):
    manager = initialized(config, make_frame, make_prediction)
    result = manager.precheck(make_frame(100.5))
    assert len(result.valid_anchor_ids) == 1
    result = manager.precheck(make_frame(100.6, change=True))
    assert len(result.invalid_anchor_ids) == 1
    assert not result.valid_anchor_ids


def test_unknown_visibility_invalid_by_default(config, make_frame, make_prediction):
    manager = initialized(config, make_frame, make_prediction)
    result = manager.precheck(make_frame(100.5, support=False))
    check = next(iter(result.diagnostics['anchors'].values()))
    assert not check['valid'] and 'visibility_unknown' in check['reasons']
    assert manager.last_masks['front'].all()


def test_uniform_patch_and_photometric_normalization(config, make_frame):
    validator = ObservationValidator(config)
    assert validator.descriptor(np.full((40, 40, 3), 200, np.uint8), (0, 0, 40, 40)) is None
    image = make_frame().cameras['front'].image
    altered = np.clip(image.astype(float)*.6+20, 0, 255).astype(np.uint8)
    assert validator.residual(validator.descriptor(image, (20, 20, 80, 80)),
                              validator.descriptor(altered, (20, 20, 80, 80))) < .05


def test_new_uncovered_region_always_invalid(config, make_frame):
    camera = make_frame().cameras['front']
    regions, mask = unexplained_regions(camera, [(10, 10, 30, 30)], [], config)
    assert not mask[15, 15] and mask[80, 90]
    regions, mask = unexplained_regions(camera, [(10, 10, 30, 30)], [(10, 10, 20, 20)], config)
    assert mask[15, 15] and regions['changed_regions']
    camera.geometry_support_mask = None
    assert unexplained_regions(camera, [(0, 0, 120, 100)], [], config)[1].all()


def test_multi_camera_all_supported_views_must_agree(config, make_frame, make_prediction):
    import copy
    initial = make_frame(); initial.cameras['side'] = copy.deepcopy(initial.cameras['front'])
    initial.cameras['side'].camera_id = 'side'
    manager = WorldStateManager(config); manager.initialize(initial, FullOutputs([make_prediction()]))
    current = make_frame(100.5)
    current.cameras['side'] = make_frame(100.5, change=True).cameras['front']
    current.cameras['side'].camera_id = 'side'
    assert manager.precheck(current).invalid_anchor_ids


def test_no_current_output_in_precheck_api():
    import inspect
    assert list(inspect.signature(WorldStateManager.precheck).parameters) == ['self', 'frame']


def test_occlusion_by_closer_anchor_is_invalid(config, make_frame, make_prediction):
    from worldstream.m1.geometry import box_geometry
    far = make_prediction(); far.geometry_world = box_geometry([0, 0, 14], [2, 2, 2])
    manager = WorldStateManager(config)
    manager.initialize(make_frame(), FullOutputs([make_prediction(), far]))
    result = manager.precheck(make_frame(100.5))
    assert len(result.invalid_anchor_ids) == 1
    assert any('projected_occlusion' in c['reasons'] for c in result.diagnostics['anchors'].values())
