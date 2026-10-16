import torch
from worldstream.m2.dependency_planner import DependencyPlanner, patch_updates
from worldstream.m2.packing import partition, unpartition


def test_full_patch_footprint(eva):
    invalid = torch.zeros(2, 40, 40, dtype=torch.bool)
    invalid[0, 0, 0] = True  # Far from token centre, still invalidates it.
    invalid[0, 7, 7] = True
    updated = patch_updates(invalid, eva.patch_embed.proj)
    assert updated.sum() == 2 and updated[0, 0, 0] and updated[0, 1, 1]


def test_window_and_global(eva):
    changed = torch.zeros(2, 10, 10, dtype=torch.bool)
    changed[0, 3, 3] = True
    plans = DependencyPlanner(eva).plan(changed)
    assert plans[0].updated.sum() == plans[1].updated.sum() == 1
    assert plans[0].tokens.sum() == 16
    assert plans[2].tokens[0].all() and not plans[2].tokens[1].any()
    assert plans[3].updated.sum() == 9


def test_conv_crosses_windows(eva):
    eva.blocks[0].use_residual_block = True
    changed = torch.zeros(2, 10, 10, dtype=torch.bool)
    changed[0, 3, 3] = True
    plans = DependencyPlanner(eva).plan(changed)
    assert plans[0].tokens[0, 4, 4]
    assert plans[1].updated.sum() == 4
    assert DependencyPlanner(eva).cacheable_prefix() == []


def test_packing_padding_and_camera_order():
    x = torch.arange(2*5*7*3).reshape(2, 5, 7, 3)
    windows, padded = partition(x, 4)
    assert windows.shape == (8, 4, 4, 3)
    assert torch.equal(unpartition(windows, 4, padded, (5, 7)), x)
    assert windows[4, 0, 0, 0] == x[1, 0, 0, 0]
