"""R0-required utilities; historical alternative runners have been retired."""

import copy

import gc

import torch

from .common import diagnostic_json, write

from .preflight import BASELINE, FINAL

def effective_config(condition, workload, dest, m3_strategy=None):
    from mmcv import Config
    if condition not in ('B0','B3'):raise ValueError('Only Native and full R0 are supported')
    cfg = Config.fromfile(BASELINE if condition == 'B0' else FINAL)
    common = Config.fromfile(FINAL)
    cfg.data.test.pipeline = copy.deepcopy(common.data.test.pipeline)
    if m3_strategy is not None:
        if condition!='B3':raise ValueError('M3 comparison retains formal M1/M2 in B3')
        if m3_strategy=='native_full':
            cfg.model.worldstream_require_m123=False
            cfg.model.m3_enable=False
        else:raise ValueError('Only frozen R0 or Native language baseline is supported')
    for node in cfg.data.test.pipeline:
        if node['type'] == 'LoadAnnoatationVQATest':
            node['load_type'] = ['conv'] if workload == 'language' else ['planning']
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.save_path = str(dest/'answers')+'/'
    cfg.data.test.test_mode = True
    cfg.data.workers_per_gpu = 0
    write(dest/'effective_config.json', dict(config=cfg._cfg_dict.to_dict(),
        note='Temporary evaluation overrides only: output path, shared metadata pipeline, worker count and language workload; official source configs unchanged.'))
    return cfg


def load_model(cfg, checkpoint, dest):
    from mmdet3d.models import build_model
    from mmcv.runner import wrap_fp16_model
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    checkpoint_data = torch.load(checkpoint, map_location='cpu')
    state = checkpoint_data.get('state_dict', checkpoint_data)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    expected = model.state_dict()
    missing = sorted(set(expected)-set(state))
    unexpected = sorted(set(state)-set(expected))
    derived_buffers = []
    for key in unexpected:
        parent, _, leaf = key.rpartition('.')
        try:
            owner = model.get_submodule(parent)
        except AttributeError:
            continue
        # The deployed transformers version registers regenerated RoPE caches
        # as nonpersistent. Match mmcv's native load behavior only for those
        # actual registered buffers; never waive a learned parameter mismatch.
        if (leaf in ('cos_cached', 'sin_cached') and leaf in owner._buffers
                and leaf in owner._non_persistent_buffers_set):
            derived_buffers.append(key)
    unexpected = sorted(set(unexpected)-set(derived_buffers))
    shapes = {k: dict(expected=list(expected[k].shape), actual=list(state[k].shape))
              for k in set(expected) & set(state) if expected[k].shape != state[k].shape}
    validation = dict(status='PASS' if not (missing or unexpected or shapes) else 'FAIL',
        checkpoint=checkpoint, expected_keys=len(expected), actual_keys=len(state),
        missing_keys=missing, unexpected_keys=unexpected, shape_mismatches=shapes,
        ignored_nonpersistent_rotary_caches=derived_buffers,
        random_initialization_accepted=False,
        metadata=diagnostic_json(checkpoint_data.get('meta', {})))
    # Large training metadata may contain Config objects; retain a readable repr.
    validation['metadata'] = repr(checkpoint_data.get('meta', {}))
    write(dest/'checkpoint_validation.json', validation)
    if validation['status'] != 'PASS':
        raise RuntimeError('Strict checkpoint compatibility failed; see checkpoint_validation.json')
    model.load_state_dict({k: v for k, v in state.items() if k not in derived_buffers}, strict=True)
    del expected, state, checkpoint_data
    gc.collect()
    return model.cuda().eval()


def reset(model):
    for head in (model.pts_bbox_head, model.map_head):
        head.reset_memory()
    for runtime in (model.m1_manager, model.m2_runtime, model.m3_runtime):
        if runtime is not None:
            runtime.reset()
