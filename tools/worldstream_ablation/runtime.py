"""Attach experiments only after strict, original checkpoint loading."""
from dataclasses import asdict
from functools import wraps
from .registry import VARIANTS


def attach(model, name, semantic_threshold=.95):
    from .state import AlternativeState
    from .vision import ExperimentalVision
    from .language import LanguagePolicy
    spec = VARIANTS[name]
    if name == 'worldstream':
        return dict(variant=asdict(spec), production_full_R0_unchanged=True)
    model.worldstream_require_m123 = False
    original_m3 = model.m3_runtime
    if spec.ogsm == 'off':
        model.m1_manager = None
    elif spec.ogsm != 'full':
        model.m1_manager = AlternativeState(model.m1_manager.config,spec.ogsm)
    if spec.davp == 'off':
        model.m2_runtime = None
    elif spec.davp != 'full':
        model.m2_runtime = ExperimentalVision(model.img_backbone,model.m2_runtime.config,spec.davp)
    if isinstance(model.m1_manager,AlternativeState):
        # The full DAVP implementation consumes the chosen state engine's masks
        # and region evidence with actual M2 source/cache lineage.
        runtime = model.m2_runtime
        model.m1_manager.runtime = runtime
        original = runtime.forward_from_m1
        @wraps(original)
        def bridge(images,manager,context):
            runtime._ablation_images = images
            return original(images,manager,context)
        runtime.forward_from_m1 = bridge
        model.m3_runtime = LanguagePolicy(original_m3,'alternative_state',model.m1_manager)
    elif spec.qcse == 'off':
        model.m3_runtime = None
    elif spec.qcse != 'full':
        model.m3_runtime = LanguagePolicy(original_m3,spec.qcse,
                                         semantic_threshold=semantic_threshold)
    return dict(variant=asdict(spec),production_full_R0_unchanged=False,
        isolated_experiment_adapter=True,semantic_threshold=semantic_threshold,
        m1_present=model.m1_manager is not None,m2_present=model.m2_runtime is not None,
        m3_present=model.m3_runtime is not None)
