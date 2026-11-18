"""One shared full-system reference; 14 unique executions for 17 table rows."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    ogsm: str = 'full'
    davp: str = 'full'
    qcse: str = 'full'


VARIANTS = {v.name: v for v in (
    Variant('native', 'Native VLM', 'off', 'off', 'off'),
    Variant('worldstream', 'WorldStream / Full OGSM / Full DAVP / Full QCSE'),
    Variant('without_ogsm', 'w/o OGSM', 'off', 'image_difference', 'candidate_only'),
    Variant('without_davp', 'w/o DAVP', davp='off'),
    Variant('without_qcse', 'w/o QCSE', qcse='off'),
    Variant('ogsm_tracking', 'Tracking-based State', ogsm='tracking'),
    Variant('ogsm_geometry', 'Geometry-based Validation', ogsm='geometry'),
    Variant('ogsm_flow', 'Flow-based Validation', ogsm='flow'),
    Variant('davp_periodic', 'Periodic Window Refresh', davp='periodic'),
    Variant('davp_feature_drift', 'Feature-Drift-based Refresh', davp='feature_drift'),
    Variant('davp_region_mapping', 'Region-to-Window Mapping', davp='region_mapping'),
    Variant('qcse_exact_cache', 'Exact Answer Cache', qcse='exact_cache'),
    Variant('qcse_semantic_cache', 'Semantic Answer Cache', qcse='semantic_cache'),
    Variant('qcse_prompt_kv', 'Prompt/KV Cache', qcse='prompt_kv'),
)}
GROUPS = dict(
    module=['native', 'worldstream', 'without_ogsm', 'without_davp', 'without_qcse'],
    ogsm=['ogsm_tracking', 'ogsm_geometry', 'ogsm_flow', 'worldstream'],
    davp=['davp_periodic', 'davp_feature_drift', 'davp_region_mapping', 'worldstream'],
    qcse=['qcse_exact_cache', 'qcse_semantic_cache', 'qcse_prompt_kv', 'worldstream'])


def select(group='all'):
    requested = list(VARIANTS) if group == 'all' else GROUPS[group]
    # Native provides actual executed-compute denominators even for design-only runs.
    return list(dict.fromkeys(['native', *requested]))


def inventory():
    return dict(variants=[asdict(v) for v in VARIANTS.values()], tables=GROUPS,
                baseline='native', full_reference='worldstream')
