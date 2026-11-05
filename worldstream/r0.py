"""Full frozen R0 parameter authority for the sole research execution route."""
import copy
import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / 'configs/worldstream_r0.json'


def model_settings():
    return json.loads(CONFIG.read_text())


def m3_settings(policy_path=None):
    settings = copy.deepcopy(model_settings()['m3_config'])
    if policy_path is not None:
        settings['selective_answer_policy'] = str(policy_path)
    return settings


def validate_fact_policy(policy):
    authority = json.loads((CONFIG.parents[1] /
        'outputs/m123_final_quality_validation_run01/reference_policy.json').read_text())
    if policy != authority:
        raise ValueError('Only the complete frozen R0 C-basic fact policy is supported')


def validate_runtime_settings(m1_enable, m1_mode, m1, m2_enable, m2, m3_enable, m3):
    """Reject CLI/config overrides that would silently change the frozen route."""
    from .m1 import M1Config
    expected = model_settings()
    actual_m1 = M1Config.load(m1).to_dict()
    if not (m1_enable and m2_enable and m3_enable):
        raise ValueError('Full R0 requires all M1/M2/M3 modules')
    if (m1_mode != expected['m1_execution_mode'] or actual_m1 != expected['m1_config']
            or dict(m2) != expected['m2_config']):
        raise ValueError('Only frozen R0 M1/M2 settings are supported')
    from .m3.config import M3Config
    M3Config(**dict(m3))
