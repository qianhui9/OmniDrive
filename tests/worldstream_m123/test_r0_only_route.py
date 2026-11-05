"""R0 preservation and rejection tests after retiring historical M3 routes."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from worldstream.r0 import model_settings, m3_settings
from worldstream.m3 import M3Config, M3Runtime
from worldstream.m3.answer_memory import AnswerMemory
from worldstream.m3.answer_validity import propose_fact_answer
from worldstream.m3.world_state import CommittedAnchor, CommittedWorldState


def state(score=.8, **overrides):
    anchor = CommittedAnchor('a','object','car',(20.,0.,0.),(),True,
        ('front',),False,False,'f',1.,1,'actual_perception',score)
    return CommittedWorldState('s','f',1.,1,1,'sig',
        (replace(anchor, **overrides),),tuple([1.,0,0,0,0,1.,0,0,0,0,1.,0,0,0,0,1.]))


def context():
    return dict(scene_id='s',frame_id='f',timestamp=1.,state_epoch=1,
        current_state_version=1,state_execution_signature='sig')


def policy():
    return json.loads(Path('outputs/m123_final_quality_validation_run01/reference_policy.json').read_text())


def test_entry_uses_complete_frozen_r0():
    from mmcv import Config
    cfg = Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    settings = model_settings()
    for key,value in settings.items():
        assert cfg.model[key] == value
    authority=json.loads(Path('outputs/m123_final_quality_validation_run01/final_research_policy.json').read_text())
    assert settings['m1_config'] == authority['M1']
    assert settings['m2_config'] == authority['M2']
    assert asdict(M3Config()) == settings['m3_config']


@pytest.mark.parametrize('field,value',[
    ('reuse_strategy','wsdr_kv'),('decode_strategy','native'),
    ('decode_strategy','history_draft'),('prefix_layout','sf_adaptive'),
    ('selective_answer_mode','off'),('selective_answer_mode','shadow'),
    ('target_aware_exact_only',False),('experimental_speculative_sampling',False),
    ('question_rng_enabled',False),('target_binding_enabled',False),
    ('temporal_risk_enabled',True),('target_completeness_enabled',True),
    ('current_detection_support',False),('draft_length',4),
])
def test_retired_or_non_r0_configuration_is_rejected(field,value):
    cfg=m3_settings();cfg[field]=value
    with pytest.raises(ValueError,match='Only frozen R0'):
        M3Config(**cfg)


def test_policy_threshold_change_is_rejected(tmp_path):
    changed=policy();changed['min_confidence']=.6
    path=tmp_path/'policy.json';path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='complete frozen R0'):
        M3Runtime(m3_settings(path))


@pytest.mark.parametrize('question',[
    'What type of vehicle is directly ahead?', 'Is there a car in front?'])
def test_current_facts_need_no_answer_history(question):
    class ForbiddenHistory:
        @property
        def records(self):raise AssertionError('Current-only accessed historical answers')
    result=propose_fact_answer(question,state(),context(),ForbiddenHistory(),('model',),
        policy(),target_binding_enabled=True)
    assert result['admitted'] and result['answer_source']=='fact_grounded_response'
    assert result['history_candidate_count']==0


@pytest.mark.parametrize('change',[
    dict(observed=False),dict(source_frame='old'),dict(source_timestamp=.5),
    dict(evidence_origin='annotation'),dict(cameras=()),dict(score=.4)])
def test_missing_or_weak_current_witness_cannot_bypass(change):
    w=state(**change)
    result=propose_fact_answer('Is there a car in front?',w,context(),AnswerMemory(),
        ('model',),policy(),target_binding_enabled=True)
    assert not result['admitted']


def test_runtime_has_no_prefix_planner_or_persistent_kv():
    runtime=M3Runtime()
    assert not hasattr(runtime,'planner') and not hasattr(runtime,'cache')
    assert not hasattr(runtime,'composer')


def test_retired_implementation_files_are_absent():
    for name in ['reuse_planner','scheduler','prefix_matcher','prefix_kv_cache',
                 'world_prefix_composer','stable_world','semantic_core','visual_memory']:
        assert not (Path('worldstream/m3')/(name+'.py')).exists()


def test_model_boundary_rejects_changed_m1_m2():
    from worldstream.r0 import validate_runtime_settings
    s=model_settings()
    args=[True,s['m1_execution_mode'],s['m1_config'],True,s['m2_config'],True,s['m3_config']]
    validate_runtime_settings(*args)
    args[4]=dict(s['m2_config'],ordinary_budget=1)
    with pytest.raises(ValueError,match='frozen R0 M1/M2'):
        validate_runtime_settings(*args)
