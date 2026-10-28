"""Research metric contracts: real transitions and unknown oracle labels."""
from tools.worldstream_m123.sf_feasibility_analysis import stats


def event(ordinal,*,scene='s',hit=False,status='correct',valid=True):
    return dict(ordinal=ordinal,scene_id=scene,frame_id=str(ordinal),log_id=scene,
        anchor_hit=hit,anchor=dict(anchor_id=scene+'a',stable_state_valid=valid,update_reason='cold'),
        core_fact_checks_offline=[dict(status=status)] if valid else [],selection=dict(rejected={}))


def test_enrichment_gaps_do_not_create_continuity():
    d=stats([event(0),event(2,hit=True)])
    assert d['transitions']==0 and d['anchor_extensions']==0
    assert d['anchor_hit_rate'] is None and d['positive_logs']==0
    assert d['sustained_sequences']==0


def test_scene_boundaries_do_not_enter_hit_denominator():
    d=stats([event(0),event(1,hit=True),event(2,scene='other'),event(3,scene='other',hit=True)])
    assert d['transitions']==2 and d['anchor_extensions']==2
    assert d['positive_logs']==2 and d['valid_sequences']==2
    assert d['sustained_sequences']==2


def test_partial_oracle_unknown_is_not_correct():
    d=stats([event(0),event(1,hit=True,status='unknown')])
    assert d['retained_fact_offline_checks']=={'unknown':1}
    assert d['core_unknown_fraction']==1.


def test_empty_core_lifetime_not_a_stable_sequence():
    d=stats([event(0,valid=False),event(1,valid=False)])
    assert d['valid_frames']==0 and d['valid_sequences']==0
    assert d['mean_valid_anchor_lifetime']==0.
