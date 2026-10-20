import pytest
from worldstream.m3.question_dependencies import analyze_questions,ready_groups,QuestionDependency


@pytest.mark.parametrize('option',['inputs_ready','shared_carriers','state_writes','ordered_execution'])
def test_unready_ordered_state_writes_and_carrier_changes_are_serial(monkeypatch,option):
    monkeypatch.setattr('worldstream.m3.question_dependencies.native_prompt_matches',lambda *a:True)
    kw=dict(inputs_ready=True,shared_carriers=True,state_writes=False,ordered_execution=False)
    kw[option]=not kw[option]
    rows=analyze_questions(None,[1,2],['first','second'],**kw)
    assert ready_groups(rows,4)==[[0],[1]] and rows[1].dependencies==(0,)


def test_single_turn_binding_and_answer_reference_are_barriers(monkeypatch):
    monkeypatch.setattr('worldstream.m3.question_dependencies.native_prompt_matches',lambda t,i,q:i!=2)
    rows=analyze_questions(None,[0,1,2,3,4],['road?','Based on your answer, why?','other','objects?','safe?'],
        inputs_ready=True,shared_carriers=True)
    assert ready_groups(rows,4)==[[0],[1],[2],[3,4]]
    assert rows[1].reason=='answer_reference' and rows[2].reason=='unbound_prompt'


def test_ready_groups_never_cross_barrier_and_preserve_submission_order():
    rows=[QuestionDependency(i,independent=i!=3) for i in range(8)]
    assert ready_groups(rows,2)==[[0,1],[2],[3],[4,5],[6,7]]
    assert ready_groups(rows,1)==[[i] for i in range(8)]
    with pytest.raises(ValueError):ready_groups(rows,0)


def test_mismatched_metadata_rejected():
    with pytest.raises(ValueError):analyze_questions(None,[1],[],inputs_ready=True,shared_carriers=True)
