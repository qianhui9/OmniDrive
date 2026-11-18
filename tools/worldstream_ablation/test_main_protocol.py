"""Meaningful regressions for mean-score quality and shared timing denominators."""
import pytest

from .unify_main_protocol import quality, throughput


def event(score, known=True):
    return dict(scorable=known, score=score if known else None,
                correct=score == 1 if known else None, fact_status='Unclassified',
                stale_eligible=False, stale_answer=None, quality=dict(reason=None))


def test_main_mean_score_keeps_partial_credit_and_excludes_unknown():
    m = quality([event(1), event(.5), event(0), event(None, False)])
    assert m['score_sum'] == 1.5
    assert m['scorable_queries'] == 3 and m['unknown_queries'] == 1
    assert m['overall_accuracy_percent'] == 50
    assert m['fully_correct_accuracy_percent'] == pytest.approx(100/3)
    assert m['stale_answer_rate_percent'] is None


def test_fps_and_qps_use_summed_e2e_not_average_scene_throughputs():
    rows = [dict(scene_id='a',e2e_ms=1000,questions=['q']),
            dict(scene_id='b',e2e_ms=3000,questions=['q','r'])]
    m = throughput(rows, samples=100)
    assert m['FPS'] == .5 and m['QPS'] == .75 and m['E2E_seconds'] == 4
    assert m['FPS'] != pytest.approx((1+1/3)/2)
    assert m['FPS_CI95'][0] <= .5 <= m['FPS_CI95'][1]
