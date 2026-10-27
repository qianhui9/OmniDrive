"""Repeated frames must not manufacture independent-scene confidence."""
from tools.worldstream_m123.final_speed_analysis import cluster_bootstrap


def test_repeating_same_scene_measurements_does_not_narrow_scene_interval():
    plan=dict(bootstrap=dict(seed=17,repetitions=10000))
    pairs=[dict(scene_id='a',saved_ms=100.),dict(scene_id='b',saved_ms=-100.)]
    once=cluster_bootstrap(pairs,plan)
    repeated=cluster_bootstrap(pairs*24,plan)
    assert once==repeated
    assert once['independent_clusters']==2
    assert once['lower']<0<once['upper']
