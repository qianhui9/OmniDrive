from tools.worldstream_m1.identity_metrics import ContinuityMetrics, SavedAssociations


def ann(identity='x',support=True):return dict(instance_id=identity,has_sensor_support=support)


def test_switch_is_not_fragment():
    m=ContinuityMetrics();m.update([ann()],{'x':'a'},{'x'});m.update([ann()],{'x':'b'},{'x'})
    assert m.counts['continuous_identity_switches']==1 and m.counts['predicted_track_fragments']==0


def test_visible_missed_frame_is_fragment():
    m=ContinuityMetrics();m.update([ann()],{'x':'a'},{'x'});m.update([ann()],{},set());m.update([ann()],{'x':'b'},{'x'})
    assert m.counts['predicted_track_fragments']==1 and m.counts['continuous_identity_switches']==0


def test_gt_gap_or_ambiguous_is_boundary():
    for anns,possible in [([],set()),([ann()],{'x'}),([ann(support=False)],set())]:
        m=ContinuityMetrics();m.update([ann()],{'x':'a'},{'x'});m.update(anns,{},possible);m.update([ann()],{'x':'b'},{'x'})
        assert m.counts['predicted_track_fragments']==0 and m.counts['continuous_identity_switches']==0


def test_saved_trace_reconstruction_and_validation():
    traces=[dict(frame_id='1',pairs=[],edges=[]),dict(frame_id='2',pairs=[['s:g2:a1',0]],
        edges=[dict(anchor_id='s:g2:a1',prediction_index=0,previous_instance='x',current_instance='x')])]
    r=SavedAssociations('s',traces)
    r.update('1',[0],{0:'x'},[ann()],{'x'});r.update('2',[0],{0:'x'},[ann()],{'x'})
    assert r.checked_edges==1 and r.continuity.counts['continuous_identity_switches']==0


def test_surviving_id_bridges_missed_observation():
    m=ContinuityMetrics();m.update([ann()],{'x':'a'},{'x'});m.update([ann()],{},set());m.update([ann()],{'x':'a'},{'x'})
    assert m.counts['predicted_track_fragments']==0 and m.counts['bridged_visible_gaps']==1
