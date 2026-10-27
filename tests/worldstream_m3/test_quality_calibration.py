"""Policy fitting is scene-disjoint and cannot retune against heldout answers."""
import copy
import pytest
from tools.worldstream_m123.benchmark.quality_calibration import (
    fit_policy,numeric_values,task_metrics,paired_quality,
)


def record(threshold,scene,wall):
    return dict(profile='full_visual',threshold=threshold,scene_id=scene,frame_id=scene+'_target',
        quality=dict(f1_loss=.001,critical_regressions=0),baseline_wall_ms=100.,candidate_wall_ms=wall,
        diagnostics=dict(reuse_type='bounded_approx',reused_tokens=64,input_tokens=100,
            cache_source_frame=scene+'_source',visual_group_spans=(('object',32,80),),
            planning_ms=1.,prefill_ms=30.,decode_ms=10.,cache_branch_ms=.1))


def test_policy_fit_reproducible_and_heldout_cannot_change_threshold():
    rows=[record(t,s,w) for t,w in [(.1,80.),(.2,50.)] for s in ('cal','val') for _ in range(3)]
    seq=dict(scene_splits={'cal':'calibration','val':'validation'})
    a,curve=fit_policy(rows,{'unit':'fixture'},seq)
    b,_=fit_policy(rows,{'unit':'fixture'},seq)
    assert a==b and a['threshold_selection']=={'full_visual':.2}
    changed=copy.deepcopy(rows)
    for r in changed:
        if r['scene_id']=='val' and r['threshold']==.2:r['quality']['f1_loss']=.5
    c,_=fit_policy(changed,{'unit':'fixture'},seq)
    assert c['threshold_selection']==a['threshold_selection']
    assert c['profiles']['full_visual']['approx']['quality_loss_upper']>.02
    assert c['profiles']['full_visual']['approx']['calibration_samples']==3
    assert c['profiles']['full_visual']['approx']['validation_samples']==3


def test_missing_independent_scene_is_an_error():
    with pytest.raises(ValueError):fit_policy([],{},dict(scene_splits={'s':'calibration'}))


def test_same_frame_inherited_kv_cannot_certify_cross_frame_visual_rule():
    rows=[record(.1,s,50.) for s in ('cal','val') for _ in range(3)]
    seq=dict(scene_splits={'cal':'calibration','val':'validation'})
    for r in rows:
        if r['scene_id']=='val':r['diagnostics']['cache_source_frame']=r['frame_id']
    policy,_=fit_policy(rows,{},seq)
    assert 'approx' not in policy['profiles']['full_visual']
    for r in rows:r['diagnostics']['reused_tokens']=32
    policy,_=fit_policy(rows,{},seq)
    assert 'approx' not in policy['profiles']['full_visual']


def test_approximate_cost_range_uses_cross_frame_samples_and_quality_keeps_descendants():
    rows=[record(.1,s,50.) for s in ('cal','val') for _ in range(3)]
    same=record(.1,'cal',10.);same['diagnostics'].update(cache_source_frame=same['frame_id'],reused_tokens=90)
    same['quality']['f1_loss']=.3
    rows.append(same)
    policy,_=fit_policy(rows,{},dict(scene_splits={'cal':'calibration','val':'validation'}),budget=1.)
    rule=policy['profiles']['full_visual']['approx']
    assert rule['cost']['min_reuse_tokens']==rule['cost']['max_reuse_tokens']==64
    assert rule['cost']['samples']==3 and rule['calibration_samples']==4
    assert rule['quality_loss_upper']>.02


def test_cross_frame_validation_must_cover_the_frozen_calibration_cost_domain():
    rows=[record(.1,s,50.) for s in ('cal','val') for _ in range(3)]
    for r in rows:
        if r['scene_id']=='val':r['diagnostics']['input_tokens']=200
    policy,_=fit_policy(rows,{},dict(scene_splits={'cal':'calibration','val':'validation'}))
    assert 'approx' not in policy['profiles']['full_visual']


def test_fitted_net_cost_deducts_the_measured_scheduler_overhead_once(tmp_path):
    import json
    from worldstream.m3.config import M3Config
    from worldstream.m3.scheduler import Scheduler
    rows=[record(.1,s,50.) for s in ('cal','val') for _ in range(3)]
    for r in rows:r['scheduler_overhead_ms']=10.
    binding={key:'unit_fixture' for key in ('model_version','weights_sha256','execution_key','generation_key','layout_signature','dtype')}
    policy,_=fit_policy(rows,binding,dict(scene_splits={'cal':'calibration','val':'validation'}))
    path=tmp_path/'policy.json';path.write_text(json.dumps(policy))
    scheduler=Scheduler(M3Config(reuse_policy=str(path)))
    # The measured paired M3 walls save 50 ms, after 10 ms of online overhead.
    chosen,_=scheduler.choose([dict(length=64,reuse_type='bounded_approx',risk=.01,rms=0.,max_token=0.,age=1,depth=1)],
        dict(binding,task_profile='full_visual',input_tokens=100),10.)
    assert chosen['saved_ms']==pytest.approx(50.)


def test_numeric_metric_checks_values_not_just_counts():
    assert numeric_values('two cars')==[2.]
    assert task_metrics('3 cars','2 cars')['numeric_exact'] is False
    assert task_metrics('two cars','2 cars')['numeric_exact'] is True
    assert task_metrics('(+1.2, -3.4)','(+1.2, -3.4)')['numeric_exact'] is True
    assert task_metrics('(+1.2, -3.5)','(+1.2, -3.4)')['numeric_exact'] is False


def test_existing_baseline_error_separated_from_new_regression():
    old=paired_quality('3 cars','3 cars','2 cars')
    new=paired_quality('3 cars','2 cars','2 cars')
    assert not old['critical_regressions'] and new['critical_regressions']>0


def test_partial_numeric_and_spatial_regression_not_hidden_by_boolean_accuracy():
    r=paired_quality('right at (9, 8)','left at (1, 8)','left front at (1, 2)')
    assert r['critical_regressions']>0
    assert r['task_metrics']['numeric_exact'] is False
    assert r['baseline_task_metrics']['numeric_exact'] is False


def test_standard_launcher_checkpoint_identity_is_content_bound(tmp_path):
    from types import SimpleNamespace
    import hashlib
    from worldstream.m3.checkpoint import attach_checkpoint_identity
    path=tmp_path/'loaded_checkpoint';path.write_bytes(b'unit_fixture')
    model=SimpleNamespace(worldstream_audit_identity={'other':'preserved'})
    result=attach_checkpoint_identity(model,path)
    assert result['checkpoint_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['other']=='preserved'


def test_actual_service_failure_only_rejects_rule_without_retuning(tmp_path):
    import json
    from types import SimpleNamespace
    from tools.worldstream_m123.benchmark.quality_calibration import qualify_service
    base=tmp_path/'base';base.mkdir();run=tmp_path/'run';run.mkdir()
    common=dict(frame_id='f',scene_id='heldout',question='which side',reference='left')
    native=dict(common,answer='left')
    actual=dict(common,answer='right',reuse_diagnostics=dict(task_profile='full_visual',reuse_type='exact'))
    (base/'per_question_metrics.jsonl').write_text(json.dumps(native)+'\n')
    (run/'per_question_metrics.jsonl').write_text(json.dumps(actual)+'\n')
    policy=dict(quality_budget=.02,threshold_selection={'full_visual':.1},
        profiles={'full_visual':{'exact':dict(quality_loss_upper=0.,critical_regressions=0,risk_limit=.1)}})
    path=tmp_path/'policy.json';path.write_text(json.dumps(policy))
    seq=tmp_path/'seq.json';seq.write_text(json.dumps(dict(scene_splits={'heldout':'validation'})))
    out=tmp_path/'qualified'
    qualify_service(SimpleNamespace(output=str(out),policy=str(path),sequence_file=str(seq),
        native_reference=str(base),service_run=str(run)))
    result=json.loads((out/'m3_reuse_policy.json').read_text())
    assert result['threshold_selection']==policy['threshold_selection']
    assert result['profiles']['full_visual']['exact']['critical_regressions']>0
    assert result['service_qualification']['threshold_retuned'] is False
