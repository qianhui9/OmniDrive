import json
import pytest
import torch

from tools.worldstream_m123.continuous_stream import question_schedule,SCOPE
from test_always_execute import formal_model,request,llm,eva


@torch.no_grad()
def test_idle_stream_keeps_real_visual_commit_and_zero_language_forward(tmp_path,eva,llm,monkeypatch):
    from mmcv.parallel import DataContainer
    from worldstream.m3 import M3Runtime
    model=formal_model(tmp_path,eva,llm,monkeypatch)
    monkeypatch.setattr(model.tokenizer,'get_vocab',lambda:{},raising=False)
    model.m3_runtime=M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',
        target_aware_exact_only=True,question_rng_enabled=True))
    model.worldstream_allow_idle_frames=True
    calls=[];h=llm.register_forward_pre_hook(lambda *a:calls.append(1))
    for t in (100.,100.1):
        meta,data=request(t);data['input_ids']=[[]];meta[0]['vlm_labels']=DataContainer([],stack=False)
        out=model.simple_test(meta,**data)
        trace=model.worldstream_execution
        assert trace['completed'] and trace['idle_visual_frame']
        assert trace['m1']['commit_completed'] and trace['m2']['execution_completed']
        assert trace['m3']['execution_completed'] and not trace['m3']['requests']
        assert not out[0]['text_out'] and not calls
    h.remove()
    assert model.pts_bbox_head.calls==model.map_head.calls==2
    assert model.m1_manager.store.commit_count==2


@pytest.mark.parametrize('frequency,expected',[('every4',6),('every2',12),('one',24),('three',72)])
def test_schedule_preserves_idle_frames_and_balances_current_requests(frequency,expected):
    qs=[question_schedule('addressable',frequency,i,['frame'+str(i)],['first']) for i in range(24)]
    assert sum(map(len,qs))==expected
    if frequency in ('every4','every2'):
        assert not qs[1] and qs[0]
    natural=question_schedule('natural','every2',2,['frame2'],['frame0'])
    assert natural==['frame2']


@pytest.mark.parametrize('scope,auto,valid',[(SCOPE,False,True),('INDEPENDENT_TEST',False,False),(SCOPE,True,False)])
def test_retrospective_policy_scope_is_explicit(scope,auto,valid,tmp_path):
    from worldstream.m3 import M3Runtime
    p=dict(status='FROZEN_RETROSPECTIVE_POLICY',evaluation_scope=scope,automatic_deployment=auto,
        main_path='current_only',position_mode='basic',min_confidence=.45,expected_normal_m3_ms=1000.,calibration_sha256='unit')
    path=tmp_path/'p.json';path.write_text(json.dumps(p))
    cfg=dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',target_aware_exact_only=True,
        question_rng_enabled=True,selective_answer_mode='online',selective_answer_policy=str(path))
    if valid:assert M3Runtime(cfg).fast_answer_policy['evaluation_scope']==SCOPE
    else:
        with pytest.raises(ValueError):M3Runtime(cfg)


def test_cluster_intervals_resample_complete_logs():
    from tools.worldstream_m123.continuous_stream_analysis import cluster_performance
    out=cluster_performance([dict(log_id='a',from_ms=20.,to_ms=10.),
        dict(log_id='b',from_ms=40.,to_ms=30.)])
    assert out['speedup']==1.5 and out['paired_frames']==2 and out['independent_logs']==2
    assert out['speedup_log_CI95'][0]>=4/3 and out['speedup_log_CI95'][1]<=2.
    assert out['speedup_log_CI95']==[4/3,2.]


def test_paired_quality_never_scores_unknown_as_correct():
    from tools.worldstream_m123.continuous_stream_report import paired_quality
    def event(cond,token,score):
        return dict(workload='addressable',frequency='three',frame_id=token,request_index=0,
            question='q',condition=cond,log_id='l',quality=dict(known=score is not None,
                score=score,severe_error=False),llm_skipped=cond=='C')
    out=paired_quality([event('B','f0',1.),event('C','f0',None),event('B','f1',0.),event('C','f1',1.)],'B','C')
    assert out['requests']==2 and out['scored']==1 and out['UNKNOWN']==1
    assert out['loss']==-1 and out['conservative_all_request_loss_bounds']==[-1.,0.]
