"""Synthetic evidence tests only. Never write production approvals."""
import copy
import json
from pathlib import Path
import pytest
import torch
from worldstream.m2.cost_model import cost_key, block_fingerprint, CostModel, SCHEMA_VERSION
from worldstream.m2 import M2Config
from tools.worldstream_m2.promote_costs import promote_candidate
from tools.worldstream_m2.window_validate import sha256


def write(path,data):path.write_text(json.dumps(data))


def evidence(tmp_path,eva):
    key=cost_key(eva.blocks[0],torch.zeros(2,10,10,32),0,model=eva)
    validation=tmp_path/'validation.json'
    cases=[dict(case='identical',difference=dict(exact=True),stages=[dict(block=0,updated_windows=0)])]
    write(validation,dict(backbone_fingerprint=block_fingerprint(eva),numerical_policy=json.loads(key)['numerical_policy'],
        natural=[],controlled_microbenchmarks=cases))
    candidate=tmp_path/'candidate.json'
    row=dict(key=key,block=0,updated=0,numerical_verified=True,quality_certified=False,
        incremental_gpu_ms=.1,full_gpu_ms=1.,incremental_wall_ms=.1,full_wall_ms=1.,
        evidence_cases=['identical'],numerical_evidence=dict(path=validation.name,sha256=sha256(validation)))
    write(candidate,dict(schema_version=SCHEMA_VERSION,backbone_fingerprint=block_fingerprint(eva),
        records=[row],validation_sha256=sha256(validation),formal_quality_policy=dict(approved_counts=[]),production_ready=False))
    artifact=tmp_path/'synthetic_task_result.json';write(artifact,dict(unit_test_fixture=True))
    tasks={name:dict(status='passed',evaluator='synthetic_fixture',metrics=dict(exact=1),
        path=artifact.name,sha256=sha256(artifact)) for name in ('detection','map','vlm')}
    quality=tmp_path/'quality.json'
    write(quality,dict(schema='m2_window_task_quality_v1',independent=True,status='passed',assessor='test_assessor',
        candidate_sha256=sha256(candidate),records=[dict(key=key,updated=0,covered_cases=['identical'],tasks=tasks)]))
    return candidate,quality,{key},row


@pytest.mark.parametrize('fault',['none','count','signature','numerical','task','quality_hash','artifact','cost','identity','scenario','candidate_not_ready'])
def test_promotion_requires_all_evidence(tmp_path,eva,fault):
    candidate,quality,keys,row=evidence(tmp_path,eva)
    c=json.loads(candidate.read_text());q=json.loads(quality.read_text());counts=[0]
    if fault=='count':counts=[1]
    if fault=='signature':keys=set()
    if fault=='numerical':c['records'][0]['numerical_verified']=False
    if fault=='task':q['records'][0]['tasks']['map']['status']='pending'
    if fault=='quality_hash':q['candidate_sha256']='invalid'
    if fault=='artifact':(tmp_path/'synthetic_task_result.json').write_text('modified')
    if fault=='cost':c['records'][0]['incremental_wall_ms']=2.
    if fault=='identity':c['backbone_fingerprint']='wrong'
    if fault=='scenario':c['records'][0]['evidence_cases']=['missing_case']
    # A provisional candidate is deliberately not production-ready; this must
    # never prevent independent assessment/promotion when all evidence exists.
    if fault in ('numerical','cost','identity','scenario'):
        write(candidate,c);q['candidate_sha256']=sha256(candidate)
    write(quality,q);output=tmp_path/'approved.json'
    if fault in ('none','candidate_not_ready'):
        result=promote_candidate(candidate,quality,keys,counts,'unit_tester',output)
        assert result['production_ready'] and CostModel(str(output)).beneficial(row['key'],0)==(True,None)
        assert json.loads(candidate.read_text())['production_ready'] is False
        # Cannot silently replace existing approvals.
        with pytest.raises(ValueError,match='overwrite'):
            promote_candidate(candidate,quality,keys,counts,'unit_tester',output)
    else:
        with pytest.raises(ValueError):promote_candidate(candidate,quality,keys,counts,'unit_tester',output)
        assert not output.exists()


@pytest.mark.parametrize('fault',['none','limits','error','pipeline','nonfinite','old_policy'])
def test_wsgrr_approval_uses_independent_errors_tasks_and_net_cost(tmp_path,eva,fault):
    candidate,quality,_,_=evidence(tmp_path,eva)
    data=json.loads(candidate.read_text());assessment=json.loads(quality.read_text())
    validation_path=tmp_path/'validation.json';validation=json.loads(validation_path.read_text())
    validation['validation_kind']='wsgrr_quantitative_v1'
    validation['controlled_microbenchmarks'][0]['difference']=dict(exact=False,max_abs=.4,mean_abs=.01)
    if fault=='nonfinite':validation['controlled_microbenchmarks'][0]['difference']['max_abs']=float('nan')
    write(validation_path,validation)
    policy=M2Config().refresh_signature()
    if fault=='old_policy':policy=('wsgrr_v1',)+policy[1:]
    key=cost_key(eva.blocks[0],torch.zeros(2,10,10,32),0,model=eva,refresh_policy=policy)
    row=data['records'][0];row['key']=key
    row.update(pipeline_incremental_wall_ms=.2,pipeline_full_wall_ms=1.,
        pipeline_incremental_gpu_ms=.2,pipeline_full_gpu_ms=1.)
    if fault=='pipeline':row['pipeline_incremental_wall_ms']=2.
    row['numerical_evidence']['sha256']=data['validation_sha256']=sha256(validation_path)
    record=assessment['records'][0];record['key']=key
    if fault!='limits':record['feature_error_limits']=dict(max_abs=.3 if fault=='error' else .5,mean_abs=.02)
    write(candidate,data);assessment['candidate_sha256']=sha256(candidate);write(quality,assessment)
    dest=tmp_path/'wsgrr_approval.json'
    if fault=='none':
        promote_candidate(candidate,quality,{key},[0],'synthetic_fixture',dest)
        assert CostModel(str(dest)).beneficial(key,0)[0]
    else:
        with pytest.raises(ValueError):promote_candidate(candidate,quality,{key},[0],'synthetic_fixture',dest)
        assert not dest.exists()
