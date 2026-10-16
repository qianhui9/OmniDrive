from types import SimpleNamespace
import json
import torch
from worldstream.m3 import M3Runtime, CarrierLayout
from worldstream.m3.config import M3Config
from worldstream.m3.scheduler import Scheduler
from worldstream.m3.visual_state import deployment_key


def test_approved_exact_cost_and_cold_capture(llm, carriers, context, tmp_path):
    version = 'verified_checkpoint'
    key = deployment_key(llm)
    rows = [dict(model_version=version, execution_key=key, device='cpu',dtype='torch.float32',
        input_tokens=n,prefix_tokens=11,exact_greedy_pass=True,baseline_ms=10.,optimized_ms=1.,
        evidence_sha256='test_evidence',cold_capture_profitable=True,min_questions=2) for n in (13,14)]
    path=tmp_path/'cost.json';path.write_text(json.dumps(dict(approved=rows)))
    runtime=M3Runtime(dict(mode='auto',cost_table=str(path),model_version=version,min_prefix_tokens=1))
    ids=[torch.tensor([1,3,-200,4,5]),torch.tensor([1,3,-200,6,7,8])]
    _,diag=runtime.generate(llm,ids,carriers,context=context,max_new_tokens=2)
    assert diag['requests'][0]['cache_stored']
    assert diag['requests'][1]['reused_tokens']==11
    # Evidence bound to a different device/backend key must not admit a policy.
    rows[0]['execution_key']=rows[1]['execution_key']='other_gpu'
    path.write_text(json.dumps(dict(approved=rows)))
    other=M3Runtime(dict(mode='auto',cost_table=str(path),model_version=version,min_prefix_tokens=1))
    _,diag=other.generate(llm,ids,carriers,context=context,max_new_tokens=2)
    assert all(r['mode']=='full' for r in diag['requests'])


def test_quality_alone_cannot_admit_auto_compact(llm, carriers, context, tmp_path):
    version='verified_checkpoint';key=deployment_key(llm)
    quality=dict(approved=[dict(model_version=version,execution_key=key,selector='task',keep_ratio=.5,
        task='object_relation',frozen_targets_pass=True,heldout_samples=10,protocol_sha256='frozen',evidence_sha256='passed')])
    path=tmp_path/'quality.json';path.write_text(json.dumps(quality))
    runtime=M3Runtime(dict(mode='auto',quality_certificate=str(path),model_version=version))
    _,diag=runtime.generate(llm,[torch.tensor([1,-200,4])],carriers,context=context,
        layout=CarrierLayout(4,1,4),questions=['distance of car in front'],max_new_tokens=2)
    assert not diag['requests'][0]['selection']['admitted']
    assert diag['requests'][0]['selection']['admission_reason']=='no_profitable_compact_cost_measurement'
    assert diag['requests'][0]['visual_tokens']==9


def test_safety_unknown_cannot_be_certified_for_compact(tmp_path):
    for task in ('global_safety','unknown'):
        policy=Scheduler(M3Config())
        allowed,reason=policy.compact_allowed(dict(approximate=True,task=task),{})
        assert not allowed
