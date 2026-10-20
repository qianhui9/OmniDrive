"""Evidence admission and profiler accounting, not task-quality certificates."""
from types import SimpleNamespace as NS
import pytest
import torch
from tools.worldstream_m123.stateful_opportunity.report import opportunity_gate
from tools.worldstream_m123.large_boundary.profile import aggregate_events,aggregate_trace,stage_for
from tools.worldstream_m123.large_boundary.report import removal_bound
import json


def qualified():
    return dict(historical_sources_legal=True,mathematical_closure=True,state_evidence_qualified=True,
        avoids_native_repeated_work=True,net_cost_probe_positive=True,whole_service_upper_bound_worthwhile=True,
        validation_plan_executable=True,optimistic_gpu_savings_upper_ms=500.,minimum_required_savings_ms=300.)


def test_positive_local_cost_does_not_admit_research():
    r=opportunity_gate(dict(net_cost_probe_positive=True))
    assert not r['research_go'] and not r['production_go']
    assert not r['state_evidence_qualified']
    assert r['active_task_quality']==r['active_e2e']=='Not Run'


@pytest.mark.parametrize('missing',['historical_sources_legal','mathematical_closure','state_evidence_qualified',
    'avoids_native_repeated_work','net_cost_probe_positive','whole_service_upper_bound_worthwhile','validation_plan_executable'])
def test_each_missing_claim_fails_closed(missing):
    evidence=qualified();evidence.pop(missing)
    assert not opportunity_gate(evidence)['research_go']


@pytest.mark.parametrize('value',[None,float('nan'),float('inf'),-1.,True])
def test_unknown_or_invalid_upper_bound_cannot_admit(value):
    evidence=qualified();evidence['optimistic_gpu_savings_upper_ms']=value
    assert not opportunity_gate(evidence)['research_go']


def test_ideal_scope_must_cover_state_cost_and_validation():
    evidence=qualified();evidence['optimistic_gpu_savings_upper_ms']=300.
    assert not opportunity_gate(evidence)['research_go']
    r=opportunity_gate(qualified())
    assert r['research_go'] and r['performance_opportunity_go'] and not r['production_go']


def test_profiler_kernel_not_double_counted_through_nested_scopes():
    parent=NS(name='lb::eva',cpu_parent=None)
    inner=NS(name='lb::eva/block_0',cpu_parent=parent)
    launch=NS(name='aten::mm',cpu_parent=inner,device_type=torch.autograd.DeviceType.CPU,
              kernels=[NS(name='gemm',duration=1000.)])
    device=NS(name='gemm',device_type=torch.autograd.DeviceType.CUDA,
              time_range=NS(elapsed_us=lambda:1000.))
    r=aggregate_events([launch,device])
    assert r['attribution_valid'] and r['device_compute_kernel_ms']==1.
    assert r['stages']['eva']['kernel_ms']==1.
    assert r['inclusive_scope_kernel_ms']['eva|eva']==1.
    assert r['inclusive_scope_kernel_ms']['eva|eva/block_0']==1.


def test_incomplete_attribution_is_reported_not_repaired_by_scaling():
    d=NS(name='gemm',device_type=torch.autograd.DeviceType.CUDA,time_range=NS(elapsed_us=lambda:1000.))
    r=aggregate_events([d])
    assert not r['attribution_valid'] and r['attached_activity_ms']==0.
    assert r['device_compute_kernel_ms']==1.


def test_memcpy_is_not_compute_and_sampling_is_not_decode():
    p=NS(name='lb::llm_generation',cpu_parent=None)
    c=NS(name='copy',cpu_parent=p,device_type=torch.autograd.DeviceType.CPU,
         kernels=[NS(name='Memcpy DtoH',duration=500.)])
    d=NS(name='Memcpy DtoH',device_type=torch.autograd.DeviceType.CUDA,time_range=NS(elapsed_us=lambda:500.))
    r=aggregate_events([c,d])
    assert r['attribution_valid'] and r['device_compute_kernel_ms']==0.
    assert r['stages']['llm_sampling_and_preparation']['transfer_ms']==.5


def test_trace_correlation_avoids_duplicated_cpu_attachments(tmp_path):
    path=tmp_path/'trace.json'
    events=[dict(ph='X',cat='user_annotation',name='lb::eva',tid=1,ts=0,dur=100),
        dict(ph='X',cat='cuda_runtime',name='cudaLaunchKernel',tid=1,ts=20,dur=1,args={'correlation':42}),
        dict(ph='X',cat='kernel',name='gemm',tid=7,ts=25,dur=50,args={'correlation':42})]
    path.write_text(json.dumps(dict(traceEvents=events)))
    r=aggregate_trace(path,dict(device_activity_count=1,device_activity_ms=.05,attribution_valid=False))
    assert r['attribution_valid'] and r['stages']['eva']['kernel_ms']==.05
    assert r['legacy_cpu_attachment_consistent'] is False


def test_missing_runtime_correlation_is_not_fabricated(tmp_path):
    path=tmp_path/'trace.json'
    path.write_text(json.dumps(dict(traceEvents=[dict(ph='X',cat='kernel',name='gemm',ts=2,dur=50,args={'correlation':8})])))
    r=aggregate_trace(path)
    assert not r['attribution_valid'] and r['missing_correlation_count']==1
    assert r['device_compute_kernel_ms']==.05


def test_oracle_cannot_remove_the_unchanged_input_pipeline():
    r=removal_bound(100.,500.,10.)
    assert r['removable_upper_ms']==90. and r['speedup_upper']==10.
    assert removal_bound(100.,20.,10.)['speedup_upper']==1.25
    with pytest.raises(ValueError):removal_bound(100.,float('nan'),10.)
