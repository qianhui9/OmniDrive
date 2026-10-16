from types import SimpleNamespace
import torch
from worldstream.m2.execution_policy import validation_budget,optimistic_window_saving
from worldstream.m2 import IncrementalVisionRuntime
import numpy as np
from .test_m1_policy import result as m1_result


def test_measured_budget_uses_current_validation_and_previous_maintenance():
    result=SimpleNamespace(diagnostics={'timings_ms':dict(precheck_total=10,input_adapter_including_transfers=2)})
    state=SimpleNamespace(last_commit_diagnostics=dict(commit_total_ms=3,output_adapter_including_transfers_ms=4))
    assert validation_budget({},result,state)==19
    assert validation_budget({'necessary_validation_ms':float('nan')})==float('inf')
    costs=SimpleNamespace(records=[dict(block=i,full_wall_ms=20,incremental_wall_ms=1,quality_certified=True) for i in (0,1)])
    assert optimistic_window_saving(costs,[0,1])==38


@torch.no_grad()
def test_unaffordable_validation_rejects_before_window_planning(eva,context,monkeypatch):
    runtime=IncrementalVisionRuntime(eva,dict(mode='auto'))
    runtime.scheduler.costs.records=[dict(block=i,key='unrelated_rejection_only',updated=0,
        full_wall_ms=20,incremental_wall_ms=1,full_gpu_ms=20,incremental_gpu_ms=1,quality_certified=True) for i in (0,1)]
    def forbidden(*a,**kw):raise AssertionError('No net saving, do not construct GPU closure')
    monkeypatch.setattr(runtime.planner,'plan',forbidden)
    result=m1_result(context,np.zeros((40,40),bool));result.diagnostics['timings_ms']={'precheck_total':100}
    _,d=runtime.forward(torch.zeros(2,3,40,40),result,execution_context=context)
    assert d['fallback_reason']=='validation_cost_exceeds_window_saving' and not d['cache_saved']


def test_camera_configs_expose_distinct_world_service_contracts():
    from mmcv import Config
    root='projects/configs/OmniDrive/'
    keep=Config.fromfile(root+'worldstream_m2_camera_auto.py').model
    output=Config.fromfile(root+'worldstream_m2_camera_output_only.py').model
    demand=Config.fromfile(root+'worldstream_m2_camera_on_demand.py').model
    assert keep.m1_enable and keep.m2_config.m1_service=='maintain'
    assert not output.m1_enable and output.m2_config.reuse_boundary=='camera'
    assert demand.m1_enable and demand.m2_config.m1_service=='output_only'
