"""Exercise actual Petr3D entry methods: reset is explicit, Head memory still advances."""
from types import SimpleNamespace
import torch
import pytest
from worldstream.m1 import WorldStateManager,M1Config
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('m1_integration_test_helpers',Path('tests/worldstream_m1/test_integration.py'))
_helpers=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_helpers)
make_model,inputs=_helpers.make_model,_helpers.inputs


@pytest.mark.parametrize('service',['maintain','reset_on_reject','output_only'])
def test_rejection_service_and_reentry(tmp_path,service):
    manager=WorldStateManager(M1Config(enabled=True))
    model=make_model(tmp_path,manager);model.training=False
    decisions=iter([True,False,True,False])
    runtime=SimpleNamespace(config=SimpleNamespace(m1_service=service),
        pre_m1_admission=lambda *a:dict(reject=next(decisions),reason='test_rejection_only'),
        forward=lambda x,*a:([x*2+1],{}))
    model.m2_runtime=runtime
    epochs=[]
    for i in range(4):
        metas,data=inputs(100+i*.5)
        model.simple_test(metas,**data)
        epochs.append(manager.store.generation)
        if service=='output_only' and i in (0,2):
            assert manager.last_result is None and not manager.store.features
            assert manager._pending is None and not manager._previous_cameras
        else:
            assert manager.last_result.timestamp==100+i*.5
            assert manager.store.last_timestamp==100+i*.5
            assert manager._pending is None
            assert all(f.computation_timestamp==100+i*.5 for f in manager.store.features.values()
                       if f.feature_id in ('backbone','object_carrier','map_carrier'))
    assert model.pts_bbox_head.calls==model.map_head.calls==4
    assert model.pts_bbox_head.memory_timestamp.item()==-403.
    if service!='maintain':assert epochs[2]>epochs[1]
