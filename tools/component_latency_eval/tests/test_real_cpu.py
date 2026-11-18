"""Small real OGSM execution verifies code replacement and state continuity."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from tools.component_latency_eval.instrumentation import Instrumentation
from tools.component_latency_eval.timing import Recorder


def test_real_ogsm_preserves_transactions_and_restores_code():
    from worldstream.m1 import WorldStateManager, M1Config, Frame, CameraFrame, FullOutputs, PerceptionAnchor
    from worldstream.m1.geometry import box_geometry
    image=np.random.RandomState(9).randint(0,256,(40,40,3),dtype=np.uint8)
    k=np.array([[30.,0,20],[0,30.,20],[0,0,1]])
    manager=WorldStateManager(M1Config(min_region_area_px=1,min_coverage=.1,temporal_workers=1))
    recorder=Recorder('cpu_smoke')
    instrumentation=Instrumentation.__new__(Instrumentation)
    instrumentation.recorder=recorder
    instrumentation.saved, instrumentation.globals, instrumentation.audit=[],{},[]
    original=manager.commit.__func__.__code__
    instrumentation.patch(manager.precheck_with_observation,('OGSM.Precheck',False,'OGSM.Total'))
    instrumentation.patch(manager.commit,('OGSM.Commit',False,'OGSM.Total'))
    try:
        for index in range(3):
            t=1.+index*.5
            camera=CameraFrame('front',image,k,np.eye(4),t,'preprocessed_image',np.eye(3),True,raw_intrinsic=k.copy())
            frame=Frame('seq',str(index),t,np.eye(4),{'front':camera},'test')
            recorder.frame('seq',str(index))
            result=manager.precheck_with_observation(frame)
            prediction=PerceptionAnchor('object','car',box_geometry([0.,0.,10.],[2.,2.,2.]),source_score=.9)
            manager.commit(frame,FullOutputs([prediction]),result)
            assert result.diagnostics['transaction_status']=='committed'
            assert manager.store.commit_count==index+1
        assert [r['component_name'] for r in recorder.rows]==['OGSM.Precheck','OGSM.Commit']*3
        assert all(r['cpu_host_ms']>=0 and 'gpu_device_ms' not in r for r in recorder.rows)
    finally:
        instrumentation.close()
    assert manager.commit.__func__.__code__ is original
