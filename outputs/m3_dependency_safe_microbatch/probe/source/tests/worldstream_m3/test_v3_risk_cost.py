import json
from pathlib import Path
from tools.system_optimization.v3_physical import upper95
from tools.system_optimization.v3_costs import calibrate


def test_zero_support_is_unknown_and_zero_errors_need_sufficient_denominator():
    assert upper95(0,0) is None
    assert upper95(0,30)>.05
    assert upper95(0,60)<.05


def test_statistical_cost_never_expands_calibration_frame_gap_domain(tmp_path):
    (tmp_path/'main').mkdir()
    protocol=dict(performance=dict(min_calibration_scenes=3,min_heldout_scenes=3,max_positive_increment_rate=.1,frame_quantile=.95))
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    key=dict(model_version='sha',lora_version='sha',execution_key='native',generation_key='sampling',input_shape=[1,6,3,8,8],question_lengths=[65],service='persistent_state',strategy='overlap',cache_state='warm',memory_budget_bytes=4096,refresh_interval=10,frame_gap_s=.5)
    data=[]
    for i in range(6):
        for mode,ms in [('m1',1000),('overlap',900)]:
            data.append(dict(mode=mode,warmup=False,index=i,repeat=0,split='calibration' if i<3 else 'heldout',generated_ids=[[[1,2]]],frame=dict(identity=dict(scene_id=str(i)),e2e_wall_ms=ms,admission=dict(workload_key=dict(key),lookup_wall_ms=1),memory=dict(tracked_cache_bytes=128),quality=dict(m1_state_digest='exact',exact_perception_memory_digest=True,numerical=dict(output=dict(max_abs=0,shape_mismatches=[]))))))
    path=tmp_path/'main/frames.jsonl'
    def run():
        path.write_text(''.join(json.dumps(r)+'\n' for r in data));calibrate(tmp_path)
        return json.loads((tmp_path/'cost_calibration.json').read_text())
    assert run()['approved_keys']==1
    protocol['performance']['frame_gap_domain_s']=[.49,.51]
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    data[1]['frame']['admission']['workload_key']['frame_gap_s']=.4
    assert run()['approved_keys']==0
    data[1]['frame']['admission']['workload_key']['frame_gap_s']=.5
    data[-1]['frame']['admission']['workload_key']['frame_gap_s']=.50001
    assert run()['approved_keys']==0


def test_admission_cache_invalidates_on_certificate_or_workload_content_change():
    from types import SimpleNamespace
    from worldstream.runtime import UnifiedRuntime
    runtime=UnifiedRuntime(dict(model_version='sha',lora_version='sha'))
    key=dict(model_version='sha',lora_version='sha',execution_key='cpu',generation_key='sampling',input_shape=[1],question_lengths=[10],service='persistent_state',strategy='overlap',cache_state='warm',memory_budget_bytes=1024,refresh_interval=10,frame_gap_s=.5)
    cost=dict(key=key,includes_all_incremental_costs=True,full_wall_ms=1000,optimized_wall_ms=800,peak_cache_bytes=16)
    quality=dict(key=key,exact_targets_pass=True,heldout_samples=6,calibration_scenes=['a'],heldout_scenes=['b'],evidence_sha256='raw',protocol_sha256='frozen')
    runtime.costs=dict(records=[cost]);runtime.quality=dict(records=[quality]);runtime._request_key=key
    model=SimpleNamespace(m3_runtime=None)
    assert runtime.optimization_admission(model,{})[0]
    cost['optimized_wall_ms']=1100
    assert not runtime.optimization_admission(model,{})[0]
    cost['optimized_wall_ms']=800;runtime._request_key=dict(key,lora_version='changed')
    assert not runtime.optimization_admission(model,{})[0]
    runtime._request_key=key;runtime._certificate_invalidated=True
    assert not runtime.optimization_admission(model,{})[0]
