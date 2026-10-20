import json
from copy import deepcopy
from tools.system_optimization.v2_costs import calibrate


def test_calibration_rejects_heldout_gap_extrapolation_and_cost_bound_violation(tmp_path):
    (tmp_path/'single').mkdir();(tmp_path/'quality_targets.json').write_text('{"test_fixture":true}')
    records=[]
    key=dict(model_version='sha',lora_version='sha',execution_key='native',generation_key='sampling',
        input_shape=[1,6,3,320,640],question_lengths=[20],service='persistent_state',strategy='overlap',
        cache_state='warm',memory_budget_bytes=1024,refresh_interval=10,frame_gap_s=.5)
    for i,delta in enumerate((-40.,-30.,-50.,-45.)):
        for mode,elapsed in (('m1',1000.),('overlap',1000.+delta)):
            records.append(dict(mode=mode,index=i,repeat=1,warmup=False,split='calibration' if i<2 else 'heldout',
                generated_ids=[[[1,2,3]]],frame=dict(e2e_wall_ms=elapsed,identity=dict(scene_id='s'+str(i)),
                    admission=dict(workload_key=deepcopy(key),lookup_wall_ms=1.),memory=dict(tracked_cache_bytes=32),
                    quality=dict(m1_state_digest='same',exact_perception_memory_digest=True,
                        numerical=dict(output=dict(max_abs=0.,shape_mismatches=[]))))))
    path=tmp_path/'single/frames.jsonl'
    def save():path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    save();assert calibrate(tmp_path)['approved_keys']==1
    records[-1]['frame']['admission']['workload_key']['frame_gap_s']=.50001
    save();result=calibrate(tmp_path)
    assert result['approved_keys']==0
    assert any(r['reason']=='heldout_gap_outside_calibrated_domain' for r in result['decisions'])
    records[-1]['frame']['admission']['workload_key']['frame_gap_s']=.5
    records[-1]['frame']['e2e_wall_ms']=985.
    save();assert calibrate(tmp_path)['decisions'][0]['reason']=='heldout_pair_exceeds_calibrated_upper_bound'
