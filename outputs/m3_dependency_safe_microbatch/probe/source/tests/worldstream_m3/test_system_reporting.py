from tools.system_optimization.report import measured_stages


def test_legacy_suffix_time_is_reclassified_by_executed_branch_not_guessed_duration():
    frame=dict(m3=dict(shared_seed=dict(prefix_tokens=10)),stages=[
        dict(name='llm_prefill',cuda_interval_ms=5),dict(name='m3_kv_branch'),
        dict(name='llm_decode',cuda_interval_ms=2),dict(name='llm_decode',cuda_interval_ms=3)])
    assert [label for label,_ in measured_stages(frame)]==['llm_prefill','m3_kv_branch','llm_prefill','llm_decode']
    assert frame['stages'][2]['name']=='llm_decode'  # raw timing is never edited


def test_shape_recorded_decode_is_not_relabelled_after_a_full_prompt_cache_hit():
    frame=dict(m3=dict(batch={}),stages=[dict(name='m3_kv_branch'),
        dict(name='llm_decode',input_shape=[1,1],forward_tokens=1)])
    assert list(measured_stages(frame))[-1][0]=='llm_decode'
