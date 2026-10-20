import json
import pytest
from tools.system_optimization.profile_analysis import analyze


def test_real_kernel_attribution_is_exclusive_and_union_is_not_sum(tmp_path):
    events=[dict(name='stream/llm_decode',ts=100,dur=100,ph='X'),
        dict(name='decodeop/model.layers.0',ts=100,dur=95,ph='X'),
        dict(name='decodeop/model.layers.0.self_attn.q_proj',ts=110,dur=20,ph='X'),
        dict(name='aten::mm',cat='cpu_op',ts=115,dur=2,ph='X',args={'External id':1}),
        dict(name='aten::add',cat='cpu_op',ts=140,dur=2,ph='X',args={'External id':2}),
        dict(name='gemm',cat='kernel',ts=120,dur=10,ph='X',args={'External id':1}),
        dict(name='add',cat='kernel',ts=125,dur=10,ph='X',args={'External id':2})]
    path=tmp_path/'synthetic_trace.json';path.write_text(json.dumps(dict(traceEvents=events)))
    result=analyze(path)
    assert result['phases']['decode']['kernel_sum_ms']==pytest.approx(.02)
    assert result['phases']['decode']['kernel_active_union_ms']==pytest.approx(.015)
    assert result['decode_operators']['model.layers.0.self_attn.q_proj']['kernel_count']==1
    assert result['decode_operators']['model.layers.0']['kernel_count']==1
    assert result['decode_layers']['0']['kernel_count']==2
    assert result['launch_gap_attribution_ms'] is None


def test_second_profiler_session_uses_actual_cuda_launch_correlation(tmp_path):
    events=[dict(name='stream/llm_decode',ts=100,dur=100,ph='X'),
        dict(name='decodeop/model.layers.1.mlp',ts=110,dur=80,ph='X'),
        dict(name='aten::mm',cat='cpu_op',ts=115,dur=10,ph='X',args={'External id':900000}),
        dict(name='cudaLaunchKernel',cat='cuda_runtime',ts=120,dur=2,ph='X',args={'External id':3000000,'correlation':3000000}),
        dict(name='gemm',cat='kernel',ts=210,dur=15,ph='X',args={'External id':3000000,'correlation':3000000})]
    path=tmp_path/'synthetic_second_session.json';path.write_text(json.dumps(dict(traceEvents=events)))
    result=analyze(path)
    assert result['kernel_host_linkage']=={'cuda_correlation':1}
    assert result['phases']['decode']['kernel_count']==1
    assert result['decode_operators']['model.layers.1.mlp']['kernel_ms']==pytest.approx(.015)
