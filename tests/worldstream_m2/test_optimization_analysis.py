"""Kernel attribution must use CUPTI correlation, not Torch 1.13 External id."""
import json
from tools.worldstream_m2.execution_analysis import trace_summary


def test_trace_correlation_is_counted_once_despite_mismatched_external_ids(tmp_path):
    events=[
        dict(cat='user_annotation',name='prefix/0/qkv',ts=100,dur=40,args={}),
        dict(cat='cpu_op',name='aten::mm',ts=105,dur=30,args={'External id':1}),
        dict(cat='cuda_runtime',name='cudaLaunchKernel',ts=110,dur=3,args={'External id':900,'correlation':77}),
        dict(cat='kernel',name='gemm',ts=300,dur=200,args={'External id':900,'correlation':77,'stream':7}),
        dict(cat='user_annotation',name='suffix/2',ts=200,dur=50,args={}),
        dict(cat='cpu_op',name='aten::mm',ts=210,dur=20,args={'External id':2}),
        dict(cat='cuda_runtime',name='cudaLaunchKernel',ts=215,dur=3,args={'External id':901,'correlation':78}),
        dict(cat='kernel',name='gemm',ts=500,dur=800,args={'External id':901,'correlation':78,'stream':7}),
    ]
    (tmp_path/'trace.json').write_text(json.dumps(dict(traceEvents=events)))
    (tmp_path/'operators.json').write_text(json.dumps(dict(rows=[])))
    result=trace_summary(tmp_path)
    assert result['kernel_active_sum_ms']==1.
    assert result['kernel_ms_by_stage']=={'prefix_qkv':.2,'suffix_22_blocks':.8}
    assert result['kernel_count']==2 and result['streams']==[7]
