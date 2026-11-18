from tools.component_latency_eval.report import paths


def test_fallback_only_when_rejection_continues():
    base=dict(verification_forward_calls=1, verification_log=[dict(rejection_position=1,output_offset=0,accepted=1)])
    receipt=dict(visual_diagnostics=dict(approx_global_residual_reuse=False),requests=[
        dict(base,output_token_ids=[1,2]), dict(base,output_token_ids=[1,2,3]),
        dict(llm_skipped=True), dict(verification_forward_calls=0,llm_skipped=False)])
    result={r['execution_path']:r for r in paths([receipt],'r')}
    assert result['fallback_after_rejection']['invocation_count']==1
    assert result['candidate_token_verification']['invocation_count']==2
    assert result['direct_answer']['invocation_rate']==.25
    assert result['global_refresh']['invocation_rate']==1
