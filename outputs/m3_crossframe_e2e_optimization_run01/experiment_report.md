# M3 cross-frame / E2E experiment

Base commit: `169d7e3023165030cabf962c12311f7c2cdf5480`. Checkpoint: `ckpts/iter_10548.pth`; SHA and environment in worker invocation/configuration/checkpoint records and `environment.json`.

## Acceptance
```json
{
  "action": "RETAIN_EXISTING_CONSERVATIVE_POLICY",
  "formal_policy_changed": false,
  "formal_policy_sha256": "326ecb70d4065ab788659b55ed540226ace4dd0e6b5751bece69edaae4140c27",
  "new_candidate_deployed": false,
  "code_correctness": true,
  "regression_log": "regression_tests_r3.log",
  "regression_log_sha256": "be3af773e3e6fabdb9be04929beb7b37787819986b336e887b6509ef4f4595ef",
  "system_absolute_quality_budgets_pass": true,
  "m3_quality_proxy_pass": true,
  "native_critical_proxy_findings_repeat0": {
    "single": 1,
    "multi": 2
  },
  "independent_cross_frame_visual_reuse": false,
  "quality_proxy_gate": false,
  "prefill_and_m3_latency_ci_positive": false,
  "model_wall_ci_positive": false,
  "cross_frame_acceptance_pass": false,
  "complete_service_acceptance_pass": false,
  "e2e_positive_paired_ci": false,
  "world_referenced_cross_frame_acceleration_claim_supported": false,
  "limitations": [
    "Six independent scenes; repeats do not increase independent sample size.",
    "Disjointness covers M3/language development and service runs. Prior full-split perception evaluations are disclosed in historical_data_audit.json.",
    "Approximate trial remains development-only. Existing Map/Full-Visual semantic failures are preserved.",
    "Proxy scores cannot certify unreviewed semantic changes."
  ],
  "calibration_decision": "No quality/latency qualified new cross-frame rule selected; no risk threshold or internal budget increase."
}
```

## Direct answers

1. Independent natural cross-frame visual KV reuse: **False**. Development trial reuse is reported separately and cannot satisfy acceptance.

2. single: cross-frame KV 0 / 20684 input tokens (0.000%); cross-frame visual tokens 0, cross-frame requests 0. All reuse 0 tokens (0.000%); same-frame exact 0. See source/target/forward evidence in `summary.json`.

2. multi: cross-frame KV 0 / 62121 input tokens (0.000%); cross-frame visual tokens 0, cross-frame requests 0. All reuse 1675 tokens (2.696%); same-frame exact 1675. See source/target/forward evidence in `summary.json`.

Development-only natural trial: 12 cross-frame visual-prefix requests, 1536 actually reused visual tokens. These observed development scenes do not count as independent acceptance. Natural motion-semantic failure and lack of a qualified cross-frame validation/cost working point prevent admission.

The 163-token trial prefix contains 35 text tokens and the first 128 Object latent-query tokens. It does not reuse Map KV merely because the question is labelled map_relations. Later Map KV cannot be stitched around a changed earlier Object prefix.

3. Actual Object/Ego/Map strict equality, normalized changes, M1/M2 state, cache age and 16/32/64 block endpoints are in `prefix_structure.json`. Native Object→Ego→Map order is retained. A changed Object prefix cannot be bypassed to reuse Map KV.

4–5. Prefill, full M3, model and full request paired savings and confidence intervals are in `performance_report.md`. Module savings are not counted as complete service savings.

6. Four official-subset score losses and native/new proxy/traffic errors are in `quality_report.md`; lexical improvements do not erase semantic failures.

7. Non-model overhead is measured at explicit request/model boundaries. Remaining bottlenecks and individual stage spans are in `per_frame_summary.jsonl`; CPU profiler hotspots are in `cpu_profile_summary.json`. CUDA stream spans include host gaps and can overlap.

8. Core changes: `ReusePlanner.plan` rejects impossible calibrated endpoints before RMS; `prefix_statistics` evaluates only policy-reachable full blocks; the offline `fit_policy` requires cross-frame visual validation for approximate costs and freezes cost ranges on calibration only. The measured old/new planner comparison uses the same real checkpoint/cache, fixed observed overhead and alternating GPU probes. `PrefillExecutor`, storage revision checks and M1/M2 algorithms are unchanged.

9. World-referenced cross-frame service acceleration claim supported: **False**.

## Policy decision
The unique formal policy is retained. New calibration selects no admissible approximate rule with independent cross-frame evidence. Map/full-visual critical history is preserved. No min-token threshold, quality budget or generation length is relaxed to create hits.

## Evidence
`per_request_summary.csv/jsonl`, `per_frame_summary.jsonl`, `execution_audits.json`, `planner_equivalence.json`, `calibration_summary.json`, and SHA-indexed local tensors/profiles. No historical result was replaced. Reproduce with `REPRODUCE.md`.


## Measured remaining cost

single: largest measured model section is m3_auto_total, host wall 755.524 ms; M3 decode CUDA span 554.448 ms; preprocessing wall 196.556 ms. Host sections and nested/overlapping CUDA spans are not added.
multi: largest measured model section is m3_auto_total, host wall 2399.441 ms; M3 decode CUDA span 1819.710 ms; preprocessing wall 221.993 ms. Host sections and nested/overlapping CUDA spans are not added.

Only the planner change has an isolated before/after measurement here: 1.6553 → 1.0055 ms on actual captured input/cache pairs. This does not establish cross-frame quality or E2E acceleration.


Across 20 actual development cross-frame comparisons, strict LCP lengths are {35: 20}. Exact visual token equality: {'object': {'equal_tokens': 0, 'compared_tokens': 5120}, 'ego': {'equal_tokens': 0, 'compared_tokens': 20}, 'map': {'equal_tokens': 0, 'compared_tokens': 5120}}. Mean carrier RMS: object 0.3423, ego 0.1970, map 0.1085. A stable world object is not an exact latent LLM KV slot. See `block_size_frontiers.json` for 16/32/64 raw risk frontiers; exact LCP is never block-rounded, and raw risk eligibility is not semantic approval.


## Validation history
Final GPU regression log is named by `regression_test_result.json`. Initial newly added test-fixture failures (CPU/CUDA device placement and a missing fixture field) remain in the earlier regression logs. The actual-checkpoint planner audit passed all 72 requests. Calibration semantic failures remain separate from passing automatic proxy scores.

Implementation and measured-evidence commit: `894264c28ce7fee3f804d9f02b62d85cc1f244df` (see `delivery.json`). Later delivery metadata does not change model execution or the retained formal policy.
