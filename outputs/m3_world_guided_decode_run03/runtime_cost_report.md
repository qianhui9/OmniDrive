# Runtime cost accounting

draft_retrieval_analysis.json provides measured retrieval/construction/tokenizer/world-diff/metadata/management/KV-crop per-request costs; abc_comparison.json adds committed snapshot, original M1/M2/heads and complete M3/full-model wall. Target selection's host wall includes waiting on asynchronous prefill; its CUDA span and the full decode/E2E include the real blocking cost. Do not add overlapping selection span to decode again. One tiny probability vector (<=4 values) is copied once; no full-logit CPU copy or candidate-by-candidate synchronization.

Measured savings use paired actual full-model requests. Online estimated native cost uses ONLY previous generated token count, actual decode forwards and prior CUDA event costs; it is an estimate, not an observed current native counterfactual. B is not weakened. Generic-prefix acceptance, M1-fact acceptance, avoided raw-first-zero validation and overall target-forward/E2E differences are separate outputs.

Historical fact parsing is included in actual online management wall, without a separate inner timer. fact_extraction_cost.json measures it in an isolated offline CPU replay and labels that scope explicitly; it does not replace actual online aggregate cost.

Instrumentation: existing worldstream/m3/runtime.py records source/candidates and host intervals; existing prefill_executor.py records selection rankings, RNG/forward counts and CUDA events; existing v2 measurement worker is parameterized for bounded development/final runs; world_guided_decode_v3.py orchestrates selection splits and offline LCP; this report module generates offline artifacts. No extra forward is added solely for timing.

| File | Change | Instrumentation purpose | Entry line |
|---|---|---|---:|
| worldstream/m3/runtime.py | modified | Current source, candidates and host construction/tokenizer/management intervals | 452 |
| worldstream/m3/prefill_executor.py | modified | Selection rankings, no-extra-forward/RNG diagnostics and CUDA events | 161 |
| tools/worldstream_m123/world_guided_decode_v2.py | modified | Reuse common full-model observer for finite development/final workloads | 112 |
| tools/worldstream_m123/world_guided_decode_v3.py | created | Disjoint cohort orchestration, offline prior-answer LCP diagnostics | 41 |
| tools/worldstream_m123/world_guided_decode_v3_report.py | created | Offline paired timing, cost, contribution and quality aggregation | 30 |

C measured module means (ms unless stated). Host enqueue/transfer waits and CUDA spans can overlap; the synchronized outer E2E is the complete-request cost. All A/B/C rows are retained in abc_comparison.json.

| Workload | QA/frame | M1 incl adapters | M2 wall | Heads wall | Snapshot | Prefill CUDA / Q | Decode wall / Q | M3 wall | E2E wall | QPS | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | 1 | 170.124 | 458.283 | 15.378 | 0.776 | 179.745 | 503.926 | 693.623 | 1369.659 | 0.730 | 15.666 |
| natural | 3 | 178.795 | 462.958 | 15.715 | 0.832 | 180.393 | 545.567 | 2194.041 | 2884.528 | 1.040 | 15.666 |
| fixed | 1 | 173.012 | 460.329 | 15.300 | 0.783 | 179.978 | 179.926 | 369.785 | 1050.762 | 0.952 | 15.666 |
| fixed | 3 | 165.759 | 461.334 | 15.729 | 0.828 | 180.297 | 263.695 | 1347.831 | 2023.492 | 1.483 | 15.666 |
