# Performance evidence

Six independent scenes × six continuous frames; three alternating rounds for each mode and method. Positive savings mean the optimized route is faster. CPU wall and overlapping CUDA spans are separate measurements. No overlapping stage sums are used as attribution.

Request/Model/M3 totals refer to one six-camera frame service call: one natural question in single, three ordered natural questions in multi. Prefill/decode forward spans sum those questions within the frame. Frame-request/s and question/s are separately reported; per-question amortized request time is labelled as amortized.

| Workload | Boundary | Reference | Prefill saved ms | M3 saved ms | Model saved ms | Request saved ms [scene 95% CI] | E2E speedup |
|---|---|---|---:|---:|---:|---|---:|
| single | free | native | 3.188 | -6.469 | -177.534 | -58.809 [-107.857, 7.115] | 0.9636 |
| single | free | zero | -0.222 | -0.669 | -1.253 | 37.005 [28.557, 46.184] | 1.0229 |
| single | fixed | native | 3.686 | -8.679 | -167.089 | -138.055 [-198.109, -70.846] | 0.9158 |
| single | fixed | zero | -0.016 | 0.247 | -3.281 | 58.438 [41.808, 73.756] | 1.0356 |
| multi | free | native | 17.939 | 7.205 | -161.704 | -169.751 [-223.744, -103.164] | 0.9484 |
| multi | free | zero | 13.465 | 15.687 | 18.469 | -1.541 [-15.650, 13.548] | 0.9995 |
| multi | fixed | native | 17.618 | -1.995 | -162.160 | -201.717 [-262.150, -138.495] | 0.9392 |
| multi | fixed | zero | 13.224 | 11.271 | 13.237 | -36.305 [-58.839, -13.183] | 0.9890 |

All raw per-scene savings, repeats, P50/P95 latencies, decode spans, peak allocated/reserved memory, cache bytes and throughput are in `summary.json`. Timed workers contain no capture, DecisionTrace or sampled cProfile/torch.profiler probes. Existing CUDA timing observers remain active equally in paired methods. Fixed-token selection overhead remains included.

Prefill/decode CUDA metrics sum actual model-forward event spans. Full HF decode-loop spans (including sampling/host gaps) remain in raw M3 per-request diagnostics; complete M3 wall includes the whole loop. These boundaries are not added together.

CUDA peak allocated/reserved memory is reset after warmup and measured over the frame stream. Owned persistent M2/M3 cache bytes and branch-workspace bytes are reported separately from the allocator peak.

The earlier ~54.5 ms non-model difference is assessed directly by paired request-minus-model measurements, plus preprocessing, collate/transfer and output transfer boundaries. It is not assumed to be M3 work.

| Workload | Fixed reuse vs zero non-model saved ms [95% CI] | Preprocess saved ms | Transfer/collate saved ms |
|---|---|---:|---:|
| single | 61.719 [46.475, 74.517] | 61.676 | 0.043 |
| multi | -49.542 [-57.732, -41.854] | -49.497 | -0.040 |

The multi-question non-model penalty is reproduced at about 49.5 ms, almost entirely in preprocessing. The single-question control instead saves about 61.7 ms outside the model despite zero KV reuse. This locates the discrepancy outside prefill and does not identify a causal M3 preprocessing instruction. The host/measurement mechanism remains unresolved; neither a positive request-only CI nor these opposite workload signs establishes KV computation acceleration.


single: largest measured model section is m3_auto_total, host wall 755.524 ms; M3 decode CUDA span 554.448 ms; preprocessing wall 196.556 ms. Host sections and nested/overlapping CUDA spans are not added.


multi: largest measured model section is m3_auto_total, host wall 2399.441 ms; M3 decode CUDA span 1819.710 ms; preprocessing wall 221.993 ms. Host sections and nested/overlapping CUDA spans are not added.


Image I/O/resize/normalization and CPU input preparation are separately profiled in `full_request_cpu_summary.json`; the profiler run is excluded from all headline measurements.


Isolated real-input planner wall: before 1.6553 ms; after 1.0055 ms. This diagnostic result is not an E2E claim. See per-profile records in `planner_microbenchmark.json`.


Development-only equal-token Zero/trial GPU control (per-question Runtime calls on two calibration scenes, three alternating rounds, unchanged real reference steps and T−1 actual decode forwards):

- Prefix 35: actual prefill CUDA savings -1.960 ms; complete M3 savings -4.713 ms. Natural quality rejection remains binding. This is not independent acceptance or full-service speedup.

- Prefix 163: actual prefill CUDA savings 18.562 ms; complete M3 savings 18.452 ms. Natural quality rejection remains binding. This is not independent acceptance or full-service speedup.


Cost scope: original natural calibration compares the native LM head on identical M1/M2 carriers with the complete M3 runtime. It is not the incremental Zero-KV comparison. Legacy rows contain planning-only cost recovery and can conservatively deduct preparation/signature again online; they are preserved and explicitly labelled. Updated calibration records the actual preparation/signature-plus-planning overhead so the fitter and Scheduler deduct it once. The separate same-runtime fixed-token Zero/trial controls provide the direct incremental measurement. No fitted policy is deployed from these diagnostic points.
