# Measured cost attribution

| Schedule / C | M1 incl adapters | M2 | Heads | M3 | Idle E2E | M3 share on QA frames |
|---|---:|---:|---:|---:|---:|---:|
| every2 | 208.338 | 461.392 | 15.094 | 111.931 | 684.1126935970452 | 0.2229 |
| every4 | 211.236 | 451.775 | 15.100 | 92.594 | 719.6932003177978 | 0.3323 |
| one | 218.855 | 455.330 | 15.066 | 213.370 | None | 0.2281 |
| three | 223.435 | 461.958 | 15.037 | 553.497 | None | 0.4300 |

All times ms, host wall. M1 adapter transfers and CUDA spans can be nested inside model phases; do not sum nested spans. Parsing/validity/Position admission are inside M3. Native visual host wall is enqueue time; incremental M2 internally synchronizes. Compare their CUDA event stream spans, not native host enqueue against incremental synchronized host time. Both include stream gaps and are not summed kernel-profiler time. Heads is the observed detection/map forward span; head postprocessing and other residual integration cost remain included in outer E2E. Per-question prefill/decode wall and CUDA event spans are distinct. M1 trends and actual M2 window reuse are in state_continuity_cost.json. No-QA frames retain vision/commit/heads and zero language forwards. Unsupported-question CPU checking overhead and natural parsing opportunities are in task_coverage_analysis.json. Throughput differences across frequencies are measured directly; no linear extrapolation of the old3Q result is used.
