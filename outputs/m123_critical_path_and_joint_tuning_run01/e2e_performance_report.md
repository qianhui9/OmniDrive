# Frozen retrospective full-system evaluation

Selected policy risk gate: FAILED; primary new severe errors=1. Improved mean accuracy does not override this failure. Independent generalization remains INCOMPLETE; production stays off.

| 预设负载 | A原生 ms/帧 | B ms/帧 | C ms/帧 | B→C | A→C | B→C log CI95 |
|---|---:|---:|---:|---:|---:|---|
| addressable/every4 | 670.269 | 764.802 | 727.639 | 1.0511× | 0.9212× | [0.9902105790142564, 1.1206467926617727] |
| addressable/every2 | 819.804 | 876.137 | 750.817 | 1.1669× | 1.0919× | [1.1074046643373774, 1.2099620377274636] |
| addressable/one | 1107.490 | 1036.250 | 858.262 | 1.2074× | 1.2904× | [1.094496648719742, 1.3306344006383666] |
| addressable/three | 2259.357 | 1724.861 | 1172.131 | 1.4716× | 1.9276× | [1.2350055485161613, 1.976635795223018] |
| mixed/three | 2518.249 | 2447.718 | 2245.601 | 1.0900× | 1.1214× | [1.0439225126107359, 1.1296317253082155] |

Each cell uses all72 continuous frames/3 exposed logs, synchronized full-model wall including answer export and excluding input preparation/audit serialization equally. No-QA frames execute vision and, for B/C, M1+M2+heads. Heavy profiler never runs in the final process. A executes native vision/generation with no M1/M2/M3 computation; the limited old-A companion is separately reported. Per-stage wall/CUDA, prefill/decode, actual forwards/skips, cold/steady, P50/P95, FPS/QPS and peak memory are in abc_comparison.json. Memory is the harness process allocation peak and can retain inactive buffers from prior conditions; it is not a minimum standalone Native deployment footprint. FPS/QPS are compute service throughput, not sensor-arrival queue throughput; no real-time deadline claim is made. P99 and3-log intervals are descriptive. The legacy feature-witness observation defect does not change these measured timings or head/QA outputs; corrected return-value comparisons are in feature_audit_report.md.
