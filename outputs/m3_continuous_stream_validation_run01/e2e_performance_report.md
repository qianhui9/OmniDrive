# Real continuous-stream E2E

WITHIN_LOG_EXTENDED_EVALUATION; all speeds come from actual synchronized complete model requests, not token scaling.

| Schedule | A ms/frame | B ms/frame | C ms/frame | B→C | A→C | B→C log CI95 | Skips / questions |
|---|---:|---:|---:|---:|---:|---|---:|
| every4 | 862.815 | 862.798 | 803.140 | 1.0743× | 1.0743× | [0.9967902734655132, 1.1790722687758959] | 8 / 18 |
| every2 | 1028.430 | 957.295 | 829.326 | 1.1543× | 1.2401× | [1.0712380564874993, 1.2890785265189573] | 22 / 36 |
| one | 1318.079 | 1146.045 | 935.334 | 1.2253× | 1.4092× | [1.0860936702635986, 1.403721014748818] | 42 / 72 |
| three | 2497.618 | 1811.899 | 1287.076 | 1.4078× | 1.9405× | [1.208290759420576, 1.845033853250353] | 126 / 216 |

Preset mixed3Q B→C 1.090836×; object3Q 1.407763×. Each schedule processes identical real six-camera streams, including idle frames. A uses native backbone/native generation plus M1 bookkeeping; B/C share deployed incremental M2. A→C, A→B, cold/steady, module wall/CUDA, response percentiles, FPS/QPS and memory are in abc_comparison.json. Cold=first frame, transition1-3, steady>=4 without reset. The FPS/QPS are compute service throughput, not a simulation of queued real-time arrivals. Small-sample P99 and3-log bootstrap intervals are descriptive. One rotated condition order per cell; no repeated-round label inflation, but residual order/clock variation is not eliminated.
