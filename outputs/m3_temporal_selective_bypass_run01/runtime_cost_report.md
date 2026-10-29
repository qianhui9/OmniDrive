# Runtime costs

Full module/phase measurements are in abc_comparison.json. Parsing, current binding, temporal risk, snapshot, prefill/decode host/CUDA, miss decision cost, FPS/QPS and peak memory are reported separately. Binding and temporal CPU times are nested in validity time. Temporal analysis runs only on CPU and launches no GPU kernel/model forward.

- addressable 1Q A: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- addressable 1Q B: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- addressable 1Q C: current binding 0.002513 ms/Q, temporal 0.129888 ms/Q; miss decision 0.26627933380730223 ms/Q.
- addressable 3Q A: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- addressable 3Q B: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- addressable 3Q C: current binding 0.002666 ms/Q, temporal 0.072517 ms/Q; miss decision 0.18000472901445447 ms/Q.
- fixed 1Q A: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- fixed 1Q B: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- fixed 1Q C: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.055107249257465206 ms/Q.
- fixed 3Q A: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- fixed 3Q B: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- fixed 3Q C: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.04513016729443162 ms/Q.
- natural 1Q A: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- natural 1Q B: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- natural 1Q C: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.07418786279029316 ms/Q.
- natural 3Q A: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- natural 3Q B: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0 ms/Q.
- natural 3Q C: current binding 0.000000 ms/Q, temporal 0.000000 ms/Q; miss decision 0.0406525839396097 ms/Q.