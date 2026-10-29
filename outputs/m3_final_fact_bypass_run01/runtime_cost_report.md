# CPU cost

Measured CPU-only over retained historical DEV snapshots, 5 warmup passes and 10 repeats. Parsing and Position admission are nested in validity; do not sum nested spans or extrapolate them into full M123 GPU gains.

- C-basic: total 0.035505 ms/question; parsing 0.000353; validity 0.025170; Position gate 0.0004582262287537257 ms per evaluated Position gate.
- C-position-restricted: total 0.035434 ms/question; parsing 0.000397; validity 0.025072; Position gate 0.0008534640073776245 ms per evaluated Position gate.
- C-position-fallback: total 0.034676 ms/question; parsing 0.000378; validity 0.024539; Position gate 0.00045082221428553265 ms per evaluated Position gate.

No new M1/M2/Heads/snapshot/prefill/decode/FPS/QPS/memory measurement is available. Historical synchronized spans stay in the original v4 report.
