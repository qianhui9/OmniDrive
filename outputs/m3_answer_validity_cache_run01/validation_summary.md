# Validation

Final host RTX3090 regression: **774 passed**, including all seven CUDA-specific checks. Thirty new validity/quality boundary tests passed. The earlier CPU-only run passed 767 tests and skipped seven CUDA tests.

The first host run passed 773 tests; its one failure was the stdin runner entry being unavailable to multiprocessing spawn. Original logs are retained in validation_initial_stdin_entry/. The corrected module entry passed the entire suite; no M1/M2 implementation was modified to fix the runner.

New full M123 GPU experiments: **0**, following the requested offline stop gate. CUDA unit/regression checks are not relabeled as new WS-AVC end-to-end experiments.
