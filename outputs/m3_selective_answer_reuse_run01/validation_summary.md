# Validation

**806 passed** on host RTX3090, including 7 CUDA-specific checks and 32 new WRSAR/grounding/contract tests. CPU-only validation: 799 passed, 7 CUDA tests skipped. The corrected native prompt preflight passed all three actual test scenes before any timed inference.

Effective full M123 model calls: 606 (24 Shadow, 96 DEV Pareto, 486 formal A/B/C). Formal language requests: 972. Actual formal M3 skips: 60. A further 282 initial-context calls are retained but excluded. All 60 skips have independently observed zero language forward calls; default deployment remains Native. Regression CUDA checks are not substituted for model experiments.
