# Existing OmniDrive local environment

Spec: outputs/m3_wrcps_optimization_run01/environment_spec.json
Spec file SHA256: eda83d61075d3610f664d5edffd4c5bf070c6dfed0822bf086252977b5c3e2e5
Interpreter: /home/qh/miniconda3/envs/OmniDrive/bin/python
GPU: RTX 3090, 24GB. CPUs 4/5 must never be used.
No rebuild or package installation. Kernel witness and real 48-frame model run passed.
Fresh documented invocation validation: PASS_WITH_SANDBOX_ACCESS_NOTE, see outputs/m3_wrcps_optimization_run01/doc_validation.json.

EQ-SSC 2026-10-28: unchanged environment warm-reused; 3090 real backward, 606 GPU regression tests, seeded witness passed. Fresh agent doc validation: outputs/m3_semantic_core_wrcps_run01/doc_validation.json (PASS_WITH_SANDBOX_ACCESS_NOTE). sandbox CUDA unavailable; documented host invocation works. No package installation.
