# Existing OmniDrive local environment

Spec: outputs/m3_wrcps_optimization_run01/environment_spec.json
Spec file SHA256: eda83d61075d3610f664d5edffd4c5bf070c6dfed0822bf086252977b5c3e2e5
Interpreter: /home/qh/miniconda3/envs/OmniDrive/bin/python
GPU: RTX 3090, 24GB. CPUs 4/5 must never be used.
No rebuild or package installation. Kernel witness and real 48-frame model run passed.
Fresh documented invocation validation: PASS_WITH_SANDBOX_ACCESS_NOTE, see outputs/m3_wrcps_optimization_run01/doc_validation.json.

EQ-SSC 2026-10-28: unchanged environment warm-reused; 3090 real backward, 606 GPU regression tests, seeded witness passed. Fresh agent doc validation: outputs/m3_semantic_core_wrcps_run01/doc_validation.json (PASS_WITH_SANDBOX_ACCESS_NOTE). sandbox CUDA unavailable; documented host invocation works. No package installation.

SF-WRCPS 2026-10-28: unchanged environment warm-reused, no package installation. Real 192-frame/576-QA capture plus 24 paired same-carrier no-core diagnostics completed. Final CUDA regression: 612 passed, seeded kernel witness 510720.0. Training was refused before model/optimizer creation because the reserved training logs had no cross-frame qualified core; no SF adapter was trained. Fresh documented host invocation validation passed without divergence: outputs/m3_sf_wrcps_optimization_run01/doc_validation.json. Reproduction: outputs/m3_sf_wrcps_optimization_run01/REPRODUCE.md.

SF feasibility 2026-10-28: same environment/spec, no installation. Completed 576 causal TRAIN/development frames with no language decode, 424-frame training preflight, 135 final controlled prefix timings and 5 separate CUDA activity probes. Data gate passed; one-question modeled net +0.084 ms/request has measurement CI crossing zero, so training was not started. Final CUDA suite: 616 passed. Fresh documented invocation: outputs/m3_sf_wrcps_feasibility_run01/doc_validation.json, PASS_WITH_SANDBOX_ACCESS_NOTE, witness510720.0, CPUs4/5 excluded. Evidence and limitations: outputs/m3_sf_wrcps_feasibility_run01/REPRODUCE.md.

WRVMP 2026-10-28: unchanged environment/spec, no installation. Captured160 enriched plus24 original natural frames, zero language decode; real regional feature/content gates passed. Completed486 finite prefix CUDA rows plus real memory creation/projection profiler probes. Natural qualified objects had15 observed grid cells, so no K64/128/256 memories; complete-current513 append controls were slower than Native. CostNO_GO, training/official quality/formal E2E NOT_RUN. Final CUDA suite632 passed, original weights SHA unchanged. Fresh documented invocation PASS_WITH_SANDBOX_ACCESS_NOTE, witness510720.0 and CPUs4/5 excluded; outputs/m3_world_visual_memory_run01/doc_validation.json and REPRODUCE.md.
