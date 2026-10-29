# WRSAR v3 reproduction

Cwd `/data/qh/phdCode/work4/OmniDrive`; base01201559e90806d3eb5a686dda6740fbd8a4262f. Existing OmniDrive Python and original checkpoint/data; no installation, download, training or weight write. All commands exclude faulty CPU4/5. GPU processes acquire `/tmp/omnidrive_m123_latency_gpu.lock` and wait for other jobs; never stop another user's process. Model/CUDA stages require host GPU execution if the sandbox hides CUDA; artifact verification is CPU-only.

Verify this completed delivery using this exact read-only invocation:

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_audit --root outputs/m3_state_calibrated_bypass_run01 --verify
```

Expected JSON `status: PASS`: frozen policy/quality/inference sources, actual language-forward hooks, current-perception receipts, disjoint logs, same-current-evidence temporal ablation, retained CPU/CUDA regressions, checkpoint hash attestation and artifact manifest. Add `--checkpoint` to rehash the original checkpoint; add `--manifest` only after finalizing all output artifacts to write the integrity audit and file manifest. No model forward occurs during verification.

This delivery measured1306 full M123 calls and118 formal M3 skips; CUDA regression833 passed. Research verdict **NO_GO**: addressable3Q B→C1.370768x and known task quality64.89%→70.21%, but one new severe left/right fast-answer error repeats in two rounds. Research switches remain off. `--verify` PASS means artifact/execution integrity, not a GO research verdict. See final_research_assessment.md and severe_error_audit.json. Temporal risk has no demonstrated independent quality benefit; TEST thresholds were never retuned.

For an independent reproduction use a NEW EMPTY root and retain all previous committed experiment artifacts, which are part of the diagnostic/data-split provenance. Run these stages sequentially, keeping measured source files unchanged during model execution:

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_audit --root /tmp/wrsar-v3-reproduction --prior-veto
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_bypass --root /tmp/wrsar-v3-reproduction --stage detection
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_audit --root /tmp/wrsar-v3-reproduction --first-frame
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_bypass --root /tmp/wrsar-v3-reproduction --stage prepare
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_bypass --root /tmp/wrsar-v3-reproduction --stage shadow
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_bypass --root /tmp/wrsar-v3-reproduction --stage calibrate
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_bypass --root /tmp/wrsar-v3-reproduction --stage pareto
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_bypass --root /tmp/wrsar-v3-reproduction --stage freeze
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_bypass --root /tmp/wrsar-v3-reproduction --stage run
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_bypass --root /tmp/wrsar-v3-reproduction --stage ablation
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_bypass --root /tmp/wrsar-v3-reproduction --stage report
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_audit --root /tmp/wrsar-v3-reproduction --support-contribution
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --output /tmp/wrsar-v3-reproduction/validation_cpu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --gpu --output /tmp/wrsar-v3-reproduction/validation_gpu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_audit --root /tmp/wrsar-v3-reproduction --summarize
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.state_calibrated_audit --root /tmp/wrsar-v3-reproduction --verify --checkpoint --manifest
```

If DEV does not qualify a policy, freeze fails and TEST must not proceed. Previous analyzed logs/frames are diagnostics only. New TEST is metadata-selected from3 new disjoint logs, first8 contiguous image/QA-complete frames each, with no object/GT/outcome selection. DEV uses6 consecutive frames from each prior development log. Two timing rounds do not become independent scenes. Natural/fixed/addressable1Q and3Q plus a separate preset mixed3Q are reported separately.

Four predeclared DEV policies: confidence0.45/risk0.35 current-only;0.45/0.75 current-only;0.65/0.75 current-only;0.45/0.75 existing temporal-risk control. The temporal option needs independently scored beneficial vetoes and a positive paired DEV quality benefit before it can compete for final E2E selection. All thresholds, weights, generation parameters and grading code are frozen before TEST. No unqualified historical identity gets positive admission credit. Prior UNKNOWN vetoes do not justify additional temporal rules.

The current-detection readout flag only affects M3 snapshots: a current committed actual detector object can use valid exposure-time calibrated camera-center projection without requiring a reusable whole-box90%-coverage image descriptor. M1/M2 stores, physical/photometric certificates, association, detector thresholds and original4m ROI remain unchanged. This is not a guarantee of detection correctness; quality is empirically evaluated on independent current visible facts. Unsupported tasks exit before target/history scans and use the unchanged K8 generator. Request RNG seeds retain the SHA256(scene,frame,question_index,repeat,experiment_seed) rule.

Raw diagnostic head/decoder/adapter/commit scalars are retained in detection/requests.json. CPU reconstruction is diagnostic replay of recorded actual outputs, not new online GPU inference. Prior review_images contain real six-camera images, GT center projections and old M1 observations, with human review marked PENDING. Current/temporal companion timing phases are separate; inspect scene CIs and exact current-detection equality. Keep UNKNOWN, all-request loss bounds and serious errors visible. No Native answer is treated as GT.

Default research switches remain off. No deployment or remote push. Authoritative answers/tokens/actual-forward records are in each stage requests.jsonl; executed source copies accompany phases. Repeated raw answer-export directories are ignored. No duplicate Tensor/model files are delivered. GPU regression model-call count0 refers only to the test process, not the separately measured full M123 calls.
