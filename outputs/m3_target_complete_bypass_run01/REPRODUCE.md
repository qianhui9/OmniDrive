# WRSAR v4 reproduction

Workdir `/data/qh/phdCode/work4/OmniDrive`, base `b89f6aeb3be4dfde9edf1aabe2c6fed8aa3899ad`, existing OmniDrive environment and RTX3090. No installation, download, training or weight writes. Every command excludes faulty CPU4/5. GPU stages acquire `/tmp/omnidrive_m123_latency_gpu.lock` and wait for other users. CUDA execution requires host GPU access if the sandbox hides CUDA. CPU verification does not run a model.

After all stages finish, verify this delivery with this exact read-only command:

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_audit --root outputs/m3_target_complete_bypass_run01 --verify
```

Expected JSON `status: PASS`: frozen policy/protocol/inference/quality sources, exact executed source copies, unchanged v3 artifacts, new DEV/TEST log isolation, same current M1 evidence in core ablation, actual language-forward hooks, current-source receipts, retained CPU/CUDA regressions, checkpoint SHA256 attestation and artifact manifest. PASS concerns execution integrity, not the research GO/NO_GO result. `go_no_go.json` is authoritative for research; deployment stays disabled. Add `--checkpoint` to reread/hash the original weights. Use `--manifest` only when finalizing artifacts.

Completed delivery: CUDA849 passed,969 full M123 calls,77 formal actual skips. Research **NO_GO**: object3Q1.344391x and known accuracy58.54%→90.24% on41/96 scored requests, but one new severe lateral fast-answer error gives1/41=2.439% (sceneCI upper8.333%), exceeding the frozen2% severe-risk gate. Mixed3Q1.116378x. Core ablation224/224 identical output-token and forward-count pairs,76/76 identical skips; no independent target-completeness quality gain. Natural3Q has one skip, but no demonstrated aggregate speed benefit; natural1Q slightly regresses. See final_research_assessment.md, result_answers.md and workload_coverage.json.

For exact experiment repetition use a NEW EMPTY root. Preserve committed historical artifacts: diagnostics and split provenance depend on them. All previously available validation logs were exposed before v4. Original v4 selection used six still-unused physical-train logs, metadata only,2 DEV/4 TEST, first8 consecutive image/QA-complete frames. Base-model training exposure is possible. Repetition must reuse the frozen metadata split explicitly; it cannot call these six logs newly held-out again. `--stage prepare` is for a genuinely new experiment requiring unused logs and intentionally refuses once they are exhausted.

Execute sequentially, keeping inference/worker/driver source files unchanged during each GPU stage:

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_audit --root /tmp/wrsar-v4-reproduction --reproduce-plan
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_diagnostics --root /tmp/wrsar-v4-reproduction
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_audit --root /tmp/wrsar-v4-reproduction --prior
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_audit --root /tmp/wrsar-v4-reproduction --regression
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_bypass --root /tmp/wrsar-v4-reproduction --stage shadow
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_bypass --root /tmp/wrsar-v4-reproduction --stage calibrate
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_bypass --root /tmp/wrsar-v4-reproduction --stage pareto
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_bypass --root /tmp/wrsar-v4-reproduction --stage freeze
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_bypass --root /tmp/wrsar-v4-reproduction --stage run
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_bypass --root /tmp/wrsar-v4-reproduction --stage ablation
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_bypass --root /tmp/wrsar-v4-reproduction --stage report
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --output /tmp/wrsar-v4-reproduction/validation_cpu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --gpu --output /tmp/wrsar-v4-reproduction/validation_gpu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_audit --root /tmp/wrsar-v4-reproduction --summarize
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.target_complete_audit --root /tmp/wrsar-v4-reproduction --checkpoint --manifest
```

If no DEV policy meets the predeclared2pp research budget and severe-risk gate, freeze refuses and TEST must not proceed. Four candidates only: basic0.45/0.75, and three completeness configurations declared in target_completeness_design.md. All generation parameters, original visual tokens and question RNG semantics are unchanged. DEV and TEST are different logs. The new research severe-risk standard was frozen before DEV/TEST and does not revise v3's historical NO_GO or provide a deployment safety qualification. UNKNOWNs never become correct labels.

Original delivery DEV selected C-basic: mean1517.542ms/frame,26/96 real skips, with1/2/3pp budgets selecting the same work point. The target companion is the DEV-fastest target policy, C3. C1/C2 reject three old errors in offline scalar replay and preserve the old Car answer, but old diagnostics cannot select TEST thresholds or establish new generalization. Shadow estimates never count as actual skips or speed. Timing noise and the two-log DEV sample limit optimality claims.

The optional adapter sink reuses existing decoded CPU scalars; it does not change M1 score filtering/commit or M2 reuse. Only C-target pays the new readout/projection/check cost; B and C-basic retain their path. Current camera-center evidence stays separate from visual-reuse qualification. Competition records are never positive answer witnesses. Only supported questions execute completeness checks. Temporal risk remains off.

P0 had25 actual full M123 calls and exact current-evidence reproduction of the three old failure cases. Shadow16 calls, online DEV160, formal672, core ablation96: planned total969. Actual final counts are in run_counts.json/integrity_audit.json. Regression process full_M123_GPU_experiments0 refers only to tests. CPU replay and artifact validation add no GPU model calls. Tests include new support separation, future/GT rejection, task-conditioned admission, safe Position availability, no false absence and unchanged basic ablation behavior.

Each GPU phase retains executed sources, configuration and raw request JSONL. Diagnostic/Shadow are pre-freeze development phases; a worker-only update before online DEV ensures B/basic do not pay optional competition-readout overhead. Their exact old source copies are preserved. Online DEV/TEST/ablation match frozen current sources. The supplemental DEV calibration was rerun only to add evidence-stratified tables; executed candidate policy bytes were verified unchanged.

A post-inference reporting erratum renamed a duplicated report metadata key from `task` to `question_intent`. It changes no inference, grading, policy, labels or scores. The original pre-TEST report source and POLICY_FREEZE remain unchanged in frozen_sources; report_assembly_fix.json records both hashes. Verification permits exactly that single textual substitution, checks every other byte against the archived source, and reports report_assembly_erratum_verified. Fresh reproduction freezes the already corrected reporting file and needs no erratum exception.

Keep all request IDs, UNKNOWNs, adverse outputs and severe cases. Do not replace official independent fact labels with detector confidence or Native answers. Repeated answer-export directories are ignored, scalar JSONL is authoritative, and no duplicate Tensor/checkpoint files are delivered. Full-model performance excludes common input loading/audit serialization while including M1/M2/heads/snapshot/generation and synchronization. No remote push or deployment is part of this delivery.
