# M3 refinement interfaces

This revision extends the existing M3; original weights, LoRA, task ground truth and frozen quality targets stay fixed. The experimental results and failure records belong to `outputs/module3_refinement/`, separate from the first delivery.

`M3Config` adds:

| Option | Behavior |
|---|---|
| `single_question_fastpath=True` | With same-frame exact execution and one question, reset the unused cache and directly call native generation before signature/expansion work. |
| `borrow_prefix=False` | Loan read-only prefix views only for the verified Transformers 4.31 native LlamaAttention concatenation implementation; otherwise retain owned clones. Dynamic attention still allocates request KV. |
| `batched_decode=False` | Offline greedy batch experiment, with a genuine shared prefix, left-padded suffixes, row-specific masks/positions, native HF EOS and output order. Sampling/custom stopping/unsupported outputs/budget rejection use native sequential generation. |
| `batch_workspace_mb=6144` | Separate conservative live-batch workspace budget; the persistent entry still obeys `memory_budget_mb`. Rejected allocation plans fall back. |
| `common_prefix_seed=False` | Offline sequential study that computes the common prefix independently of the first complete question. It changes arithmetic Shape and is not a bitwise equivalence guarantee. |
| `state_context='none'` | Offline `only` or `augment`: verified historical M1 text context replaces carriers or augments full carriers. Unknown/unusable evidence uses native full input. |

The common-prefix Auto config is `projects/configs/OmniDrive/worldstream_m3_refined_auto.py`. Its admission requires **both** quality and whole-workload net-cost evidence for the model/backend, ordered input lengths, prefix and exact generation contract. Calibration and held-out scene lists must be nonempty and disjoint, quality frozen targets must pass, cold capture/maintenance must be included and the measured cache must fit the budget. Empty approval tables leave native full generation active. An old first-question cache cost row cannot approve the independent-prefix strategy.

The world-state interface freezes `WorldStateManager.last_result` immediately after precheck, while its commit is pending. It copies only current-verified **historical** anchors, projects existing geometry and reliable velocity into Ego coordinates and preserves actual timestamp/camera/feature provenance. A current perception commit cannot retrospectively become historical evidence. M1 does not retain per-anchor source frame IDs, so those remain unknown. Observation validity does not certify Transformer equivalence or QA quality. Counts/absence/global safety/unknown tasks require full visual evidence, and source formatting/insertion is an approximation requiring separate task validation.

`precision_probe` is a reversible, serial, offline research helper. It evaluates deterministic settings and FP16 reduction settings without editing any parameter. The native loader already uses FP32 LoRA: the `fp32_lora` probe is a sanity control, not a precision improvement. `fp32_qkv_prefill` separately evaluates temporary FP32 **base** Q/K/V prefill GEMMs while preserving native one-token decode and all original parameter dtypes. Both restore original forwards/settings even after exceptions; neither is a concurrent production policy or a claim that more precision preserves the native FP16 baseline.

Experiments use the existing `OmniDrive` environment:

```bash
conda activate OmniDrive
# Fresh P0 studies, with original saved real carriers:
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 python -u -m tools.worldstream_m3.numerical_refinement --output outputs/module3_refinement/new_numerical
OMP_NUM_THREADS=1 python -u -m tools.worldstream_m3.common_prefix_probe --output outputs/module3_refinement/new_common_prefix
# Continuous eight-scene / five-question, separate numerical E2E,
# original one-question sampling, all regression tests and final analysis:
bash tools/worldstream_m3/run_refinement.sh
# Complete reproduction including extended-history and genuine base-projection
# precision probes, in a fresh output root:
bash outputs/module3_refinement/commands.sh
```

The script uses fresh timestamped directories and only observes other GPU workloads. Complete generation retains 320 max new tokens. Numerical/operator probes, CUDA profiler timings and continuous service timings are separate protocols. The exact source and its hash are recorded per run; the report refuses incomplete cell grids or changed inference sources and verifies token decoding against actual answer files.

See [refinement report](../outputs/module3_refinement/refinement_report.md) and [delivery audit](../outputs/module3_refinement/delivery_audit.json) for actual quality/admission decisions. Source-code completion and an observed offline speedup do not establish a quality-constrained production improvement.
