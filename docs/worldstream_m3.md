# State-Conditioned LLM Execution

M3 is opt-in at `Petr3D(m3_enable=True, m3_config=...)`. It runs after current Detection/Map carrier production, after the existing M1 observer, and before language generation. Training, head weights and temporal memory are untouched. The default deployment still uses sampling, temperature 0.1, top-p 0.75, one beam and max-new-tokens 320.

```python
from worldstream.m3 import M3Runtime, CarrierLayout
runtime = M3Runtime(dict(mode='exact', offline_experiment=True))
answers, diagnostics = runtime.generate(
    lm_head=model.lm_head, input_ids_list=questions,
    vision_carriers=carriers, world_state=current_m1_store,
    visual_execution=m2_diagnostics,
    context=dict(scene_id=scene, frame_id=sample, timestamp=timestamp,
                 camera_ids=camera_order),
    questions=original_question_texts, layout=CarrierLayout(256, 1, 256),
    do_sample=True, temperature=.1, top_p=.75, num_beams=1,
    max_new_tokens=320, use_cache=True)
```

Modes are `full`, `auto`, `exact`, `compact` and `combined`. `exact`/`combined` bypass a missing cost table only with `offline_experiment=True`; this flag is an explicit experimental override, never a quality certificate. `compact` keeps full inputs unless the selector has an independently approved task certificate or the offline override. Original Q/A output order and BOS/EOS conventions are preserved.

`worldstream_m3_auto.py` defaults to full generation with reason `no_calibrated_policy`. `worldstream_m123_auto.py` preserves the existing M1 history service and M2 camera certification. `worldstream_m3_exact_experiment.py` is an explicitly experimental config. M3 does not require an M1/M2 hit. A stale supplied M1 store rejects task selection; an absent store permits only a current-carrier branch candidate, still subject to quality admission. There is no anchor-to-carrier mapping.

The task selector classifies only question text. Safety/global/map/unknown requests retain all tokens. For object-relation requests, it retains all object/ego tokens plus the leading map subset. `topk` uses untrained token norm as an offline control; `random` is seeded. Both keep Ego and preserve token order. Every deletion is approximate; quality failures prevent online admission. Existing GT is used only in offline evaluation.

The cache owns native expanded embeddings, token identities, positions, mask, legacy 4.31 K/V and end logits. Exact matching compares values, finite status, dtype/device, model/LoRA live versions, execution backend/config, scene and ordered cameras. Different text IDs cannot match merely through identical embeddings. Branches own cloned tensors. A complete identical request reuses cached end logits; a shorter complete match recomputes its last token. Cross-frame mode additionally requires a finite timestamp and refreshes the current frame after text-prefix reuse, retaining the original prefix computation provenance as flat spans. Never use `.data` mutations to change deployment weights; call standard `load_state_dict`/runtime reset and provide the new verified `model_version`.

Supported inference is batch one, Transformers 4.31.0 legacy Llama cache, one greedy/sampled sequence, fixed RoPE configuration. Beam search, supplied external cache, alternate generation configs, hidden-state/attention outputs, supplied runtime padding masks and scaled/dynamic RoPE delegate to the native generator. The native expander and low-level prefill executor separately support and test padding/positions. No environment upgrade is performed.

The stored-entry memory budget includes the owned KV, expanded input and end logits. A live decoder's branch clone is **additional workspace**, separately reported. `memory_budget_mb=0` causes independent full prefill. There is no cross-question mutable cache. A new scene/camera/model signature resets reuse. Prefix matching time, copies, capture and preparation remain inside the measured request latency.

Auto calibration artifacts use an `approved` array. Cost rows require verified `model_version`, stable `execution_key` (`deployment_key(lm_head)` includes config, module types, tensor shapes/dtypes, GPU, framework and math flags), device/dtype, exact input/prefix lengths, evidence hash, passing greedy check and measured positive net benefit. Cold capture additionally requires `cold_capture_profitable=True` and `min_questions`. A task certificate requires matching model/version/execution key, selector/ratio/task, `frozen_targets_pass=True`, nonzero held-out samples and protocol/evidence SHA256. Unknown/safety tasks are never admitted to compact auto. No fabricated or failed measurement should populate these arrays.

Reproduce in the existing environment:

```bash
conda activate OmniDrive
# Rebuild analysis/figures/audits from completed records, with no GPU inference:
bash outputs/module3/commands.sh
# Repeat the whole protocol in fresh directories on an authorized host GPU:
bash tools/worldstream_m3/resume_after_access.sh
```

Use a fresh output directory. The experiment loads original local weights, enables real original driving questions, freezes a question-text-only subset, rotates method order by scene/replay and resets native state per sequence. Source snapshots, all failed/exploratory runs, real carrier tensors, original GT, output token IDs, CUDA events and profiler traces are retained. Full E2E generation always retains the original 320-token limit; separate one-token probes isolate prefill and never count as full-service speedup.

See [experiment report](../outputs/module3/experiment_report.md) for measured support and limitations. F1/normalized exact/numeric-list diagnostics are not a road-safety or relation-semantic certification.

The completed GPU protocol contains 216 multiquestion frame executions, 72 native one-question frame executions, 24 additional quantity-quality records, and **197 passing M1/M2/M3 tests with no skips**. Raw single-question head outputs and temporal memory match across all measured method pairs. Source digests, saved answer files, generation call counts and original typed perception snapshots are checked by `tools.worldstream_m3.finalize_delivery` in [delivery_audit.json](../outputs/module3/delivery_audit.json).

Offline same-frame prefix reuse reduces five-question Prefill P50 from 907.70 to 294.82 ms (3.079×), and VLM P50 from 8253.36 to 7646.69 ms (1.079×). This is **not an approved online improvement**: full generated token sequences agree in 118/120 measured questions. Equal input dependencies do not guarantee identical FP16 results across different prefill shapes; a near-tied logit changes the selected token. Compact input also fails the frozen quality gate. Consequently Auto retains native full execution, and both approval arrays remain empty. Native one-question deployment has zero actual reused KV tokens and no demonstrated reuse benefit.

The real different-question trace reuses 558 tokens and computes only an 18-token suffix; the identical-input trace performs no Transformer prefill. Both are in [traces](../outputs/module3/traces/), with query dimensions and hashes in the audit. Quantity supplementation is evaluated separately and never mixed into the frozen latency sample. Task selection compresses only 5/30 questions in its LLM-only control (4.16% aggregate carrier reduction); the 513→385 reduction applies only to eligible requests.
