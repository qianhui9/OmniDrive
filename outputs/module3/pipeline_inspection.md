# M3 pipeline inspection

Environment: OmniDrive conda environment, Python 3.9.25, PyTorch 1.13.1+cu117, Transformers 4.31.0, PEFT 0.7.0. Local RTX 3090 becomes accessible with host GPU permissions; sandbox alone cannot access the driver. Checkpoints `ckpts/pretrain_qformer/` and `ckpts/iter_10548.pth` exist. Git was clean at task start; no AGENTS.md was found in the repository or ancestor paths.

The actual local path is `Petr3D.simple_test_pts → cat(det_query,map_query) → PeftModelForCausalLM.generate → injected LlavaLlamaForCausalLM.generate → native multimodal expansion → HF GenerationMixin`. LoRA r=128, alpha=16 on q/k/v/o projections remains injected; M3 calls the PEFT base model **with those same injected layers**. No model parameter changes.

Detection has 256 learned extra queries + one CAN bus/ego carrier; map has 256 extra queries. Carrier width is already LLM hidden size 4096. See carrier_semantics_audit.md for why proposal/anchor IDs cannot select individual carriers.

The test data loader prepends `<image>\nYou are driving in {location}. ` to each original question. Conversation serialization adds its original system/role prefix. Actual image-token positions and exact embedding LCPs are measured in run02/input_audit.json; no claim is based on raw text equality alone. Deployment config loads only planning (one question/frame). The multiquestion experiment explicitly enables original planning, short action and original conv/VQA files; it records all question counts and selects a deterministic question-text-only subset for expensive inference. Original questions and GT are preserved.

`IMAGE_TOKEN_INDEX=-200` is removed and replaced by 513 visual embeddings. For one marker, expanded length = raw text length −1 +513. Native assembler strips supplied padding, right-pads batches, assigns arange positions, and returns None for position/mask if the caller supplied None. For batch one the HF generator reconstructs an all-one attention mask and consecutive positions. When explicit original positions are supplied to native multimodal expansion, it replaces them with its own arange; M3 audits that actual behavior. Multiple image markers consume separate image-feature rows; insufficient rows fail exactly as in native code.

Original `Llava.generate` explicitly rejects external `inputs_embeds`. Transformers 4.31 Llama.prepare_inputs_for_generation also slices an input with `past_key_values` to its **last token**; passing a multi-token suffix to that interface would silently lose tokens. M3 instead executes suffix prefill using original `forward(inputs_embeds=suffix, past_key_values=owned_prefix_copy, position_ids=suffix_positions, attention_mask=full_mask)`. A request-local GenerationMixin facade retains the installed HF greedy/sample processors, RNG consumption, dummy BOS convention, EOS, stopping criteria and max-new-token logic. Subsequent decode delegates to the original decoder. Images are never inserted a second time.

Cache entries own detached exact input copies, legacy per-layer (K,V) `[B,heads,length,head_dim]`, end logits and original computation provenance. Matching includes embedding values, text identities, visual ordering, positions, mask, dtype/device, live model/LoRA state and scene/camera scope. Each branch owns cloned K/V; shorter complete matches leave one token to compute its logits. Validation never changes computation source. The online default uses no optimization without compatible measured cost/quality evidence; offline exact and compact experiments are explicitly named.

The original file-existence shortcut is retained only when M3 is off. With M3 enabled every question actually executes. All experiments use fresh per-replay/per-method output paths, including the original baseline. Prefill/Decode totals count **every actual question**; CPU enqueue intervals are not reported as synchronized GPU latency.

Primary versioned interface reference: [Transformers v4.31.0 source](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/llama/modeling_llama.py), checked against the installed source (no environment upgrades).

Completed protocol measurements use six scenes, two consecutive frames each, one warmup replay and two measured replays, with six rotated methods. There are 13–17 original questions per frame when all native question files are enabled; the timed subset is five. Raw text lengths are 56–146 and multimodal lengths 569–659; image index is 35. Same-frame expanded prefix lengths are 557–570; adjacent frames share only 35 text tokens. Each frame still computes native EVA/detection/map before all language questions. The final source for input measurements is clean_multi_20261016_220113/input_audit.json; earlier run02 observations are exploratory only.

| Five-question frame, GPU forward intervals | Baseline P50/P95 ms | Offline exact-prefix P50/P95 ms |
|---|---:|---:|
| Cumulative Prefill | 907.70 / 910.60 | 294.82 / 298.52 |
| Cumulative Decode | 6771.38 / 7838.12 | 6758.65 / 7825.51 |
| Complete VLM wall | 8253.36 / 9330.80 | 7646.69 / 8725.73 |

These forward timers include the causal LM vocabulary projection. Input preparation, carrier assembly, EVA, heads, M1/M2 and matching/copy costs are separately retained in baseline_profile.json and latency_results.json; their individual medians must not be summed into an alleged total median. The default one-question deployment is measured independently, including original sampling and raw perception/memory outputs. No actual KV reuse occurs in that case. Exact-prefix experiments fail full-sequence equality (118/120), so these offline speedups do not authorize Auto optimization. See experiment_report.md and delivery_audit.json for finalized evidence and the quality decision.
