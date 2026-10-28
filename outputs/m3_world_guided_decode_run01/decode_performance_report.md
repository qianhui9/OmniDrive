# WSG-SD decode evidence: NO_GO for exact deployment

Current base: `ae5a357b6a2619500c9d6798b7efc5cc753e827e`; research branch `research/m3-world-guided-decode`. Actual RTX3090, unchanged checkpoint `6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d`. All 513 native visual tokens retained. No training or extra draft model.

Native instrumented 24-frame/72-question profile includes one built-in loader overview template and two original dataset questions per frame. Excluding one cold frame, the combined means are: prefill **179.670 ms**, decode forward CUDA **1747.996 ms**; prefill/decode shares **9.32%/90.68%** of their combined forward time. Decode wall **1777.267 ms**, M3 question wall **1957.960 ms**, generated tokens **70.19**. Original dataset QA alone averages **23.43 tokens**, prefill **180.527 ms**, decode **566.035 ms**. Mean three-question frame **6533.123 ms**. M1/M2/heads host wall means **171.728/416.940/16.511 ms**. These host durations can contain original synchronization; they are not isolated kernel times.

Original natural dataset QA strict question availability: **2/48 = 4.17%**. The native loader's 24 generated overview-template prompts have one further exact match; mixed-loader availability is 3/72. These are distinct workload scopes, not full-dataset coverage. Repeated-question application workload uses the two actual scene-first dataset questions on distinct subsequent images: bounded 9-frame/18-question stream, 12 history opportunities (**66.67%**). Native overview-template repetition is a separately labeled diagnostic and is not counted as original QA.

## Paired real decode cost gate

378 measured trials: 21 history-available question/frame tasks, 3 modes, 3 rotating orders, greedy and native sampling separately. These comprise two original natural QA opportunities, twelve original fixed-QA opportunities, and seven loader-template diagnostics, explicitly separated below. Fixed frames are the first two subsequent distinct frames per scene. Selection is by input opportunity, not measured outcome. Source drafts are prior real **sampled** native answers; they are never ground truth or the current answer.

| Contract | Workload | K | Decode wall ratio A/B | CUDA ratio A/B | Greedy identical / pairs | Output tokens A / B |
|---|---|---:|---:|---:|---|---|
| greedy | natural | 4 | 2.200 | 2.205 | 6 / 6 | 168 / 168 |
| greedy | natural | 8 | 2.642 | 2.649 | 6 / 6 | 168 / 168 |
| greedy | fixed | 4 | 3.007 | 3.020 | 36 / 36 | 786 / 786 |
| greedy | fixed | 8 | 3.992 | 4.016 | 36 / 36 | 786 / 786 |
| greedy | native_template_natural | 4 | 1.597 | 1.597 | 3 / 3 | 540 / 540 |
| greedy | native_template_natural | 8 | 1.723 | 1.723 | 3 / 3 | 540 / 540 |
| greedy | native_template_fixed | 4 | 1.174 | 1.173 | 15 / 18 | 2949 / 2949 |
| greedy | native_template_fixed | 8 | 1.217 | 1.216 | 15 / 18 | 2949 / 2913 |
| sampling | natural | 4 | 2.633 | 2.675 | RNG differs / 6 | 165 / 166 |
| sampling | natural | 8 | 4.454 | 4.612 | RNG differs / 6 | 165 / 165 |
| sampling | fixed | 4 | 2.775 | 2.825 | RNG differs / 36 | 791 / 795 |
| sampling | fixed | 8 | 3.715 | 3.818 | RNG differs / 36 | 791 / 789 |
| sampling | native_template_natural | 4 | 1.337 | 1.340 | RNG differs / 3 | 496 / 539 |
| sampling | native_template_natural | 8 | 1.415 | 1.419 | RNG differs / 3 | 496 / 539 |
| sampling | native_template_fixed | 4 | 1.146 | 1.145 | RNG differs / 18 | 3006 / 3008 |
| sampling | native_template_fixed | 8 | 1.151 | 1.150 | RNG differs / 18 | 3006 / 3043 |

The table contains raw diagnostic ratios, including changed-output cases. It is **not evidence of exact acceleration**. Sampling output lengths differ naturally, and the known block-versus-serial numerical difference prevents declaring target-law equivalence on the deployed model. Per-round ratios, paired tokens, forward counts and raw logs are in `decode_comparison_summary.json` and `decode_trials.jsonl`.

Forward cost by input length is retained in the JSON. A nominal K=8 round normally forwards 9 tokens because it incorporates one pending correction/bonus; the initial round forwards 8. Likewise K=4 normally forwards 5. Compare actual costs, not theoretical counts. Rejection K/V cropping is a view operation; its host enqueue time is measured, not a fabricated isolated CUDA cost. Rejected suffix storage lives only until the next concatenation/request ends. Native decode uses no historical K/V. Cache capture is disabled equally for cost trials (zero persistent cache budget); tests separately verify owned-prefix lifecycle.

Generation-wall A includes native multimodal assembly while B has embeddings prepared before timing; therefore those generation ratios cannot qualify complete M3 speedup. Decode CUDA and post-prefill stream spans are the fair local comparison. Draft tuple materialization is recorded; full runtime metadata/hash management is not measured in this bounded executor gate.

## Decision

Real RTX3090 native greedy token mismatch for both block lengths; batched numerical contract not qualified. Stop before complex M1 scheduling and formal sampling/E2E deployment.

M1 question-aware diff, world-guided scheduler, B→C comparisons, formal single-/three-question M123 trials and approximate task-quality acceptance are **NOT_RUN/null**. No complete M3 or M123 acceleration is claimed. Formal source configuration, M1/M2, EVA/heads, native visual carrier and pretrained weights are unchanged.

## Instrumentation changelog

| File | Change | Purpose |
|---|---|---|
| worldstream/m3/prefill_executor.py | modified | Native-forward counts/events and optional request-local history verifier |
| tools/worldstream_m123/world_guided_decode.py | new | Actual continuous native profiling, history audit and paired CUDA gate |
| tools/worldstream_m123/speculative_numeric_probe.py | new | Forced-prefix real-model causal/numeric audit |
| tools/worldstream_m123/speculative_sampling_validation.py | new | Actual small-Llama joint-law and residual boundary diagnostics |
| tools/worldstream_m123/world_guided_decode_checks.py | new | Serialized actual-GPU regression and witness |
| tools/worldstream_m123/world_guided_decode_report.py | new | Reproducible evidence aggregation, no new measurements |

The profiling launcher is independent of serving; ordinary native generation does not create CUDA timing events unless profile=True.
