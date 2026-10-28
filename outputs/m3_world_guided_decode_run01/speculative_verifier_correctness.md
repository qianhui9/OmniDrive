# Correctness gates

CPU tests and actual tiny-Llama CUDA tests cover causal verification, wrong/invalid proposals, first and middle rejection, all acceptance, EOS/stopping/length, processors, GenerationConfig, padding positions, owned prefill K/V, scene/version/question/expiry and multiple same-frame questions. Empty/invalid draft executes the original HF loop, preserving its seed behavior.

Actual large-model adversarial smoke cases: PASS. Broader paired large-model greedy cases: **FAIL**, 6 nonidentical pairs across both block configurations and three rounds. A smoke pass cannot override this failure. Raw first failure is preserved in `strict_gate_failure/`; repeated comparisons are in the final CUDA JSON.

Real forced-prefix probe: future-only suffix changes preserve earlier logits: **True**. Same forced prefix with serial vs block forwards has **0** argmax disagreements over 110 positions. Full logits deltas, margins and K/V differences are saved in `gpu_numeric_probe.json`. This isolates numerical differences from free-running answer drift; it does not authorize approximate deployment.

The additional exact-schedule probe mirrors the actual rejection/crop schedule and reports **1** argmax disagreement(s). The first argmax disagreement in the mirrored schedule is at output index 80: independent serial token 322 versus block-derived K/V token 29889; serial top-two margin 0.0, max absolute logit difference 0.0234375. This demonstrates a numerical decision difference with the same conditioned tokens and rejection crop. This result must not be substituted by the earlier fixed-chunk probe.

Original dataset QA pairs alone pass 84/84 token comparisons in this sample. Across the whole native workload, 6/126 comparisons fail. Workload-specific counts remain in the comparison table. This post-capture input-provenance subgroup does not establish universal exactness or deployed sampling equivalence.

Speculative sampling q=delta derivation: `sampling_derivation.md`. Transition and real small-model joint-law diagnostic: PASS. Deployed-model native numerical distribution equivalence: **NOT_QUALIFIED**. Formal sampling remains native.
