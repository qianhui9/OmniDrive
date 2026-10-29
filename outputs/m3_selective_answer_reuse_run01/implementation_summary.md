# WRSAR delivery

Decision: **NO_GO**. Measured results do not meet all declared scoped quality/coverage/scene-E2E criteria; inspect errors, UNKNOWN coverage and paired scene intervals. No test threshold tuning.

Implemented question-referenced fact coverage in task_conditioner/answer_validity, source-preserving AnswerMemory reuse, an opt-in shadow/online decision before M3 expansion in the existing Runtime, and the existing exact K8 target-aware verifier on every miss. No new model, weight update, training, M1 tracker or M2 visual algorithm. Defaults remain unchanged and selective_answer_mode is off.

Actual full M123 GPU calls: 606; formal A/B/C language requests: 972; actual M3 skips: 60. Source counts: {'fact_grounded_response': 60}. Runtime diagnostics and an independent language-forward hook agree: fast requests execute zero LLM calls and zero expansion. Shadow executes every target generation and is never counted as skip speedup.

DEV Shadow selected a confidence floor 0.45 for the default 2pp relative quality budget. Empirical confidence bins are not error probabilities. All budgets/thresholds were fixed before TEST; repeated rounds are not independent scenes. Original natural, fixed and introduced addressable workloads remain separate. Quality is limited to independently labeled current required facts; unavailable absence/road/safety labels remain UNKNOWN.

Default deployment remains prefill-mandatory. Explicit opt-in WRSAR responses complete the same M123 execution contract only with a current committed perception receipt, admitted policy, zero expansion/LLM calls and declared nonexecution of Prefill/Decode. Final B/C measurements enable that contract; A uses the existing native-full benchmark override. An independent forward hook checks the actual calls. This is research qualification, not deployment activation.


The actual frozen configuration is NO_GO under its predeclared zero-new-serious-error criterion. The newly wrong right-front answer occurs on an LLM miss: B answered right-rear correctly for the same current annotated target. It is not an erroneous cache hit. A shared frame-level random stream may shift after earlier generation is skipped; this explanation is an inference, not an independently isolated RNG experiment. No threshold or seed was retuned on TEST to remove the case. The 60 actual fast responses are all fact_grounded_response; full original-answer reuse remains zero.

The study establishes a narrow measured M1-to-language bypass contribution, not independent benefit from historical answer caching or world-identity persistence. Natural/fixed three-question B→C gains were not demonstrated. Broader natural QA, negative visibility and driving safety remain unqualified; this evidence is insufficient for a general M3 paper claim or automatic deployment.
