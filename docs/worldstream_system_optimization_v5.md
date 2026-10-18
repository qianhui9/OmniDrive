# Fifth-round feasibility experiments

The serving model, EVA attention, tokenizer and checkpoint are unchanged. New
direction certification and compact contexts are research tools and are not
enabled by any runtime configuration. Historical retrieval and direct answers
are intentionally absent until their prerequisite evidence exists.

`worldstream/m1/fact_certifier.py` separates candidate values from observation
certification. Missing foreground ownership or unique source/current identity
rejects before a calibrated interval can authorize a sign. Detector confidence
is not a metre error bound. Calibration uses joint xy scene maxima; insufficient
scene support produces an unbounded interval. An observation-level certification
does not grant deployment approval.

`tools/system_optimization/compact_context.py` contains both the failed detailed
serialization and the frozen short format. These consume only current decoded
predictions and ego telemetry. The short format loses object size, velocity,
height, orientation, appearance and fine map shape, and is evaluated as an
approximation. `full` passes the original tensors through unchanged.

The v5 tools separate the work as follows:

- `v5.py`: original Planning GPU input probes on frozen disjoint scene splits.
- `v5_semantics.py` and `v5_semantic_review.py`: fixed question-only sampling and
  explicit analyst text review. Visual grounding and human annotations are not
  claimed; unresolved fields remain Unknown.
- `v5_system.py`: fresh Overlap/Auto calibration with stage instrumentation off,
  preserving the full persistent-state service and measuring all admission and
  state-maintenance costs.
- `v5_facts.py`: persist annotation-free certifications first, then load
  calibration Oracle labels, freeze the diagnostic interval artifact, and only
  afterward score independent scenes.
- `v5_report.py`: actual token/GPU/quality results, failed requests included in
  quality denominators, and explicit Not Run gates for illegal combinations.
- `v5_pipeline.py`: wait for the input probe and idle GPU, then finish system
  experiments, complete GPU-visible regressions, and reporting.

Outputs and exact commands are in `outputs/system_optimization_v5/`. Frozen
protocols and measured traces must not be overwritten. Reproduction requires a
fresh output root selected with `OMNIDRIVE_V5_OUTPUT`; the final source freezes
the short format directly, and the original
detailed-format OOM is a separate failed calibration experiment. All execution
uses `/home/qh/miniconda3/envs/OmniDrive/bin/python` with
`taskset -c 0-3,6-31`. GPU waiting never terminates another process.

The only serving optimization is avoiding a repeated string hash when the full
live parameter/buffer/version/backend witness is unchanged. Every request still
enumerates that witness. The implementation fingerprint now also covers the
admission function, so old certificates cannot authorize the changed code.
