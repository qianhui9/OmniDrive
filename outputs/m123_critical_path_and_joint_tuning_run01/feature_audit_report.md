# Corrected feature witness

The legacy worker attached its feature witness to the final global-block stage callback. That callback is absent when a global suffix is reused, so those feature-only receipts could refer to the last full-suffix frame. This was an external observation defect; the actual M2 return still fed the native heads and M3 correctly. Timings, detections, independent GT scores and LLM forward/skip counts remain valid as measured.

548 legacy feature rows are explicitly marked invalid in legacy_feature_witness_validity.json; raw records are preserved. A48-call continuous audit captures the actual returned NCHW tensor every frame, validates sixteen complete Full/Incremental feature comparisons, and reuses the Incremental arm to compare original/optimized M1. All sixteen returned-feature, state, ID, detection, refresh and answer pairs are exactly equal for the M1 implementation control, including residual-reuse frames. No new parameter selection or serving-runtime change occurred.

See corrected_feature_audit.json for the complete current-feature errors versus Full and the partial independent QA score. Main A/B/C data were not rerun or replaced; the corrected observation was isolated after their frozen source guard passed.
