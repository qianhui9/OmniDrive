# Question-conditioned target evidence

One existing M3 Runtime and K8 fallback are retained. All production switches default off. No detector, encoder, VLM, M1 association or M2 reuse algorithm is rerun or redesigned. M1 filtering stays0.3; M3 Answer Witness filtering stays0.35 before the frozen admission confidence threshold.

An Answer Witness is an actual current committed observation with current source receipt, calibrated camera support, finite geometry and adequate score. Its detector confidence is not its probability of being the question's correct target. A Competition Evidence record comes from an already decoded current detector box, before the M1 score filter. The optional adapter sink copies existing CPU scalars; only candidates with score≥0.15 and valid current camera-center projection enter the immutable snapshot. These records cannot supply answer facts, repair a missed target, create an M1 Anchor or authorize M2 reuse. Historical predictions and GT are not accepted.

For a nearest selector, retain competitors in the question taxonomy and original ROI only if their category (Category) or direction sector (Position) could change the answer. The unchanged geometric gap scale is `m=max(2m,0.15*d_selected)`. For a decoded competitor:

```
gap = d_candidate - d_selected
dominance = clamp(1 - gap/m, 0, 1)
competition_risk = min(1, score_candidate/score_selected) * dominance
target_risk = max(1-score_selected, all relevant competition_risks)
```

This is a relative support/ranking score, not a probability. Same-answer and unrelated competitors contribute nothing. Missing candidates outside the decoder's existing top300 remain unobservable; camera-center projection does not certify visibility or object-set completeness. Existing ROI/sector ambiguity checks remain unchanged. Position uses a DEV candidate confidence floor; nearest center-projection witnesses use a DEV candidate minimum camera count. Multi-camera support still does not certify visibility, identity or correct ranking.

Positive existence needs one reliable current positive Witness, not complete detection of other objects. It does not scan competition records and never turns missing Anchors into No. Unsupported tasks exit before current target/history/completeness scans and retain the original generator. Motion still requires independently supported velocity; no velocity is invented.

Four candidates were declared before new DEV inference:

| Configuration | Completeness | Position confidence | Target-risk ceiling | Center-projection cameras for nearest |
|---|---:|---:|---:|---:|
| C0 basic | off | existing | existing | existing |
| C1 target | on |0.55|0.50|2|
| C2 target | on |0.65|0.35|2|
| C3 target | on |0.45|0.65|1|

All share the unchanged0.45 positive confidence admission and0.75 legacy binding ceiling. DEV real online E2E determines the best eligible candidate; TEST never chooses thresholds. C-basic and B retain their original evidence-readout path. C-target pays its own decoded scalar readout in the M1 output adapter, center projection in snapshot and per-supported-question analysis. The ablation includes these costs. Binding/extraction/completeness are nested in validity timing and must not be summed again.

Quality uses independently loaded current annotations and original QA references, with the previous visible-nearest/2m ambiguity/1m sector conventions preserved. Physical object-class matching and correct visible question-target matching are reported separately; unmatched identity is UNKNOWN, not an assumed false positive. Required-fact errors can remain KNOWN even when the predicted object's identity is UNKNOWN. The symmetric observable severe metric includes explicit opposite longitudinal/lateral assertions and incorrect negative existence; exact LLM target identity is not generally observable and is reported separately for fast Witnesses. These restricted labels do not certify driving safety.

`protocol_freeze.json` predeclares mean quality budgets1/2/3pp (default2pp), paired scene intervals and conservative all-request unknown bounds. Research admission additionally requires new severe error rate and its empirical scene-bootstrap upper interval≤2pp of scored requests. This new limited descriptive-QA research criterion does not revise historical v3 NO_GO or authorize deployment. Few logs, rare errors, correlated rounds and incomplete labels limit population-level guarantees; all UNKNOWNs and descriptive counts remain visible.

All18 physical validation logs were previously exposed. New data therefore use the six locally available physical-train logs absent from historical experiment-plan scene/log/frame metadata:2 DEV logs and4 TEST logs, each first8 contiguous image/QA-complete frames, selected by metadata only. Base-model training exposure is possible and disclosed; these are independent experimental streams, not official unseen-model held-out claims. TEST has one timing round and rotates method order by scene/workload to prioritize independent logs over repeated frames. Mixed3Q contains one preset structured slot and two original questions; its ratio is not a natural-population estimate.

Old25-call diagnostics reproduce the v3 current detector/observation evidence exactly. Correct pedestrian candidates were filtered at M1 score0.3 or M3 witness0.35; the motorcycle survived all stages but was not selected ahead of a weakly justified near Car. CPU replay of old scalar outputs shows C1/C2 reject all3 old errors while all4 candidates keep the old Car answer. This is diagnostic regression evidence, not new TEST quality or E2E acceleration.
