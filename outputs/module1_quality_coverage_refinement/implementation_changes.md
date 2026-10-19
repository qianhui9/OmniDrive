# Implementation changes

Baseline: actual clean HEAD `78c3a5e`. `baseline_source/` and its SHA manifest retain the prior M1 implementation. Old experiment outputs are read only.

## Association and certification

`identity_tracking_policy` defaults to `shadow`. All legal temporal edges are measured and claim-checked before any tracking-policy decision. Shadow relations carry source epoch, geometry version, actual source frame, target prediction, target committed state version, protocol, qualification reasons and whether they affected ordinary tracking. `get_identity_hypotheses()` returns defensive copies.

`certified_only` permits only independently qualified relations to alter an otherwise rejected geometric association. `experimental` retains the research recovery function, but reassigns permitted edges one-to-one and records `experimental_identity_hypothesis`. Such a recovery clears historical verified timestamps and certified counts. Ordinary geometric tracking remains explicitly unqualified. New observation geometry and features still come from current actual perception; old features are not relabeled as freshly computed.

Hypotheses are published with the forked StateStore transaction. Failure cannot publish a partial history, qualification or ID update. Source-store digest includes association basis and identity qualification state. Readout of physical claims remains separately bound to exact geometry, source/current frames, epoch, camera coordinates, execution and producer protocol.

## Visible surface consensus

The new opt-in `surface_consensus` policy keeps the existing measured flow, FB, texture, photometry, unfolded-warp and ray tests. It does not change the old 0.5 degree or 0.5px geometric requirements. It replaces rejection of an entire complete-neighbourhood grid unit with measured connected support and train-only multiple inverse-depth hypotheses. Overlapping scheduling domains do not define object or surface membership.

Spatial checkerboard holdout measurements never fit or rank a model. Each model needs at least four fitting and four non-collinear withheld supports. Incompatible key support is carved out. Each retained pixel needs an actual measured correspondence, a consistent plane prediction, observed hull support and membership in a supported connected component. There is no superpixel, box or convex-hull filling of unmeasured pixels. Real finite history can supply a longer-baseline key only if it also predicts the immediate measured source; observed temporal conflicts reject support.

The old boundary guard is disabled in the new research configuration because the old population showed no final decision benefit. It remains a measured on/off ablation. Optional split/connectivity/withheld/key-only/history ablations have fixed thresholds; no weighted scoring or post-test reselection is introduced. A coherent wrong RGB flow may still pass: the output names a conditional observation hypothesis, never physical ownership.

## Object-surface identity enhancement

`temporal_object` extends the finite-history producer rather than replacing it. When ordinary crop observation fails, it tests predicted translation as a motion hypothesis, triangulates actual tracks under that motion, requires membership in the source and translated current predicted box, and checks depth separation from a real independently tracked exterior ring in both views. Missing geometry, low parallax, lack of a ring or indistinguishable depth remains Unknown.

The hybrid preserves old supported crop observations with their original meaning. `identity_object_required` is an explicit surface-only ablation. Raw crop/ring counts and failure reasons remain visible; a conditional selected-track population is not substituted for the original 5798 geometric-pair denominator. The new producer/population and all configuration/code choices are separately protocol-bound. It cannot inherit an old producer's independent qualification. Its claim remains identity only: no stationarity, geometry, velocity or pixel ownership certification.

## Evaluation and limits

No online annotations, LiDAR scoring samples, Oracle correspondences, future frames, GT masks or extra uninstalled models are used. All real quality evaluation uses the unchanged sparse annotated reference outside M1. Startup and periodic refresh stay in the area denominator. No accepted sparse samples means FAR is null. Log-level risks, unscorable candidates and accepted support are reported independently.

Instrumentation reproduces the old 39 false and 155 correct accepts exactly across all 1320 nonstartup camera rows. Those diagnostics were frozen before the new mechanism. A supplementary rejection audit scores the actual old flow at candidate reference points; it distinguishes observed rejected support from a causal on/off ablation. Development and leave-one-log-out sensitivity reuse exposed logs and never become independent validation.
