# Implementation audit

Baseline HEAD: 0371c49ead2465ac766af2b32ca88c26629c553c. Initial git status was clean. Existing strict configurations and historical output directories were preserved.

- worldstream/m1/state_types.py adds ComputationStateBundle, ValidityLease, optional M1Result.computation_state and metadata-only FullOutputs.execution_receipt. Existing fields and constructors remain compatible.
- worldstream/m1/computation_state.py builds committed source snapshots, actual measured RGB-derived grayscale sample links and revocable masks. It distinguishes metadata from available tensor/cache state and physical observation from actual computation frame/version.
- state_manager.py keeps precheck(frame); adds precheck_with_observation(frame, validate_observation). Dense/optional surface work can be deferred while Commit, camera history, world anchors and existing identity work continue. Receipt construction and cached-camera lineage correction happen inside the private Commit worker before atomic publication. No M2 tensors are copied.
- temporal_visibility.py adds a dense-verification scheduling flag; sparse measurements required by state/identity maintenance remain active. No new plane, patch fitting, score or multi-view algorithm was added.
- novelty_detector.py conservatively separates new ordinary candidates from the legacy strict invalidation contract. In the lightweight policy, strict masks refresh fully. Optional original strict surface configurations retain their original behavior.
- M1Config and the new YAML opt in to the source-linked path. Defaults remain legacy. Lease exports use the existing maximum state horizon.
- m2/m1_adapter.py retains invalidation_mask and adds computation_admission. It validates context, epoch, cache publication, source frame/age, camera leases and the Dirty/Candidate/Unknown partition; only Dirty enters the exact dependency update mask. Unknown and Candidate require independent full input equality and dependency closure.
- m2/runtime.py adds a default-legacy interface switch, checks both camera/window boundaries, preserves the existing Exact planner/scheduler and quality gates, records actual early-window computation lineage and operation counts, and fails closed on forced refresh, expired lineage and aborted transactions. Approved zero-update/full-update paths do not transfer window-ID lists to CPU. Truly partial window execution needs its actual IDs to record provenance.
- m2/state_cache.py builds image and provenance allocations before publishing fields. The new injected-OOM test reproduces why this change is necessary: the old code could publish a new image before metadata allocation completed. No changes to window_executor.py or packing.py were needed.
- The final audit distinguishes cache boundaries explicitly: a prefix window cache never claims to retain the final global backbone feature. Its individual block/window production records are exported separately with actual pixel footprints. A dedicated regression covers this distinction.
- camera_feature_cache.py adds actual producer, dependency and computation/cache version metadata. Existing camera source frames remain immutable under reuse.
- Petr3D glue passes current timestamps/epoch, requests observation work according to the cheap rejection result, and supplies M2 metadata receipts at Commit. Original Backbone/Head/LLM math, model weights and training were not modified.
- projects/configs/OmniDrive/worldstream_m2_source_linked.py opts in to source-linked camera Auto with persistent M1 state. Legacy M2 configurations remain available. No approximate path was added.

Validation entry points:

- tests/worldstream_m1/test_source_linked_state.py: real source links, metadata-only sources, leases, transaction order, continuous maintenance, older computation/newer observation, shadow identity isolation and actual Petr3D entry output parity with deterministic test doubles.
- tests/worldstream_m2/test_source_linked_admission.py: full Candidate and full Unknown populations, exact changed/unchanged inputs, dependency/global barriers, metadata/lease/coordinate mismatches, dirty/full/aborted/expired overrides, camera refresh and atomic OOM publication. JUnit contains actual engineering observations.
- tools/worldstream_m1/state_interface_replay.py: fresh nuScenes continuous six-camera replay through the actual deployed image transforms. Original historical detections enter Commit; independent sparse LiDAR/annotations score only afterward.
- tools/worldstream_m2/state_interface_integration.py: five-arm full original-model CUDA entry, separate natural/repeat invocations, current heads/memory/LLM, unique language paths and existing quality targets. The completed GPU execution is retained in gpu_validation_v2/.
- tools/worldstream_m1/state_interface_report.py derives JSON, reports and four figures from current raw replay/JUnit. It does not import past GPU results as current evidence.

Scope limits: natural driving produced no actual reuse or E2E acceleration. The optional lease and contract exports are interface ablations over shared hard safety checks, not independently tuned algorithms. Candidate physical quality remains unestablished where sparse references do not intersect accepted pixels.

## GPU follow-up

Added state_interface_cuda_batch.py for idle-GPU admission, seeded kernel witness, full CUDA suite, 8-frame original-model perception parity, and five-arm natural/repeat VLM tests. Integration follows the same replay scenes, preserves native Memory, and compares actually executed EVA intermediate outputs. state_interface_gpu_report.py merges actual GPU evidence and preserves CPU-stage documents. Early outside-sandbox attempts were blocked by approval-service timeouts; later approved launches succeeded on the RTX 3090. gpu_validation_v2 completed 575 CUDA-visible regression cases, perception parity, and full natural/repeat comparisons. Both full-task populations passed their unchanged quality gates.

## GPU execution provenance correction

A recompute→reuse diagnostic reproduced an ambiguous version namespace: the same computation frame was exported with M1 feature-slot version 2 immediately after recompute and M2 producer version 3 after reuse. build_bundle now consistently prefers the committed M2 camera producer receipt, exports computation_version_namespace, and retains feature_record_version separately. The first GPU natural run was intentionally interrupted and retained in gpu_validation/. Final execution uses gpu_validation_v2/ and records each Precheck source snapshot. A dedicated regression covers the stable producer version through cache publications 3 and 4.

## Current-signature zero-update measurement

The historical camera-cost signatures omit the explicit fp16_reduced_precision_reduction field; Auto therefore correctly rejected them with unprofiled_camera_cost. No historical record or production quality gate was loosened. --zero-cost-followup runs a separate GPU regression and controlled full-model experiment in gpu_validation_zero_cost/.

The new profiler creates exactly one current-signature record for changed_cameras=0, measures full versus cached feature-copy cost, checks bitwise full-backbone equality, and leaves quality_certified=False. A dedicated regression checks that profiling cannot itself grant task qualification. The controlled source_linked_zero_probe uses explicit experimental Exact mode only for bitwise repeated all-camera inputs and forbids partial camera updates. Only after the unchanged full Head/Memory/LLM quality gate passes and actual cache hits occur may the experiment-local record become qualified for Auto. Nonzero updates remain unqualified, and the historical cost table is untouched. Exact probe timings are not a production speed claim.

The new opt-in source-linked project config selects that freshly measured table explicitly; missing or incompatible records still trigger full recompute. This is development numerical/task evidence for exact zero updates, not independent physical certification. state_interface_delivery_audit.py checks all required files, current runtime/replay hashes, unchanged protected code, CUDA regression, full-task quality and source lineage before marking the delivery audit passed.
