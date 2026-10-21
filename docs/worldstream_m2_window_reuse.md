# Strict source-linked window reuse

The opt-in native configuration is
`projects/configs/OmniDrive/worldstream_m2_source_linked_window.py`.
It keeps the existing Geometry association policy and disables experimental
conditional reuse. The inherited camera configuration remains available.

PETR3D reads the current pending M1 snapshot through `forward_from_m1`. M1
refresh contracts use actual preprocessed input pixels. Dirty hints and expired
sources force recomputation; candidates and unknown pixels both require an
independent finite, exact pixel comparison. Conv2d receptive fields and the
DependencyPlanner determine affected windows. Only the independent prefix before
global attention or cross-window residual convolution is cacheable. All later
blocks and task heads execute normally. Unsupported projection geometry fails
closed; the deployed EVA uses non-overlapping, unpadded patch embedding.

An expired window source invalidates its input dependency footprint; the planner
then closes dependencies again. A camera observation lease failure invalidates
that camera. Scene, epoch, model signature, transaction and cache-version faults
remain global failures. Each camera uses its own observation horizon. Window
source records must match the retained producer frame, timestamp and version.

`mode='auto'` retains existing measured cost and quality gates. Missing or
incompatible profiles cause full execution. `mode='exact',cache_capture='always'`
is a mechanism experiment that forces legal sparse work. Its partial packed
outputs carry unqualified numerical lineage. Every forced Exact request
reports `task_quality_certified=False`, including full refresh and zero updates,
because an independent complete task certificate is required;
full publication starts with certified reference outputs. Switching between
Exact and Auto invalidates the numerical execution signature. Exact is not a promise
of speed or downstream numerical quality. FP32 partial-window packing under
TF32 is conservatively rejected by the runtime. Disabling TF32 alone did
not resolve the real FP32/FlashAttention packed-shape error: Q/K/V rounding
changes before the existing FP16 cast. Forced Exact partial execution remains
unqualified; Auto requires the existing downstream-certified count and exact
signature. The window-study experiment disables TF32 for **all** paired arms
and records this policy; this setting alone is not a quality certificate. Half-precision kernels still require their existing quality
validation. Conditional approximate reuse stays disabled and uncertified.

Counts in `candidate_windows`, `verified_exact_windows`, `recomputed_windows`,
`actual_skipped_windows` and `actual_skipped_block_computations` use
**block-window pairs in the cacheable prefix**. The last two counts describe
actual skipped executor calls, and intentionally agree for this executor.
`actual_reused_spatial_windows` counts reused windows at the first prefix block.
Per-block and per-camera counters are also emitted from completed Block hooks
even when cache capture is disabled. OOM or runtime failure while publishing
a cache clears it and returns the already completed current output once;
unpublished computations cannot appear as a newly published cache. Publication
independently checks that the retained input and every prefix output are finite.
This GPU validation cost is included in the native measurements. Candidate/dirty/unknown
counts can overlap at window boundaries; they are not a partition of windows.
Camera counts are separate and never added to window counts. M2 timing is host
wall/enqueue timing, including required `.nonzero`/finite-publication scalar
synchronization. Capturing and non-capturing requests have different implicit
synchronization, so these components are not comparable GPU kernel durations;
synchronized end-to-end
measurements and peak GPU allocation are recorded by the native experiment.

Example (choose a fresh output directory):

```bash
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.state_interface_integration --window-study --case local --frames 4 --scenes 1 --warmup 1 --warmup-frames 4 --repeats 2 --output outputs/window_local_fresh
taskset -c 0-3,6-31 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.state_interface_integration --window-study --case natural --frames 20 --scenes 3 --warmup 1 --warmup-frames 4 --repeats 2 --output outputs/window_natural_fresh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m2.window_study_summary outputs/window_natural_fresh
```

`local` retains the first actual preprocessed image and changes one pixel in
camera zero each later frame; metadata and downstream tasks remain live.
It proves the mechanism, not natural-scene acceleration. Natural sequences
retain actual input images. Warmup sequences are excluded from summaries.
Outputs include protocol, source snapshots/hashes, per-request counters and
lineage, paired full-task quality differences, memory snapshots and language.

Malformed current/source timestamps return full execution without publication.
Historical preprocessing coordinates and state epoch must match. Invalid
compressed mask payloads become validation errors, so diagnostic decoding cannot
break a successful full fallback. `--warmup-frames 4` only shortens warmup;
all measured scenes still use the complete requested frame sequence.

Dependency admission verifies the original EVA/PatchEmbed implementations,
registered prefix module types, and pointwise one-axis LayerNorm. A nonlocal
replacement retaining the same parameter objects fails closed. The cache
signature also tracks module classes and forward implementations.
