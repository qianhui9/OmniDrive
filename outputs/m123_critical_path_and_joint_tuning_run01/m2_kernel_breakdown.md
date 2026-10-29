# Actual CUDA work

| Probe | Prefix kernels / ms | Global kernels / ms | Grid kernel ms | Cache kernel ms |
|---|---:|---:|---:|---:|
| 000_01 | 94 / 38.431 | 939 / 430.344 | 0.000 | 1.593 |
| 000_06 | 94 / 38.366 | 939 / 432.468 | 0.000 | 1.595 |
| 001_01 | 92 / 24.950 | 11 / 0.905 | 0.353 | 1.590 |
| 001_06 | 92 / 31.994 | 939 / 434.159 | 0.346 | 1.596 |

000 is same-executor Full and001 is Incremental, frames01 and06 of the same8-frame prefix. On frame01 the incremental global suffix is reused; on06 it refreshes. Full and Incremental share weights, dtype, resolution, token layout, native block functions and synchronization. Only the private M2 refresh hint changes in Full; M1 processing is preserved. Each ordered condition has its own continuous closed-loop state and cache; downstream state differences can arise from actual approximate outputs.

On frame06 the cached prefix falls from38.366ms to31.994ms of actual kernels, while the global suffix remains432.468ms versus434.159ms. On frame01 the global update falls from430.344ms/939 kernels to0.905ms/11 kernels. Partial-window GEMMs/attention perform less work even though gather/scatter can add kernel launches; a kernel-count reduction is not proportional to a window-count reduction. There is no full-prefix recompute followed by cosmetic cache replacement in the partial path. The patch embedding always executes once. Cache finite validation/ownership and global attention on refresh frames are retained correctness costs.

The prefix packing costs about0.7ms in Full; partial output-grid reconstruction about0.35ms; refresh selection about0.62ms kernels and cache publication about1.6ms kernels. Detailed copy/runtime-launch totals and kernel names are in m2_detailed_profile.json. Correlation IDs map GPU events to CPU launch ranges. Nested ranges overlap; kernel durations measure work, CUDA events measure elapsed stream spans including launch gaps/waits, and synchronized outer E2E measures the full model. Their differences are not a hardware SM-idle measurement. CPU adapter D2H and cuda_runtime synchronize calls are separately retained in raw profiles.

In the initial16-frame Incremental sequence,12 frames refresh the global suffix because drift exceeds the existing threshold. This dominates the observed limited benefit despite many prefix windows reused. Remaining executor overhead is small compared with required suffix computation; stop M2 implementation edits and test a finite set of existing refresh parameters. No architecture, window geometry, new risk selector or kernel library is introduced.
