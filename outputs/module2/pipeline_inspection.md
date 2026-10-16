# Local pipeline inspection

Environment: 1.13.1+cu117, CUDA 11.7, NVIDIA GeForce RTX 3090.
Inspected config: `worldstream_m1_coverage_observe.py`, inherited deployed EVA config.
Weights: `ckpts/iter_10548.pth`; actual checkpoint is loaded strictly in all GPU runs.

Input [6,3,640,640], patch Conv2d kernel=stride=16, padding=0, dilation=1.
Patch output [6,40,40,1024]. Absolute positions are interpolated on this grid.
Window size 16 TOKENS = 256 image pixels. Pad 40x40 to 48x48: 9 windows per camera,
54 independent attention batches, each 256 tokens (edge padding participates in attention).
Window blocks: [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22]. Global blocks: [2, 5, 8, 11, 14, 17, 20, 23].
Cache only blocks 0,1. First global barrier is block 2; later blocks execute completely.
Projection and position embeddings always execute completely. No residual bottleneck is
enabled in deployed config; planner supports its 3x3 dependency, executor falls back before it.

`Block._forward`: norm1 -> zero pad -> window partition -> original Attention ->
unpartition/crop -> residual -> norm2 -> SwiGLU -> residual. Sparse path preserves
norm-before-padding, local RoPE, head dimension, and the independent batch axis.
MLP on edge padding performs surplus work; cropped padding never leaks to valid positions.
Global RoPE has 1600 entries, window RoPE has 256 entries. FlashAttention is enabled.
Backbone returns list with NCHW tensor; Petr3D reshapes to B,N,C,H,W after optional neck.
GridMask returns input unchanged in eval; training never uses M2. Neck/ROI/detection/map/
temporal memory/LLM execute in original order. TRT export has its own non-flash path;
no selective TRT execution is claimed.

Live preprocessing: raw 1600x900 -> resize by max(320/900,640/1600)=0.4 -> 640x360,
bottom crop to 640x320 (40 rows removed above) -> anisotropic resize to 640x640 ->
normalization -> size-divisor-32 pad. Intrinsics are updated by actual transforms.
Live M1 sees de-normalized preprocessed images (640x640), matching model coordinates.
Old M1 replay masks at 640x360 are explicitly rejected: naively resizing these masks would
omit crop and interpolation footprints. No raw-to-token equality is inferred from flow.

M1 has 3480/1013760 complete 16-pixel image tiles, zero certified deep tokens;
its transformer_dependency_certified=False remains respected. M2 independently compares
all actual input pixels at their same position and unions this invalidation with M1 masks.
Unknown pixels always invalidate their full patch and attention window. Cache includes
scene, camera order, shape, frame/version, parameter/buffer pointers and mutation versions,
precision/backend policy and execution signature. Model load and inference reset clear it.
