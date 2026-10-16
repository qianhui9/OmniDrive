# M2: exact incremental EVA execution

M2 is opt-in. Use `projects/configs/OmniDrive/worldstream_m2_exact.py` to exercise
legal sparse execution and `worldstream_m2_auto.py` for measured-cost selection.
The original model/config remain the default. No training or checkpoint changes.

```python
from worldstream.m2 import IncrementalVisionRuntime, M2Config
runtime = IncrementalVisionRuntime(backbone, M2Config(mode='exact'))
with torch.no_grad():
    features, diagnostics = runtime.forward(images, m1_result, manager,
        dict(scene_id=scene, frame_id=frame, camera_ids=camera_order,
             coordinate_space='preprocessed_image', execution_signature=signature))
```

`images` are actual normalized NCHW model inputs, after GridMask. Training/grad
execution falls back. Without M1, exact whole-input comparison is a controlled
experiment; it does not represent M1 coverage. With M1, invalid pixels are unioned
with independently detected changes. Unknown/missing coordinate transforms fail
closed. In particular, raw replay masks cannot be resized into model masks.

Only early independent window blocks before the first global or convolutional
barrier are cached. EVA deployment caches blocks 0/1, always runs PatchEmbed and
positions, and fully executes blocks 2..23. Detection, map, temporal memory and
language execute normally. The cache owns one previous input and two padded
window tensors. Default 512 MiB budget includes a conservative 3x replacement/
scratch estimate; available CUDA memory is checked and sparse OOM retries dense.
Budget=0 disables caching. `reset()` explicitly frees history and scratch.

The executor gathers camera-major windows into a contiguous GPU buffer and runs
original norm/attention/RoPE/SwiGLU/residual weights only on that batch. A per-slot
validity tensor preserves EVA's normalize-before-padding semantics. Updated output
windows scatter into a copy of historical outputs; no full attention runs before
masking. The selected tensor batch really shrinks.

Cache signatures include model parameters and buffers (storage pointers/mutation
versions), topology, autocast/TF32 policy and external execution signature. Scene,
camera order, shape, duplicate frame and optional source frame/version are checked.
Use standard PyTorch in-place updates/load_state_dict; as elsewhere in PyTorch,
mutating `.data` outside version tracking is unsupported. Model load and model test
sequence resets invalidate M2. Calling consumers must preserve frame order.

Auto requires matching measured costs: torch/CUDA/device, shape, dtype, topology,
attention implementation and precision flags. It only accepts a measured update
count when total packed GPU interval AND wall cost beat dense. No extrapolation or
weighted score. Unknown table/shape/count uses dense. Exact mode is experimental
and may deliberately execute a slower but legal sparse batch.

Reproduction commands are in `outputs/module2/commands.sh`. The GPU runner uses
the production checkpoint, original FlashAttention and a full original EVA oracle.
CUDA events include launch gaps; profiler kernel-active time is separately labelled.
CUDA Graph measurements apply to fixed addresses/count and are experimental only;
the streaming runtime always retains the ordinary packed CUDA fallback.

Results and limitations: [experiment report](../outputs/module2/experiment_report.md).
