"""Audit local EVA and preprocessing before sparse execution experiments."""
from pathlib import Path
import json
import subprocess
from .common import write


def main():
    import torch
    from mmcv import Config
    cfg = Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py')
    b = dict(cfg.model.img_backbone)
    windows = b['window_block_indexes']
    record = dict(torch=torch.__version__, cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(), device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        config=b, global_blocks=[i for i in range(b['depth']) if i not in windows],
        gpu=subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout,
        git_status=subprocess.run(['git', 'status', '--short'], capture_output=True, text=True).stdout,
        m1=json.loads(Path('outputs/module1_refinement/coverage_final/m2_candidates.json').read_text()))
    write('outputs/module2/environment.json', record)
    Path('outputs/module2/pipeline_inspection.md').write_text(f'''# Local pipeline inspection

Environment: {record['torch']}, CUDA {record['cuda']}, {record['device']}.
Inspected config: `worldstream_m1_coverage_observe.py`, inherited deployed EVA config.
Weights: `ckpts/iter_10548.pth`; actual checkpoint is loaded strictly in all GPU runs.

Input [6,3,640,640], patch Conv2d kernel=stride=16, padding=0, dilation=1.
Patch output [6,40,40,1024]. Absolute positions are interpolated on this grid.
Window size 16 TOKENS = 256 image pixels. Pad 40x40 to 48x48: 9 windows per camera,
54 independent attention batches, each 256 tokens (edge padding participates in attention).
Window blocks: {windows}. Global blocks: {record['global_blocks']}.
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
''')


if __name__ == '__main__': main()
