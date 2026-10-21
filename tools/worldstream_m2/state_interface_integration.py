"""Five-arm original OmniDrive full-task CUDA regression with fresh language paths.

Uses existing frozen quality and cost gates. Natural and repeated inputs are
separate invocations. This is an audited workload including diagnostic copies.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import random
import resource
import numpy as np
import torch
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from tools.worldstream_m1.io_utils import write_json
from .rearchitecture.integration import load
from .rearchitecture.common import arrays, memory, compare, quality_pass
from .refine_integration import planning_metric


def archive_sources(destination, sources, repository_root=None):
    """Snapshot one read per file; absolute paths must never replace originals."""
    repository_root=Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
    manifest={}
    for source in sources:
        source=Path(source).resolve()
        relative=source.relative_to(repository_root)
        payload=source.read_bytes()
        target=Path(destination)/'source'/relative
        if target.resolve()==source:
            raise ValueError('Archive destination aliases its input source')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(payload)
        manifest[str(relative)]=hashlib.sha256(payload).hexdigest()
    write_json(Path(destination)/'source_manifest.json',manifest)
    return manifest


@torch.no_grad()
def profile_zero_camera_cost(backbone, images, path):
    """Measure this exact signature; task qualification is initially withheld."""
    from worldstream.m2.camera_executor import camera_cost_key
    from worldstream.m2.profiler import measure, errors
    runtime = IncrementalVisionRuntime(backbone, dict(mode="exact", reuse_boundary="camera"))
    context = dict(scene_id="zero_cost_probe", frame_id="0", camera_ids=list(range(len(images))))
    runtime.forward(images, execution_context=context)
    cache = runtime.camera_executor.cache
    saved = copy.copy(vars(cache))
    def reuse():
        cache.__dict__.update(saved)
        return runtime.forward(images, execution_context=dict(context, frame_id="1"))[0][0]
    reference = backbone(images)[0]
    actual = reuse()
    if runtime.last_diagnostics.get("actual_reused_cameras") != len(images):
        raise AssertionError("Zero-update profiler did not reuse every camera")
    error = errors(actual, reference)
    if not torch.equal(actual, reference):
        raise AssertionError("Zero-update feature copy differs from the full backbone")
    reuse_cost = measure(reuse, warmup=3, repeats=10)
    full_cost = measure(lambda: backbone(images), warmup=3, repeats=10)
    table = dict(schema_version=1, records=[dict(
        key=camera_cost_key(backbone, images), changed_cameras=0,
        quality_certified=False, reuse_wall_ms=reuse_cost['wall_ms'],
        full_wall_ms=full_cost['wall_ms'], reuse_gpu_ms=reuse_cost['gpu_ms'],
        full_gpu_ms=full_cost['gpu_ms'], reuse=reuse_cost, full=full_cost,
        feature_error=error, qualification="Pending fresh full-task zero-update probe")])
    write_json(path, table)
    runtime.reset()
    return table
