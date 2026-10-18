"""Serializable frame costs with explicit computation lineage and timing scopes."""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import perf_counter, process_time


@dataclass(frozen=True)
class FrameIdentity:
    scene_id: str
    frame_id: str
    timestamp: float
    camera_ids: tuple
    model_version: str
    lora_version: str
    execution_signature: str


@dataclass
class FrameExecutionRecord:
    identity: FrameIdentity
    question_count: int
    service: str
    stages: list = field(default_factory=list)
    m1: dict = field(default_factory=dict)
    m2: dict = field(default_factory=dict)
    m3: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    fallback_reasons: list = field(default_factory=list)
    lifecycle_resets: list = field(default_factory=list)
    admission: dict = field(default_factory=dict)
    e2e_wall_ms: float = None
    inference_wall_ms: float = None
    input_preparation_wall_ms: float = None
    gpu_kernel_ms: float = None  # Only a real profiler trace can supply this.
    status: str = 'running'
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)

    def append(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            handle.write(json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)+'\n')


class StageRecorder:
    """Nested wall/CPU and CUDA intervals. GPU intervals are NOT kernel time.

    Direct child wall durations are subtracted once; inclusive parent CUDA
    intervals are never summed with their children. Synchronize once per frame.
    """
    def __init__(self, record, cuda=False):
        self.record, self.cuda = record, cuda
        self.stack, self.events = [], []

    @contextmanager
    def stage(self, name, source=None):
        index = len(self.record.stages)
        row = dict(name=name, parent=self.stack[-1] if self.stack else None,
                   computed_source=source or asdict(self.record.identity))
        self.record.stages.append(row)
        self.stack.append(index)
        if self.cuda:
            import torch
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
        tick, cpu = perf_counter(), process_time()
        try:
            if self.cuda:
                import torch
                scope = torch.autograd.profiler.record_function('stream/'+name)
            else:
                scope = nullcontext()
            with scope:
                yield row
        finally:
            row.update(wall_ms=(perf_counter()-tick)*1000,
                       cpu_process_ms=(process_time()-cpu)*1000)
            if self.cuda:
                b.record()
                self.events.append((row, a, b))
            self.stack.pop()

    def finish(self):
        for row, a, b in self.events:
            row['cuda_interval_ms'] = a.elapsed_time(b)
        for i, row in enumerate(self.record.stages):
            children = sum(c['wall_ms'] for c in self.record.stages if c['parent'] == i)
            row['exclusive_wall_ms'] = max(0., row['wall_ms']-children)
        self.events.clear()
