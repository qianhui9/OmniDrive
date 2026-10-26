"""Sampled diagnostic profiling; never installed in the serving runtime."""
import cProfile
import json
from pathlib import Path
import pstats
import torch


class RuntimeProfile:
    def __init__(self, directory):
        self.directory = Path(directory)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self.cpu = cProfile.Profile()
        self.gpu = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False)
        self.gpu.__enter__()
        self.cpu.enable()
        return self

    def __exit__(self, *error):
        self.cpu.disable()
        self.gpu.__exit__(*error)
        self.cpu.dump_stats(str(self.directory/'cpu.prof'))
        self.gpu.export_chrome_trace(str(self.directory/'cuda.json'))
        stats = pstats.Stats(self.cpu).stats
        names = ('source_digest', '_history_digest', 'frame_digest', 'fork', 'freeze',
                 'export_m2_inputs', '_precheck', '_commit', 'deepcopy', 'prepare_inputs_for_generation')
        rows = {name: dict(calls=sum(v[1] for k,v in stats.items() if k[2]==name),
            cumulative_ms=sum(v[3]*1000 for k,v in stats.items() if k[2]==name)) for name in names}
        (self.directory/'cpu_summary.json').write_text(json.dumps(rows, indent=2))
