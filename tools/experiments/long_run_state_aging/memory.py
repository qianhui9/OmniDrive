"""Allocator peaks for one complete five-query frame; no cache mutation."""
from pathlib import Path
import resource

MIB = 1024 ** 2


def host_rss_mb():
    # Linux reports kB here. This is a point sample, never called a stage peak.
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 1024
    return None


class FrameMemory:
    def __init__(self, cuda):
        self.cuda = cuda

    def start(self):
        self.host_before_mb = host_rss_mb()
        # reset_peak_memory_stats resets statistics only, never empty_cache.
        self.cuda.reset_peak_memory_stats(0)

    def stop(self):
        return dict(gpu_peak_allocated_mb=self.cuda.max_memory_allocated(0) / MIB,
            gpu_peak_reserved_mb=self.cuda.max_memory_reserved(0) / MIB,
            gpu_end_allocated_mb=self.cuda.memory_allocated(0) / MIB,
            host_rss_before_mb=self.host_before_mb,
            host_rss_after_mb=host_rss_mb(),
            host_process_lifetime_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
