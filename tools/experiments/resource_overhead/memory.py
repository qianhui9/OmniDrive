"""Read-only storage accounting and bounded, background process sampling."""
import ctypes
import resource
import sys
import threading
import time
import types
from collections import Counter

import numpy as np
import psutil
import torch


def storage(tensor):
    value = tensor.untyped_storage() if hasattr(tensor, 'untyped_storage') else tensor.storage()
    nbytes = value.nbytes() if hasattr(value, 'nbytes') else value.size() * tensor.element_size()
    return (str(tensor.device), value.data_ptr(), nbytes), nbytes


class SizeLedger:
    """One ledger per snapshot: aliases are assigned to the first owner.

    Exact storage payloads and estimated Python heap are separate. Never walks
    nn.Modules, functions, classes, or model parameter/buffer storage.
    """
    def __init__(self, model=None, excluded=None):
        self.objects, self.storages = set(), set()
        self.excluded = excluded if excluded is not None else set()
        if model is not None:
            self.excluded = {storage(t)[0] for t in list(model.parameters()) + list(model.buffers())}

    def measure(self, value):
        out = Counter(cpu_storage_bytes=0, gpu_storage_bytes=0, estimated_python_bytes=0,
                      tensor_elements=0, numpy_elements=0, unique_storages=0, opaque_objects=0)

        def visit(obj):
            if obj is None or id(obj) in self.objects:
                return
            self.objects.add(id(obj))
            if isinstance(obj, (torch.nn.Module, types.ModuleType, type)) or callable(obj):
                return
            if isinstance(obj, torch.Tensor):
                key, size = storage(obj)
                if key in self.excluded:
                    return
                out['estimated_python_bytes'] += sys.getsizeof(obj)
                if key not in self.storages:
                    self.storages.add(key)
                    out['gpu_storage_bytes' if obj.is_cuda else 'cpu_storage_bytes'] += size
                    out['unique_storages'] += 1
                    out['tensor_elements'] += obj.numel()
                return
            if isinstance(obj, np.ndarray):
                owner = obj
                while isinstance(owner.base, np.ndarray):
                    owner = owner.base
                # frombuffer arrays keep their bytes/memoryview owner alive.
                if owner.base is not None:
                    out['estimated_python_bytes'] += sys.getsizeof(obj)
                    visit(owner.base)
                else:
                    out['estimated_python_bytes'] += max(0, sys.getsizeof(obj) - (obj.nbytes if obj.flags.owndata else 0))
                    key = ('cpu', owner.__array_interface__['data'][0], owner.nbytes)
                    if key not in self.storages:
                        self.storages.add(key)
                        out['cpu_storage_bytes'] += owner.nbytes
                        out['numpy_elements'] += owner.size
                        out['unique_storages'] += 1
                return
            out['estimated_python_bytes'] += sys.getsizeof(obj)
            if isinstance(obj, memoryview):
                visit(obj.obj)
            elif isinstance(obj, dict):
                for key, item in obj.items():
                    visit(key)
                    visit(item)
            elif isinstance(obj, (tuple, list, set, frozenset)):
                for item in obj:
                    visit(item)
            elif hasattr(obj, '__dict__'):
                visit(vars(obj))
            elif hasattr(type(obj), '__slots__'):
                slots = type(obj).__slots__
                for name in ([slots] if isinstance(slots, str) else slots):
                    if hasattr(obj, name):
                        visit(getattr(obj, name))
            elif not isinstance(obj, (str, bytes, bytearray, int, float, bool, complex, np.generic)):
                out['opaque_objects'] += 1
        visit(value)
        result = dict(out)
        result['cpu_bytes'] = out['cpu_storage_bytes'] + out['estimated_python_bytes']
        result['gpu_bytes'] = out['gpu_storage_bytes']
        return result


def module_snapshot(model):
    """Full retained runtime graph, plus named cache payloads (subset columns).

    Runtime graphs include diagnostics/metadata, DAVP planner/executor caches,
    and OGSM previous cameras. Model weights and shared storage are excluded.
    Named payloads are independently measured subsets; never add them to totals.
    """
    ledger = SizeLedger(model)
    result = []
    for name, attr in [('OGSM', 'm1_manager'), ('DAVP', 'm2_runtime'), ('QCSE', 'm3_runtime')]:
        runtime = getattr(model, attr, None)
        roots = [runtime] if runtime is not None else []
        if name == 'OGSM' and runtime is not None:
            roots += [getattr(model, key, None) for key in
                      ('m3_committed_state', 'm3_verified_state', '_m1_frame', '_m2_context')]
        # Iterate roots without counting instrumentation-created containers.
        sizes = Counter()
        for root in roots:
            sizes.update(ledger.measure(root))
        row = dict(component=name, enabled=runtime is not None, **dict(sizes), entries=0, elements=0)
        if name == 'OGSM' and runtime is not None:
            anchors = runtime.store.anchors
            row.update(object_anchors=sum(a.anchor_type == 'object' for a in anchors.values()),
                       map_anchors=sum(a.anchor_type == 'map' for a in anchors.values()),
                       entries=len(anchors) + len(runtime.store.features), elements=len(anchors))
            subset = SizeLedger(excluded=ledger.excluded).measure(runtime.store)
            row['world_state_cpu_bytes'] = subset['cpu_bytes']
            row['world_state_gpu_bytes'] = subset['gpu_bytes']
        elif name == 'DAVP' and runtime is not None:
            cache = runtime.cache
            windows = sum(len(t) for t in cache.outputs.values())
            row.update(entries=windows + int(cache.global_anchor is not None), elements=windows,
                       cached_windows=windows, cached_layers=len(cache.outputs))
            subsets = [('window_features', cache.outputs), ('global_anchor', cache.global_anchor),
                       ('global_residual', getattr(cache.global_anchor, 'residual', None))]
            for label, value in subsets:
                subset = SizeLedger(excluded=ledger.excluded).measure(value)
                row[label + '_cpu_bytes'] = subset['cpu_bytes']
                row[label + '_gpu_bytes'] = subset['gpu_bytes']
        elif name == 'QCSE' and runtime is not None:
            records = runtime.answer_memory.records
            row.update(entries=len(records), elements=sum(len(r.tokens) for r in records.values()),
                       historical_candidate_tokens=sum(len(r.tokens) for r in records.values()),
                       prompt_tokens=sum(len(r.prompt_ids) for r in records.values()), persistent_kv_bytes=0)
            row['candidate_cache_bytes'] = SizeLedger(excluded=ledger.excluded).measure(runtime.answer_memory)['cpu_bytes']
        for key in ('cpu_bytes', 'gpu_bytes', 'cpu_storage_bytes', 'gpu_storage_bytes', 'estimated_python_bytes'):
            row.setdefault(key, 0)
        result.append(row)
    return result


class NVML:
    """Use the installed NVML C ABI when pynvml is unavailable (v2 process struct)."""
    class Process(ctypes.Structure):
        _fields_ = [('pid', ctypes.c_uint), ('usedGpuMemory', ctypes.c_ulonglong),
                    ('gpuInstanceId', ctypes.c_uint), ('computeInstanceId', ctypes.c_uint)]

    def __init__(self, device, pid):
        self.pid, self.error, self.lib = pid, None, None
        try:
            self.lib = ctypes.CDLL('libnvidia-ml.so.1')
            self.check(self.lib.nvmlInit_v2())
            self.handle = ctypes.c_void_p()
            self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(device), ctypes.byref(self.handle)))
            self.read()
        except (OSError, RuntimeError, AttributeError) as exc:
            self.error = str(exc)

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError('NVML return code ' + str(code))

    def read(self):
        if self.error:
            return None
        count = ctypes.c_uint(64)
        values = (self.Process * count.value)()
        code = self.lib.nvmlDeviceGetComputeRunningProcesses_v2(self.handle, ctypes.byref(count), values)
        if code == 7:  # NVML_ERROR_INSUFFICIENT_SIZE
            values = (self.Process * count.value)()
            code = self.lib.nvmlDeviceGetComputeRunningProcesses_v2(self.handle, ctypes.byref(count), values)
        self.check(code)
        matches = [v.usedGpuMemory for v in values[:count.value] if v.pid == self.pid]
        return sum(matches) if matches and all(v != 2**64-1 for v in matches) else None

    def close(self):
        if self.lib is not None:
            self.lib.nvmlShutdown()


class ProcessMonitor:
    def __init__(self, device=0, interval=.05, uss_interval=1., gpu=True):
        self.process = psutil.Process()
        self.interval, self.uss_interval = interval, uss_interval
        self.nvml = NVML(device, self.process.pid) if gpu else None
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.errors = set()
        self.phases = {}
        self.phase = 'load'
        self.last_uss_at, self.uss = 0., None
        self.thread = threading.Thread(target=self.loop, name='resource-sampler', daemon=True)

    def sample(self):
        now = time.perf_counter()
        rss = self.process.memory_info().rss
        if now - self.last_uss_at >= self.uss_interval:
            try:
                self.uss = getattr(self.process.memory_full_info(), 'uss', None)
            except (psutil.Error, OSError) as exc:
                self.errors.add(str(exc))
            self.last_uss_at = now
        nvml = None
        if self.nvml is not None:
            try:
                nvml = self.nvml.read()
            except (RuntimeError, AttributeError) as exc:
                self.errors.add(str(exc))
        record = dict(cpu_rss_bytes=rss, cpu_uss_bytes=self.uss, nvml_process_gpu_bytes=nvml)
        with self.lock:
            stat = self.phases.setdefault(self.phase, dict(samples=0, rss_sum=0, rss_peak=0, uss_peak=None, nvml_peak=None))
            stat['samples'] += 1
            stat['rss_sum'] += rss
            stat['rss_peak'] = max(stat['rss_peak'], rss)
            for key, value in [('uss_peak', self.uss), ('nvml_peak', nvml)]:
                if value is not None:
                    stat[key] = max(stat[key] or 0, value)
        return record

    def loop(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.sample()
            except (psutil.Error, OSError) as exc:
                self.errors.add(str(exc))

    def start(self):
        self.sample()
        self.thread.start()

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase

    def finish(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        if self.nvml:
            self.nvml.close()
        return dict(phases=self.phases, errors=sorted(self.errors),
                    nvml_error=self.nvml.error if self.nvml else 'disabled',
                    process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    sampling_interval_seconds=self.interval, uss_interval_seconds=self.uss_interval)


def cuda_snapshot():
    torch.cuda.synchronize()
    return dict(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                gpu_reserved_bytes=torch.cuda.memory_reserved(),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
