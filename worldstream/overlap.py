"""One CPU precheck in flight; the main thread owns only full EVA until join.

The manager is exclusively leased to the worker. Failure/timeout must drain the
worker before the coordinator invalidates the manager or temporal heads.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from threading import get_ident,get_native_id
from time import perf_counter


class M1Worker:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='m1-precheck')
        self.future=None
        self.diagnostics={}

    def submit(self,manager,frame):
        if self.future is not None:raise RuntimeError('M1 worker already owns a frame')
        submitted=perf_counter()
        def work():
            start=perf_counter()
            try:return manager.precheck(frame)
            finally:self.diagnostics.update(worker_thread=get_ident(),worker_start_s=start,
                queue_delay_ms=(start-submitted)*1000,
                worker_native_thread=get_native_id(),worker_end_s=perf_counter(),worker_ms=(perf_counter()-start)*1000)
        self.diagnostics=dict(submitted_s=submitted,main_thread=get_ident())
        import torch
        with torch.autograd.profiler.record_function('m1_overlap_clock_sync'):
            self.diagnostics['clock_sync_s']=perf_counter()
        self.future=self.pool.submit(work)

    def join(self,timeout_s):
        if self.future is None:raise RuntimeError('No pending M1 precheck')
        start=perf_counter();future=self.future
        try:
            try:return future.result(timeout=timeout_s)
            except FutureTimeout:
                # Python/OpenCV cannot be safely killed. Drain before reset;
                # timeout reports failure, never permission to use stale state.
                self.diagnostics['timed_out']=True
                try:future.result()
                except Exception:pass
                raise TimeoutError('M1 precheck timeout; worker drained before invalidation')
        finally:
            self.future=None
            self.diagnostics.update(join_wait_ms=(perf_counter()-start)*1000,joined_s=perf_counter())

    def drain(self):
        if self.future is not None:
            try:self.future.result()
            except Exception:pass
            finally:self.future=None

    def close(self):
        self.drain();self.pool.shutdown(wait=True)
