"""Host spans and deferred current-stream CUDA events; no per-token hooks."""
from contextlib import contextmanager
from time import perf_counter_ns

FIELDS = ('trial_id', 'sequence_id', 'frame_id', 'query_id', 'component_name',
          'execution_path', 'sample_unit', 'cpu_host_ms', 'gpu_device_ms',
          'component_wall_ms', 'invocation_count', 'execution_status',
          'refreshed_windows', 'reused_windows', 'aggregation', 'parent_component')


class Recorder:
    def __init__(self, trial_id, cuda=None, enabled=True, query_sync=True):
        self.trial_id, self.cuda, self.enabled = trial_id, cuda, enabled
        self.query_sync = query_sync
        self.context = {}
        self.rows, self.pending = [], []
        self.query_count = 0

    def frame(self, sequence_id, frame_id):
        self.context = dict(sequence_id=sequence_id, frame_id=frame_id, query_id='')
        self.query_count = 0

    @contextmanager
    def span(self, name, gpu=False, path='', parent='', sync=False, query=False):
        if query:
            sync = sync and self.query_sync
        if query:
            self.context['query_id'] = self.query_count
            self.query_count += 1
        active = self.enabled or sync
        if not active:
            try:
                yield {}
            finally:
                if query:
                    self.context['query_id'] = ''
            return
        if sync and self.cuda is not None:
            self.cuda.synchronize()
        row = dict(trial_id=self.trial_id, **self.context, component_name=name,
                   execution_path=path, sample_unit='query' if self.context.get('query_id', '') != '' else 'frame',
                   invocation_count=1, execution_status='COMPLETED', aggregation='invocation',
                   parent_component=parent)
        events = None
        if gpu and self.cuda is not None:
            stream = self.cuda.current_stream()
            events = (self.cuda.Event(enable_timing=True), self.cuda.Event(enable_timing=True), stream)
            events[0].record(stream)
        start = perf_counter_ns()
        try:
            yield row
        except BaseException:
            row['execution_status'] = 'FAILED'
            raise
        finally:
            row['cpu_host_ms'] = (perf_counter_ns() - start) / 1e6
            if events:
                events[1].record(events[2])
                self.pending.append((row, *events[:2]))
            if sync and self.cuda is not None:
                self.cuda.synchronize()
            if sync or not gpu:
                row['component_wall_ms'] = (perf_counter_ns() - start) / 1e6
            self.rows.append(row)
            if query:
                self.context['query_id'] = ''

    def resolve(self, already_synchronized=False):
        """One batch fence, or reuse the caller's completed E2E fence."""
        if self.pending and not already_synchronized:
            self.cuda.synchronize()
        for row, start, end in self.pending:
            row['gpu_device_ms'] = start.elapsed_time(end)
        self.pending.clear()

    def clear(self):
        if self.pending:
            raise RuntimeError('Resolve CUDA events before clearing records')
        self.rows.clear()
