from contextlib import nullcontext
import pytest
from tools.component_latency_eval.timing import Recorder


class Event:
    def __init__(self, cuda):
        self.cuda = cuda
        self.recorded = False
    def record(self, stream):
        assert stream is self.cuda.stream
        self.recorded = True
    def elapsed_time(self, end):
        assert self.recorded and end.recorded and self.cuda.fences
        return 2.5


class Cuda:
    def __init__(self):
        self.fences = 0
        self.stream = object()
    def Event(self, enable_timing):
        assert enable_timing
        return Event(self)
    def current_stream(self):
        return self.stream
    def synchronize(self):
        self.fences += 1


def test_events_deferred_and_actual_stream():
    cuda = Cuda()
    r = Recorder('trial', cuda)
    r.frame('seq', 'frame')
    for _ in range(3):
        with r.span('GPU', gpu=True):
            pass
    assert cuda.fences == 0
    assert all('gpu_device_ms' not in row for row in r.rows)
    assert all('component_wall_ms' not in row for row in r.rows)
    r.resolve()
    assert cuda.fences == 1
    assert [row['gpu_device_ms'] for row in r.rows] == [2.5]*3
    r.clear()


def test_cpu_failure_and_blanks():
    r = Recorder('trial')
    r.frame('seq', 'frame')
    with pytest.raises(ValueError), r.span('CPU'):
        raise ValueError('expected')
    assert r.rows[0]['execution_status'] == 'FAILED'
    assert r.rows[0]['cpu_host_ms'] >= 0
    assert 'gpu_device_ms' not in r.rows[0]


def test_query_unit_and_only_e2e_fences():
    cuda = Cuda()
    r = Recorder('trial', cuda)
    r.frame('seq', 'frame')
    with r.span('QCSE.Total', gpu=True):
        for _ in range(2):
            with r.span('Query E2E', gpu=True, sync=True, query=True):
                with r.span('route'):
                    pass
    assert cuda.fences == 4
    assert [row['query_id'] for row in r.rows if row['component_name']=='route'] == [0,1]
    assert r.rows[-1]['sample_unit'] == 'frame'
    r.resolve(already_synchronized=True)
    assert cuda.fences == 4


def test_control_has_no_component_events():
    cuda = Cuda()
    r = Recorder('baseline', cuda, enabled=False, query_sync=False)
    r.frame('seq', 'frame')
    with r.span('Frame E2E', sync=True):
        with r.span('Query E2E', gpu=True, sync=True, query=True):
            with r.span('device', gpu=True):
                pass
    assert cuda.fences == 2
    assert [row['component_name'] for row in r.rows] == ['Frame E2E']
    assert not r.pending
