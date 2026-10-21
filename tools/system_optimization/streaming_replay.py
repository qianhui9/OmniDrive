"""Trace-driven FCFS replay using real measured E2E durations only.

This models queueing, not the behavior/quality of an aged online world state.
Every measured frame remains a request. No synthetic throughput samples,
coalescing, deadlines-based dropping, or repeated traces are introduced.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .audit import write


def distribution(values):
    return dict(count=len(values),p50=float(np.percentile(values,50)) if values else None,
        p95=float(np.percentile(values,95)) if values else None,
        maximum=max(values) if values else None)


def replay(requests, deadline_s=.5):
    """Input schema uses explicit sensor/request arrival and measured service."""
    if not math.isfinite(deadline_s) or deadline_s<=0:raise ValueError('Invalid deadline')
    if not requests:raise ValueError('Empty measured request stream')
    fields=('sensor_arrival_s','request_arrival_s','measured_service_s')
    for r in requests:
        if any(k not in r or not math.isfinite(r[k]) for k in fields):raise ValueError('Missing/invalid measured event')
        if r['measured_service_s']<=0 or r['request_arrival_s']<r['sensor_arrival_s']:
            raise ValueError('Invalid service or arrival ordering')
    if len({r['request_id'] for r in requests})!=len(requests):raise ValueError('Duplicate request identity')
    if any(b['request_arrival_s']<a['request_arrival_s'] for a,b in zip(requests,requests[1:])):
        raise ValueError('Requests must be sorted by arrival')
    origin=requests[0]['sensor_arrival_s']; available=requests[0]['request_arrival_s']
    horizon=requests[-1]['request_arrival_s'];events=[]
    for r in requests:
        start=max(available,r['request_arrival_s']); finish=start+r['measured_service_s'];available=finish
        events.append(dict(r,inference_start_s=start,inference_finish_s=finish,
            queue_wait_s=start-r['request_arrival_s'],output_age_s=finish-r['sensor_arrival_s'],
            deadline_missed=finish-r['sensor_arrival_s']>deadline_s))
    completed=[e for e in events if e['inference_finish_s']<=horizon]
    started=[e for e in events if e['inference_start_s']<=horizon]
    unresolved=[e for e in events if e['inference_finish_s']>horizon]
    pending_misses=sum(horizon-e['sensor_arrival_s']>deadline_s for e in unresolved)
    drain=available-origin
    return dict(requests=len(events),completed_at_last_arrival=len(completed),
        pending_at_last_arrival=len(unresolved),queued_not_started_at_last_arrival=len(events)-len(started),
        completed_after_drain=len(events),request_arrival_equals_sensor_arrival=all(r['request_arrival_s']==r['sensor_arrival_s'] for r in requests),
        observation_horizon_s=horizon-origin,drain_finish_s=drain,
        deadline_s=deadline_s,deadline_miss_rate_after_drain=sum(e['deadline_missed'] for e in events)/len(events),
        already_missed_deadlines_at_last_arrival=sum(e['deadline_missed'] for e in completed)+pending_misses,
        queue_wait_s=distribution([e['queue_wait_s'] for e in events]),
        output_age_s=distribution([e['output_age_s'] for e in events]),
        measured_service_s=distribution([e['measured_service_s'] for e in events]),
        effective_completed_per_s_including_drain=len(events)/drain,
        steady_state_throughput=None,steady_state_reason='finite real trace; no trace tiling or fabricated steady-state observations',
        actual_queue_memory_growth_bytes=None,memory_reason='queue is simulated, no image payload residency allocated or measured',events=events)


def run(root):
    root=Path(root);groups=defaultdict(list);hashes={};memory=defaultdict(list)
    for path in sorted(root.glob('final_test_*/frames.jsonl')):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['warmup']:continue
            f=r['frame']; scene=f['identity']['scene_id'];mode=r['mode']
            groups[(scene,mode)].append(dict(request_id=f['identity']['frame_id'],
                sensor_arrival_s=f['identity']['timestamp'],request_arrival_s=f['identity']['timestamp'],
                measured_service_s=f['e2e_wall_ms']/1000.,measured_source=str(path)))
            memory[mode].append(f['memory'])
    if not groups:raise ValueError('No completed real final-test frames')
    records=[]
    for (scene,mode),requests in sorted(groups.items()):
        records.append(dict(scene=scene,mode=mode,**replay(sorted(requests,key=lambda r:r['request_arrival_s']))))
    modes={}
    for mode in sorted({r['mode'] for r in records}):
        rows=[r for r in records if r['mode']==mode];events=[e for r in rows for e in r['events']]
        modes[mode]=dict(scenes=len(rows),requests=len(events),
            service_contract='current_output' if mode=='original' else 'persistent_state',
            completed_at_last_arrival=sum(r['completed_at_last_arrival'] for r in rows),
            pending_at_last_arrival=sum(r['pending_at_last_arrival'] for r in rows),
            queue_wait_s=distribution([e['queue_wait_s'] for e in events]),
            output_age_s=distribution([e['output_age_s'] for e in events]),
            deadline_miss_rate_after_drain=sum(e['deadline_missed'] for e in events)/len(events),
            effective_completed_per_s_including_drain=sum(r['requests'] for r in rows)/sum(r['drain_finish_s'] for r in rows),
            measured_tracked_cache_peak_bytes=max(m.get('tracked_cache_bytes',0) for m in memory[mode]),
            actual_queue_memory_growth_bytes=None)
    value=dict(status='completed_trace_driven_replay',sources_sha256=hashes,
        contract='FCFS; one worker per scene; every original Planning request retained; no drops/coalescing',
        service_comparison='m1/overlap/unified_auto share persistent-state responsibility; original is separately labeled current-output service',
        assumptions=['sensor timestamp is replay arrival; request arrives with sensor',
            'actual isolated per-frame E2E durations reused once each; queueing itself adds no modeled inference cost',
            'scene resets independent; no cross-scene hidden state or repeated duration samples',
            'Auto/Overlap quality proved for recorded inputs only; aged live inference under backlog not tested'],
        memory_scope='measured cache payload only; actual queue memory and memory growth unmeasured',
        modes=modes,scenes=records)
    write(root/'streaming_replay_results.json',value)
    with (root/'traces/streaming_events.jsonl').open('w') as f:
        for r in records:
            for e in r['events']:f.write(json.dumps(dict(scene=r['scene'],mode=r['mode'],**e))+'\n')
    return value
