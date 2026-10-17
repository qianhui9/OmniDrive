# Profiling instrumentation changelog

Standalone probes and offline audits are excluded from ordinary E2E timing. Common request scopes and token capture are included. Core worker remains bounded to one Store owner.

| File | Type | Instrumentation | Lines |
|---|---|---|---|
| `worldstream/overlap.py` | modified | Worker queue delay, existing monotonic start/end/join diagnostics | See source snapshot, content hash in delivery_manifest.json |
| `projects/mmdet3d_plugin/models/detectors/petr3d.py` | modified | Actual post-head task verification wall time and profiler scope | See source snapshot, content hash in delivery_manifest.json |
| `tools/system_optimization/v3.py` | created | Actual runner hooks, post-timer source/temporal snapshots and environment records | See source snapshot, content hash in delivery_manifest.json |
| `tools/system_optimization/v3_probes.py` | created | Real GPU timeline, key cProfile and certificate matching controls | See source snapshot, content hash in delivery_manifest.json |
| `tools/system_optimization/v3_monitor.py` | created | Read-only CPU affinity, threads, shared GPU processes, clocks and temperature | See source snapshot, content hash in delivery_manifest.json |
| `tools/system_optimization/v3_report.py` | created | Actual kernel attribution, scene statistics, dependency bound and standalone plots | See source snapshot, content hash in delivery_manifest.json |
| `tools/system_optimization/v3_physical.py` | created | Offline case evidence, risk replay, verifier wall/process CPU measurements | See source snapshot, content hash in delivery_manifest.json |
| `tools/system_optimization/v2_state_supply.py` | modified | Reuse long-stream native checkpoints with optional post-perception mode | See source snapshot, content hash in delivery_manifest.json |
| `tools/system_optimization/v2_state_quality.py` | modified | Offline original annotation scoring for a specified supplemental run | See source snapshot, content hash in delivery_manifest.json |

Additional shared helpers: `physical_oracle.py` accepts a supplemental query subdirectory, preserving its default; `v2_state_supply.py` accepts explicit checkpoint ordinals and a post-perception base mode, preserving v2 defaults. New thread/resource variations reuse the existing ResidentSampler. All controls remain outside scene-level performance statistics.
