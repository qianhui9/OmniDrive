#!/usr/bin/env bash
# Commands used for v4. Frozen outputs are intentionally never overwritten.
# Run from /data/qh/phdCode/work4/OmniDrive; a fresh reproduction needs a new
# output root in v4.py/v4_facts.py/v4_report.py and a newly frozen protocol.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/omnidrive-v4-mpl
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.task_dependency_audit
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.v4 --freeze
taskset -c 0-3,6-31 bash tools/system_optimization/run_v4.sh
# Read-only monitor launched alongside the GPU runner:
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.system_optimization.v4_monitor
# After manual review of original metric questions, before ANY new Fact Oracle:
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.v4_facts --freeze
# After the real captures complete:
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.v4_facts
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.v4_report
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.system_optimization.v4_delivery
# The commands after GPU execution were also orchestrated through:
# taskset -c 0-3,6-31 bash tools/system_optimization/finalize_v4.sh
