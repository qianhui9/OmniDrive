#!/usr/bin/env bash
# Evidence record of completed commands. Original result directories are protected.
# Fresh complete reproduction: bash tools/worldstream_m1/reproduce_refinement.sh outputs/module1_refinement_reproduced
# Interpreter for every command: /home/qh/miniconda3/envs/OmniDrive/bin/python
# Environment: MPLCONFIGDIR=/tmp/omnidrive-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m tools.worldstream_m1.diagnose_validity
python -u -m tools.worldstream_m1.run_refinement
# Before optimization; hash-identical source snapshot is archived in reproducibility/before_optimization.
python -u -m tools.worldstream_m1.benchmark_refinement --output outputs/module1_refinement/latency_before_optimization.json
# GPU commands executed on the authorized host, where /dev/nvidia* are available.
python -u -m tools.worldstream_m1.gpu_refinement
python -u -m tools.worldstream_m1.benchmark_refinement
python -m tools.worldstream_m1.map_demo --config configs/worldstream_m1_refinement.yaml --output outputs/module1_refinement/map_annotation_demo
python -m tools.worldstream_m1.diagnose_constraints
python -u -m tools.worldstream_m1.run_refinement --output outputs/module1_refinement/optimized_replay --methods new_full
python -u -m tools.worldstream_m1.gpu_refinement --output outputs/module1_refinement/gpu_optimized --skip-feature-oracle
python -m pytest tests/worldstream_m1 -q --junitxml=outputs/module1_refinement/unit_tests.xml
python -m tools.worldstream_m1.summarize_refinement
