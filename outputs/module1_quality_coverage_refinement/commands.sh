#!/usr/bin/env bash
# Commands actually used during this task, with final research-config name.
# Run from repository root in a fresh output workspace. Runners refuse to
# overwrite completed scene traces. Preserve this directory when reproducing.

taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.refinement_data_audit
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.refinement_diagnostics --workers 4
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.refinement_funnel_support
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.refinement_data_manifest
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.refinement_experiments --stage development --workers 4 --freeze-source
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.refinement_report
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_quality_coverage_refinement/regression_cpu.xml

# GPU commands require host device access and successful approval. Their actual
# executed / Not Run status is recorded in original_model_parity.json.
taskset -c 0-3,6-31 nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q --junitxml=outputs/module1_quality_coverage_refinement/regression_cuda.xml
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m1.model_smoke --compare --perception-only --frames 8 --m1-config configs/worldstream_m1_quality_coverage_refinement.yaml --output outputs/module1_quality_coverage_refinement/gpu_parity

# This bounded final-source replay was also executed after provenance hardening:
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.refinement_experiments --stage development --workers 4 --methods surface_consensus,temporal_object --output outputs/module1_quality_coverage_refinement/post_hardening_replay --freeze-source
# A fresh repeat of all methods can use another --output and --freeze-source.
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.refinement_identity_panels
taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_refinement_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m1.refinement_finalize
