# Reproduce the frozen cross-frame experiment

Run from `/data/qh/phdCode/work4/OmniDrive`. Use the existing `OmniDrive`
environment at `/home/qh/miniconda3/envs/OmniDrive/bin/python`. All commands
below inherit CPU affinity `0-3,6-31`; hardware-fault CPUs 4/5 are excluded.
The local RTX 3090, original nuScenes data and `ckpts/iter_10548.pth` are
required. No model weights or data are downloaded or modified.

The recorded baseline is `169d7e3023165030cabf962c12311f7c2cdf5480` (also
GitHub main at the start). The frozen `before_m3/` source is diagnostic only;
all model executions use the existing production runtime and PrefillExecutor.
Final implementation/delivery commits are recorded in `delivery.json`.

First inspect GPU idleness; never stop another user's process:

```sh
taskset -c 0-3,6-31 nvidia-smi
```

Copy only the predeclared inputs into a **new** output directory. This
reproduces the exact scene/question selection rather than treating previously
observed acceptance scenes as fresh development data:

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python - <<'PY'
from pathlib import Path
import shutil
source = Path('outputs/m3_crossframe_e2e_optimization_run01')
target = Path('outputs/m3_crossframe_e2e_reproduce_run01')
target.mkdir(exist_ok=False)
for name in ('experiment_plan.json', 'development_sequence.json',
             'single_sequence.json', 'multi_sequence.json',
             'frozen_m3_policy.json', 'question_sources.json',
             'historical_data_audit.json', 'calibration_semantic_review.json',
             'historical_semantic_failure.json', 'semantic_state_attribution.json'):
    shutil.copy2(source/name, target/name)
shutil.copytree(source/'before_m3', target/'before_m3',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
PY
```

Run the captured development diagnostic, one limited calibration, real-input
planner equivalence/microbenchmark, then the complete three-method campaign:

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.crossframe_e2e --root outputs/m3_crossframe_e2e_reproduce_run01 --stage diagnostic
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.crossframe_e2e --root outputs/m3_crossframe_e2e_reproduce_run01 --stage calibrate
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.crossframe_e2e --root outputs/m3_crossframe_e2e_reproduce_run01 --stage micro
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.crossframe_e2e --root outputs/m3_crossframe_e2e_reproduce_run01 --stage all
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.crossframe_e2e --root outputs/m3_crossframe_e2e_reproduce_run01 --stage post
```

Each GPU stage uses the shared existing job lock and waits for GPU memory to
drop below 500 MiB. Workers are sequential. A completed worker is preserved;
partial runs cause an error and must be inspected, never silently overwritten.
Regression retries retain separate failure logs. `commands.json` records the
actual worker invocations and return codes.

There are 4 development scenes (2 calibration, 2 validation) and 6 independent
acceptance scenes, 6 consecutive frames per scene. First natural questions
retain original Conv/VQA file order: 1 per frame for `single`, 3 for `multi`.
All four dependency profiles are represented without repeating/reordering
questions. Scene selection uses pre-execution metadata only.

For **each** workload, natural generation and equal-token generation both use
three rounds with orders Native/Zero/Reuse, Reuse/Zero/Native, Zero/Native/Reuse.
Fixed token IDs and step counts are taken from the same workload's actual
Native free generation. Every prefill/suffix and incremental decode forward
still executes; forced-token runs are never scored as natural quality.

The Zero control calls the same planner/cache/executor but forces historical
reuse length to zero. Only Native, Zero and the one formal optimized route are
timed. M1/M2 configuration and model weights remain unchanged. The calibration
does not write the formal policy. Existing semantic rejection records remain
binding. The before/after planner probe holds observed scheduler overhead fixed
for decision equivalence and alternates measurements on real expanded inputs.
Its extra probes are excluded from all service timing conclusions.

The `post` stage adds a limited two-calibration-scene Zero/trial fixed-token
control (three alternating rounds), independently observes actual prefill
input/KV/mask/position shapes and every decode input, profiles six full
development service calls including preprocessing, runs a seeded CUDA witness,
hashes the used external resources, and generates the reports/SHA index. None
of these diagnostic runs enters headline service timing or independent quality
acceptance. Passing post-stage artifacts are reused on a report-only retry.

`crossframe_report` checks exact frame/question/input digest pairing,
perception-output equality between Zero/Reuse, actual prefill forward lengths,
planned/actual reuse agreement, equal decode histories/forward counts, and
checkpoint/worker integrity. Confidence intervals resample **scenes** after
averaging repeats within scenes (10,000 draws, seed 20261027).

Official nuScenes/OpenLane evaluator functions score the declared subset;
subset scores must not be presented as full-validation scores. Inspect changed
answers and `calibration_semantic_review.json` separately from proxy Token F1.
The new crossing assertion is rejected despite lexical F1 improvement.

Large model captures, raw predictions and profiler traces stay on this server.
Their paths, sizes and SHA256 are in `raw_artifacts_sha256.json`; do not commit
private camera images, datasets or checkpoints. Compact per-request/frame
summaries, original natural answers, policies, source manifests, test logs and
the required reports are the reviewable delivery artifacts.
