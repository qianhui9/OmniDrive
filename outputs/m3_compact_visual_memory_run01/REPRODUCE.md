# WR-CVMS reproduction

Run from `/data/qh/phdCode/work4/OmniDrive`. Existing environment is OmniDrive; no installation or environment rebuild. CPU4/5 are faulty and must remain excluded. GPU jobs use the existing exclusive file lock and wait until memory.used < 500 MiB, never stop another process. In this Codex sandbox CUDA is not exposed: run GPU commands on the host through approved execution, with the same command and environment.

Base HEAD: `7e26bc10a6861caf29727817241f011e75b7b521`. Research branch `research/m3-compact-visual-memory`. Frozen native checkpoint `ckpts/iter_10548.pth`; existing pretrain weights/data must already be present. No downloads. Environment spec and ledger: `outputs/m3_wrcps_optimization_run01/environment_spec.json`, `.aris/compute/local.md`; unchanged spec, warm reuse. Raw `.pt` captures and untrained cost artifacts are kept locally and bound in source_and_artifact_bindings.json; they are not trained deployment checkpoints.

## Exact environment witness

```bash
taskset -c 0-3,6-31 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -c 'import os,torch; torch.manual_seed(0); assert not ({4,5}&set(os.sched_getaffinity(0))); x=torch.arange(64,dtype=torch.float32,device="cuda").reshape(8,8); value=float((x@x.T).sum()); assert value==510720.0; print("WITNESS",value,torch.cuda.get_device_name(0)); print("AFFINITY",sorted(os.sched_getaffinity(0)))'
```

Expected `WITNESS 510720.0 NVIDIA GeForce RTX 3090`, affinity excludes 4/5. Run only when the GPU is idle and no experiment holds its lock.

## New capture and bounded cost controls

The target root below MUST NOT already exist. The prepare step copies the two frozen sequences, not outcome-selected frames. Each GPU command runs sequentially; wait for completion before the next.

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.compact_visual_memory --stage prepare --root outputs/m3_compact_visual_memory_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory --stage audit --root outputs/m3_compact_visual_memory_reproduce01/natural_live
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_visual_memory --stage audit --root outputs/m3_compact_visual_memory_reproduce01/enriched_live
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.compact_visual_memory --stage cost --root outputs/m3_compact_visual_memory_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.compact_visual_memory --stage transactions --root outputs/m3_compact_visual_memory_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.compact_visual_memory_report --root outputs/m3_compact_visual_memory_reproduce01
```

Audit recomputes M1/M2 with zero language generation and saves native carriers/questions plus real EVA regions; eager and lazy bridge costs share each current frame. These audit inference times are not full-request service measurements. Cost controls use the frozen actual model and three source/target scene pairs, nine measured repeats per question/mode over three alternating rounds. Native, Compact ZeroKV and Compact ExactKV all have 513 visual tokens; extra text-only control isolates Stable8 savings. Untrained embeddings are used only for structure/cost. Cost generation has max_new_tokens=1, hence zero decode forwards. Real no-memory identity check also generates up to eight tokens on the same RNG/carrier.

The separate transactions stage probes cold/cached projection and memory version checks on the actual fifteen-cell natural regions, without loading the full model. It is not added again to live bridge costs.

Final coverage verifies exact embeddings, positions and masks through PrefixMatcher for the original captured IDs on all 184 frames. It is a causal prefix opportunity audit, not actual service reuse. K16 is a one-pair cost/coverage diagnostic, never a parallel training path. Scheduler approvals and forced control lengths are distinct fields.

The report stops with NO_GO when coverage-weighted cost fails. If a new run returns GO, its report command deliberately raises until conditional training/quality are performed; never label it complete automatically. Training requires assert_training_ready before constructing an optimizer. Trained deployment additionally requires cost/training/quality provenance in CompactVisualComposer.load_adapter. Production config remains native.

## Existing evidence only

CPU lifecycle replay (overwrites the coverage summary with PREFIX_PENDING; rerun cost only in a fresh root for final actual-prefix proof):

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.compact_visual_memory --stage coverage --root outputs/m3_compact_visual_memory_reproduce01
```

Re-render reports from retained final evidence, without LLM/GPU:

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.compact_visual_memory_report
```

## Regression suite

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q
```

CUDA-only tests skip inside the sandbox; execute on the host when idle for final GPU validation. Stage reports label every unexecuted training/quality/formal E2E field NOT_RUN/null. Existing earlier experiment directories and the official config are not modified.

## Separate stage probe on this retained run

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.compact_visual_memory --stage transactions
```
