# Reproduce P0/P1

Use OmniDrive Python and exclude faulty CPUs 4/5. All prerequisites are existing local nuScenes tables/metadata/QA, v2/v3 JSONL and the local pretrain_qformer tokenizer. No network, downloads or model forward is needed. Use a NEW output root; the study refuses to overwrite an existing result.

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_study --root outputs/m3_answer_validity_cache_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest -q tests
```

P0 source: outputs/m3_world_guided_decode_run03/e2e_trials.jsonl. Its B is the old non-target-aware verifier; its C is the existing target-aware world-guided path. Those historical conditions are never relabeled as WS-AVC A/B/C. P1 source: outputs/m3_world_guided_decode_run02/p3/e2e_trials.jsonl. Only round0 three-QA actual native outputs seed original history; corresponding C committed perception snapshots provide actual world evidence. Question tokenization uses only the existing local tokenizer.

P0 and P1 scene/log disjointness is enforced. All current answer/QA/annotation comparison happens after the causal validity decision. Predictions from M1 are not ground truth. Repeated rounds are not independent labels. QA scores are selected reference fact recall; annotation scores are partial positive-witness checks with coverage. Unknown facts cannot satisfy a quality budget.

Formal A/B/C: NOT_RUN_OFFLINE_GATE. Required future B is target-aware exact history; future C must use exactly that generator on cache misses. Do not reuse old v3 B as B. Actual online skips and new GPU model experiments are zero. Host CUDA regression tests (when available) use the existing /tmp/omnidrive_m123_latency_gpu.lock, check the GPU is idle, and never terminate other processes. CPU-only sandbox tests skip unavailable CUDA checks.

Host GPU tests must use this module entry to acquire the GPU lock and support multiprocessing spawn (use a NEW validation directory):

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --gpu --output outputs/m3_answer_validity_cache_reproduce01/validation_gpu
```

For CPU tests, use the same validation command without `--gpu`; it hides CUDA. The direct pytest command above is for a CPU-only sandbox; on a GPU host use the wrapper to honor the shared GPU lock. Final validation summary, task-coverage convenience tables and the artifact manifest are delivery attestations generated after the measured study, separate from its source-bound measurements.
