# WSG-SD v3 reproduction

Base commit: `2bb65d8090194713e602fd9557cdd797b3aea658`. Branch: `research/m3-world-guided-decode-v3`. Use the existing OmniDrive interpreter `/home/qh/miniconda3/envs/OmniDrive/bin/python` and workspace `/data/qh/phdCode/work4/OmniDrive`. Every execution starts with `taskset -c 0-3,6-31`; CPUs4/5 are faulty. No package installation, new model, draft LLM, adapter training or checkpoint change. The existing environment spec is `outputs/m3_wrcps_optimization_run01/environment_spec.json`, ledger `.aris/compute/local.md`; its unchanged warm environment is reused.

Prerequisites are the existing local six-camera nuScenes data/QA/metadata, `ckpts/iter_10548.pth`, `ckpts/pretrain_qformer/`, v1/v2 retained outputs and the existing semantic qualification policy. No downloads are performed. GPU launchers use `/tmp/omnidrive_m123_latency_gpu.lock`, wait for GPU memory.used<500MiB and never stop other processes. In Codex's sandbox CUDA requires host execution; ordinary CPU report rendering works inside it.

## New-root reproduction

Use a NEW output root. The worker refuses to overwrite its e2e/capture directories. Commands below use `outputs/m3_world_guided_decode_v3_reproduce01` as an example.

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode_v3 --stage prepare --root outputs/m3_world_guided_decode_v3_reproduce01 --exclude-plan outputs/m3_world_guided_decode_run03/initial_scope_mismatch/experiment_plan.json
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode_v3 --stage offline --root outputs/m3_world_guided_decode_v3_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/omnidrive-wsg3-mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.world_guided_decode_v3 --stage development --root outputs/m3_world_guided_decode_v3_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/omnidrive-wsg3-mpl /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.world_guided_decode_v3 --stage run --root outputs/m3_world_guided_decode_v3_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode_checks --root outputs/m3_world_guided_decode_v3_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode_v3_report --root outputs/m3_world_guided_decode_v3_reproduce01
```

`offline` constructs candidates from previous native outputs/worlds on v2 development scenes, then compares with current sampled native trajectories. It is a diagnostic LCP upper bound on those reference trajectories, not a stochastic acceptance measurement, quality truth or an online oracle. `development` is one bounded old-scene fixed-three-QA B/C round (24 full requests), reusing the same framework. The retained development source precedes final historical-cost calibration and the clear-path full-scene safety correction; byte-exact source copies are under `development_fixed/executed_sources/`. New-root development with the final code may differ and is not silently substituted for the retained run. `POLICY_FREEZE.md` describes this finite boundary.

`run` executes 432 full A/B/C model requests: three new scene/log streams, four distinct contiguous frames each, natural/fixed questions, one/three questions per frame, three rotating A/B/C orders. Then it performs same-carrier greedy probes on original QA. Heldout scenes/logs exclude original v1/v2 development, excluded initial v3 scope-debug streams and M1 qualification calibration/validation logs. Selection uses no model outcome. The generated overview at loader index0 is explicitly removed; original questions at indices1..3 remain in order. Fixed workloads repeat the first real frame's questions on changing images, without transferring source-frame QA answers as current truth.

## Execution and timing contract

A is original native generation. B is the ordinary v2 exact-history K8 verifier, without M1-conditioned composition or target-aware selection. C uses the same executor with <=4 deterministic proposals, current processed prefill probabilities and past acceptance/cost statistics. All three retain six cameras, full current Object256+Ego1+Map256 visual input, same model/weights, current context and max_new_tokens320. Sampling remains do_sample=True, temperature0.1, top_p0.75, num_beams1, top_k50 (installed default), use_cache=True. Greedy diagnostics only override do_sample=False.

Each condition builds its OWN prior-answer memory; scene resets are identical. No online native/current answers, labels or future frames enter C. No history decode KV or persistent visual-prefill KV is inherited. All candidates require current target verification. The prefill selector draws no RNG and adds no target forward; target acceptance/residual/bonus uses the existing law. FP16 block/serial numerical differences remain explicitly approximate; strict native distribution equality is not claimed.

The same original outer model call is CUDA-synchronized before and after for A/B/C, including M1/M2/heads/current multimodal assembly/full-prefill/decode/C snapshot and retrieval/selection/answer serialization. Data collation, capture saving and offline reporting are outside equally. First scene frames are marked cold. Raw wall includes actual CPU/GPU blocking; native generation on pre-expanded carriers is never substituted for complete E2E. Target selector host wall includes pending prefill wait; its CUDA span is already inside decode wall, so do not add it twice.

## Evidence and limits

`execution_sources.json` binds every measured source SHA256; the worker checks unchanged bytes at completion. The code_head field is the git parent while changes are pending; per-file hashes bind the executed implementation. Original checkpoint hash is verified by the final GPU checker. `artifact_manifest.json` indexes raw rows/logs and local-only captures. Large `.pt` carrier files and duplicate per-frame answer exports stay local; full generated token IDs/answers/acceptance/crop and timing records are committed in JSONL.

Only AFTER inference, the report reads exact current-frame QA facts and independent nuScenes object geometry. Native answers are never truth. Rule extraction abstains on unavailable/ambiguous labels; class/action reference recall is not exhaustive accuracy. Fixed questions missing from current-frame labels remain UNKNOWN. Sampling first-token frequencies use n=3 and do not prove distribution equivalence. Greedy no-draft agreement does not validate speculation. Historical fact extraction is timed within online management aggregate; its separate CPU replay is explicitly labeled offline.

Re-render retained reports with:

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode_v3_report
```

Default formal deployment stays native. GO/PARTIAL_GO/NO_GO/INCOMPLETE is a research decision, not automatic deployment authorization.

The initial v3 primary worker was interrupted after320 complete frames due to a demonstrated road-vs-vehicle scope bug, not a numeric mismatch. All its data are retained in initial_scope_mismatch/ and excluded from final summaries. Those scenes/logs are development only. The corrected explicit road-type scope and negated full-scene dependencies were tested, and a fresh three-scene/log cohort was selected deterministically. No extra template search followed.

Final retained validation:744 tests passed in the actual RTX3090 environment; see validation_summary.md and delivery_audit.json. quality_coverage_by_execution.json exposes the absence of labels on actively speculating fixed-stream outputs; zero loss on scored no-draft rows is not budget qualification.
