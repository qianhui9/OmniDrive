# WSG-SD v2 reproduction

Use the existing repository `/data/qh/phdCode/work4/OmniDrive` and interpreter `/home/qh/miniconda3/envs/OmniDrive/bin/python`. CPU4/5 are faulty; every execution below starts with `taskset -c 0-3,6-31`. No package installation, model download, base-model training or weight edits. The unchanged environment spec/ledger is `outputs/m3_wrcps_optimization_run01/environment_spec.json` / `.aris/compute/local.md`.

GPU commands need host execution in this Codex sandbox. Launchers acquire `/tmp/omnidrive_m123_latency_gpu.lock`, then wait for memory.used <500 MiB. They never stop other tasks. The only interrupted process in this run was our own initial v2 job with the confirmed asynchronous-camera timestamp bug; its entire attempt is retained under `initial_camera_timestamp_mismatch/` and excluded.

Base commit: `4f3fe1d165002aca0ed39bd8f96f00cb0d6fbca0`. Branch: `research/m3-world-guided-decode-v2`. Actual measured source files are SHA-bound in `execution_sources.json` / `p3/execution_sources.json`, checked again at worker completion. The initial P2 source tree is frozen under `p2_executed_sources/`; it is evidence, not a second production engine. Jobs ran with local research changes, so the code_head field records the git parent, while per-file SHA256 identifies the executed sources. The delivery commit contains the final implementation and both phases' evidence.

Prerequisites are local `ckpts/iter_10548.pth`, `ckpts/pretrain_qformer/`, validation info/images/QA and nuScenes metadata/annotations. Full original checkpoints and raw `.pt` carriers remain local and hash-indexed; they are not committed or replaced. The existing independent car-continuity policy is `outputs/m3_sf_wrcps_optimization_run01/semantic_qualification.json`. Its source calibration/validation scene/logs, along with the original three development scene/logs, are excluded from the new three-scene stream.

The native loader prepends a generated overview template. The primary workload deliberately discards that first prompt and takes the first one/three actual dataset conversation/VQA questions, preserving their original order. Fixed workloads repeat the source frame's real questions on DIFFERENT subsequent images. The known long-overview failure is separate in runtime/numeric controls. No dataset answer enters runtime retrieval or online scheduling.

## Reproduce the final P3 policy only

Choose a NEW output root; the worker refuses to overwrite measured directories. These commands rerun the primary final three-question A/B/C experiment, avoiding an unnecessary repeat of all P2 single-question controls.

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode_v2 --stage prepare --root outputs/m3_world_guided_decode_v2_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.world_guided_decode_v2 --stage run --root outputs/m3_world_guided_decode_v2_reproduce01 --three-questions-only
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.world_guided_decode_v2_numeric --root outputs/m3_world_guided_decode_v2_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.world_guided_decode_checks --root outputs/m3_world_guided_decode_v2_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode_v2_report --root outputs/m3_world_guided_decode_v2_reproduce01
```

The run includes a same-carrier old/new runtime control, the EXACT previous 24 frames/48 original questions using their answer-free frozen ID manifest, then 216 complete fresh-scene A/B/C three-question requests and same-carrier greedy probes. Retained P3 used `--skip-controls` to reuse the already measured P2 old/new runtime control; it still reran the 48-QA coverage. A standalone reproduction can omit that flag, as above, to create its own control evidence.

For a new final-policy single-/three-question experiment, omit `--three-questions-only`; that is NEW evidence, not the retained P2 policy. P2 originally measured 432 requests using its more conservative initial policy. P3 made exactly one main-policy revision after P2 found no C verification. P2 single-question results remain labeled P2; final-policy single-question E2E is NOT_RUN in this retained study. Do not silently transfer a speedup between policies or workloads. To inspect/re-execute P2 exactly, use its frozen sources in an isolated checkout with the same local data/checkpoints; do not overwrite the current production tree.

## Fair boundaries and artifacts

A is the original native language generation inside the unchanged M1/M2/heads model flow. B/C use the SAME PrefillExecutor and a fresh FULL current 513-token prefill with no persistent prefix K/V. Each condition builds its own history from only its prior outputs. A/B/C time the same model request from before M1 through answer serialization, with CUDA completion synchronized at both boundaries. Data loading/collation, raw capture saving and offline scoring are outside equally. C snapshot/retrieval/diff costs remain inside. Source/frame/epoch/version and asynchronous camera observation support are saved, without future/GT information.

Native generation is `do_sample=True, temperature=0.1, top_p=0.75, num_beams=1, max_new_tokens=320, use_cache=True`, plus installed defaults (including top_k=50). No shorter max length, forced token sequence, altered user question or reduced visual representation is used. Greedy changes only do_sample for correctness/quality probes. It is NOT the deployed sampling contract.

Three rotating condition orders are used, scene memory reset identically, with first scene frames separately marked cold. Read `abc_comparison.json` for primary final P3 and separately labeled P2 module/scene/round results; `effective_draft_coverage.json` for accepted-token plus measured local net-cost coverage; and raw `.jsonl` for all IDs, outputs, rejection positions, module times and peak GPU allocation.

Quality evaluation happens strictly AFTER inference using exact CURRENT-FRAME QA labels and independent nuScenes annotations. Fixed source-frame answers are never current-frame GT. Rule-extracted facts have explicit coverage/abstentions; reference class/action recall is not exhaustive accuracy. Sampling frequencies with only three rounds are diagnostics, not a distribution-equivalence proof. Existing v1 sampler transition/real small-model law evidence is preserved, and current regression exercises the same acceptance/residual code. Full native FP16 distribution equality is NOT_CLAIMED.

## Re-render retained reports / final tests

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode_v2_report
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 tests/worldstream_m123 -q
```

CUDA-specific tests skip in the sandbox. The serialized host checker above records actual GPU tests, affinity, kernel witness and checkpoint SHA256. Formal source configuration remains native; research qualification does not authorize automatic deployment. `go_no_go.json` separates Exact, approximate quality coverage, measured E2E and M1 incremental contribution.
