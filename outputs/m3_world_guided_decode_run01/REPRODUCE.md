# Reproduce WSG-SD measurements and failed deployment gate

Work in `/data/qh/phdCode/work4/OmniDrive`, environment `/home/qh/miniconda3/envs/OmniDrive/bin/python`. Exclude faulty CPU4/5 with **every** execution command below. Warm-reuse the existing environment/spec in `.aris/compute/local.md`; install nothing, modify no weights. CUDA is unavailable in the Codex sandbox; GPU commands need host execution. Every GPU launcher uses `/tmp/omnidrive_m123_latency_gpu.lock` and waits for memory.used < 500 MiB; never stop another process.

Base HEAD `ae5a357b6a2619500c9d6798b7efc5cc753e827e`; research branch `research/m3-world-guided-decode`. Existing `ckpts/iter_10548.pth`, pretrained tokenizer/model files and nuScenes validation data are required. Raw `.pt` carriers stay local, SHA-indexed at delivery.

The native loader inserts a random overview template BEFORE actual dataset questions. This run profiled its first three questions: one generated overview plus the first two original QA entries. The question manifest freezes all input IDs and wording, contains **no answers**, and allows exact replay of this loader randomness. `workload_scope_audit.json` records the post-capture provenance correction; all trials remain available. The raw QA cohort is 48 questions; loader templates are not original dataset QA. Fixed raw QA repeats two real first-frame dataset questions over subsequent distinct images; generated overview repetition is separately labeled.

Run the following stages sequentially. Choose a NEW root and preserve this retained evidence. These commands use `outputs/m3_world_guided_decode_reproduce01`.

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.world_guided_decode --stage profile --root outputs/m3_world_guided_decode_reproduce01 --question-manifest outputs/m3_world_guided_decode_run01/question_manifest.json
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode --stage overlap --root outputs/m3_world_guided_decode_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.world_guided_decode --stage benchmark --root outputs/m3_world_guided_decode_reproduce01
```

The strict benchmark is **expected to stop nonzero** on actual greedy divergence. Do not label this an infrastructure failure or proceed to formal deployment. Preserve its files, then explicitly run an approximate cost diagnostic if desired:

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode --stage archive-failure --root outputs/m3_world_guided_decode_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.world_guided_decode --stage benchmark --allow-numerical-diagnostic --root outputs/m3_world_guided_decode_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.speculative_sampling_validation --root outputs/m3_world_guided_decode_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.speculative_numeric_probe --root outputs/m3_world_guided_decode_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.speculative_numeric_probe --schedule --root outputs/m3_world_guided_decode_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -u -m tools.worldstream_m123.world_guided_decode_checks --root outputs/m3_world_guided_decode_reproduce01
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode_report --root outputs/m3_world_guided_decode_reproduce01
```

The numeric probe intentionally replays the final task of the preserved strict-gate failure. Its original fixed-chunk probe compares the same forced tokens with 8/9-token blocks and serial forwards. The schedule probe mirrors the actual verifier's rejection/crop/continuation schedule and computes an independent serial reference. These are numeric/causal witnesses, not performance trials.

The cost gate contains three rotating mode orders for native/history4/history8, with greedy and original sampling reported separately. Historical proposals always originate in earlier real model outputs, never current targets or GT. Native original generation parameters remain `do_sample=True, temperature=0.1, top_p=0.75, num_beams=1, max_new_tokens=320, use_cache=True`; installed HF defaults, including top_k=50 and BOS/EOS, are retained. Greedy changes only do_sample for the correctness diagnostic. This run does not shorten or force target answers.

Actual decode spans are paired on the same current 513-token carrier. Generation-wall A includes its native assembly, whereas B uses pre-expanded embeddings; do **not** interpret that generation ratio as fair complete-M3 speedup. No prefix sharing, historical decode K/V, composer training or extra draft LLM is used. Full M123 paired E2E and approximate task-quality acceptance are NOT_RUN.

The retained run's initial capture failure (`native_profile_failed_timestamp`) occurred in the capture writer after model execution. No observations from it enter results. The timestamp container fix reads dataset metadata; its failure log is retained. Large-model adversarial smoke tests passed, but broader real greedy pairs failed, so smoke tests must not override the final gate.

To re-render the retained reports without GPU or model reload:

```bash
taskset -c 0-3,6-31 env PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.world_guided_decode_report
```

Ordinary serving defaults to native decode. The optional `M3Config(decode_strategy='history_draft', reuse_strategy='exact_prefix')` is a **research option**, forces fresh full current-visual prefill and records only answer Token IDs/provenance. Sampling with a nonempty draft additionally requires explicit `experimental_speculative_sampling=True`. This opt-in does not imply deployment qualification. M1 world-diff/scheduler implementation was stopped at the failed prerequisite and is NOT_RUN; do not claim WSG-SD/M1 incremental or E2E success from the history-only decoder.
