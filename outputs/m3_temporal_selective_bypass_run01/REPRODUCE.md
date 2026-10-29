# WRSAR v2 reproduction

Working directory: `/data/qh/phdCode/work4/OmniDrive`. Existing Python environment `OmniDrive`, RTX3090, original `ckpts/iter_10548.pth` and six-camera nuScenes data. No installation, download, training or weight update. Every command excludes faulty CPU4/5. GPU stages acquire `/tmp/omnidrive_m123_latency_gpu.lock`, wait for other jobs, and never stop another user's process.

The delivered run already exists. Verify it with this exact read-only command:

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_audit --root outputs/m3_temporal_selective_bypass_run01
```

Expected: JSON `status: PASS`, all frozen/source/actual-forward checks passing. On a CUDA-restricted sandbox, model stages and CUDA regression need host GPU execution; CPU artifact verification does not need GPU. Verification validates the retained checkpoint hash attestation; add `--checkpoint` to recompute the original checkpoint hash. `--manifest` writes the integrity audit and reviewable-artifact manifest after all other files are finalized.

For a new run, use an EMPTY output path and execute these stages in order, substituting that same new root at every step. Each GPU stage is a fresh process using the same existing model and worker; do not modify measured sources while a stage runs.

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_bypass --root /tmp/wrsar-v2-reproduction --stage prepare
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_bypass --root /tmp/wrsar-v2-reproduction --stage diagnostic
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_bypass --root /tmp/wrsar-v2-reproduction --stage shadow
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_bypass --root /tmp/wrsar-v2-reproduction --stage calibrate
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_bypass --root /tmp/wrsar-v2-reproduction --stage pareto
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_bypass --root /tmp/wrsar-v2-reproduction --stage freeze
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_audit --root /tmp/wrsar-v2-reproduction --freeze-quality
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_bypass --root /tmp/wrsar-v2-reproduction --stage run
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_bypass --root /tmp/wrsar-v2-reproduction --stage ablation
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_bypass --root /tmp/wrsar-v2-reproduction --stage report
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_audit --root /tmp/wrsar-v2-reproduction --interpret
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --gpu --output /tmp/wrsar-v2-reproduction/validation_gpu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.temporal_selective_audit --root /tmp/wrsar-v2-reproduction --checkpoint --manifest
```

If DEV does not qualify a policy, freeze fails; do not proceed to TEST or silently alter seeds/thresholds. The historical severe-error scene and two DEV frames are causal diagnostics only. TEST uses three distinct new logs and four contiguous frames each, selected without GT/detector outputs. The question RNG key is SHA256(scene, frame, question index, repeat, experiment seed), independent of A/B/C and prior skipped questions. Native sampling and the existing K8 block verifier retain their original generation settings.

Current binding risk combines relevant competing witnesses, ROI/sector boundary and detection/camera support. Temporal risk adds a previous committed observation transformed through actual old/global/current Ego poses, plus reliable velocity or an explicitly provisional motion envelope (3m/s for pedestrian,12m/s otherwise,0.75m base uncertainty,1.5s maximum state age). These envelopes and risk values are heuristics, not certified physical/error bounds. History can veto an uncertain nearest target; it never supplies a current positive answer. Diagnostic `temporal_risk_enabled` changes only that risk term in C-current.

Authoritative request/answer/token/actual-forward data are in each stage's `requests.jsonl`; executed sources and hashes accompany every phase. Repeated per-frame export directories are ignored because they duplicate authoritative records. No tensors are duplicated in the delivery. Phase CUDA means must not be summed to manufacture E2E. Repeated rounds are not independent scenes; UNKNOWN facts are not correct by default. All research features remain off in default deployment.
