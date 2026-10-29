# Reproduce continuous stream evaluation

Existing OmniDrive environment, RTX3090, no install/train/download. GPU uses /tmp/omnidrive_m123_latency_gpu.lock and waits for other users; CPU4/5 excluded.

Integrity verification (automatically restores evaluation/requests.jsonl from its lossless gzip archive if absent; both archive and raw SHA256 checked against receipt_archive.json):

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.continuous_stream verify
```

Exact repetition must copy the original frozen selection/protocol to a NEW root; rerunning metadata selection after these results exist must exclude these now-executed frames and is a different study. Repeated original frames are regression/performance repetitions, not fresh extension data.

```sh
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.continuous_stream dev --root /tmp/continuous-exact-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -c "from pathlib import Path; from tools.worldstream_m123.continuous_stream_analysis import attest_companion; attest_companion(Path('/tmp/continuous-exact-replay'))"
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.continuous_stream evaluation --root /tmp/continuous-exact-replay
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --output /tmp/continuous-exact-replay/validation_cpu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.answer_validity_validate --gpu --output /tmp/continuous-exact-replay/validation_gpu
taskset -c 0-3,6-31 /home/qh/miniconda3/envs/OmniDrive/bin/python -m tools.worldstream_m123.continuous_stream report --root /tmp/continuous-exact-replay
```

Before running the commands, create the replay root and copy extended_frame_selection.json, protocol_freeze.json and continuous_qa_protocol.md verbatim from this run; also copy historical_manifest.json, historical_execution_inventory.json, additional_selected_frame_audit.json and non_M3_exposure_audit.json for provenance and reporting. Do not copy completed stage directories, selected policies, receipt archives or freeze files: the replay must generate its own. The driver validates selection hashes. DEV freezes the retrospective policy before evaluation. Each condition starts cold then runs the complete stream with no internal reset. Source hashes in dev/evaluation/execution_sources.json and policy_sources.json cover runtime/worker/grading; presentation assembly is separately attested. DEV executed the archived worker in dev/executed_sources. Before evaluation, scheduling was extended to conditionally run the predeclared Position fallback if actual selected-C Position bypass occurred; evaluation_source_revision.json preserves the exact old/new worker hashes. No inference or grading change occurred at this boundary. Generation settings remain original; using a different GPU/clock or condition order can change timings. All labels and statistics have explicit denominators and unknowns. No independent TEST or strict2% rare-risk qualification.
