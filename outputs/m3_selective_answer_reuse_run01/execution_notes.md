# Execution notes

The initial prompt encoder used the raw site location instead of NuScenesDataset's city prefix. Its fixed first-frame equality assertion failed before any fixed/addressable TEST model call. All initial outputs (24 Shadow + 96 DEV Pareto + 162 native-only TEST calls) are excluded and retained under initial_prompt_context/. No initial TEST quality labels or answers were used to choose a policy. Boston/Singapore native-loader token equality tests and an actual per-scene preflight now catch this before any timed inference.

Selecting three additional image-complete independent logs was not possible with the local metadata/data availability. The original metadata-selected DEV-disjoint TEST cohort is retained unchanged. Every replacement condition is rerun; none of the initial timings/answers are mixed into final A/B/C. Confidence candidates remain the original .45/.65/.85. Only DEV Shadow selects policy.

Two shell/env host-execution requests timed out in automatic approval before any process started. A narrower fixed Python module entry was approved. Thread limits, offline flags and stage logs are now set by that module. The resource policy and CPU affinity were maintained; no other process was stopped.

B/C explicitly enable the M123 completion contract. Default mode still requires PrefillExecutor completion. An opt-in fast response requires current committed perception evidence, an admitted frozen research policy, zero language calls and zero expansion, and declares Prefill/Decode not executed. An independent forward hook validates the actual zero-call claim. Each GPU stage retains its exact executed sources and hashes. The replacement Shadow predates the stricter completion receipt guard; it nevertheless executed all normal LLM requests. DEV Pareto and final B/C exercise the new guard.

Default deployment configuration and original weights are unchanged. This research does not claim human safety review or complete visual absence labels.
