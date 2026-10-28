# Controlled feasibility performance

Cost decision: NO_GO. Not a formal deployment or E2E benchmark.

K8 B−C executor wall: -0.9900 ms. K8 A−C executor wall: -2.9842 ms. Stable8-only CUDA (text-only minus text+8): 0.6123 ms. These differ: fixed text savings cannot be attributed to world memory.

After extra costs, conditional hit net -5.5686 ms; natural coverage-weighted single-question net -1.5398 ms/frame. Whole-scene bootstrap CI95: [-1.762521320258348, -1.412706985713247]. Three cost scenes are a bounded exploratory control, not a population confidence guarantee. Break-even probability None; maximum under observed coverage 0.250000; actual opportunity probability 0.208333.

No-memory bridge mean 0.2257 ms; create bridge 1.3735 ms; keep bridge 0.9449 ms. Composer cold 0.2398 ms, retained 0.1970 ms; preparation 0.4548 ms; planner including matching 0.9877 ms. All figures are newly measured. Component nesting is documented in bridge_cost_breakdown.json.

Natural coverage is 24 finite development frames across three logs, only one memory-bearing scene. Enriched scenes are availability-selected and cannot stand in for natural deployment coverage. Actual prefixes are checked on original native IDs/carriers for all frames, including no-memory/layout/scene transitions. No formal scheduler approvals are substituted with controlled reuse lengths.

Training, memory-bearing task quality, free generation A/B/C, equal-step decode E2E and final acceptance: NOT_RUN due to cost NO_GO. No M1/M2 full-request FPS/QPS claim. Investigate larger-compute reusable vision-language intermediate states with their own causal validity and accuracy gates; these results establish no such alternative speedup.
