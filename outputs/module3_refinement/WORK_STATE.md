# M3 refinement final state

All authorized implementation and validation stages are complete. No GPU jobs remain.

- Final report: refinement_report.md; audit: delivery_audit.json; precise outcomes: delivery_status.json.
- Main 480 real frame executions: multi_20261017_005514. Numeric E2E30: numeric_e2e_20261017_005514. Original single sampling60: single_20261017_005514.
- Genuine baseQKV FP32 E2E36: projection_e2e_01. Actual16frame/80question precision comparisons64rows: state_sweep_02/projection_probe (byte-for-byte preserved controls from state_sweep_01).
- Real M1 history: state_sweep_02, 3scenes16frames=48, no usable verified task context. Feasibility No-Go on this workload.
- Tests: 212 passed, 0 skipped in existing OmniDrive environment.
- Exact equality154/160 (original6scenes118/120, heldout36/40). Batch152/160, E2EP50~1.842x offline. Strict targets not met. No production optimization approvals. Default native full generate.
- Single question:8/8IDs and perception summaries match, no net latency improvement.
- All original weights/LoRA/GT and outputs/module3 untouched. Pilot failures and source compatibility exception are documented in experiment_notes.md and delivery_audit.json.
- Full fresh-directory reproduction: bash outputs/module3_refinement/commands.sh. No commit was requested or created.
