# Current-only fact bypass

`M3Runtime` remains the only language runtime. Its default
`selective_answer_mode='off'` is unchanged. A research policy with
`main_path='current_only'` answers supported questions from current committed M1
facts and skips input expansion, prefill and decode. A miss executes the existing
Target-Aware K8 generator. Question-level RNG isolation and the distinction
between current detection evidence and M2 visual reuse qualification remain.

The current-only branch never looks up full historical answers. It rejects
Temporal Risk and Target Completeness in both configuration and policy. The
existing legacy branch and frozen source snapshots remain for reproducing
historical experiments. No M1/M2 or model-weight changes are required.

The policy can select `position_mode='basic'`, `'restricted'`, or `'fallback'`.
Restricted mode adds exactly one test: selected current detector support must
exceed `position_min_confidence`. That value must come from DEV. It is not a
calibrated probability of correct nearest-target selection. Fallback mode is a
diagnostic ablation. Category and positive Existence use the existing admission
criteria. A missing object witness never justifies answering No.

The current delivery in `outputs/m3_final_fact_bypass_run01` is
`INCOMPLETE_INDEPENDENT_VALIDATION`: local official validation logs were already
used, and historical v4 scenes are in the model training partition. Its three
candidate JSONs are deliberately `DIAGNOSTIC_ONLY_NOT_FROZEN`; the online runtime
rejects them. A new DEV optimum or TEST speedup is not claimed. The historical
best measured DEV configuration remains `target_dev_0`; v4 remains NO_GO.

Offline evaluation in `fact_bypass_identity.py` distinguishes physical-object,
question-target and answer-level correctness. Matches use current frame/time,
class-independent one-to-one 3D assignment, a 2m distance limit and a .5m ambiguity
margin. Duplicate detections cannot share a GT correctness receipt. A common
answer among ambiguous instances may receive an answer label while its instance
identity remains UNKNOWN. Visibility is an annotation/projection proxy; ambiguous
visibility stays UNKNOWN. The old severe error remains in the frozen audit even
when this stricter label protocol cannot decide it.

See the run's `REPRODUCE.md` for exact CPU/CUDA regression and audit commands.
