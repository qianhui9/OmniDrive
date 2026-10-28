# Current-state provenance and conditional implementation

The existing `VerifiedWorldState` in `worldstream/m3/world_state.py` is explicitly a pre-commit, frozen historical precheck. It does not contain the current detector commit. It cannot be used as `W_t`.

The detector invokes M1 commit via its perception-output observer before language generation. A future question-aware diff must read only the published `manager.store`, require the current frame in `committed_frame_ids`, bind `last_timestamp`, `commit_count`, `generation`, execution signature and source frame, and compare against an immutable copy of the **previous committed** snapshot. No `_pending_store`, future frame or GT is allowed. Existing source-linked association and tracking algorithms must remain untouched.

Partial M1 observations cannot establish complete-scene absence, counts, signals or driving safety. Identity continuation alone is not semantic correctness. Missing dependencies, uncertain association, missing motion, scene reset and unfamiliar question types must lower draft length to zero. Any proposed history token still requires current target verification. The diff must never edit the visual carrier or directly return an answer.

Conditional output schema: `QuestionAwareWorldDelta(scene_id, previous_frame, current_frame, question_signature, task_profile, relevant_anchor_ids, matched_anchor_ids, changed_attributes, unknown_dependencies, evidence_versions, draft_risk, suggested_draft_length, decision_reason)`.

Implementation/accuracy experiment: NOT_RUN. The history-only CUDA gate shows local speed potential, but the universal real-model greedy/numerical gate fails and production sampling is not qualified. Complex M1 scheduling therefore stops at this prerequisite. This document is a provenance audit and design, not a claim that current-state guidance has been implemented or validated.
