# Prior temporal-veto evidence

9 three-question records reduce to 3 independent Scene/Frame/Question/Selected-Target cases on 1 scene. Counts: {'unknown_veto': 3}.

All original nearest-class ambiguity and visibility conditions remain. Current QA mentions multiple vehicles but does not resolve the nearest target. Exact coordinate nearest is not promoted to a reliable visual semantic label when multiple different classes lie within the frozen2m ambiguity margin. No human review has been performed. UNKNOWN is neither beneficial nor unnecessary. No new Temporal Risk formula is justified by this audit alone.

- 0f5cbb909e9544dd861dc2ff63ac4d13: A car is directly ahead. -> Directly ahead is a large white truck.; unknown_veto; {'known': False, 'reason': 'annotated_nearest_class_ambiguous'}; 11 additional LLM forwards.

- da74550a10a8408299d7e6b1735963ae: A car is directly ahead. -> Directly ahead is a large white truck.; unknown_veto; {'known': False, 'reason': 'annotated_nearest_class_ambiguous'}; 3 additional LLM forwards.

- cf86af90b622499b8214df0024a6ea29: A car is directly ahead. -> Directly ahead is a large white truck.; unknown_veto; {'known': False, 'reason': 'annotated_nearest_class_ambiguous'}; 3 additional LLM forwards.

The three distinct frame cases belong to one contiguous scene episode and are correlated; they are not three independent scenes.
