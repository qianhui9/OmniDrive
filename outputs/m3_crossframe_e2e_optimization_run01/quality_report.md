# Quality evidence

Official unchanged evaluators on the declared 36-frame subset; these are not full-validation scores. Absolute score loss must be ≤ 0.10. Internal M3 quality budget remains 0.02. Negative loss means an improvement.

| Workload | mAP loss | NDS loss | Lane DET_l loss | VLM Token F1 loss | Four budgets |
|---|---:|---:|---:|---:|---|
| single | -0.002054 | -0.002884 | 0.003996 | 0.004688 | True |
| multi | -0.002054 | -0.002884 | 0.003996 | 0.001018 | True |

## single
Full official per-scene metrics, populations and proxy error summaries are retained in `summary.json` and the official evaluator directories.

M3-only mean Token F1 loss vs zero historical KV: 0.000000; changed answers 0; critical proxy regressions 0. Native and candidate proxy errors are separately retained in `single_quality_pairs.json` and `summary.json`. Traffic assertions requiring review: 0.

Native→optimized spatial/numeric/motion proxy findings in repeat 0: 1; repeats do not increase independent evidence. Native baseline proxy errors: {'numeric_exact': 1, 'spatial_label_match': 10, 'motion_label_match': 4, 'safety_label_match': 1}; optimized proxy errors: {'numeric_exact': 1, 'spatial_label_match': 10, 'motion_label_match': 4, 'safety_label_match': 1}. These are parser findings, not independent human confirmation of new hard semantic failures. The overall semantic proxy gate remains conservative.


## multi
Full official per-scene metrics, populations and proxy error summaries are retained in `summary.json` and the official evaluator directories.

M3-only mean Token F1 loss vs zero historical KV: 0.000000; changed answers 0; critical proxy regressions 0. Native and candidate proxy errors are separately retained in `multi_quality_pairs.json` and `summary.json`. Traffic assertions requiring review: 0.

Native→optimized spatial/numeric/motion proxy findings in repeat 0: 2; repeats do not increase independent evidence. Native baseline proxy errors: {'numeric_exact': 1, 'spatial_label_match': 37, 'motion_label_match': 13, 'safety_label_match': 4}; optimized proxy errors: {'numeric_exact': 1, 'spatial_label_match': 36, 'motion_label_match': 13, 'safety_label_match': 4}. These are parser findings, not independent human confirmation of new hard semantic failures. The overall semantic proxy gate remains conservative.

Historical double-yellow/no-passing/lane-count hallucination remains a rejection. Its valid M1 state and 94.4% M2 refresh did not establish semantic safety. The development-only 128-visual-token trial does not remove that failure. Higher lexical F1 does not override it. Changed-answer evidence and traffic-assertion flags remain available for direct scene review; parser success alone is not semantic certification.

New development failure: at `f38ef51177c84c2a9fa32584b79e601e`, the 163-token trial changed waiting pedestrians to people currently crossing, contradicting the unchanged annotation that no pedestrians are crossing. Token F1 improved by 0.0680 and automatic critical count was zero. This is preserved as a separate motion-semantic rejection in `calibration_semantic_review.json`, with all six image paths/SHA and state signals. It does not become a qualified Object Approx rule.

Both the historical Map failure and the new crossing failure executed all 22 global M2 suffix blocks, skipped zero suffix blocks, and published a fresh global anchor (age after update 0). World-change scores were 0.7083 and 1.0, respectively. Freshness alone cannot reauthorize approximation; these sparse observations do not establish a reliable safe/unsafe threshold. Full state/refresh evidence and source SHA are in `semantic_state_attribution.json`.
