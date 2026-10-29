# Active v3 request quality

Reused 648 saved answer outputs, including 155 actively verified requests. No new generation.

| Condition | Workload | Execution | Requests | Current QA scored | Annotation scored requests | False assertions |
|---|---|---|---:|---:|---:|---:|
| A | fixed | native_no_reuse | 108 | 15 | 21 | 9 |
| A | natural | native_no_reuse | 108 | 54 | 22 | 6 |
| B | fixed | history_draft_active | 81 | 0 | 21 | 9 |
| B | fixed | native_no_reuse | 27 | 15 | 0 | 0 |
| B | natural | native_no_reuse | 108 | 54 | 22 | 6 |
| C | fixed | history_draft_active | 74 | 0 | 18 | 9 |
| C | fixed | native_no_reuse | 34 | 15 | 3 | 0 |
| C | natural | native_no_reuse | 108 | 54 | 22 | 6 |
| A | fixed | all_requests | 108 | 15 | 21 | 9 |
| A | natural | all_requests | 108 | 54 | 22 | 6 |
| B | fixed | all_requests | 108 | 15 | 21 | 9 |
| B | natural | all_requests | 108 | 54 | 22 | 6 |
| C | fixed | all_requests | 108 | 15 | 21 | 9 |
| C | natural | all_requests | 108 | 54 | 22 | 6 |

Annotation assertions and QA fact recall have separate denominators. Positive current-frame visible annotation witnesses can confirm existence or contradict explicit absence. Missing annotations, crossing intent, exact count, road semantics, signals and global safety remain UNKNOWN without exact reliable QA facts. These checks do not qualify the complete driving-quality budget.

Repeated rounds are not independent scenes. All unscored requests remain in coverage denominators. No human validation was performed. See active_request_quality.jsonl and active_request_findings.json.
