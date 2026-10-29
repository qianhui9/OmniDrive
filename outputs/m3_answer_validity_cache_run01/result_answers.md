# Research answers

1. M1 provides 11 candidate-referenced current object witnesses (11 unique frame/anchor pairs), with class/ego geometry and partial identity. These are M1 estimates, not GT certifications. M1 lacks signal state, visual appearance, global safety and complete negative coverage. Details: m1_fact_coverage.json and offline_requests.jsonl.

2. Local positive object existence/category questions have the best supported potential; actual highest verified reuse type: NONE in this cohort. Broad scene, road, signal and safety questions lack M1 fact coverage.

3. The current-frame selected-task-fact validity rates, with UNKNOWN excluded only from the scored denominator, are in answer_validity_analysis.json. They are not full-answer validity rates; that rate is UNKNOWN.

4. Threshold-specific error rates are in quality_latency_pareto.json. With no scored predicted positives, precision/false-reuse is null, not perfect. Full validity accuracy is UNKNOWN.

5. For budgets 1/2/3pp, provisional DEV coverage is [('natural', 0.01, 0.0), ('natural', 0.02, 0.0), ('natural', 0.03, 0.0), ('fixed', 0.01, 0.0), ('fixed', 0.02, 0.0), ('fixed', 0.03, 0.0)]. No budget is empirically qualified.

6. Zero requests actually skipped current M3 expansion, Prefill or Decode. P2 was gated out.

7. Isolated CPU replay combined mean is 0.3743 ms. Online current snapshot/check/management cost is NOT_RUN.

8. Fixed QA WS-AVC full E2E speedup: NOT_RUN.

9. Original natural QA WS-AVC full E2E speedup: NOT_RUN.

10. B→C independent M1 E2E benefit: NOT_ESTABLISHED. Old v3 B is not the requested target-aware baseline.

11. Online cache-hit quality loss: UNKNOWN (zero online hits). Active historical v3 QA/annotation coverage is reported separately; partial annotation checks cannot qualify 2pp driving quality.

12. No new online false reuse occurred. Existing v3 object-existence contradictions arise in actual model generations. Separating a bad historical answer from temporal invalidation requires reliable labels at both source and current frame; incomplete labels leave attribution UNKNOWN.

13. No qualified quality–E2E Pareto configuration. Keep Native deployment. Three offline rule-score thresholds are candidates only.

14. No supported cache hit at any of the three predeclared DEV thresholds; current M1 observations do not cover full historical answers in this workload.

15. Answer-level reuse is still a plausible direction for local object facts, but these retained streams do not support it as an empirically validated M3 systems-paper contribution. Broad questions require evidence M1 currently does not expose; do not add templates to force GO.
