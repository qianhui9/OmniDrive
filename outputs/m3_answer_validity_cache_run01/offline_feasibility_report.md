# Bounded offline feasibility

No supported cache hit at any of the three predeclared DEV thresholds; current M1 observations do not cover full historical answers in this workload.

- natural: 36 requests, 0 history candidates, 0 locally eligible answers; rejection reasons {}.
- fixed: 36 requests, 27 history candidates, 1 locally eligible answers; rejection reasons {'M1_does_not_cover_task': 24, 'missing_answer_referenced_observation': 1, 'unbound_answer_class': 1}.

Only existing v2 development scenes/logs were used for scoring thresholds. Natural questions were never replaced by repeated questions. Fixed questions and their actual generated answers were replayed separately on their real changing frames. No target forward was run. Quality references are read after the causal decision. Missing current-frame labels remain UNKNOWN.

No online runtime was added: the requested P1 stop condition precedes P2/P3. No finite threshold search can turn missing road/signal/safety evidence into object evidence. No quality budget or E2E benefit is claimed.
