# Real RTX3090 Shadow

24 complete M123 model calls / 60 language requests, including 36 addressable and 24 separate diagnostics. The target-aware LLM executed for every request; actual skips: 0.

Addressable candidates: 30; independently scored: 26; task-fact correct: 24. Full-history opportunities: 0; fact-driven opportunities: 30. See raw decisions and all UNKNOWN cases in shadow_quality_requests.jsonl. No human review was performed.

Three predeclared confidence floors were compared against the SAME paired B outputs using current-frame labels. The calibration uses empirical task loss with coverage and uncertainty, not the previous uncalibrated weighted validity score. Selected online research policy: wrsar_dev_2pp. Missing labels prevent full driving-quality certification. Shadow costs are observed, but savings are theoretical because the LLM was not skipped.
