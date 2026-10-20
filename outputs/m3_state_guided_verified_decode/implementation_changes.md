# Implementation changes

- `worldstream/m3/historical_draft.py`: bounded immutable history and strict full text prompt/scene/camera/task/model/tokenizer/generation contracts. Explicit completed, parsed and submitted commits; no logits, GT or visual facts stored.
- `worldstream/m3/verified_sampling.py`: deterministic proposal acceptance and normalized rejection primitive, independently enumerated and Monte Carlo tested. No generation loop installation.
- `tools/worldstream_m3/sgvd/`: real complete-model history census, offline probability probes, paired CUDA costs, numerical target/KV checks, fallback KV analysis, frozen complete-service regression and artifact generation.
- Tests cover history ownership/budgets/reset/protocol/task invalidation and sampling support, EOS/joint law, extreme probabilities and processor order.
- Original Native Decode, M3Runtime/config/PrefixKVCache safety policy, M1/M2 and model files are unmodified. SGVD execution and optimization were stopped at the real target-distribution gate; no unused enable switch or unqualified decoder is installed.
